"""
Track 4 Live Baseline — Run tasks with live Claude API, no interventions.

For each task in the task bank:
1. Call Claude for multi-step reasoning (4 steps per task)
2. Embed each step via sentence-transformer (real 384-dim embeddings)
3. Trace geometric trajectory via GeometricTracer
4. Score response quality via quality_scorer
5. Save results as JSON

This is the live control arm: real reasoning, no interventions.
API calls: 16 tasks × 4 steps = 64 Claude calls.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.task_bank import get_task_bank
from experiments.shared.quality_scorer import score_response
from experiments.shared.geometric_tracer import GeometricTracer
from experiments.shared.llm_client import ClaudeReasoningClient
from experiments.shared.embedder import ResponseEmbedder

# Configuration
STEPS_PER_TASK = 4
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "track4_live_baseline"


def build_reference_texts() -> list:
    """
    Build reference corpus texts from task prompts and evaluation criteria.

    Uses the 16 task prompts + their evaluation criteria (~80 texts)
    to create a semantically meaningful reference set.
    """
    tasks = get_task_bank()
    texts = []
    for task in tasks:
        texts.append(task.prompt)
        for criterion in task.evaluation_criteria:
            texts.append(criterion)
    return texts


def run_live_baseline() -> dict:
    """Run live baseline reasoning across all tasks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 4 Live Baseline — timestamp={timestamp}")
    print("Initializing Claude client and embedder...")

    client = ClaudeReasoningClient()
    embedder = ResponseEmbedder()

    # Build reference corpus from real task text
    print("Building reference corpus from task bank...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    n_ref = len(reference)
    k = min(50, n_ref - 1)
    tracer = GeometricTracer(reference, k=k)

    tasks = get_task_bank()
    results = []

    print(f"Running {len(tasks)} tasks ({STEPS_PER_TASK} steps each)...\n")

    for task in tasks:
        print(f"  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        # Generate reasoning via Claude API
        step_texts = client.generate_reasoning(task.prompt, n_steps=STEPS_PER_TASK)

        # Embed each step
        step_embeddings = embedder.embed_batch(step_texts)

        # Normalize for geometric tracing
        norms = np.linalg.norm(step_embeddings, axis=1, keepdims=True)
        step_embeddings_norm = step_embeddings / (norms + 1e-8)

        # Trace geometric trajectory
        trace = tracer.trace_reasoning(step_embeddings_norm, task_id=task.id)

        # Score full response
        full_response = "\n\n".join(step_texts)
        quality = score_response(full_response)

        result = {
            'task_id': task.id,
            'domain': task.domain,
            'prompt': task.prompt,
            'response': full_response,
            'steps': [{'text': t} for t in step_texts],
            'trace': trace.to_dict(),
            'quality_score': quality.to_dict(),
            'expected_signatures': task.expected_signatures,
        }
        results.append(result)

        print(f"    Quality: {quality.composite:.3f} | "
              f"Dominant sig: {trace.summary['dominant_signature']}")

    # Summary
    scores = [r['quality_score']['composite'] for r in results]
    summary = {
        'timestamp': timestamp,
        'n_tasks': len(tasks),
        'steps_per_task': STEPS_PER_TASK,
        'embedding_dim': reference.shape[1],
        'n_reference': n_ref,
        'model': client.model,
        'temperature': client.temperature,
        'mean_quality': float(np.mean(scores)),
        'std_quality': float(np.std(scores)),
        'per_domain': {},
    }

    for domain in set(r['domain'] for r in results):
        domain_scores = [r['quality_score']['composite']
                         for r in results if r['domain'] == domain]
        summary['per_domain'][domain] = {
            'mean_quality': float(np.mean(domain_scores)),
            'n_tasks': len(domain_scores),
        }

    output = {
        'config': {
            'embedding_dim': reference.shape[1],
            'n_reference': n_ref,
            'steps_per_task': STEPS_PER_TASK,
            'model': client.model,
            'temperature': client.temperature,
            'max_tokens': client.max_tokens,
        },
        'summary': summary,
        'results': results,
    }

    # Save results
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_results.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nLive baseline complete. Mean quality: {summary['mean_quality']:.3f}")
    print(f"Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_live_baseline()
