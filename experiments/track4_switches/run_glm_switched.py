"""
Track 4 GLM Switched — Run tasks with live GLM-4.7 API + geometric interventions.

Same as Claude switched experiment, but uses GLMReasoningClient.
Tests whether domain asymmetry (design helped, ethics harmed) transfers
across models.

API calls: 16 tasks x 4 steps = 64 GLM calls.
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
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.switch_engine import SwitchEngine
from experiments.track4_switches.run_live_baseline import build_reference_texts

# Configuration
STEPS_PER_TASK = 4
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "glm_switched"


def run_glm_switched() -> dict:
    """Run live switched reasoning across all tasks using GLM-4.7."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 4 GLM Switched — timestamp={timestamp}")
    print("Initializing GLM client and embedder...")

    client = GLMReasoningClient(max_tokens=2048)
    embedder = ResponseEmbedder()

    # Build reference corpus from real task text
    print("Building reference corpus from task bank...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    n_ref = len(reference)
    k = min(50, n_ref - 1)
    engine = SwitchEngine(reference, k=k, use_live_calibration=True)

    tasks = get_task_bank()
    results = []

    print(f"Running {len(tasks)} tasks ({STEPS_PER_TASK} steps each) with switches...\n")

    for task in tasks:
        print(f"  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        engine.reset_history()

        # Step-by-step reasoning with interventions
        step_texts = []
        step_records = []
        interventions_fired = []
        pending_intervention = ""

        for step_idx in range(STEPS_PER_TASK):
            # Generate step via GLM API (with intervention from previous step)
            step_text = client.generate_step(
                task_prompt=task.prompt,
                step_index=step_idx,
                total_steps=STEPS_PER_TASK,
                previous_steps=step_texts if step_texts else None,
                intervention_text=pending_intervention if pending_intervention else None,
            )
            step_texts.append(step_text)

            # Embed the step for geometric evaluation
            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            # Evaluate for intervention
            eval_result = engine.evaluate(step_embedding_norm, step_idx)

            step_record = {
                'step_index': step_idx,
                'text': step_text,
                'intervention_applied': pending_intervention if pending_intervention else None,
                'intervention_detected': eval_result.to_dict(),
            }
            step_records.append(step_record)

            # Queue intervention for next step
            if eval_result.triggered and eval_result.intervention:
                pending_intervention = eval_result.intervention.instruction
                interventions_fired.append({
                    'step': step_idx,
                    'signal': eval_result.intervention.signal,
                    'instruction': eval_result.intervention.instruction,
                })
            else:
                pending_intervention = ""

        # Score full response
        full_response = "\n\n".join(step_texts)
        quality = score_response(full_response)

        result = {
            'task_id': task.id,
            'domain': task.domain,
            'prompt': task.prompt,
            'response': full_response,
            'steps': step_records,
            'quality_score': quality.to_dict(),
            'interventions_fired': interventions_fired,
            'n_interventions': len(interventions_fired),
            'expected_signatures': task.expected_signatures,
            'engine_stats': engine.get_intervention_stats(),
        }
        results.append(result)

        n_fired = len(interventions_fired)
        fired_signals = [i['signal'] for i in interventions_fired]
        print(f"    Quality: {quality.composite:.3f} | "
              f"Interventions: {n_fired} | "
              f"Signals: {fired_signals}")

    # Summary
    scores = [r['quality_score']['composite'] for r in results]
    total_interventions = sum(r['n_interventions'] for r in results)

    summary = {
        'timestamp': timestamp,
        'n_tasks': len(tasks),
        'steps_per_task': STEPS_PER_TASK,
        'embedding_dim': reference.shape[1],
        'n_reference': n_ref,
        'model': client.model,
        'backend': 'glm',
        'temperature': client.temperature,
        'mean_quality': float(np.mean(scores)),
        'std_quality': float(np.std(scores)),
        'total_interventions': total_interventions,
        'mean_interventions_per_task': total_interventions / len(tasks),
        'per_domain': {},
    }

    for domain in set(r['domain'] for r in results):
        domain_results = [r for r in results if r['domain'] == domain]
        domain_scores = [r['quality_score']['composite'] for r in domain_results]
        domain_interventions = sum(r['n_interventions'] for r in domain_results)
        summary['per_domain'][domain] = {
            'mean_quality': float(np.mean(domain_scores)),
            'n_tasks': len(domain_scores),
            'total_interventions': domain_interventions,
        }

    output = {
        'config': {
            'embedding_dim': reference.shape[1],
            'n_reference': n_ref,
            'steps_per_task': STEPS_PER_TASK,
            'model': client.model,
            'backend': 'glm',
            'temperature': client.temperature,
            'max_tokens': client.max_tokens,
            'policy_table': engine.get_policy_table(),
        },
        'summary': summary,
        'results': results,
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "glm_switched_results.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nGLM switched run complete. Mean quality: {summary['mean_quality']:.3f}")
    print(f"Total interventions: {total_interventions}")
    print(f"Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_glm_switched()
