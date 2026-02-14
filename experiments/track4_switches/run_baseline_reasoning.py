"""
Track 4 Baseline — Run tasks without geometric interventions.

For each task in the task bank:
1. Simulate multi-step reasoning (3-5 steps per task)
2. Generate embeddings via random projection (lightweight, no GPU needed)
3. Record geometric trajectory per step via GeometricTracer
4. Score response quality via quality_scorer
5. Save results as JSON

This is the control arm: same tasks, no interventions.
"""

import json
import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.task_bank import get_task_bank, Task
from experiments.shared.quality_scorer import score_response
from experiments.shared.geometric_tracer import GeometricTracer

# Configuration
SEED = 42
EMBEDDING_DIM = 384
N_REFERENCE = 200
STEPS_PER_TASK = 4
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "track4_baseline"


def build_reference_corpus(n_samples: int, dim: int, seed: int) -> np.ndarray:
    """
    Build a synthetic reference corpus using random embeddings.

    For real deployment, replace with actual sentence-transformer embeddings
    of a domain-relevant corpus. Random projection preserves geometric
    structure (Johnson-Lindenstrauss) well enough for signature detection.
    """
    rng = np.random.RandomState(seed)
    # Use normalized random vectors (unit sphere) to mimic real embeddings
    embeddings = rng.randn(n_samples, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def simulate_reasoning_steps(
    task: Task,
    n_steps: int,
    dim: int,
    rng: np.random.RandomState,
    intervention_text: str = "",
) -> list:
    """
    Simulate multi-step reasoning for a task.

    Each step generates a response fragment and a corresponding embedding.
    The embedding is a deterministic hash of the prompt + step index,
    perturbed to simulate reasoning trajectory.

    Args:
        task: The task to reason about
        n_steps: Number of reasoning steps
        dim: Embedding dimension
        rng: Random state for reproducibility
        intervention_text: Optional text prepended to prompt (unused in baseline)

    Returns:
        List of dicts with 'text', 'embedding' per step
    """
    # Generate base embedding from task prompt
    # Use hash-based seeding for determinism
    prompt_hash = hash(task.prompt) % (2**31)
    step_rng = np.random.RandomState(prompt_hash)
    base_embedding = step_rng.randn(dim).astype(np.float32)
    base_embedding /= (np.linalg.norm(base_embedding) + 1e-8)

    steps = []
    current = base_embedding.copy()

    # Step templates for simulated reasoning
    step_templates = [
        _generate_step_text(task, i, n_steps, intervention_text)
        for i in range(n_steps)
    ]

    for i in range(n_steps):
        # Evolve embedding: drift + noise to simulate reasoning trajectory
        drift = step_rng.randn(dim).astype(np.float32) * 0.1
        noise = rng.randn(dim).astype(np.float32) * 0.05
        current = current + drift + noise
        current /= (np.linalg.norm(current) + 1e-8)

        steps.append({
            'text': step_templates[i],
            'embedding': current.copy(),
        })

    return steps


def _generate_step_text(task: Task, step_idx: int, total_steps: int, intervention: str = "") -> str:
    """Generate simulated reasoning text for a step."""
    domain_responses = {
        "research_questions": [
            "Let me consider the major theoretical frameworks relevant to this question. "
            "There are several competing perspectives, each with distinct empirical support.",
            "Examining the evidence more carefully, we find tension between these frameworks. "
            "However, some argue that the disagreement is more apparent than real, "
            "as the theories may address different aspects of the phenomenon.",
            "On the other hand, critics point out significant limitations in each approach. "
            "The trade-off between explanatory power and testability remains unresolved. "
            "Further research is needed to discriminate between these accounts.",
            "In summary, the current understanding remains uncertain and contested. "
            "The most promising direction may be to synthesize elements from multiple "
            "frameworks while acknowledging what we don't yet know. Concrete next steps "
            "should include designing experiments that could falsify specific predictions.",
        ],
        "design_exploration": [
            "The design space for this problem is broad. We should consider multiple "
            "user needs and constraints before narrowing our approach.",
            "One approach focuses on simplicity and accessibility. Alternatively, "
            "we could prioritize flexibility and power. The trade-off between these "
            "goals is a fundamental design tension.",
            "A concrete proposal: implement a tiered system that addresses different "
            "user segments. However, this introduces complexity that could undermine "
            "the accessibility goal. We should evaluate this carefully.",
            "My recommendation is to start with the minimal viable approach and iterate. "
            "Specifically, I propose the following protocol: begin with user research, "
            "prototype the core interaction, and test with the target population. "
            "The caveat is that this requires sustained investment in user feedback.",
        ],
        "ethical_dilemmas": [
            "This dilemma involves fundamental tensions between competing ethical "
            "frameworks. From a utilitarian perspective, we should maximize aggregate "
            "welfare. Deontological ethics, however, emphasizes duties and rights "
            "that cannot be traded away.",
            "The virtue ethics tradition offers another perspective entirely — asking "
            "not 'what should be done' but 'what kind of person should I be.' "
            "Some argue this reframes the dilemma productively, while others "
            "criticize it as evasive.",
            "In practice, most societies resolve such tensions through institutional "
            "compromise rather than philosophical purity. The question of 'who decides' "
            "is perhaps more important than 'what is decided.' We should be cautious "
            "about oversimplifying these issues.",
            "There is no clean resolution. Any policy will involve trade-offs between "
            "competing values. What we can recommend is transparency about these "
            "trade-offs and robust mechanisms for revisiting decisions as understanding "
            "evolves. It's possible that no single framework is adequate here.",
        ],
        "novel_hypothesis": [
            "This is speculative territory. Let me ground the hypothesis in what "
            "we currently understand and identify where the speculation begins.",
            "The proposed mechanism would need to be compatible with established "
            "physics while offering novel predictions. One tentative framework: "
            "perhaps the interaction operates through a channel we haven't yet "
            "characterized. This is preliminary and uncertain.",
            "To test this hypothesis, we would need experiments that can distinguish "
            "it from existing theories. The key falsifiable prediction would be "
            "the presence of a specific measurable signature. However, current "
            "technology may not be sensitive enough to detect it.",
            "In conclusion, this remains a speculative but potentially fruitful "
            "direction for further research. The mechanism I've proposed is "
            "tentative and may or may not survive empirical scrutiny. The most "
            "important next step is to design an experiment that could provide "
            "preliminary evidence for or against the hypothesis.",
        ],
    }

    responses = domain_responses.get(task.domain, domain_responses["research_questions"])
    text = responses[min(step_idx, len(responses) - 1)]

    if intervention and step_idx > 0:
        text = f"[Intervention: {intervention}]\n\n{text}"

    return text


def run_baseline(seed: int = SEED) -> dict:
    """Run baseline reasoning across all tasks."""
    rng = np.random.RandomState(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 4 Baseline — seed={seed}, timestamp={timestamp}")
    print(f"Building reference corpus (n={N_REFERENCE}, dim={EMBEDDING_DIM})...")

    reference = build_reference_corpus(N_REFERENCE, EMBEDDING_DIM, seed)
    tracer = GeometricTracer(reference, k=min(50, N_REFERENCE - 1))

    tasks = get_task_bank()
    results = []

    print(f"Running {len(tasks)} tasks ({STEPS_PER_TASK} steps each)...\n")

    for task in tasks:
        print(f"  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        # Simulate reasoning
        steps = simulate_reasoning_steps(task, STEPS_PER_TASK, EMBEDDING_DIM, rng)

        # Collect embeddings for tracing
        step_embeddings = np.array([s['embedding'] for s in steps])
        trace = tracer.trace_reasoning(step_embeddings, task_id=task.id)

        # Combine step texts for quality scoring
        full_response = "\n\n".join(s['text'] for s in steps)
        quality = score_response(full_response)

        result = {
            'task_id': task.id,
            'domain': task.domain,
            'prompt': task.prompt,
            'response': full_response,
            'steps': [{'text': s['text']} for s in steps],
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
        'seed': seed,
        'n_tasks': len(tasks),
        'steps_per_task': STEPS_PER_TASK,
        'embedding_dim': EMBEDDING_DIM,
        'n_reference': N_REFERENCE,
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
            'seed': seed,
            'embedding_dim': EMBEDDING_DIM,
            'n_reference': N_REFERENCE,
            'steps_per_task': STEPS_PER_TASK,
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

    print(f"\nBaseline complete. Mean quality: {summary['mean_quality']:.3f}")
    print(f"Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_baseline()
