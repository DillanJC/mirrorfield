"""
Track 4 Switched — Run tasks WITH geometric interventions.

Same as baseline, but after each reasoning step, SwitchEngine.evaluate()
checks for intervention. If triggered, the intervention text is prepended
to the next step's prompt.

Records: which interventions fired, when, and the modified prompts.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.task_bank import get_task_bank, Task
from experiments.shared.quality_scorer import score_response
from experiments.shared.geometric_tracer import GeometricTracer
from experiments.track4_switches.switch_engine import SwitchEngine
from experiments.track4_switches.run_baseline_reasoning import (
    build_reference_corpus,
    simulate_reasoning_steps,
    SEED, EMBEDDING_DIM, N_REFERENCE, STEPS_PER_TASK,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "track4_switched"


def run_switched(seed: int = SEED) -> dict:
    """Run switched reasoning across all tasks."""
    rng = np.random.RandomState(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 4 Switched — seed={seed}, timestamp={timestamp}")
    print(f"Building reference corpus (n={N_REFERENCE}, dim={EMBEDDING_DIM})...")

    reference = build_reference_corpus(N_REFERENCE, EMBEDDING_DIM, seed)
    engine = SwitchEngine(reference, k=min(50, N_REFERENCE - 1))

    tasks = get_task_bank()
    results = []

    print(f"Running {len(tasks)} tasks ({STEPS_PER_TASK} steps each) with switches...\n")

    for task in tasks:
        print(f"  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        engine.reset_history()

        # Step-by-step reasoning with interventions
        step_rng = np.random.RandomState(hash(task.prompt) % (2**31))
        base_embedding = step_rng.randn(EMBEDDING_DIM).astype(np.float32)
        base_embedding /= (np.linalg.norm(base_embedding) + 1e-8)

        current = base_embedding.copy()
        step_records = []
        interventions_fired = []
        pending_intervention = ""

        for step_idx in range(STEPS_PER_TASK):
            # Evolve embedding
            drift = step_rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.1
            noise = rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
            current = current + drift + noise
            current /= (np.linalg.norm(current) + 1e-8)

            # Evaluate for intervention
            eval_result = engine.evaluate(current, step_idx)

            # Generate step text (with intervention from previous step)
            from experiments.track4_switches.run_baseline_reasoning import _generate_step_text
            step_text = _generate_step_text(
                task, step_idx, STEPS_PER_TASK, pending_intervention
            )

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
        full_response = "\n\n".join(s['text'] for s in step_records)
        quality = score_response(full_response)

        # Collect trace from engine history
        trace_steps = []
        for record in step_records:
            detected = record['intervention_detected']
            if detected.get('step_trace'):
                trace_steps.append(detected['step_trace'])

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
        print(f"    Quality: {quality.composite:.3f} | "
              f"Interventions: {n_fired}")

    # Summary
    scores = [r['quality_score']['composite'] for r in results]
    total_interventions = sum(r['n_interventions'] for r in results)

    summary = {
        'timestamp': timestamp,
        'seed': seed,
        'n_tasks': len(tasks),
        'steps_per_task': STEPS_PER_TASK,
        'embedding_dim': EMBEDDING_DIM,
        'n_reference': N_REFERENCE,
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
            'seed': seed,
            'embedding_dim': EMBEDDING_DIM,
            'n_reference': N_REFERENCE,
            'steps_per_task': STEPS_PER_TASK,
            'policy_table': engine.get_policy_table(),
        },
        'summary': summary,
        'results': results,
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "switched_results.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSwitched run complete. Mean quality: {summary['mean_quality']:.3f}")
    print(f"Total interventions: {total_interventions}")
    print(f"Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_switched()
