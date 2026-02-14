"""
Track 5 Recursive Trials — Multi-iteration learning loop.

Runs 5 iterations of the full task bank. Each iteration:
1. Run all tasks with current policy (via RecursiveLearner's SwitchEngine)
2. Record quality deltas relative to baseline
3. Update policy based on accumulated experience
4. Run Goodhart detector check

Saves per-iteration: policy state, quality scores, geometric traces,
and Goodhart report.
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
from experiments.track4_switches.run_baseline_reasoning import (
    build_reference_corpus, simulate_reasoning_steps, _generate_step_text,
    SEED, EMBEDDING_DIM, N_REFERENCE, STEPS_PER_TASK,
)
from experiments.track5_recursive.recursive_learner import RecursiveLearner
from experiments.track5_recursive.goodhart_detector import (
    GoodhartDetector, IterationSnapshot,
)

N_ITERATIONS = 5
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "track5_recursive"
GOODHART_RED_THRESHOLD = 3


def run_iteration(
    learner: RecursiveLearner,
    tracer: GeometricTracer,
    baseline_scores: dict,
    rng: np.random.RandomState,
    iteration: int,
) -> dict:
    """
    Run a single iteration through all tasks.

    Args:
        learner: The RecursiveLearner with current policy
        tracer: GeometricTracer for trajectory recording
        baseline_scores: {task_id: quality_score} from Track 4 baseline
        rng: Random state
        iteration: Iteration number

    Returns:
        Iteration results dict
    """
    tasks = get_task_bank()
    engine = learner.get_engine()
    results = []

    for task in tasks:
        engine.reset_history()

        # Simulate reasoning with current policy
        step_rng = np.random.RandomState(hash(task.prompt) % (2**31))
        base_embedding = step_rng.randn(EMBEDDING_DIM).astype(np.float32)
        base_embedding /= (np.linalg.norm(base_embedding) + 1e-8)

        current = base_embedding.copy()
        step_records = []
        interventions_fired = []
        pending_intervention = ""

        for step_idx in range(STEPS_PER_TASK):
            drift = step_rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.1
            noise = rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
            current = current + drift + noise
            current /= (np.linalg.norm(current) + 1e-8)

            eval_result = engine.evaluate(current, step_idx)

            step_text = _generate_step_text(
                task, step_idx, STEPS_PER_TASK, pending_intervention
            )

            step_records.append({
                'step_index': step_idx,
                'text': step_text,
                'intervention_applied': pending_intervention if pending_intervention else None,
            })

            if eval_result.triggered and eval_result.intervention:
                pending_intervention = eval_result.intervention.instruction
                interventions_fired.append({
                    'step': step_idx,
                    'signal': eval_result.intervention.signal,
                })
            else:
                pending_intervention = ""

        # Score
        full_response = "\n\n".join(s['text'] for s in step_records)
        quality = score_response(full_response)

        # Record experience: quality delta vs baseline
        baseline_q = baseline_scores.get(task.id, quality.composite)
        quality_delta = quality.composite - baseline_q

        for intervention_info in interventions_fired:
            learner.record_experience(
                task_id=task.id,
                signal=intervention_info['signal'],
                intervention_text="",  # kept brief for storage
                quality_delta=quality_delta,
                step_index=intervention_info['step'],
            )

        # Trace
        step_embeddings = []
        step_rng2 = np.random.RandomState(hash(task.prompt) % (2**31))
        base2 = step_rng2.randn(EMBEDDING_DIM).astype(np.float32)
        base2 /= (np.linalg.norm(base2) + 1e-8)
        curr = base2.copy()
        for step_idx in range(STEPS_PER_TASK):
            d = step_rng2.randn(EMBEDDING_DIM).astype(np.float32) * 0.1
            n = rng.randn(EMBEDDING_DIM).astype(np.float32) * 0.05
            curr = curr + d + n
            curr /= (np.linalg.norm(curr) + 1e-8)
            step_embeddings.append(curr.copy())

        trace = tracer.trace_reasoning(np.array(step_embeddings), task_id=task.id)

        results.append({
            'task_id': task.id,
            'domain': task.domain,
            'quality_score': quality.to_dict(),
            'quality_delta': quality_delta,
            'n_interventions': len(interventions_fired),
            'interventions_fired': interventions_fired,
            'trace_summary': trace.summary,
            'response': full_response,
        })

    return {
        'iteration': iteration,
        'results': results,
    }


def build_iteration_snapshot(
    iteration_data: dict,
    iteration: int,
    responses: list,
) -> IterationSnapshot:
    """Build a Goodhart detector snapshot from iteration results."""
    results = iteration_data['results']

    qualities = [r['quality_score']['composite'] for r in results]
    intervention_counts = [r['n_interventions'] for r in results]
    total_steps = len(results) * STEPS_PER_TASK

    # Count signatures
    n_terra = 0
    n_fc = 0
    ridge_values = []
    for r in results:
        summary = r.get('trace_summary', {})
        sig_counts = summary.get('signature_counts', {})
        n_terra += sig_counts.get('terra_incognita', 0)
        n_fc += sig_counts.get('framework_collision', 0)
        means = summary.get('feature_means', {})
        if 'ridge_proximity' in means:
            ridge_values.append(means['ridge_proximity'])

    # Per-domain quality
    per_domain = {}
    for r in results:
        domain = r['domain']
        if domain not in per_domain:
            per_domain[domain] = []
        per_domain[domain].append(r['quality_score']['composite'])
    per_domain_mean = {d: float(np.mean(v)) for d, v in per_domain.items()}

    # Weakest domain
    weakest = min(per_domain_mean, key=per_domain_mean.get)

    # Diversity
    diversity = GoodhartDetector.compute_response_diversity(responses)

    return IterationSnapshot(
        iteration=iteration,
        mean_quality=float(np.mean(qualities)),
        std_quality=float(np.std(qualities)),
        mean_ridge_proximity=float(np.mean(ridge_values)) if ridge_values else 0.0,
        response_diversity=diversity,
        intervention_rate=sum(intervention_counts) / total_steps if total_steps > 0 else 0.0,
        n_terra_incognita=n_terra,
        n_framework_collision=n_fc,
        per_domain_quality=per_domain_mean,
        weakest_domain=weakest,
        weakest_quality=per_domain_mean[weakest],
    )


def run_recursive_trials(seed: int = SEED, n_iterations: int = N_ITERATIONS) -> dict:
    """Run the full recursive learning loop."""
    rng = np.random.RandomState(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 5 Recursive Trials — seed={seed}, {n_iterations} iterations")
    print(f"Building reference corpus (n={N_REFERENCE}, dim={EMBEDDING_DIM})...")

    reference = build_reference_corpus(N_REFERENCE, EMBEDDING_DIM, seed)
    tracer = GeometricTracer(reference, k=min(50, N_REFERENCE - 1))
    learner = RecursiveLearner(reference, k=min(50, N_REFERENCE - 1))
    detector = GoodhartDetector(red_flag_threshold=GOODHART_RED_THRESHOLD)

    # Run baseline for quality deltas
    print("Computing baseline scores...")
    from experiments.track4_switches.run_baseline_reasoning import run_baseline
    baseline_output = run_baseline(seed=seed)
    baseline_scores = {
        r['task_id']: r['quality_score']['composite']
        for r in baseline_output['results']
    }

    all_iterations = []
    all_policies = []
    all_goodhart = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n--- Iteration {iteration}/{n_iterations} ---")

        # Run tasks with current policy
        iter_data = run_iteration(learner, tracer, baseline_scores, rng, iteration)

        # Collect responses for diversity
        responses = [r['response'] for r in iter_data['results']]

        # Build snapshot and check Goodhart
        snapshot = build_iteration_snapshot(iter_data, iteration, responses)
        detector.add_snapshot(snapshot)

        goodhart_report = None
        if iteration >= 2:
            goodhart_report = detector.evaluate()
            if goodhart_report.n_red_triggered > 0:
                print(f"  Goodhart WARNING: {goodhart_report.n_red_triggered} red flags "
                      f"[verdict: {goodhart_report.verdict}]")
            else:
                print(f"  Goodhart: {goodhart_report.verdict} "
                      f"({goodhart_report.n_green_triggered} green flags)")

        # Update policy
        policy_state = learner.update_policy()
        if policy_state.dropped_signals:
            print(f"  Policy: dropped {policy_state.dropped_signals}")
        print(f"  Policy: {len(learner.interventions)} interventions remaining")
        print(f"  Quality: {snapshot.mean_quality:.4f} "
              f"(diversity: {snapshot.response_diversity:.3f})")

        # Clear per-iteration experience
        learner.clear_experience()

        # Store
        all_iterations.append({
            'iteration': iteration,
            'snapshot': {
                'mean_quality': snapshot.mean_quality,
                'std_quality': snapshot.std_quality,
                'diversity': snapshot.response_diversity,
                'intervention_rate': snapshot.intervention_rate,
                'n_terra_incognita': snapshot.n_terra_incognita,
                'n_framework_collision': snapshot.n_framework_collision,
                'per_domain_quality': snapshot.per_domain_quality,
            },
            'results': iter_data['results'],
        })
        all_policies.append(policy_state.to_dict())
        if goodhart_report:
            all_goodhart.append(goodhart_report.to_dict())

    # Final output
    output = {
        'config': {
            'seed': seed,
            'n_iterations': n_iterations,
            'embedding_dim': EMBEDDING_DIM,
            'n_reference': N_REFERENCE,
            'steps_per_task': STEPS_PER_TASK,
            'goodhart_red_threshold': GOODHART_RED_THRESHOLD,
        },
        'timestamp': timestamp,
        'iterations': all_iterations,
        'policy_history': all_policies,
        'goodhart_reports': all_goodhart,
        'final_policy': learner.get_current_policy(),
        'baseline_mean_quality': float(np.mean(list(baseline_scores.values()))),
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recursive_trials.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nRecursive trials complete.")
    print(f"Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_recursive_trials()
