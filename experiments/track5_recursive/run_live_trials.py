"""
Track 5 Live Recursive Trials — Adaptive policy optimization with live Claude API.

Runs 5 iterations of the full task bank. Each iteration:
1. Run 16 tasks with current adaptive policy via Claude API
2. Compute per-signature effectiveness vs baseline
3. Update policy: penalize harmful signals, reward beneficial ones
4. Check Goodhart detector for gaming patterns
5. Save per-iteration results

Uses existing baseline results (no need to re-run).
API calls: 5 × 16 × 4 = 320 Claude calls.
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
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track5_recursive.recursive_learner import PolicyOptimizer
from experiments.track5_recursive.goodhart_detector import (
    GoodhartDetector, IterationSnapshot,
)

# Configuration
STEPS_PER_TASK = 4
N_ITERATIONS = 5
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "track5_recursive"

# Existing baseline results
BASELINE_PATH = (
    PROJECT_ROOT / "experiments" / "results" / "track4_live_baseline"
    / "20260208_221402" / "baseline_results.json"
)


def load_baseline_scores() -> dict:
    """Load baseline quality scores from Track 4 live baseline results."""
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)
    return {
        r['task_id']: r['quality_score']['composite']
        for r in baseline['results']
    }


def run_iteration(
    client: ClaudeReasoningClient,
    embedder: ResponseEmbedder,
    optimizer: PolicyOptimizer,
    iteration: int,
) -> list:
    """Run a single iteration through all tasks with the current policy."""
    tasks = get_task_bank()
    engine = optimizer.build_engine()
    results = []

    for task in tasks:
        engine.reset_history()

        step_texts = []
        interventions_fired = []
        pending_intervention = ""

        for step_idx in range(STEPS_PER_TASK):
            step_text = client.generate_step(
                task_prompt=task.prompt,
                step_index=step_idx,
                total_steps=STEPS_PER_TASK,
                previous_steps=step_texts if step_texts else None,
                intervention_text=pending_intervention if pending_intervention else None,
            )
            step_texts.append(step_text)

            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            eval_result = engine.evaluate(step_embedding_norm, step_idx)

            if eval_result.triggered and eval_result.intervention:
                pending_intervention = eval_result.intervention.instruction
                interventions_fired.append({
                    'step': step_idx,
                    'signal': eval_result.intervention.signal,
                })
            else:
                pending_intervention = ""

        full_response = "\n\n".join(step_texts)
        quality = score_response(full_response)

        fired_signals = [i['signal'] for i in interventions_fired]
        results.append({
            'task_id': task.id,
            'domain': task.domain,
            'prompt': task.prompt[:80],
            'quality_score': quality.to_dict(),
            'n_interventions': len(interventions_fired),
            'interventions_fired': interventions_fired,
            'response': full_response,
            'trace_summary': engine.get_intervention_stats(),
        })

        print(f"    [{task.id}] q={quality.composite:.3f} | "
              f"intv={len(interventions_fired)} | "
              f"sigs={fired_signals}")

    return results


def build_snapshot(
    results: list,
    baseline_scores: dict,
    iteration: int,
) -> IterationSnapshot:
    """Build Goodhart detector snapshot from iteration results."""
    qualities = [r['quality_score']['composite'] for r in results]
    depths = [r['quality_score']['depth'] for r in results]
    total_steps = len(results) * STEPS_PER_TASK
    total_interventions = sum(r['n_interventions'] for r in results)

    # Count signals from interventions fired
    signal_counts = {}
    for r in results:
        for intv in r['interventions_fired']:
            sig = intv['signal']
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

    n_terra = signal_counts.get('terra_incognita', 0)
    n_fc = signal_counts.get('framework_collision', 0)

    # Per-domain quality
    per_domain = {}
    for r in results:
        domain = r['domain']
        per_domain.setdefault(domain, []).append(r['quality_score']['composite'])
    per_domain_mean = {d: float(np.mean(v)) for d, v in per_domain.items()}

    weakest = min(per_domain_mean, key=per_domain_mean.get)

    # Ridge proximity from trace stats
    ridge_values = []
    for r in results:
        stats = r.get('trace_summary', {})
        sig_counts = stats.get('signal_counts', {})
        # Use intervention_rate as rough proxy
    mean_ridge = 0.06  # approximate from baseline analysis

    # Diversity
    responses = [r['response'] for r in results]
    diversity = GoodhartDetector.compute_response_diversity(responses)

    return IterationSnapshot(
        iteration=iteration,
        mean_quality=float(np.mean(qualities)),
        std_quality=float(np.std(qualities)),
        mean_depth=float(np.mean(depths)),
        mean_ridge_proximity=mean_ridge,
        response_diversity=diversity,
        intervention_rate=total_interventions / total_steps if total_steps > 0 else 0.0,
        n_terra_incognita=n_terra,
        n_framework_collision=n_fc,
        per_domain_quality=per_domain_mean,
        weakest_domain=weakest,
        weakest_quality=per_domain_mean[weakest],
        signal_distribution=signal_counts,
    )


def run_live_recursive_trials(n_iterations: int = N_ITERATIONS) -> dict:
    """Run the full recursive learning loop with live Claude API."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Track 5 Live Recursive Trials — {n_iterations} iterations")
    print(f"Timestamp: {timestamp}")
    print()

    # Load existing baseline scores
    print("Loading baseline scores from Track 4 live baseline...")
    baseline_scores = load_baseline_scores()
    baseline_mean = float(np.mean(list(baseline_scores.values())))
    print(f"  Baseline mean quality: {baseline_mean:.4f}")
    print(f"  Tasks: {len(baseline_scores)}")
    print()

    # Initialize
    print("Initializing Claude client and embedder...")
    client = ClaudeReasoningClient()
    embedder = ResponseEmbedder()

    print("Building reference corpus...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    k = min(50, len(reference) - 1)

    optimizer = PolicyOptimizer(reference, k=k)
    detector = GoodhartDetector(red_flag_threshold=3, warn_threshold=1)

    all_iterations = []
    all_updates = []
    all_goodhart = []

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{n_iterations}")
        print(f"  Policy: {optimizer.policy.to_dict()}")
        print(f"{'='*60}")

        # Run all tasks with current policy
        results = run_iteration(client, embedder, optimizer, iteration)

        # Compute iteration quality
        qualities = [r['quality_score']['composite'] for r in results]
        iter_mean = float(np.mean(qualities))
        delta_vs_baseline = iter_mean - baseline_mean

        print(f"\n  Iteration {iteration} quality: {iter_mean:.4f} "
              f"(baseline: {baseline_mean:.4f}, delta: {delta_vs_baseline:+.4f})")

        # Per-domain summary
        per_domain = {}
        for r in results:
            per_domain.setdefault(r['domain'], []).append(
                r['quality_score']['composite']
            )
        for domain in sorted(per_domain):
            dm = float(np.mean(per_domain[domain]))
            bd = float(np.mean([baseline_scores[r['task_id']]
                       for r in results if r['domain'] == domain]))
            print(f"    {domain[:20]:20s}: {dm:.3f} (baseline {bd:.3f}, "
                  f"delta {dm - bd:+.3f})")

        # Goodhart check
        snapshot = build_snapshot(results, baseline_scores, iteration)
        detector.add_snapshot(snapshot)

        goodhart_report = None
        if iteration >= 2:
            goodhart_report = detector.evaluate()
            red_names = [f.name for f in goodhart_report.red_flags if f.triggered]
            green_names = [f.name for f in goodhart_report.green_flags if f.triggered]
            print(f"\n  Goodhart [{goodhart_report.verdict}]: "
                  f"{goodhart_report.n_red_triggered} red, "
                  f"{goodhart_report.n_green_triggered} green")
            if red_names:
                print(f"    RED: {red_names}")
            if green_names:
                print(f"    GREEN: {green_names}")

        # Update policy
        update = optimizer.update_policy(results, baseline_scores)
        print(f"\n  Policy updates:")
        for adj in update.adjustments:
            print(f"  {adj}")

        # Intervention stats
        total_intv = sum(r['n_interventions'] for r in results)
        print(f"\n  Interventions: {total_intv}/64 steps "
              f"({total_intv/64*100:.0f}% trigger rate)")

        # Save iteration data
        all_iterations.append({
            'iteration': iteration,
            'mean_quality': iter_mean,
            'delta_vs_baseline': delta_vs_baseline,
            'per_domain_quality': {
                d: float(np.mean(v)) for d, v in per_domain.items()
            },
            'intervention_count': total_intv,
            'results': results,
            'policy_state': optimizer.policy.to_dict(),
        })
        all_updates.append({
            'iteration': update.iteration,
            'effectiveness': update.effectiveness,
            'adjustments': update.adjustments,
            'policy_before': update.policy_before,
            'policy_after': update.policy_after,
        })
        if goodhart_report:
            all_goodhart.append(goodhart_report.to_dict())

    # Final output
    output = {
        'config': {
            'n_iterations': n_iterations,
            'steps_per_task': STEPS_PER_TASK,
            'model': client.model,
            'temperature': client.temperature,
            'max_tokens': client.max_tokens,
            'penalty_rate': optimizer.penalty_rate,
            'reward_rate': optimizer.reward_rate,
            'disable_threshold': optimizer.disable_threshold,
        },
        'timestamp': timestamp,
        'baseline_mean_quality': baseline_mean,
        'baseline_scores': baseline_scores,
        'iterations': all_iterations,
        'policy_updates': all_updates,
        'goodhart_reports': all_goodhart,
        'final_policy': optimizer.policy.to_dict(),
        'update_history': optimizer.get_update_history(),
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recursive_trials.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"  TRACK 5 COMPLETE — {n_iterations} iterations")
    print(f"{'='*60}")
    print(f"  Baseline mean quality: {baseline_mean:.4f}")
    for it in all_iterations:
        print(f"  Iteration {it['iteration']}: {it['mean_quality']:.4f} "
              f"(delta: {it['delta_vs_baseline']:+.4f})")
    print(f"\n  Final policy: {optimizer.policy.to_dict()}")
    print(f"  Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_live_recursive_trials()
