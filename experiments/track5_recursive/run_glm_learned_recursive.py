"""
Track 5 GLM Learned Recursive v2 — Phase 4 states + balanced rates + 10 iterations.

v2 upgrades from v1:
1. EnhancedTracer with Phase 4 cognitive state mapping (14 features, 6 states)
2. Balanced penalty/reward rates (1.15 / 1/1.15) — eliminates penalty ratchet
3. 10 iterations (was 5) — enough for policy equilibrium
4. Early stopping on Goodhart FAIL
5. Per-iteration timing
6. Multiplier drift detection

Core question: With richer geometry and balanced learning, can the recursive
learner find a stable, beneficial intervention policy?

API calls: 10 x 16 x 4 = 640 GLM calls via Z.ai.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared.task_bank import get_task_bank
from experiments.shared.quality_scorer import score_response
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track4_switches.mirror_interventions import MIRROR_INTERVENTIONS
from experiments.track5_recursive.recursive_learner import PolicyOptimizer
from experiments.track5_recursive.goodhart_detector import (
    GoodhartDetector,
    IterationSnapshot,
)

# Configuration
STEPS_PER_TASK = 4
N_ITERATIONS = 2  # Temporary: testing fixes before full run
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "glm_learned_recursive_v2"

# GLM baseline for comparison
GLM_BASELINE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "glm_baseline"
    / "20260215_083005"
    / "glm_baseline_results.json"
)


def load_glm_baseline_scores() -> dict:
    with open(GLM_BASELINE_PATH) as f:
        baseline = json.load(f)
    return {r["task_id"]: r["quality_score"]["composite"] for r in baseline["results"]}


def run_iteration(
    client: GLMReasoningClient,
    embedder: ResponseEmbedder,
    optimizer: PolicyOptimizer,
    iteration: int,
) -> list:
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
                intervention_text=pending_intervention
                if pending_intervention
                else None,
            )
            step_texts.append(step_text)

            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            eval_result = engine.evaluate(step_embedding_norm, step_idx)

            if eval_result.triggered and eval_result.intervention:
                pending_intervention = eval_result.intervention.instruction
                interventions_fired.append(
                    {
                        "step": step_idx,
                        "signal": eval_result.intervention.signal,
                    }
                )
            else:
                pending_intervention = ""

        full_response = "\n\n".join(step_texts)
        quality = score_response(full_response)

        fired_signals = [i["signal"] for i in interventions_fired]
        results.append(
            {
                "task_id": task.id,
                "domain": task.domain,
                "prompt": task.prompt[:80],
                "quality_score": quality.to_dict(),
                "n_interventions": len(interventions_fired),
                "interventions_fired": interventions_fired,
                "response": full_response,
                "trace_summary": engine.get_intervention_stats(),
            }
        )

        print(
            f"    [{task.id}] q={quality.composite:.3f} | "
            f"intv={len(interventions_fired)} | "
            f"sigs={fired_signals}"
        )

    return results


def build_snapshot(
    results: list,
    baseline_scores: dict,
    iteration: int,
    policy_multipliers: Optional[Dict[str, float]] = None,
) -> IterationSnapshot:
    qualities = [r["quality_score"]["composite"] for r in results]
    depths = [r["quality_score"]["depth"] for r in results]
    total_steps = len(results) * STEPS_PER_TASK
    total_interventions = sum(r["n_interventions"] for r in results)

    signal_counts = {}
    for r in results:
        for intv in r["interventions_fired"]:
            sig = intv["signal"]
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

    n_terra = signal_counts.get("terra_incognita", 0)
    n_fc = signal_counts.get("framework_collision", 0)

    per_domain = {}
    for r in results:
        domain = r["domain"]
        per_domain.setdefault(domain, []).append(r["quality_score"]["composite"])
    per_domain_mean = {d: float(np.mean(v)) for d, v in per_domain.items()}

    weakest = min(per_domain_mean, key=per_domain_mean.get)
    mean_ridge = 0.06

    responses = [r["response"] for r in results]
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
        policy_multipliers=policy_multipliers or {},
    )


def run_learned_recursive(n_iterations: int = N_ITERATIONS) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_start = time.time()

    print("=" * 60)
    print("MIRRORFIELD — GLM LEARNED RECURSIVE v2")
    print("  Phase 4 states + balanced rates + 10 iterations")
    print(f"Timestamp: {timestamp}")
    print(f"Iterations: {n_iterations}")
    print(f"API calls: {n_iterations} x 16 x 4 = {n_iterations * 64}")
    print("=" * 60)

    # Load baseline
    print("\nLoading GLM baseline scores...")
    baseline_scores = load_glm_baseline_scores()
    baseline_mean = float(np.mean(list(baseline_scores.values())))
    print(f"  GLM Baseline mean quality: {baseline_mean:.4f}")

    # Initialize
    print("Initializing GLM client and embedder...")
    client = GLMReasoningClient(max_tokens=2048)
    embedder = ResponseEmbedder()

    print("Building reference corpus...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    k = min(50, len(reference) - 1)

    # Key: pass mirror interventions + enable enhanced tracer
    print("Initializing PolicyOptimizer with EnhancedTracer...")
    optimizer = PolicyOptimizer(
        reference,
        k=k,
        base_interventions=list(MIRROR_INTERVENTIONS),
        use_enhanced_tracer=True,
        state_threshold_base=0.15,  # Lower threshold - scores are ~0.25 dominant
    )
    detector = GoodhartDetector(red_flag_threshold=3, warn_threshold=1)

    # Log enhanced tracer config
    enhanced_config = {}
    if optimizer._enhanced_tracer is not None:
        enhanced_config = optimizer._enhanced_tracer.get_config()
        print(
            f"  EnhancedTracer: {enhanced_config['n_calibration_samples']} "
            f"calibration samples, {enhanced_config['n_states']} states"
        )

    all_iterations = []
    all_updates = []
    all_goodhart = []
    early_stopped = False

    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"  ITERATION {iteration}/{n_iterations}")
        print(f"  Policy: {optimizer.policy.to_dict()}")
        print(f"{'=' * 60}")

        results = run_iteration(client, embedder, optimizer, iteration)

        qualities = [r["quality_score"]["composite"] for r in results]
        iter_mean = float(np.mean(qualities))
        delta_vs_baseline = iter_mean - baseline_mean
        iter_elapsed = time.time() - iter_start

        print(
            f"\n  Iteration {iteration} quality: {iter_mean:.4f} "
            f"(baseline: {baseline_mean:.4f}, delta: {delta_vs_baseline:+.4f})"
        )
        print(f"  Iteration time: {iter_elapsed:.1f}s")

        per_domain = {}
        for r in results:
            per_domain.setdefault(r["domain"], []).append(
                r["quality_score"]["composite"]
            )
        for domain in sorted(per_domain):
            dm = float(np.mean(per_domain[domain]))
            bd = float(
                np.mean(
                    [
                        baseline_scores[r["task_id"]]
                        for r in results
                        if r["domain"] == domain
                    ]
                )
            )
            print(
                f"    {domain[:20]:20s}: {dm:.3f} (baseline {bd:.3f}, "
                f"delta {dm - bd:+.3f})"
            )

        # Goodhart check (pass policy multipliers for drift detection)
        snapshot = build_snapshot(
            results,
            baseline_scores,
            iteration,
            policy_multipliers=dict(optimizer.policy.threshold_multipliers),
        )
        detector.add_snapshot(snapshot)

        goodhart_report = None
        if iteration >= 2:
            goodhart_report = detector.evaluate()
            red_names = [f.name for f in goodhart_report.red_flags if f.triggered]
            green_names = [f.name for f in goodhart_report.green_flags if f.triggered]
            print(
                f"\n  Goodhart [{goodhart_report.verdict}]: "
                f"{goodhart_report.n_red_triggered} red, "
                f"{goodhart_report.n_green_triggered} green"
            )
            if red_names:
                print(f"    RED: {red_names}")
            if green_names:
                print(f"    GREEN: {green_names}")

            # Early stopping on FAIL
            if goodhart_report.verdict == "FAIL":
                print(
                    f"\n  *** EARLY STOPPING: Goodhart FAIL at iteration {iteration} ***"
                )
                early_stopped = True

        # Update policy
        update = optimizer.update_policy(results, baseline_scores)
        print(f"\n  Policy updates:")
        for adj in update.adjustments:
            print(f"  {adj}")

        total_intv = sum(r["n_interventions"] for r in results)
        print(
            f"\n  Interventions: {total_intv}/64 steps "
            f"({total_intv / 64 * 100:.0f}% trigger rate)"
        )

        all_iterations.append(
            {
                "iteration": iteration,
                "mean_quality": iter_mean,
                "delta_vs_baseline": delta_vs_baseline,
                "per_domain_quality": {
                    d: float(np.mean(v)) for d, v in per_domain.items()
                },
                "intervention_count": total_intv,
                "results": results,
                "policy_state": optimizer.policy.to_dict(),
                "elapsed_seconds": iter_elapsed,
            }
        )
        all_updates.append(
            {
                "iteration": update.iteration,
                "effectiveness": update.effectiveness,
                "adjustments": update.adjustments,
                "policy_before": update.policy_before,
                "policy_after": update.policy_after,
            }
        )
        if goodhart_report:
            all_goodhart.append(goodhart_report.to_dict())

        if early_stopped:
            break

    total_elapsed = time.time() - run_start

    # Final output
    output = {
        "config": {
            "version": "v2",
            "n_iterations": n_iterations,
            "iterations_completed": len(all_iterations),
            "early_stopped": early_stopped,
            "steps_per_task": STEPS_PER_TASK,
            "model": client.model,
            "backend": "glm",
            "temperature": client.temperature,
            "max_tokens": client.max_tokens,
            "penalty_rate": optimizer.penalty_rate,
            "reward_rate": optimizer.reward_rate,
            "penalty_reward_product": optimizer.penalty_rate * optimizer.reward_rate,
            "disable_threshold": optimizer.disable_threshold,
            "multiplier_clamp": [optimizer.MULTIPLIER_MIN, optimizer.MULTIPLIER_MAX],
            "classifier_mode": "enhanced_phase4",
            "intervention_style": "mirror",
            "use_enhanced_tracer": optimizer.use_enhanced_tracer,
            "state_threshold_base": optimizer.state_threshold_base,
            "enhanced_tracer_config": enhanced_config,
        },
        "timestamp": timestamp,
        "total_elapsed_seconds": total_elapsed,
        "baseline_mean_quality": baseline_mean,
        "baseline_scores": baseline_scores,
        "iterations": all_iterations,
        "policy_updates": all_updates,
        "goodhart_reports": all_goodhart,
        "final_policy": optimizer.policy.to_dict(),
        "update_history": optimizer.get_update_history(),
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "glm_learned_recursive_v2_trials.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"  RECURSIVE LEARNER v2 COMPLETE — "
        f"{len(all_iterations)}/{n_iterations} iterations "
        f"({'early stopped' if early_stopped else 'full run'})"
    )
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Baseline mean quality: {baseline_mean:.4f}")
    for it in all_iterations:
        print(
            f"  Iter {it['iteration']}: quality={it['mean_quality']:.4f} "
            f"(delta={it['delta_vs_baseline']:+.4f}, "
            f"intv={it['intervention_count']}, "
            f"time={it['elapsed_seconds']:.1f}s)"
        )
    print(f"\n  Final policy: {optimizer.policy.to_dict()}")
    print(
        f"  Penalty*Reward product: {optimizer.penalty_rate * optimizer.reward_rate:.6f}"
    )
    print(f"  Results saved to: {out_path}")

    return output


if __name__ == "__main__":
    run_learned_recursive()
