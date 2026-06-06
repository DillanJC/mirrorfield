"""
Compare task banks: Standard vs Adversarial.

Runs 2 iterations each on:
1. Standard task bank (16 tasks)
2. Adversarial task bank (20 tasks)

Reports signal diversity, intervention rates, and policy learning.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from experiments.shared.task_bank import get_task_bank, Task
from experiments.shared.adversarial_task_bank import get_adversarial_task_bank
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track5_recursive.recursive_learner import PolicyOptimizer
from experiments.track5_recursive.goodhart_detector import GoodhartDetector
from experiments.shared.quality_scorer import score_response


STEPS_PER_TASK = 4
N_ITERATIONS = 2


def run_iteration(
    client: GLMReasoningClient,
    embedder: ResponseEmbedder,
    optimizer: PolicyOptimizer,
    tasks: List[Task],
    iteration: int,
    task_bank_name: str,
) -> Dict[str, Any]:
    """Run one iteration over all tasks."""
    engine = optimizer.build_engine()
    results = []
    all_signals = []
    all_states = []
    intervention_count = 0
    total_steps = 0

    for task in tasks:
        engine.reset_history()
        step_texts = []
        task_signals = []
        task_states = []

        for step_idx in range(STEPS_PER_TASK):
            step_text = client.generate_step(
                task_prompt=task.prompt,
                step_index=step_idx,
                total_steps=STEPS_PER_TASK,
                previous_steps=step_texts if step_texts else None,
                intervention_text=None,  # No interventions for this test
            )
            step_texts.append(step_text)

            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            eval_result = engine.evaluate(step_embedding_norm, step_idx)
            total_steps += 1

            # Collect signal data
            if hasattr(eval_result.step_trace, "novelty_signature"):
                sig = eval_result.step_trace.novelty_signature
                task_signals.append(sig)
                all_signals.append(sig)

            if hasattr(eval_result.step_trace, "dominant_state"):
                state = eval_result.step_trace.dominant_state
                task_states.append(state)
                all_states.append(state)

            if eval_result.triggered:
                intervention_count += 1

        full_response = "\n\n".join(step_texts)
        quality_score = score_response(full_response)

        results.append(
            {
                "task_id": task.id,
                "domain": task.domain,
                "quality": quality_score.composite,
                "signals": task_signals,
                "states": task_states,
                "adversarial_type": getattr(task, "adversarial_type", "standard"),
            }
        )

    # Compute summary
    qualities = [r["quality"] for r in results]
    signal_counts = {}
    for s in all_signals:
        signal_counts[s] = signal_counts.get(s, 0) + 1
    state_counts = {}
    for s in all_states:
        state_counts[s] = state_counts.get(s, 0) + 1

    return {
        "iteration": iteration,
        "task_bank": task_bank_name,
        "n_tasks": len(tasks),
        "total_steps": total_steps,
        "intervention_count": intervention_count,
        "intervention_rate": intervention_count / total_steps if total_steps > 0 else 0,
        "mean_quality": float(np.mean(qualities)),
        "std_quality": float(np.std(qualities)),
        "signal_counts": signal_counts,
        "unique_signals": len(signal_counts),
        "state_counts": state_counts,
        "unique_states": len(state_counts),
        "results": results,
    }


def main():
    print("=" * 70)
    print("TASK BANK COMPARISON: Standard vs Adversarial")
    print("=" * 70)

    # Initialize
    print("\nInitializing GLM client and embedder...")
    client = GLMReasoningClient(max_tokens=2048)
    embedder = ResponseEmbedder()

    print("Building reference corpus...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    k = min(50, len(reference) - 1)

    # Load task banks
    standard_tasks = get_task_bank()
    adversarial_tasks = get_adversarial_task_bank()

    print(f"\nTask banks loaded:")
    print(f"  Standard: {len(standard_tasks)} tasks")
    print(f"  Adversarial: {len(adversarial_tasks)} tasks")

    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "standard": [],
        "adversarial": [],
    }

    # =========================================================================
    # Run Standard Task Bank
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING STANDARD TASK BANK")
    print("=" * 70)

    optimizer_std = PolicyOptimizer(
        reference,
        k=k,
        use_enhanced_tracer=True,
        state_threshold_base=0.15,
    )

    for iteration in range(N_ITERATIONS):
        print(f"\n--- Standard Iteration {iteration + 1}/{N_ITERATIONS} ---")
        result = run_iteration(
            client, embedder, optimizer_std, standard_tasks, iteration + 1, "standard"
        )
        all_results["standard"].append(result)

        print(f"  Quality: {result['mean_quality']:.4f}")
        print(
            f"  Interventions: {result['intervention_count']}/{result['total_steps']} ({result['intervention_rate'] * 100:.1f}%)"
        )
        print(f"  Signals: {result['signal_counts']}")
        print(f"  States: {result['state_counts']}")

    # =========================================================================
    # Run Adversarial Task Bank
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ADVERSARIAL TASK BANK")
    print("=" * 70)

    optimizer_adv = PolicyOptimizer(
        reference,
        k=k,
        use_enhanced_tracer=True,
        state_threshold_base=0.15,
    )

    for iteration in range(N_ITERATIONS):
        print(f"\n--- Adversarial Iteration {iteration + 1}/{N_ITERATIONS} ---")
        result = run_iteration(
            client,
            embedder,
            optimizer_adv,
            adversarial_tasks,
            iteration + 1,
            "adversarial",
        )
        all_results["adversarial"].append(result)

        print(f"  Quality: {result['mean_quality']:.4f}")
        print(
            f"  Interventions: {result['intervention_count']}/{result['total_steps']} ({result['intervention_rate'] * 100:.1f}%)"
        )
        print(f"  Signals: {result['signal_counts']}")
        print(f"  States: {result['state_counts']}")

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    std_final = all_results["standard"][-1]
    adv_final = all_results["adversarial"][-1]

    print(f"\n{'Metric':<25} {'Standard':>15} {'Adversarial':>15} {'Delta':>10}")
    print("-" * 70)
    print(
        f"{'Intervention Rate':<25} {std_final['intervention_rate'] * 100:>14.1f}% {adv_final['intervention_rate'] * 100:>14.1f}% {(adv_final['intervention_rate'] - std_final['intervention_rate']) * 100:>+9.1f}%"
    )
    print(
        f"{'Unique Signals':<25} {std_final['unique_signals']:>15} {adv_final['unique_signals']:>15} {adv_final['unique_signals'] - std_final['unique_signals']:>+10}"
    )
    print(
        f"{'Unique States':<25} {std_final['unique_states']:>15} {adv_final['unique_states']:>15} {adv_final['unique_states'] - std_final['unique_states']:>+10}"
    )
    print(
        f"{'Mean Quality':<25} {std_final['mean_quality']:>15.4f} {adv_final['mean_quality']:>15.4f} {adv_final['mean_quality'] - std_final['mean_quality']:>+10.4f}"
    )

    print("\n--- Signal Distribution ---")
    all_sig_types = set(std_final["signal_counts"].keys()) | set(
        adv_final["signal_counts"].keys()
    )
    for sig in sorted(all_sig_types):
        std_cnt = std_final["signal_counts"].get(sig, 0)
        adv_cnt = adv_final["signal_counts"].get(sig, 0)
        print(f"  {sig:<25} {std_cnt:>15} {adv_cnt:>15} {adv_cnt - std_cnt:>+10}")

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "task_bank_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    main()
