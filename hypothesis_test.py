import sys

sys.path.insert(0, "C:/Users/User/mirrorfield")

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from experiments.shared.task_bank import get_task_bank
from experiments.shared.adversarial_task_bank import get_adversarial_task_bank
from experiments.shared.llm_client import GLMReasoningClient
from experiments.shared.embedder import ResponseEmbedder
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track5_recursive.recursive_learner import PolicyOptimizer
from experiments.track4_switches.switch_engine import DEFAULT_INTERVENTIONS

print("=" * 70)
print("HYPOTHESIS TEST: When do interventions help vs hurt?")
print("=" * 70)
print("\nH1: Interventions fundamentally harmful")
print("H2: Too aggressive (sweet spot exists)")
print("H3: Task-type dependent")
print("=" * 70)

print("\n[INIT] Setting up...")
client = GLMReasoningClient(max_tokens=512)
embedder = ResponseEmbedder()

print("[INIT] Building reference corpus...")
ref_texts = build_reference_texts()
reference = embedder.build_reference_corpus(ref_texts)
k = min(50, len(reference) - 1)

STEPS = 3
N_TASKS = 3


def score_quality(response: str) -> float:
    words = response.lower().split()
    word_count = len(words)

    length_score = 1.0 - abs(min(word_count, 600) - 400) / 400
    length_score = max(0, min(1, length_score))

    structure_markers = [
        "first",
        "second",
        "third",
        "finally",
        "however",
        "therefore",
        "conclusion",
        "additionally",
        "furthermore",
    ]
    structure_count = sum(1 for m in structure_markers if m in response.lower())
    structure_score = min(1.0, structure_count / 4)

    if word_count > 0:
        diversity = len(set(words)) / word_count
    else:
        diversity = 0
    diversity_score = min(1.0, diversity * 1.5)

    reasoning_words = [
        "because",
        "since",
        "implies",
        "suggests",
        "indicates",
        "means",
        "shows",
        "evidence",
        "example",
        "consider",
    ]
    reasoning_count = sum(1 for w in reasoning_words if w in response.lower())
    reasoning_score = min(1.0, reasoning_count / 5)

    transitions = ["however", "but", "although", "therefore", "thus", "consequently"]
    transition_count = sum(1 for t in transitions if t in response.lower())
    coherence_score = min(1.0, transition_count / 3)

    return (
        length_score * 0.2
        + structure_score * 0.2
        + diversity_score * 0.2
        + reasoning_score * 0.2
        + coherence_score * 0.2
    )


def run_condition(
    tasks,
    threshold_base: float,
    condition_name: str,
    description: str,
    use_interventions: bool = True,
):
    print(f"\n{'=' * 70}")
    print(f"CONDITION: {condition_name}")
    print(f"{description}")
    print(f"{'=' * 70}")

    optimizer = PolicyOptimizer(
        reference,
        k=k,
        base_interventions=list(DEFAULT_INTERVENTIONS) if use_interventions else [],
        use_enhanced_tracer=True,
        state_threshold_base=threshold_base,
    )
    engine = optimizer.build_engine()

    results = []
    intervention_count = 0
    total_steps = 0
    signals = []

    for i, task in enumerate(tasks):
        print(f"  Task {i + 1}/{len(tasks)}: {task.id[:25]}...", end=" ", flush=True)
        engine.reset_history()
        step_texts = []

        for step_idx in range(STEPS):
            step_text = client.generate_step(
                task_prompt=task.prompt,
                step_index=step_idx,
                total_steps=STEPS,
                previous_steps=step_texts if step_texts else None,
            )

            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            eval_result = engine.evaluate(step_embedding_norm, step_idx)
            total_steps += 1

            if hasattr(eval_result.step_trace, "novelty_signature"):
                signals.append(eval_result.step_trace.novelty_signature)

            if eval_result.triggered:
                intervention_count += 1

            step_texts.append(step_text)

        quality = score_quality("\n\n".join(step_texts))
        results.append({"task_id": task.id, "quality": quality})
        print(f"q={quality:.3f}")

    qualities = [r["quality"] for r in results]
    signal_counts = {s: signals.count(s) for s in set(signals)}
    rate = intervention_count / total_steps if total_steps > 0 else 0

    print(f"\n  Mean: {np.mean(qualities):.4f}, Std: {np.std(qualities):.4f}")
    print(
        f"  Intervention rate: {rate * 100:.1f}% ({intervention_count}/{total_steps})"
    )

    return {
        "mean": float(np.mean(qualities)),
        "std": float(np.std(qualities)),
        "rate": rate,
        "signals": signal_counts,
        "qualities": qualities,
    }


standard_tasks = get_task_bank()[:N_TASKS]
adversarial_tasks = get_adversarial_task_bank()[:N_TASKS]

print(
    f"\n[TASKS] Standard: {len(standard_tasks)}, Adversarial: {len(adversarial_tasks)}"
)

# CONDITION A: Baseline (no interventions)
result_a = run_condition(
    standard_tasks,
    threshold_base=0.15,
    condition_name="A: BASELINE (0% interventions)",
    description="No interventions - pure generation",
    use_interventions=False,
)

# CONDITION B: Conservative threshold
result_b = run_condition(
    standard_tasks,
    threshold_base=0.25,
    condition_name="B: CONSERVATIVE (15% target)",
    description="Higher threshold, fewer interventions",
)

# CONDITION C: Adversarial tasks
result_c = run_condition(
    adversarial_tasks,
    threshold_base=0.15,
    condition_name="C: ADVERSARIAL TASKS (25% target)",
    description="Same threshold, harder prompts",
)

# ANALYSIS
print("\n" + "=" * 70)
print("HYPOTHESIS TEST RESULTS")
print("=" * 70)

print(f"\n{'Condition':<30} {'Quality':>10} {'Rate':>10} {'vs Baseline':>12}")
print("-" * 65)
print(
    f"{'A: Baseline (0%)':<30} {result_a['mean']:>10.4f} {result_a['rate'] * 100:>9.1f}% {'—':>12}"
)
print(
    f"{'B: Conservative (~15%)':<30} {result_b['mean']:>10.4f} {result_b['rate'] * 100:>9.1f}% {result_b['mean'] - result_a['mean']:>+12.4f}"
)
print(
    f"{'C: Adversarial (~25%)':<30} {result_c['mean']:>10.4f} {result_c['rate'] * 100:>9.1f}% {result_c['mean'] - result_a['mean']:>+12.4f}"
)

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

delta_b = result_b["mean"] - result_a["mean"]
delta_c = result_c["mean"] - result_a["mean"]

if delta_b > 0.02:
    print("\n[+] H2 SUPPORTED: Conservative rate improves quality")
    print(
        f"    Interventions help at {result_b['rate'] * 100:.0f}% rate (+{delta_b * 100:.1f}%)"
    )
elif delta_b > -0.02 and result_b["rate"] < 0.25:
    print("\n[~] H2 INCONCLUSIVE: Conservative rate neutral")
    print("    May need even lower rate")
else:
    print("\n[-] H2 REJECTED: Even conservative rate hurts")

if delta_c > delta_b + 0.05:
    print("\n[+] H3 SUPPORTED: Adversarial tasks benefit more")
elif delta_c < delta_b - 0.05:
    print("\n[-] H3 REJECTED: Adversarial tasks hurt more")
else:
    print("\n[~] H3 INCONCLUSIVE: Similar effect on both task types")

if delta_b < -0.02 and delta_c < -0.02:
    print("\n[!] H1 CANNOT BE REJECTED: Interventions may be fundamentally harmful")
    print("    Consider pivoting to training data filtering instead")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
