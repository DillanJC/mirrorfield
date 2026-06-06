"""
Track 4 GLM Learned Classifier — Re-run switched experiments with data-driven classifier.

The original switched experiments used a hand-coded classifier that returned
100% "coherent", so interventions only fired from the low_pr fallback.

This script re-runs both directive and mirror interventions with the new
learned classifier that actually detects geometric anomalies.

API calls: 2 x (16 tasks x 4 steps) = 128 GLM calls total.

Results are saved with classifier_mode="learned" metadata for comparison
against the old "rules" baseline.
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
from experiments.track4_switches.switch_engine import SwitchEngine, Intervention
from experiments.track4_switches.run_live_baseline import build_reference_texts
from experiments.track4_switches.mirror_interventions import MIRROR_INTERVENTIONS

STEPS_PER_TASK = 4


def run_switched_experiment(
    client: GLMReasoningClient,
    embedder: ResponseEmbedder,
    engine: SwitchEngine,
    tasks: list,
    label: str,
) -> list:
    """Run a full switched experiment and return per-task results."""
    results = []

    for task in tasks:
        print(f"  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        engine.reset_history()

        step_texts = []
        step_records = []
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

            step_record = {
                'step_index': step_idx,
                'text': step_text,
                'intervention_applied': pending_intervention if pending_intervention else None,
                'intervention_detected': eval_result.to_dict(),
            }
            step_records.append(step_record)

            if eval_result.triggered and eval_result.intervention:
                pending_intervention = eval_result.intervention.instruction
                interventions_fired.append({
                    'step': step_idx,
                    'signal': eval_result.intervention.signal,
                    'instruction': eval_result.intervention.instruction,
                })
            else:
                pending_intervention = ""

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

    return results


def build_summary(results: list, reference: np.ndarray, client, extra: dict) -> dict:
    """Build experiment summary dict."""
    scores = [r['quality_score']['composite'] for r in results]
    total_interventions = sum(r['n_interventions'] for r in results)

    summary = {
        'n_tasks': len(results),
        'steps_per_task': STEPS_PER_TASK,
        'embedding_dim': reference.shape[1],
        'n_reference': len(reference),
        'model': client.model,
        'backend': 'glm',
        'temperature': client.temperature,
        'mean_quality': float(np.mean(scores)),
        'std_quality': float(np.std(scores)),
        'total_interventions': total_interventions,
        'mean_interventions_per_task': total_interventions / len(results),
        'per_domain': {},
        **extra,
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

    return summary


def run_learned_classifier_experiments():
    """Run directive and mirror switched experiments with the learned classifier."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("MIRRORFIELD — GLM LEARNED CLASSIFIER EXPERIMENTS")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Initialize shared components
    print("\nInitializing GLM client and embedder...")
    client = GLMReasoningClient(max_tokens=2048)
    embedder = ResponseEmbedder()

    print("Building reference corpus from task bank...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    n_ref = len(reference)
    k = min(50, n_ref - 1)

    tasks = get_task_bank()

    # Baselines for comparison
    glm_baseline_quality = 0.812  # from previous experiment
    glm_directive_old_quality = 0.778  # old classifier, directive
    glm_mirror_old_quality = 0.789  # old classifier, mirror

    # ========================================
    # EXPERIMENT 1: Directive + Learned Classifier
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Directive Interventions + Learned Classifier")
    print(f"API calls: {len(tasks)} tasks x {STEPS_PER_TASK} steps = {len(tasks) * STEPS_PER_TASK}")
    print("=" * 60 + "\n")

    engine_directive = SwitchEngine(reference, k=k, use_live_calibration=True)
    results_directive = run_switched_experiment(
        client, embedder, engine_directive, tasks, "directive_learned"
    )

    summary_directive = build_summary(results_directive, reference, client, {
        'timestamp': timestamp,
        'intervention_style': 'directive',
        'classifier_mode': 'learned',
    })

    # Save directive results
    out_dir_d = PROJECT_ROOT / "experiments" / "results" / "glm_learned_directive" / timestamp
    out_dir_d.mkdir(parents=True, exist_ok=True)
    out_path_d = out_dir_d / "glm_learned_directive_results.json"

    with open(out_path_d, 'w') as f:
        json.dump({
            'config': {
                'embedding_dim': reference.shape[1],
                'n_reference': n_ref,
                'steps_per_task': STEPS_PER_TASK,
                'model': client.model,
                'backend': 'glm',
                'temperature': client.temperature,
                'max_tokens': client.max_tokens,
                'classifier_mode': 'learned',
                'intervention_style': 'directive',
                'policy_table': engine_directive.get_policy_table(),
            },
            'summary': summary_directive,
            'results': results_directive,
        }, f, indent=2)

    d_quality = summary_directive['mean_quality']
    d_interventions = summary_directive['total_interventions']
    print(f"\nDirective + Learned: quality={d_quality:.3f}, interventions={d_interventions}")
    print(f"  vs baseline:        {d_quality - glm_baseline_quality:+.3f}")
    print(f"  vs old directive:   {d_quality - glm_directive_old_quality:+.3f}")
    print(f"Saved to: {out_path_d}")

    # ========================================
    # EXPERIMENT 2: Mirror + Learned Classifier
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Mirror Interventions + Learned Classifier")
    print(f"API calls: {len(tasks)} tasks x {STEPS_PER_TASK} steps = {len(tasks) * STEPS_PER_TASK}")
    print("=" * 60 + "\n")

    engine_mirror = SwitchEngine(
        reference, k=k,
        interventions=list(MIRROR_INTERVENTIONS),
        use_live_calibration=True,
    )
    results_mirror = run_switched_experiment(
        client, embedder, engine_mirror, tasks, "mirror_learned"
    )

    summary_mirror = build_summary(results_mirror, reference, client, {
        'timestamp': timestamp,
        'intervention_style': 'mirror',
        'classifier_mode': 'learned',
    })

    # Save mirror results
    out_dir_m = PROJECT_ROOT / "experiments" / "results" / "glm_learned_mirror" / timestamp
    out_dir_m.mkdir(parents=True, exist_ok=True)
    out_path_m = out_dir_m / "glm_learned_mirror_results.json"

    with open(out_path_m, 'w') as f:
        json.dump({
            'config': {
                'embedding_dim': reference.shape[1],
                'n_reference': n_ref,
                'steps_per_task': STEPS_PER_TASK,
                'model': client.model,
                'backend': 'glm',
                'temperature': client.temperature,
                'max_tokens': client.max_tokens,
                'classifier_mode': 'learned',
                'intervention_style': 'mirror',
                'policy_table': engine_mirror.get_policy_table(),
            },
            'summary': summary_mirror,
            'results': results_mirror,
        }, f, indent=2)

    m_quality = summary_mirror['mean_quality']
    m_interventions = summary_mirror['total_interventions']
    print(f"\nMirror + Learned: quality={m_quality:.3f}, interventions={m_interventions}")
    print(f"  vs baseline:      {m_quality - glm_baseline_quality:+.3f}")
    print(f"  vs old mirror:    {m_quality - glm_mirror_old_quality:+.3f}")
    print(f"Saved to: {out_path_m}")

    # ========================================
    # COMPARISON TABLE
    # ========================================
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Experiment':<30s} {'Quality':>8s} {'vs Base':>8s} {'Intv':>6s}")
    print("-" * 60)
    print(f"{'GLM Baseline (no intv)':<30s} {glm_baseline_quality:>8.3f} {'---':>8s} {'0':>6s}")
    print(f"{'Directive + old classifier':<30s} {glm_directive_old_quality:>8.3f} {glm_directive_old_quality - glm_baseline_quality:>+8.3f} {'~8':>6s}")
    print(f"{'Mirror + old classifier':<30s} {glm_mirror_old_quality:>8.3f} {glm_mirror_old_quality - glm_baseline_quality:>+8.3f} {'~8':>6s}")
    print(f"{'Directive + LEARNED':<30s} {d_quality:>8.3f} {d_quality - glm_baseline_quality:>+8.3f} {d_interventions:>6d}")
    print(f"{'Mirror + LEARNED':<30s} {m_quality:>8.3f} {m_quality - glm_baseline_quality:>+8.3f} {m_interventions:>6d}")
    print("=" * 60)

    # Save comparison
    comparison_path = PROJECT_ROOT / "experiments" / "results" / "learned_classifier_comparison.json"
    comparison = {
        'timestamp': timestamp,
        'classifier_mode': 'learned',
        'baselines': {
            'glm_no_intervention': glm_baseline_quality,
            'glm_directive_rules': glm_directive_old_quality,
            'glm_mirror_rules': glm_mirror_old_quality,
        },
        'learned_results': {
            'directive': {
                'mean_quality': d_quality,
                'delta_vs_baseline': d_quality - glm_baseline_quality,
                'delta_vs_old_directive': d_quality - glm_directive_old_quality,
                'total_interventions': d_interventions,
                'per_domain': summary_directive['per_domain'],
            },
            'mirror': {
                'mean_quality': m_quality,
                'delta_vs_baseline': m_quality - glm_baseline_quality,
                'delta_vs_old_mirror': m_quality - glm_mirror_old_quality,
                'total_interventions': m_interventions,
                'per_domain': summary_mirror['per_domain'],
            },
        },
    }
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to: {comparison_path}")


if __name__ == "__main__":
    run_learned_classifier_experiments()
