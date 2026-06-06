"""
Track 5 Terrain Mapper Experiment — Continuous geometric interventions via GLM-4.7.

Replaces the discrete classifier pipeline with the continuous TerrainMapper.
Single-pass: 16 tasks x 4 steps = 64 GLM API calls.

Compares against:
- GLM baseline (no intervention): 0.812
- Mirror + learned classifier: 0.779
- Mirror + blind classifier: 0.783
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
from experiments.shared.terrain_mapper import TerrainMapper, extract_baseline_features
from experiments.track4_switches.run_live_baseline import build_reference_texts

# Configuration
STEPS_PER_TASK = 4
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "terrain_mapper"

# Baseline results path
GLM_BASELINE_PATH = (
    PROJECT_ROOT / "experiments" / "results" / "glm_baseline"
    / "20260215_083005" / "glm_baseline_results.json"
)

# Comparison baselines
BASELINES = {
    'glm_no_intervention': 0.812,
    'mirror_learned_classifier': 0.779,
    'mirror_blind_classifier': 0.783,
}


def run_terrain_experiment():
    """Run the continuous terrain mapper experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("MIRRORFIELD — CONTINUOUS TERRAIN MAPPER EXPERIMENT")
    print(f"Timestamp: {timestamp}")
    print("=" * 60)

    # Step 1: Load GLM baseline features
    print("\n[1/4] Loading GLM baseline features...")
    baseline_features = extract_baseline_features(str(GLM_BASELINE_PATH))
    print(f"  Baseline feature matrix: {baseline_features.shape}")
    assert baseline_features.shape == (64, 7), (
        f"Expected (64, 7) baseline features, got {baseline_features.shape}"
    )

    # Step 2: Build reference corpus and embedder
    print("\n[2/4] Initializing GLM client and embedder...")
    client = GLMReasoningClient(max_tokens=2048)
    embedder = ResponseEmbedder()

    print("  Building reference corpus from task bank...")
    ref_texts = build_reference_texts()
    reference = embedder.build_reference_corpus(ref_texts)
    n_ref = len(reference)
    k = min(50, n_ref - 1)
    print(f"  Reference corpus: {n_ref} texts, k={k}")

    # Step 3: Create TerrainMapper
    print("\n[3/4] Creating TerrainMapper...")
    mapper = TerrainMapper(
        baseline_features=baseline_features,
        reference_embeddings=reference,
        k=k,
        cooldown_steps=1,
        threshold_observation=0.45,
        threshold_detailed=0.65,
    )

    # Quick sanity check: run baseline features through percentile conversion
    print("  Sanity check: baseline feature percentile distribution...")
    test_percentiles = []
    for i in range(min(5, len(baseline_features))):
        pct = mapper.calibrator.to_percentiles(baseline_features[i])
        test_percentiles.append(pct)
    test_percentiles = np.array(test_percentiles)
    print(f"  First 5 baseline samples — mean percentile per feature:")
    from experiments.shared.terrain_mapper import FEATURE_NAMES
    for j, name in enumerate(FEATURE_NAMES):
        print(f"    {name}: {test_percentiles[:, j].mean():.1f}")

    # Step 4: Run experiment
    print(f"\n[4/4] Running experiment: {16} tasks x {STEPS_PER_TASK} steps = 64 API calls")
    print("=" * 60)

    tasks = get_task_bank()
    all_results = []
    all_anomaly_scores = []

    for task in tasks:
        print(f"\n  [{task.id}] {task.domain}: {task.prompt[:60]}...")

        mapper.reset()

        step_texts = []
        step_records = []
        pending_intervention = ""

        for step_idx in range(STEPS_PER_TASK):
            # Generate step
            step_text = client.generate_step(
                task_prompt=task.prompt,
                step_index=step_idx,
                total_steps=STEPS_PER_TASK,
                previous_steps=step_texts if step_texts else None,
                intervention_text=pending_intervention if pending_intervention else None,
            )
            step_texts.append(step_text)

            # Embed and normalize
            step_embedding = embedder.embed(step_text)
            norm = np.linalg.norm(step_embedding)
            step_embedding_norm = step_embedding / (norm + 1e-8)

            # Evaluate through terrain mapper
            profile = mapper.evaluate(step_embedding_norm, step_idx)
            all_anomaly_scores.append(profile.anomaly_score)

            # Record step
            step_record = {
                'step_index': step_idx,
                'text': step_text,
                'intervention_applied': pending_intervention if pending_intervention else None,
                'terrain_profile': profile.to_dict(),
            }
            step_records.append(step_record)

            # Set next intervention
            if profile.intervention_text:
                pending_intervention = profile.intervention_text
                print(f"    Step {step_idx}: anomaly={profile.anomaly_score:.3f} "
                      f"[{profile.intervention_intensity}]")
            else:
                pending_intervention = ""
                print(f"    Step {step_idx}: anomaly={profile.anomaly_score:.3f} [none]")

        # Score full response
        full_response = "\n\n".join(step_texts)
        quality = score_response(full_response)

        # Count interventions
        n_interventions = sum(
            1 for r in step_records
            if r['terrain_profile']['intervention_intensity'] != 'none'
        )
        intervention_types = [
            r['terrain_profile']['intervention_intensity']
            for r in step_records
            if r['terrain_profile']['intervention_intensity'] != 'none'
        ]

        result = {
            'task_id': task.id,
            'domain': task.domain,
            'prompt': task.prompt,
            'response': full_response,
            'steps': step_records,
            'quality_score': quality.to_dict(),
            'n_interventions': n_interventions,
            'intervention_types': intervention_types,
            'expected_signatures': task.expected_signatures,
        }
        all_results.append(result)

        print(f"    Quality: {quality.composite:.3f} | "
              f"Interventions: {n_interventions} ({intervention_types})")

    # Compute summary
    scores = [r['quality_score']['composite'] for r in all_results]
    mean_quality = float(np.mean(scores))
    std_quality = float(np.std(scores))
    total_interventions = sum(r['n_interventions'] for r in all_results)

    per_domain = {}
    for r in all_results:
        domain = r['domain']
        per_domain.setdefault(domain, []).append(r['quality_score']['composite'])
    per_domain_summary = {
        d: {'mean_quality': float(np.mean(v)), 'n_tasks': len(v)}
        for d, v in per_domain.items()
    }

    mapper_stats = mapper.get_stats()

    # Anomaly score distribution
    anomaly_arr = np.array(all_anomaly_scores)
    anomaly_dist = {
        'mean': float(np.mean(anomaly_arr)),
        'std': float(np.std(anomaly_arr)),
        'min': float(np.min(anomaly_arr)),
        'max': float(np.max(anomaly_arr)),
        'p25': float(np.percentile(anomaly_arr, 25)),
        'p50': float(np.percentile(anomaly_arr, 50)),
        'p75': float(np.percentile(anomaly_arr, 75)),
        'n_above_045': int(np.sum(anomaly_arr >= 0.45)),
        'n_above_065': int(np.sum(anomaly_arr >= 0.65)),
    }

    # Build output
    output = {
        'config': {
            'embedding_dim': reference.shape[1],
            'n_reference': n_ref,
            'steps_per_task': STEPS_PER_TASK,
            'model': client.model,
            'backend': 'glm',
            'temperature': client.temperature,
            'max_tokens': client.max_tokens,
            'cooldown_steps': 1,
            'threshold_observation': 0.45,
            'threshold_detailed': 0.65,
        },
        'summary': {
            'timestamp': timestamp,
            'n_tasks': len(all_results),
            'steps_per_task': STEPS_PER_TASK,
            'mean_quality': mean_quality,
            'std_quality': std_quality,
            'total_interventions': total_interventions,
            'intervention_rate': total_interventions / (len(all_results) * STEPS_PER_TASK),
            'per_domain': per_domain_summary,
            'mapper_stats': mapper_stats,
            'anomaly_distribution': anomaly_dist,
        },
        'results': all_results,
    }

    # Save
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "terrain_experiment_results.json"

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS — CONTINUOUS TERRAIN MAPPER")
    print("=" * 60)
    print(f"{'Experiment':<35s} {'Quality':>8s} {'vs Base':>8s} {'Intv':>6s}")
    print("-" * 60)
    for name, baseline_q in BASELINES.items():
        print(f"{name:<35s} {baseline_q:>8.3f} {'---':>8s} {'---':>6s}")
    print(f"{'Terrain Mapper':<35s} {mean_quality:>8.3f} "
          f"{mean_quality - BASELINES['glm_no_intervention']:>+8.3f} "
          f"{total_interventions:>6d}")
    print("=" * 60)

    # Per-domain breakdown
    print(f"\n{'Domain':<25s} {'Terrain':>8s} {'Baseline':>8s} {'Delta':>8s}")
    print("-" * 55)

    # Load baseline per-domain for comparison
    with open(GLM_BASELINE_PATH) as f:
        baseline_data = json.load(f)
    baseline_per_domain = baseline_data['summary']['per_domain']

    for domain in sorted(per_domain_summary.keys()):
        t_q = per_domain_summary[domain]['mean_quality']
        b_q = baseline_per_domain.get(domain, {}).get('mean_quality', 0)
        print(f"{domain:<25s} {t_q:>8.3f} {b_q:>8.3f} {t_q - b_q:>+8.3f}")
    print("-" * 55)

    # Anomaly score distribution
    print(f"\nAnomaly Score Distribution (64 steps):")
    print(f"  Mean: {anomaly_dist['mean']:.3f} | Std: {anomaly_dist['std']:.3f}")
    print(f"  Min: {anomaly_dist['min']:.3f} | Max: {anomaly_dist['max']:.3f}")
    print(f"  P25: {anomaly_dist['p25']:.3f} | P50: {anomaly_dist['p50']:.3f} | P75: {anomaly_dist['p75']:.3f}")
    print(f"  Steps >= 0.45 (observation): {anomaly_dist['n_above_045']}/64 "
          f"({anomaly_dist['n_above_045']/64*100:.0f}%)")
    print(f"  Steps >= 0.65 (detailed):    {anomaly_dist['n_above_065']}/64 "
          f"({anomaly_dist['n_above_065']/64*100:.0f}%)")

    # Sample intervention texts
    print(f"\nSample Intervention Texts:")
    sample_count = 0
    for r in all_results:
        for s in r['steps']:
            tp = s['terrain_profile']
            if tp['intervention_intensity'] != 'none' and sample_count < 3:
                print(f"\n  [{r['task_id']}] Step {s['step_index']} "
                      f"({tp['intervention_intensity']}, anomaly={tp['anomaly_score']:.3f}):")
                text = tp.get('intervention_text', '')
                # Indent each line
                for line in text.split('\n'):
                    print(f"    {line}")
                sample_count += 1

    print(f"\nResults saved to: {out_path}")
    return output


if __name__ == "__main__":
    run_terrain_experiment()
