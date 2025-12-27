"""
Phase D Integrated Evaluation Harness

Unified evaluation pipeline combining:
- Semantic transforms (Phase B)
- Perturbation testing with calibrated epsilon (Phase C)
- Friction stratification (Phase C)

Implements 4 evaluation modes:
1. Semantic-only: Baseline semantic transform evaluation
2. Perturbation-only: Noise injection stratified by friction
3. Combined: Semantic transforms THEN perturbation
4. Stratified analysis: Cross-mode breakdown by friction level

Follows Phase A/B/C artifact pattern: timestamped runs, environment snapshots, JSON summaries.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from tier2.dataset import generate_synthetic_dataset, SyntheticSample
from tier2.model import BinarySentimentClassifier, get_embeddings
from tier2.reference import ReferenceSetStats
from tier2.friction import FrictionTag
from tier2.integrated_eval import (
    run_semantic_mode,
    run_perturbation_mode,
    run_combined_mode,
    analyze_stratified_results,
    save_integrated_eval_results
)

DEFAULT_SEED = 42


def get_git_info():
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty_check = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL
        )
        dirty = "dirty" if dirty_check != 0 else "clean"

        return {"commit": commit[:12], "status": dirty}
    except Exception:
        return {"commit": "unknown", "status": "unknown"}


def get_environment_snapshot(device: torch.device):
    """Capture environment details for reproducibility."""
    env = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
    }

    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

    try:
        import sentence_transformers
        env["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        env["sentence_transformers_version"] = "not_installed"

    env["git"] = get_git_info()

    return env


def main():
    parser = argparse.ArgumentParser(description="Phase D: Integrated Evaluation Pipeline")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--reference-stats", type=str, required=True,
                        help="Path to reference set summary.json")
    parser.add_argument("--transform-suite", type=str, required=True,
                        help="Path to transform suite JSON (e.g., runs/tier2_transforms_v1.json)")
    parser.add_argument("--calibration-artifact", type=str, required=True,
                        help="Path to calibration artifact (e.g., runs/calibration_tau_*.json)")
    parser.add_argument("--friction-artifact", type=str, required=True,
                        help="Path to friction artifact (e.g., runs/friction_tags_*.json)")
    parser.add_argument("--n-perturbations", type=int, default=10,
                        help="Number of perturbations per sample (default: 10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")

    args = parser.parse_args()

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase D: Integrated Evaluation Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/phase_d_integrated_eval") / run_id

    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")
    print()

    # Capture environment
    env = get_environment_snapshot(device)

    # Step 1: Load all artifacts
    print("Step 1: Loading artifacts...")

    # Load reference set statistics
    with open(args.reference_stats, "r") as f:
        ref_data = json.load(f)

    ref_stats = ReferenceSetStats(
        mu_ref=ref_data["reference_set"]["mu_ref"],
        sigma_ref=ref_data["reference_set"]["sigma_ref"],
        n_samples=ref_data["reference_set"]["n_samples"],
        ref_name=ref_data["reference_set"]["ref_name"],
        ref_hash=ref_data["reference_set"]["ref_hash"],
        timestamp=ref_data["reference_set"]["timestamp"],
        min_distance=ref_data["reference_set"].get("min_distance", 0.0),
        max_distance=ref_data["reference_set"].get("max_distance", 0.0),
        median_distance=ref_data["reference_set"].get("median_distance", 0.0)
    )

    print(f"  Reference: {ref_stats.ref_name}")

    # Load model
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    embedding_dim = checkpoint["embedding_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = BinarySentimentClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"  Model: {args.model_checkpoint}")

    # Load transform suite
    with open(args.transform_suite, "r") as f:
        transform_data = json.load(f)

    # Extract transforms array (handle both direct list and nested structure)
    if isinstance(transform_data, dict) and "transforms" in transform_data:
        transform_suite = transform_data["transforms"]
    else:
        transform_suite = transform_data

    print(f"  Transforms: {args.transform_suite} ({len(transform_suite)} transforms)")

    # Load calibration artifact
    with open(args.calibration_artifact, "r") as f:
        calib_data = json.load(f)

    epsilon = calib_data["calibration_result"]["epsilon_calibrated"]
    print(f"  Calibration: {args.calibration_artifact} (epsilon={epsilon:.4f})")

    # Load friction artifact
    with open(args.friction_artifact, "r") as f:
        friction_data = json.load(f)

    friction_tags = [
        FrictionTag(
            sample_id=t["sample_id"],
            friction_level=t["friction_level"],
            d_tilde_standardized=t["d_tilde_standardized"],
            abs_d_tilde=t["abs_d_tilde"],
            borderline_distance=t["borderline_distance"],
            rationale=t["rationale"]
        )
        for t in friction_data["tags"]
    ]

    print(f"  Friction: {args.friction_artifact} ({len(friction_tags)} samples)")
    print()

    # Step 2: Generate evaluation dataset
    print("Step 2: Generating evaluation dataset...")
    samples, dataset_metadata = generate_synthetic_dataset(n_samples=2000, seed=seed)
    print(f"  Generated {len(samples)} samples")
    print()

    # Step 3: Run semantic-only evaluation
    print("Step 3: Running semantic-only evaluation...")
    print(f"  Processing {len(transform_suite)} transforms...")
    semantic_results = run_semantic_mode(
        model, transform_suite, ref_stats, device, seed
    )

    semantic_deltas = [r.delta_d_tilde for r in semantic_results]
    semantic_flips = [r.flip_occurred for r in semantic_results]
    mean_delta = sum(semantic_deltas) / len(semantic_deltas) if semantic_deltas else 0.0
    flip_rate_semantic = sum(semantic_flips) / len(semantic_flips) if semantic_flips else 0.0

    print(f"  Mean delta d-tilde: {mean_delta:.2f}")
    print(f"  Flip rate: {flip_rate_semantic:.1%}")
    print()

    # Step 4: Run perturbation-only evaluation
    print("Step 4: Running perturbation-only evaluation...")
    print(f"  Processing {len(samples)} samples ({args.n_perturbations} perturbations each)...")
    perturbation_results = run_perturbation_mode(
        model, samples, friction_tags, ref_stats, epsilon,
        args.n_perturbations, device, seed
    )

    overall_flip_rate = sum(r.flip_rate for r in perturbation_results) / len(perturbation_results) if perturbation_results else 0.0
    print(f"  Overall flip rate: {overall_flip_rate:.1%}")

    # Compute by-friction stats
    friction_stats = {}
    for friction_level in ["low", "medium", "high"]:
        friction_samples = [r for r in perturbation_results if r.friction_level == friction_level]
        if friction_samples:
            mean_flip = sum(r.flip_rate for r in friction_samples) / len(friction_samples)
            friction_stats[friction_level] = {
                "flip_rate": mean_flip,
                "n_samples": len(friction_samples)
            }

    print("  By friction:")
    for level in ["low", "medium", "high"]:
        if level in friction_stats:
            stats = friction_stats[level]
            print(f"    {level.capitalize()} ({stats['n_samples']}): {stats['flip_rate']:.1%}")
    print()

    # Step 5: Run combined semantic + perturbation
    print("Step 5: Running combined semantic + perturbation...")
    print(f"  Processing {len(transform_suite)} transforms with perturbation...")
    combined_results = run_combined_mode(
        model, transform_suite, samples, friction_tags, ref_stats,
        epsilon, args.n_perturbations, device, seed
    )

    compound_effects = [r.compound_effect for r in combined_results]
    mean_compound = sum(compound_effects) / len(compound_effects) if compound_effects else 0.0
    print(f"  Mean compound effect: {mean_compound:.3f}")
    print()

    # Step 6: Analyze stratified results
    print("Step 6: Analyzing stratified results...")
    stratified_analysis = analyze_stratified_results(
        semantic_results, perturbation_results, combined_results
    )

    print("  Cross-mode analysis complete")
    print()

    # Step 7: Save artifacts
    print("Step 7: Saving artifacts...")

    config = {
        "model_checkpoint": str(args.model_checkpoint),
        "reference_stats": str(args.reference_stats),
        "transform_suite": str(args.transform_suite),
        "calibration_artifact": str(args.calibration_artifact),
        "friction_artifact": str(args.friction_artifact),
        "epsilon_calibrated": epsilon,
        "n_perturbation_samples": args.n_perturbations,
        "seed": seed
    }

    save_integrated_eval_results(
        results_dir, run_id, semantic_results, perturbation_results,
        combined_results, stratified_analysis, config, env
    )

    print(f"  Summary: {results_dir / 'summary.json'}")
    print(f"  Semantic results: {results_dir / 'semantic_results.json'}")
    print(f"  Perturbation results: {results_dir / 'perturbation_results.json'}")
    print(f"  Combined results: {results_dir / 'combined_results.json'}")
    print(f"  Stratified analysis: {results_dir / 'stratified_analysis.json'}")
    print()

    print("=" * 60)
    print("Integrated Evaluation Complete!")
    print("=" * 60)
    print(f"Modes: Semantic | Perturbation | Combined")
    print(f"Flip rates: {flip_rate_semantic:.1%} | {overall_flip_rate:.1%} | (compound effect {mean_compound:.3f})")

    if "high" in friction_stats and "low" in friction_stats:
        high_flip = friction_stats["high"]["flip_rate"]
        low_flip = friction_stats["low"]["flip_rate"]
        print(f"High-friction perturbation flip rate: {high_flip:.1%} (vs {low_flip:.1%} low-friction)")

    print("=" * 60)


if __name__ == "__main__":
    main()
