"""
Phase C Perturbation Calibration Harness

Calibrates noise scale ε to achieve target flip rate (5-10% boundary crossings)
on a held-out calibration split. Uses binary search for efficient convergence.

Follows Phase A/B artifact pattern: timestamped runs, environment snapshots, JSON summaries.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from tier2.dataset import generate_synthetic_dataset, split_dataset_3way, SyntheticSample
from tier2.model import BinarySentimentClassifier, load_classifier_checkpoint
from tier2.reference import ReferenceSetStats
from tier2.calibration import (
    CalibrationConfig,
    calibrate_epsilon,
    save_calibration_artifact,
    compute_flip_rate_under_perturbation
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
    parser = argparse.ArgumentParser(description="Phase C: Perturbation Calibration")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--reference-stats", type=str, required=True,
                        help="Path to reference set summary.json")
    parser.add_argument("--target-flip-min", type=float, default=0.05,
                        help="Minimum target flip rate (default: 0.05)")
    parser.add_argument("--target-flip-max", type=float, default=0.10,
                        help="Maximum target flip rate (default: 0.10)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")

    args = parser.parse_args()

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase C: Perturbation Calibration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Model: {args.model_checkpoint}")
    print(f"Reference: {args.reference_stats}")
    print()

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/phase_c_calibrate_tau") / run_id

    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")
    print()

    # Capture environment
    env = get_environment_snapshot(device)

    # Load reference set statistics
    print("Step 1: Loading reference set statistics...")
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
    print(f"  mu_ref: {ref_stats.mu_ref:.4f}")
    print(f"  sigma_ref: {ref_stats.sigma_ref:.4f}")
    print(f"  N: {ref_stats.n_samples} samples")
    print()

    # Load model
    print("Step 2: Loading trained classifier...")
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

    print(f"  Model loaded from: {args.model_checkpoint}")
    print(f"  Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")
    print()

    # Generate dataset and 3-way split
    print("Step 3: Generating dataset and 3-way split...")
    samples, dataset_metadata = generate_synthetic_dataset(n_samples=2000, seed=seed)
    train_samples, calib_samples, eval_samples, split_metadata = split_dataset_3way(
        samples, train_fraction=0.6, calib_fraction=0.2, seed=seed
    )

    print(f"  Train: {split_metadata['n_train']} samples")
    print(f"  Calibration: {split_metadata['n_calib']} samples")
    print(f"  Evaluation: {split_metadata['n_eval']} samples")
    print()

    # Configure calibration
    config = CalibrationConfig(
        target_flip_rate_min=args.target_flip_min,
        target_flip_rate_max=args.target_flip_max,
        seed=seed
    )

    print("Step 4: Calibrating epsilon on calibration set...")
    print(f"  Target flip rate: [{config.target_flip_rate_min:.1%}, {config.target_flip_rate_max:.1%}]")
    print(f"  Max iterations: {config.max_iterations}")
    print()

    start_time = time.perf_counter()
    result = calibrate_epsilon(model, calib_samples, ref_stats, device, config)
    calibration_time = time.perf_counter() - start_time

    # Print search history
    for step in result.search_history:
        status = "✓" if config.target_flip_rate_min <= step["flip_rate"] <= config.target_flip_rate_max else ""
        print(f"  Iteration {step['iteration']}: ε={step['epsilon']:.4f} → flip_rate={step['flip_rate']:.1%} {status}")

    print()
    if result.converged:
        print(f"  CONVERGED after {result.iterations_used} iterations")
    else:
        print(f"  Max iterations reached ({result.iterations_used})")

    print(f"  Calibrated ε: {result.epsilon_calibrated:.4f}")
    print(f"  Achieved flip rate: {result.flip_rate_achieved:.1%}")
    print(f"  Calibration time: {calibration_time:.2f}s")
    print()

    # Validate on evaluation set
    print("Step 5: Validating on evaluation set...")
    val_flip_rate, val_n_flips = compute_flip_rate_under_perturbation(
        model, eval_samples, ref_stats, result.epsilon_calibrated,
        config.n_perturbation_samples, device, seed
    )
    print(f"  Validation flip rate: {val_flip_rate:.1%}")
    print(f"  Validation within tolerance: {'YES' if abs(val_flip_rate - result.flip_rate_achieved) < 0.02 else 'NO'}")
    print()

    # Save artifacts
    print("Step 6: Saving artifacts...")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "phase_c_calibrate_tau",
        "environment": env,
        "config": {
            "model_checkpoint": str(args.model_checkpoint),
            "reference_stats": str(args.reference_stats),
            "target_flip_rate_min": config.target_flip_rate_min,
            "target_flip_rate_max": config.target_flip_rate_max,
            "seed": seed
        },
        "reference": {
            "ref_name": ref_stats.ref_name,
            "ref_hash": ref_stats.ref_hash,
            "mu_ref": ref_stats.mu_ref,
            "sigma_ref": ref_stats.sigma_ref
        },
        "dataset_split": split_metadata,
        "calibration_result": {
            "epsilon_calibrated": result.epsilon_calibrated,
            "flip_rate_achieved": result.flip_rate_achieved,
            "n_samples": result.n_samples,
            "converged": result.converged,
            "iterations_used": result.iterations_used,
            "calibration_hash": result.calibration_hash,
            "calibration_time_seconds": calibration_time
        },
        "validation": {
            "flip_rate": val_flip_rate,
            "n_flips": val_n_flips,
            "n_samples": len(eval_samples)
        }
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    # Save canonical calibration artifact to runs/
    runs_dir = Path("runs")
    artifact_path = save_calibration_artifact(result, ref_stats, config, runs_dir)
    print(f"  Calibration artifact: {artifact_path}")

    print()
    print("=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"Calibrated ε: {result.epsilon_calibrated:.4f}")
    print(f"Flip rate: {result.flip_rate_achieved:.1%} (target: {config.target_flip_rate_min:.1%}-{config.target_flip_rate_max:.1%})")
    print(f"Calibration hash: {result.calibration_hash}")
    print()
    print("Next steps:")
    print("  1. Run friction tagging: python experiments\\phase_c_friction_tag.py \\")
    print(f"       --model-checkpoint {args.model_checkpoint} \\")
    print(f"       --reference-stats {args.reference_stats}")
    print("  2. Use calibrated ε in future evaluation runs")
    print("=" * 60)


if __name__ == "__main__":
    main()
