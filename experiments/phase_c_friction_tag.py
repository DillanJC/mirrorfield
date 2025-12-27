"""
Phase C Friction Tagging Harness

Tags reference set samples by expected "friction" (model uncertainty/difficulty).
Categorizes samples as low/medium/high friction based on boundary distance |d̃(x)|.

Follows Phase A/B artifact pattern: timestamped runs, environment snapshots, JSON summaries.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

from tier2.dataset import generate_synthetic_dataset, SyntheticSample
from tier2.model import BinarySentimentClassifier, load_classifier_checkpoint, get_embeddings
from tier2.reference import ReferenceSetStats
from tier2.metrics import compute_raw_boundary_distance, compute_standardized_distance
from tier2.friction import (
    FrictionConfig,
    compute_friction_tags,
    analyze_friction_distribution,
    save_friction_artifact
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
    parser = argparse.ArgumentParser(description="Phase C: Friction Tagging")
    parser.add_argument("--model-checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--reference-stats", type=str, required=True,
                        help="Path to reference set summary.json")
    parser.add_argument("--theta-borderline", type=float, default=0.5,
                        help="Borderline threshold (default: 0.5)")
    parser.add_argument("--theta-high-friction", type=float, default=0.25,
                        help="High friction threshold (default: 0.25)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")

    args = parser.parse_args()

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Phase C: Friction Tagging")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Model: {args.model_checkpoint}")
    print(f"Reference: {args.reference_stats}")
    print()

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/phase_c_friction_tag") / run_id

    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")
    print()

    # Capture environment
    env = get_environment_snapshot(device)

    # Configure friction tagging
    config = FrictionConfig(
        theta_borderline=args.theta_borderline,
        theta_high_friction=args.theta_high_friction
    )

    print("Configuration:")
    print(f"  theta_borderline: {config.theta_borderline}")
    print(f"  theta_high_friction: {config.theta_high_friction}")
    print()

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
    print()

    # Generate dataset (use full dataset as reference set)
    print("Step 3: Generating dataset...")
    samples, dataset_metadata = generate_synthetic_dataset(n_samples=2000, seed=seed)
    print(f"  Generated {len(samples)} samples")
    print()

    # Compute boundary distances
    print("Step 4: Computing boundary distances...")
    texts = [s.text for s in samples]
    embeddings = get_embeddings(texts, device, seed=seed)

    with torch.no_grad():
        logits = model(embeddings)
        d_raw = compute_raw_boundary_distance(logits)
        d_tilde = compute_standardized_distance(d_raw, ref_stats.mu_ref, ref_stats.sigma_ref)

    print(f"  Processed {len(samples)} samples")
    print()

    # Apply friction tagging
    print("Step 5: Applying friction tagging rules...")
    tags = compute_friction_tags(samples, d_tilde, config)

    # Analyze distribution
    distribution = analyze_friction_distribution(tags)

    print()
    print("Friction Distribution:")
    print(f"  Low friction:    {distribution['counts']['low']} samples ({distribution['fractions']['low']:.1%})")
    print(f"  Medium friction: {distribution['counts']['medium']} samples ({distribution['fractions']['medium']:.1%})")
    print(f"  High friction:   {distribution['counts']['high']} samples ({distribution['fractions']['high']:.1%})")
    print()

    # Save artifacts
    print("Step 6: Saving artifacts...")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "phase_c_friction_tag",
        "environment": env,
        "config": {
            "model_checkpoint": str(args.model_checkpoint),
            "reference_stats": str(args.reference_stats),
            "theta_borderline": config.theta_borderline,
            "theta_high_friction": config.theta_high_friction,
            "seed": seed
        },
        "reference": {
            "ref_name": ref_stats.ref_name,
            "ref_hash": ref_stats.ref_hash,
            "mu_ref": ref_stats.mu_ref,
            "sigma_ref": ref_stats.sigma_ref
        },
        "dataset": dataset_metadata,
        "friction_distribution": distribution
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    # Save canonical friction artifact to runs/
    runs_dir = Path("runs")
    artifact_path = save_friction_artifact(
        tags, ref_stats.ref_name, ref_stats.ref_hash, config, runs_dir
    )
    print(f"  Friction artifact: {artifact_path}")

    print()
    print("=" * 60)
    print("Friction Tagging Complete!")
    print("=" * 60)
    print(f"Total samples: {distribution['total_samples']}")
    print(f"High-friction samples: {distribution['counts']['high']} ({distribution['fractions']['high']:.1%})")
    print()
    print("Next steps:")
    print("  1. Use friction tags to stratify evaluation results")
    print("  2. Analyze model behavior on high-friction inputs")
    print("  3. Check for 'friction suppression' (flat outputs on high-friction samples)")
    print("=" * 60)


if __name__ == "__main__":
    main()
