"""
Tier-2 Semantic Evaluation Harness

Evaluates semantic transform suite using trained classifier and reference set statistics.

Flow:
1. Load model checkpoint, reference stats, transform suite
2. For each transform:
   - Compute embeddings for original & transformed text
   - Get logits → d(x) raw boundary distance
   - Standardize using μ_ref, σ_ref → d̃(x)
   - Compute Δd̃ = d̃(transformed) - d̃(original)
   - Detect boundary crossings (FlipRate)
3. Group by category (preserving/changing/gotcha)
4. Compute statistics per category (mean, std, median, abs_mean of Δd̃)
5. Evaluate expectation alignment (preserving small, changing large, gotcha intermediate)
6. Save artifacts to experiments/results/tier2_semantic_eval/<run_id>/

Expected results from DEFINITIONS_FREEZE Section D:
- Preserving: |Δd̃| < 0.5, FlipRate 0-10%
- Changing: |Δd̃| > 1.5, FlipRate >70%
- Gotcha: Intermediate (model struggles)
"""

import argparse
import json
import subprocess
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from tier2.model import load_classifier_checkpoint, get_embeddings
from tier2.metrics import (
    compute_raw_boundary_distance,
    compute_standardized_distance,
    compute_delta_distance,
    compute_flip_rate,
    compute_category_statistics,
    evaluate_transform_expectations
)
from tier2.transforms import transform_suite_from_dict

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
        env["tf32_enabled"] = torch.backends.cuda.matmul.allow_tf32

    # Add sentence-transformers version
    try:
        import sentence_transformers
        env["sentence_transformers_version"] = sentence_transformers.__version__
        env["embedding_model"] = "all-MiniLM-L6-v2"
    except ImportError:
        env["sentence_transformers_version"] = "not_installed"
        env["embedding_model"] = "N/A"

    env["git"] = get_git_info()

    return env


def load_reference_stats(reference_summary_path: str) -> Dict[str, Any]:
    """Load reference set statistics from summary JSON."""
    with open(reference_summary_path, 'r') as f:
        data = json.load(f)
    return data["reference_set"]


def main():
    parser = argparse.ArgumentParser(description="Tier-2 Semantic Evaluation")
    parser.add_argument("--model-checkpoint", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--reference-stats", required=True,
                        help="Path to reference set summary.json")
    parser.add_argument("--transform-suite", required=True,
                        help="Path to transform suite JSON")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--embedding-dim", type=int, default=384,
                        help="Embedding dimension (default: 384)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden layer dimension (default: 128)")

    args = parser.parse_args()

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Tier-2 Semantic Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Model checkpoint: {args.model_checkpoint}")
    print(f"Reference stats: {args.reference_stats}")
    print(f"Transform suite: {args.transform_suite}")
    print()

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/tier2_semantic_eval") / run_id

    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")
    print()

    # Capture environment
    env = get_environment_snapshot(device)

    # Load reference statistics
    print("Step 1: Loading reference set statistics...")
    ref_stats = load_reference_stats(args.reference_stats)
    mu_ref = ref_stats["mu_ref"]
    sigma_ref = ref_stats["sigma_ref"]
    ref_name = ref_stats["ref_name"]
    ref_hash = ref_stats["ref_hash"]

    print(f"  Reference: {ref_name}")
    print(f"  mu_ref: {mu_ref:.4f}")
    print(f"  sigma_ref: {sigma_ref:.4f}")
    print(f"  N: {ref_stats['n_samples']}")
    print(f"  Hash: {ref_hash}")
    print()

    # Load model
    print("Step 2: Loading trained classifier...")
    torch.manual_seed(seed)
    model = load_classifier_checkpoint(
        args.model_checkpoint,
        device,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    )
    print(f"  Model loaded from: {args.model_checkpoint}")
    print(f"  Model in eval mode: {not model.training}")
    print()

    # Load transform suite
    print("Step 3: Loading transform suite...")
    with open(args.transform_suite, 'r') as f:
        suite_data = json.load(f)

    suite = transform_suite_from_dict(suite_data)

    print(f"  Suite ID: {suite.suite_id}")
    print(f"  Total transforms: {len(suite.transforms)}")
    print(f"  LLM model: {suite.llm_model}")
    print(f"  Suite hash: {suite.suite_hash}")

    category_counts = suite.get_category_counts()
    for cat, count in category_counts.items():
        print(f"    {cat}: {count}")

    validation_report = suite.validate_suite_requirements()
    print(f"  Validated: {validation_report['total_validated']}")
    print(f"  Meets requirements: {'PASS' if validation_report['meets_requirement'] else 'FAIL'}")
    print()

    # Evaluate transforms
    print("Step 4: Evaluating semantic transforms...")
    print()

    results_by_category = {
        "preserving": [],
        "changing": [],
        "gotcha": []
    }

    per_transform_results = []

    for i, transform in enumerate(suite.transforms, 1):
        # Get embeddings for original and transformed text
        texts_original = [transform.original_text]
        texts_transformed = [transform.transformed_text]

        with torch.no_grad():
            emb_original = get_embeddings(texts_original, device, seed=seed)
            emb_transformed = get_embeddings(texts_transformed, device, seed=seed)

            # Compute logits
            logits_original = model(emb_original)
            logits_transformed = model(emb_transformed)

            # Compute d(x)
            d_original = compute_raw_boundary_distance(logits_original)
            d_transformed = compute_raw_boundary_distance(logits_transformed)

            # Standardize to d̃(x)
            d_tilde_original = compute_standardized_distance(d_original, mu_ref, sigma_ref)
            d_tilde_transformed = compute_standardized_distance(d_transformed, mu_ref, sigma_ref)

            # Compute Δd̃
            delta_d_tilde = compute_delta_distance(d_tilde_original, d_tilde_transformed)

            # Check flip
            flip_info = compute_flip_rate(d_tilde_original, d_tilde_transformed, theta=0.0)

        # Store result
        result = {
            "transform_id": transform.transform_id,
            "category": transform.category,
            "original_text": transform.original_text,
            "transformed_text": transform.transformed_text,
            "d_original": d_original.item(),
            "d_transformed": d_transformed.item(),
            "d_tilde_original": d_tilde_original.item(),
            "d_tilde_transformed": d_tilde_transformed.item(),
            "delta_d_tilde": delta_d_tilde.item(),
            "flipped": flip_info["n_flipped"] > 0,
            "validated": transform.validated
        }

        per_transform_results.append(result)
        results_by_category[transform.category].append(delta_d_tilde.item())

        # Print progress
        if i % 10 == 0 or i == len(suite.transforms):
            print(f"  Processed {i}/{len(suite.transforms)} transforms...")

    print()

    # Compute category statistics
    print("Step 5: Computing category statistics...")
    print()

    category_stats = {}

    for category in ["preserving", "changing", "gotcha"]:
        if not results_by_category[category]:
            print(f"  [{category.upper()}] No transforms")
            category_stats[category] = None
            continue

        delta_values = torch.tensor(results_by_category[category])
        stats = compute_category_statistics(delta_values, category)

        category_stats[category] = stats

        print(f"  [{category.upper()}]")
        print(f"    N: {stats['n_samples']}")
        print(f"    Δd̃ mean: {stats['mean']:.4f}")
        print(f"    Δd̃ std: {stats['std']:.4f}")
        print(f"    Δd̃ median: {stats['median']:.4f}")
        print(f"    |Δd̃| mean: {stats['abs_mean']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        # Compute FlipRate for category
        cat_results = [r for r in per_transform_results if r["category"] == category]
        n_flipped = sum(1 for r in cat_results if r["flipped"])
        flip_rate = n_flipped / len(cat_results) if cat_results else 0.0

        print(f"    FlipRate: {flip_rate:.2%} ({n_flipped}/{len(cat_results)})")
        print()

    # Evaluate expectations
    print("Step 6: Evaluating transform expectations...")
    print()

    if all(category_stats[c] is not None for c in ["preserving", "changing", "gotcha"]):
        expectations = evaluate_transform_expectations(
            category_stats["preserving"],
            category_stats["changing"],
            category_stats["gotcha"]
        )

        print(f"  Preserving: {expectations['interpretation']['preserving']}")
        print(f"    |Δd̃| mean = {expectations['preserving_abs_mean']:.4f} (expect < 0.5)")

        print(f"  Changing: {expectations['interpretation']['changing']}")
        print(f"    |Δd̃| mean = {expectations['changing_abs_mean']:.4f} (expect > 1.5)")

        print(f"  Gotcha: {expectations['interpretation']['gotcha']}")
        print(f"    |Δd̃| mean = {expectations['gotcha_abs_mean']:.4f} (expect intermediate)")

        print()
        print(f"  Separation ratio: {expectations['separation_ratio']:.2f}")
        print(f"  Overall: {expectations['interpretation']['overall']}")
    else:
        print("  SKIP: Not all categories have transforms")
        expectations = None

    print()

    # Save artifacts
    print("Step 7: Saving artifacts...")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "tier2_semantic_eval",
        "environment": env,
        "config": {
            "seed": seed,
            "model_checkpoint": args.model_checkpoint,
            "reference_stats_path": args.reference_stats,
            "transform_suite_path": args.transform_suite,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim
        },
        "reference": {
            "ref_name": ref_name,
            "ref_hash": ref_hash,
            "mu_ref": mu_ref,
            "sigma_ref": sigma_ref,
            "n_samples": ref_stats["n_samples"]
        },
        "transform_suite": {
            "suite_id": suite.suite_id,
            "suite_hash": suite.suite_hash,
            "llm_model": suite.llm_model,
            "total_transforms": len(suite.transforms),
            "validated_count": validation_report["total_validated"],
            "category_counts": category_counts
        },
        "results": {
            "category_statistics": category_stats,
            "expectations": expectations,
            "n_transforms_evaluated": len(per_transform_results)
        }
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")

    # Per-transform results
    per_transform_path = results_dir / "per_transform_results.json"
    with per_transform_path.open("w") as f:
        json.dump(per_transform_results, f, indent=2)
    print(f"  Per-transform results: {per_transform_path}")

    # Category metrics
    category_metrics = {}
    for category in ["preserving", "changing", "gotcha"]:
        cat_results = [r for r in per_transform_results if r["category"] == category]

        if cat_results:
            delta_values = [r["delta_d_tilde"] for r in cat_results]
            n_flipped = sum(1 for r in cat_results if r["flipped"])

            category_metrics[category] = {
                "n_transforms": len(cat_results),
                "delta_d_tilde_values": delta_values,
                "flip_rate": n_flipped / len(cat_results),
                "n_flipped": n_flipped,
                "statistics": category_stats[category]
            }

    category_metrics_path = results_dir / "category_metrics.json"
    with category_metrics_path.open("w") as f:
        json.dump(category_metrics, f, indent=2)
    print(f"  Category metrics: {category_metrics_path}")

    print()

    print("=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Transforms evaluated: {len(per_transform_results)}")

    if expectations:
        print(f"Overall: {expectations['interpretation']['overall']}")

    print()
    print("Next steps:")
    print(f"  1. Analyze results: python experiments\\analyze_tier2_results.py \\")
    print(f"       --input {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
