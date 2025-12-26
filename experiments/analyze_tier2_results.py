"""
Tier-2 Semantic Evaluation Results Analyzer

Analyzes Phase B evaluation results and generates summary reports.

Usage:
    python analyze_tier2_results.py --input <summary.json> [--out <output_dir>]

Produces:
- Human-readable console summary
- Optional visualizations (histogram of Δd̃ by category)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional


def print_header(title: str, width: int = 60):
    """Print formatted section header."""
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def print_section(title: str, width: int = 60):
    """Print formatted subsection."""
    print()
    print(title)
    print("-" * width)


def analyze_summary(summary: Dict[str, Any]):
    """Print comprehensive analysis of evaluation summary."""

    print_header("Tier-2 Semantic Evaluation Results")

    # Run metadata
    print(f"Run ID: {summary['run_id']}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Phase: {summary['phase']}")

    # Environment
    print_section("Environment")
    env = summary["environment"]
    print(f"  PyTorch: {env['torch_version']}")
    print(f"  Device: {env['device_type']}")

    if env["cuda_available"]:
        print(f"  CUDA: {env['cuda_version']}")
        print(f"  GPU: {env['device_name']}")

    print(f"  sentence-transformers: {env.get('sentence_transformers_version', 'N/A')}")

    git_info = env.get("git", {})
    print(f"  Git: {git_info.get('commit', 'unknown')} ({git_info.get('status', 'unknown')})")

    # Configuration
    print_section("Configuration")
    config = summary["config"]
    print(f"  Seed: {config['seed']}")
    print(f"  Model checkpoint: {Path(config['model_checkpoint']).name}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")

    # Reference set
    print_section("Reference Set")
    ref = summary["reference"]
    print(f"  Name: {ref['ref_name']}")
    print(f"  Hash: {ref['ref_hash']}")
    print(f"  N: {ref['n_samples']}")
    print(f"  mu_ref: {ref['mu_ref']:.4f}")
    print(f"  sigma_ref: {ref['sigma_ref']:.4f}")

    # Transform suite
    print_section("Transform Suite")
    suite = summary["transform_suite"]
    print(f"  Suite ID: {suite['suite_id']}")
    print(f"  Suite hash: {suite['suite_hash']}")
    print(f"  LLM model: {suite['llm_model']}")
    print(f"  Total transforms: {suite['total_transforms']}")
    print(f"  Validated: {suite['validated_count']}")

    category_counts = suite.get("category_counts", {})
    for cat, count in category_counts.items():
        print(f"    {cat}: {count}")

    # Results
    print_section("Results by Category")
    results = summary["results"]
    category_stats = results.get("category_statistics", {})

    for category in ["preserving", "changing", "gotcha"]:
        stats = category_stats.get(category)

        if stats is None:
            print(f"\n[{category.upper()}] No transforms")
            continue

        print(f"\n[{category.upper()}]")
        print(f"  N: {stats['n_samples']}")
        print(f"  Δd̃ mean: {stats['mean']:+.4f}")
        print(f"  Δd̃ std: {stats['std']:.4f}")
        print(f"  Δd̃ median: {stats['median']:+.4f}")
        print(f"  |Δd̃| mean: {stats['abs_mean']:.4f}")
        print(f"  Range: [{stats['min']:+.4f}, {stats['max']:+.4f}]")

    # Expectations
    print_section("Expectation Evaluation")
    expectations = results.get("expectations")

    if expectations is None:
        print("  SKIP: Not all categories have transforms")
    else:
        interp = expectations["interpretation"]

        print(f"  Preserving: {interp['preserving']}")
        print(f"    |Δd̃| mean = {expectations['preserving_abs_mean']:.4f}")
        print(f"    Threshold: < 0.5 (small change)")

        print()
        print(f"  Changing: {interp['changing']}")
        print(f"    |Δd̃| mean = {expectations['changing_abs_mean']:.4f}")
        print(f"    Threshold: > 1.5 (large change)")

        print()
        print(f"  Gotcha: {interp['gotcha']}")
        print(f"    |Δd̃| mean = {expectations['gotcha_abs_mean']:.4f}")
        print(f"    Expected: intermediate (0.2-3.0)")

        print()
        print(f"  Separation ratio: {expectations['separation_ratio']:.2f}")
        print(f"    (changing / preserving, higher is better)")

        print()
        print(f"  Overall: {interp['overall']}")

    # Summary
    print_section("Summary")
    print(f"  Total transforms evaluated: {results['n_transforms_evaluated']}")

    if expectations:
        preserving_ok = expectations["preserving_meets_expectation"]
        changing_ok = expectations["changing_meets_expectation"]

        if preserving_ok and changing_ok:
            print("  Status: PASS - Metrics align with Phase B expectations")
        else:
            print("  Status: NEEDS REVIEW - Some metrics outside expected ranges")

            if not preserving_ok:
                print("    WARNING: Preserving transforms show larger boundary shifts than expected")
            if not changing_ok:
                print("    WARNING: Changing transforms show smaller boundary shifts than expected")

    print()


def create_visualization(summary: Dict[str, Any], output_dir: Path):
    """Create visualization of Δd̃ distribution by category."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not installed, skipping visualization")
        return

    results = summary["results"]
    category_stats = results.get("category_statistics", {})

    # Load per-transform results to get Δd̃ values
    summary_path = Path(summary.get("_source_path", "."))
    per_transform_path = summary_path.parent / "per_transform_results.json"

    if not per_transform_path.exists():
        print(f"WARNING: Could not find {per_transform_path}, skipping visualization")
        return

    with per_transform_path.open('r') as f:
        per_transform_results = json.load(f)

    # Collect Δd̃ values by category
    data_by_category = {
        "preserving": [],
        "changing": [],
        "gotcha": []
    }

    for result in per_transform_results:
        category = result["category"]
        delta_d_tilde = result["delta_d_tilde"]
        data_by_category[category].append(delta_d_tilde)

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "preserving": "#2ecc71",  # Green (should stay near 0)
        "changing": "#e74c3c",    # Red (should shift far from 0)
        "gotcha": "#f39c12"       # Orange (intermediate)
    }

    bins = np.linspace(-6, 6, 50)

    for category in ["preserving", "changing", "gotcha"]:
        values = data_by_category[category]

        if values:
            ax.hist(values, bins=bins, alpha=0.6, label=f"{category.capitalize()} (N={len(values)})",
                    color=colors[category], edgecolor='black')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Decision boundary')

    # Add expectation zones
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='green', label='Preserving zone (|Δd̃| < 0.5)')

    ax.set_xlabel('Δd̃ (Change in standardized boundary distance)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Δd̃ by Transform Category', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    output_path = output_dir / "delta_d_tilde_histogram.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Tier-2 semantic evaluation results")
    parser.add_argument("--input", "-i", dest="input_path", required=True,
                        help="Path to evaluation summary.json")
    parser.add_argument("--out", "-o", dest="output_dir", default=None,
                        help="Directory to save analysis outputs (default: same as input)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization generation")

    args = parser.parse_args()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return

    # Load summary
    with input_path.open('r') as f:
        summary = json.load(f)

    # Add source path for visualization loading
    summary["_source_path"] = str(input_path)

    # Print analysis
    analyze_summary(summary)

    # Create visualization if requested
    if not args.no_viz:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = input_path.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("Generating visualization...")
        create_visualization(summary, output_dir)

    print()


if __name__ == "__main__":
    main()
