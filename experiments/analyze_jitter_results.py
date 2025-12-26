import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze jitter experiment summaries.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=None,
        help="Path to the jitter results JSON file (deprecated, use --input).",
    )
    parser.add_argument(
        "--input",
        "-i",
        dest="input_path",
        default=None,
        help="Path to the jitter results JSON file.",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="output_dir",
        default=None,
        help="Directory to save analysis outputs (JSON + plots). If not specified, only prints to console.",
    )
    args = parser.parse_args()

    # Determine input path (--input takes precedence over positional arg)
    if args.input_path:
        args.final_input = args.input_path
    elif args.json_path:
        args.final_input = args.json_path
    else:
        # Default fallback
        args.final_input = "experiments/jitter_graph_results_run01.json"

    return args


def load_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        print(
            "No jitter results found yet – run python experiments/jitter_graph_playground.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def print_report(summary: Dict[str, Any]) -> List[float]:
    cfg = summary.get("config", {})
    global_stats = summary.get("global", {})
    per_node = summary.get("per_node", [])
    extremes = summary.get("extremes", {})

    variances = [node.get("variance", 0.0) for node in per_node]
    if not variances:
        print("No per-node variances found in summary.")
        return []

    median_var = statistics.median(variances)

    print("=== Jitter Summary ===")
    print("Config:")
    print(f"  noise_std: {cfg.get('noise_std')}")
    print(f"  num_samples: {cfg.get('num_samples')}")
    print(f"  k_extremes: {cfg.get('k_extremes')}")

    print("Global stats:")
    print(f"  mean_variance: {global_stats.get('mean_variance')}")
    print(f"  var_of_variances: {global_stats.get('var_of_variances')}")

    print("Per-node variance stats:")
    print(f"  min: {min(variances):.6f}")
    print(f"  max: {max(variances):.6f}")
    print(f"  median: {median_var:.6f}")

    print("Lowest jitter nodes:")
    for item in extremes.get("lowest", []):
        print(f"  node {item.get('node')}: {item.get('variance'):.6f}")

    print("Highest jitter nodes:")
    for item in extremes.get("highest", []):
        print(f"  node {item.get('node')}: {item.get('variance'):.6f}")

    return variances


def save_analysis(summary: Dict[str, Any], variances: List[float], output_dir: Path):
    """Save analysis results as JSON."""
    if not variances:
        return

    analysis = {
        "input_summary": summary,
        "computed_stats": {
            "min_variance": min(variances),
            "max_variance": max(variances),
            "median_variance": statistics.median(variances),
            "mean_variance": statistics.mean(variances),
            "stdev_variance": statistics.stdev(variances) if len(variances) > 1 else 0.0,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis.json"

    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)

    print(f"Analysis saved to {analysis_path.as_posix()}")


def maybe_plot_histogram(variances: List[float], output_dir: Path = None):
    if not variances:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        # Use non-interactive backend if saving to file
        if output_dir:
            matplotlib.use('Agg')
    except Exception:  # pragma: no cover
        print("Matplotlib not available; skipping histogram.")
        return

    bins = min(20, max(5, len(variances)))
    plt.figure(figsize=(6, 4))
    plt.hist(variances, bins=bins, color="steelblue", edgecolor="black")
    plt.title("Per-node Variance Histogram")
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "variance_histogram.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Histogram saved to {plot_path.as_posix()}")
        plt.close()
    else:
        plt.show()


def main():
    args = parse_args()
    summary_path = Path(args.final_input)
    summary = load_summary(summary_path)
    variances = print_report(summary)

    output_dir = Path(args.output_dir) if args.output_dir else None

    if output_dir:
        save_analysis(summary, variances, output_dir)

    maybe_plot_histogram(variances, output_dir)


if __name__ == "__main__":
    main()
