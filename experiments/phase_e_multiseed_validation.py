"""Phase E Multi-Seed Validation — Robustness Check

This script runs Phase E validation across multiple seeds to verify:
1. Ridge independence is seed-invariant (corr(ridge, bd) < 0.9)
2. Falsifier verdicts are consistent
3. Geometry feature distributions are stable

Similar to Phase D 20-seed validation, this validates the robustness of
Phase E geometry features across different random initializations.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.validate_phase_e_on_real_data import (
    load_synthetic_data,
    validate_phase_e,
)


def run_multiseed_validation(
    seeds: list,
    n_ref: int = 1000,
    n_query: int = 500,
    D: int = 768,
    device: str = "cpu",
    output_dir: str = "runs",
):
    """Run Phase E validation across multiple seeds.

    Args:
        seeds: list of random seeds to test
        n_ref: reference set size
        n_query: query set size
        D: embedding dimension
        device: 'cuda' or 'cpu'
        output_dir: directory for output manifest

    Returns:
        dict with aggregated results
    """
    results = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed {i}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # Load data with this seed
        ref_emb, query_emb, bd, target = load_synthetic_data(
            n_ref=n_ref,
            n_query=n_query,
            D=D,
            seed=seed,
        )

        # Run validation
        config = {
            "random_seed": seed,
            "n_neighbors": 32,
            "k_curvature": 16,
            "r_components": 4,
            "batch_size": 256,
            "device": device,
            "use_amp": False,
        }

        result = validate_phase_e(ref_emb, query_emb, bd, target, config)

        # Extract key metrics
        results.append({
            "seed": int(seed),
            "verdict": result["falsifier"]["verdict"],
            "delta_r2": float(result["falsifier"]["delta_r2"]),
            "info_density": float(result["falsifier"]["info_density"]),
            "corr_bd_geom": float(result["falsifier"]["correlations"]["boundary_distance_vs_geometry_score"]),
            "corr_ridge_bd": float(result["falsifier"]["correlations"]["ridge_proximity_vs_boundary_distance"]),
            "ridge_independence_ok": bool(result["ridge_independence_ok"]),
            "curvature_mean": float(result["geometry_stats"]["curvature_mean"]),
            "curvature_std": float(result["geometry_stats"]["curvature_std"]),
            "ridge_mean": float(result["geometry_stats"]["ridge_mean"]),
            "ridge_std": float(result["geometry_stats"]["ridge_std"]),
            "timing": {k: float(v) for k, v in result["timing"].items()},
        })

    # Aggregate statistics
    print(f"\n\n{'='*60}")
    print(f"MULTI-SEED VALIDATION SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}\n")

    # Verdict distribution
    verdict_counts = {}
    for r in results:
        verdict = r["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    print("Verdict Distribution:")
    for verdict, count in sorted(verdict_counts.items()):
        pct = 100.0 * count / len(results)
        print(f"  {verdict}: {count}/{len(seeds)} ({pct:.1f}%)")

    # Ridge independence
    ridge_independence_pass = sum(r["ridge_independence_ok"] for r in results)
    ridge_independence_rate = 100.0 * ridge_independence_pass / len(results)
    print(f"\nRidge Independence:")
    print(f"  PASS: {ridge_independence_pass}/{len(seeds)} ({ridge_independence_rate:.1f}%)")

    # Correlation statistics
    corr_bd_geom = [r["corr_bd_geom"] for r in results]
    corr_ridge_bd = [r["corr_ridge_bd"] for r in results]
    print(f"\nCorrelation Statistics:")
    print(f"  corr(bd, geom_score): mean={np.mean(corr_bd_geom):.3f}, std={np.std(corr_bd_geom):.3f}")
    print(f"  corr(ridge, bd): mean={np.mean(corr_ridge_bd):.3f}, std={np.std(corr_ridge_bd):.3f}")

    # ΔR² statistics
    delta_r2 = [r["delta_r2"] for r in results]
    info_density = [r["info_density"] for r in results]
    print(f"\nExplanatory Power:")
    print(f"  ΔR²: mean={np.mean(delta_r2):.4f}, std={np.std(delta_r2):.4f}")
    print(f"  Info density: mean={np.mean(info_density):.3f}, std={np.std(info_density):.3f}")

    # Geometry feature statistics
    curv_mean = [r["curvature_mean"] for r in results]
    curv_std = [r["curvature_std"] for r in results]
    ridge_mean = [r["ridge_mean"] for r in results]
    ridge_std = [r["ridge_std"] for r in results]
    print(f"\nGeometry Features:")
    print(f"  Curvature mean: {np.mean(curv_mean):.4f} ± {np.std(curv_mean):.4f}")
    print(f"  Curvature std: {np.mean(curv_std):.4f} ± {np.std(curv_std):.4f}")
    print(f"  Ridge mean: {np.mean(ridge_mean):.4f} ± {np.std(ridge_mean):.4f}")
    print(f"  Ridge std: {np.mean(ridge_std):.4f} ± {np.std(ridge_std):.4f}")

    # Performance
    timing_transform = [r["timing"]["transform_sec"] for r in results]
    print(f"\nPerformance:")
    print(f"  Transform time: {np.mean(timing_transform):.3f} ± {np.std(timing_transform):.3f} seconds")

    print(f"\n{'='*60}\n")

    # Create manifest
    manifest = {
        "validation_type": "phase_e_multiseed",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "n_ref": n_ref,
            "n_query": n_query,
            "D": D,
            "device": device,
        },
        "summary": {
            "verdict_distribution": verdict_counts,
            "ridge_independence_pass_rate": ridge_independence_rate,
            "correlations": {
                "bd_geom_mean": float(np.mean(corr_bd_geom)),
                "bd_geom_std": float(np.std(corr_bd_geom)),
                "ridge_bd_mean": float(np.mean(corr_ridge_bd)),
                "ridge_bd_std": float(np.std(corr_ridge_bd)),
            },
            "explanatory_power": {
                "delta_r2_mean": float(np.mean(delta_r2)),
                "delta_r2_std": float(np.std(delta_r2)),
                "info_density_mean": float(np.mean(info_density)),
                "info_density_std": float(np.std(info_density)),
            },
            "geometry_features": {
                "curvature_mean": float(np.mean(curv_mean)),
                "curvature_mean_std": float(np.std(curv_mean)),
                "ridge_mean": float(np.mean(ridge_mean)),
                "ridge_mean_std": float(np.std(ridge_mean)),
            },
        },
        "per_seed_results": results,
    }

    # Save manifest
    output_path = Path(output_dir) / f"phase_e_multiseed_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to: {output_path}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase E multi-seed validation")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds to test")
    parser.add_argument("--n-ref", type=int, default=1000, help="Reference set size")
    parser.add_argument("--n-query", type=int, default=500, help="Query set size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds (default: 42,123,456,789,999,17,100,200,333,500)",
    )

    args = parser.parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        # Default: 10 seeds similar to Phase D initial batch
        seeds = [42, 123, 456, 789, 999, 17, 100, 200, 333, 500]
        seeds = seeds[:args.n_seeds]

    print(f"\nPhase E Multi-Seed Validation")
    print(f"Testing {len(seeds)} seeds: {seeds}")
    print(f"Config: n_ref={args.n_ref}, n_query={args.n_query}, device={args.device}\n")

    manifest = run_multiseed_validation(
        seeds=seeds,
        n_ref=args.n_ref,
        n_query=args.n_query,
        device=args.device,
        output_dir=args.output_dir,
    )

    print("\nValidation complete!")
