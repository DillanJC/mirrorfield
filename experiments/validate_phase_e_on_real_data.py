"""Phase E Validation — Real Data Integration Test

This script validates Phase E geometry bundle on real Phase D data.
It serves as both an integration test and a benchmark for the full pipeline.

Validation checklist:
1. Load Phase D artifacts (embeddings + boundary_distance)
2. Build geometry bundle on reference set
3. Transform query set
4. Validate outputs
5. Run falsifier to get verdict
6. Report performance
"""

import json
import time
import numpy as np
import argparse
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.bundle import GeometryBundle
from geometry.scoring import compute_geometry_score
from phase_e_falsifier import PhaseEFalsifier, Verdict


def find_latest_phase_d_run():
    """Find most recent Phase D run."""
    results_dir = Path("experiments/results/phase_d_integrated_eval")
    if not results_dir.exists():
        return None

    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()], reverse=True)
    for run_dir in run_dirs:
        if (run_dir / "summary.json").exists():
            return run_dir
    return None


def load_synthetic_data(n_ref: int = 1000, n_query: int = 500, D: int = 768, seed: int = 42):
    """Load synthetic data for validation.

    This is used when no Phase D artifacts are available.
    """
    print(f"Loading synthetic data: n_ref={n_ref}, n_query={n_query}, D={D}")
    rng = np.random.RandomState(seed)

    ref_embeddings = rng.randn(n_ref, D).astype(np.float32)
    query_embeddings = rng.randn(n_query, D).astype(np.float32)
    boundary_distance = rng.randn(n_query).astype(np.float32)

    # Synthetic target for falsifier (e.g., flip outcomes)
    # Create a target that has some correlation with boundary_distance
    # but also some independent signal
    noise = rng.randn(n_query) * 0.5
    target = (boundary_distance + noise > 0).astype(np.float32)

    return ref_embeddings, query_embeddings, boundary_distance, target


def validate_phase_e(
    ref_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    boundary_distance: np.ndarray,
    target: np.ndarray,
    config: dict = None,
):
    """Run full Phase E validation pipeline.

    Args:
        ref_embeddings: (N_ref, D) reference embeddings
        query_embeddings: (N_query, D) query embeddings
        boundary_distance: (N_query,) from Phase D
        target: (N_query,) target for falsifier R²
        config: geometry bundle config

    Returns:
        dict with validation results
    """
    print(f"\n{'='*60}")
    print(f"Phase E Validation Pipeline")
    print(f"{'='*60}")
    print(f"Reference set: {ref_embeddings.shape}")
    print(f"Query set: {query_embeddings.shape}")
    print(f"Boundary distance: {boundary_distance.shape}")
    print(f"Target: {target.shape}")
    print(f"{'='*60}\n")

    # Default config
    if config is None:
        config = {
            "random_seed": 42,
            "n_neighbors": 32,
            "k_curvature": 16,
            "r_components": 4,
            "batch_size": 256,
            "device": "cuda",
            "use_amp": False,
        }

    # Validate config
    print("Validating config...")
    GeometryBundle.validate_config(config)

    # Build kNN index
    print("Building kNN index on reference set...")
    t0 = time.time()
    nn = NearestNeighbors(
        n_neighbors=config["n_neighbors"],
        algorithm='auto',
        metric='euclidean',
        n_jobs=-1,
    )
    nn.fit(ref_embeddings)
    dt_index = time.time() - t0
    print(f"  Time: {dt_index:.3f}s")

    # Create geometry bundle
    print("Creating geometry bundle...")
    bundle = GeometryBundle(ref_points=ref_embeddings, nn_index=nn, config=config)

    # Transform queries
    print(f"Transforming {len(query_embeddings)} queries...")
    t0 = time.time()
    geometry_features = bundle.transform(query_embeddings, boundary_distance)
    dt_transform = time.time() - t0
    throughput = len(query_embeddings) / dt_transform
    print(f"  Time: {dt_transform:.3f}s")
    print(f"  Throughput: {throughput:.0f} queries/sec")

    # Extract features
    curvature = geometry_features["local_curvature"]
    ridge = geometry_features["ridge_proximity"]
    flags = geometry_features["geom_flags"]

    # Compute composite score
    geometry_score = compute_geometry_score(curvature, ridge)

    # Statistics
    print(f"\nGeometry Feature Statistics:")
    print(f"  Curvature: mean={np.mean(curvature):.4f}, std={np.std(curvature):.4f}")
    print(f"  Ridge: mean={np.mean(ridge):.4f}, std={np.std(ridge):.4f}")
    print(f"  Geometry score: mean={np.mean(geometry_score):.4f}, std={np.std(geometry_score):.4f}")

    # Flag analysis
    flag_counts = {}
    for sample_flags in flags:
        for flag in sample_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    print(f"\nGeometry Flags:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(flags)
        print(f"  {flag}: {count} ({pct:.1f}%)")

    # Run falsifier
    print(f"\nRunning Phase E falsifier...")
    falsifier_result = PhaseEFalsifier.evaluate(
        boundary_distance=boundary_distance,
        geometry_score=geometry_score,
        ridge_proximity=ridge,
        target=target,
    )

    print(f"\n{'='*60}")
    print(f"FALSIFIER VERDICT: {falsifier_result.verdict.value}")
    print(f"{'='*60}")
    print(f"  ΔR² = {falsifier_result.delta_r2:.4f}")
    print(f"  Info density = {falsifier_result.info_density:.3f}")
    print(f"  R²(dist only) = {falsifier_result.r2_dist_only:.4f}")
    print(f"  R²(dist+geom) = {falsifier_result.r2_dist_geom:.4f}")
    print(f"\nCorrelations:")
    print(f"  corr(bd, geom_score) = {falsifier_result.corr_bd_geom:.3f}")
    print(f"  corr(ridge, bd) = {falsifier_result.corr_ridge_bd:.3f}")
    print(f"\nReason: {falsifier_result.evidence.get('reason', 'N/A')}")
    print(f"{'='*60}\n")

    # Ridge independence sanity check
    ridge_independence_ok = abs(falsifier_result.corr_ridge_bd) < 0.9
    print(f"Ridge independence check: {'PASS' if ridge_independence_ok else 'FAIL'}")
    print(f"  |corr(ridge, bd)| = {abs(falsifier_result.corr_ridge_bd):.3f} {'<' if ridge_independence_ok else '>='} 0.9")

    return {
        "config": config,
        "timing": {
            "knn_index_build_sec": dt_index,
            "transform_sec": dt_transform,
            "throughput_per_sec": throughput,
        },
        "geometry_stats": {
            "curvature_mean": float(np.mean(curvature)),
            "curvature_std": float(np.std(curvature)),
            "ridge_mean": float(np.mean(ridge)),
            "ridge_std": float(np.std(ridge)),
        },
        "flags": flag_counts,
        "falsifier": falsifier_result.to_dict(),
        "ridge_independence_ok": ridge_independence_ok,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Phase E on real/synthetic data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n-ref", type=int, default=1000, help="Synthetic: ref set size")
    parser.add_argument("--n-query", type=int, default=500, help="Synthetic: query set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size")

    args = parser.parse_args()

    # Load data
    if args.synthetic:
        ref_emb, query_emb, bd, target = load_synthetic_data(
            n_ref=args.n_ref,
            n_query=args.n_query,
            seed=args.seed,
        )
    else:
        # Try to load real Phase D data
        phase_d_run = find_latest_phase_d_run()
        if phase_d_run is None:
            print("No Phase D runs found. Using synthetic data instead.")
            ref_emb, query_emb, bd, target = load_synthetic_data(seed=args.seed)
        else:
            print(f"Phase D artifacts not yet implemented for loading.")
            print(f"Using synthetic data for validation.")
            ref_emb, query_emb, bd, target = load_synthetic_data(seed=args.seed)

    # Run validation
    config = {
        "random_seed": args.seed,
        "n_neighbors": 32,
        "k_curvature": 16,
        "r_components": 4,
        "batch_size": args.batch_size,
        "device": args.device,
        "use_amp": False,
    }

    results = validate_phase_e(ref_emb, query_emb, bd, target, config)

    print("\n✓ Phase E validation complete")
