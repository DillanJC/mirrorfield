"""Phase E Falsifier Test 2: Model Shift — Embedder Swap

This test answers: Is geometry signal embedder-specific or universal?

Strategy:
1. Use same Phase D friction structure (boundary_distance distribution)
2. Generate embeddings with TWO different strategies:
   - Strategy A: Current approach (friction-based clusters)
   - Strategy B: Alternative embedding (PCA-reduced, different seed, different structure)
3. Run Phase E on both
4. Compare verdicts

Expected outcomes:
- If both → REAL_SIGNAL: Geometry is robust to embedder choice
- If one → COSMETIC, other → REAL_SIGNAL: Geometry is embedder-specific artifact
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from geometry.bundle import GeometryBundle
from phase_e_falsifier import PhaseEFalsifier
from geometry.scoring import compute_geometry_score


def generate_embeddings_strategy_a(boundary_distance, friction_levels, D=768, seed=42):
    """Original strategy: Friction-based clusters."""
    rng = np.random.RandomState(seed)
    n_samples = len(boundary_distance)

    embeddings = []
    for i, (bd, friction) in enumerate(zip(boundary_distance, friction_levels)):
        base = rng.randn(D) * 0.1

        if friction == "low":
            offset = np.array([1.0, 0.0, 0.0] + [0.0]*(D-3))
        elif friction == "medium":
            offset = np.array([0.0, 1.0, 0.0] + [0.0]*(D-3))
        else:  # high
            offset = np.array([0.0, 0.0, 1.0] + [0.0]*(D-3))

        noise = rng.randn(D) * (0.5 + abs(bd))
        emb = base + offset * 2.0 + noise * 0.3
        embeddings.append(emb)

    return np.array(embeddings, dtype=np.float32)


def generate_embeddings_strategy_b(boundary_distance, friction_levels, D=768, seed=100):
    """Alternative strategy: Boundary-distance-driven with PCA reduction."""
    rng = np.random.RandomState(seed)
    n_samples = len(boundary_distance)

    # Generate high-dimensional random embeddings
    high_dim = 2048
    embeddings_high = rng.randn(n_samples, high_dim).astype(np.float32)

    # Add boundary-distance-correlated structure
    for i, bd in enumerate(boundary_distance):
        # Add structured component based on boundary distance
        direction = rng.randn(high_dim)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        embeddings_high[i] += direction * bd * 5.0

    # PCA reduce to target dimension
    pca = PCA(n_components=D, random_state=seed)
    embeddings = pca.fit_transform(embeddings_high).astype(np.float32)

    # Add per-sample noise
    embeddings += rng.randn(n_samples, D).astype(np.float32) * 0.1

    return embeddings


def run_test2(run_dir: str = None, seed: int = 42):
    """Run Test 2: Model shift (different embedding strategies)."""
    print(f"\n{'='*70}")
    print(f"PHASE E FALSIFIER TEST 2: MODEL SHIFT (Embedder Swap)")
    print(f"{'='*70}\n")

    # Load Phase D friction structure
    if run_dir is None:
        phase_d_dir = Path("experiments/results/phase_d_integrated_eval")
        runs = sorted([d for d in phase_d_dir.iterdir() if d.is_dir()], reverse=True)
        if not runs:
            raise FileNotFoundError("No Phase D runs found")
        run_dir = runs[0]
    else:
        run_dir = Path(run_dir)

    print(f"Using Phase D run: {run_dir.name}")

    # Load friction data
    summary_path = run_dir / "summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    friction_artifact = summary["config"]["friction_artifact"]
    friction_path = Path("runs") / Path(friction_artifact).name

    with open(friction_path, "r") as f:
        friction_data = json.load(f)

    boundary_distances = []
    friction_levels = []
    for tag in friction_data["tags"]:
        boundary_distances.append(tag["d_tilde_standardized"])
        friction_levels.append(tag["friction_level"])

    boundary_distances = np.array(boundary_distances, dtype=np.float32)
    n_samples = len(boundary_distances)

    print(f"Loaded {n_samples} samples from Phase D")
    print(f"Friction distribution: {friction_data['friction_distribution']}\n")

    # Generate target
    rng = np.random.RandomState(seed)
    target = np.array([
        0.019 if f == "low" else 0.118 if f == "medium" else 0.259
        for f in friction_levels
    ], dtype=np.float32)
    target = target + rng.randn(len(target)) * 0.05
    target = np.clip(target, 0, 1)

    # Test both embedding strategies
    results = {}

    for strategy_name, strategy_func, strategy_seed in [
        ("Strategy_A_Friction_Clusters", generate_embeddings_strategy_a, 42),
        ("Strategy_B_PCA_Reduced", generate_embeddings_strategy_b, 100),
    ]:
        print(f"\n{'-'*70}")
        print(f"Testing: {strategy_name}")
        print(f"{'-'*70}\n")

        # Generate embeddings
        embeddings = strategy_func(boundary_distances, friction_levels, D=768, seed=strategy_seed)

        print(f"Generated embeddings: {embeddings.shape}")
        print(f"  mean={np.mean(embeddings):.3f}, std={np.std(embeddings):.3f}")

        # Split into reference and query
        n_ref = int(0.8 * n_samples)
        n_query = n_samples - n_ref

        ref_embeddings = embeddings[:n_ref]
        query_embeddings = embeddings[n_ref:]
        query_bd = boundary_distances[n_ref:]
        query_target = target[n_ref:]

        # Build kNN index
        nn = NearestNeighbors(n_neighbors=32, algorithm='auto', metric='euclidean', n_jobs=-1)
        nn.fit(ref_embeddings)

        # Create geometry bundle
        config = {
            "random_seed": seed,
            "n_neighbors": 32,
            "k_curvature": 16,
            "r_components": 4,
            "batch_size": 256,
            "device": "cpu",
            "use_amp": False,
        }

        bundle = GeometryBundle(ref_points=ref_embeddings, nn_index=nn, config=config)

        # Transform queries
        geometry_features = bundle.transform(query_embeddings, query_bd)
        curvature = geometry_features["local_curvature"]
        ridge = geometry_features["ridge_proximity"]
        geometry_score = compute_geometry_score(curvature, ridge)

        print(f"Geometry features:")
        print(f"  Curvature: mean={np.mean(curvature):.4f}, std={np.std(curvature):.4f}")
        print(f"  Ridge: mean={np.mean(ridge):.4f}, std={np.std(ridge):.4f}")

        # Run falsifier
        falsifier_result = PhaseEFalsifier.evaluate(
            boundary_distance=query_bd,
            geometry_score=geometry_score,
            ridge_proximity=ridge,
            target=query_target,
        )

        print(f"\nVERDICT: {falsifier_result.verdict.value}")
        print(f"  ΔR² = {falsifier_result.delta_r2:.4f}")
        print(f"  Info density = {falsifier_result.info_density:.3f}")
        print(f"  corr(bd, geom) = {falsifier_result.corr_bd_geom:.3f}")
        print(f"  corr(ridge, bd) = {falsifier_result.corr_ridge_bd:.3f}")

        results[strategy_name] = {
            "verdict": falsifier_result.verdict.value,
            "delta_r2": float(falsifier_result.delta_r2),
            "info_density": float(falsifier_result.info_density),
            "correlations": {
                "bd_geom": float(falsifier_result.corr_bd_geom),
                "ridge_bd": float(falsifier_result.corr_ridge_bd),
            },
            "geometry_stats": {
                "curvature_mean": float(np.mean(curvature)),
                "ridge_mean": float(np.mean(ridge)),
            },
        }

    # Compare results
    print(f"\n{'='*70}")
    print(f"MODEL SHIFT COMPARISON")
    print(f"{'='*70}\n")

    for strategy, result in results.items():
        print(f"{strategy}:")
        print(f"  Verdict: {result['verdict']}")
        print(f"  ΔR²: {result['delta_r2']:.4f}")
        print(f"  Info density: {result['info_density']:.3f}\n")

    verdicts_match = (results["Strategy_A_Friction_Clusters"]["verdict"] ==
                     results["Strategy_B_PCA_Reduced"]["verdict"])

    print(f"Verdict consistency: {'SAME' if verdicts_match else 'DIFFERENT'}")

    if not verdicts_match:
        print("⚠️  WARNING: Geometry signal is EMBEDDER-SPECIFIC (potential artifact)")
    else:
        print("✓ Geometry signal is ROBUST across embedders")

    # Save results
    output_dir = Path("runs") / f"phase_e_test2_model_shift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "test_id": "test2_model_shift_embedder_swap",
        "timestamp": datetime.now().isoformat(),
        "phase_d_run": str(run_dir),
        "strategies": results,
        "comparison": {
            "verdicts_match": verdicts_match,
            "verdict_a": results["Strategy_A_Friction_Clusters"]["verdict"],
            "verdict_b": results["Strategy_B_PCA_Reduced"]["verdict"],
            "delta_r2_difference": abs(
                results["Strategy_A_Friction_Clusters"]["delta_r2"] -
                results["Strategy_B_PCA_Reduced"]["delta_r2"]
            ),
        },
    }

    output_path = output_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_path}\n")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase E Test 2: Model shift")
    parser.add_argument("--run-dir", type=str, default=None, help="Phase D run directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    result = run_test2(run_dir=args.run_dir, seed=args.seed)

    print("\nTest 2 (Model Shift) complete!")
    print(f"Verdicts: {result['comparison']['verdict_a']} vs {result['comparison']['verdict_b']}")
    print(f"Consistent: {result['comparison']['verdicts_match']}")
