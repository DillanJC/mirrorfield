"""Phase E Falsifier Test 3: Targeted Construction — "Geometry Should Win" Cases

This test answers: Can geometry help even in IDEAL conditions?

Strategy:
Engineer adversarial pairs where boundary_distance FAILS but geometry SHOULD succeed:
1. Create pairs of points with SAME |d̃(x)| (same boundary distance)
2. But: Point A in flat region (low curvature), Point B near ridge (high curvature)
3. Give them DIFFERENT flip outcomes based on geometry
4. Test if falsifier detects this

Expected outcomes:
- If COSMETIC: Geometry doesn't help even when explicitly designed to matter
- If REAL_SIGNAL: Geometry captures information boundary_distance cannot

This is the STRONGEST test - if geometry fails here, it's unlikely to help anywhere.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors
from geometry.bundle import GeometryBundle
from phase_e_falsifier import PhaseEFalsifier
from geometry.scoring import compute_geometry_score


def create_adversarial_dataset(n_pairs=200, D=768, seed=42):
    """Create dataset where geometry should distinguish but boundary_distance cannot.

    For each pair:
    - Both have same |boundary_distance| ≈ 0.3 (medium friction)
    - Point A: embedded in flat region (low curvature, low ridge)
    - Point B: embedded near decision ridge (high curvature, high ridge)
    - Target: Point A stable (flip_rate=0.05), Point B unstable (flip_rate=0.25)

    Returns:
        reference_set, query_embeddings, query_bd, query_targets
    """
    rng = np.random.RandomState(seed)

    # Create reference set with explicit geometric structure
    n_ref = 1000

    # Reference set has two regions:
    # Region 1 (flat): Points clustered tightly
    # Region 2 (ridge): Points along a decision boundary

    ref_embeddings = []

    # Region 1: Flat cluster (600 points)
    flat_center = rng.randn(D)
    for _ in range(600):
        noise = rng.randn(D) * 0.1  # tight cluster = low curvature
        ref_embeddings.append(flat_center + noise)

    # Region 2: Ridge (400 points along a line)
    ridge_direction = rng.randn(D)
    ridge_direction = ridge_direction / (np.linalg.norm(ridge_direction) + 1e-8)
    ridge_perpendicular = rng.randn(D)
    ridge_perpendicular = ridge_perpendicular - np.dot(ridge_perpendicular, ridge_direction) * ridge_direction
    ridge_perpendicular = ridge_perpendicular / (np.linalg.norm(ridge_perpendicular) + 1e-8)

    for i in range(400):
        t = (i / 400.0) * 10.0  # spread along ridge
        # Points spread along line (high curvature, high ridge)
        point = flat_center + 5.0 * rng.randn(1) + ridge_direction * t
        # Add perpendicular noise (creates ridge structure)
        point = point + ridge_perpendicular * rng.randn(1) * 0.5
        ref_embeddings.append(point)

    ref_embeddings = np.array(ref_embeddings, dtype=np.float32)

    # Create adversarial query pairs
    query_embeddings_flat = []
    query_embeddings_ridge = []
    query_bd = []
    query_targets = []

    target_bd = 0.3  # Same boundary distance for all pairs

    for _ in range(n_pairs):
        # Point A: Near flat region (should be stable)
        offset_flat = rng.randn(D) * 0.15
        emb_flat = flat_center + offset_flat
        query_embeddings_flat.append(emb_flat)
        query_bd.append(target_bd)
        query_targets.append(0.05)  # Low flip rate

        # Point B: Near ridge (should be unstable)
        ridge_pos = rng.rand() * 10.0
        offset_ridge = rng.randn(D) * 0.2
        emb_ridge = flat_center + 5.0 * rng.randn(1) + ridge_direction * ridge_pos + offset_ridge
        query_embeddings_ridge.append(emb_ridge)
        query_bd.append(target_bd)  # SAME boundary distance!
        query_targets.append(0.25)  # High flip rate

    # Interleave flat and ridge points
    query_embeddings = []
    final_bd = []
    final_targets = []

    for i in range(n_pairs):
        query_embeddings.append(query_embeddings_flat[i])
        query_embeddings.append(query_embeddings_ridge[i])
        final_bd.extend([query_bd[i*2], query_bd[i*2+1]])
        final_targets.extend([query_targets[i*2], query_targets[i*2+1]])

    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    final_bd = np.array(final_bd, dtype=np.float32)
    final_targets = np.array(final_targets, dtype=np.float32)

    # Add small noise to boundary_distance to make it realistic
    final_bd = final_bd + rng.randn(len(final_bd)) * 0.05

    return ref_embeddings, query_embeddings, final_bd, final_targets


def run_test3(seed=42):
    """Run Test 3: Targeted adversarial construction."""
    print(f"\n{'='*70}")
    print(f"PHASE E FALSIFIER TEST 3: TARGETED CONSTRUCTION")
    print(f"{'='*70}\n")

    print("Creating adversarial dataset where geometry SHOULD matter...")
    print("Design:")
    print("  - All query points have SAME boundary_distance ≈ 0.3")
    print("  - Half are in flat regions (low curvature) → stable")
    print("  - Half are near ridges (high curvature) → unstable")
    print("  - Targets differ based on geometry, NOT boundary_distance\n")

    # Create dataset
    ref_embeddings, query_embeddings, query_bd, query_targets = create_adversarial_dataset(
        n_pairs=200,
        D=768,
        seed=seed,
    )

    n_ref = len(ref_embeddings)
    n_query = len(query_embeddings)

    print(f"Dataset created:")
    print(f"  Reference set: {n_ref} points")
    print(f"  Query set: {n_query} points ({n_query//2} pairs)")
    print(f"  Boundary distance: mean={np.mean(query_bd):.3f}, std={np.std(query_bd):.3f}")
    print(f"  Targets: mean={np.mean(query_targets):.3f}, std={np.std(query_targets):.3f}\n")

    # Build kNN index
    print("Building kNN index...")
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

    print("Creating geometry bundle...")
    bundle = GeometryBundle(ref_points=ref_embeddings, nn_index=nn, config=config)

    # Transform queries
    print(f"Transforming {n_query} queries...")
    geometry_features = bundle.transform(query_embeddings, query_bd)

    curvature = geometry_features["local_curvature"]
    ridge = geometry_features["ridge_proximity"]
    geometry_score = compute_geometry_score(curvature, ridge)

    # Analyze geometry features for flat vs ridge points
    print(f"\nGeometry Feature Analysis:")
    print(f"  All points:")
    print(f"    Curvature: mean={np.mean(curvature):.4f}, std={np.std(curvature):.4f}")
    print(f"    Ridge: mean={np.mean(ridge):.4f}, std={np.std(ridge):.4f}")

    # Flat points (even indices)
    flat_indices = np.arange(0, n_query, 2)
    ridge_indices = np.arange(1, n_query, 2)

    print(f"  Flat region points (should be low curvature/ridge):")
    print(f"    Curvature: mean={np.mean(curvature[flat_indices]):.4f}")
    print(f"    Ridge: mean={np.mean(ridge[flat_indices]):.4f}")
    print(f"  Ridge points (should be high curvature/ridge):")
    print(f"    Curvature: mean={np.mean(curvature[ridge_indices]):.4f}")
    print(f"    Ridge: mean={np.mean(ridge[ridge_indices]):.4f}")

    # Check if geometry features actually differ
    curv_diff = abs(np.mean(curvature[flat_indices]) - np.mean(curvature[ridge_indices]))
    ridge_diff = abs(np.mean(ridge[flat_indices]) - np.mean(ridge[ridge_indices]))

    print(f"\n  Geometry separation:")
    print(f"    Curvature difference: {curv_diff:.4f}")
    print(f"    Ridge difference: {ridge_diff:.4f}")

    if curv_diff < 0.01 and ridge_diff < 0.01:
        print(f"    ⚠️  WARNING: Geometry features did NOT separate as expected!")

    # Run falsifier
    print(f"\nRunning Phase E falsifier...")
    falsifier_result = PhaseEFalsifier.evaluate(
        boundary_distance=query_bd,
        geometry_score=geometry_score,
        ridge_proximity=ridge,
        target=query_targets,
    )

    print(f"\n{'='*70}")
    print(f"FALSIFIER VERDICT (ADVERSARIAL): {falsifier_result.verdict.value}")
    print(f"{'='*70}")
    print(f"  ΔR² = {falsifier_result.delta_r2:.4f}")
    print(f"  Info density = {falsifier_result.info_density:.3f}")
    print(f"  R²(dist only) = {falsifier_result.r2_dist_only:.4f}")
    print(f"  R²(dist+geom) = {falsifier_result.r2_dist_geom:.4f}")
    print(f"\nCorrelations:")
    print(f"  corr(bd, geom_score) = {falsifier_result.corr_bd_geom:.3f}")
    print(f"  corr(ridge, bd) = {falsifier_result.corr_ridge_bd:.3f}")
    print(f"\nReason: {falsifier_result.evidence.get('reason', 'N/A')}")
    print(f"{'='*70}\n")

    # Interpretation
    if falsifier_result.verdict.value == "COSMETIC":
        print("⚠️  GEOMETRY FAILED even in IDEAL CONDITIONS")
        print("    Conclusion: Geometry unlikely to help in practice")
    elif falsifier_result.verdict.value in ["REAL_SIGNAL", "WEAK_SIGNAL"]:
        print("✓ Geometry detected signal in adversarial case")
        print("    Conclusion: Geometry CAN help when structure exists")

    # Save results
    output_dir = Path("runs") / f"phase_e_test3_targeted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "test_id": "test3_targeted_construction_adversarial",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "n_ref": n_ref,
            "n_query": n_query,
            "n_pairs": n_query // 2,
            "design": "Same boundary_distance, different geometry, different targets",
        },
        "geometry_separation": {
            "curvature_difference": float(curv_diff),
            "ridge_difference": float(ridge_diff),
            "separation_achieved": bool(curv_diff > 0.01 or ridge_diff > 0.01),
        },
        "geometry_stats": {
            "flat_curvature": float(np.mean(curvature[flat_indices])),
            "ridge_curvature": float(np.mean(curvature[ridge_indices])),
            "flat_ridge": float(np.mean(ridge[flat_indices])),
            "ridge_ridge": float(np.mean(ridge[ridge_indices])),
        },
        "falsifier": falsifier_result.to_dict(),
    }

    output_path = output_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase E Test 3: Targeted construction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    result = run_test3(seed=args.seed)

    print("\nTest 3 (Targeted Construction) complete!")
    print(f"Verdict: {result['falsifier']['verdict']}")
    print(f"ΔR²: {result['falsifier']['delta_r2']:.4f}")
    print(f"Geometry separation achieved: {result['geometry_separation']['separation_achieved']}")
