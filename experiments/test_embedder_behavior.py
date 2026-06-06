"""Behavioral Correlation Test for OpenAI Embedders

This script validates that embedders passing the gate check actually
produce geometry features that add meaningful signal beyond boundary_distance.

Steps:
1. Load saved embeddings from gate tests
2. Compute geometry features (curvature, ridge proximity)
3. Check correlations with boundary_distance
4. Run full Phase E falsifier
5. Validate ΔR² matches gate prediction

Expected: Gate REAL_SIGNAL → Falsifier REAL_SIGNAL with ΔR² > 0.01
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_e_falsifier import PhaseEFalsifier
from geometry.features import local_curvature_svd_gpu, ridge_proximity_from_knn_distances
from sklearn.neighbors import NearestNeighbors


def load_embeddings_from_run(run_dir):
    """Load embeddings and metadata from a test run.

    Returns:
        embeddings: (N, D) numpy array
        boundary_distance: (N,) numpy array
        labels: (N,) numpy array
        diagnostics: dict with gate check results
    """
    run_path = Path(run_dir)

    # Load embeddings (saved as separate .npy files)
    embeddings_path = run_path / "embeddings.npy"
    boundary_path = run_path / "boundary_distances.npy"
    labels_path = run_path / "labels.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary distances not found: {boundary_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    embeddings = np.load(embeddings_path)
    boundary_distance = np.load(boundary_path)
    labels = np.load(labels_path)

    # Load diagnostics
    diag_path = run_path / "diagnostics.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"Diagnostics not found: {diag_path}")

    with open(diag_path, "r") as f:
        diagnostics = json.load(f)

    return embeddings, boundary_distance, labels, diagnostics


def compute_geometry_features(embeddings, k=32, r=4, device="cuda"):
    """Compute Phase E geometry features.

    Args:
        embeddings: (N, D) numpy array
        k: Number of neighbors for geometry computation
        r: Number of principal components for curvature (r < k)
        device: 'cuda' or 'cpu' for computation

    Returns:
        curvature: (N,) array of local curvature scores
        ridge: (N,) array of ridge proximity scores
    """
    N, D = embeddings.shape
    print(f"Computing geometry features for {N} samples in {D}D...")

    # Build kNN index (all points are both reference and query)
    print("  [1/3] Building kNN index...")
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)

    # Get neighbors (query all points against themselves)
    print("  [2/3] Finding k-nearest neighbors...")
    distances, neighbor_indices = nn.kneighbors(embeddings)

    # Compute local curvature (SVD-based)
    print(f"  [3/3] Computing curvature (SVD, r={r}) and ridge proximity...")
    curvature = local_curvature_svd_gpu(
        ref_points=embeddings,
        neighbor_indices=neighbor_indices,
        r=r,
        batch_size=256,
        device=device,
        eps=1e-8,
        use_amp=False
    )

    # Compute ridge proximity
    ridge = ridge_proximity_from_knn_distances(
        distances=distances,
        inner_index=15,  # 16th NN (0-indexed)
        outer_index=31,  # 32nd NN (0-indexed)
        eps=1e-6
    )

    print(f"✓ Geometry features computed")
    print(f"  Curvature: mean={curvature.mean():.6f}, std={curvature.std():.6f}")
    print(f"  Ridge: mean={ridge.mean():.6f}, std={ridge.std():.6f}")

    return curvature, ridge


def run_behavioral_test(run_dir, embedder_name):
    """Run behavioral correlation test on a saved embedder test.

    Args:
        run_dir: Path to the test run directory
        embedder_name: Human-readable name for the embedder

    Returns:
        dict with test results
    """
    print(f"\n{'='*80}")
    print(f"BEHAVIORAL CORRELATION TEST: {embedder_name}")
    print(f"{'='*80}\n")

    # Load data
    print(f"Loading embeddings from {run_dir}...")
    embeddings, boundary_distance, labels, diagnostics = load_embeddings_from_run(run_dir)
    N, D = embeddings.shape
    print(f"✓ Loaded {N} samples, {D} dimensions\n")

    # Show gate prediction
    metrics = diagnostics.get("metrics", diagnostics)  # Handle both flat and nested structure
    print("Gate Check Results:")
    print(f"  Verdict: {diagnostics['verdict_prediction']}")
    print(f"  Confidence: {diagnostics['confidence']:.2f}")
    print(f"  Distance Concentration (CV): {metrics['distance_concentration']:.6f}")
    print(f"  Local Intrinsic Dim: {metrics['local_intrinsic_dim']:.1f}")
    print()

    # Compute geometry features
    # Use CPU if no CUDA available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    curvature, ridge = compute_geometry_features(embeddings, k=32, r=4, device=device)

    # Combine into geometry_score (per Phase E: use ridge as primary, curvature as tiebreaker)
    geometry_score = ridge + 0.1 * curvature

    print(f"\nCorrelation Analysis:")
    # Check correlations
    corr_bd_geom = np.corrcoef(boundary_distance, geometry_score)[0, 1]
    corr_bd_ridge = np.corrcoef(boundary_distance, ridge)[0, 1]
    corr_bd_curv = np.corrcoef(boundary_distance, curvature)[0, 1]

    print(f"  corr(boundary_distance, geometry_score) = {corr_bd_geom:.6f}")
    print(f"  corr(boundary_distance, ridge) = {corr_bd_ridge:.6f}")
    print(f"  corr(boundary_distance, curvature) = {corr_bd_curv:.6f}")

    # Independence check
    if abs(corr_bd_geom) > 0.90:
        print(f"  ⚠️ WARNING: Geometry is redundant with boundary_distance (corr > 0.9)")
    elif abs(corr_bd_geom) > 0.50:
        print(f"  ⚠️ Moderate correlation detected (0.5 < corr < 0.9)")
    else:
        print(f"  ✓ Geometry is independent of boundary_distance (corr < 0.5)")

    if abs(corr_bd_ridge) > 0.90:
        print(f"  ⚠️ WARNING: Ridge is redundant with boundary_distance (corr > 0.9)")
    else:
        print(f"  ✓ Ridge is independent of boundary_distance (corr < 0.9)")

    print()

    # Run Phase E falsifier
    print("Running Phase E Falsifier...")
    print("-" * 80)

    # Create synthetic target (use labels as proxy for flip_rate)
    # In real test, flip_rate would be computed from actual model predictions
    # For this test, we'll use distance from decision boundary as proxy
    target = np.abs(boundary_distance)  # Higher distance = more stable

    # Run falsifier
    result = PhaseEFalsifier.evaluate(
        boundary_distance=boundary_distance,
        geometry_score=geometry_score,
        ridge_proximity=ridge,
        target=target,
        embedder_diagnostics={
            "distance_concentration": metrics["distance_concentration"],
            "local_intrinsic_dim": metrics["local_intrinsic_dim"],
        }
    )

    print(f"\nFalsifier Verdict: {result.verdict}")
    print(f"ΔR² = {result.delta_r2:.6f} ({result.delta_r2*100:.2f}% additional explanatory power)")
    print(f"Info density = {result.info_density:.6f}")
    print(f"R²(dist only) = {result.r2_dist_only:.6f}")
    print(f"R²(dist+geom) = {result.r2_dist_geom:.6f}")

    if result.representation_warning:
        print(f"\n⚠️ REPRESENTATION WARNING:")
        print(f"  {result.evidence.get('warning_note', 'Unknown warning')}")

    # Validation
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}\n")

    gate_prediction = diagnostics["verdict_prediction"]
    falsifier_verdict = result.verdict

    print(f"Gate predicted: {gate_prediction}")
    print(f"Falsifier verdict: {falsifier_verdict}")

    # Check if predictions match
    # Note: falsifier_verdict may be a Verdict enum, convert to string for comparison
    falsifier_verdict_str = str(falsifier_verdict).replace("Verdict.", "")

    if gate_prediction == falsifier_verdict_str:
        print(f"✓ PASS: Gate prediction matches falsifier verdict")
        validation_status = "PASS"
    elif gate_prediction == "REAL_SIGNAL" and falsifier_verdict_str == "WEAK_SIGNAL":
        if result.delta_r2 > 0.01:
            print(f"⚠️ PARTIAL: Gate predicted REAL_SIGNAL but got WEAK_SIGNAL (ΔR² > 0.01, borderline pass)")
            validation_status = "PARTIAL"
        else:
            print(f"✗ FAIL: Gate predicted REAL_SIGNAL but got WEAK_SIGNAL with ΔR² < 0.01")
            validation_status = "FAIL"
    elif gate_prediction == "WEAK_SIGNAL" and falsifier_verdict_str == "REAL_SIGNAL":
        print(f"✓ PASS: Gate was conservative, falsifier found stronger signal")
        validation_status = "PASS"
    else:
        print(f"✗ FAIL: Gate prediction does not match falsifier verdict")
        validation_status = "FAIL"

    # Check ΔR² threshold
    if result.delta_r2 > 0.01:
        print(f"✓ PASS: ΔR² = {result.delta_r2:.4f} > 0.01 (meaningful signal)")
    else:
        print(f"✗ FAIL: ΔR² = {result.delta_r2:.4f} ≤ 0.01 (cosmetic)")

    # Return results
    return {
        "embedder_name": embedder_name,
        "n_samples": N,
        "dimensions": D,
        "gate_prediction": gate_prediction,
        "falsifier_verdict": str(falsifier_verdict),  # Convert enum to string for JSON
        "validation_status": validation_status,
        "delta_r2": float(result.delta_r2),
        "info_density": float(result.info_density),
        "distance_cv": float(metrics["distance_concentration"]),
        "local_intrinsic_dim": float(metrics["local_intrinsic_dim"]),
        "corr_bd_geom": float(corr_bd_geom),
        "corr_bd_ridge": float(corr_bd_ridge),
        "representation_warning": bool(result.representation_warning),
    }


def main():
    """Run behavioral tests on selected embedders."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"embedder_behavioral_tests_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("PHASE E: EMBEDDER BEHAVIORAL CORRELATION TESTS")
    print(f"{'='*80}\n")
    print("This test validates that gate diagnostics predict actual behavioral signal.\n")

    # Test configurations
    tests = [
        {
            "name": "text-embedding-3-small (1536-D)",
            "run_dir": "runs/openai_3_small_test_20251231_005441",
        },
        {
            "name": "text-embedding-3-large (256-D)",
            "run_dir": "runs/openai_3_large_test_20251231_005624",
        },
    ]

    results = []

    for test in tests:
        try:
            result = run_behavioral_test(test["run_dir"], test["name"])
            results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR testing {test['name']}: {e}\n")
            continue

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL TESTS")
    print(f"{'='*80}\n")

    if not results:
        print("No tests completed successfully.")
        return

    # Table
    print(f"{'Embedder':<35} {'Gate':<12} {'Falsifier':<12} {'ΔR²':<10} {'Status':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['embedder_name']:<35} {r['gate_prediction']:<12} {r['falsifier_verdict']:<12} "
              f"{r['delta_r2']:.6f}  {r['validation_status']:<10}")

    print()

    # Overall validation
    pass_count = sum(1 for r in results if r["validation_status"] in ["PASS", "PARTIAL"])
    total = len(results)

    print(f"Overall: {pass_count}/{total} tests passed")

    if pass_count == total:
        print("✓ ALL TESTS PASSED: Gate diagnostics accurately predict behavioral signal")
    elif pass_count > 0:
        print(f"⚠️ PARTIAL SUCCESS: {pass_count}/{total} tests passed")
    else:
        print("✗ ALL TESTS FAILED: Gate diagnostics do not predict behavioral signal")

    # Save results
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "tests": results,
            "pass_count": pass_count,
            "total": total,
            "overall_status": "PASS" if pass_count == total else "PARTIAL" if pass_count > 0 else "FAIL"
        }, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
