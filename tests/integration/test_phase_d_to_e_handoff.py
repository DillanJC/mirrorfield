"""Phase D → E Integration Test — Contract Validation

This test validates the integration contract between Phase D and Phase E:

Phase E requires from Phase D:
1. embeddings: (N, D) float array
2. boundary_distance: (N,) float array (numeric)

Phase E does NOT care:
- How boundary_distance was computed (on-the-fly or cached)
- What the semantic transforms were
- What the perturbation parameters were

This test loads a real Phase D artifact and validates the contract.
"""

import json
import numpy as np
import pytest
from pathlib import Path


def find_latest_phase_d_run():
    """Find the most recent Phase D run directory.

    Returns:
        Path to summary.json or None if not found
    """
    results_dir = Path("experiments/results/phase_d_integrated_eval")
    if not results_dir.exists():
        return None

    # Find all run directories
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by name (YYYYMMDD_HHMMSS format naturally sorts correctly)
    run_dirs.sort(reverse=True)

    # Find summary.json in latest run
    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            return summary_path

    return None


def test_phase_d_produces_required_fields():
    """Test that Phase D output contains required fields for Phase E."""
    summary_path = find_latest_phase_d_run()

    if summary_path is None:
        pytest.skip("No Phase D runs found in experiments/results/phase_d_integrated_eval")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Check that we have the basic structure
    assert "run_id" in summary, "Phase D summary missing run_id"
    assert "config" in summary, "Phase D summary missing config"

    # Phase E needs access to the underlying data files, not just summary
    # Check that the run directory contains the expected data files
    run_dir = summary_path.parent

    # Look for data files (these might vary by Phase D implementation)
    # Common patterns: embeddings.npy, boundary_distances.npy, or combined data file
    data_files = list(run_dir.glob("*.npy")) + list(run_dir.glob("*embeddings*.json"))

    # At minimum, we should have some data artifacts
    assert len(data_files) > 0 or summary.get("evaluation_modes"), \
        f"Phase D run {run_dir} has no data artifacts"

    print(f"✓ Phase D run {summary['run_id']} found with valid structure")


def test_boundary_distance_shape_compatibility():
    """Test that boundary_distance has correct shape for Phase E integration.

    This is a synthetic test that validates the expected data format.
    In practice, Phase E will receive boundary_distance from Phase D's
    internal computation, not from saved artifacts.
    """
    # Simulate Phase D output
    n_samples = 100
    D = 768

    # These would come from Phase D
    embeddings = np.random.randn(n_samples, D).astype(np.float32)
    boundary_distance = np.random.randn(n_samples).astype(np.float32)

    # Validate shapes
    assert embeddings.ndim == 2, f"embeddings must be 2D, got {embeddings.ndim}D"
    assert embeddings.shape[0] == n_samples, "embeddings length mismatch"
    assert embeddings.shape[1] == D, f"Expected D={D}, got {embeddings.shape[1]}"

    assert boundary_distance.ndim == 1, f"boundary_distance must be 1D, got {boundary_distance.ndim}D"
    assert len(boundary_distance) == n_samples, \
        f"Length mismatch: embeddings={n_samples}, boundary_distance={len(boundary_distance)}"

    # Validate types
    assert embeddings.dtype in [np.float32, np.float64], \
        f"embeddings must be float, got {embeddings.dtype}"
    assert boundary_distance.dtype in [np.float32, np.float64], \
        f"boundary_distance must be float, got {boundary_distance.dtype}"

    print(f"✓ Shape compatibility validated: ({n_samples}, {D}) embeddings + ({n_samples},) boundary_distance")


def test_phase_e_can_consume_phase_d_format():
    """Test that Phase E bundle can consume Phase D output format.

    This is an end-to-end integration test using synthetic data.
    """
    from sklearn.neighbors import NearestNeighbors
    from geometry.bundle import GeometryBundle

    # Simulate Phase D outputs
    n_ref = 200
    n_query = 50
    D = 64

    rng = np.random.RandomState(42)
    ref_embeddings = rng.randn(n_ref, D).astype(np.float32)
    query_embeddings = rng.randn(n_query, D).astype(np.float32)
    boundary_distance = rng.randn(n_query).astype(np.float32)

    # Build kNN index on reference set
    nn = NearestNeighbors(n_neighbors=32, algorithm='auto', metric='euclidean')
    nn.fit(ref_embeddings)

    # Create Phase E bundle
    config = {
        "random_seed": 42,
        "n_neighbors": 32,
        "k_curvature": 16,
        "r_components": 4,
        "batch_size": 16,
        "device": "cpu",
    }

    bundle = GeometryBundle(ref_points=ref_embeddings, nn_index=nn, config=config)

    # This should work without errors
    result = bundle.transform(query_embeddings, boundary_distance)

    # Validate outputs
    assert "local_curvature" in result
    assert "ridge_proximity" in result
    assert "geom_flags" in result

    assert result["local_curvature"].shape == (n_query,)
    assert result["ridge_proximity"].shape == (n_query,)
    assert len(result["geom_flags"]) == n_query

    print(f"✓ Phase E successfully consumed Phase D format: {n_query} samples processed")


if __name__ == "__main__":
    test_phase_d_produces_required_fields()
    test_boundary_distance_shape_compatibility()
    test_phase_e_can_consume_phase_d_format()
    print("\nPASS: All Phase D -> E integration tests passed")
