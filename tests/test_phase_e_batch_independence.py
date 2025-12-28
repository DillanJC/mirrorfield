"""Phase E Test — Batch Independence (Reproducibility Guarantee)

This test validates that geometry features are batch-independent:
- Shuffling query order must produce permuted outputs (not different values)
- Duplicating queries must produce duplicated outputs
- Different batch sizes must produce identical results

This is a critical correctness constraint for Phase E.
"""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from geometry.bundle import GeometryBundle
from geometry.features import local_curvature_svd_gpu, ridge_proximity_from_knn_distances


def test_curvature_batch_independence():
    """Test that curvature is independent of batch size and ordering."""
    # Create synthetic reference set
    rng = np.random.RandomState(42)
    n_ref = 1000
    D = 64
    ref_points = rng.randn(n_ref, D).astype(np.float32)

    # Create query points
    n_query = 100
    query = rng.randn(n_query, D).astype(np.float32)

    # Fit kNN index
    nn = NearestNeighbors(n_neighbors=16, algorithm='auto', metric='euclidean')
    nn.fit(ref_points)
    distances, indices = nn.kneighbors(query)

    # Compute curvature with batch_size=32
    curv_batch32 = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices,
        r=4,
        batch_size=32,
        device="cpu",
    )

    # Compute curvature with batch_size=10
    curv_batch10 = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices,
        r=4,
        batch_size=10,
        device="cpu",
    )

    # Results should be identical
    np.testing.assert_allclose(
        curv_batch32,
        curv_batch10,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Curvature changed with different batch sizes"
    )


def test_curvature_shuffle_invariance():
    """Test that shuffling queries produces permuted outputs."""
    # Create synthetic reference set
    rng = np.random.RandomState(42)
    n_ref = 1000
    D = 64
    ref_points = rng.randn(n_ref, D).astype(np.float32)

    # Create query points
    n_query = 50
    query = rng.randn(n_query, D).astype(np.float32)

    # Fit kNN index
    nn = NearestNeighbors(n_neighbors=16, algorithm='auto', metric='euclidean')
    nn.fit(ref_points)
    distances, indices = nn.kneighbors(query)

    # Compute curvature in original order
    curv_original = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices,
        r=4,
        batch_size=16,
        device="cpu",
    )

    # Shuffle query order
    perm = rng.permutation(n_query)
    query_shuffled = query[perm]
    distances_shuffled, indices_shuffled = nn.kneighbors(query_shuffled)

    # Compute curvature in shuffled order
    curv_shuffled = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices_shuffled,
        r=4,
        batch_size=16,
        device="cpu",
    )

    # Unshuffle and compare
    curv_unshuffled = np.empty_like(curv_shuffled)
    curv_unshuffled[perm] = curv_shuffled

    np.testing.assert_allclose(
        curv_original,
        curv_unshuffled,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Curvature changed under query shuffling"
    )


def test_curvature_duplicate_queries():
    """Test that duplicating queries produces duplicated outputs."""
    # Create synthetic reference set
    rng = np.random.RandomState(42)
    n_ref = 1000
    D = 64
    ref_points = rng.randn(n_ref, D).astype(np.float32)

    # Create query points
    n_query = 25
    query = rng.randn(n_query, D).astype(np.float32)

    # Duplicate queries
    query_doubled = np.vstack([query, query])

    # Fit kNN index
    nn = NearestNeighbors(n_neighbors=16, algorithm='auto', metric='euclidean')
    nn.fit(ref_points)

    # Compute for original
    distances, indices = nn.kneighbors(query)
    curv_original = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices,
        r=4,
        batch_size=16,
        device="cpu",
    )

    # Compute for doubled
    distances_doubled, indices_doubled = nn.kneighbors(query_doubled)
    curv_doubled = local_curvature_svd_gpu(
        ref_points=ref_points,
        neighbor_indices=indices_doubled,
        r=4,
        batch_size=16,
        device="cpu",
    )

    # First half should match original
    np.testing.assert_allclose(
        curv_original,
        curv_doubled[:n_query],
        rtol=1e-5,
        atol=1e-7,
        err_msg="First half of doubled queries doesn't match original"
    )

    # Second half should also match original
    np.testing.assert_allclose(
        curv_original,
        curv_doubled[n_query:],
        rtol=1e-5,
        atol=1e-7,
        err_msg="Second half of doubled queries doesn't match original"
    )


def test_ridge_batch_independence():
    """Test that ridge proximity is batch-independent."""
    # Ridge proximity is computed from kNN distances, so it's inherently
    # batch-independent. This test validates the implementation.
    rng = np.random.RandomState(42)
    n_query = 100
    k = 32

    # Create synthetic distance matrix
    distances = rng.rand(n_query, k).astype(np.float32)
    distances.sort(axis=1)  # Ensure ascending order as kNN returns

    # Compute ridge for all queries
    ridge_all = ridge_proximity_from_knn_distances(distances)

    # Compute ridge for first half
    ridge_half1 = ridge_proximity_from_knn_distances(distances[:50])

    # Compute ridge for second half
    ridge_half2 = ridge_proximity_from_knn_distances(distances[50:])

    # Concatenate halves
    ridge_concat = np.concatenate([ridge_half1, ridge_half2])

    # Should match
    np.testing.assert_allclose(
        ridge_all,
        ridge_concat,
        rtol=1e-7,
        atol=1e-9,
        err_msg="Ridge proximity changed when computed in batches"
    )


def test_bundle_batch_independence():
    """Test that GeometryBundle.transform is batch-independent."""
    # Create synthetic reference set
    rng = np.random.RandomState(42)
    n_ref = 500
    D = 64
    ref_points = rng.randn(n_ref, D).astype(np.float32)

    # Fit kNN index
    nn = NearestNeighbors(n_neighbors=32, algorithm='auto', metric='euclidean')
    nn.fit(ref_points)

    # Create config
    config = {
        "random_seed": 42,
        "n_neighbors": 32,
        "k_curvature": 16,
        "r_components": 4,
        "batch_size": 32,
        "device": "cpu",
    }

    # Create bundle
    bundle = GeometryBundle(ref_points=ref_points, nn_index=nn, config=config)

    # Create query
    n_query = 50
    query = rng.randn(n_query, D).astype(np.float32)
    bd = rng.randn(n_query).astype(np.float32)

    # Transform with original config
    result1 = bundle.transform(query, bd)

    # Transform with different batch size
    bundle.config["batch_size"] = 10
    result2 = bundle.transform(query, bd)

    # Curvature should match
    np.testing.assert_allclose(
        result1["local_curvature"],
        result2["local_curvature"],
        rtol=1e-5,
        atol=1e-7,
        err_msg="Curvature changed with different batch size in bundle"
    )

    # Ridge should match
    np.testing.assert_allclose(
        result1["ridge_proximity"],
        result2["ridge_proximity"],
        rtol=1e-7,
        atol=1e-9,
        err_msg="Ridge changed with different batch size in bundle"
    )


if __name__ == "__main__":
    test_curvature_batch_independence()
    test_curvature_shuffle_invariance()
    test_curvature_duplicate_queries()
    test_ridge_batch_independence()
    test_bundle_batch_independence()
    print("✓ All batch independence tests passed")
