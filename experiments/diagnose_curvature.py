"""
Diagnostic script for local curvature computation issue.

Investigates why local_curvature values are all ~0 in Phase E test.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors


def diagnose_curvature_single_point(query, reference, k=50):
    """
    Diagnose curvature computation for a single query point.

    Returns detailed diagnostics about eigenvalues and covariance.
    """
    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
    nn.fit(reference)
    distances, indices = nn.kneighbors(query.reshape(1, -1))

    # Get neighbors
    neighbors = reference[indices[0]]

    # Center around query point
    centered = neighbors - query

    # Compute covariance
    cov = np.cov(centered.T)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)

    # Sort eigenvalues
    eigenvalues_sorted = np.sort(eigenvalues)

    diagnostics = {
        'distances': distances[0],
        'distance_mean': distances[0].mean(),
        'distance_std': distances[0].std(),
        'centered_shape': centered.shape,
        'centered_norm_mean': np.linalg.norm(centered, axis=1).mean(),
        'cov_shape': cov.shape,
        'cov_trace': np.trace(cov),
        'cov_det': np.linalg.det(cov),
        'cov_frobenius_norm': np.linalg.norm(cov, 'fro'),
        'eigenvalues_min': eigenvalues_sorted[0],
        'eigenvalues_max': eigenvalues_sorted[-1],
        'eigenvalues_ratio': eigenvalues_sorted[0] / (eigenvalues_sorted[-1] + 1e-15),
        'eigenvalues_all': eigenvalues_sorted,
        'rank': np.linalg.matrix_rank(cov),
        'condition_number': np.linalg.cond(cov)
    }

    return diagnostics


def test_synthetic_data():
    """Test on synthetic data to establish baseline."""
    print("="*80)
    print("TEST 1: Synthetic Random Data")
    print("="*80 + "\n")

    np.random.seed(42)
    reference = np.random.randn(200, 256).astype(np.float32)
    query = np.random.randn(256).astype(np.float32)

    diag = diagnose_curvature_single_point(query, reference, k=50)

    print(f"Distance statistics:")
    print(f"  Mean: {diag['distance_mean']:.4f}")
    print(f"  Std: {diag['distance_std']:.4f}")
    print()

    print(f"Centered neighbors:")
    print(f"  Shape: {diag['centered_shape']}")
    print(f"  Mean norm: {diag['centered_norm_mean']:.4f}")
    print()

    print(f"Covariance matrix:")
    print(f"  Shape: {diag['cov_shape']}")
    print(f"  Trace: {diag['cov_trace']:.4f}")
    print(f"  Frobenius norm: {diag['cov_frobenius_norm']:.4f}")
    print(f"  Rank: {diag['rank']}")
    print(f"  Condition number: {diag['condition_number']:.2e}")
    print()

    print(f"Eigenvalues:")
    print(f"  Min (λ_min): {diag['eigenvalues_min']:.6e}")
    print(f"  Max (λ_max): {diag['eigenvalues_max']:.6e}")
    print(f"  Ratio (λ_min/λ_max): {diag['eigenvalues_ratio']:.6e}")
    print(f"  Top 10 eigenvalues: {diag['eigenvalues_all'][-10:]}")
    print()

    # Check if ratio would be set to 0
    if diag['eigenvalues_max'] > 1e-10:
        curvature = diag['eigenvalues_ratio']
        print(f"✓ Curvature computed: {curvature:.6e}")
    else:
        print(f"✗ Curvature set to 0 (λ_max too small)")

    print()
    return diag


def test_real_data():
    """Test on real sentiment classification data."""
    print("="*80)
    print("TEST 2: Real Sentiment Data")
    print("="*80 + "\n")

    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    if not data_path.exists():
        print("⚠️  SKIP: Real data not found")
        return None

    embeddings = np.load(data_path / "embeddings.npy")

    print(f"Loaded data: N={len(embeddings)}, D={embeddings.shape[1]}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Embedding norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print()

    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    query = embeddings[split]  # First query point

    diag = diagnose_curvature_single_point(query, reference, k=50)

    print(f"Distance statistics:")
    print(f"  Mean: {diag['distance_mean']:.4f}")
    print(f"  Std: {diag['distance_std']:.4f}")
    print()

    print(f"Centered neighbors:")
    print(f"  Shape: {diag['centered_shape']}")
    print(f"  Mean norm: {diag['centered_norm_mean']:.4f}")
    print()

    print(f"Covariance matrix:")
    print(f"  Shape: {diag['cov_shape']}")
    print(f"  Trace: {diag['cov_trace']:.4f}")
    print(f"  Frobenius norm: {diag['cov_frobenius_norm']:.4f}")
    print(f"  Rank: {diag['rank']}")
    print(f"  Condition number: {diag['condition_number']:.2e}")
    print()

    print(f"Eigenvalues:")
    print(f"  Min (λ_min): {diag['eigenvalues_min']:.6e}")
    print(f"  Max (λ_max): {diag['eigenvalues_max']:.6e}")
    print(f"  Ratio (λ_min/λ_max): {diag['eigenvalues_ratio']:.6e}")
    print(f"  Top 10 eigenvalues: {diag['eigenvalues_all'][-10:]}")
    print(f"  Bottom 10 eigenvalues: {diag['eigenvalues_all'][:10]}")
    print()

    # Check if ratio would be set to 0
    if diag['eigenvalues_max'] > 1e-10:
        curvature = diag['eigenvalues_ratio']
        print(f"✓ Curvature computed: {curvature:.6e}")
    else:
        print(f"✗ Curvature set to 0 (λ_max too small)")

    print()

    # Test multiple query points
    print("Testing 10 random query points:")
    ratios = []
    for i in range(10):
        query_i = embeddings[split + i]
        diag_i = diagnose_curvature_single_point(query_i, reference, k=50)
        ratios.append(diag_i['eigenvalues_ratio'])
        print(f"  Point {i}: ratio = {diag_i['eigenvalues_ratio']:.6e}")

    print()
    print(f"Ratio statistics (n=10):")
    print(f"  Mean: {np.mean(ratios):.6e}")
    print(f"  Std: {np.std(ratios):.6e}")
    print(f"  Min: {np.min(ratios):.6e}")
    print(f"  Max: {np.max(ratios):.6e}")
    print()

    return diag


def test_dtype_precision():
    """Test if float32 vs float64 matters."""
    print("="*80)
    print("TEST 3: Float32 vs Float64 Precision")
    print("="*80 + "\n")

    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    if not data_path.exists():
        print("⚠️  SKIP: Real data not found")
        return

    embeddings = np.load(data_path / "embeddings.npy")
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    query = embeddings[split]

    # Test with float32
    reference_f32 = reference.astype(np.float32)
    query_f32 = query.astype(np.float32)
    diag_f32 = diagnose_curvature_single_point(query_f32, reference_f32, k=50)

    # Test with float64
    reference_f64 = reference.astype(np.float64)
    query_f64 = query.astype(np.float64)
    diag_f64 = diagnose_curvature_single_point(query_f64, reference_f64, k=50)

    print("Float32 results:")
    print(f"  λ_min: {diag_f32['eigenvalues_min']:.6e}")
    print(f"  λ_max: {diag_f32['eigenvalues_max']:.6e}")
    print(f"  Ratio: {diag_f32['eigenvalues_ratio']:.6e}")
    print()

    print("Float64 results:")
    print(f"  λ_min: {diag_f64['eigenvalues_min']:.6e}")
    print(f"  λ_max: {diag_f64['eigenvalues_max']:.6e}")
    print(f"  Ratio: {diag_f64['eigenvalues_ratio']:.6e}")
    print()

    print("Difference:")
    print(f"  Δ(λ_min): {abs(diag_f32['eigenvalues_min'] - diag_f64['eigenvalues_min']):.6e}")
    print(f"  Δ(λ_max): {abs(diag_f32['eigenvalues_max'] - diag_f64['eigenvalues_max']):.6e}")
    print(f"  Δ(Ratio): {abs(diag_f32['eigenvalues_ratio'] - diag_f64['eigenvalues_ratio']):.6e}")
    print()


def main():
    print("\n" + "="*80)
    print("CURVATURE DIAGNOSTIC")
    print("="*80 + "\n")

    # Test 1: Synthetic data
    diag_synthetic = test_synthetic_data()

    # Test 2: Real data
    diag_real = test_real_data()

    # Test 3: Precision
    test_dtype_precision()

    # Summary
    print("="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80 + "\n")

    if diag_real is not None:
        print("Root cause analysis:")
        print()

        # Check if it's a rank deficiency issue
        if diag_real['rank'] < diag_real['cov_shape'][0]:
            print(f"⚠️  RANK DEFICIENCY DETECTED")
            print(f"   Covariance matrix rank: {diag_real['rank']}/{diag_real['cov_shape'][0]}")
            print(f"   This is expected when k={50} < D={256}")
            print(f"   With k neighbors in D dimensions, covariance is at most rank k")
            print()

        # Check if eigenvalues are too small
        if diag_real['eigenvalues_ratio'] < 1e-6:
            print(f"⚠️  EXTREMELY SMALL EIGENVALUE RATIO")
            print(f"   Ratio: {diag_real['eigenvalues_ratio']:.6e}")
            print(f"   This causes float32 underflow to 0")
            print()

        # Check condition number
        if diag_real['condition_number'] > 1e10:
            print(f"⚠️  ILL-CONDITIONED COVARIANCE MATRIX")
            print(f"   Condition number: {diag_real['condition_number']:.2e}")
            print(f"   Matrix is numerically unstable")
            print()

        print("Recommended fixes:")
        print("1. Use float64 for eigenvalue computation")
        print("2. Store result as log(λ_min/λ_max) to handle small ratios")
        print("3. Add epsilon to denominator: λ_min / (λ_max + ε)")
        print("4. Consider using SVD instead of eigendecomposition")
        print()


if __name__ == "__main__":
    main()
