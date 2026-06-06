"""
Test improved curvature computation using SVD.

Compares:
1. Original eigenvalue method (unstable)
2. SVD method (stable)
3. Variance explained ratio (alternative)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors


def compute_curvature_original(query, neighbors):
    """Original method: eigenvalues of covariance matrix."""
    centered = neighbors - query
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)

    if eigenvalues[-1] > 1e-10:
        ratio = eigenvalues[0] / eigenvalues[-1]
    else:
        ratio = 0.0

    return ratio


def compute_curvature_svd(query, neighbors):
    """
    Improved method: SVD of centered neighbor matrix.

    More stable than covariance eigendecomposition when k << D.
    Returns ratio of smallest to largest singular value.
    """
    centered = neighbors - query
    # SVD of (k, D) matrix directly
    # U: (k, k), S: (k,), Vt: (k, D)
    # NOTE: S is sorted in DESCENDING order: S[0] = largest, S[-1] = smallest
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Curvature = smallest/largest singular value (measures anisotropy)
    # Small ratio (close to 0) = high anisotropy (stretched, low curvature)
    # Large ratio (close to 1) = isotropy (spherical, high curvature)
    if S[0] > 1e-10:
        ratio = S[-1] / S[0]  # smallest/largest
    else:
        ratio = 0.0

    return ratio, S


def compute_curvature_variance_explained(query, neighbors):
    """
    Alternative: Variance explained by principal direction.

    Returns: 1 - (variance in principal direction / total variance)
    High value = low curvature (anisotropic)
    Low value = high curvature (isotropic)
    """
    centered = neighbors - query
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Variance = S^2
    variance = S ** 2
    total_var = variance.sum()
    principal_var = variance[-1]  # Largest variance

    if total_var > 1e-10:
        ratio = 1.0 - (principal_var / total_var)
    else:
        ratio = 0.0

    return ratio, variance


def test_all_methods():
    """Compare all three methods on real data."""
    print("="*80)
    print("CURVATURE METHOD COMPARISON")
    print("="*80 + "\n")

    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    if not data_path.exists():
        print("⚠️  SKIP: Real data not found")
        return

    embeddings = np.load(data_path / "embeddings.npy")
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:split+20]  # Test on 20 points

    print(f"Testing on {len(queries)} query points")
    print()

    # Find neighbors
    nn = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='euclidean')
    nn.fit(reference)

    results = []

    for i, query in enumerate(queries):
        distances, indices = nn.kneighbors(query.reshape(1, -1))
        neighbors = reference[indices[0]]

        # Method 1: Original (covariance eigenvalues)
        curv_orig = compute_curvature_original(query, neighbors)

        # Method 2: SVD
        curv_svd, singular_values = compute_curvature_svd(query, neighbors)

        # Method 3: Variance explained
        curv_varexp, variance = compute_curvature_variance_explained(query, neighbors)

        results.append({
            'original': curv_orig,
            'svd': curv_svd,
            'variance_explained': curv_varexp,
            'singular_values': singular_values,
            'variance': variance
        })

        if i < 5:  # Print first 5 in detail
            print(f"Point {i}:")
            print(f"  Original (eigenvalue ratio): {curv_orig:.6e}")
            print(f"  SVD (singular value ratio): {curv_svd:.6e}")
            print(f"  Variance explained: {curv_varexp:.6f}")
            print(f"  Singular values (top 5): {singular_values[-5:]}")
            print(f"  Singular values (bottom 5): {singular_values[:5]}")
            print()

    # Summary statistics
    print("="*80)
    print("SUMMARY STATISTICS (n=20)")
    print("="*80 + "\n")

    orig_vals = [r['original'] for r in results]
    svd_vals = [r['svd'] for r in results]
    varexp_vals = [r['variance_explained'] for r in results]

    print("Original method (covariance eigenvalues):")
    print(f"  Mean: {np.mean(orig_vals):.6e}")
    print(f"  Std: {np.std(orig_vals):.6e}")
    print(f"  Min: {np.min(orig_vals):.6e}")
    print(f"  Max: {np.max(orig_vals):.6e}")
    print(f"  % non-zero: {100 * np.count_nonzero(orig_vals) / len(orig_vals):.1f}%")
    print()

    print("SVD method (singular value ratio):")
    print(f"  Mean: {np.mean(svd_vals):.6e}")
    print(f"  Std: {np.std(svd_vals):.6e}")
    print(f"  Min: {np.min(svd_vals):.6e}")
    print(f"  Max: {np.max(svd_vals):.6e}")
    print(f"  % non-zero: {100 * np.count_nonzero(svd_vals) / len(svd_vals):.1f}%")
    print()

    print("Variance explained method:")
    print(f"  Mean: {np.mean(varexp_vals):.6f}")
    print(f"  Std: {np.std(varexp_vals):.6f}")
    print(f"  Min: {np.min(varexp_vals):.6f}")
    print(f"  Max: {np.max(varexp_vals):.6f}")
    print(f"  % non-zero: {100 * np.count_nonzero(varexp_vals) / len(varexp_vals):.1f}%")
    print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80 + "\n")

    if np.count_nonzero(orig_vals) == 0:
        print("✗ Original method: ALL VALUES ARE ZERO (completely broken)")
    else:
        print(f"⚠️  Original method: Only {100 * np.count_nonzero(orig_vals) / len(orig_vals):.0f}% non-zero")

    print()
    print("✓ RECOMMENDED: Use SVD method")
    print("  - More numerically stable")
    print("  - Works directly on (k, D) matrix")
    print("  - Avoids covariance matrix entirely")
    print()

    print("Alternative: Variance explained ratio")
    print("  - Different interpretation (fraction of variance NOT in principal direction)")
    print("  - Range: [0, 1] (easier to interpret)")
    print("  - May be more intuitive for ML models")
    print()


if __name__ == "__main__":
    test_all_methods()
