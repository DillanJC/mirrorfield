"""
Investigate Dark River Detection

Analyzes why we're getting 0 dark river detections on real data.
Examines feature distributions and threshold appropriateness.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle


def analyze_feature_distributions():
    """Analyze feature distributions on real data."""
    print("="*80)
    print("DARK RIVER INVESTIGATION")
    print("="*80 + "\n")

    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    if not data_path.exists():
        print("⚠️  SKIP: Real data not found")
        return

    # Load data
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    print(f"Loaded data: N={len(embeddings)}, D={embeddings.shape[1]}")
    print()

    # Split
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    print(f"Reference: N={len(reference)}")
    print(f"Query: N={len(queries)}")
    print()

    # Compute geometry
    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries)

    # Extract features
    features = bundle.get_feature_matrix(results)
    flags = bundle.get_flags(results)

    # Feature names
    feature_names = [
        'knn_mean_distance',
        'knn_std_distance',
        'knn_min_distance',
        'knn_max_distance',
        'local_curvature',
        'ridge_proximity',
        'dist_to_ref_nearest'
    ]

    # Full distributions
    print("="*80)
    print("FEATURE DISTRIBUTIONS")
    print("="*80 + "\n")

    for i, name in enumerate(feature_names):
        values = features[:, i]
        print(f"{name}:")
        print(f"  Mean: {values.mean():.6f}")
        print(f"  Std: {values.std():.6f}")
        print(f"  Min: {values.min():.6f}")
        print(f"  Max: {values.max():.6f}")
        print(f"  Median: {np.median(values):.6f}")
        print(f"  25th percentile: {np.percentile(values, 25):.6f}")
        print(f"  75th percentile: {np.percentile(values, 75):.6f}")
        print(f"  95th percentile: {np.percentile(values, 95):.6f}")
        print(f"  99th percentile: {np.percentile(values, 99):.6f}")
        print()

    # Dark river analysis
    print("="*80)
    print("DARK RIVER ANALYSIS")
    print("="*80 + "\n")

    curvature = features[:, 4]
    ridge = features[:, 5]

    print("Current thresholds:")
    print(f"  Curvature < 0.5")
    print(f"  Ridge proximity > 2.0")
    print()

    # How many meet each criterion?
    low_curv = curvature < 0.5
    high_ridge = ridge > 2.0

    print(f"Points with low curvature (< 0.5): {low_curv.sum()}/{len(curvature)} ({100*low_curv.mean():.1f}%)")
    print(f"Points with high ridge (> 2.0): {high_ridge.sum()}/{len(ridge)} ({100*high_ridge.mean():.1f}%)")
    print(f"Points meeting BOTH criteria: {(low_curv & high_ridge).sum()}/{len(curvature)}")
    print()

    # What if we used different thresholds?
    print("Threshold sensitivity analysis:")
    print()

    curvature_thresholds = [0.01, 0.02, 0.05, 0.1, 0.5]
    ridge_thresholds = [0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0]

    print("Ridge proximity thresholds (with curvature < 0.5):")
    for ridge_thresh in ridge_thresholds:
        low_curv = curvature < 0.5
        high_ridge = ridge > ridge_thresh
        count = (low_curv & high_ridge).sum()
        pct = 100 * count / len(curvature)
        print(f"  Ridge > {ridge_thresh:.2f}: {count:3d} detections ({pct:5.1f}%)")
    print()

    print("Curvature thresholds (with ridge > 0.25):")
    for curv_thresh in curvature_thresholds:
        low_curv = curvature < curv_thresh
        high_ridge = ridge > 0.25
        count = (low_curv & high_ridge).sum()
        pct = 100 * count / len(curvature)
        print(f"  Curvature < {curv_thresh:.2f}: {count:3d} detections ({pct:5.1f}%)")
    print()

    # Ridge proximity interpretation
    print("="*80)
    print("RIDGE PROXIMITY INTERPRETATION")
    print("="*80 + "\n")

    print("Ridge proximity = σ/μ (std/mean of k-NN distances)")
    print()
    print("Physical meaning:")
    print("  High ridge (> 2.0): σ > 2×μ → highly variable neighbor distances")
    print("                      Indicates rapid density changes (near boundary?)")
    print()
    print("  Medium ridge (~0.2): σ ≈ 0.2×μ → consistent neighbor distances")
    print("                       Indicates smooth, uniform density")
    print()
    print("Observation:")
    ridge_max = ridge.max()
    ridge_95 = np.percentile(ridge, 95)
    print(f"  Max ridge in dataset: {ridge_max:.3f}")
    print(f"  95th percentile: {ridge_95:.3f}")
    print()

    if ridge_max < 1.0:
        print("⚠️  FINDING: Ridge proximity values are ALL < 1.0")
        print("   This suggests the embedding space has very uniform density")
        print("   Threshold of 2.0 is unrealistic for this data")
        print()

    # Correlation with boundary distance
    print("="*80)
    print("CORRELATION WITH BOUNDARY DISTANCE")
    print("="*80 + "\n")

    from scipy.stats import pearsonr

    for i, name in enumerate(feature_names):
        corr, pval = pearsonr(features[:, i], query_boundaries)
        print(f"{name}:")
        print(f"  Correlation: {corr:+.3f} (p={pval:.2e})")
    print()

    # Find points closest to boundary
    print("="*80)
    print("BOUNDARY PROXIMITY ANALYSIS")
    print("="*80 + "\n")

    # Sort by boundary distance (closest to boundary first)
    sorted_indices = np.argsort(query_boundaries)
    closest_10 = sorted_indices[:10]
    farthest_10 = sorted_indices[-10:]

    print("10 points CLOSEST to boundary:")
    print(f"  Boundary distances: {query_boundaries[closest_10]}")
    print(f"  Curvature: {curvature[closest_10]}")
    print(f"  Ridge proximity: {ridge[closest_10]}")
    print()

    print("10 points FARTHEST from boundary:")
    print(f"  Boundary distances: {query_boundaries[farthest_10]}")
    print(f"  Curvature: {curvature[farthest_10]}")
    print(f"  Ridge proximity: {ridge[farthest_10]}")
    print()

    # Recommended thresholds
    print("="*80)
    print("RECOMMENDED THRESHOLDS")
    print("="*80 + "\n")

    # Use percentile-based thresholds
    curvature_p50 = np.percentile(curvature, 50)
    ridge_p75 = np.percentile(ridge, 75)
    ridge_p90 = np.percentile(ridge, 90)
    ridge_p95 = np.percentile(ridge, 95)

    print("Data-driven threshold suggestions:")
    print()
    print(f"Option A (Conservative - flag ~5%):")
    print(f"  Curvature < {curvature_p50:.3f} (median)")
    print(f"  Ridge > {ridge_p95:.3f} (95th percentile)")
    low_curv = curvature < curvature_p50
    high_ridge = ridge > ridge_p95
    count = (low_curv & high_ridge).sum()
    print(f"  Would detect: {count}/{len(curvature)} ({100*count/len(curvature):.1f}%)")
    print()

    print(f"Option B (Moderate - flag ~10-15%):")
    print(f"  Curvature < {curvature_p50:.3f} (median)")
    print(f"  Ridge > {ridge_p90:.3f} (90th percentile)")
    low_curv = curvature < curvature_p50
    high_ridge = ridge > ridge_p90
    count = (low_curv & high_ridge).sum()
    print(f"  Would detect: {count}/{len(curvature)} ({100*count/len(curvature):.1f}%)")
    print()

    print(f"Option C (Liberal - flag ~25%):")
    print(f"  Curvature < {curvature_p50:.3f} (median)")
    print(f"  Ridge > {ridge_p75:.3f} (75th percentile)")
    low_curv = curvature < curvature_p50
    high_ridge = ridge > ridge_p75
    count = (low_curv & high_ridge).sum()
    print(f"  Would detect: {count}/{len(curvature)} ({100*count/len(curvature):.1f}%)")
    print()


if __name__ == "__main__":
    analyze_feature_distributions()
