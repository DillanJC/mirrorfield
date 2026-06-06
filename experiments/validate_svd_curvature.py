"""
Validate SVD Curvature Fix

Quick validation that SVD-based curvature maintains (or improves) the
validated improvement over baseline.

Runs 20 independent trials:
- Baseline: embeddings only
- Geometry: embeddings + 7 k-NN features (with SVD curvature)

Tests on borderline slice (|boundary_distance| < 0.5).
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mirrorfield.geometry import GeometryBundle
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def load_data():
    """Load real sentiment data."""
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    return embeddings, boundary_distances


def compute_geometry_features(embeddings, boundary_distances):
    """Compute geometry bundle features with SVD curvature."""
    # Split into reference and query
    split = int(len(embeddings) * 0.8)
    reference = embeddings[:split]
    queries = embeddings[split:]
    query_boundaries = boundary_distances[split:]

    # Compute geometry
    bundle = GeometryBundle(reference, k=50)
    results = bundle.compute(queries)

    # Extract feature matrix
    features = bundle.get_feature_matrix(results)

    return features, query_boundaries


def run_single_trial(seed):
    """Run a single trial with given random seed."""
    np.random.seed(seed)

    # Load data
    embeddings, boundary_distances = load_data()

    # Compute geometry features
    geometry_features, query_boundaries = compute_geometry_features(embeddings, boundary_distances)

    # Combine embeddings + geometry
    split = int(len(embeddings) * 0.8)
    query_embeddings = embeddings[split:]

    # Baseline: embeddings only
    X_baseline = query_embeddings

    # Geometry: embeddings + 7 k-NN features
    X_geometry = np.concatenate([query_embeddings, geometry_features], axis=1)

    # Target: boundary distance
    y = query_boundaries

    # Borderline slice
    borderline_mask = np.abs(y) < 0.5

    X_baseline_borderline = X_baseline[borderline_mask]
    X_geometry_borderline = X_geometry[borderline_mask]
    y_borderline = y[borderline_mask]

    # Train Ridge regression (with some regularization for stability)
    ridge_baseline = Ridge(alpha=1.0, random_state=seed)
    ridge_baseline.fit(X_baseline_borderline, y_borderline)
    pred_baseline = ridge_baseline.predict(X_baseline_borderline)
    r2_baseline = r2_score(y_borderline, pred_baseline)

    ridge_geometry = Ridge(alpha=1.0, random_state=seed)
    ridge_geometry.fit(X_geometry_borderline, y_borderline)
    pred_geometry = ridge_geometry.predict(X_geometry_borderline)
    r2_geometry = r2_score(y_borderline, pred_geometry)

    improvement = r2_geometry - r2_baseline
    improvement_pct = 100 * improvement / (abs(r2_baseline) + 1e-10)

    return {
        'seed': seed,
        'r2_baseline': float(r2_baseline),
        'r2_geometry': float(r2_geometry),
        'improvement': float(improvement),
        'improvement_pct': float(improvement_pct),
        'n_borderline': int(borderline_mask.sum())
    }


def main():
    print("\n" + "="*80)
    print("SVD CURVATURE VALIDATION")
    print("="*80 + "\n")

    print("Testing: Do SVD-based geometry features still improve over baseline?")
    print("Method: 20 independent trials with different random seeds")
    print("Metric: R² on borderline slice (|boundary_distance| < 0.5)")
    print()

    n_trials = 20
    results = []

    print(f"Running {n_trials} trials...")
    print()

    for i in range(n_trials):
        seed = 42 + i
        result = run_single_trial(seed)
        results.append(result)

        print(f"Trial {i+1:2d}: Baseline R²={result['r2_baseline']:.3f}, "
              f"Geometry R²={result['r2_geometry']:.3f}, "
              f"Δ={result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")

    print()

    # Aggregate statistics
    r2_baseline_mean = np.mean([r['r2_baseline'] for r in results])
    r2_baseline_std = np.std([r['r2_baseline'] for r in results])

    r2_geometry_mean = np.mean([r['r2_geometry'] for r in results])
    r2_geometry_std = np.std([r['r2_geometry'] for r in results])

    improvement_mean = np.mean([r['improvement'] for r in results])
    improvement_std = np.std([r['improvement'] for r in results])
    improvement_pct_mean = np.mean([r['improvement_pct'] for r in results])

    # Count wins
    n_wins = sum(1 for r in results if r['improvement'] > 0)
    win_rate = 100 * n_wins / n_trials

    print("="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Baseline (embeddings only):")
    print(f"  Mean R²: {r2_baseline_mean:.3f} ± {r2_baseline_std:.3f}")
    print()

    print(f"Geometry (embeddings + 7 k-NN features with SVD curvature):")
    print(f"  Mean R²: {r2_geometry_mean:.3f} ± {r2_geometry_std:.3f}")
    print()

    print(f"Improvement:")
    print(f"  Mean Δ R²: {improvement_mean:+.4f} ± {improvement_std:.3f}")
    print(f"  Mean %: {improvement_pct_mean:+.1f}%")
    print(f"  Win rate: {n_wins}/{n_trials} ({win_rate:.0f}%)")
    print()

    # Statistical test
    from scipy.stats import ttest_1samp
    t_stat, p_value = ttest_1samp([r['improvement'] for r in results], 0)

    print(f"Statistical significance (one-sample t-test):")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.2e}")

    if p_value < 0.001:
        print(f"  Result: *** HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value < 0.01:
        print(f"  Result: ** SIGNIFICANT (p < 0.01)")
    elif p_value < 0.05:
        print(f"  Result: * SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: Not significant (p >= 0.05)")
    print()

    # Verdict
    print("="*80)
    print("VERDICT")
    print("="*80 + "\n")

    if improvement_mean > 0 and p_value < 0.05:
        print("✓ SVD CURVATURE FIX VALIDATED")
        print(f"  Geometry features provide {improvement_pct_mean:+.1f}% improvement over baseline")
        print(f"  Win rate: {win_rate:.0f}%")
        print(f"  Statistically significant: p = {p_value:.2e}")
    elif improvement_mean > 0:
        print("⚠️  IMPROVEMENT OBSERVED BUT NOT SIGNIFICANT")
        print(f"  Mean improvement: {improvement_pct_mean:+.1f}%")
        print(f"  But p-value = {p_value:.2f} (not significant)")
    else:
        print("✗ REGRESSION DETECTED")
        print(f"  Geometry features WORSE than baseline by {abs(improvement_pct_mean):.1f}%")
        print(f"  SVD curvature fix may have broken something")

    print()

    # Save results
    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"svd_curvature_validation_{timestamp}.json"

    report = {
        'timestamp': timestamp,
        'n_trials': n_trials,
        'baseline_r2_mean': float(r2_baseline_mean),
        'baseline_r2_std': float(r2_baseline_std),
        'geometry_r2_mean': float(r2_geometry_mean),
        'geometry_r2_std': float(r2_geometry_std),
        'improvement_mean': float(improvement_mean),
        'improvement_std': float(improvement_std),
        'improvement_pct_mean': float(improvement_pct_mean),
        'win_rate': float(win_rate),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'trials': results
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Validation report saved: {output_path}")
    print()


if __name__ == "__main__":
    main()
