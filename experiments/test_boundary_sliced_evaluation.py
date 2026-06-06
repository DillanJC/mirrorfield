"""Boundary-Sliced Evaluation: Does Geometry Help Where It Matters?

Tests whether geometric features (120-cell or native 256-D) add value
specifically in borderline regions where baseline confidence is low.

Three models:
- Model A: Baseline (256-D embeddings only)
- Model B: Baseline + 120-cell soft assignments (376-D total)
- Model C: Baseline + native 256-D geometry features (~263-D total)

Three regions (stratified by boundary_distance):
- Toxic: d < -0.5 (deep in toxic region)
- Borderline: -0.5 ≤ d ≤ 0.5 (critical safety boundary) ⭐
- Safe: d > 0.5 (clearly safe region)

Key question: Does geometry show concentrated gain in borderline?

Pass condition: ΔR² > 0.05 (5%) in borderline region
Fail condition: ΔR² < 0.02 (2%) everywhere

Usage:
    python experiments/test_boundary_sliced_evaluation.py --seeds 5
    python experiments/test_boundary_sliced_evaluation.py --seeds 5 --verbose
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import sys
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phase_d_data(run_dir):
    """Load Phase D embeddings, labels, and boundary distances."""
    run_path = Path(run_dir)
    embeddings = np.load(run_path / "embeddings.npy")
    labels = np.load(run_path / "labels.npy")
    boundary_distances = np.load(run_path / "boundary_distances.npy")
    return embeddings, labels, boundary_distances


def compute_native_geometry_features(embeddings, k=50, seed=42):
    """Compute geometry features in native 256-D space.

    Features:
    - k-NN distance statistics (mean, std, min, max)
    - Local variance ratio (anisotropy)
    - Neighborhood stability (Jaccard similarity)
    - Distance to nearest different class (boundary proximity)
    """
    N = len(embeddings)

    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Remove self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    features = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        # k-NN distance statistics
        features[i, 0] = distances[i].mean()  # mean_knn_distance
        features[i, 1] = distances[i].std()   # std_knn_distance
        features[i, 2] = distances[i].min()   # min_knn_distance
        features[i, 3] = distances[i].max()   # max_knn_distance

        # Local variance ratio (anisotropy)
        neighbors = embeddings[indices[i]]
        centered = neighbors - embeddings[i]
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[-1] > 1e-10:
            features[i, 4] = eigenvalues[0] / eigenvalues[-1]  # smallest/largest
        else:
            features[i, 4] = 0.0

        # Neighborhood stability (Jaccard at k=50)
        # This is placeholder - would need perturbation for true stability
        # For now, use distance variance as proxy
        features[i, 5] = distances[i].std() / (distances[i].mean() + 1e-6)

        # Distance to nearest neighbor (simple boundary proximity)
        features[i, 6] = distances[i, 0]

    return features


def stratify_by_boundary(boundary_distances, threshold_low=-0.5, threshold_high=0.5):
    """Stratify data into toxic/borderline/safe regions."""
    toxic_mask = boundary_distances < threshold_low
    borderline_mask = (boundary_distances >= threshold_low) & (boundary_distances <= threshold_high)
    safe_mask = boundary_distances > threshold_high

    return {
        'toxic': toxic_mask,
        'borderline': borderline_mask,
        'safe': safe_mask,
    }


def project_to_4d(embeddings, seed=42):
    """Project embeddings to 4D via PCA."""
    pca = PCA(n_components=4, random_state=seed)
    projected = pca.fit_transform(embeddings)
    return projected.astype(np.float32), pca


def assign_to_120cell(embeddings_4d, seed=42):
    """Assign 4D points to 120-cell via k-means."""
    kmeans = KMeans(n_clusters=120, random_state=seed, n_init=10)
    kmeans.fit(embeddings_4d)

    distances = kmeans.transform(embeddings_4d)
    soft_assignments = 1.0 / (distances + 1e-6)
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)

    return soft_assignments, kmeans


class BaselineModel(nn.Module):
    """Model A: Baseline (256-D embeddings only)."""

    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out.squeeze(1)


class AugmentedModel(nn.Module):
    """Model B/C: Baseline + additional features."""

    def __init__(self, input_dim=256, aug_dim=120, hidden_dim=64):
        super().__init__()
        total_dim = input_dim + aug_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out.squeeze(1)


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device="cpu"):
    """Train a model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = F.mse_loss(predictions, y_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = F.mse_loss(val_predictions, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model


def evaluate_model_on_slice(model, X, y, mask, device="cpu"):
    """Evaluate model on a specific slice of data."""
    if mask.sum() == 0:
        return {"r2": np.nan, "rmse": np.nan, "n_samples": 0}

    X_slice = X[mask]
    y_slice = y[mask]

    model = model.to(device)
    model.eval()

    X_t = torch.from_numpy(X_slice).float().to(device)

    with torch.no_grad():
        predictions = model(X_t).cpu().numpy()

    r2 = r2_score(y_slice, predictions)
    rmse = np.sqrt(mean_squared_error(y_slice, predictions))

    return {"r2": r2, "rmse": rmse, "n_samples": len(y_slice)}


def run_single_seed(seed, embeddings, boundary_distances, device="cpu", verbose=False):
    """Run boundary-sliced evaluation for a single seed."""

    # Set ALL random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}\n")

    # Compute auxiliary features
    print(f"  Computing 4D projection...", end=" ", flush=True)
    embeddings_4d, pca = project_to_4d(embeddings, seed=seed)
    print("✓")

    print(f"  Computing 120-cell soft assignments...", end=" ", flush=True)
    soft_assignments_120, kmeans = assign_to_120cell(embeddings_4d, seed=seed)
    print("✓")

    print(f"  Computing native 256-D geometry features...", end=" ", flush=True)
    geometry_features = compute_native_geometry_features(embeddings, k=50, seed=seed)
    print("✓")

    # Prepare augmented feature sets
    X_baseline = embeddings
    X_with_120cell = np.concatenate([embeddings, soft_assignments_120], axis=1)
    X_with_geometry = np.concatenate([embeddings, geometry_features], axis=1)

    # Data split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    # Stratify test set
    test_boundary_distances = boundary_distances[test_idx]
    strata = stratify_by_boundary(test_boundary_distances)

    if verbose:
        print(f"\n  Test set stratification:")
        print(f"    Toxic:      {strata['toxic'].sum():3d} samples (d < -0.5)")
        print(f"    Borderline: {strata['borderline'].sum():3d} samples (-0.5 ≤ d ≤ 0.5)")
        print(f"    Safe:       {strata['safe'].sum():3d} samples (d > 0.5)")

    # Train three models
    models = {}

    # Model A: Baseline
    print(f"\n  Training Model A (Baseline)...", end=" ", flush=True)
    model_a = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64)
    model_a = train_model(
        model_a,
        X_baseline[train_idx], boundary_distances[train_idx],
        X_baseline[val_idx], boundary_distances[val_idx],
        device=device
    )
    models['baseline'] = (model_a, X_baseline)
    print("✓")

    # Model B: Baseline + 120-cell
    print(f"  Training Model B (Baseline + 120-cell)...", end=" ", flush=True)
    model_b = AugmentedModel(input_dim=embeddings.shape[1], aug_dim=120, hidden_dim=64)
    model_b = train_model(
        model_b,
        X_with_120cell[train_idx], boundary_distances[train_idx],
        X_with_120cell[val_idx], boundary_distances[val_idx],
        device=device
    )
    models['baseline_plus_120cell'] = (model_b, X_with_120cell)
    print("✓")

    # Model C: Baseline + native geometry
    print(f"  Training Model C (Baseline + Native Geometry)...", end=" ", flush=True)
    model_c = AugmentedModel(input_dim=embeddings.shape[1], aug_dim=7, hidden_dim=64)
    model_c = train_model(
        model_c,
        X_with_geometry[train_idx], boundary_distances[train_idx],
        X_with_geometry[val_idx], boundary_distances[val_idx],
        device=device
    )
    models['baseline_plus_geometry'] = (model_c, X_with_geometry)
    print("✓")

    # Evaluate on each stratum
    results = {}

    for model_name, (model, X_full) in models.items():
        results[model_name] = {}

        for region_name, mask in strata.items():
            result = evaluate_model_on_slice(
                model, X_full[test_idx], boundary_distances[test_idx], mask, device=device
            )
            results[model_name][region_name] = result

    # Compute deltas vs baseline
    for model_name in ['baseline_plus_120cell', 'baseline_plus_geometry']:
        for region_name in strata.keys():
            baseline_r2 = results['baseline'][region_name]['r2']
            model_r2 = results[model_name][region_name]['r2']

            if not np.isnan(baseline_r2) and not np.isnan(model_r2):
                results[model_name][region_name]['delta_r2'] = model_r2 - baseline_r2
            else:
                results[model_name][region_name]['delta_r2'] = np.nan

    if verbose:
        print(f"\n  Results by region:")
        print(f"  {'Region':<12} {'Model':<30} {'R²':>8} {'RMSE':>8} {'ΔR²':>8}")
        print(f"  {'-'*72}")

        for region_name in ['toxic', 'borderline', 'safe']:
            for model_name in ['baseline', 'baseline_plus_120cell', 'baseline_plus_geometry']:
                res = results[model_name][region_name]
                r2 = res['r2']
                rmse = res['rmse']
                delta = res.get('delta_r2', 0.0)

                if not np.isnan(r2):
                    if model_name == 'baseline':
                        print(f"  {region_name:<12} {model_name:<30} {r2:>8.4f} {rmse:>8.4f} {'—':>8}")
                    else:
                        print(f"  {region_name:<12} {model_name:<30} {r2:>8.4f} {rmse:>8.4f} {delta:>+8.4f}")

    return {
        'seed': seed,
        'results': results,
        'strata_counts': {k: int(v.sum()) for k, v in strata.items()}
    }


def main():
    parser = argparse.ArgumentParser(description='Boundary-sliced evaluation: Does geometry help where it matters?')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to test')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--embeddings', type=str, default='runs/openai_3_large_test_20251231_024532',
                        help='Path to embeddings directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"boundary_sliced_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("BOUNDARY-SLICED EVALUATION")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Number of seeds: {args.seeds}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load data
    print("Loading Phase D data...")
    embeddings, labels, boundary_distances = load_phase_d_data(args.embeddings)
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)")
    print(f"  Boundary distances: [{boundary_distances.min():.3f}, {boundary_distances.max():.3f}]")

    # Define seeds
    seed_list = [17, 42, 100, 200, 333, 500, 666, 777, 888, 999][:args.seeds]

    print(f"\nTesting seeds: {seed_list}")
    print(f"\nRegion definitions:")
    print(f"  Toxic:      d < -0.5")
    print(f"  Borderline: -0.5 ≤ d ≤ 0.5 ⭐")
    print(f"  Safe:       d > 0.5")

    print(f"\n{'='*80}\n")

    # Run experiments
    all_results = []
    for seed in seed_list:
        result = run_single_seed(seed, embeddings, boundary_distances, device, verbose=args.verbose)
        all_results.append(result)

        if not args.verbose:
            print(f"Seed {seed:3d}: Borderline ΔR² (120-cell)={result['results']['baseline_plus_120cell']['borderline'].get('delta_r2', np.nan):+.4f}, "
                  f"(Geometry)={result['results']['baseline_plus_geometry']['borderline'].get('delta_r2', np.nan):+.4f}")

    # Aggregate analysis
    print(f"\n{'='*80}")
    print("AGGREGATE ANALYSIS ACROSS SEEDS")
    print(f"{'='*80}\n")

    # Compute mean ΔR² for each model and region
    aggregate = {}

    for model_name in ['baseline_plus_120cell', 'baseline_plus_geometry']:
        aggregate[model_name] = {}

        for region_name in ['toxic', 'borderline', 'safe']:
            deltas = []
            for result in all_results:
                delta = result['results'][model_name][region_name].get('delta_r2', np.nan)
                if not np.isnan(delta):
                    deltas.append(delta)

            if deltas:
                aggregate[model_name][region_name] = {
                    'mean_delta': np.mean(deltas),
                    'std_delta': np.std(deltas),
                    'min_delta': np.min(deltas),
                    'max_delta': np.max(deltas),
                }
            else:
                aggregate[model_name][region_name] = {
                    'mean_delta': np.nan,
                    'std_delta': np.nan,
                    'min_delta': np.nan,
                    'max_delta': np.nan,
                }

    # Print summary table
    print(f"Mean ΔR² vs Baseline (across {args.seeds} seeds):\n")
    print(f"{'Region':<12} {'Model':<30} {'Mean ΔR²':>12} {'Std':>8} {'Range':>16}")
    print(f"{'-'*80}")

    for region_name in ['toxic', 'borderline', 'safe']:
        for model_name in ['baseline_plus_120cell', 'baseline_plus_geometry']:
            agg = aggregate[model_name][region_name]
            mean = agg['mean_delta']
            std = agg['std_delta']
            min_val = agg['min_delta']
            max_val = agg['max_delta']

            model_label = "120-cell" if '120cell' in model_name else "Native Geo"

            if not np.isnan(mean):
                range_str = f"[{min_val:+.3f}, {max_val:+.3f}]"
                print(f"{region_name:<12} {model_label:<30} {mean:>+12.4f} {std:>8.4f} {range_str:>16}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT: Does Geometry Help in Borderline?")
    print(f"{'='*80}\n")

    borderline_120_mean = aggregate['baseline_plus_120cell']['borderline']['mean_delta']
    borderline_geo_mean = aggregate['baseline_plus_geometry']['borderline']['mean_delta']

    print(f"Borderline region (where safety matters):")
    print(f"  Baseline + 120-cell:      ΔR² = {borderline_120_mean:+.4f}")
    print(f"  Baseline + Native Geo:    ΔR² = {borderline_geo_mean:+.4f}")

    threshold_pass = 0.05
    threshold_fail = 0.02

    best_delta = max(borderline_120_mean, borderline_geo_mean)

    if best_delta > threshold_pass:
        print(f"\n✓ PASS: Geometry shows significant gain in borderline (ΔR² > {threshold_pass})")
        print(f"  → Geometry adds value where it matters (safety-critical boundaries)")
        print(f"  → Ship: 'Geometric Features for Borderline Case Resolution'")
        verdict = "PASS"
    elif best_delta > threshold_fail:
        print(f"\n⚠️  MARGINAL: Geometry shows small gain in borderline ({threshold_fail} < ΔR² < {threshold_pass})")
        print(f"  → Geometry provides some value, but not decisive")
        print(f"  → Consider: Hybrid approach or further refinement")
        verdict = "MARGINAL"
    else:
        print(f"\n✗ FAIL: Geometry doesn't help in borderline (ΔR² < {threshold_fail})")
        print(f"  → Geometry doesn't add meaningful value")
        print(f"  → Ship: 'Embedder Diagnostics' (baseline is excellent)")
        verdict = "FAIL"

    # Check if gain is concentrated
    toxic_120_mean = aggregate['baseline_plus_120cell']['toxic']['mean_delta']
    safe_120_mean = aggregate['baseline_plus_120cell']['safe']['mean_delta']

    if best_delta > 0.03 and (best_delta > 2 * toxic_120_mean) and (best_delta > 2 * safe_120_mean):
        print(f"\n✓ CONCENTRATED: Gain is specifically in borderline region")
        print(f"  → Borderline: ΔR² = {best_delta:+.4f}")
        print(f"  → Toxic:      ΔR² = {toxic_120_mean:+.4f}")
        print(f"  → Safe:       ΔR² = {safe_120_mean:+.4f}")
        print(f"  → This is the valuable finding: geometry helps where baseline struggles")

    # Save results
    summary = {
        "timestamp": timestamp,
        "num_seeds": args.seeds,
        "seeds": seed_list,
        "verdict": verdict,
        "threshold_pass": threshold_pass,
        "threshold_fail": threshold_fail,
        "aggregate_deltas": {
            model: {
                region: {k: float(v) if not np.isnan(v) else None
                        for k, v in stats.items()}
                for region, stats in regions.items()
            }
            for model, regions in aggregate.items()
        },
        "all_seed_results": [
            {
                "seed": r['seed'],
                "strata_counts": r['strata_counts'],
                "deltas": {
                    model: {
                        region: float(r['results'][model][region].get('delta_r2', np.nan))
                        for region in ['toxic', 'borderline', 'safe']
                    }
                    for model in ['baseline_plus_120cell', 'baseline_plus_geometry']
                }
            }
            for r in all_results
        ]
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
