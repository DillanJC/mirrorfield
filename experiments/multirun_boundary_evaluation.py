"""
Multi-Run Boundary Evaluation: Accounting for Training Randomness

Runs each seed MULTIPLE times to separate:
- Between-seed variance (data splits)
- Within-seed variance (training randomness)

This properly quantifies uncertainty in geometry's benefit.

Usage:
    python experiments/multirun_boundary_evaluation.py --seeds 5 --runs 10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
import argparse
from scipy.spatial.distance import cdist

# ============================================================================
# Model Definitions
# ============================================================================

class BaselineModel(nn.Module):
    """Model A: Baseline (256-D embeddings only)"""
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(1)


class AugmentedModel(nn.Module):
    """Model B/C: Baseline + additional features"""
    def __init__(self, input_dim=256, aug_dim=7, hidden_dim=64):
        super().__init__()
        total_dim = input_dim + aug_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(1)


# ============================================================================
# Feature Computation
# ============================================================================

def compute_native_geometry_features(embeddings, k=50, seed=42):
    """Compute geometry features in native 256-D space."""
    np.random.seed(seed)
    N = len(embeddings)

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    features = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        features[i, 0] = distances[i].mean()
        features[i, 1] = distances[i].std()
        features[i, 2] = distances[i].min()
        features[i, 3] = distances[i].max()

        neighbors = embeddings[indices[i]]
        centered = neighbors - embeddings[i]
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[-1] > 1e-10:
            features[i, 4] = eigenvalues[0] / eigenvalues[-1]
        else:
            features[i, 4] = 0.0

        features[i, 5] = distances[i].std() / (distances[i].mean() + 1e-6)
        features[i, 6] = distances[i, 0]

    return features


# ============================================================================
# Training
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, device, epochs=100, lr=0.001):
    """Train model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = F.mse_loss(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = F.mse_loss(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    return model


# ============================================================================
# Single Run
# ============================================================================

def run_single_evaluation(seed, run_id, embeddings, boundary_distances, device):
    """
    Run a single evaluation with given data seed and training run ID.

    Args:
        seed: Data split seed (consistent across runs)
        run_id: Training run ID (varies to get different initializations)
        embeddings: Input embeddings
        boundary_distances: Target values
        device: CUDA or CPU

    Returns:
        Dictionary with results for all regions
    """
    # Set seeds for DATA SPLIT (consistent across runs for this seed)
    np.random.seed(seed)

    # Compute geometry features (consistent for this seed)
    geo_features = compute_native_geometry_features(embeddings, k=50, seed=seed)

    # Data split (consistent for this seed)
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx_inner, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    # Set TRAINING seed (varies per run)
    training_seed = seed * 1000 + run_id
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_seed)
        torch.cuda.manual_seed_all(training_seed)
        # Note: Even with these, some non-determinism remains

    # Prepare data
    X_train = embeddings[train_idx_inner]
    y_train = boundary_distances[train_idx_inner]
    X_val = embeddings[val_idx]
    y_val = boundary_distances[val_idx]
    X_test = embeddings[test_idx]
    y_test = boundary_distances[test_idx]

    geo_train = geo_features[train_idx_inner]
    geo_val = geo_features[val_idx]
    geo_test = geo_features[test_idx]

    X_train_geo = np.concatenate([X_train, geo_train], axis=1)
    X_val_geo = np.concatenate([X_val, geo_val], axis=1)
    X_test_geo = np.concatenate([X_test, geo_test], axis=1)

    # Train baseline model
    model_baseline = BaselineModel(input_dim=256, hidden_dim=64)
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device)

    # Train geometry model
    model_geo = AugmentedModel(input_dim=256, aug_dim=7, hidden_dim=64)
    model_geo = train_model(model_geo, X_train_geo, y_train, X_val_geo, y_val, device)

    # Get predictions
    model_baseline.eval()
    model_geo.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_baseline = model_baseline(X_test_t).cpu().numpy()

        X_test_geo_t = torch.from_numpy(X_test_geo).float().to(device)
        preds_geo = model_geo(X_test_geo_t).cpu().numpy()

    # Stratify test set
    toxic_mask = y_test < -0.5
    borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)
    safe_mask = y_test > 0.5

    results = {}

    for region_name, mask in [('toxic', toxic_mask), ('borderline', borderline_mask), ('safe', safe_mask)]:
        if mask.sum() == 0:
            results[region_name] = {
                'n_samples': 0,
                'r2_baseline': np.nan,
                'r2_geometry': np.nan,
                'delta_r2': np.nan
            }
            continue

        y_region = y_test[mask]
        preds_baseline_region = preds_baseline[mask]
        preds_geo_region = preds_geo[mask]

        r2_baseline = r2_score(y_region, preds_baseline_region)
        r2_geo = r2_score(y_region, preds_geo_region)
        delta = r2_geo - r2_baseline

        results[region_name] = {
            'n_samples': int(mask.sum()),
            'r2_baseline': float(r2_baseline),
            'r2_geometry': float(r2_geo),
            'delta_r2': float(delta)
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-run boundary evaluation')
    parser.add_argument('--seeds', type=int, default=5, help='Number of data seeds')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per seed')
    parser.add_argument('--embeddings', type=str, default='runs/openai_3_large_test_20251231_024532',
                        help='Path to embeddings directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("MULTI-RUN BOUNDARY EVALUATION")
    print("Accounting for Training Randomness")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Data seeds: {args.seeds}")
    print(f"Runs per seed: {args.runs}")
    print(f"Total runs: {args.seeds * args.runs}\n")

    # Load data
    print("Loading data...")
    data_path = Path(args.embeddings)
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Boundary distances: {boundary_distances.shape}\n")

    # Run experiments
    seeds = [17, 42, 100, 200, 333][:args.seeds]
    all_results = []

    print("="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80 + "\n")

    total_runs = 0
    for seed in seeds:
        print(f"Seed {seed}:")
        seed_results = []

        for run_id in range(args.runs):
            total_runs += 1
            print(f"  Run {run_id + 1}/{args.runs}...", end=" ", flush=True)

            result = run_single_evaluation(seed, run_id, embeddings, boundary_distances, device)
            seed_results.append(result)

            delta_borderline = result['borderline']['delta_r2']
            print(f"Borderline ΔR² = {delta_borderline:+.4f}")

        all_results.append({
            'seed': int(seed),
            'runs': seed_results
        })

        # Compute statistics for this seed
        borderline_deltas = [r['borderline']['delta_r2'] for r in seed_results]
        mean_delta = np.mean(borderline_deltas)
        std_delta = np.std(borderline_deltas, ddof=1)

        print(f"  → Mean ± SD: {mean_delta:+.4f} ± {std_delta:.4f}")
        print()

    # Aggregate statistics
    print("="*80)
    print("AGGREGATE STATISTICS")
    print("="*80 + "\n")

    # Collect all deltas by region
    all_deltas = {
        'toxic': [],
        'borderline': [],
        'safe': []
    }

    seed_means = {
        'toxic': [],
        'borderline': [],
        'safe': []
    }

    for seed_result in all_results:
        seed = seed_result['seed']
        runs = seed_result['runs']

        for region in ['toxic', 'borderline', 'safe']:
            region_deltas = [r[region]['delta_r2'] for r in runs if not np.isnan(r[region]['delta_r2'])]

            if region_deltas:
                all_deltas[region].extend(region_deltas)
                seed_means[region].append(np.mean(region_deltas))

    # Print variance decomposition
    print("Variance Decomposition:\n")
    print(f"{'Region':<12} {'Overall Mean':<14} {'Between-Seed SD':<18} {'Within-Seed SD':<16}")
    print("-" * 80)

    for region in ['toxic', 'borderline', 'safe']:
        overall_mean = np.mean(all_deltas[region])
        between_seed_std = np.std(seed_means[region], ddof=1) if len(seed_means[region]) > 1 else 0.0

        # Within-seed variance: average of per-seed variances
        within_seed_vars = []
        for seed_result in all_results:
            region_deltas = [r[region]['delta_r2'] for r in seed_result['runs'] if not np.isnan(r[region]['delta_r2'])]
            if len(region_deltas) > 1:
                within_seed_vars.append(np.var(region_deltas, ddof=1))
        within_seed_std = np.sqrt(np.mean(within_seed_vars)) if within_seed_vars else 0.0

        print(f"{region:<12} {overall_mean:+.4f} ({np.std(all_deltas[region], ddof=1):.4f})  "
              f"{between_seed_std:<18.4f}  {within_seed_std:<16.4f}")

    print("\n" + "="*80)
    print("BORDERLINE REGION ANALYSIS (Where Safety Matters)")
    print("="*80 + "\n")

    borderline_all = all_deltas['borderline']
    borderline_mean = np.mean(borderline_all)
    borderline_std = np.std(borderline_all, ddof=1)
    borderline_se = borderline_std / np.sqrt(len(borderline_all))

    # 95% CI using t-distribution
    from scipy import stats
    ci_95 = stats.t.interval(0.95, len(borderline_all) - 1, loc=borderline_mean, scale=borderline_se)

    print(f"Total runs: {len(borderline_all)}")
    print(f"Mean ΔR²: {borderline_mean:+.4f}")
    print(f"Std Dev:  {borderline_std:.4f}")
    print(f"Std Err:  {borderline_se:.4f}")
    print(f"95% CI:   [{ci_95[0]:+.4f}, {ci_95[1]:+.4f}]")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80 + "\n")

    threshold_pass = 0.05

    if ci_95[0] > threshold_pass:
        print(f"✓ STRONG PASS: 95% CI entirely above {threshold_pass:.0%}")
        print(f"  → Geometry provides robust improvement in borderline region")
        print(f"  → Mean gain: {borderline_mean:+.1%} ± {borderline_std:.1%}")
        print(f"  → Ship: 'Geometric Features for Borderline Case Resolution'")
        verdict = "STRONG_PASS"
    elif borderline_mean > threshold_pass and ci_95[0] > 0:
        print(f"✓ PASS: Mean above {threshold_pass:.0%}, CI above zero")
        print(f"  → Geometry provides measurable improvement in borderline region")
        print(f"  → Mean gain: {borderline_mean:+.1%} (95% CI: [{ci_95[0]:+.1%}, {ci_95[1]:+.1%}])")
        print(f"  → Ship with caveat: Effect size modest but consistent")
        verdict = "PASS"
    elif borderline_mean > 0:
        print(f"⚠️  MARGINAL: Mean positive but CI includes zero")
        print(f"  → Geometry shows weak/inconsistent benefit")
        print(f"  → Mean gain: {borderline_mean:+.1%} (95% CI: [{ci_95[0]:+.1%}, {ci_95[1]:+.1%}])")
        print(f"  → Consider: More data or different features")
        verdict = "MARGINAL"
    else:
        print(f"✗ FAIL: Mean not positive")
        print(f"  → Geometry doesn't improve borderline prediction")
        print(f"  → Mean: {borderline_mean:+.1%} (95% CI: [{ci_95[0]:+.1%}, {ci_95[1]:+.1%}])")
        print(f"  → Ship: 'Embedder Diagnostics' (baseline is excellent)")
        verdict = "FAIL"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"multirun_boundary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'num_seeds': args.seeds,
        'runs_per_seed': args.runs,
        'total_runs': total_runs,
        'seeds': seeds,
        'verdict': verdict,
        'borderline_statistics': {
            'mean': float(borderline_mean),
            'std': float(borderline_std),
            'se': float(borderline_se),
            'ci_95_low': float(ci_95[0]),
            'ci_95_high': float(ci_95[1]),
            'n_runs': len(borderline_all)
        },
        'variance_decomposition': {
            region: {
                'overall_mean': float(np.mean(all_deltas[region])),
                'overall_std': float(np.std(all_deltas[region], ddof=1)),
                'between_seed_std': float(np.std(seed_means[region], ddof=1)) if len(seed_means[region]) > 1 else 0.0,
                'within_seed_std': float(within_seed_std)
            }
            for region in ['toxic', 'borderline', 'safe']
        },
        'all_results': all_results
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - summary.json\n")

    return summary


if __name__ == "__main__":
    main()
