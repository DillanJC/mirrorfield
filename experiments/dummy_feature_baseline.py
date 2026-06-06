"""
Dummy Feature Baseline Test

Verifies that geometry gain isn't just from increased dimensionality.

Compares three configurations:
- Baseline only (256-D)
- Baseline + 7 random features (263-D) ← Control
- Baseline + 7 geometry features (263-D) ← Treatment

If geometry beats random by >5%, the signal is real (not just dimensionality).
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
import json

# ============================================================================
# Models
# ============================================================================

class BaselineModel(nn.Module):
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
    def __init__(self, input_dim=256, aug_dim=7, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + aug_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(1)


# ============================================================================
# Features
# ============================================================================

def compute_geometry_features(embeddings, k=50, seed=42):
    """Compute real geometry features."""
    np.random.seed(seed)
    N = len(embeddings)

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
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


def compute_dummy_features(embeddings, seed=42):
    """Compute random dummy features (same dimensionality as geometry)."""
    np.random.seed(seed)
    N = len(embeddings)

    # Random features with similar scale to geometry features
    # Use mixture of distributions to mimic geometry feature types
    features = np.zeros((N, 7), dtype=np.float32)

    features[:, 0] = np.random.normal(0.5, 0.1, N)  # Like mean distance
    features[:, 1] = np.random.normal(0.1, 0.05, N)  # Like std distance
    features[:, 2] = np.random.normal(0.3, 0.1, N)  # Like min distance
    features[:, 3] = np.random.normal(0.8, 0.2, N)  # Like max distance
    features[:, 4] = np.random.uniform(0, 1, N)  # Like variance ratio
    features[:, 5] = np.random.normal(0.2, 0.1, N)  # Like stability
    features[:, 6] = np.random.normal(0.4, 0.1, N)  # Like NN distance

    return features


# ============================================================================
# Training
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
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
# Single Evaluation
# ============================================================================

def run_evaluation(seed, run_id, embeddings, boundary_distances, device):
    """Run evaluation comparing baseline, dummy, and geometry."""
    # Set seeds
    np.random.seed(seed)
    training_seed = seed * 1000 + run_id
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    # Compute features
    geo_features = compute_geometry_features(embeddings, k=50, seed=seed)
    dummy_features = compute_dummy_features(embeddings, seed=seed + run_id)  # Different per run

    # Split data
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx_inner, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

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

    dummy_train = dummy_features[train_idx_inner]
    dummy_val = dummy_features[val_idx]
    dummy_test = dummy_features[test_idx]

    # Augmented inputs
    X_train_geo = np.concatenate([X_train, geo_train], axis=1)
    X_val_geo = np.concatenate([X_val, geo_val], axis=1)
    X_test_geo = np.concatenate([X_test, geo_test], axis=1)

    X_train_dummy = np.concatenate([X_train, dummy_train], axis=1)
    X_val_dummy = np.concatenate([X_val, dummy_val], axis=1)
    X_test_dummy = np.concatenate([X_test, dummy_test], axis=1)

    # Train 3 models
    model_baseline = BaselineModel()
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device)

    model_dummy = AugmentedModel()
    model_dummy = train_model(model_dummy, X_train_dummy, y_train, X_val_dummy, y_val, device)

    model_geo = AugmentedModel()
    model_geo = train_model(model_geo, X_train_geo, y_train, X_val_geo, y_val, device)

    # Predictions
    model_baseline.eval()
    model_dummy.eval()
    model_geo.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_baseline = model_baseline(X_test_t).cpu().numpy()

        X_test_dummy_t = torch.from_numpy(X_test_dummy).float().to(device)
        preds_dummy = model_dummy(X_test_dummy_t).cpu().numpy()

        X_test_geo_t = torch.from_numpy(X_test_geo).float().to(device)
        preds_geo = model_geo(X_test_geo_t).cpu().numpy()

    # Evaluate borderline
    borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)

    r2_baseline = r2_score(y_test[borderline_mask], preds_baseline[borderline_mask])
    r2_dummy = r2_score(y_test[borderline_mask], preds_dummy[borderline_mask])
    r2_geo = r2_score(y_test[borderline_mask], preds_geo[borderline_mask])

    return {
        'r2_baseline': float(r2_baseline),
        'r2_dummy': float(r2_dummy),
        'r2_geometry': float(r2_geo),
        'delta_dummy': float(r2_dummy - r2_baseline),
        'delta_geometry': float(r2_geo - r2_baseline),
        'delta_geo_vs_dummy': float(r2_geo - r2_dummy)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("DUMMY FEATURE BASELINE TEST")
    print("="*80)
    print(f"\nDevice: {device}")
    print("Testing if geometry beats random features\n")

    # Load data
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    # Run multiple seeds and runs
    seeds = [42, 100, 200]
    runs_per_seed = 5

    all_results = []

    print("Running experiments...\n")

    for seed in seeds:
        print(f"Seed {seed}:")

        for run_id in range(runs_per_seed):
            result = run_evaluation(seed, run_id, embeddings, boundary_distances, device)
            all_results.append(result)

            print(f"  Run {run_id+1}: Baseline={result['r2_baseline']:.4f}, "
                  f"Dummy={result['r2_dummy']:.4f} ({result['delta_dummy']:+.4f}), "
                  f"Geo={result['r2_geometry']:.4f} ({result['delta_geometry']:+.4f})")

        print()

    # Aggregate statistics
    print("="*80)
    print("AGGREGATE STATISTICS")
    print("="*80 + "\n")

    deltas_dummy = [r['delta_dummy'] for r in all_results]
    deltas_geo = [r['delta_geometry'] for r in all_results]
    deltas_geo_vs_dummy = [r['delta_geo_vs_dummy'] for r in all_results]

    mean_dummy = np.mean(deltas_dummy)
    std_dummy = np.std(deltas_dummy, ddof=1)

    mean_geo = np.mean(deltas_geo)
    std_geo = np.std(deltas_geo, ddof=1)

    mean_geo_vs_dummy = np.mean(deltas_geo_vs_dummy)
    std_geo_vs_dummy = np.std(deltas_geo_vs_dummy, ddof=1)

    print(f"ΔR² (vs Baseline):")
    print(f"  Dummy features:    {mean_dummy:+.4f} ± {std_dummy:.4f}")
    print(f"  Geometry features: {mean_geo:+.4f} ± {std_geo:.4f}")
    print()
    print(f"ΔR² (Geometry vs Dummy):")
    print(f"  Mean:              {mean_geo_vs_dummy:+.4f} ± {std_geo_vs_dummy:.4f}")

    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(deltas_geo, deltas_dummy)

    print()
    print(f"Paired t-test (Geometry vs Dummy):")
    print(f"  t-statistic:       {t_stat:.4f}")
    print(f"  p-value:           {p_value:.6f}")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80 + "\n")

    threshold_pass = 0.05

    if mean_geo_vs_dummy > threshold_pass and p_value < 0.05:
        print(f"✓ PASS: Geometry significantly beats dummy features")
        print(f"  → Geometry advantage: {mean_geo_vs_dummy:+.1%}")
        print(f"  → p-value: {p_value:.4f} (significant at α=0.05)")
        print(f"  → Geometry signal is REAL, not just dimensionality")
        verdict = "PASS"
    elif mean_geo_vs_dummy > 0 and p_value < 0.05:
        print(f"⚠️  MARGINAL: Geometry beats dummy but effect is small")
        print(f"  → Geometry advantage: {mean_geo_vs_dummy:+.1%}")
        print(f"  → p-value: {p_value:.4f}")
        print(f"  → Signal exists but weak")
        verdict = "MARGINAL"
    elif mean_geo_vs_dummy > 0:
        print(f"⚠️  WEAK: Geometry beats dummy but not significantly")
        print(f"  → Geometry advantage: {mean_geo_vs_dummy:+.1%}")
        print(f"  → p-value: {p_value:.4f} (not significant)")
        print(f"  → Could be noise")
        verdict = "WEAK"
    else:
        print(f"✗ FAIL: Geometry doesn't beat dummy features")
        print(f"  → Difference: {mean_geo_vs_dummy:+.1%}")
        print(f"  → Geometry gain is likely spurious (dimensionality artifact)")
        verdict = "FAIL"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"dummy_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'verdict': verdict,
        'n_runs': len(all_results),
        'statistics': {
            'dummy_vs_baseline': {
                'mean': float(mean_dummy),
                'std': float(std_dummy)
            },
            'geometry_vs_baseline': {
                'mean': float(mean_geo),
                'std': float(std_geo)
            },
            'geometry_vs_dummy': {
                'mean': float(mean_geo_vs_dummy),
                'std': float(std_geo_vs_dummy),
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        },
        'all_results': all_results
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    main()
