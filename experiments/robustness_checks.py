"""
Robustness Checks: Varying k-NN and Borderline Thresholds

Tests whether the +6.4% geometry gain is robust to:
1. Different k-NN values (k=25, 50, 100)
2. Different borderline thresholds (±0.3, ±0.5, ±0.7)

Runs 3 training runs per configuration (faster than full 10-run validation).
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
# Geometry Features with Variable k
# ============================================================================

def compute_geometry_features(embeddings, k=50, seed=42):
    """Compute geometry features with specified k."""
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

def run_evaluation(seed, run_id, embeddings, boundary_distances, device, k=50, threshold=0.5):
    """Run single evaluation with given parameters."""
    # Set seeds
    np.random.seed(seed)
    training_seed = seed * 1000 + run_id
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    # Geometry features
    geo_features = compute_geometry_features(embeddings, k=k, seed=seed)

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

    X_train_geo = np.concatenate([X_train, geo_train], axis=1)
    X_val_geo = np.concatenate([X_val, geo_val], axis=1)
    X_test_geo = np.concatenate([X_test, geo_test], axis=1)

    # Train models
    model_baseline = BaselineModel()
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device)

    model_geo = AugmentedModel()
    model_geo = train_model(model_geo, X_train_geo, y_train, X_val_geo, y_val, device)

    # Predictions
    model_baseline.eval()
    model_geo.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_baseline = model_baseline(X_test_t).cpu().numpy()

        X_test_geo_t = torch.from_numpy(X_test_geo).float().to(device)
        preds_geo = model_geo(X_test_geo_t).cpu().numpy()

    # Evaluate borderline with given threshold
    borderline_mask = (y_test >= -threshold) & (y_test <= threshold)

    if borderline_mask.sum() < 5:
        return None

    r2_baseline = r2_score(y_test[borderline_mask], preds_baseline[borderline_mask])
    r2_geo = r2_score(y_test[borderline_mask], preds_geo[borderline_mask])

    return {
        'delta_r2': float(r2_geo - r2_baseline),
        'n_borderline': int(borderline_mask.sum())
    }


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("ROBUSTNESS CHECKS")
    print("="*80)
    print(f"\nDevice: {device}")
    print("Testing sensitivity to k-NN and threshold parameters\n")

    # Load data
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    # Test configurations
    k_values = [25, 50, 100]
    threshold_values = [0.3, 0.5, 0.7]
    seeds = [42, 100, 200]  # 3 seeds for robustness
    runs_per_seed = 3  # 3 runs per seed (faster than 10)

    all_results = []

    # Test k-NN variations
    print("="*80)
    print("TEST 1: k-NN Sensitivity (threshold=0.5)")
    print("="*80 + "\n")

    for k in k_values:
        print(f"k={k}:")
        config_deltas = []

        for seed in seeds:
            for run_id in range(runs_per_seed):
                result = run_evaluation(seed, run_id, embeddings, boundary_distances,
                                       device, k=k, threshold=0.5)
                if result:
                    config_deltas.append(result['delta_r2'])
                    print(f"  Seed {seed}, Run {run_id+1}: ΔR² = {result['delta_r2']:+.4f}")

        mean_delta = np.mean(config_deltas)
        std_delta = np.std(config_deltas, ddof=1) if len(config_deltas) > 1 else 0.0

        all_results.append({
            'test': 'knn_sensitivity',
            'k': k,
            'threshold': 0.5,
            'mean_delta': float(mean_delta),
            'std_delta': float(std_delta),
            'n_runs': len(config_deltas)
        })

        print(f"  → Mean ± SD: {mean_delta:+.4f} ± {std_delta:.4f}\n")

    # Test threshold variations
    print("="*80)
    print("TEST 2: Threshold Sensitivity (k=50)")
    print("="*80 + "\n")

    for threshold in threshold_values:
        print(f"Threshold=±{threshold}:")
        config_deltas = []

        for seed in seeds:
            for run_id in range(runs_per_seed):
                result = run_evaluation(seed, run_id, embeddings, boundary_distances,
                                       device, k=50, threshold=threshold)
                if result:
                    config_deltas.append(result['delta_r2'])
                    print(f"  Seed {seed}, Run {run_id+1}: ΔR² = {result['delta_r2']:+.4f} (n={result['n_borderline']})")

        mean_delta = np.mean(config_deltas)
        std_delta = np.std(config_deltas, ddof=1) if len(config_deltas) > 1 else 0.0

        all_results.append({
            'test': 'threshold_sensitivity',
            'k': 50,
            'threshold': threshold,
            'mean_delta': float(mean_delta),
            'std_delta': float(std_delta),
            'n_runs': len(config_deltas)
        })

        print(f"  → Mean ± SD: {mean_delta:+.4f} ± {std_delta:.4f}\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print("k-NN Sensitivity (threshold=±0.5):")
    print(f"{'k':<6} {'Mean ΔR²':<12} {'Std Dev':<10} {'Robust?':<10}")
    print("-" * 50)
    for result in all_results:
        if result['test'] == 'knn_sensitivity':
            robust = "✓" if result['mean_delta'] > 0.04 else "⚠️"
            print(f"{result['k']:<6} {result['mean_delta']:+.4f}       {result['std_delta']:.4f}     {robust}")

    print("\nThreshold Sensitivity (k=50):")
    print(f"{'Threshold':<12} {'Mean ΔR²':<12} {'Std Dev':<10} {'Robust?':<10}")
    print("-" * 50)
    for result in all_results:
        if result['test'] == 'threshold_sensitivity':
            robust = "✓" if result['mean_delta'] > 0.04 else "⚠️"
            print(f"±{result['threshold']:<10} {result['mean_delta']:+.4f}       {result['std_delta']:.4f}     {robust}")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80 + "\n")

    knn_results = [r for r in all_results if r['test'] == 'knn_sensitivity']
    threshold_results = [r for r in all_results if r['test'] == 'threshold_sensitivity']

    knn_robust = all(r['mean_delta'] > 0.04 for r in knn_results)
    threshold_robust = all(r['mean_delta'] > 0.04 for r in threshold_results)

    if knn_robust and threshold_robust:
        print("✓ ROBUST: Geometry gain holds across all tested configurations")
        print(f"  → k-NN: All k values show >4% improvement")
        print(f"  → Threshold: All thresholds show >4% improvement")
        verdict = "ROBUST"
    elif knn_robust or threshold_robust:
        print("⚠️  PARTIALLY ROBUST: Some configurations show consistent gain")
        if knn_robust:
            print(f"  → k-NN: Robust across k values")
        else:
            print(f"  → k-NN: Sensitive to k choice")
        if threshold_robust:
            print(f"  → Threshold: Robust across thresholds")
        else:
            print(f"  → Threshold: Sensitive to threshold choice")
        verdict = "PARTIAL"
    else:
        print("✗ NOT ROBUST: Results vary significantly across configurations")
        verdict = "SENSITIVE"

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"robustness_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'verdict': verdict,
        'k_values_tested': k_values,
        'threshold_values_tested': threshold_values,
        'seeds_used': seeds,
        'runs_per_config': runs_per_seed,
        'results': all_results
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    main()
