"""
H₄ Polytope Pilot Test - Quick 10-Run Validation

Fast test to get initial answer: Does H₄ beat k-NN?
Uses 2 seeds × 5 runs = 10 total evaluations

If pilot shows strong signal → run full 50-run validation
If pilot shows no signal → H₄ hypothesis likely false
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
from scipy.stats import ttest_rel
import json
import sys

# Ensure output isn't buffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================================
# Feature Computation (same as main script)
# ============================================================================

def compute_polytope_features(embeddings, k=50, seed=42):
    np.random.seed(seed)
    N = len(embeddings)

    pca = PCA(n_components=4, random_state=seed)
    embeddings_4d = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()

    kmeans = KMeans(n_clusters=120, random_state=seed, n_init=10, max_iter=300)
    cell_assignments = kmeans.fit_predict(embeddings_4d)
    cell_centers = kmeans.cluster_centers_

    features = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        point_4d = embeddings_4d[i]
        assigned_cell = cell_assignments[i]

        features[i, 0] = np.linalg.norm(point_4d - cell_centers[assigned_cell])
        features[i, 1:5] = point_4d

        distances_to_centers = np.linalg.norm(
            cell_centers - point_4d[np.newaxis, :], axis=1
        )
        distances_to_centers[assigned_cell] = np.inf
        features[i, 5] = distances_to_centers.min()
        features[i, 6] = variance_explained

    return features, variance_explained


def compute_knn_features(embeddings, k=50, seed=42):
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


def run_single_evaluation(seed, run_id, embeddings, boundary_distances, device):
    np.random.seed(seed)
    training_seed = seed * 1000 + run_id
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    # Compute features
    knn_features = compute_knn_features(embeddings, k=50, seed=seed)
    polytope_features, variance_explained = compute_polytope_features(embeddings, k=50, seed=seed)

    # Split data
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx_inner, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    X_train = embeddings[train_idx_inner]
    y_train = boundary_distances[train_idx_inner]
    X_val = embeddings[val_idx]
    y_val = boundary_distances[val_idx]
    X_test = embeddings[test_idx]
    y_test = boundary_distances[test_idx]

    knn_train = knn_features[train_idx_inner]
    knn_val = knn_features[val_idx]
    knn_test = knn_features[test_idx]

    polytope_train = polytope_features[train_idx_inner]
    polytope_val = polytope_features[val_idx]
    polytope_test = polytope_features[test_idx]

    X_train_knn = np.concatenate([X_train, knn_train], axis=1)
    X_val_knn = np.concatenate([X_val, knn_val], axis=1)
    X_test_knn = np.concatenate([X_test, knn_test], axis=1)

    X_train_polytope = np.concatenate([X_train, polytope_train], axis=1)
    X_val_polytope = np.concatenate([X_val, polytope_val], axis=1)
    X_test_polytope = np.concatenate([X_test, polytope_test], axis=1)

    # Train models
    model_baseline = BaselineModel()
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device)

    model_knn = AugmentedModel()
    model_knn = train_model(model_knn, X_train_knn, y_train, X_val_knn, y_val, device)

    model_polytope = AugmentedModel()
    model_polytope = train_model(model_polytope, X_train_polytope, y_train,
                                   X_val_polytope, y_val, device)

    # Evaluate on borderline
    borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)

    if borderline_mask.sum() < 10:
        return None

    model_baseline.eval()
    model_knn.eval()
    model_polytope.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_baseline = model_baseline(X_test_t).cpu().numpy()

        X_test_knn_t = torch.from_numpy(X_test_knn).float().to(device)
        preds_knn = model_knn(X_test_knn_t).cpu().numpy()

        X_test_polytope_t = torch.from_numpy(X_test_polytope).float().to(device)
        preds_polytope = model_polytope(X_test_polytope_t).cpu().numpy()

    r2_baseline = r2_score(y_test[borderline_mask], preds_baseline[borderline_mask])
    r2_knn = r2_score(y_test[borderline_mask], preds_knn[borderline_mask])
    r2_polytope = r2_score(y_test[borderline_mask], preds_polytope[borderline_mask])

    return {
        'r2_baseline': float(r2_baseline),
        'r2_knn': float(r2_knn),
        'r2_polytope': float(r2_polytope),
        'delta_knn': float(r2_knn - r2_baseline),
        'delta_polytope': float(r2_polytope - r2_baseline),
        'delta_polytope_vs_knn': float(r2_polytope - r2_knn),
        'variance_explained': float(variance_explained),
        'n_borderline': int(borderline_mask.sum())
    }


def main():
    print("="*80, flush=True)
    print("H₄ POLYTOPE PILOT TEST (10 runs)", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(flush=True)

    # Load data
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    print(f"Data loaded: N={len(embeddings)}, D={embeddings.shape[1]}", flush=True)
    print(flush=True)

    # Quick test: 2 seeds × 5 runs
    seeds = [42, 100]
    runs_per_seed = 5

    all_results = []

    print("Running pilot (10 evaluations)...", flush=True)
    print(flush=True)

    for seed in seeds:
        print(f"Seed {seed}:", flush=True)

        for run_id in range(runs_per_seed):
            result = run_single_evaluation(seed, run_id, embeddings, boundary_distances, device)

            if result:
                all_results.append(result)
                print(f"  Run {run_id+1}: "
                      f"base={result['r2_baseline']:.3f}, "
                      f"knn={result['r2_knn']:.3f} ({result['delta_knn']:+.3f}), "
                      f"poly={result['r2_polytope']:.3f} ({result['delta_polytope']:+.3f}), "
                      f"poly-knn={result['delta_polytope_vs_knn']:+.3f}", flush=True)
        print(flush=True)

    # Results
    print("="*80, flush=True)
    print("PILOT RESULTS", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    deltas_knn = [r['delta_knn'] for r in all_results]
    deltas_polytope = [r['delta_polytope'] for r in all_results]
    deltas_poly_vs_knn = [r['delta_polytope_vs_knn'] for r in all_results]

    mean_knn = np.mean(deltas_knn)
    mean_polytope = np.mean(deltas_polytope)
    mean_poly_vs_knn = np.mean(deltas_poly_vs_knn)
    std_poly_vs_knn = np.std(deltas_poly_vs_knn, ddof=1)

    variance_explained = np.mean([r['variance_explained'] for r in all_results])

    print(f"k-NN vs Baseline:     {mean_knn:+.1%}", flush=True)
    print(f"Polytope vs Baseline: {mean_polytope:+.1%}", flush=True)
    print(f"Polytope vs k-NN:     {mean_poly_vs_knn:+.1%} ± {std_poly_vs_knn:.1%}", flush=True)
    print(flush=True)
    print(f"4D PCA variance explained: {variance_explained:.1%}", flush=True)
    print(flush=True)

    # Quick verdict
    print("="*80, flush=True)
    print("PILOT VERDICT", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    if mean_poly_vs_knn > 0.02:
        print(f"✓ PROMISING: Polytope beats k-NN by {mean_poly_vs_knn:+.1%}", flush=True)
        print(f"  → Recommend full 50-run validation", flush=True)
        verdict = "PROMISING"
    elif abs(mean_poly_vs_knn) < 0.01:
        print(f"✗ NO SIGNAL: Polytope ≈ k-NN (diff={mean_poly_vs_knn:+.1%})", flush=True)
        print(f"  → H₄ likely adds no value", flush=True)
        print(f"  → 4D projection loses {1-variance_explained:.1%} of variance", flush=True)
        verdict = "NO_SIGNAL"
    else:
        print(f"✗ POLYTOPE WORSE: Polytope loses {mean_poly_vs_knn:+.1%} vs k-NN", flush=True)
        print(f"  → 4D projection loses critical information", flush=True)
        verdict = "WORSE"

    print(flush=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"h4_pilot_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'verdict': verdict,
        'n_runs': len(all_results),
        'polytope_vs_knn': {
            'mean': float(mean_poly_vs_knn),
            'std': float(std_poly_vs_knn)
        },
        'variance_explained': float(variance_explained),
        'all_results': all_results
    }

    with open(output_dir / "pilot_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {output_dir}", flush=True)

    return summary


if __name__ == "__main__":
    main()
