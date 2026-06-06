"""
Test H₄ Polytope Hypothesis: Does 120-cell structure beat simple k-NN?

Critical test of Mirrorfield's core geometric hypothesis:
Does H₄ polytope (120-cell) discretization provide better boundary detection
than simple k-NN geometric features?

Comparison:
- Baseline: 256-D embeddings only (known: R²≈0.34)
- k-NN: Baseline + 7 k-NN features (known: R²≈0.40, +6.4%)
- Polytope: Baseline + H₄ polytope features (unknown: R²=???)
- Dummy: Baseline + random 4D projection (control)

Multi-run validation: 50 independent runs (5 seeds × 10 runs each)

Success criteria:
- polytope_r2 > knn_r2 + 0.02 (+2% additional gain) → H₄ validated
- polytope_r2 ≈ knn_r2 (within ±1%) → H₄ adds no value
- polytope_r2 < knn_r2 → H₄ worse than simple geometry

Usage:
    python experiments/test_h4_polytope_hypothesis.py
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

# ============================================================================
# Polytope Feature Computation
# ============================================================================

def compute_polytope_features(embeddings, k=50, seed=42):
    """
    Compute H₄ polytope features:
    1. Project to 4D via PCA
    2. Assign to 120-cell via k-means
    3. Extract geometric features from polytope structure

    Returns 7 features matching k-NN feature dimensionality.
    """
    np.random.seed(seed)
    N = len(embeddings)

    # Step 1: Project to 4D
    pca = PCA(n_components=4, random_state=seed)
    embeddings_4d = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()

    # Step 2: Assign to 120-cell (k-means as proxy)
    kmeans = KMeans(n_clusters=120, random_state=seed, n_init=10, max_iter=300)
    cell_assignments = kmeans.fit_predict(embeddings_4d)
    cell_centers = kmeans.cluster_centers_

    # Step 3: Compute features
    features = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        point_4d = embeddings_4d[i]
        assigned_cell = cell_assignments[i]

        # Feature 1: Distance to assigned cell center
        features[i, 0] = np.linalg.norm(point_4d - cell_centers[assigned_cell])

        # Feature 2-5: Coordinates in 4D space (captures position in polytope)
        features[i, 1:5] = point_4d

        # Feature 6: Distance to nearest cell center (not assigned)
        distances_to_centers = np.linalg.norm(
            cell_centers - point_4d[np.newaxis, :], axis=1
        )
        distances_to_centers[assigned_cell] = np.inf  # Exclude assigned
        features[i, 5] = distances_to_centers.min()

        # Feature 7: Variance explained ratio (how much 4D captures)
        features[i, 6] = variance_explained

    return features, {
        'variance_explained': float(variance_explained),
        'n_cells_used': len(np.unique(cell_assignments)),
        'mean_cell_size': float(np.bincount(cell_assignments).mean())
    }


def compute_knn_features(embeddings, k=50, seed=42):
    """Compute k-NN features (baseline comparison)."""
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
    """Random 4D projection (control)."""
    np.random.seed(seed)
    N = len(embeddings)

    # Random projection to 4D
    random_projection = np.random.randn(embeddings.shape[1], 4)
    random_projection /= np.linalg.norm(random_projection, axis=0)
    embeddings_4d_random = embeddings @ random_projection

    # Pad to 7 features with additional random noise
    features = np.random.randn(N, 7).astype(np.float32) * 0.1
    features[:, :4] = embeddings_4d_random

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
# Single Run Evaluation
# ============================================================================

def run_single_evaluation(seed, run_id, embeddings, boundary_distances, device):
    """
    Run single evaluation comparing all 4 conditions:
    - Baseline: embeddings only
    - k-NN: embeddings + k-NN features
    - Polytope: embeddings + H₄ polytope features
    - Dummy: embeddings + random 4D projection
    """
    # Set seeds
    np.random.seed(seed)
    training_seed = seed * 1000 + run_id
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)

    # Compute all features
    knn_features = compute_knn_features(embeddings, k=50, seed=seed)
    polytope_features, polytope_stats = compute_polytope_features(embeddings, k=50, seed=seed)
    dummy_features = compute_dummy_features(embeddings, seed=seed + run_id)

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

    knn_train = knn_features[train_idx_inner]
    knn_val = knn_features[val_idx]
    knn_test = knn_features[test_idx]

    polytope_train = polytope_features[train_idx_inner]
    polytope_val = polytope_features[val_idx]
    polytope_test = polytope_features[test_idx]

    dummy_train = dummy_features[train_idx_inner]
    dummy_val = dummy_features[val_idx]
    dummy_test = dummy_features[test_idx]

    # Augmented inputs
    X_train_knn = np.concatenate([X_train, knn_train], axis=1)
    X_val_knn = np.concatenate([X_val, knn_val], axis=1)
    X_test_knn = np.concatenate([X_test, knn_test], axis=1)

    X_train_polytope = np.concatenate([X_train, polytope_train], axis=1)
    X_val_polytope = np.concatenate([X_val, polytope_val], axis=1)
    X_test_polytope = np.concatenate([X_test, polytope_test], axis=1)

    X_train_dummy = np.concatenate([X_train, dummy_train], axis=1)
    X_val_dummy = np.concatenate([X_val, dummy_val], axis=1)
    X_test_dummy = np.concatenate([X_test, dummy_test], axis=1)

    # Train 4 models
    model_baseline = BaselineModel()
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device)

    model_knn = AugmentedModel()
    model_knn = train_model(model_knn, X_train_knn, y_train, X_val_knn, y_val, device)

    model_polytope = AugmentedModel()
    model_polytope = train_model(model_polytope, X_train_polytope, y_train,
                                   X_val_polytope, y_val, device)

    model_dummy = AugmentedModel()
    model_dummy = train_model(model_dummy, X_train_dummy, y_train, X_val_dummy, y_val, device)

    # Predictions on borderline region
    borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)

    if borderline_mask.sum() < 10:
        return None

    model_baseline.eval()
    model_knn.eval()
    model_polytope.eval()
    model_dummy.eval()

    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        preds_baseline = model_baseline(X_test_t).cpu().numpy()

        X_test_knn_t = torch.from_numpy(X_test_knn).float().to(device)
        preds_knn = model_knn(X_test_knn_t).cpu().numpy()

        X_test_polytope_t = torch.from_numpy(X_test_polytope).float().to(device)
        preds_polytope = model_polytope(X_test_polytope_t).cpu().numpy()

        X_test_dummy_t = torch.from_numpy(X_test_dummy).float().to(device)
        preds_dummy = model_dummy(X_test_dummy_t).cpu().numpy()

    # Evaluate on borderline
    r2_baseline = r2_score(y_test[borderline_mask], preds_baseline[borderline_mask])
    r2_knn = r2_score(y_test[borderline_mask], preds_knn[borderline_mask])
    r2_polytope = r2_score(y_test[borderline_mask], preds_polytope[borderline_mask])
    r2_dummy = r2_score(y_test[borderline_mask], preds_dummy[borderline_mask])

    return {
        'r2_baseline': float(r2_baseline),
        'r2_knn': float(r2_knn),
        'r2_polytope': float(r2_polytope),
        'r2_dummy': float(r2_dummy),
        'delta_knn': float(r2_knn - r2_baseline),
        'delta_polytope': float(r2_polytope - r2_baseline),
        'delta_dummy': float(r2_dummy - r2_baseline),
        'delta_polytope_vs_knn': float(r2_polytope - r2_knn),
        'n_borderline': int(borderline_mask.sum()),
        'polytope_stats': polytope_stats
    }


# ============================================================================
# Main Multi-Run Evaluation
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("H₄ POLYTOPE HYPOTHESIS TEST")
    print("="*80)
    print(f"\nDevice: {device}")
    print("\nCritical Question: Does 120-cell structure beat simple k-NN?\n")

    # Load data
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
    embeddings = np.load(data_path / "embeddings.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    print(f"Data: N={len(embeddings)}, D={embeddings.shape[1]}")
    print(f"Boundary distances: [{boundary_distances.min():.2f}, {boundary_distances.max():.2f}]\n")

    # Multi-run evaluation
    seeds = [17, 42, 100, 200, 333]
    runs_per_seed = 10

    all_results = []

    print("Running 50 independent evaluations...")
    print("Format: Baseline | k-NN | Polytope | Dummy\n")

    for seed in seeds:
        print(f"Seed {seed}:")

        for run_id in range(runs_per_seed):
            result = run_single_evaluation(seed, run_id, embeddings, boundary_distances, device)

            if result:
                all_results.append(result)

                print(f"  Run {run_id+1:2d}: "
                      f"R²={{base={result['r2_baseline']:.3f}, "
                      f"knn={result['r2_knn']:.3f} ({result['delta_knn']:+.3f}), "
                      f"poly={result['r2_polytope']:.3f} ({result['delta_polytope']:+.3f}), "
                      f"dummy={result['r2_dummy']:.3f} ({result['delta_dummy']:+.3f})}}")
        print()

    # Aggregate statistics
    print("="*80)
    print("AGGREGATE RESULTS")
    print("="*80 + "\n")

    deltas_knn = [r['delta_knn'] for r in all_results]
    deltas_polytope = [r['delta_polytope'] for r in all_results]
    deltas_dummy = [r['delta_dummy'] for r in all_results]
    deltas_polytope_vs_knn = [r['delta_polytope_vs_knn'] for r in all_results]

    mean_knn = np.mean(deltas_knn)
    std_knn = np.std(deltas_knn, ddof=1)

    mean_polytope = np.mean(deltas_polytope)
    std_polytope = np.std(deltas_polytope, ddof=1)

    mean_dummy = np.mean(deltas_dummy)
    std_dummy = np.std(deltas_dummy, ddof=1)

    mean_poly_vs_knn = np.mean(deltas_polytope_vs_knn)
    std_poly_vs_knn = np.std(deltas_polytope_vs_knn, ddof=1)

    print(f"ΔR² (vs Baseline):")
    print(f"  k-NN features:     {mean_knn:+.4f} ± {std_knn:.4f}")
    print(f"  Polytope features: {mean_polytope:+.4f} ± {std_polytope:.4f}")
    print(f"  Dummy features:    {mean_dummy:+.4f} ± {std_dummy:.4f}")
    print()
    print(f"ΔR² (Polytope vs k-NN):")
    print(f"  Mean:              {mean_poly_vs_knn:+.4f} ± {std_poly_vs_knn:.4f}")

    # Statistical test
    t_stat, p_value = ttest_rel(deltas_polytope, deltas_knn)

    print()
    print(f"Paired t-test (Polytope vs k-NN):")
    print(f"  t-statistic:       {t_stat:.4f}")
    print(f"  p-value:           {p_value:.6f}")

    # Polytope info
    variance_explained = np.mean([r['polytope_stats']['variance_explained']
                                  for r in all_results])
    print()
    print(f"Polytope Statistics:")
    print(f"  4D PCA variance explained: {variance_explained:.1%}")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT: H₄ POLYTOPE HYPOTHESIS")
    print("="*80 + "\n")

    threshold_meaningful = 0.02  # +2% improvement

    if mean_poly_vs_knn > threshold_meaningful and p_value < 0.05:
        print(f"✓ H₄ VALIDATED: Polytope significantly beats k-NN")
        print(f"  → Polytope advantage: {mean_poly_vs_knn:+.1%}")
        print(f"  → p-value: {p_value:.6f} (significant at α=0.05)")
        print(f"  → 120-cell structure provides additional geometric signal")
        verdict = "H4_VALIDATED"
    elif abs(mean_poly_vs_knn) < 0.01 and p_value > 0.05:
        print(f"✗ H₄ INVALIDATED: Polytope equivalent to k-NN")
        print(f"  → Difference: {mean_poly_vs_knn:+.1%} (negligible)")
        print(f"  → p-value: {p_value:.4f} (not significant)")
        print(f"  → 120-cell adds no value beyond simple k-NN geometry")
        verdict = "H4_EQUIVALENT"
    elif mean_poly_vs_knn < -threshold_meaningful and p_value < 0.05:
        print(f"✗ H₄ REJECTED: Polytope significantly worse than k-NN")
        print(f"  → Polytope disadvantage: {mean_poly_vs_knn:+.1%}")
        print(f"  → p-value: {p_value:.6f}")
        print(f"  → 4D projection loses important information")
        verdict = "H4_REJECTED"
    else:
        print(f"⚠️  INCONCLUSIVE: Effect too small or unstable")
        print(f"  → Difference: {mean_poly_vs_knn:+.1%}")
        print(f"  → p-value: {p_value:.4f}")
        print(f"  → Need more data or different approach")
        verdict = "INCONCLUSIVE"

    # Information loss analysis
    print()
    print("="*80)
    print("INFORMATION LOSS ANALYSIS")
    print("="*80 + "\n")

    print(f"4D PCA captures {variance_explained:.1%} of variance")
    print(f"Information loss: {1-variance_explained:.1%}")
    print()

    if variance_explained < 0.5:
        print("⚠️  WARNING: 4D projection loses >50% of variance")
        print("   → This may explain why polytope underperforms")
        print("   → Native 256-D geometry may be irreducible")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"h4_polytope_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'verdict': verdict,
        'n_runs': len(all_results),
        'statistics': {
            'knn_vs_baseline': {
                'mean': float(mean_knn),
                'std': float(std_knn)
            },
            'polytope_vs_baseline': {
                'mean': float(mean_polytope),
                'std': float(std_polytope)
            },
            'dummy_vs_baseline': {
                'mean': float(mean_dummy),
                'std': float(std_dummy)
            },
            'polytope_vs_knn': {
                'mean': float(mean_poly_vs_knn),
                'std': float(std_poly_vs_knn),
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        },
        'polytope_info': {
            'variance_explained': float(variance_explained),
            'information_loss': float(1 - variance_explained)
        },
        'all_results': all_results
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")

    return summary


if __name__ == "__main__":
    main()
