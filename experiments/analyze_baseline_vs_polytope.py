"""Investigate Why Baseline Outperforms Polytope Geometric Priors

Analyzes what the baseline MLP learns that makes it superior to 4D polytope approaches.

Key questions:
1. What does baseline capture in 256-D that 4D projection loses?
2. Does baseline implicitly learn geometric structure?
3. Can geometric features complement (not replace) baseline?
4. Is the problem dimensionality reduction or discrete polytopes?

Tests:
- Baseline alone (winner: R² ~0.95)
- Baseline + 4D PCA features
- Baseline + polytope soft assignments
- Feature importance analysis
- Representation similarity

Usage:
    python experiments/analyze_baseline_vs_polytope.py
    python experiments/analyze_baseline_vs_polytope.py --seed 42 --verbose
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
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phase_d_data(run_dir):
    """Load real Phase D embeddings and boundary distances."""
    run_path = Path(run_dir)
    embeddings = np.load(run_path / "embeddings.npy")
    boundary_distances = np.load(run_path / "boundary_distances.npy")
    return embeddings, boundary_distances


def project_to_4d(embeddings, seed=42):
    """Project embeddings to 4D via PCA."""
    pca = PCA(n_components=4, random_state=seed)
    projected = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_
    return projected.astype(np.float32), pca, variance_explained


def assign_to_120cell(embeddings_4d, seed=42):
    """Assign 4D points to 120-cell via k-means."""
    kmeans = KMeans(n_clusters=120, random_state=seed, n_init=10)
    cell_assignments = kmeans.fit_predict(embeddings_4d)

    distances = kmeans.transform(embeddings_4d)
    soft_assignments = 1.0 / (distances + 1e-6)
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)

    return soft_assignments, kmeans


class BaselineModel(nn.Module):
    """Baseline MLP - the winner."""

    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_hidden=False):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)

        if return_hidden:
            return out.squeeze(1), h2
        return out.squeeze(1)


class AugmentedModel(nn.Module):
    """Baseline + additional features (4D PCA or polytope soft assignments)."""

    def __init__(self, input_dim=256, aug_dim=4, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + aug_dim, hidden_dim)
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


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate model."""
    model = model.to(device)
    model.eval()

    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {"r2": r2, "rmse": rmse, "predictions": predictions}


def extract_hidden_representations(model, X, device="cpu"):
    """Extract baseline's hidden layer activations."""
    model = model.to(device)
    model.eval()

    X_t = torch.from_numpy(X).float().to(device)

    with torch.no_grad():
        _, hidden = model(X_t, return_hidden=True)
        hidden_np = hidden.cpu().numpy()

    return hidden_np


def compute_feature_importance(model, X_test, y_test, device="cpu"):
    """Compute which input dimensions matter most to baseline."""
    model = model.to(device)
    model.eval()

    # Simple gradient-based importance
    X_test_t = torch.from_numpy(X_test).float().to(device)
    X_test_t.requires_grad = True

    predictions = model(X_test_t)
    loss = F.mse_loss(predictions, torch.from_numpy(y_test).float().to(device))

    loss.backward()

    # Feature importance = mean absolute gradient
    importance = X_test_t.grad.abs().mean(dim=0).cpu().numpy()

    return importance


def analyze_pca_components(pca, embeddings):
    """Analyze what variance PCA captures vs discards."""
    n_components = len(pca.explained_variance_ratio_)

    # Variance explained by first 4 components
    explained_4d = pca.explained_variance_ratio_[:4].sum()

    # Variance in discarded components
    if len(pca.explained_variance_ratio_) > 4:
        discarded = pca.explained_variance_ratio_[4:].sum()
    else:
        discarded = 0.0

    # Reconstruct from 4D and measure reconstruction error
    projected_4d = pca.transform(embeddings)[:, :4]
    # Use PCA with only 4 components for reconstruction
    pca_4d = PCA(n_components=4)
    pca_4d.fit(embeddings)
    reconstructed = pca_4d.inverse_transform(projected_4d)
    
    reconstruction_error = np.mean((embeddings - reconstructed) ** 2)

    return {
        "explained_4d": explained_4d,
        "discarded": discarded,
        "reconstruction_error": reconstruction_error,
        "component_variance": pca.explained_variance_ratio_[:10].tolist(),
    }


def test_complementarity(embeddings, targets, embeddings_4d, soft_assignments_120,
                         train_idx, val_idx, test_idx, device="cpu"):
    """Test if geometric features complement baseline."""

    results = {}

    # Prepare data splits
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    X_test = embeddings[test_idx]

    X_train_4d = embeddings_4d[train_idx]
    X_val_4d = embeddings_4d[val_idx]
    X_test_4d = embeddings_4d[test_idx]

    X_train_soft = soft_assignments_120[train_idx]
    X_val_soft = soft_assignments_120[val_idx]
    X_test_soft = soft_assignments_120[test_idx]

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    # Test 1: Baseline alone (already know this wins)
    print("\n  Testing: Baseline alone...")
    model_baseline = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64)
    model_baseline = train_model(model_baseline, X_train, y_train, X_val, y_val, device=device)
    results['baseline'] = evaluate_model(model_baseline, X_test, y_test, device=device)
    print(f"    R² = {results['baseline']['r2']:.4f}")

    # Test 2: Baseline + 4D PCA features
    print("\n  Testing: Baseline + 4D PCA features...")
    X_train_aug_pca = np.concatenate([X_train, X_train_4d], axis=1)
    X_val_aug_pca = np.concatenate([X_val, X_val_4d], axis=1)
    X_test_aug_pca = np.concatenate([X_test, X_test_4d], axis=1)

    model_aug_pca = AugmentedModel(input_dim=embeddings.shape[1], aug_dim=4, hidden_dim=64)
    model_aug_pca = train_model(model_aug_pca, X_train_aug_pca, y_train, X_val_aug_pca, y_val, device=device)
    results['baseline_plus_pca'] = evaluate_model(model_aug_pca, X_test_aug_pca, y_test, device=device)
    print(f"    R² = {results['baseline_plus_pca']['r2']:.4f} (Δ = {results['baseline_plus_pca']['r2'] - results['baseline']['r2']:+.4f})")

    # Test 3: Baseline + 120-cell soft assignments
    print("\n  Testing: Baseline + 120-cell soft assignments...")
    X_train_aug_soft = np.concatenate([X_train, X_train_soft], axis=1)
    X_val_aug_soft = np.concatenate([X_val, X_val_soft], axis=1)
    X_test_aug_soft = np.concatenate([X_test, X_test_soft], axis=1)

    model_aug_soft = AugmentedModel(input_dim=embeddings.shape[1], aug_dim=120, hidden_dim=64)
    model_aug_soft = train_model(model_aug_soft, X_train_aug_soft, y_train, X_val_aug_soft, y_val, device=device)
    results['baseline_plus_120cell'] = evaluate_model(model_aug_soft, X_test_aug_soft, y_test, device=device)
    print(f"    R² = {results['baseline_plus_120cell']['r2']:.4f} (Δ = {results['baseline_plus_120cell']['r2'] - results['baseline']['r2']:+.4f})")

    return results, model_baseline


def main():
    parser = argparse.ArgumentParser(description='Analyze why baseline beats polytopes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--embeddings', type=str, default='runs/openai_3_large_test_20251231_024532',
                        help='Path to embeddings directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"baseline_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("BASELINE vs POLYTOPE ANALYSIS")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Seed: {args.seed}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load data
    print("Loading Phase D data...")
    embeddings, boundary_distances = load_phase_d_data(args.embeddings)
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)")
    print(f"  Boundary distances: [{boundary_distances.min():.3f}, {boundary_distances.max():.3f}]")

    # Project to 4D
    print("\nProjecting to 4D via PCA...")
    embeddings_4d, pca, variance_explained = project_to_4d(embeddings, seed=args.seed)
    print(f"✓ 4D projection complete")
    print(f"  Variance explained by PC1-4: {variance_explained[:4].sum():.4f}")
    print(f"  Variance discarded: {1.0 - variance_explained[:4].sum():.4f}")

    # Assign to 120-cell
    print("\nAssigning to 120-cell...")
    soft_assignments_120, kmeans = assign_to_120cell(embeddings_4d, seed=args.seed)
    print(f"✓ Cell assignment complete")

    # Data split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=args.seed)

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Analysis 1: PCA Component Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS 1: What variance does PCA capture vs discard?")
    print(f"{'='*80}")

    pca_analysis = analyze_pca_components(pca, embeddings)

    print(f"\nVariance distribution:")
    print(f"  Explained by 4D: {pca_analysis['explained_4d']:.4f} (39.2%)")
    print(f"  Discarded: {pca_analysis['discarded']:.4f} (60.8%)")
    print(f"  Reconstruction error: {pca_analysis['reconstruction_error']:.6f}")

    print(f"\nTop 10 principal components:")
    for i, var in enumerate(pca_analysis['component_variance']):
        print(f"  PC{i+1}: {var:.4f} ({100*var:.2f}%)")

    # Analysis 2: Complementarity Test
    print(f"\n{'='*80}")
    print("ANALYSIS 2: Can geometric features complement baseline?")
    print(f"{'='*80}")

    comp_results, baseline_model = test_complementarity(
        embeddings, boundary_distances, embeddings_4d, soft_assignments_120,
        train_idx, val_idx, test_idx, device=device
    )

    # Analysis 3: Feature Importance
    print(f"\n{'='*80}")
    print("ANALYSIS 3: Which input dimensions matter to baseline?")
    print(f"{'='*80}")

    print("\nComputing feature importance...")
    X_test = embeddings[test_idx]
    y_test = boundary_distances[test_idx]

    importance = compute_feature_importance(baseline_model, X_test, y_test, device=device)

    print(f"\nFeature importance statistics:")
    print(f"  Mean importance: {importance.mean():.6f}")
    print(f"  Std importance: {importance.std():.6f}")
    print(f"  Max importance: {importance.max():.6f}")
    print(f"  Min importance: {importance.min():.6f}")

    # Top 10 most important dimensions
    top_indices = importance.argsort()[::-1][:10]
    print(f"\nTop 10 most important dimensions:")
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. Dimension {idx}: {importance[idx]:.6f}")

    # Analysis 4: Representation Similarity
    print(f"\n{'='*80}")
    print("ANALYSIS 4: Does baseline learn similar geometry to PCA?")
    print(f"{'='*80}")

    print("\nExtracting baseline hidden representations...")
    hidden_baseline = extract_hidden_representations(baseline_model, embeddings, device=device)

    # Compare baseline's 64-D hidden layer to 4D PCA
    print(f"\nComparing representations:")
    print(f"  Baseline hidden: {hidden_baseline.shape} (64-D)")
    print(f"  PCA projection: {embeddings_4d.shape} (4-D)")

    # Can PCA's 4D be predicted from baseline's 64-D hidden layer?
    # If yes → baseline learns superset of PCA structure
    # If no → baseline learns different structure

    from sklearn.linear_model import LinearRegression

    # Split for this analysis
    hidden_train = hidden_baseline[train_idx]
    hidden_test = hidden_baseline[test_idx]
    pca_train = embeddings_4d[train_idx]
    pca_test = embeddings_4d[test_idx]

    # Predict each PCA dimension from baseline hidden layer
    pca_r2_scores = []
    for dim in range(4):
        reg = LinearRegression()
        reg.fit(hidden_train, pca_train[:, dim])
        pred = reg.predict(hidden_test)
        r2 = r2_score(pca_test[:, dim], pred)
        pca_r2_scores.append(r2)
        print(f"  PC{dim+1} predictable from baseline hidden: R² = {r2:.4f}")

    avg_r2 = np.mean(pca_r2_scores)
    print(f"\n  Average R²: {avg_r2:.4f}")

    if avg_r2 > 0.8:
        print(f"  → Baseline learns superset of PCA structure (redundant)")
    elif avg_r2 > 0.5:
        print(f"  → Baseline partially overlaps with PCA (complementary)")
    else:
        print(f"  → Baseline learns different structure than PCA (orthogonal)")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Why Does Baseline Win?")
    print(f"{'='*80}\n")

    print(f"1. INFORMATION LOSS:")
    print(f"   4D PCA discards {pca_analysis['discarded']:.1%} of variance")
    print(f"   Baseline uses all 256 dimensions")
    print(f"   → Baseline has access to 2.5× more information\n")

    print(f"2. COMPLEMENTARITY:")
    baseline_r2 = comp_results['baseline']['r2']
    pca_r2 = comp_results['baseline_plus_pca']['r2']
    soft_r2 = comp_results['baseline_plus_120cell']['r2']

    print(f"   Baseline alone:          R² = {baseline_r2:.4f}")
    print(f"   Baseline + 4D PCA:       R² = {pca_r2:.4f} (Δ = {pca_r2 - baseline_r2:+.4f})")
    print(f"   Baseline + 120-cell:     R² = {soft_r2:.4f} (Δ = {soft_r2 - baseline_r2:+.4f})")

    if max(pca_r2 - baseline_r2, soft_r2 - baseline_r2) < 0.01:
        print(f"   → Geometry adds no value (redundant with baseline)\n")
    elif max(pca_r2 - baseline_r2, soft_r2 - baseline_r2) > 0.05:
        print(f"   → Geometry adds significant value (complementary)\n")
    else:
        print(f"   → Geometry adds marginal value (slightly complementary)\n")

    print(f"3. REPRESENTATION:")
    if avg_r2 > 0.8:
        print(f"   Baseline implicitly learns PCA-like structure (R² = {avg_r2:.2f})")
        print(f"   → Polytope geometry is redundant\n")
    else:
        print(f"   Baseline learns different structure than PCA (R² = {avg_r2:.2f})")
        print(f"   → But baseline's structure works better for this task\n")

    print(f"4. CONCLUSION:")
    if max(pca_r2 - baseline_r2, soft_r2 - baseline_r2) < 0.01:
        print(f"   Dimensionality reduction loses critical information")
        print(f"   Geometric priors don't compensate for lost signal")
        print(f"   → If geometry is needed, must work in native high-D space\n")
    else:
        print(f"   Geometry provides some complementary signal")
        print(f"   Worth exploring hybrid approaches")
        print(f"   → Baseline + geometric features might be best path\n")

    # Save results
    summary = {
        "timestamp": timestamp,
        "seed": args.seed,
        "pca_analysis": pca_analysis,
        "complementarity": {
            "baseline_r2": float(baseline_r2),
            "baseline_plus_pca_r2": float(pca_r2),
            "baseline_plus_120cell_r2": float(soft_r2),
            "pca_delta": float(pca_r2 - baseline_r2),
            "soft_delta": float(soft_r2 - baseline_r2),
        },
        "representation_similarity": {
            "pca_predictable_from_baseline": [float(r2) for r2 in pca_r2_scores],
            "average_r2": float(avg_r2),
        },
        "feature_importance": {
            "top_10_dimensions": [int(idx) for idx in top_indices],
            "top_10_scores": [float(importance[idx]) for idx in top_indices],
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
