"""
Bootstrap Confidence Intervals for Boundary-Sliced Evaluation

Tests statistical robustness of geometry gains in borderline region.
For each seed, computes 1000 bootstrap samples and 95% CI for ΔR².

Pass condition: All seeds show CI entirely above zero
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import r2_score
from sklearn.utils import resample

# ============================================================================
# Model Definitions
# ============================================================================

class BaselineModel(nn.Module):
    """Model A: Baseline only (256-D embeddings)"""
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out.squeeze(1)


class AugmentedModel(nn.Module):
    """Model B/C: Baseline + additional features"""
    def __init__(self, input_dim=256, aug_dim=7, hidden_dim=64):
        super().__init__()
        total_dim = input_dim + aug_dim
        self.fc1 = nn.Linear(total_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out.squeeze(1)


# ============================================================================
# Native Geometry Features (from baseline analysis)
# ============================================================================

def compute_native_geometry_features(embeddings, k=50, seed=42):
    """
    Compute geometry features in native 256-D space.

    Features:
    - k-NN distance statistics (mean, std, min, max)
    - Local variance ratio (anisotropy measure)
    - Neighborhood stability (density estimate)
    - Distance to nearest neighbor

    Returns: (N, 7) array
    """
    from sklearn.neighbors import NearestNeighbors

    np.random.seed(seed)
    N = len(embeddings)

    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Remove self (first neighbor)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Distance statistics
    mean_dist = distances.mean(axis=1)
    std_dist = distances.std(axis=1)
    min_dist = distances.min(axis=1)
    max_dist = distances.max(axis=1)

    # Local variance ratio (PCA on neighborhood)
    local_variance_ratio = np.zeros(N)
    for i in range(N):
        neighbors = embeddings[indices[i]]
        cov = np.cov(neighbors.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals.sum() > 0:
            local_variance_ratio[i] = eigvals[0] / eigvals.sum()  # First PC ratio
        else:
            local_variance_ratio[i] = 0.0

    # Neighborhood stability (inverse of density)
    neighborhood_stability = 1.0 / (mean_dist + 1e-8)

    # Distance to nearest neighbor
    nn_dist = distances[:, 0]

    features = np.column_stack([
        mean_dist,
        std_dist,
        min_dist,
        max_dist,
        local_variance_ratio,
        neighborhood_stability,
        nn_dist
    ])

    return features


# ============================================================================
# Training
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, device, epochs=100, lr=0.001, verbose=False):
    """Train a model with validation-based early stopping."""
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
        loss = torch.nn.functional.mse_loss(predictions, y_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = torch.nn.functional.mse_loss(val_predictions, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

    return model


def evaluate_model(model, X, y, device):
    """Evaluate model and return predictions."""
    model = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    return preds


# ============================================================================
# Bootstrap Analysis
# ============================================================================

def bootstrap_confidence_interval(y_true, y_pred_baseline, y_pred_geometry, n_iterations=1000, ci_percentile=95):
    """
    Compute bootstrap confidence interval for ΔR² = R²(geometry) - R²(baseline)

    Args:
        y_true: Ground truth values
        y_pred_baseline: Baseline model predictions
        y_pred_geometry: Geometry model predictions
        n_iterations: Number of bootstrap samples
        ci_percentile: Confidence interval percentile (e.g., 95 for 95% CI)

    Returns:
        dict with mean_delta, ci_low, ci_high, bootstrap_deltas
    """
    n = len(y_true)
    bootstrap_deltas = []

    for i in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n), replace=True, n_samples=n)

        y_true_boot = y_true[indices]
        y_pred_baseline_boot = y_pred_baseline[indices]
        y_pred_geometry_boot = y_pred_geometry[indices]

        # Compute R² for each model
        r2_baseline = r2_score(y_true_boot, y_pred_baseline_boot)
        r2_geometry = r2_score(y_true_boot, y_pred_geometry_boot)

        # Compute delta
        delta = r2_geometry - r2_baseline
        bootstrap_deltas.append(delta)

    bootstrap_deltas = np.array(bootstrap_deltas)

    # Compute confidence interval
    alpha = (100 - ci_percentile) / 2
    ci_low, ci_high = np.percentile(bootstrap_deltas, [alpha, 100 - alpha])
    mean_delta = bootstrap_deltas.mean()

    return {
        'mean_delta': mean_delta,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'bootstrap_deltas': bootstrap_deltas
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_bootstrap_analysis(
    seed,
    embeddings,
    boundary_distances,
    native_geometry_features,
    device,
    threshold_low=-0.5,
    threshold_high=0.5,
    n_bootstrap=1000,
    verbose=False
):
    """Run bootstrap analysis for a single seed.

    CRITICAL: Trains on ALL data, but evaluates bootstrap CIs on borderline region only.
    This matches the original boundary-sliced evaluation methodology.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

    # Split into train/val/test on ALL data (not just borderline)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    X_train = embeddings[train_idx]
    y_train = boundary_distances[train_idx]
    geo_train = native_geometry_features[train_idx]

    X_val = embeddings[val_idx]
    y_val = boundary_distances[val_idx]
    geo_val = native_geometry_features[val_idx]

    X_test = embeddings[test_idx]
    y_test = boundary_distances[test_idx]
    geo_test = native_geometry_features[test_idx]

    # Identify borderline samples in test set
    borderline_mask = (y_test >= threshold_low) & (y_test <= threshold_high)
    n_borderline = borderline_mask.sum()

    if verbose:
        print(f"\nFull test set: {len(y_test)} samples")
        print(f"Borderline in test set: {n_borderline} samples")

    if n_borderline < 10:
        print(f"WARNING: Only {n_borderline} borderline samples - skipping seed {seed}")
        return None

    # Train baseline model on ALL training data
    if verbose:
        print("\nTraining baseline model on ALL training data...")
    baseline_model = BaselineModel(input_dim=256, hidden_dim=64)
    baseline_model = train_model(baseline_model, X_train, y_train, X_val, y_val, device, epochs=100, verbose=False)

    # Train geometry model on ALL training data
    if verbose:
        print("Training geometry model on ALL training data...")
    X_train_geo = np.concatenate([X_train, geo_train], axis=1)
    X_val_geo = np.concatenate([X_val, geo_val], axis=1)
    X_test_geo = np.concatenate([X_test, geo_test], axis=1)
    geometry_model = AugmentedModel(input_dim=256, aug_dim=7, hidden_dim=64)
    geometry_model = train_model(geometry_model, X_train_geo, y_train, X_val_geo, y_val, device, epochs=100, verbose=False)

    # Get predictions on full test set
    y_pred_baseline = evaluate_model(baseline_model, X_test, y_test, device)
    y_pred_geometry = evaluate_model(geometry_model, X_test_geo, y_test, device)

    # Extract borderline samples only
    y_test_borderline = y_test[borderline_mask]
    y_pred_baseline_borderline = y_pred_baseline[borderline_mask]
    y_pred_geometry_borderline = y_pred_geometry[borderline_mask]

    # Compute direct R² on borderline region
    r2_baseline = r2_score(y_test_borderline, y_pred_baseline_borderline)
    r2_geometry = r2_score(y_test_borderline, y_pred_geometry_borderline)
    direct_delta = r2_geometry - r2_baseline

    if verbose:
        print(f"\nDirect evaluation on BORDERLINE region:")
        print(f"  Baseline R²:  {r2_baseline:.4f}")
        print(f"  Geometry R²:  {r2_geometry:.4f}")
        print(f"  ΔR²:          {direct_delta:+.4f}")

    # Bootstrap analysis on borderline region only
    if verbose:
        print(f"\nRunning {n_bootstrap} bootstrap iterations on borderline region...")

    bootstrap_results = bootstrap_confidence_interval(
        y_test_borderline,
        y_pred_baseline_borderline,
        y_pred_geometry_borderline,
        n_iterations=n_bootstrap,
        ci_percentile=95
    )

    ci_low = bootstrap_results['ci_low']
    ci_high = bootstrap_results['ci_high']
    mean_delta = bootstrap_results['mean_delta']

    if verbose:
        print(f"\nBootstrap Results:")
        print(f"  Mean ΔR²:     {mean_delta:+.4f}")
        print(f"  95% CI:       [{ci_low:+.4f}, {ci_high:+.4f}]")

        if ci_low > 0:
            print(f"  ✓ CI entirely above zero (statistically significant)")
        else:
            print(f"  ✗ CI includes zero (not statistically significant)")

    return {
        'seed': int(seed),
        'n_borderline': int(n_borderline),
        'n_test': int(len(y_test)),
        'r2_baseline': float(r2_baseline),
        'r2_geometry': float(r2_geometry),
        'direct_delta': float(direct_delta),
        'bootstrap': {
            'mean_delta': float(mean_delta),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'n_iterations': int(n_bootstrap)
        }
    }


def main():
    print("Bootstrap Confidence Interval Analysis")
    print("Testing statistical robustness of geometry gains in borderline region")
    print("="*60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load Phase D data
    print("\nLoading Phase D data...")
    data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")

    embeddings = np.load(data_path / "embeddings.npy")
    labels = np.load(data_path / "labels.npy")
    boundary_distances = np.load(data_path / "boundary_distances.npy")

    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Boundary distances: {boundary_distances.shape}")

    # Run bootstrap analysis for each seed
    seeds = [17, 42, 100, 200, 333]
    all_results = []

    for seed in seeds:
        # Set ALL random seeds for full reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Compute geometry features WITH SEED (critical for reproducibility!)
        print(f"\n  Computing native 256-D geometry features for seed {seed}...")
        native_geometry_features = compute_native_geometry_features(embeddings, k=50, seed=seed)

        result = run_bootstrap_analysis(
            seed=seed,
            embeddings=embeddings,
            boundary_distances=boundary_distances,
            native_geometry_features=native_geometry_features,
            device=device,
            threshold_low=-0.5,
            threshold_high=0.5,
            n_bootstrap=1000,
            verbose=True
        )

        if result is not None:
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Bootstrap Confidence Intervals")
    print(f"{'='*60}\n")

    print("Seed    ΔR²      95% CI                    Significant?")
    print("-" * 60)

    all_significant = True
    for result in all_results:
        seed = result['seed']
        mean_delta = result['bootstrap']['mean_delta']
        ci_low = result['bootstrap']['ci_low']
        ci_high = result['bootstrap']['ci_high']

        is_significant = ci_low > 0
        sig_marker = "✓" if is_significant else "✗"

        print(f"{seed:<6} {mean_delta:+.4f}   [{ci_low:+.4f}, {ci_high:+.4f}]   {sig_marker}")

        if not is_significant:
            all_significant = False

    print(f"\n{'='*60}")

    if all_significant:
        print("✓ PASS: All seeds show CI entirely above zero")
        print("  → Geometry gain is statistically robust")
        print("  → Ready for publication")
    else:
        print("✗ FAIL: Some seeds have CI including zero")
        print("  → Geometry gain not consistently significant")
        print("  → Need further investigation")

    print(f"{'='*60}\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("C:/Users/User/mirrorfield/runs") / f"bootstrap_ci_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'timestamp': timestamp,
        'n_bootstrap': 1000,
        'ci_percentile': 95,
        'seeds': seeds,
        'threshold_low': -0.5,
        'threshold_high': 0.5,
        'all_significant': all_significant,
        'verdict': 'PASS' if all_significant else 'FAIL',
        'results': all_results
    }

    summary_path = output_dir / "bootstrap_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print(f"  - bootstrap_summary.json")

    return summary


if __name__ == "__main__":
    main()
