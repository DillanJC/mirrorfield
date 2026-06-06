"""Test 2: H₄ vs Generic 4D Ablation

Tests whether H₄ (120-cell) symmetry adds value over generic 4D geometric priors.

Models:
- Model A: H₄-structured GNN (120 cells, Coxeter symmetry)
- Model B: Generic 4D GNN (same capacity, no H₄ constraint)
- Model C: Baseline (direct from embeddings)

Task: Predict geometric properties (ridge proximity, curvature) from embeddings
Critical test: Does A significantly beat B?

Timeline: ~2 hours (data prep + training + evaluation)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import sys
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_embeddings_and_geometry(run_dir):
    """Load embeddings and precomputed geometry features."""
    run_path = Path(run_dir)

    embeddings = np.load(run_path / "embeddings.npy")

    # Check if behavioral test results exist (has geometry features)
    behavioral_dir = Path("runs")
    behavioral_runs = list(behavioral_dir.glob("embedder_behavioral_tests_*/"))

    if behavioral_runs:
        # Load from most recent behavioral test
        latest = sorted(behavioral_runs)[-1]
        print(f"✓ Found behavioral test results: {latest.name}")
        # For now, we'll compute geometry on the fly

    return embeddings


def project_to_4d(embeddings, method="pca", seed=42):
    """Project embeddings to 4D manifold.

    Args:
        method: "pca" or "learned" (for now just PCA)
    """
    print(f"\nProjecting {embeddings.shape[0]} samples from {embeddings.shape[1]}-D to 4-D...")

    if method == "pca":
        pca = PCA(n_components=4, random_state=seed)
        projected = pca.fit_transform(embeddings)
        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"✓ PCA projection complete")
        print(f"  Variance explained: {variance_explained:.4f}")
        return projected.astype(np.float32), pca

    else:
        raise NotImplementedError(f"Projection method '{method}' not implemented")


def assign_to_h4_cells(embeddings_4d, num_cells=120, seed=42):
    """Assign 4D points to 120-cell Voronoi regions.

    For proof-of-concept, use k-means with 120 clusters.
    True H₄ would use 120-cell vertices as cluster centers.
    """
    print(f"\nAssigning points to {num_cells} cells...")

    kmeans = KMeans(n_clusters=num_cells, random_state=seed, n_init=10)
    cell_assignments = kmeans.fit_predict(embeddings_4d)

    # Get soft assignments (distances to all centroids)
    distances = kmeans.transform(embeddings_4d)  # (N, 120)
    soft_assignments = 1.0 / (distances + 1e-6)  # Inverse distance
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)

    print(f"✓ Cell assignment complete")
    print(f"  Cluster sizes: min={np.bincount(cell_assignments).min()}, "
          f"max={np.bincount(cell_assignments).max()}, "
          f"mean={np.bincount(cell_assignments).mean():.1f}")

    return cell_assignments, soft_assignments, kmeans


def build_h4_adjacency(num_cells=120):
    """Build adjacency matrix based on H₄ (120-cell) structure.

    For proof-of-concept, use a hierarchical structure:
    - Each cell connects to ~12 neighbors (dodecahedral faces)
    - Hierarchical levels: cells grouped into 10 super-cells of 12 each

    True H₄ would use Coxeter group structure.
    """
    print(f"\nBuilding H₄-structured adjacency matrix...")

    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # Hierarchical structure: 10 groups of 12 cells
    num_groups = 10
    cells_per_group = num_cells // num_groups

    for i in range(num_cells):
        group_i = i // cells_per_group

        # Connect within group (dodecahedral faces)
        group_start = group_i * cells_per_group
        group_end = min(group_start + cells_per_group, num_cells)
        for j in range(group_start, group_end):
            if i != j:
                adjacency[i, j] = 1.0

        # Connect to adjacent groups (hierarchical)
        for g_offset in [-1, 1]:
            neighbor_group = (group_i + g_offset) % num_groups
            neighbor_start = neighbor_group * cells_per_group
            neighbor_end = min(neighbor_start + cells_per_group, num_cells)

            # Connect to 3 cells in neighbor group (120-cell structure)
            for k in range(3):
                j = neighbor_start + (k * cells_per_group // 3)
                if j < neighbor_end:
                    adjacency[i, j] = 0.5  # Weaker connection

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    print(f"✓ H₄ adjacency built")
    print(f"  Average degree: {(adjacency > 0).sum(axis=1).mean():.1f}")

    return adjacency


def build_generic_adjacency(num_cells=120, seed=42):
    """Build random adjacency matrix (same average degree as H₄)."""
    print(f"\nBuilding generic (random) adjacency matrix...")

    rng = np.random.RandomState(seed)

    # Match H₄ average degree (~15 neighbors)
    target_degree = 15

    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)
    for i in range(num_cells):
        # Sample random neighbors
        num_neighbors = target_degree
        neighbors = rng.choice(num_cells, size=num_neighbors, replace=False)
        for j in neighbors:
            if i != j:
                adjacency[i, j] = rng.uniform(0.5, 1.0)

    # Symmetrize
    adjacency = (adjacency + adjacency.T) / 2

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    print(f"✓ Generic adjacency built")
    print(f"  Average degree: {(adjacency > 0).sum(axis=1).mean():.1f}")

    return adjacency


class H4GNN(nn.Module):
    """H₄-structured GNN (Model A)."""

    def __init__(self, num_cells=120, latent_dim=4, adjacency=None, hidden_dim=64):
        super().__init__()
        self.num_cells = num_cells
        self.latent_dim = latent_dim

        # Store adjacency as buffer (not a trainable parameter)
        if adjacency is not None:
            self.register_buffer("adjacency", torch.from_numpy(adjacency))
        else:
            self.adjacency = None

        # Cell feature embeddings (learned)
        self.cell_embeddings = nn.Embedding(num_cells, hidden_dim)

        # Message passing layers
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, soft_assignments):
        """
        Args:
            soft_assignments: (batch, num_cells) - soft assignment to cells

        Returns:
            predictions: (batch,) - predicted target values
        """
        batch_size = soft_assignments.shape[0]

        # Get cell embeddings
        cell_ids = torch.arange(self.num_cells, device=soft_assignments.device)
        cell_features = self.cell_embeddings(cell_ids)  # (num_cells, hidden_dim)

        # Message passing (1 layer for proof-of-concept)
        if self.adjacency is not None:
            # Aggregate messages from neighbors
            messages = []
            for i in range(self.num_cells):
                neighbors = self.adjacency[i] > 0
                if neighbors.sum() > 0:
                    # Combine self with neighbors
                    self_feat = cell_features[i].unsqueeze(0).expand(neighbors.sum(), -1)
                    neighbor_feats = cell_features[neighbors]
                    combined = torch.cat([self_feat, neighbor_feats], dim=1)
                    msg = self.message_mlp(combined)  # (num_neighbors, hidden_dim)
                    # Weight by adjacency
                    weights = self.adjacency[i, neighbors].unsqueeze(1)
                    msg = (msg * weights).sum(dim=0)  # (hidden_dim,)
                else:
                    msg = torch.zeros_like(cell_features[i])
                messages.append(msg)

            messages = torch.stack(messages, dim=0)  # (num_cells, hidden_dim)
            cell_features = cell_features + self.update_mlp(messages)

        # Aggregate cell features based on soft assignments
        batch_features = torch.matmul(soft_assignments, cell_features)  # (batch, hidden_dim)

        # Predict output
        predictions = self.output_head(batch_features).squeeze(1)  # (batch,)

        return predictions


class BaselineModel(nn.Module):
    """Baseline MLP (Model C)."""

    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(1)


def create_synthetic_task(embeddings, k=64, seed=42):
    """Create a task where geometry should matter.

    Task: Predict ridge proximity (ratio of outer to inner shell radius).
    This requires understanding 4D geometry, not just raw embeddings.
    """
    print(f"\nCreating synthetic task (ridge proximity prediction)...")

    N = len(embeddings)

    # Build kNN index
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # Compute ridge proximity (r_outer / r_inner)
    inner_index = k // 4  # 25th percentile
    outer_index = k - 1   # Last neighbor

    r_inner = distances[:, inner_index]
    r_outer = distances[:, outer_index]
    ridge_proximity = r_outer / (r_inner + 1e-6)

    print(f"✓ Synthetic task created")
    print(f"  Ridge proximity: mean={ridge_proximity.mean():.3f}, "
          f"std={ridge_proximity.std():.3f}, "
          f"range=[{ridge_proximity.min():.3f}, {ridge_proximity.max():.3f}]")

    return ridge_proximity


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, device="cpu"):
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

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = F.mse_loss(val_predictions, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    return model, best_val_loss


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate a trained model."""
    model = model.to(device)
    model.eval()

    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return {"r2": r2, "mse": mse, "rmse": rmse, "predictions": predictions}


def main():
    """Run Test 2: H₄ vs Generic 4D ablation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"h4_ablation_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("TEST 2: H₄ vs GENERIC 4D ABLATION")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings_and_geometry("runs/openai_3_large_test_20251231_024532")
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)")

    # Create synthetic task
    targets = create_synthetic_task(embeddings, k=64, seed=42)

    # Project to 4D
    embeddings_4d, pca = project_to_4d(embeddings, method="pca", seed=42)

    # Assign to cells
    cell_assignments, soft_assignments, kmeans = assign_to_h4_cells(embeddings_4d, num_cells=120, seed=42)

    # Build adjacencies
    h4_adjacency = build_h4_adjacency(num_cells=120)
    generic_adjacency = build_generic_adjacency(num_cells=120, seed=42)

    # Train/val/test split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Prepare data for each model
    X_train_embed = embeddings[train_idx]
    X_val_embed = embeddings[val_idx]
    X_test_embed = embeddings[test_idx]

    X_train_soft = soft_assignments[train_idx]
    X_val_soft = soft_assignments[val_idx]
    X_test_soft = soft_assignments[test_idx]

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    results = {}

    # Model A: H₄-structured GNN
    print(f"\n{'='*80}")
    print("MODEL A: H₄-STRUCTURED GNN")
    print(f"{'='*80}")

    model_a = H4GNN(num_cells=120, latent_dim=4, adjacency=h4_adjacency, hidden_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model_a.parameters()):,}")

    model_a, best_val_loss_a = train_model(
        model_a, X_train_soft, y_train, X_val_soft, y_val,
        epochs=100, lr=0.001, device=device
    )

    results_a = evaluate_model(model_a, X_test_soft, y_test, device=device)
    results["model_a_h4"] = results_a
    print(f"\nTest Results (Model A):")
    print(f"  R² = {results_a['r2']:.4f}")
    print(f"  RMSE = {results_a['rmse']:.4f}")

    # Model B: Generic 4D GNN
    print(f"\n{'='*80}")
    print("MODEL B: GENERIC 4D GNN (NO H₄ CONSTRAINT)")
    print(f"{'='*80}")

    model_b = H4GNN(num_cells=120, latent_dim=4, adjacency=generic_adjacency, hidden_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    model_b, best_val_loss_b = train_model(
        model_b, X_train_soft, y_train, X_val_soft, y_val,
        epochs=100, lr=0.001, device=device
    )

    results_b = evaluate_model(model_b, X_test_soft, y_test, device=device)
    results["model_b_generic"] = results_b
    print(f"\nTest Results (Model B):")
    print(f"  R² = {results_b['r2']:.4f}")
    print(f"  RMSE = {results_b['rmse']:.4f}")

    # Model C: Baseline
    print(f"\n{'='*80}")
    print("MODEL C: BASELINE (DIRECT FROM EMBEDDINGS)")
    print(f"{'='*80}")

    model_c = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model_c.parameters()):,}")

    model_c, best_val_loss_c = train_model(
        model_c, X_train_embed, y_train, X_val_embed, y_val,
        epochs=100, lr=0.001, device=device
    )

    results_c = evaluate_model(model_c, X_test_embed, y_test, device=device)
    results["model_c_baseline"] = results_c
    print(f"\nTest Results (Model C):")
    print(f"  R² = {results_c['r2']:.4f}")
    print(f"  RMSE = {results_c['rmse']:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: COMPARATIVE PERFORMANCE")
    print(f"{'='*80}\n")

    print(f"{'Model':<30} {'R²':<10} {'RMSE':<10}")
    print("-" * 50)
    print(f"{'A: H₄-structured GNN':<30} {results_a['r2']:>9.4f}  {results_a['rmse']:>9.4f}")
    print(f"{'B: Generic 4D GNN':<30} {results_b['r2']:>9.4f}  {results_b['rmse']:>9.4f}")
    print(f"{'C: Baseline MLP':<30} {results_c['r2']:>9.4f}  {results_c['rmse']:>9.4f}")

    # Critical comparison
    print(f"\n{'='*80}")
    print("CRITICAL COMPARISON: Does H₄ beat Generic 4D?")
    print(f"{'='*80}\n")

    delta_r2_ab = results_a['r2'] - results_b['r2']
    delta_r2_ac = results_a['r2'] - results_c['r2']
    delta_r2_bc = results_b['r2'] - results_c['r2']

    print(f"ΔR² (A vs B): {delta_r2_ab:+.4f}")
    print(f"ΔR² (A vs C): {delta_r2_ac:+.4f}")
    print(f"ΔR² (B vs C): {delta_r2_bc:+.4f}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}\n")

    threshold_significant = 0.05  # 5% R² improvement

    if delta_r2_ab > threshold_significant:
        print(f"✓ PASS: H₄ structure beats generic 4D (ΔR² = {delta_r2_ab:+.4f} > {threshold_significant})")
        print(f"  → H₄ symmetry is SPECIAL. Proceed with full architecture.")
        verdict = "H4_IS_SPECIAL"
    elif delta_r2_ab > -threshold_significant:
        print(f"⚠️  PARTIAL: H₄ ≈ Generic 4D (ΔR² = {delta_r2_ab:+.4f} ≈ 0)")
        print(f"  → 4D helps, but H₄ symmetry doesn't matter. Drop H₄ commitment.")
        verdict = "4D_HELPS_H4_DOESNT_MATTER"
    else:
        print(f"✗ FAIL: H₄ worse than generic 4D (ΔR² = {delta_r2_ab:+.4f} < -{threshold_significant})")
        print(f"  → H₄ structure is harmful. Use generic 4D or baseline.")
        verdict = "H4_HARMFUL"

    if delta_r2_bc > threshold_significant:
        print(f"\n✓ Both 4D models beat baseline (ΔR² = {delta_r2_bc:+.4f})")
        print(f"  → 4D geometric priors are valuable (Phase E validated).")
    elif delta_r2_bc > -threshold_significant:
        print(f"\n⚠️  4D models ≈ baseline (ΔR² = {delta_r2_bc:+.4f})")
        print(f"  → Geometry is complementary only.")
    else:
        print(f"\n✗ 4D models worse than baseline (ΔR² = {delta_r2_bc:+.4f})")
        print(f"  → 4D projection hurts performance.")

    # Save results
    summary = {
        "timestamp": timestamp,
        "verdict": verdict,
        "delta_r2_ab": float(delta_r2_ab),
        "delta_r2_ac": float(delta_r2_ac),
        "delta_r2_bc": float(delta_r2_bc),
        "model_a_r2": float(results_a['r2']),
        "model_b_r2": float(results_b['r2']),
        "model_c_r2": float(results_c['r2']),
        "threshold": threshold_significant,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
