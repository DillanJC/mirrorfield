"""Comprehensive 4D Polytope Comparison

Tests all 6 regular convex 4-polytopes to identify optimal geometric structure:
1. 5-cell (4-simplex) - 5 tetrahedral cells
2. 8-cell (tesseract) - 8 cubic cells
3. 16-cell (hyperoctahedron) - 16 tetrahedral cells
4. 24-cell - 24 octahedral cells (unique to 4D!)
5. 120-cell (H₄) - 120 dodecahedral cells
6. 600-cell - 600 tetrahedral cells

Each polytope is tested on ridge proximity prediction task.
Critical question: Which geometric structure best captures embedding geometry?
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


def load_embeddings(run_dir):
    """Load embeddings from previous test run."""
    run_path = Path(run_dir)
    embeddings = np.load(run_path / "embeddings.npy")
    return embeddings


def project_to_4d(embeddings, method="pca", seed=42):
    """Project embeddings to 4D manifold."""
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


def assign_to_cells(embeddings_4d, num_cells, seed=42):
    """Assign 4D points to polytope cells via k-means."""
    print(f"\nAssigning points to {num_cells} cells...")

    kmeans = KMeans(n_clusters=num_cells, random_state=seed, n_init=10)
    cell_assignments = kmeans.fit_predict(embeddings_4d)

    # Get soft assignments (inverse distance weighting)
    distances = kmeans.transform(embeddings_4d)
    soft_assignments = 1.0 / (distances + 1e-6)
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)

    print(f"✓ Cell assignment complete")
    print(f"  Cluster sizes: min={np.bincount(cell_assignments).min()}, "
          f"max={np.bincount(cell_assignments).max()}, "
          f"mean={np.bincount(cell_assignments).mean():.1f}")

    return cell_assignments, soft_assignments, kmeans


def build_5cell_adjacency():
    """5-cell: Complete graph (all cells connected)."""
    num_cells = 5
    adjacency = np.ones((num_cells, num_cells), dtype=np.float32)
    np.fill_diagonal(adjacency, 0)  # No self-loops

    # Normalize
    adjacency = adjacency / 4.0  # Each cell has 4 neighbors

    return adjacency


def build_8cell_adjacency():
    """8-cell (tesseract): Cubic lattice structure."""
    num_cells = 8
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # Arrange as 2×2×2 cube, each cube face connects 2 cells
    # Vertices of hypercube at binary coordinates
    vertices = [(i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]

    for idx_i, v_i in enumerate(vertices):
        for idx_j, v_j in enumerate(vertices):
            if idx_i != idx_j:
                # Connect if Hamming distance = 1 (differ in exactly 1 coordinate)
                hamming = sum(a != b for a, b in zip(v_i, v_j))
                if hamming == 1:
                    adjacency[idx_i, idx_j] = 1.0

    # Normalize (each cell has 3 neighbors in tesseract)
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


def build_16cell_adjacency():
    """16-cell (hyperoctahedron): Dual of 8-cell."""
    num_cells = 16
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # 16-cell vertices: (±1, 0, 0, 0) and permutations (8 vertices)
    # Each cell (tetrahedron) connects to 4 others
    # Organized as: 2 groups of 8 cells

    for i in range(num_cells):
        group_i = i // 8  # Which half
        local_i = i % 8    # Position within half

        # Connect within same half (4 neighbors)
        for offset in [1, 2, 3, 4]:
            j = (group_i * 8) + ((local_i + offset) % 8)
            adjacency[i, j] = 1.0

        # Connect to opposite half (4 neighbors)
        for offset in [0, 2, 4, 6]:
            j = ((1 - group_i) * 8) + ((local_i + offset) % 8)
            adjacency[i, j] = 0.5

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


def build_24cell_adjacency():
    """24-cell: Self-dual, unique to 4D. 8 neighbors per cell."""
    num_cells = 24
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # 24-cell can be constructed from tesseract + 16-cell vertices
    # Each cell (octahedron) connects to 8 others
    # Organize as 3 groups of 8 cells

    num_groups = 3
    cells_per_group = 8

    for i in range(num_cells):
        group_i = i // cells_per_group
        local_i = i % cells_per_group

        # Connect within group (3 neighbors)
        for offset in [1, 3, 5]:
            j = (group_i * cells_per_group) + ((local_i + offset) % cells_per_group)
            adjacency[i, j] = 1.0

        # Connect to other groups (5 neighbors each)
        for g_offset in [1, 2]:
            neighbor_group = (group_i + g_offset) % num_groups
            for offset in [0, 2, 4, 6]:
                j = (neighbor_group * cells_per_group) + ((local_i + offset) % cells_per_group)
                adjacency[i, j] = 0.7

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


def build_120cell_adjacency():
    """120-cell (H₄): Hierarchical dodecahedral structure."""
    num_cells = 120
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # Hierarchical: 10 groups of 12 cells
    num_groups = 10
    cells_per_group = 12

    for i in range(num_cells):
        group_i = i // cells_per_group

        # Connect within group (11 neighbors)
        group_start = group_i * cells_per_group
        group_end = min(group_start + cells_per_group, num_cells)
        for j in range(group_start, group_end):
            if i != j:
                adjacency[i, j] = 1.0

        # Connect to adjacent groups (6 neighbors)
        for g_offset in [-1, 1]:
            neighbor_group = (group_i + g_offset) % num_groups
            neighbor_start = neighbor_group * cells_per_group
            neighbor_end = min(neighbor_start + cells_per_group, num_cells)

            for k in range(3):
                j = neighbor_start + (k * cells_per_group // 3)
                if j < neighbor_end:
                    adjacency[i, j] = 0.5

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


def build_600cell_adjacency():
    """600-cell: Largest regular 4-polytope. 20 neighbors per cell."""
    num_cells = 600
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    # Hierarchical: 30 groups of 20 cells (icosahedral structure)
    num_groups = 30
    cells_per_group = 20

    for i in range(num_cells):
        group_i = i // cells_per_group
        local_i = i % cells_per_group

        # Connect within group (12 neighbors - icosahedron)
        for offset in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]:
            j = (group_i * cells_per_group) + ((local_i + offset) % cells_per_group)
            adjacency[i, j] = 1.0

        # Connect to adjacent groups (8 neighbors)
        for g_offset in [-2, -1, 1, 2]:
            neighbor_group = (group_i + g_offset) % num_groups
            for k in [0, 5, 10, 15]:
                j = (neighbor_group * cells_per_group) + ((local_i + k) % cells_per_group)
                adjacency[i, j] = 0.5

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


def build_generic_adjacency(num_cells, target_degree=15, seed=42):
    """Generic random adjacency (baseline)."""
    rng = np.random.RandomState(seed)
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    for i in range(num_cells):
        neighbors = rng.choice(num_cells, size=target_degree, replace=False)
        for j in neighbors:
            if i != j:
                adjacency[i, j] = rng.uniform(0.5, 1.0)

    # Symmetrize
    adjacency = (adjacency + adjacency.T) / 2

    # Normalize
    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)

    return adjacency


class PolytopeGNN(nn.Module):
    """GNN with polytope-structured adjacency."""

    def __init__(self, num_cells, adjacency=None, hidden_dim=64):
        super().__init__()
        self.num_cells = num_cells

        if adjacency is not None:
            self.register_buffer("adjacency", torch.from_numpy(adjacency))
        else:
            self.adjacency = None

        # Cell embeddings
        self.cell_embeddings = nn.Embedding(num_cells, hidden_dim)

        # Message passing
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

        # Output
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, soft_assignments):
        batch_size = soft_assignments.shape[0]

        # Cell features
        cell_ids = torch.arange(self.num_cells, device=soft_assignments.device)
        cell_features = self.cell_embeddings(cell_ids)

        # Message passing
        if self.adjacency is not None:
            messages = []
            for i in range(self.num_cells):
                neighbors = self.adjacency[i] > 0
                if neighbors.sum() > 0:
                    self_feat = cell_features[i].unsqueeze(0).expand(neighbors.sum(), -1)
                    neighbor_feats = cell_features[neighbors]
                    combined = torch.cat([self_feat, neighbor_feats], dim=1)
                    msg = self.message_mlp(combined)
                    weights = self.adjacency[i, neighbors].unsqueeze(1)
                    msg = (msg * weights).sum(dim=0)
                else:
                    msg = torch.zeros_like(cell_features[i])
                messages.append(msg)

            messages = torch.stack(messages, dim=0)
            cell_features = cell_features + self.update_mlp(messages)

        # Aggregate
        batch_features = torch.matmul(soft_assignments, cell_features)
        predictions = self.output_head(batch_features).squeeze(1)

        return predictions


class BaselineModel(nn.Module):
    """Baseline MLP."""

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
    """Create ridge proximity prediction task."""
    print(f"\nCreating synthetic task (ridge proximity prediction)...")

    N = len(embeddings)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    inner_index = k // 4
    outer_index = k - 1

    r_inner = distances[:, inner_index]
    r_outer = distances[:, outer_index]
    ridge_proximity = r_outer / (r_inner + 1e-6)

    print(f"✓ Synthetic task created")
    print(f"  Ridge proximity: mean={ridge_proximity.mean():.3f}, "
          f"std={ridge_proximity.std():.3f}")

    return ridge_proximity


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device="cpu", verbose=False):
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

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return model, best_val_loss


def evaluate_model(model, X_test, y_test, device="cpu"):
    """Evaluate model."""
    model = model.to(device)
    model.eval()

    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        predictions = model(X_test_t).cpu().numpy()

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return {"r2": r2, "mse": mse, "rmse": rmse}


def main():
    """Test all 6 regular 4-polytopes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"polytope_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE 4D POLYTOPE COMPARISON")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings("runs/openai_3_large_test_20251231_024532")
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)")

    # Create task
    targets = create_synthetic_task(embeddings, k=64, seed=42)

    # Project to 4D
    embeddings_4d, pca = project_to_4d(embeddings, method="pca", seed=42)

    # Data split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    print(f"\nData split:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    X_train_embed = embeddings[train_idx]
    X_val_embed = embeddings[val_idx]
    X_test_embed = embeddings[test_idx]

    # Define polytopes to test
    polytopes = [
        {"name": "5-cell", "num_cells": 5, "builder": build_5cell_adjacency},
        {"name": "8-cell", "num_cells": 8, "builder": build_8cell_adjacency},
        {"name": "16-cell", "num_cells": 16, "builder": build_16cell_adjacency},
        {"name": "24-cell", "num_cells": 24, "builder": build_24cell_adjacency},
        {"name": "120-cell", "num_cells": 120, "builder": build_120cell_adjacency},
        {"name": "600-cell", "num_cells": 600, "builder": build_600cell_adjacency},
    ]

    results = {}

    # Test each polytope
    for poly in polytopes:
        print(f"\n{'='*80}")
        print(f"TESTING: {poly['name'].upper()} ({poly['num_cells']} cells)")
        print(f"{'='*80}")

        # Build adjacency
        adjacency = poly['builder']()
        avg_degree = (adjacency > 0).sum(axis=1).mean()
        print(f"Average degree: {avg_degree:.1f} neighbors per cell")

        # Assign to cells
        _, soft_assignments, _ = assign_to_cells(embeddings_4d, poly['num_cells'], seed=42)

        X_train_soft = soft_assignments[train_idx]
        X_val_soft = soft_assignments[val_idx]
        X_test_soft = soft_assignments[test_idx]

        # Train model
        model = PolytopeGNN(num_cells=poly['num_cells'], adjacency=adjacency, hidden_dim=64)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, val_loss = train_model(
            model, X_train_soft, y_train, X_val_soft, y_val,
            epochs=100, lr=0.001, device=device, verbose=False
        )

        # Evaluate
        result = evaluate_model(model, X_test_soft, y_test, device=device)
        results[poly['name']] = result

        print(f"\nTest Results:")
        print(f"  R² = {result['r2']:.4f}")
        print(f"  RMSE = {result['rmse']:.4f}")

    # Baseline
    print(f"\n{'='*80}")
    print("BASELINE (DIRECT FROM EMBEDDINGS)")
    print(f"{'='*80}")

    model_baseline = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64)
    print(f"Parameters: {sum(p.numel() for p in model_baseline.parameters()):,}")

    model_baseline, _ = train_model(
        model_baseline, X_train_embed, y_train, X_val_embed, y_val,
        epochs=100, lr=0.001, device=device, verbose=False
    )

    result_baseline = evaluate_model(model_baseline, X_test_embed, y_test, device=device)
    results['baseline'] = result_baseline

    print(f"\nTest Results:")
    print(f"  R² = {result_baseline['r2']:.4f}")
    print(f"  RMSE = {result_baseline['rmse']:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Polytope':<20} {'Cells':>6} {'R²':>10} {'RMSE':>10} {'ΔR² vs Baseline':>18}")
    print("-" * 75)

    # Sort by R² (descending)
    sorted_results = sorted(
        [(name, res) for name, res in results.items() if name != 'baseline'],
        key=lambda x: x[1]['r2'],
        reverse=True
    )

    for name, res in sorted_results:
        num_cells = next(p['num_cells'] for p in polytopes if p['name'] == name)
        delta_r2 = res['r2'] - result_baseline['r2']
        print(f"{name:<20} {num_cells:>6} {res['r2']:>10.4f} {res['rmse']:>10.4f} {delta_r2:>+18.4f}")

    print(f"{'Baseline':<20} {'N/A':>6} {result_baseline['r2']:>10.4f} {result_baseline['rmse']:>10.4f} {'—':>18}")

    # Best polytope
    best_name, best_result = sorted_results[0]
    best_delta = best_result['r2'] - result_baseline['r2']

    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}\n")

    print(f"Best polytope: {best_name.upper()}")
    print(f"  R² = {best_result['r2']:.4f}")
    print(f"  ΔR² vs baseline = {best_delta:+.4f}")

    if best_delta > 0.05:
        print(f"\n✓ STRONG SIGNAL: {best_name} significantly beats baseline")
        print(f"  → Geometric structure is valuable for this task")
        verdict = "GEOMETRIC_STRUCTURE_HELPS"
    elif best_delta > -0.05:
        print(f"\n⚠️  WEAK SIGNAL: {best_name} ≈ baseline")
        print(f"  → Geometric structure provides marginal value")
        verdict = "MARGINAL_VALUE"
    else:
        print(f"\n✗ NEGATIVE: {best_name} worse than baseline")
        print(f"  → Geometric structure hurts performance")
        verdict = "HARMFUL"

    # Save results
    summary = {
        "timestamp": timestamp,
        "verdict": verdict,
        "best_polytope": best_name,
        "best_r2": float(best_result['r2']),
        "baseline_r2": float(result_baseline['r2']),
        "delta_r2": float(best_delta),
        "all_results": {
            name: {
                "r2": float(res['r2']),
                "rmse": float(res['rmse']),
                "delta_r2": float(res['r2'] - result_baseline['r2'])
            }
            for name, res in results.items()
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
