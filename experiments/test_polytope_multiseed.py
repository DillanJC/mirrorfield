"""Multi-Seed Robustness Test for 4D Polytope Comparison

Tests whether the 16-cell winner is stable across multiple random seeds.

Critical validation: Does 16-cell win on ≥80% of seeds with stable margins?

Usage:
    python experiments/test_polytope_multiseed.py --seeds 5
    python experiments/test_polytope_multiseed.py --seeds 10 --verbose
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

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_embeddings(run_dir):
    """Load embeddings from previous test run."""
    run_path = Path(run_dir)
    embeddings = np.load(run_path / "embeddings.npy")
    return embeddings


def project_to_4d(embeddings, method="pca", seed=42):
    """Project embeddings to 4D manifold."""
    if method == "pca":
        pca = PCA(n_components=4, random_state=seed)
        projected = pca.fit_transform(embeddings)
        variance_explained = pca.explained_variance_ratio_.sum()
        return projected.astype(np.float32), pca, variance_explained
    else:
        raise NotImplementedError(f"Projection method '{method}' not implemented")


def assign_to_cells(embeddings_4d, num_cells, seed=42):
    """Assign 4D points to polytope cells via k-means."""
    kmeans = KMeans(n_clusters=num_cells, random_state=seed, n_init=10)
    cell_assignments = kmeans.fit_predict(embeddings_4d)

    distances = kmeans.transform(embeddings_4d)
    soft_assignments = 1.0 / (distances + 1e-6)
    soft_assignments /= soft_assignments.sum(axis=1, keepdims=True)

    return cell_assignments, soft_assignments, kmeans


def build_5cell_adjacency():
    """5-cell: Complete graph."""
    num_cells = 5
    adjacency = np.ones((num_cells, num_cells), dtype=np.float32)
    np.fill_diagonal(adjacency, 0)
    adjacency = adjacency / 4.0
    return adjacency


def build_8cell_adjacency():
    """8-cell (tesseract): Cubic lattice."""
    num_cells = 8
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    vertices = [(i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]]

    for idx_i, v_i in enumerate(vertices):
        for idx_j, v_j in enumerate(vertices):
            if idx_i != idx_j:
                hamming = sum(a != b for a, b in zip(v_i, v_j))
                if hamming == 1:
                    adjacency[idx_i, idx_j] = 1.0

    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)
    return adjacency


def build_16cell_adjacency():
    """16-cell (hyperoctahedron): Dual of 8-cell."""
    num_cells = 16
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    for i in range(num_cells):
        group_i = i // 8
        local_i = i % 8

        for offset in [1, 2, 3, 4]:
            j = (group_i * 8) + ((local_i + offset) % 8)
            adjacency[i, j] = 1.0

        for offset in [0, 2, 4, 6]:
            j = ((1 - group_i) * 8) + ((local_i + offset) % 8)
            adjacency[i, j] = 0.5

    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)
    return adjacency


def build_24cell_adjacency():
    """24-cell: Self-dual, unique to 4D."""
    num_cells = 24
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    num_groups = 3
    cells_per_group = 8

    for i in range(num_cells):
        group_i = i // cells_per_group
        local_i = i % cells_per_group

        for offset in [1, 3, 5]:
            j = (group_i * cells_per_group) + ((local_i + offset) % cells_per_group)
            adjacency[i, j] = 1.0

        for g_offset in [1, 2]:
            neighbor_group = (group_i + g_offset) % num_groups
            for offset in [0, 2, 4, 6]:
                j = (neighbor_group * cells_per_group) + ((local_i + offset) % cells_per_group)
                adjacency[i, j] = 0.7

    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)
    return adjacency


def build_120cell_adjacency():
    """120-cell (H₄): Hierarchical dodecahedral structure."""
    num_cells = 120
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    num_groups = 10
    cells_per_group = 12

    for i in range(num_cells):
        group_i = i // cells_per_group

        group_start = group_i * cells_per_group
        group_end = min(group_start + cells_per_group, num_cells)
        for j in range(group_start, group_end):
            if i != j:
                adjacency[i, j] = 1.0

        for g_offset in [-1, 1]:
            neighbor_group = (group_i + g_offset) % num_groups
            neighbor_start = neighbor_group * cells_per_group
            neighbor_end = min(neighbor_start + cells_per_group, num_cells)

            for k in range(3):
                j = neighbor_start + (k * cells_per_group // 3)
                if j < neighbor_end:
                    adjacency[i, j] = 0.5

    row_sums = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / (row_sums + 1e-6)
    return adjacency


def build_600cell_adjacency():
    """600-cell: Largest regular 4-polytope."""
    num_cells = 600
    adjacency = np.zeros((num_cells, num_cells), dtype=np.float32)

    num_groups = 30
    cells_per_group = 20

    for i in range(num_cells):
        group_i = i // cells_per_group
        local_i = i % cells_per_group

        for offset in [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]:
            j = (group_i * cells_per_group) + ((local_i + offset) % cells_per_group)
            adjacency[i, j] = 1.0

        for g_offset in [-2, -1, 1, 2]:
            neighbor_group = (group_i + g_offset) % num_groups
            for k in [0, 5, 10, 15]:
                j = (neighbor_group * cells_per_group) + ((local_i + k) % cells_per_group)
                adjacency[i, j] = 0.5

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

        self.cell_embeddings = nn.Embedding(num_cells, hidden_dim)

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

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, soft_assignments):
        batch_size = soft_assignments.shape[0]

        cell_ids = torch.arange(self.num_cells, device=soft_assignments.device)
        cell_features = self.cell_embeddings(cell_ids)

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
    N = len(embeddings)
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    inner_index = k // 4
    outer_index = k - 1

    r_inner = distances[:, inner_index]
    r_outer = distances[:, outer_index]
    ridge_proximity = r_outer / (r_inner + 1e-6)

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


def run_single_seed(seed, embeddings, device, verbose=False):
    """Run polytope comparison for a single seed."""

    if verbose:
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}\n")

    # Create task
    targets = create_synthetic_task(embeddings, k=64, seed=seed)

    # Project to 4D
    embeddings_4d, pca, variance_explained = project_to_4d(embeddings, method="pca", seed=seed)

    # Data split
    indices = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    y_train = targets[train_idx]
    y_val = targets[val_idx]
    y_test = targets[test_idx]

    X_train_embed = embeddings[train_idx]
    X_val_embed = embeddings[val_idx]
    X_test_embed = embeddings[test_idx]

    # Define polytopes
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
        if verbose:
            print(f"Testing {poly['name']}...", end=" ", flush=True)

        adjacency = poly['builder']()
        _, soft_assignments, _ = assign_to_cells(embeddings_4d, poly['num_cells'], seed=seed)

        X_train_soft = soft_assignments[train_idx]
        X_val_soft = soft_assignments[val_idx]
        X_test_soft = soft_assignments[test_idx]

        model = PolytopeGNN(num_cells=poly['num_cells'], adjacency=adjacency, hidden_dim=64)
        model, val_loss = train_model(
            model, X_train_soft, y_train, X_val_soft, y_val,
            epochs=100, lr=0.001, device=device, verbose=False
        )

        result = evaluate_model(model, X_test_soft, y_test, device=device)
        results[poly['name']] = result

        if verbose:
            print(f"R² = {result['r2']:.4f}")

    # Baseline
    if verbose:
        print(f"Testing baseline...", end=" ", flush=True)

    model_baseline = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64)
    model_baseline, _ = train_model(
        model_baseline, X_train_embed, y_train, X_val_embed, y_val,
        epochs=100, lr=0.001, device=device, verbose=False
    )

    result_baseline = evaluate_model(model_baseline, X_test_embed, y_test, device=device)
    results['baseline'] = result_baseline

    if verbose:
        print(f"R² = {result_baseline['r2']:.4f}")

    # Add deltas
    for name in results:
        if name != 'baseline':
            results[name]['delta_r2'] = results[name]['r2'] - results['baseline']['r2']

    # Find winner
    polytope_results = {k: v for k, v in results.items() if k != 'baseline'}
    winner = max(polytope_results.items(), key=lambda x: x[1]['r2'])

    return {
        'seed': seed,
        'results': results,
        'winner': winner[0],
        'winner_r2': winner[1]['r2'],
        'winner_delta': winner[1]['delta_r2'],
        'variance_explained': variance_explained,
    }


def main():
    parser = argparse.ArgumentParser(description='Multi-seed polytope robustness test')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to test')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--embeddings', type=str, default='runs/openai_3_large_test_20251231_024532',
                        help='Path to embeddings directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"polytope_multiseed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("MULTI-SEED POLYTOPE ROBUSTNESS TEST")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Number of seeds: {args.seeds}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load embeddings
    print("Loading embeddings...")
    embeddings = load_embeddings(args.embeddings)
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)\n")

    # Define seeds
    seed_list = [17, 42, 100, 200, 333, 500, 666, 777, 888, 999][:args.seeds]

    print(f"Testing seeds: {seed_list}\n")
    print(f"{'='*80}\n")

    # Run experiments
    all_results = []
    for seed in seed_list:
        result = run_single_seed(seed, embeddings, device, verbose=args.verbose)
        all_results.append(result)

        print(f"Seed {seed:3d}: Winner = {result['winner']:10s} (R² = {result['winner_r2']:+.4f}, ΔR² = {result['winner_delta']:+.4f})")

    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print(f"{'='*80}\n")

    # Count wins per polytope
    win_counts = {}
    for result in all_results:
        winner = result['winner']
        win_counts[winner] = win_counts.get(winner, 0) + 1

    print(f"Win counts (out of {args.seeds} seeds):")
    for name, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / args.seeds
        print(f"  {name:15s}: {count:2d} wins ({percentage:5.1f}%)")

    # Compute statistics per polytope
    print(f"\nPerformance statistics across seeds:\n")
    print(f"{'Polytope':<15} {'Mean R²':>10} {'Std R²':>10} {'Mean ΔR²':>10} {'Std ΔR²':>10}")
    print("-" * 70)

    polytope_names = ['5-cell', '8-cell', '16-cell', '24-cell', '120-cell', '600-cell']

    stats = {}
    for name in polytope_names:
        r2_values = [r['results'][name]['r2'] for r in all_results]
        delta_values = [r['results'][name]['delta_r2'] for r in all_results]

        stats[name] = {
            'mean_r2': np.mean(r2_values),
            'std_r2': np.std(r2_values),
            'mean_delta': np.mean(delta_values),
            'std_delta': np.std(delta_values),
            'r2_values': r2_values,
            'delta_values': delta_values,
        }

        print(f"{name:<15} {stats[name]['mean_r2']:>10.4f} {stats[name]['std_r2']:>10.4f} "
              f"{stats[name]['mean_delta']:>10.4f} {stats[name]['std_delta']:>10.4f}")

    # Verdict
    print(f"\n{'='*80}")
    print("ROBUSTNESS VERDICT")
    print(f"{'='*80}\n")

    most_common_winner = max(win_counts.items(), key=lambda x: x[1])
    winner_name = most_common_winner[0]
    winner_percentage = 100 * most_common_winner[1] / args.seeds

    print(f"Most consistent winner: {winner_name}")
    print(f"Win rate: {winner_percentage:.1f}% ({most_common_winner[1]}/{args.seeds} seeds)")
    print(f"Mean R²: {stats[winner_name]['mean_r2']:.4f} ± {stats[winner_name]['std_r2']:.4f}")
    print(f"Mean ΔR² vs baseline: {stats[winner_name]['mean_delta']:.4f} ± {stats[winner_name]['std_delta']:.4f}")

    if winner_percentage >= 80:
        print(f"\n✓ ROBUST: {winner_name} wins on ≥80% of seeds")
        print(f"  → Finding is stable, safe to build on")
        verdict = "ROBUST"
    elif winner_percentage >= 60:
        print(f"\n⚠️  MODERATE: {winner_name} wins on 60-80% of seeds")
        print(f"  → Finding is likely real but with some variance")
        verdict = "MODERATE"
    else:
        print(f"\n✗ UNSTABLE: No clear winner (best wins <60% of seeds)")
        print(f"  → Finding may be seed-dependent, investigate further")
        verdict = "UNSTABLE"

    # Check margin stability
    margin_cv = stats[winner_name]['std_delta'] / abs(stats[winner_name]['mean_delta'])
    print(f"\nMargin stability (CV of ΔR²): {margin_cv:.3f}")

    if margin_cv < 0.3:
        print(f"✓ Margin is stable (CV < 0.3)")
    elif margin_cv < 0.5:
        print(f"⚠️  Margin has moderate variance (CV = 0.3-0.5)")
    else:
        print(f"✗ Margin is unstable (CV > 0.5)")

    # Save results
    summary = {
        "timestamp": timestamp,
        "num_seeds": args.seeds,
        "seeds": seed_list,
        "verdict": verdict,
        "winner": winner_name,
        "win_rate": winner_percentage,
        "win_counts": win_counts,
        "statistics": {
            name: {
                "mean_r2": float(stat['mean_r2']),
                "std_r2": float(stat['std_r2']),
                "mean_delta_r2": float(stat['mean_delta']),
                "std_delta_r2": float(stat['std_delta']),
            }
            for name, stat in stats.items()
        },
        "all_results": [
            {
                "seed": r['seed'],
                "winner": r['winner'],
                "winner_r2": float(r['winner_r2']),
                "winner_delta": float(r['winner_delta']),
                "variance_explained": float(r['variance_explained']),
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
