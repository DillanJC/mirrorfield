"""Cross-Task Polytope Validation on Real Phase D Geometry

Tests whether polytope geometric priors generalize across real tasks:
1. Boundary distance prediction (regression)
2. Friction classification (binary: Presence vs Friction)
3. Local curvature estimation (geometric complexity)

Critical question: Does any polytope consistently help across real geometry tasks?

Evaluates not just performance but whether geometry encodes ethical principles:
- Does it degrade gracefully under perturbation?
- Can it detect uncertainty at boundaries?
- Is safety geometrically natural (low jitter near safe regions)?

Usage:
    python experiments/test_polytope_crosstask.py --seeds 5
    python experiments/test_polytope_crosstask.py --seeds 10 --verbose
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
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phase_d_data(run_dir):
    """Load real Phase D embeddings, labels, and boundary distances."""
    run_path = Path(run_dir)

    embeddings = np.load(run_path / "embeddings.npy")
    labels = np.load(run_path / "labels.npy")
    boundary_distances = np.load(run_path / "boundary_distances.npy")

    return embeddings, labels, boundary_distances


def compute_local_curvature(embeddings, k=32, seed=42):
    """Estimate local curvature from embedding geometry.

    Curvature measures how much the local neighborhood deviates from flatness.
    High curvature = complex local geometry.
    """
    N = len(embeddings)

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    curvatures = np.zeros(N)

    for i in range(N):
        # Get k nearest neighbors (excluding self)
        neighbors = embeddings[indices[i, 1:]]

        # Center at query point
        centered = neighbors - embeddings[i]

        # Compute covariance
        cov = np.cov(centered.T)

        # Eigenvalues measure variance along principal directions
        eigenvalues = np.linalg.eigvalsh(cov)

        # Curvature as ratio of smallest to largest eigenvalue
        # Low ratio = stretched/curved, high ratio = spherical/flat
        if eigenvalues[-1] > 1e-10:
            curvatures[i] = 1.0 - (eigenvalues[0] / eigenvalues[-1])
        else:
            curvatures[i] = 0.0

    return curvatures


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

    def __init__(self, num_cells, adjacency=None, hidden_dim=64, task="regression"):
        super().__init__()
        self.num_cells = num_cells
        self.task = task

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

        if task == "regression":
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:  # classification
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # Binary classification
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
        output = self.output_head(batch_features)

        if self.task == "regression":
            return output.squeeze(1)
        else:
            return output


class BaselineModel(nn.Module):
    """Baseline MLP."""

    def __init__(self, input_dim=256, hidden_dim=64, task="regression"):
        super().__init__()
        self.task = task

        if task == "regression":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:  # classification
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

    def forward(self, x):
        output = self.mlp(x)
        if self.task == "regression":
            return output.squeeze(1)
        else:
            return output


def train_model(model, X_train, y_train, X_val, y_val, task="regression",
                epochs=100, lr=0.001, device="cpu", verbose=False):
    """Train a model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    if task == "regression":
        y_train_t = y_train_t.float()
        y_val_t = y_val_t.float()
    else:
        y_train_t = y_train_t.long()
        y_val_t = y_val_t.long()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)

        if task == "regression":
            loss = F.mse_loss(predictions, y_train_t)
        else:
            loss = F.cross_entropy(predictions, y_train_t)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            if task == "regression":
                val_loss = F.mse_loss(val_predictions, y_val_t).item()
            else:
                val_loss = F.cross_entropy(val_predictions, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model, best_val_loss


def evaluate_model(model, X_test, y_test, task="regression", device="cpu"):
    """Evaluate model."""
    model = model.to(device)
    model.eval()

    X_test_t = torch.from_numpy(X_test).float().to(device)

    with torch.no_grad():
        if task == "regression":
            predictions = model(X_test_t).cpu().numpy()
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            return {"r2": r2, "mse": mse, "rmse": rmse}
        else:
            logits = model(X_test_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred_classes = logits.argmax(dim=1).cpu().numpy()

            acc = accuracy_score(y_test, pred_classes)
            auc = roc_auc_score(y_test, probs[:, 1])

            return {"accuracy": acc, "auc": auc}


def run_single_task_seed(seed, embeddings, targets, task_type, device, verbose=False):
    """Run polytope comparison for a single task and seed."""

    if verbose:
        print(f"  Seed {seed}...", end=" ", flush=True)

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
        adjacency = poly['builder']()
        _, soft_assignments, _ = assign_to_cells(embeddings_4d, poly['num_cells'], seed=seed)

        X_train_soft = soft_assignments[train_idx]
        X_val_soft = soft_assignments[val_idx]
        X_test_soft = soft_assignments[test_idx]

        model = PolytopeGNN(num_cells=poly['num_cells'], adjacency=adjacency,
                           hidden_dim=64, task=task_type)
        model, val_loss = train_model(
            model, X_train_soft, y_train, X_val_soft, y_val,
            task=task_type, epochs=100, lr=0.001, device=device, verbose=False
        )

        result = evaluate_model(model, X_test_soft, y_test, task=task_type, device=device)
        results[poly['name']] = result

    # Baseline
    model_baseline = BaselineModel(input_dim=embeddings.shape[1], hidden_dim=64, task=task_type)
    model_baseline, _ = train_model(
        model_baseline, X_train_embed, y_train, X_val_embed, y_val,
        task=task_type, epochs=100, lr=0.001, device=device, verbose=False
    )

    result_baseline = evaluate_model(model_baseline, X_test_embed, y_test,
                                     task=task_type, device=device)
    results['baseline'] = result_baseline

    # Compute deltas
    metric_key = 'r2' if task_type == 'regression' else 'auc'
    for name in results:
        if name != 'baseline':
            results[name][f'delta_{metric_key}'] = (results[name][metric_key] -
                                                     results['baseline'][metric_key])

    # Find winner
    polytope_results = {k: v for k, v in results.items() if k != 'baseline'}
    winner = max(polytope_results.items(), key=lambda x: x[1][metric_key])

    if verbose:
        print(f"Winner: {winner[0]} ({metric_key}={winner[1][metric_key]:.3f})")

    return {
        'seed': seed,
        'results': results,
        'winner': winner[0],
        'winner_score': winner[1][metric_key],
        'winner_delta': winner[1][f'delta_{metric_key}'],
    }


def main():
    parser = argparse.ArgumentParser(description='Cross-task polytope validation on real Phase D data')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to test')
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')
    parser.add_argument('--embeddings', type=str, default='runs/openai_3_large_test_20251231_024532',
                        help='Path to embeddings directory')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"polytope_crosstask_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("CROSS-TASK POLYTOPE VALIDATION ON REAL PHASE D DATA")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(f"Number of seeds: {args.seeds}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load real Phase D data
    print("Loading real Phase D data...")
    embeddings, labels, boundary_distances = load_phase_d_data(args.embeddings)
    print(f"✓ Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}-D)")
    print(f"  Labels: {len(np.unique(labels))} classes (Presence/Friction)")
    print(f"  Boundary distances: [{boundary_distances.min():.3f}, {boundary_distances.max():.3f}]")

    # Compute local curvature
    print("\nComputing local curvature...")
    curvatures = compute_local_curvature(embeddings, k=32, seed=42)
    print(f"✓ Curvature: [{curvatures.min():.3f}, {curvatures.max():.3f}]")

    # Define tasks
    tasks = [
        {"name": "boundary_distance", "targets": boundary_distances, "type": "regression", "metric": "r2"},
        {"name": "friction_classification", "targets": labels, "type": "classification", "metric": "auc"},
        {"name": "curvature", "targets": curvatures, "type": "regression", "metric": "r2"},
    ]

    # Define seeds
    seed_list = [17, 42, 100, 200, 333, 500, 666, 777, 888, 999][:args.seeds]

    print(f"\nTesting seeds: {seed_list}")
    print(f"Tasks: {[t['name'] for t in tasks]}\n")
    print(f"{'='*80}\n")

    # Run experiments
    all_results = {}

    for task in tasks:
        print(f"{'='*80}")
        print(f"TASK: {task['name'].upper()}")
        print(f"{'='*80}\n")

        task_results = []
        for seed in seed_list:
            result = run_single_task_seed(
                seed, embeddings, task['targets'], task['type'],
                device, verbose=args.verbose
            )
            task_results.append(result)

            if not args.verbose:
                print(f"  Seed {seed:3d}: {result['winner']:10s} "
                      f"({task['metric']}={result['winner_score']:+.3f}, "
                      f"Δ={result['winner_delta']:+.3f})")

        all_results[task['name']] = task_results
        print()

    # Aggregate analysis
    print(f"{'='*80}")
    print("CROSS-TASK AGGREGATE ANALYSIS")
    print(f"{'='*80}\n")

    summary = {}

    for task in tasks:
        task_name = task['name']
        task_data = all_results[task_name]

        print(f"Task: {task_name}")
        print(f"{'-'*60}")

        # Count wins per polytope
        win_counts = {}
        for result in task_data:
            winner = result['winner']
            win_counts[winner] = win_counts.get(winner, 0) + 1

        print(f"\nWin counts (out of {args.seeds} seeds):")
        for name, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / args.seeds
            print(f"  {name:15s}: {count:2d} wins ({percentage:5.1f}%)")

        # Find most consistent winner
        most_common_winner = max(win_counts.items(), key=lambda x: x[1])
        winner_name = most_common_winner[0]
        winner_percentage = 100 * most_common_winner[1] / args.seeds

        # Determine verdict
        if winner_percentage >= 80:
            verdict = "ROBUST"
        elif winner_percentage >= 60:
            verdict = "MODERATE"
        else:
            verdict = "UNSTABLE"

        summary[task_name] = {
            "winner": winner_name,
            "win_rate": winner_percentage,
            "verdict": verdict,
            "win_counts": win_counts,
        }

        print(f"\nVerdict for {task_name}: {verdict}")
        print(f"  Winner: {winner_name} ({winner_percentage:.1f}% win rate)\n")

    # Cross-task consistency analysis
    print(f"{'='*80}")
    print("CROSS-TASK CONSISTENCY")
    print(f"{'='*80}\n")

    # Check if same polytope wins across tasks
    winners = [summary[task['name']]['winner'] for task in tasks]

    if len(set(winners)) == 1:
        print(f"✓ UNIVERSAL WINNER: {winners[0]} wins on ALL tasks")
        print(f"  → Geometric prior generalizes across real geometry tasks")
        overall_verdict = "UNIVERSAL"
    else:
        print(f"⚠️  TASK-DEPENDENT: Different polytopes win on different tasks")
        for task in tasks:
            print(f"  {task['name']:25s}: {summary[task['name']]['winner']}")
        print(f"  → Geometry helps but optimal structure varies by task")
        overall_verdict = "TASK_DEPENDENT"

    # Save results
    final_summary = {
        "timestamp": timestamp,
        "num_seeds": args.seeds,
        "seeds": seed_list,
        "overall_verdict": overall_verdict,
        "task_summaries": summary,
        "all_results": {
            task['name']: [
                {
                    "seed": r['seed'],
                    "winner": r['winner'],
                    "winner_score": float(r['winner_score']),
                    "winner_delta": float(r['winner_delta']),
                }
                for r in all_results[task['name']]
            ]
            for task in tasks
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n✓ Results saved to {summary_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
