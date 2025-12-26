import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

DEFAULT_SEED = 42


def get_git_info():
    """Get git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()

        dirty_check = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL
        )
        dirty = "dirty" if dirty_check != 0 else "clean"

        return {"commit": commit[:12], "status": dirty}
    except Exception:
        return {"commit": "unknown", "status": "unknown"}


def get_environment_snapshot(device: torch.device):
    """Capture environment details for reproducibility."""
    env = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_type": device.type,
    }

    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        env["tf32_enabled"] = torch.backends.cuda.matmul.allow_tf32

    env["git"] = get_git_info()

    return env


def build_grid_edges() -> List[Tuple[int, int]]:
    """Return undirected neighbor pairs for a 4x4 grid."""
    edges: List[Tuple[int, int]] = []
    for row in range(4):
        for col in range(4):
            node = row * 4 + col
            if col < 3:
                right = node + 1
                edges.append((node, right))
                edges.append((right, node))
            if row < 3:
                down = node + 4
                edges.append((node, down))
                edges.append((down, node))
    return edges


def make_graph(device: torch.device, seed: int = DEFAULT_SEED):
    edges = build_grid_edges()
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    torch.manual_seed(seed)
    x = torch.randn(16, 8, device=device)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2], device=device)
    return edge_index, x, labels


class ToyGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        # Sum neighbor features for each destination node.
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        combined = torch.cat([x, agg], dim=-1)
        return self.mlp(combined)


class ToyGraphNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ToyGNNLayer(8, 16)
        self.layer2 = ToyGNNLayer(16, 16)
        self.head = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, edge_index)
        x = torch.relu(x)
        x = self.layer2(x, edge_index)
        x = torch.relu(x)
        return self.head(x)


def train(device: torch.device, seed: int = DEFAULT_SEED):
    print(f"Using device: {device}")

    edge_index, x, labels = make_graph(device, seed=seed)
    model = ToyGraphNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(1, 201):
        # Forward, loss, and backward pass on the full graph.
        logits = model(x, edge_index)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print(f"Step {step:03d}/200 - loss: {loss.item():.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        # Report prediction histogram for a quick sanity check.
        final_logits = model(x, edge_index)
        preds = final_logits.argmax(dim=-1)
        counts = torch.bincount(preds, minlength=3)

    print(f"Final loss: {loss.item():.6f}")
    print(f"Total training time: {elapsed:.3f} seconds")
    print("Predicted class counts:")
    for cls, cnt in enumerate(counts.tolist()):
        print(f"  class {cls}: {cnt}")

    return loss.item(), elapsed, counts.tolist()


def save_summary(final_loss: float, elapsed: float, class_counts: list, env: dict, seed: int, path: Path):
    """Save run summary with config and environment snapshot."""
    payload = {
        "run_id": path.parent.name,
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "config": {
            "graph_size": "4x4_grid",
            "num_nodes": 16,
            "node_features": 8,
            "num_classes": 3,
            "steps": 200,
            "lr": 1e-2,
            "seed": seed,
        },
        "training": {
            "final_loss": final_loss,
            "elapsed_seconds": elapsed,
        },
        "results": {
            "predicted_class_counts": class_counts,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Summary saved to {path.as_posix()}")


def main():
    seed = DEFAULT_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/toy_graph") / run_id
    summary_path = results_dir / "summary.json"

    print(f"Seed: {seed}")
    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")

    # Capture environment
    env = get_environment_snapshot(device)

    # Train and save results
    final_loss, elapsed, class_counts = train(device, seed=seed)
    save_summary(final_loss, elapsed, class_counts, env, seed, summary_path)


if __name__ == "__main__":
    main()
