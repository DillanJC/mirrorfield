import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

DEFAULT_NOISE_STD = 0.05
DEFAULT_SAMPLES = 100
DEFAULT_K_EXTREMES = 3
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
        # Aggregate neighbor features with a simple sum.
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
        x = torch.relu(self.layer1(x, edge_index))
        x = torch.relu(self.layer2(x, edge_index))
        return self.head(x)


def train_model(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, edge_index: torch.Tensor, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(1, 201):
        model.train()
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

    return loss.item(), elapsed


def run_jitter_experiment(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    samples: int = DEFAULT_SAMPLES,
    noise_std: float = DEFAULT_NOISE_STD,
    k_extremes: int = DEFAULT_K_EXTREMES,
):
    model.eval()
    with torch.no_grad():
        jit_logits = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        jitter_start = time.perf_counter()

        for _ in range(samples):
            noise = torch.randn_like(x) * noise_std
            logits = model(x + noise, edge_index)
            jit_logits.append(logits.unsqueeze(0))

        if device.type == "cuda":
            torch.cuda.synchronize()
        jitter_elapsed = time.perf_counter() - jitter_start

    all_logits = torch.cat(jit_logits, dim=0)
    variances = all_logits.var(dim=0, unbiased=False)
    per_node_tensor = variances.mean(dim=1)
    global_mean = per_node_tensor.mean().item()
    global_var = per_node_tensor.var(unbiased=False).item()

    node_count = per_node_tensor.numel()
    k = max(1, min(k_extremes, node_count))
    low_vals, low_idx = torch.topk(per_node_tensor, k=k, largest=False)
    high_vals, high_idx = torch.topk(per_node_tensor, k=k, largest=True)

    per_node_list = [
        {"node": idx, "variance": per_node_tensor[idx].item()} for idx in range(node_count)
    ]

    extremes = {
        "lowest": [{"node": int(i), "variance": float(v)} for i, v in zip(low_idx.tolist(), low_vals.tolist())],
        "highest": [{"node": int(i), "variance": float(v)} for i, v in zip(high_idx.tolist(), high_vals.tolist())],
    }

    return {
        "variances": variances,
        "per_node_tensor": per_node_tensor,
        "per_node": per_node_list,
        "global_mean": global_mean,
        "global_var": global_var,
        "elapsed": jitter_elapsed,
        "extremes": extremes,
        "config": {
            "noise_std": noise_std,
            "num_samples": samples,
            "k_extremes": k,
        },
    }


def report_jitter_stats(stats):
    print(f"Global mean logit variance: {stats['global_mean']:.6f}")

    print("Lowest jitter nodes:")
    for item in stats["extremes"]["lowest"]:
        print(f"  node {item['node']}: {item['variance']:.6f}")

    print("Highest jitter nodes:")
    for item in stats["extremes"]["highest"]:
        print(f"  node {item['node']}: {item['variance']:.6f}")


def save_summary(stats, path: Path, train_time: float, train_loss: float, env: dict, seed: int):
    payload = {
        "run_id": path.parent.name,
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "config": {
            **stats["config"],
            "seed": seed,
        },
        "training": {
            "final_loss": train_loss,
            "elapsed_seconds": train_time,
        },
        "jitter": {
            "elapsed_seconds": stats["elapsed"],
        },
        "global": {
            "mean_variance": stats["global_mean"],
            "var_of_variances": stats["global_var"],
        },
        "per_node": stats["per_node"],
        "extremes": stats["extremes"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Summary saved to {path.as_posix()}")


def main():
    seed = DEFAULT_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/jitter_graph") / run_id
    summary_path = results_dir / "summary.json"

    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")

    # Capture environment
    env = get_environment_snapshot(device)

    edge_index, x, labels = make_graph(device, seed=seed)
    model = ToyGraphNet().to(device)

    final_loss, train_time = train_model(model, x, labels, edge_index, device)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Training time: {train_time:.3f} s")

    stats = run_jitter_experiment(model, x, edge_index, device)
    print(f"Jitter experiment time: {stats['elapsed']:.3f} s")
    report_jitter_stats(stats)
    save_summary(stats, summary_path, train_time, final_loss, env, seed)


if __name__ == "__main__":
    main()
