import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


@dataclass
class Config:
    num_samples: int = 10_000
    input_dim: int = 32
    hidden_dim: int = 64
    steps: int = 160
    batch_size: int = 256
    lr: float = 1e-3


def make_synthetic_regression(cfg: Config, device: torch.device, seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    inputs = torch.randn(cfg.num_samples, cfg.input_dim, device=device)
    ground_truth = torch.randn(cfg.input_dim, 1, device=device)
    targets = inputs @ ground_truth + 0.1 * torch.randn(cfg.num_samples, 1, device=device)
    return inputs, targets


def build_model(cfg: Config):
    return nn.Sequential(
        nn.Linear(cfg.input_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, 1),
    )


def train(cfg: Config, device: torch.device, seed: int = DEFAULT_SEED):
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available, training on CPU.")

    model = build_model(cfg).to(device)
    inputs, targets = make_synthetic_regression(cfg, device, seed=seed)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, cfg.num_samples, (cfg.batch_size,), device=device)
        batch_inputs = inputs[idx]
        batch_targets = targets[idx]

        preds = model(batch_inputs)
        loss = criterion(preds, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            print(f"Step {step:03d}/{cfg.steps} - loss: {loss.item():.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Final loss: {loss.item():.6f}")
    print(f"Total training time: {elapsed:.2f} seconds")

    return loss.item(), elapsed


def save_summary(cfg: Config, final_loss: float, elapsed: float, env: dict, seed: int, path: Path):
    """Save run summary with config and environment snapshot."""
    payload = {
        "run_id": path.parent.name,
        "timestamp": datetime.now().isoformat(),
        "environment": env,
        "config": {
            "num_samples": cfg.num_samples,
            "input_dim": cfg.input_dim,
            "hidden_dim": cfg.hidden_dim,
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "seed": seed,
        },
        "training": {
            "final_loss": final_loss,
            "elapsed_seconds": elapsed,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Summary saved to {path.as_posix()}")


def main():
    seed = DEFAULT_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # Generate timestamped run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/gpu_playground") / run_id
    summary_path = results_dir / "summary.json"

    print(f"Seed: {seed}")
    print(f"Run ID: {run_id}")
    print(f"Output: {results_dir.as_posix()}")

    # Capture environment
    env = get_environment_snapshot(device)

    # Train and save results
    final_loss, elapsed = train(cfg, device, seed=seed)
    save_summary(cfg, final_loss, elapsed, env, seed, summary_path)


if __name__ == "__main__":
    main()
