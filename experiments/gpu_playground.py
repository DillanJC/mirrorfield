import time
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    num_samples: int = 10_000
    input_dim: int = 32
    hidden_dim: int = 64
    steps: int = 160
    batch_size: int = 256
    lr: float = 1e-3


def make_synthetic_regression(cfg: Config, device: torch.device):
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


def train(cfg: Config, device: torch.device):
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available, training on CPU.")

    model = build_model(cfg).to(device)
    inputs, targets = make_synthetic_regression(cfg, device)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    train(cfg, device)


if __name__ == "__main__":
    main()
