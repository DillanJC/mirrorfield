import time
from typing import List, Tuple

import torch
from torch import nn


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


def make_graph(device: torch.device):
    edges = build_grid_edges()
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    torch.manual_seed(42)
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


def train(device: torch.device):
    print(f"Using device: {device}")

    edge_index, x, labels = make_graph(device)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)


if __name__ == "__main__":
    main()
