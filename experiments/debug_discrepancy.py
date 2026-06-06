"""
Debug Discrepancy Between Boundary-Sliced and Bootstrap

Saves detailed artifacts to compare methodologies.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json

# Set seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
embeddings = np.load(data_path / "embeddings.npy")
boundary_distances = np.load(data_path / "boundary_distances.npy")

print(f"Total samples: {len(embeddings)}")
print(f"Seed: {SEED}")
print(f"Device: {device}\n")

# Split data (EXACTLY as both scripts do)
indices = np.arange(len(embeddings))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
train_idx_inner, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED)

print(f"Train samples: {len(train_idx_inner)}")
print(f"Val samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}\n")

# Identify borderline in test set
test_bd = boundary_distances[test_idx]
borderline_mask = (test_bd >= -0.5) & (test_bd <= 0.5)
borderline_count = borderline_mask.sum()

print(f"Borderline in test set: {borderline_count}")
print(f"Borderline indices (first 10): {np.where(borderline_mask)[0][:10]}")
print(f"Borderline boundary_distances (first 10): {test_bd[borderline_mask][:10]}\n")

# Save artifacts
output_dir = Path(f"C:/Users/User/mirrorfield/runs/debug_seed{SEED}")
output_dir.mkdir(parents=True, exist_ok=True)

artifacts = {
    'seed': int(SEED),
    'train_idx': train_idx_inner.tolist(),
    'val_idx': val_idx.tolist(),
    'test_idx': test_idx.tolist(),
    'borderline_mask': borderline_mask.tolist(),
    'borderline_count': int(borderline_count),
    'test_boundary_distances': test_bd.tolist(),
    'borderline_bd_values': test_bd[borderline_mask].tolist()
}

with open(output_dir / "split_artifacts.json", 'w') as f:
    json.dump(artifacts, f, indent=2)

print(f"Saved artifacts to: {output_dir}")
print("\nFirst 5 test samples:")
print(f"  Index | Boundary Distance | In Borderline?")
for i in range(5):
    print(f"  {test_idx[i]:<5} | {test_bd[i]:+.4f} | {borderline_mask[i]}")
