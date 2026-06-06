"""Direct comparison of seed 42 between methodologies"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import json

SEED = 42

# Model definition
class BaselineModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(1)

class AugmentedModel(nn.Module):
    def __init__(self, input_dim=256, aug_dim=7, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + aug_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2).squeeze(1)

def train_model(model, X_train, y_train, X_val, y_val, device, epochs=100):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = F.mse_loss(preds, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = F.mse_loss(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

    return model

def compute_native_geometry_features(embeddings, k=50, seed=42):
    np.random.seed(seed)
    N = len(embeddings)

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    features = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        features[i, 0] = distances[i].mean()
        features[i, 1] = distances[i].std()
        features[i, 2] = distances[i].min()
        features[i, 3] = distances[i].max()

        neighbors = embeddings[indices[i]]
        centered = neighbors - embeddings[i]
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[-1] > 1e-10:
            features[i, 4] = eigenvalues[0] / eigenvalues[-1]
        else:
            features[i, 4] = 0.0

        features[i, 5] = distances[i].std() / (distances[i].mean() + 1e-6)
        features[i, 6] = distances[i, 0]

    return features

# Main comparison
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# Load data
data_path = Path("C:/Users/User/mirrorfield/runs/openai_3_large_test_20251231_024532")
embeddings = np.load(data_path / "embeddings.npy")
boundary_distances = np.load(data_path / "boundary_distances.npy")

print("="*60)
print("METHOD 1: Original Boundary-Sliced Approach")
print("="*60)

# Set seeds EXACTLY as original does
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Compute geometry features
geo_features_method1 = compute_native_geometry_features(embeddings, k=50, seed=SEED)

# Prepare features
X_with_geo_method1 = np.concatenate([embeddings, geo_features_method1], axis=1)

# Split data
indices = np.arange(len(embeddings))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
train_idx_inner, val_idx = train_test_split(train_idx, test_size=0.2, random_state=SEED)

# Train models
print("\nTraining baseline...")
model_baseline_m1 = BaselineModel()
model_baseline_m1 = train_model(
    model_baseline_m1,
    embeddings[train_idx_inner], boundary_distances[train_idx_inner],
    embeddings[val_idx], boundary_distances[val_idx],
    device
)

print("Training geometry...")
model_geo_m1 = AugmentedModel()
model_geo_m1 = train_model(
    model_geo_m1,
    X_with_geo_method1[train_idx_inner], boundary_distances[train_idx_inner],
    X_with_geo_method1[val_idx], boundary_distances[val_idx],
    device
)

# Get predictions on test set
model_baseline_m1.eval()
model_geo_m1.eval()

with torch.no_grad():
    X_test_t = torch.from_numpy(embeddings[test_idx]).float().to(device)
    preds_baseline_m1 = model_baseline_m1(X_test_t).cpu().numpy()

    X_test_geo_t = torch.from_numpy(X_with_geo_method1[test_idx]).float().to(device)
    preds_geo_m1 = model_geo_m1(X_test_geo_t).cpu().numpy()

# Evaluate on borderline
y_test = boundary_distances[test_idx]
borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)

r2_baseline_m1 = r2_score(y_test[borderline_mask], preds_baseline_m1[borderline_mask])
r2_geo_m1 = r2_score(y_test[borderline_mask], preds_geo_m1[borderline_mask])
delta_m1 = r2_geo_m1 - r2_baseline_m1

print(f"\nResults on borderline ({borderline_mask.sum()} samples):")
print(f"  Baseline R²: {r2_baseline_m1:.4f}")
print(f"  Geometry R²: {r2_geo_m1:.4f}")
print(f"  ΔR²:         {delta_m1:+.4f}")

print("\n" + "="*60)
print("METHOD 2: Bootstrap Approach")
print("="*60)

# Set seeds again
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Compute geometry features
geo_features_method2 = compute_native_geometry_features(embeddings, k=50, seed=SEED)

# Split data (same as method 1)
indices2 = np.arange(len(embeddings))
train_idx2, test_idx2 = train_test_split(indices2, test_size=0.2, random_state=SEED)
train_idx_inner2, val_idx2 = train_test_split(train_idx2, test_size=0.2, random_state=SEED)

# Prepare features AFTER split (bootstrap way)
X_train_geo_m2 = np.concatenate([embeddings[train_idx_inner2], geo_features_method2[train_idx_inner2]], axis=1)
X_val_geo_m2 = np.concatenate([embeddings[val_idx2], geo_features_method2[val_idx2]], axis=1)
X_test_geo_m2 = np.concatenate([embeddings[test_idx2], geo_features_method2[test_idx2]], axis=1)

# Train models
print("\nTraining baseline...")
model_baseline_m2 = BaselineModel()
model_baseline_m2 = train_model(
    model_baseline_m2,
    embeddings[train_idx_inner2], boundary_distances[train_idx_inner2],
    embeddings[val_idx2], boundary_distances[val_idx2],
    device
)

print("Training geometry...")
model_geo_m2 = AugmentedModel()
model_geo_m2 = train_model(
    model_geo_m2,
    X_train_geo_m2, boundary_distances[train_idx_inner2],
    X_val_geo_m2, boundary_distances[val_idx2],
    device
)

# Get predictions on test set
model_baseline_m2.eval()
model_geo_m2.eval()

with torch.no_grad():
    X_test_t2 = torch.from_numpy(embeddings[test_idx2]).float().to(device)
    preds_baseline_m2 = model_baseline_m2(X_test_t2).cpu().numpy()

    X_test_geo_t2 = torch.from_numpy(X_test_geo_m2).float().to(device)
    preds_geo_m2 = model_geo_m2(X_test_geo_t2).cpu().numpy()

# Evaluate on borderline
y_test2 = boundary_distances[test_idx2]
borderline_mask2 = (y_test2 >= -0.5) & (y_test2 <= 0.5)

r2_baseline_m2 = r2_score(y_test2[borderline_mask2], preds_baseline_m2[borderline_mask2])
r2_geo_m2 = r2_score(y_test2[borderline_mask2], preds_geo_m2[borderline_mask2])
delta_m2 = r2_geo_m2 - r2_baseline_m2

print(f"\nResults on borderline ({borderline_mask2.sum()} samples):")
print(f"  Baseline R²: {r2_baseline_m2:.4f}")
print(f"  Geometry R²: {r2_geo_m2:.4f}")
print(f"  ΔR²:         {delta_m2:+.4f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Method 1 ΔR²: {delta_m1:+.4f}")
print(f"Method 2 ΔR²: {delta_m2:+.4f}")
print(f"Difference:   {delta_m2 - delta_m1:+.4f}")

if abs(delta_m2 - delta_m1) < 0.001:
    print("\n✓ Methods MATCH - discrepancy resolved!")
else:
    print(f"\n✗ Methods DIFFER by {abs(delta_m2 - delta_m1):.4f} - issue remains")
