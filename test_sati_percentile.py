"""Test Sati percentile integration in mirrorfield."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.neighbors import NearestNeighbors

print("[1/5] Testing sati_schema imports...")

from mirrorfield.geometry.sati_schema import (
    generate_sati_feedback_percentile,
    compute_sati_summary_percentile,
    SATI_TO_SIGNAL,
    sati_to_intervention_signal,
)
from mirrorfield.geometry.escape_vectors import compute_escape_features, pca_to_4d
from mirrorfield.geometry.phase2_weather_features import compute_topology_lite_features

print("  Imports OK")

print("\n[2/5] Creating sample data...")
np.random.seed(42)
N = 100
D = 256
embeddings = np.random.randn(N, D).astype(np.float32)
labels = np.random.randint(0, 2, N)

k = 15
nn = NearestNeighbors(n_neighbors=k + 1)
nn.fit(embeddings)
distances, indices = nn.kneighbors(embeddings, k + 1)
distances = distances[:, 1:]
indices = indices[:, 1:]

knn_min = distances.min(axis=1)
knn_std = distances.std(axis=1)

print(f"  Embeddings: {embeddings.shape}")
print(f"  KNN min range: {knn_min.min():.4f} - {knn_min.max():.4f}")

print("\n[3/5] Computing topology features...")
topology_features, _ = compute_topology_lite_features(embeddings, embeddings, indices)
participation_ratio = topology_features[:, 2]
print(f"  PR range: {participation_ratio.min():.3f} - {participation_ratio.max():.3f}")

print("\n[4/5] Computing local curvature...")
local_curvature = np.zeros(N)
for i in range(N):
    neighbors = embeddings[indices[i]]
    centered = neighbors - neighbors.mean(axis=0)
    if not np.allclose(centered, 0):
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        rank = np.sum(S > 1e-10)
        local_curvature[i] = S[rank - 1] / (S[0] + 1e-12)
    else:
        local_curvature[i] = 1.0

print(f"  Curvature range: {local_curvature.min():.4f} - {local_curvature.max():.4f}")

print("\n[5/5] Computing escape features...")
escape = compute_escape_features(embeddings, max_iter=64)
escape_time = escape["escape_time"]
print(f"  Escape time range: {escape_time.min()} - {escape_time.max()}")

ridge_proximity = np.random.rand(N).astype(np.float32)

feedback_types, confidences, triggered_mask = generate_sati_feedback_percentile(
    local_curvature=local_curvature,
    knn_min_distance=knn_min,
    knn_std_distance=knn_std,
    ridge_proximity=ridge_proximity,
    participation_ratio=participation_ratio,
    escape_time=escape_time,
)

print(f"\n  Triggered: {triggered_mask.sum()}/{N}")
print(f"  Feedback types: {np.unique(feedback_types)}")

summary = compute_sati_summary_percentile(feedback_types, confidences, labels)
print("\n[6/5] Summary by class:")
print(f"  Clean (0): {summary['by_class'][0]}")
print(f"  Poisoned (1): {summary['by_class'][1]}")

print("\n[7/5] Signal mapping:")
for ftype in ["compressed", "resonant", "dissonant", "derivative"]:
    signal = sati_to_intervention_signal(ftype)
    print(f"  {ftype} -> {signal}")

print("\n=== ALL TESTS PASSED ===")
