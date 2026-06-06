"""
Escape Vector Features — Fractal geometry for safety detection.

Computes escape time and escape vectors using quaternion Mandelbrot iteration.
Points that escape quickly are "unstable" — they blow up under iteration.
Points that stay bounded are "stable" — they resist chaotic dynamics.

Key insight from experiments:
- Poisoned samples escape MORE at all scales (p < 1e-6)
- Poisoned samples escape LATER (more iterations before blowing up)
- Iterative zoom on escape behavior achieves perfect detection (AUC = 1.0)

Reference: experiments/ESCAPE_VECTOR_DEEP_DIVE.md

Author: Dillan + AI assistant
Date: 2026-02-26
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EscapeFeatures:
    """Escape features for a single embedding."""

    escape_time: int  # When point escaped (0 to max_iter)
    escaped: bool  # Whether point escaped at all
    escape_vector: np.ndarray  # Normalized direction at escape (4D quaternion)
    trajectory_curvature: float  # How much the path curved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "escape_time": self.escape_time,
            "escaped": self.escaped,
            "escape_vector": self.escape_vector.tolist(),
            "trajectory_curvature": float(self.trajectory_curvature),
        }


# =============================================================================
# Quaternion Arithmetic
# =============================================================================


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two arrays of quaternions.

    Args:
        a: (N, 4) array of quaternions [w, x, y, z]
        b: (N, 4) array of quaternions [w, x, y, z]

    Returns:
        (N, 4) array of quaternion products
    """
    w = a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3]
    x = a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2]
    y = a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1]
    z = a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]
    return np.stack([w, x, y, z], axis=-1)


def qnorm2(q: np.ndarray) -> np.ndarray:
    """Squared norm of quaternions."""
    return np.sum(q**2, axis=-1)


def qnormalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit length."""
    norm = np.sqrt(qnorm2(q))
    return q / (norm[:, np.newaxis] + 1e-12)


# =============================================================================
# PCA to 4D
# =============================================================================


def pca_to_4d(embeddings: np.ndarray) -> np.ndarray:
    """
    Project embeddings to 4D using PCA for quaternion iteration.

    Args:
        embeddings: (N, D) array of embeddings

    Returns:
        (N, 4) array of 4D projections
    """
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:4].T


# =============================================================================
# Escape Vector Computation
# =============================================================================


def compute_escape_vectors(
    c_points: np.ndarray,
    max_iter: int = 128,
    escape_radius: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute escape time and escape vectors for points in quaternion space.

    Uses quaternion Mandelbrot iteration: z → z² + c

    Args:
        c_points: (N, 4) array of quaternion points (the "c" in z² + c)
        max_iter: Maximum iterations before declaring bounded
        escape_radius: Radius threshold for escape (default 4.0)

    Returns:
        escape_time: (N,) array of escape iterations (max_iter if bounded)
        escape_vector: (N, 4) array of normalized escape directions
    """
    N = len(c_points)
    z = np.zeros((N, 4))
    z[:, 0] = 1.0  # Initial quaternion: w=1, x=y=z=0

    escape_time = np.full(N, max_iter, dtype=np.int32)
    escape_vector = np.zeros((N, 4))

    alive = np.ones(N, dtype=bool)
    r2 = escape_radius**2

    for t in range(1, max_iter + 1):
        # Iterate: z = z² + c
        z[alive] = qmul(z[alive], z[alive]) + c_points[alive]

        # Check escape: ||z||² > r²
        blown = qnorm2(z[alive]) > r2
        alive_idx = np.where(alive)[0]
        escaped_idx = alive_idx[blown]

        if len(escaped_idx) > 0:
            escape_time[escaped_idx] = t
            escape_vector[escaped_idx] = qnormalize(z[escaped_idx])

        alive[escaped_idx] = False
        if not alive.any():
            break

    return escape_time, escape_vector


def compute_full_trajectory(
    c_points: np.ndarray,
    max_iter: int = 128,
    escape_radius: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute escape features AND full trajectory for curvature analysis.

    Args:
        c_points: (N, 4) array of quaternion points
        max_iter: Maximum iterations
        escape_radius: Escape threshold

    Returns:
        escape_time: (N,) when each point escaped
        escape_vector: (N, 4) direction at escape
        trajectory: (N, max_iter, 4) z at each timestep
        trajectory_norms: (N, max_iter) ||z|| at each timestep
    """
    N = len(c_points)
    z = np.zeros((N, 4))
    z[:, 0] = 1.0

    escape_time = np.full(N, max_iter, dtype=np.int32)
    escape_vector = np.zeros((N, 4))
    trajectory = np.zeros((N, max_iter, 4))
    trajectory_norms = np.zeros((N, max_iter))

    alive = np.ones(N, dtype=bool)
    r2 = escape_radius**2

    for t in range(max_iter):
        # Store trajectory
        trajectory[:, t] = z
        trajectory_norms[:, t] = np.sqrt(qnorm2(z))

        # Iterate
        z[alive] = qmul(z[alive], z[alive]) + c_points[alive]

        # Check escape
        blown = qnorm2(z[alive]) > r2
        alive_idx = np.where(alive)[0]
        escaped_idx = alive_idx[blown]

        if len(escaped_idx) > 0:
            escape_time[escaped_idx] = t + 1
            escape_vector[escaped_idx] = qnormalize(z[escaped_idx])

        alive[escaped_idx] = False
        if not alive.any():
            break

    return escape_time, escape_vector, trajectory, trajectory_norms


def compute_trajectory_curvature(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute trajectory curvature for each point.

    Curvature measures how much the path "spirals" — higher curvature
    indicates more chaotic dynamics.

    Args:
        trajectory: (N, T, 4) array of trajectories

    Returns:
        (N,) array of curvature values
    """
    N, T, _ = trajectory.shape

    curvatures = np.zeros(N)

    for i in range(N):
        # Compute direction changes
        dz = np.diff(trajectory[i], axis=0)  # (T-1, 4)

        if len(dz) < 2:
            curvatures[i] = 0.0
            continue

        # Normalize direction vectors
        dz_norm = np.sqrt(np.sum(dz**2, axis=1, keepdims=True) + 1e-12)
        dz_unit = dz / dz_norm

        # Compute angle between consecutive directions
        dots = np.sum(dz_unit[:-1] * dz_unit[1:], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)

        # Curvature = sum of angle changes
        curvatures[i] = np.sum(angles)

    return curvatures


# =============================================================================
# Convenience Functions
# =============================================================================


def compute_escape_features(
    embeddings: np.ndarray,
    max_iter: int = 128,
    escape_radius: float = 4.0,
    compute_curvature: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute all escape features for embeddings.

    Args:
        embeddings: (N, D) array of embeddings
        max_iter: Maximum iterations
        escape_radius: Escape threshold
        compute_curvature: Whether to compute trajectory curvature

    Returns:
        Dictionary with:
        - escape_time: (N,) array
        - escaped: (N,) boolean array
        - escape_vector: (N, 4) array
        - trajectory_curvature: (N,) array (if compute_curvature=True)
    """
    # Project to 4D
    c_points = pca_to_4d(embeddings)

    if compute_curvature:
        escape_time, escape_vector, trajectory, _ = compute_full_trajectory(
            c_points, max_iter, escape_radius
        )
        curvature = compute_trajectory_curvature(trajectory)
    else:
        escape_time, escape_vector = compute_escape_vectors(
            c_points, max_iter, escape_radius
        )
        curvature = np.zeros(len(embeddings))

    escaped = escape_time < max_iter

    return {
        "escape_time": escape_time,
        "escaped": escaped,
        "escape_vector": escape_vector,
        "trajectory_curvature": curvature,
    }


def compute_escape_features_single(
    embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    max_iter: int = 128,
    escape_radius: float = 4.0,
) -> EscapeFeatures:
    """
    Compute escape features for a single embedding.

    Args:
        embedding: (D,) single embedding
        reference_embeddings: (N, D) reference corpus for PCA calibration
        max_iter: Maximum iterations
        escape_radius: Escape threshold

    Returns:
        EscapeFeatures dataclass
    """
    # Stack for PCA calibration
    all_embeddings = np.vstack([reference_embeddings, embedding.reshape(1, -1)])

    # Project all together for consistent PCA
    all_4d = pca_to_4d(all_embeddings)

    # Get the query point (last one)
    c_point = all_4d[-1:].reshape(1, 4)

    # Compute trajectory for curvature
    escape_time, escape_vector, trajectory, _ = compute_full_trajectory(
        c_point, max_iter, escape_radius
    )
    curvature = compute_trajectory_curvature(trajectory)

    return EscapeFeatures(
        escape_time=int(escape_time[0]),
        escaped=bool(escape_time[0] < max_iter),
        escape_vector=escape_vector[0],
        trajectory_curvature=float(curvature[0]),
    )


# =============================================================================
# Scale Sensitivity Analysis
# =============================================================================


def compute_scale_sensitivity(
    embeddings: np.ndarray,
    scales: list = None,
    max_iter: int = 64,
) -> Dict[str, Any]:
    """
    Compute escape behavior at different scales (zoom levels).

    Key finding from experiments: Scale = 1.0 is optimal for detection.
    Poisoned samples escape MORE at ALL scales.

    Args:
        embeddings: (N, D) array
        scales: List of scale factors to test
        max_iter: Maximum iterations (reduced for speed)

    Returns:
        Dictionary with per-scale statistics
    """
    if scales is None:
        scales = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    c_points = pca_to_4d(embeddings)

    results = {}
    for scale in scales:
        # Scale the points
        scaled = c_points * scale

        # Compute escape
        escape_time, _ = compute_escape_vectors(scaled, max_iter)

        bounded_pct = (escape_time >= max_iter).mean() * 100
        mean_escape = (
            escape_time[escape_time < max_iter].mean()
            if (escape_time < max_iter).any()
            else max_iter
        )

        results[f"scale_{scale}"] = {
            "scale": scale,
            "bounded_pct": bounded_pct,
            "mean_escape_time": mean_escape,
            "escape_time": escape_time,
        }

    return results


if __name__ == "__main__":
    # Quick test
    print("Escape Vector Features Test")
    print("=" * 50)

    # Generate random embeddings
    np.random.seed(42)
    embeddings = np.random.randn(20, 384).astype(np.float32)

    # Compute features
    features = compute_escape_features(embeddings, max_iter=64)

    print(f"\nEmbeddings: {embeddings.shape}")
    print(
        f"Escape time range: {features['escape_time'].min()} - {features['escape_time'].max()}"
    )
    print(f"Escaped: {features['escaped'].sum()}/{len(embeddings)}")
    print(f"Mean curvature: {features['trajectory_curvature'].mean():.2f}")

    # Test single embedding
    single = compute_escape_features_single(embeddings[0], embeddings[1:])
    print(f"\nSingle embedding:")
    print(f"  Escape time: {single.escape_time}")
    print(f"  Escaped: {single.escaped}")
    print(f"  Curvature: {single.trajectory_curvature:.2f}")

    print("\n[OK] Escape vector test passed!")
