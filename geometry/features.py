"""Phase E Geometry Features — GPU-Batched Low-Rank SVD Curvature + Ridge Proximity

This module implements the core geometric feature extraction:
1. Local curvature via GPU-batched SVD (NOT covariance eigendecomp)
2. Ridge proximity as density gradient proxy

Key Mathematical Identity:
For centered X ∈ ℝ^(k×D), the covariance eigenvalues are proportional to s_i^2
where s_i are singular values of X (returned in descending order by torch.linalg.svdvals).

Non-negotiables:
- Batch independence: all kNN queries against fixed reference set
- Low-rank correct: use SVD on k×D neighborhoods, never D×D covariance
- Local RNG only: no global seed calls in runtime paths
"""

from __future__ import annotations
import numpy as np
import torch


def local_curvature_svd_gpu(
    ref_points: np.ndarray,
    neighbor_indices: np.ndarray,
    r: int = 4,
    batch_size: int = 256,
    device: str = "cuda",
    eps: float = 1e-8,
    use_amp: bool = False,
) -> np.ndarray:
    """Compute local curvature as residual energy beyond top-r principal components.

    This function uses GPU-batched low-rank SVD to measure how much variance in a
    local neighborhood cannot be explained by the top r principal components.
    Higher curvature = more complex local geometry.

    Mathematical approach:
    1. For each query point's k-nearest neighbors in embedding space
    2. Center the neighborhood: X_centered = X - mean(X)
    3. Compute singular values: s = svdvals(X_centered)  [descending order]
    4. Compute energy: e_i = s_i^2
    5. Residual energy: sum(e[r:]) / sum(e)

    Args:
        ref_points: (N_ref, D) float32 array of reference embeddings
        neighbor_indices: (N_query, k) int64 array of indices into ref_points
        r: number of principal components to keep (must be < k)
        batch_size: number of queries to process per GPU batch
        device: 'cuda' or 'cpu' (GPU strongly recommended for performance)
        eps: small constant to prevent division by zero
        use_amp: enable automatic mixed precision (disable if SVD precision issues occur)

    Returns:
        curvature: (N_query,) float32 array of local curvature values in [0, 1]
                   0 = perfectly flat (all variance in top-r components)
                   1 = maximally curved (no variance in top-r components)

    Implementation notes:
    - Uses torch.linalg.svdvals (singular values only, no U/V computation)
    - Processes queries in batches to manage GPU memory
    - Returns to CPU after each batch and clears GPU cache
    - Thread-safe: no global state modification
    """
    assert neighbor_indices.ndim == 2, "neighbor_indices must be (N_query, k)"
    n_query, k = neighbor_indices.shape
    assert 0 <= r < k, f"r must be < k (got r={r}, k={k})"

    # Setup device
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    ref_t = torch.from_numpy(ref_points).to(device=dev, dtype=torch.float32)
    idx_t = torch.from_numpy(neighbor_indices).to(device=dev, dtype=torch.long)

    # Preallocate output
    out = np.empty((n_query,), dtype=np.float32)

    # Setup autocast context if available and requested
    autocast_ctx = torch.amp.autocast if hasattr(torch, "amp") else None
    use_autocast = bool(use_amp and dev.type == "cuda" and autocast_ctx is not None)

    with torch.no_grad():
        for start in range(0, n_query, batch_size):
            end = min(n_query, start + batch_size)
            batch_idx = idx_t[start:end]              # (B, k)
            X = ref_t[batch_idx]                      # (B, k, D)
            Xc = X - X.mean(dim=1, keepdim=True)      # center each neighborhood

            # Compute singular values (descending order)
            if use_autocast:
                with autocast_ctx("cuda"):
                    s = torch.linalg.svdvals(Xc)      # (B, k)
                    e = s * s                          # energy = singular_value^2
            else:
                s = torch.linalg.svdvals(Xc)          # (B, k)
                e = s * s

            # Compute residual energy fraction
            total = e.sum(dim=1)                      # (B,)
            residual = e[:, r:].sum(dim=1)            # energy beyond top-r components
            curv = (residual / (total + eps)).to(torch.float32)

            # Copy to output
            out[start:end] = curv.detach().cpu().numpy()

            # Clear GPU cache to prevent OOM on long runs
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    return out


def ridge_proximity_from_knn_distances(
    distances: np.ndarray,
    inner_index: int = 15,  # 16th nearest neighbor (0-indexed)
    outer_index: int = 31,  # 32nd nearest neighbor (0-indexed)
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute ridge proximity as density gradient proxy.

    Ridge proximity measures the ratio of outer-shell to inner-shell radius
    in k-nearest-neighbor space. High values indicate the point is near a
    density discontinuity (separatrix / decision boundary ridge).

    Mathematical interpretation:
    - r_inner = radius to 16th NN (inner shell)
    - r_outer = radius to 32nd NN (outer shell)
    - ridge = r_outer / r_inner

    Higher ridge values indicate:
    - Point is near a region where density drops sharply
    - Likely near decision boundary or mode boundary
    - "Architect mode" territory in Dark River taxonomy

    Index clarification (CRITICAL):
    With n_neighbors=32, sklearn's kneighbors returns distances shaped (N, 32):
    - distances[:, 0]  = 1st nearest neighbor
    - distances[:, 15] = 16th nearest neighbor (inner_index=15, 0-indexed)
    - distances[:, 31] = 32nd nearest neighbor (outer_index=31, 0-indexed)

    Args:
        distances: (N, k) float array from kneighbors output
        inner_index: 0-indexed position for inner shell radius (default: 15 → 16th NN)
        outer_index: 0-indexed position for outer shell radius (default: 31 → 32nd NN)
        eps: small constant to prevent division by zero

    Returns:
        ridge: (N,) float array of ridge proximity values
               Typical range: [1.0, 3.0]
               Values > 2.0 indicate strong density gradients
    """
    assert distances.ndim == 2 and distances.shape[1] > outer_index, \
        f"distances must be (N, k) with k > {outer_index}, got shape {distances.shape}"

    r_inner = distances[:, inner_index]  # radius to 16th NN
    r_outer = distances[:, outer_index]  # radius to 32nd NN

    return r_outer / (r_inner + eps)
