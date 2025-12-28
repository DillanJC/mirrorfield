"""Phase E Geometry Bundle — Transform Engine with Local RNG Discipline

This module provides the GeometryBundle class which orchestrates:
1. kNN lookup against fixed reference set
2. Local curvature computation (GPU-batched SVD)
3. Ridge proximity computation
4. Geometry-derived flags (Dark River candidates, mode markers)

Non-negotiables:
- Batch independence: all queries against fixed reference, never query batch
- Local RNG only: no global np.random.seed() or torch.manual_seed()
- Reproducibility: deterministic per-sample RNG if needed
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, Any, Optional

from geometry.features import (
    local_curvature_svd_gpu,
    ridge_proximity_from_knn_distances,
)
from geometry.schema import NAMES


class GeometryBundle:
    """Geometry feature extraction bundle with local RNG discipline.

    This class encapsulates the geometry computation pipeline while maintaining
    strict reproducibility and batch-independence guarantees.

    Attributes:
        ref_points: (N_ref, D) reference embeddings (fixed)
        nn_index: sklearn NearestNeighbors index (fitted on ref_points)
        config: configuration dict
        base_seed: base random seed for reproducibility
        numpy_rng: local numpy RandomState (not global)
        torch_rng: local torch Generator (not global)
    """

    def __init__(
        self,
        ref_points: np.ndarray,
        nn_index: Any,  # sklearn.neighbors.NearestNeighbors
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize geometry bundle.

        Args:
            ref_points: (N_ref, D) float32 reference embeddings
            nn_index: fitted sklearn NearestNeighbors index
            config: configuration dict with keys:
                - random_seed: base seed (default: 42)
                - n_neighbors: k for kNN (default: 32)
                - k_curvature: k for curvature computation (default: 16)
                - r_components: top-r components to keep (default: 4)
                - batch_size: GPU batch size (default: 256)
                - device: 'cuda' or 'cpu' (default: 'cuda')
                - use_amp: enable mixed precision (default: False)
        """
        self.ref_points = ref_points
        self.nn_index = nn_index
        self.config = dict(config or {})

        # Local RNG discipline: store RNG state locally, never modify global state
        self.base_seed = int(self.config.get("random_seed", 42))
        self.numpy_rng = np.random.RandomState(self.base_seed)
        self.torch_rng = torch.Generator().manual_seed(self.base_seed)

    def _rng_for_sample(self, i: int) -> np.random.RandomState:
        """Create deterministic per-sample RNG.

        This ensures reproducibility even when processing samples in different batches
        or different orderings.

        Args:
            i: sample index

        Returns:
            numpy RandomState seeded with base_seed + i
        """
        return np.random.RandomState(self.base_seed + int(i))

    def transform(
        self,
        query_emb: np.ndarray,
        boundary_distance: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Extract geometry features for query embeddings.

        This is the main entry point for Phase E geometry computation.

        Args:
            query_emb: (N_query, D) float32 query embeddings
            boundary_distance: (N_query,) float32 boundary distances from Phase D
                               (computed on-the-fly or from cached tags)

        Returns:
            dict with keys matching geometry.schema.NAMES:
                - local_curvature: (N_query,) float32
                - ridge_proximity: (N_query,) float32
                - geom_flags: (N_query,) list of lists of flag strings

        Raises:
            AssertionError: if input shapes don't match or config is invalid
        """
        assert query_emb.ndim == 2, f"query_emb must be (N, D), got {query_emb.shape}"
        assert boundary_distance.ndim == 1, f"boundary_distance must be (N,), got {boundary_distance.shape}"
        n_query = len(query_emb)
        assert len(boundary_distance) == n_query, \
            f"Length mismatch: query_emb={n_query}, boundary_distance={len(boundary_distance)}"

        # Extract config
        n_neighbors = self.config.get("n_neighbors", 32)
        k_curv = self.config.get("k_curvature", 16)
        r_components = self.config.get("r_components", 4)
        batch_size = self.config.get("batch_size", 256)
        device = self.config.get("device", "cuda")
        use_amp = self.config.get("use_amp", False)

        # Validate config
        assert r_components < k_curv, \
            f"r_components must be < k_curvature (got r={r_components}, k={k_curv})"
        assert k_curv <= n_neighbors, \
            f"k_curvature must be <= n_neighbors (got k={k_curv}, n={n_neighbors})"

        # 1) kNN against reference set only (batch-independent)
        distances, indices = self.nn_index.kneighbors(query_emb, n_neighbors=n_neighbors)

        # 2) Local curvature via GPU-batched SVD
        # Use first k_curv neighbors for curvature computation
        curv_idx = indices[:, :k_curv]
        curvature = local_curvature_svd_gpu(
            ref_points=self.ref_points,
            neighbor_indices=curv_idx,
            r=r_components,
            batch_size=batch_size,
            device=device,
            use_amp=use_amp,
        )

        # 3) Ridge proximity from full kNN distances
        ridge = ridge_proximity_from_knn_distances(
            distances=distances,
            inner_index=15,  # 16th NN (0-indexed)
            outer_index=31,  # 32nd NN (0-indexed)
        )

        # 4) Geometry-derived flags (Dark River, mode markers)
        flags = self._compute_flags(
            curvature=curvature,
            ridge=ridge,
            boundary_distance=boundary_distance,
        )

        return {
            NAMES.CURVATURE: curvature,
            NAMES.RIDGE: ridge,
            NAMES.FLAGS: flags,
        }

    def _compute_flags(
        self,
        curvature: np.ndarray,
        ridge: np.ndarray,
        boundary_distance: np.ndarray,
    ) -> list:
        """Compute geometry-derived flags for each sample.

        Flags are heuristic markers derived from geometry that can help identify
        interesting regions (Dark River candidates, mode boundaries, etc.).

        Flag taxonomy:
        - dark_river_candidate: Low curvature + high ridge (flat region near boundary)
        - observer_mode: Low curvature + near centroid (stable interior region)
        - architect_mode: High ridge (near decision boundary)

        Args:
            curvature: (N,) local curvature values
            ridge: (N,) ridge proximity values
            boundary_distance: (N,) boundary distances from Phase D

        Returns:
            flags: list of N lists, each containing 0+ flag strings
        """
        # Derive reference statistics for threshold calibration
        # Use boundary_distance std as a scale reference
        ref_std = float(np.std(boundary_distance) + 1e-8)

        flags = []
        JC = curvature  # Gemini alias: Jitter Curvature
        SD = ridge      # Gemini alias: Separatrix Density
        CA = boundary_distance  # Centroid Anchor (in Phase D terms)

        for i in range(len(JC)):
            sample_flags = []

            # Dark River candidate: flat (low curvature) + near ridge (high density gradient)
            if JC[i] < 0.5 and SD[i] > 2.0:
                sample_flags.append("dark_river_candidate")

            # Observer mode: low curvature + near centroid (stable region)
            if JC[i] < 1.0 and CA[i] < 1.5 * ref_std:
                sample_flags.append("observer_mode")

            # Architect mode: high ridge (near decision boundary)
            if SD[i] > 2.0:
                sample_flags.append("architect_mode")

            flags.append(sample_flags)

        return flags

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration parameters.

        Raises:
            AssertionError: if config is invalid
            Warning: if config has risky settings
        """
        r = config.get("r_components", 4)
        k = config.get("k_curvature", 16)
        n = config.get("n_neighbors", 32)
        batch_size = config.get("batch_size", 256)
        seed = config.get("random_seed")

        # Critical constraints
        assert r < k, f"r_components must be < k_curvature (got r={r}, k={k})"
        assert k <= n, f"k_curvature must be <= n_neighbors (got k={k}, n={n})"

        # Warnings
        if seed is None:
            import warnings
            warnings.warn("No random_seed specified in config; using default=42")

        if not (64 <= batch_size <= 512):
            import warnings
            warnings.warn(f"batch_size={batch_size} is outside recommended range [64, 512]")

        # GPU memory warnings (heuristic for 8GB VRAM)
        device = config.get("device", "cuda")
        if device == "cuda" and batch_size > 512:
            import warnings
            warnings.warn(
                f"batch_size={batch_size} may exceed 8GB VRAM for D=768 embeddings. "
                "Consider reducing to 256-512."
            )
