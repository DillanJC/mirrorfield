"""Phase E Geometry Bundle — Core Module

This module provides geometry-based features for boundary evaluation:
- Local curvature (low-rank SVD)
- Ridge proximity (density gradient)
- Geometry scoring and bundling

All computations are batch-independent (reference-set only).
"""

from .schema import NAMES, SectionFNames
from .features import (
    local_curvature_svd_gpu,
    ridge_proximity_from_knn_distances,
)
from .bundle import GeometryBundle

__all__ = [
    "NAMES",
    "SectionFNames",
    "local_curvature_svd_gpu",
    "ridge_proximity_from_knn_distances",
    "GeometryBundle",
]
