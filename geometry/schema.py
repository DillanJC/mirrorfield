"""Phase E Geometry Schema — Frozen Contract v0.1

This module defines the canonical field names for geometry features.
These names are locked to prevent drift across experiments and refactors.

Design principle: Keep this minimal and stable to avoid refactor tax.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionFNames:
    """Canonical field names for Phase E geometry bundle.

    Attributes:
        schema_version: Version identifier for contract stability
        PROJ: Projected embedding (if using dimensionality reduction)
        D_CENTROID: Distance to reference set centroid
        D_NEAREST: Distance to nearest reference point
        CURVATURE: Local curvature (residual energy beyond top-r components)
                   Gemini alias: JC (Jitter Curvature)
        RIDGE: Ridge proximity (density gradient proxy)
               Gemini alias: SD (Separatrix Density)
        FLAGS: Geometry-derived flags (dark_river_candidate, etc.)
    """
    schema_version: str = "F-0.1-draft"
    PROJ: str = "proj_emb"
    D_CENTROID: str = "dist_to_ref_mean"
    D_NEAREST: str = "dist_to_ref_nearest"
    CURVATURE: str = "local_curvature"
    RIDGE: str = "ridge_proximity"
    FLAGS: str = "geom_flags"


# Singleton instance for import convenience
NAMES = SectionFNames()
