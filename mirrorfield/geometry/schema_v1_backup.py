"""
Geometry Bundle Schema — Phase E Contract (Algebra v1)

Frozen dataclass defining output structure for geometric features.
Single source of truth for output keys.

Design principles:
- Schema version tracking for forward compatibility
- NAME_MAP for ParallaxChain (swappable keys without refactor)
- No literal output-key strings elsewhere in codebase
- All records include _schema_version
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

__version__ = "1.0.0"

# Schema version for this output format
SCHEMA_VERSION = "geometry_v1.0"


@dataclass(frozen=True)
class GeometryOutputSchema:
    """
    Frozen contract for geometry bundle outputs.

    All geometry computations must produce records matching this schema.
    """

    # === Core Distance Features ===
    dist_to_ref_mean: float
    """Mean distance from query point to reference set (Centroid Anchor / CA)"""

    dist_to_ref_nearest: float
    """Distance to nearest reference point (1-NN distance)"""

    # === k-NN Statistics (k=50 default) ===
    knn_mean_distance: float
    """Mean distance to k nearest neighbors"""

    knn_std_distance: float
    """Standard deviation of k-NN distances (local uniformity)"""

    knn_min_distance: float
    """Minimum k-NN distance (already captured in dist_to_ref_nearest, but kept for completeness)"""

    knn_max_distance: float
    """Maximum k-NN distance (neighborhood extent)"""

    # === Geometric Structure Features ===
    local_curvature: float
    """
    Local manifold curvature proxy (Jitter Curvature / JC)

    Computed as eigenvalue ratio: λ_min / λ_max of local covariance
    Low values (< 0.5) indicate anisotropic neighborhoods (manifold stretching)
    """

    ridge_proximity: float
    """
    Density gradient proxy / ridge detection (Separatrix Density / SD)

    Computed as: σ_dist / μ_dist (coefficient of variation)
    High values (> 2.0) indicate rapid density changes
    """

    # === Flags (Phase State Markers) ===
    dark_river_candidate: bool
    """
    Dark River hypothesis: low curvature + high ridge

    Formula: JC < 0.5 AND SD > 2.0
    Indicates regions near decision boundary with high geometric instability
    """

    observer_mode: bool
    """
    Observer mode: low curvature + close to reference centroid

    Formula: JC < 1.0 AND CA < 1.5 * ref_std
    Indicates regions where model is uncertain but data is "normal"
    """

    # === Metadata (Artifact Discipline) ===
    _schema_version: str
    """Schema version for forward compatibility"""

    _ref_hash: str
    """Hash of reference set used (for reproducibility)"""

    _config_hash: str
    """Hash of configuration (k, distance metric, etc.)"""

    k_neighbors: int
    """Number of neighbors used for k-NN statistics"""


# ParallaxChain: NAME_MAP adapter for swappable keys
NAME_MAP = {
    # Canonical name → Alias (if you want to rename without refactoring)
    "dist_to_ref_mean": "centroid_anchor",
    "dist_to_ref_nearest": "nearest_neighbor_dist",
    "local_curvature": "jitter_curvature",
    "ridge_proximity": "separatrix_density",
    "dark_river_candidate": "dark_river_flag",
}


def validate_output(record: Dict[str, Any]) -> bool:
    """
    Validate that a record conforms to GeometryOutputSchema.

    Args:
        record: Dictionary to validate

    Returns:
        True if valid, raises ValueError if not
    """
    required_fields = [
        'dist_to_ref_mean', 'dist_to_ref_nearest',
        'knn_mean_distance', 'knn_std_distance',
        'knn_min_distance', 'knn_max_distance',
        'local_curvature', 'ridge_proximity',
        'dark_river_candidate', 'observer_mode',
        '_schema_version', '_ref_hash', '_config_hash', 'k_neighbors'
    ]

    missing = [f for f in required_fields if f not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Type checks
    float_fields = [
        'dist_to_ref_mean', 'dist_to_ref_nearest',
        'knn_mean_distance', 'knn_std_distance',
        'knn_min_distance', 'knn_max_distance',
        'local_curvature', 'ridge_proximity'
    ]

    for field in float_fields:
        if not isinstance(record[field], (float, np.floating)):
            raise ValueError(f"{field} must be float, got {type(record[field])}")

    # Boolean flags
    if not isinstance(record['dark_river_candidate'], (bool, np.bool_)):
        raise ValueError(f"dark_river_candidate must be bool")
    if not isinstance(record['observer_mode'], (bool, np.bool_)):
        raise ValueError(f"observer_mode must be bool")

    # Schema version
    if record['_schema_version'] != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: {record['_schema_version']} != {SCHEMA_VERSION}")

    return True


def create_record(
    dist_to_ref_mean: float,
    dist_to_ref_nearest: float,
    knn_mean_distance: float,
    knn_std_distance: float,
    knn_min_distance: float,
    knn_max_distance: float,
    local_curvature: float,
    ridge_proximity: float,
    dark_river_candidate: bool,
    observer_mode: bool,
    ref_hash: str,
    config_hash: str,
    k_neighbors: int
) -> Dict[str, Any]:
    """
    Create a validated geometry output record.

    This is the canonical way to create geometry outputs.
    Ensures schema compliance.
    """
    record = {
        'dist_to_ref_mean': float(dist_to_ref_mean),
        'dist_to_ref_nearest': float(dist_to_ref_nearest),
        'knn_mean_distance': float(knn_mean_distance),
        'knn_std_distance': float(knn_std_distance),
        'knn_min_distance': float(knn_min_distance),
        'knn_max_distance': float(knn_max_distance),
        'local_curvature': float(local_curvature),
        'ridge_proximity': float(ridge_proximity),
        'dark_river_candidate': bool(dark_river_candidate),
        'observer_mode': bool(observer_mode),
        '_schema_version': SCHEMA_VERSION,
        '_ref_hash': ref_hash,
        '_config_hash': config_hash,
        'k_neighbors': int(k_neighbors)
    }

    validate_output(record)
    return record
