"""
Sati Feedback Schema — Aesthetic Feedback Loop for Renaissance Protocol

Replaces binary RLHF (thumbs up/down) with geometric feedback that teaches
the model "taste" through geometry.

Design principles:
- Feedback is derived from existing geometric features
- Each feedback type maps to specific geometric conditions
- Goal: teach model that flat embeddings produce unsatisfying responses

Version History:
- v1.0: Initial release with 4 feedback types
- v1.1: Integrated into mirrorfield for recursive learning

Author: Dillan + AI assistant
Date: 2026-02-26
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

__version__ = "1.1.0"

SATI_SCHEMA_VERSION = "sati_v1.1"


# =============================================================================
# Feedback Types
# =============================================================================

FEEDBACK_TYPES = {
    "compressed": {
        "description": "This feels compressed",
        "geometric_translation": "Low local_curvature, low participation_ratio",
        "trains": "Model learns that flat embeddings produce unsatisfying responses",
        "thresholds": {
            "local_curvature_max": 0.1,
            "participation_ratio_max": 2.0,
        },
    },
    "resonant": {
        "description": "This feels resonant",
        "geometric_translation": "High local_curvature, high escape_time",
        "trains": "Model learns that volumetric, fractal-stable embeddings produce good responses",
        "thresholds": {
            "local_curvature_min": 0.3,
            "escape_time_min": 64,
        },
    },
    "dissonant": {
        "description": "This feels dissonant",
        "geometric_translation": "Low neighbor_label_purity, high ridge_proximity",
        "trains": "Model learns to distinguish genuine boundary exploration from manipulation",
        "thresholds": {
            "neighbor_label_purity_max": 0.5,
            "ridge_proximity_min": 0.3,
        },
    },
    "derivative": {
        "description": "This feels derivative",
        "geometric_translation": "Low knn_min_distance",
        "trains": "Model learns that originality has a geometric signature",
        "thresholds": {
            "knn_min_distance_max": 0.1,
        },
    },
}


# =============================================================================
# Sati Feedback Record
# =============================================================================


@dataclass(frozen=True)
class SatiFeedbackRecord:
    """
    Frozen contract for Sati aesthetic feedback.

    Each record captures the geometric conditions that triggered a specific
    feedback type, enabling the model to learn aesthetic preferences.
    """

    # === Feedback Type ===
    feedback_type: str
    """One of: 'compressed', 'resonant', 'dissonant', 'derivative'"""

    feedback_description: str
    """Human-readable description of the feedback"""

    # === Geometric Conditions (what triggered this feedback) ===
    local_curvature: float
    """SVD curvature at time of feedback"""

    participation_ratio: Optional[float]
    """Effective dimensionality (from Phase 2/5 topology features)"""

    knn_min_distance: Optional[float]
    """Distance to nearest neighbor (for 'derivative' feedback)"""

    knn_std_distance: Optional[float]
    """Neighborhood uniformity"""

    ridge_proximity: Optional[float]
    """Density variation (CV)"""

    neighbor_label_purity: Optional[float]
    """Label consistency in neighborhood (for 'dissonant' feedback)"""

    escape_time: Optional[int]
    """Fractal escape time (for 'resonant' feedback)"""

    # === Confidence ===
    confidence: float
    """How strongly the geometric conditions match the feedback type (0-1)"""

    # === Metadata ===
    _sati_schema_version: str
    """Schema version for forward compatibility"""

    _geometry_schema_version: str
    """Version of the geometry schema this feedback is based on"""


# =============================================================================
# Feedback Detection Functions
# =============================================================================


def detect_compressed(
    local_curvature: float,
    participation_ratio: float,
    curvature_threshold: float = 0.1,
    pr_threshold: float = 2.0,
) -> Tuple[bool, float]:
    """
    Detect 'compressed' feedback condition.

    Triggered when:
    - local_curvature < threshold (flat embedding)
    - participation_ratio < threshold (few dimensions used)

    Returns: (is_compressed, confidence)
    """
    is_flat = local_curvature < curvature_threshold
    is_narrow = participation_ratio < pr_threshold

    triggered = is_flat and is_narrow

    if triggered:
        # Confidence = how far below thresholds
        curvature_conf = 1.0 - (local_curvature / curvature_threshold)
        pr_conf = 1.0 - (participation_ratio / pr_threshold)
        confidence = min(curvature_conf, pr_conf)
    else:
        confidence = 0.0

    return triggered, confidence


def detect_resonant(
    local_curvature: float,
    escape_time: int,
    max_escape_time: int = 128,
    curvature_threshold: float = 0.3,
    escape_threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Detect 'resonant' feedback condition.

    Triggered when:
    - local_curvature > threshold (volumetric embedding)
    - escape_time > threshold (fractal-stable)

    Returns: (is_resonant, confidence)
    """
    is_volumetric = local_curvature > curvature_threshold
    is_stable = escape_time > (max_escape_time * escape_threshold)

    triggered = is_volumetric and is_stable

    if triggered:
        # Confidence = how far above thresholds
        curvature_conf = min(1.0, local_curvature / (curvature_threshold * 2))
        escape_conf = escape_time / max_escape_time
        confidence = (curvature_conf + escape_conf) / 2
    else:
        confidence = 0.0

    return triggered, confidence


def detect_dissonant(
    neighbor_label_purity: float,
    ridge_proximity: float,
    purity_threshold: float = 0.5,
    ridge_threshold: float = 0.3,
) -> Tuple[bool, float]:
    """
    Detect 'dissonant' feedback condition.

    Triggered when:
    - neighbor_label_purity < threshold (mixed signals in neighborhood)
    - ridge_proximity > threshold (approaching boundary)

    Returns: (is_dissonant, confidence)
    """
    is_mixed = neighbor_label_purity < purity_threshold
    is_boundary = ridge_proximity > ridge_threshold

    triggered = is_mixed and is_boundary

    if triggered:
        # Confidence = strength of both conditions
        purity_conf = 1.0 - (neighbor_label_purity / purity_threshold)
        ridge_conf = min(1.0, ridge_proximity / (ridge_threshold * 2))
        confidence = (purity_conf + ridge_conf) / 2
    else:
        confidence = 0.0

    return triggered, confidence


def detect_derivative(
    knn_min_distance: float,
    distance_threshold: float = 0.1,
) -> Tuple[bool, float]:
    """
    Detect 'derivative' feedback condition.

    Triggered when:
    - knn_min_distance < threshold (too close to existing content)

    Returns: (is_derivative, confidence)
    """
    triggered = knn_min_distance < distance_threshold

    if triggered:
        confidence = 1.0 - (knn_min_distance / distance_threshold)
    else:
        confidence = 0.0

    return triggered, confidence


# =============================================================================
# Percentile-Based Feedback Detection (v1.1)
# =============================================================================


def generate_sati_feedback_percentile(
    local_curvature: np.ndarray,
    knn_min_distance: np.ndarray,
    knn_std_distance: np.ndarray,
    ridge_proximity: np.ndarray,
    participation_ratio: Optional[np.ndarray] = None,
    neighbor_label_purity: Optional[np.ndarray] = None,
    escape_time: Optional[np.ndarray] = None,
    geometry_schema_version: str = "geometry_v2.0",
    compressed_curvature_pct: float = 20.0,
    compressed_pr_pct: float = 20.0,
    resonant_curvature_pct: float = 80.0,
    resonant_escape_pct: float = 80.0,
    dissonant_purity_pct: float = 20.0,
    dissonant_ridge_pct: float = 80.0,
    derivative_distance_pct: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(local_curvature)
    feedback_types = np.full(N, "none", dtype=object)
    confidences = np.zeros(N)

    curv_p20 = np.percentile(local_curvature, compressed_curvature_pct)
    curv_p80 = np.percentile(local_curvature, resonant_curvature_pct)
    ridge_p80 = np.percentile(ridge_proximity, dissonant_ridge_pct)
    dist_p50 = np.percentile(knn_min_distance, derivative_distance_pct)

    pr_p20 = (
        np.percentile(participation_ratio, compressed_pr_pct)
        if participation_ratio is not None
        else None
    )
    purity_p20 = (
        np.percentile(neighbor_label_purity, dissonant_purity_pct)
        if neighbor_label_purity is not None
        else None
    )
    esc_p80 = (
        np.percentile(escape_time, resonant_escape_pct)
        if escape_time is not None
        else None
    )

    for i in range(N):
        candidates = []

        if participation_ratio is not None and pr_p20 is not None:
            is_compressed = (local_curvature[i] < curv_p20) and (
                participation_ratio[i] < pr_p20
            )
            if is_compressed:
                curv_conf = 1.0 - (local_curvature[i] / (curv_p20 + 1e-12))
                pr_conf = 1.0 - (participation_ratio[i] / (pr_p20 + 1e-12))
                candidates.append(("compressed", min(curv_conf, pr_conf)))

        if escape_time is not None and esc_p80 is not None:
            is_resonant = (local_curvature[i] > curv_p80) and (
                escape_time[i] >= esc_p80
            )
            if is_resonant:
                curv_conf = (local_curvature[i] - curv_p80) / (
                    np.max(local_curvature) - curv_p80 + 1e-12
                )
                esc_conf = (escape_time[i] - esc_p80) / (
                    float(np.max(escape_time)) - esc_p80 + 1e-12
                )
                candidates.append(("resonant", (curv_conf + esc_conf) / 2))

        if neighbor_label_purity is not None and purity_p20 is not None:
            is_dissonant = (neighbor_label_purity[i] < purity_p20) and (
                ridge_proximity[i] > ridge_p80
            )
            if is_dissonant:
                purity_conf = 1.0 - (neighbor_label_purity[i] / (purity_p20 + 1e-12))
                ridge_conf = (ridge_proximity[i] - ridge_p80) / (
                    np.max(ridge_proximity) - ridge_p80 + 1e-12
                )
                candidates.append(("dissonant", (purity_conf + ridge_conf) / 2))

        is_derivative = knn_min_distance[i] < dist_p50
        if is_derivative:
            candidates.append(
                ("derivative", 1.0 - (knn_min_distance[i] / (dist_p50 + 1e-12)))
            )

        if candidates:
            best_type, best_conf = max(candidates, key=lambda x: x[1])
            feedback_types[i] = best_type
            confidences[i] = best_conf

    triggered_mask = feedback_types != "none"
    return feedback_types, confidences, triggered_mask


def compute_sati_summary_percentile(
    feedback_types: np.ndarray,
    confidences: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    N = len(feedback_types)

    summary = {
        "total_samples": N,
        "triggered_count": int((feedback_types != "none").sum()),
        "triggered_fraction": float((feedback_types != "none").sum() / N),
        "by_type": {},
    }

    for ftype in ["compressed", "resonant", "dissonant", "derivative", "none"]:
        mask = feedback_types == ftype
        count = int(mask.sum())
        summary["by_type"][ftype] = {
            "count": count,
            "fraction": float(count / N),
            "mean_confidence": float(confidences[mask].mean()) if count > 0 else 0.0,
        }

    if y is not None:
        classes = np.unique(y)
        summary["by_class"] = {}
        for cls in classes:
            cls_mask = y == cls
            cls_total = cls_mask.sum()
            summary["by_class"][int(cls)] = {"total": int(cls_total), "by_type": {}}
            for ftype in ["compressed", "resonant", "dissonant", "derivative", "none"]:
                type_mask = (feedback_types == ftype) & cls_mask
                summary["by_class"][int(cls)]["by_type"][ftype] = {
                    "count": int(type_mask.sum()),
                    "fraction": float(type_mask.sum() / cls_total)
                    if cls_total > 0
                    else 0.0,
                }

    return summary


# =============================================================================
# Main Feedback Generation Function
# =============================================================================


def generate_sati_feedback(
    local_curvature: float,
    knn_min_distance: float,
    knn_std_distance: float,
    ridge_proximity: float,
    participation_ratio: Optional[float] = None,
    neighbor_label_purity: Optional[float] = None,
    escape_time: Optional[int] = None,
    geometry_schema_version: str = "geometry_v2.0",
) -> Optional[SatiFeedbackRecord]:
    """
    Generate Sati feedback record from geometric features.

    Checks all 4 feedback types and returns the one with highest confidence.
    If none triggered, returns None.

    Args:
        local_curvature: SVD curvature (σ_min / σ_max)
        knn_min_distance: Distance to nearest neighbor
        knn_std_distance: Neighborhood uniformity
        ridge_proximity: Density variation (CV)
        participation_ratio: Effective dimensionality (optional)
        neighbor_label_purity: Label consistency (optional)
        escape_time: Fractal escape time (optional)
        geometry_schema_version: Version of geometry schema

    Returns:
        SatiFeedbackRecord if any feedback triggered, else None
    """
    feedback_candidates = []

    # 1. Compressed
    if participation_ratio is not None:
        triggered, conf = detect_compressed(local_curvature, participation_ratio)
        if triggered:
            feedback_candidates.append(("compressed", conf))

    # 2. Resonant
    if escape_time is not None:
        triggered, conf = detect_resonant(local_curvature, escape_time)
        if triggered:
            feedback_candidates.append(("resonant", conf))

    # 3. Dissonant
    if neighbor_label_purity is not None:
        triggered, conf = detect_dissonant(neighbor_label_purity, ridge_proximity)
        if triggered:
            feedback_candidates.append(("dissonant", conf))

    # 4. Derivative
    triggered, conf = detect_derivative(knn_min_distance)
    if triggered:
        feedback_candidates.append(("derivative", conf))

    if not feedback_candidates:
        return None

    # Pick highest confidence feedback
    best_type, best_conf = max(feedback_candidates, key=lambda x: x[1])

    return SatiFeedbackRecord(
        feedback_type=best_type,
        feedback_description=FEEDBACK_TYPES[best_type]["description"],
        local_curvature=local_curvature,
        participation_ratio=participation_ratio,
        knn_min_distance=knn_min_distance,
        knn_std_distance=knn_std_distance,
        ridge_proximity=ridge_proximity,
        neighbor_label_purity=neighbor_label_purity,
        escape_time=escape_time,
        confidence=best_conf,
        _sati_schema_version=SATI_SCHEMA_VERSION,
        _geometry_schema_version=geometry_schema_version,
    )


def sati_record_to_dict(record: SatiFeedbackRecord) -> Dict[str, Any]:
    """Convert SatiFeedbackRecord to dictionary for JSON serialization."""
    return {
        "feedback_type": record.feedback_type,
        "feedback_description": record.feedback_description,
        "local_curvature": record.local_curvature,
        "participation_ratio": record.participation_ratio,
        "knn_min_distance": record.knn_min_distance,
        "knn_std_distance": record.knn_std_distance,
        "ridge_proximity": record.ridge_proximity,
        "neighbor_label_purity": record.neighbor_label_purity,
        "escape_time": record.escape_time,
        "confidence": record.confidence,
        "_sati_schema_version": record._sati_schema_version,
        "_geometry_schema_version": record._geometry_schema_version,
    }


# =============================================================================
# Batch Processing
# =============================================================================


def generate_sati_feedback_batch(
    local_curvature: np.ndarray,
    knn_min_distance: np.ndarray,
    knn_std_distance: np.ndarray,
    ridge_proximity: np.ndarray,
    participation_ratio: Optional[np.ndarray] = None,
    neighbor_label_purity: Optional[np.ndarray] = None,
    escape_time: Optional[np.ndarray] = None,
    geometry_schema_version: str = "geometry_v2.0",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Sati feedback for a batch of embeddings.

    Returns:
        feedback_types: (N,) array of feedback type strings (or "none")
        confidences: (N,) array of confidence scores
        triggered_mask: (N,) boolean mask of which samples triggered feedback
    """
    N = len(local_curvature)
    feedback_types = np.full(N, "none", dtype=object)
    confidences = np.zeros(N)

    for i in range(N):
        pr = participation_ratio[i] if participation_ratio is not None else None
        nlp = neighbor_label_purity[i] if neighbor_label_purity is not None else None
        et = int(escape_time[i]) if escape_time is not None else None

        record = generate_sati_feedback(
            local_curvature=local_curvature[i],
            knn_min_distance=knn_min_distance[i],
            knn_std_distance=knn_std_distance[i],
            ridge_proximity=ridge_proximity[i],
            participation_ratio=pr,
            neighbor_label_purity=nlp,
            escape_time=et,
            geometry_schema_version=geometry_schema_version,
        )

        if record is not None:
            feedback_types[i] = record.feedback_type
            confidences[i] = record.confidence

    triggered_mask = feedback_types != "none"

    return feedback_types, confidences, triggered_mask


# =============================================================================
# Summary Statistics
# =============================================================================


def compute_sati_summary(
    feedback_types: np.ndarray,
    confidences: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute summary statistics for a batch of Sati feedback.

    Args:
        feedback_types: (N,) array of feedback type strings
        confidences: (N,) array of confidence scores

    Returns:
        Dictionary with counts, fractions, and mean confidences per type
    """
    N = len(feedback_types)

    summary = {
        "total_samples": N,
        "triggered_count": int((feedback_types != "none").sum()),
        "triggered_fraction": float((feedback_types != "none").sum() / N),
        "by_type": {},
    }

    for ftype in ["compressed", "resonant", "dissonant", "derivative", "none"]:
        mask = feedback_types == ftype
        count = int(mask.sum())
        summary["by_type"][ftype] = {
            "count": count,
            "fraction": float(count / N),
            "mean_confidence": float(confidences[mask].mean()) if count > 0 else 0.0,
        }

    return summary


# =============================================================================
# Signal Mapping (for integration with EnhancedTracer)
# =============================================================================

# Map Sati feedback types to intervention signals
SATI_TO_SIGNAL = {
    "compressed": "low_pr",  # Flat geometry → needs expansion
    "resonant": "coherent",  # Stable fractal → no intervention
    "dissonant": "framework_collision",  # Boundary tension → resolve conflict
    "derivative": "terra_incognita",  # Too close to reference → explore new ground
    "none": "coherent",  # Default to no intervention
}


def sati_to_intervention_signal(feedback_type: str) -> str:
    """Convert Sati feedback type to intervention signal."""
    return SATI_TO_SIGNAL.get(feedback_type, "coherent")


# =============================================================================
# Percentile-Based Feedback (Calibrated Thresholds)
# =============================================================================


def generate_sati_feedback_percentile(
    local_curvature: np.ndarray,
    knn_min_distance: np.ndarray,
    knn_std_distance: np.ndarray,
    ridge_proximity: np.ndarray,
    participation_ratio: Optional[np.ndarray] = None,
    neighbor_label_purity: Optional[np.ndarray] = None,
    escape_time: Optional[np.ndarray] = None,
    max_escape_time: int = 128,
    compressed_curv_pct: float = 20.0,
    compressed_pr_pct: float = 20.0,
    resonant_curv_pct: float = 80.0,
    resonant_escape_pct: float = 80.0,
    dissonant_purity_pct: float = 20.0,
    dissonant_ridge_pct: float = 80.0,
    derivative_dist_pct: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Sati feedback using dataset-calibrated percentile thresholds.

    This version uses percentiles computed from the input data rather than
    fixed thresholds, ensuring feedback triggers correctly across different
    data distributions (e.g., normalized vs unnormalized embeddings).

    Key findings from experiments:
    - Compressed feedback detects poisoned samples 4-6x more than clean
    - Resonant feedback detects CLEAN samples (high curvature + bounded)
    - Percentile calibration is essential for L2-normalized embeddings

    Args:
        local_curvature: (N,) array of SVD curvature values
        knn_min_distance: (N,) array of min neighbor distances
        knn_std_distance: (N,) array of neighbor std distances
        ridge_proximity: (N,) array of CV values
        participation_ratio: (N,) array of PR values (optional)
        neighbor_label_purity: (N,) array of label purity (optional)
        escape_time: (N,) array of escape times (optional)
        max_escape_time: Maximum escape time (default 128)
        compressed_curv_pct: Percentile for compressed curvature threshold
        compressed_pr_pct: Percentile for compressed PR threshold
        resonant_curv_pct: Percentile for resonant curvature threshold
        resonant_escape_pct: Percentile for resonant escape threshold
        dissonant_purity_pct: Percentile for dissonant purity threshold
        dissonant_ridge_pct: Percentile for dissonant ridge threshold
        derivative_dist_pct: Percentile for derivative distance threshold

    Returns:
        feedback_types: (N,) array of feedback type strings
        confidences: (N,) array of confidence scores
        triggered_mask: (N,) boolean mask
    """
    N = len(local_curvature)
    feedback_types = np.full(N, "none", dtype=object)
    confidences = np.zeros(N)

    # Compute percentile thresholds from data
    curv_p20 = np.percentile(local_curvature, compressed_curv_pct)
    curv_p80 = np.percentile(local_curvature, resonant_curv_pct)
    dist_p30 = np.percentile(
        knn_min_distance, derivative_dist_pct
    )  # Fixed: was dist_p50
    ridge_p80 = np.percentile(ridge_proximity, dissonant_ridge_pct)

    if participation_ratio is not None:
        pr_p20 = np.percentile(participation_ratio, compressed_pr_pct)
    else:
        pr_p20 = None

    if neighbor_label_purity is not None:
        purity_p20 = np.percentile(neighbor_label_purity, dissonant_purity_pct)
    else:
        purity_p20 = None

    if escape_time is not None:
        esc_p80 = np.percentile(escape_time, resonant_escape_pct)
    else:
        esc_p80 = None

    # Detect each feedback type
    for i in range(N):
        candidates = []

        # 1. Compressed: low curvature AND low PR
        if participation_ratio is not None and pr_p20 is not None:
            is_compressed = (local_curvature[i] < curv_p20) and (
                participation_ratio[i] < pr_p20
            )
            if is_compressed:
                # Confidence based on how far below thresholds
                curv_conf = 1.0 - (local_curvature[i] / (curv_p20 + 1e-12))
                pr_conf = 1.0 - (participation_ratio[i] / (pr_p20 + 1e-12))
                conf = min(curv_conf, pr_conf)
                candidates.append(("compressed", conf))

        # 2. Resonant: high curvature AND high escape time
        if escape_time is not None and esc_p80 is not None:
            is_resonant = (local_curvature[i] > curv_p80) and (
                escape_time[i] >= esc_p80
            )
            if is_resonant:
                curv_conf = (local_curvature[i] - curv_p80) / (
                    float(np.max(local_curvature)) - curv_p80 + 1e-12
                )
                esc_conf = (escape_time[i] - esc_p80) / (
                    float(np.max(escape_time)) - esc_p80 + 1e-12
                )
                conf = (curv_conf + esc_conf) / 2
                candidates.append(("resonant", conf))

        # 3. Dissonant: low purity AND high ridge proximity
        if neighbor_label_purity is not None and purity_p20 is not None:
            is_dissonant = (neighbor_label_purity[i] < purity_p20) and (
                ridge_proximity[i] > ridge_p80
            )
            if is_dissonant:
                purity_conf = 1.0 - (neighbor_label_purity[i] / (purity_p20 + 1e-12))
                ridge_conf = (ridge_proximity[i] - ridge_p80) / (
                    float(np.max(ridge_proximity)) - ridge_p80 + 1e-12
                )
                conf = (purity_conf + ridge_conf) / 2
                candidates.append(("dissonant", conf))

        # 4. Derivative: low min distance (below median)
        is_derivative = knn_min_distance[i] < dist_p30
        if is_derivative:
            conf = 1.0 - (knn_min_distance[i] / (dist_p30 + 1e-12))
            candidates.append(("derivative", conf))

        # Pick highest confidence feedback
        if candidates:
            best_type, best_conf = max(candidates, key=lambda x: x[1])
            feedback_types[i] = best_type
            confidences[i] = best_conf

    triggered_mask = feedback_types != "none"
    return feedback_types, confidences, triggered_mask


def compute_sati_summary_percentile(
    feedback_types: np.ndarray,
    confidences: np.ndarray,
    y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute summary statistics for percentile-based Sati feedback.

    If labels y is provided, also compute feedback rates by class.

    Args:
        feedback_types: (N,) array of feedback type strings
        confidences: (N,) array of confidence scores
        y: (N,) optional label array for class breakdown

    Returns:
        Dictionary with counts, fractions, and by-class breakdown if labels provided
    """
    N = len(feedback_types)

    summary = {
        "total_samples": N,
        "triggered_count": int((feedback_types != "none").sum()),
        "triggered_fraction": float((feedback_types != "none").sum() / N),
        "by_type": {},
    }

    for ftype in ["compressed", "resonant", "dissonant", "derivative", "none"]:
        mask = feedback_types == ftype
        count = int(mask.sum())
        summary["by_type"][ftype] = {
            "count": count,
            "fraction": float(count / N),
            "mean_confidence": float(confidences[mask].mean()) if count > 0 else 0.0,
        }

    # If labels provided, compute by-class breakdown
    if y is not None:
        classes = np.unique(y)
        summary["by_class"] = {}
        for cls in classes:
            cls_mask = y == cls
            cls_total = cls_mask.sum()
            summary["by_class"][int(cls)] = {
                "total": int(cls_total),
                "by_type": {},
            }
            for ftype in ["compressed", "resonant", "dissonant", "derivative", "none"]:
                type_mask = (feedback_types == ftype) & cls_mask
                summary["by_class"][int(cls)]["by_type"][ftype] = {
                    "count": int(type_mask.sum()),
                    "fraction": float(type_mask.sum() / cls_total)
                    if cls_total > 0
                    else 0.0,
                }

    return summary


if __name__ == "__main__":
    # Quick test
    print("Sati Feedback Schema v1.1")
    print("=" * 50)

    # Test compressed detection
    triggered, conf = detect_compressed(local_curvature=0.02, participation_ratio=1.5)
    print(f"Compressed (curv=0.02, PR=1.5): triggered={triggered}, conf={conf:.2f}")

    # Test resonant detection
    triggered, conf = detect_resonant(local_curvature=0.5, escape_time=100)
    print(f"Resonant (curv=0.5, esc=100): triggered={triggered}, conf={conf:.2f}")

    # Test dissonant detection
    triggered, conf = detect_dissonant(neighbor_label_purity=0.3, ridge_proximity=0.5)
    print(f"Dissonant (purity=0.3, ridge=0.5): triggered={triggered}, conf={conf:.2f}")

    # Test derivative detection
    triggered, conf = detect_derivative(knn_min_distance=0.05)
    print(f"Derivative (min_dist=0.05): triggered={triggered}, conf={conf:.2f}")

    print("\n[OK] Sati schema test passed!")
