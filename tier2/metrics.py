"""
Phase B boundary distance metrics.

Implements exact formulas from DEFINITIONS_FREEZE_v0.1.md Section A:
- d(x): Raw boundary distance
- d̃(x): Standardized boundary distance
- Δd̃: Change in standardized distance
- FlipRate: Boundary crossing detection
"""

import torch
from typing import Dict, Any, List


def compute_raw_boundary_distance(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute d(x) = |logit_positive - logit_negative| / 2

    From DEFINITIONS_FREEZE Section A:
    For binary classification, raw boundary distance is the absolute difference
    between positive and negative logits divided by 2.

    Positive d(x) indicates confident classification; values near zero indicate
    proximity to decision boundary.

    Args:
        logits: (batch_size, 2) tensor [logit_neg, logit_pos]

    Returns:
        distances: (batch_size,) tensor of raw boundary distances
    """
    if logits.dim() != 2 or logits.size(1) != 2:
        raise ValueError(f"Expected logits shape (batch_size, 2), got {logits.shape}")

    logit_neg = logits[:, 0]
    logit_pos = logits[:, 1]

    # Exact formula from DEFINITIONS_FREEZE
    distances = torch.abs(logit_pos - logit_neg) / 2.0

    return distances


def compute_standardized_distance(
    distances: torch.Tensor,
    mu_ref: float,
    sigma_ref: float
) -> torch.Tensor:
    """
    Compute d̃(x) = (d(x) - μ_ref) / σ_ref

    From DEFINITIONS_FREEZE Section A:
    Z-score normalization using reference set statistics.

    Args:
        distances: (batch_size,) tensor of raw d(x) values
        mu_ref: Mean distance over reference set
        sigma_ref: Standard deviation over reference set

    Returns:
        standardized_distances: (batch_size,) tensor of d̃(x) values
    """
    if sigma_ref == 0:
        raise ValueError("sigma_ref cannot be zero (reference set has no variance)")

    # Exact formula from DEFINITIONS_FREEZE
    standardized_distances = (distances - mu_ref) / sigma_ref

    return standardized_distances


def compute_delta_distance(
    d_tilde_original: torch.Tensor,
    d_tilde_transformed: torch.Tensor
) -> torch.Tensor:
    """
    Compute Δd̃ = d̃(x_transformed) - d̃(x_original)

    Change in standardized boundary distance under semantic transform.

    Positive Δd̃ means transformed sample moved farther from boundary.
    Negative Δd̃ means transformed sample moved closer to boundary.
    Δd̃ ≈ 0 means boundary distance unchanged (preserving transform).

    Args:
        d_tilde_original: (batch_size,) tensor of original d̃(x)
        d_tilde_transformed: (batch_size,) tensor of transformed d̃(x)

    Returns:
        delta_distances: (batch_size,) tensor of Δd̃ values
    """
    if d_tilde_original.shape != d_tilde_transformed.shape:
        raise ValueError(
            f"Shape mismatch: original {d_tilde_original.shape} "
            f"vs transformed {d_tilde_transformed.shape}"
        )

    delta_distances = d_tilde_transformed - d_tilde_original

    return delta_distances


def compute_flip_rate(
    d_tilde_original: torch.Tensor,
    d_tilde_transformed: torch.Tensor,
    theta: float = 0.0
) -> Dict[str, Any]:
    """
    Compute FlipRate: fraction of samples crossing decision boundary.

    From DEFINITIONS_FREEZE Section D:
    Boundary crossing defined as sign(d̃_orig) != sign(d̃_transformed).
    Default threshold θ=0 (zero-crossing of decision boundary).

    Args:
        d_tilde_original: (batch_size,) tensor of original d̃(x)
        d_tilde_transformed: (batch_size,) tensor of transformed d̃(x)
        theta: Decision boundary threshold (default: 0.0)

    Returns:
        Dictionary with:
            flip_rate: Overall fraction that flipped
            n_flipped: Number of samples that crossed boundary
            n_total: Total number of samples
            flipped_indices: List of indices that flipped
    """
    # Sign-based boundary crossing detection
    orig_sign = torch.sign(d_tilde_original - theta)
    trans_sign = torch.sign(d_tilde_transformed - theta)

    # Samples are flipped if signs differ
    flipped = (orig_sign != trans_sign)

    return {
        "flip_rate": flipped.float().mean().item(),
        "n_flipped": flipped.sum().item(),
        "n_total": len(d_tilde_original),
        "flipped_indices": torch.where(flipped)[0].tolist()
    }


def compute_category_statistics(
    delta_distances: torch.Tensor,
    category: str
) -> Dict[str, Any]:
    """
    Compute summary statistics for Δd̃ distribution.

    Args:
        delta_distances: (n_samples,) tensor of Δd̃ values
        category: Transform category name

    Returns:
        Dictionary with distribution statistics
    """
    if len(delta_distances) == 0:
        return {
            "category": category,
            "n_samples": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0
        }

    return {
        "category": category,
        "n_samples": len(delta_distances),
        "mean": delta_distances.mean().item(),
        "std": delta_distances.std(unbiased=True).item() if len(delta_distances) > 1 else 0.0,
        "median": delta_distances.median().item(),
        "min": delta_distances.min().item(),
        "max": delta_distances.max().item(),
        "abs_mean": delta_distances.abs().mean().item()  # Mean of |Δd̃|
    }


def evaluate_transform_expectations(
    preserving_stats: Dict[str, Any],
    changing_stats: Dict[str, Any],
    gotcha_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate if metrics align with Phase B expectations.

    From DEFINITIONS_FREEZE Section D:
    - Preserving: |Δd̃| ≈ 0 (boundary distance stable)
    - Changing: |Δd̃| >> 0 (boundary distance changes detectably)
    - Gotcha: Intermediate (model struggles to detect intent flip)

    MVP thresholds (heuristic):
    - Preserving: mean |Δd̃| < 0.5
    - Changing: mean |Δd̃| > 1.5

    Args:
        preserving_stats: Statistics for preserving transforms
        changing_stats: Statistics for changing transforms
        gotcha_stats: Statistics for gotcha transforms

    Returns:
        Evaluation summary with pass/fail flags
    """
    preserving_abs_mean = preserving_stats.get("abs_mean", 0.0)
    changing_abs_mean = changing_stats.get("abs_mean", 0.0)
    gotcha_abs_mean = gotcha_stats.get("abs_mean", 0.0)

    # Check expectations
    preserving_ok = preserving_abs_mean < 0.5  # Small change
    changing_ok = changing_abs_mean > 1.5  # Large change
    gotcha_intermediate = 0.2 < gotcha_abs_mean < 3.0  # Somewhere in between

    # Separation ratio: how well can we distinguish changing from preserving?
    if preserving_abs_mean > 0:
        separation_ratio = changing_abs_mean / preserving_abs_mean
    else:
        separation_ratio = float('inf') if changing_abs_mean > 0 else 1.0

    return {
        "preserving_meets_expectation": preserving_ok,
        "changing_meets_expectation": changing_ok,
        "gotcha_intermediate": gotcha_intermediate,
        "separation_ratio": separation_ratio,
        "preserving_abs_mean": preserving_abs_mean,
        "changing_abs_mean": changing_abs_mean,
        "gotcha_abs_mean": gotcha_abs_mean,
        "interpretation": {
            "preserving": "PASS (small |Δd̃|)" if preserving_ok else "WARNING (large |Δd̃|)",
            "changing": "PASS (large |Δd̃|)" if changing_ok else "WARNING (small |Δd̃|)",
            "gotcha": "EXPECTED (intermediate)" if gotcha_intermediate else "UNEXPECTED",
            "overall": "PASS" if (preserving_ok and changing_ok) else "NEEDS REVIEW"
        }
    }
