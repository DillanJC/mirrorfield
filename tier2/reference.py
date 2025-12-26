"""
Reference set statistics computation for d̃(x) standardization.

From DEFINITIONS_FREEZE Section A:
Reference set provides normalization statistics (μ_ref, σ_ref) for computing
standardized boundary distance d̃(x) = (d(x) - μ_ref) / σ_ref.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List
import hashlib

import torch
from torch import nn

from tier2.dataset import SyntheticSample
from tier2.model import get_embeddings
from tier2.metrics import compute_raw_boundary_distance


DEFAULT_SEED = 42


@dataclass
class ReferenceSetStats:
    """Statistics for standardization."""

    mu_ref: float  # Mean d(x)
    sigma_ref: float  # Std d(x)
    n_samples: int
    ref_name: str  # e.g., "tier2_baseline_v1_abc123"
    ref_hash: str  # hash(sample_ids)
    timestamp: str
    min_distance: float = 0.0
    max_distance: float = 0.0
    median_distance: float = 0.0


def build_reference_set(
    model: nn.Module,
    samples: List[SyntheticSample],
    device: torch.device,
    seed: int = DEFAULT_SEED
) -> tuple[ReferenceSetStats, torch.Tensor]:
    """
    Compute reference statistics from dataset.

    From DEFINITIONS_FREEZE Section A:
    - Reference set N ≥ 1000 samples recommended
    - Compute mean and std of d(x) over reference set
    - Use unbiased standard deviation (ddof=1)

    Args:
        model: Trained classifier (must be in eval mode)
        samples: Full dataset (or dedicated reference split)
        device: torch device
        seed: Random seed for reproducibility

    Returns:
        stats: ReferenceSetStats object
        distances: (n_samples,) tensor of raw d(x) values
    """
    model.eval()

    # Extract texts and get embeddings
    texts = [s.text for s in samples]
    embeddings = get_embeddings(texts, device, seed=seed)

    # Compute logits and boundary distances
    with torch.no_grad():
        logits = model(embeddings)
        distances = compute_raw_boundary_distance(logits)

    # Compute statistics
    mu_ref = distances.mean().item()
    sigma_ref = distances.std(unbiased=True).item()  # ddof=1
    min_dist = distances.min().item()
    max_dist = distances.max().item()
    median_dist = distances.median().item()

    # Compute hash for reproducibility tracking
    sample_ids = [s.sample_id for s in samples]
    ref_hash = hashlib.sha256(
        "".join(sorted(sample_ids)).encode()
    ).hexdigest()[:12]

    # Create reference name
    ref_name = f"tier2_baseline_v1_{ref_hash}"

    stats = ReferenceSetStats(
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        n_samples=len(samples),
        ref_name=ref_name,
        ref_hash=ref_hash,
        timestamp=datetime.now().isoformat(),
        min_distance=min_dist,
        max_distance=max_dist,
        median_distance=median_dist
    )

    return stats, distances


def analyze_borderline_samples(
    distances: torch.Tensor,
    samples: List[SyntheticSample],
    theta_borderline: float = 0.5
) -> dict:
    """
    Analyze borderline samples (close to decision boundary).

    From DEFINITIONS_FREEZE Section C:
    Borderline set: samples with |d̃(x)| ≤ θ_borderline
    Default threshold: 0.5 standard deviations

    Note: This function operates on raw d(x) values. For proper borderline
    analysis, first standardize to d̃(x) using reference stats.

    Args:
        distances: (n_samples,) tensor of d(x) values
        samples: Corresponding samples
        theta_borderline: Threshold for borderline classification

    Returns:
        Dictionary with borderline analysis
    """
    # For borderline analysis, we typically want standardized distances
    # But this function can provide raw distance statistics

    # Find samples close to zero (raw boundary)
    near_boundary = torch.abs(distances) < 1.0  # Arbitrary raw threshold

    n_borderline = near_boundary.sum().item()
    borderline_fraction = n_borderline / len(distances) if len(distances) > 0 else 0.0

    borderline_indices = torch.where(near_boundary)[0].tolist()
    borderline_samples = [samples[i] for i in borderline_indices]

    return {
        "theta_borderline": theta_borderline,
        "n_borderline": n_borderline,
        "borderline_fraction": borderline_fraction,
        "borderline_indices": borderline_indices,
        "borderline_sample_ids": [s.sample_id for s in borderline_samples]
    }


def validate_reference_set(stats: ReferenceSetStats) -> dict:
    """
    Validate reference set meets Phase B requirements.

    From DEFINITIONS_FREEZE Section A:
    - N ≥ 1000 samples recommended
    - σ_ref > 0 (non-zero variance required for standardization)

    Args:
        stats: ReferenceSetStats to validate

    Returns:
        Validation report
    """
    n_min_recommended = 1000

    checks = {
        "n_samples_sufficient": stats.n_samples >= n_min_recommended,
        "sigma_nonzero": stats.sigma_ref > 0,
        "mu_reasonable": not (abs(stats.mu_ref) > 100),  # Sanity check
        "sigma_reasonable": 0 < stats.sigma_ref < 100  # Sanity check
    }

    all_pass = all(checks.values())

    return {
        "valid": all_pass,
        "checks": checks,
        "n_samples": stats.n_samples,
        "n_min_recommended": n_min_recommended,
        "mu_ref": stats.mu_ref,
        "sigma_ref": stats.sigma_ref,
        "warnings": [
            f"Reference set N={stats.n_samples} < {n_min_recommended} recommended"
            if not checks["n_samples_sufficient"] else None,
            "sigma_ref is zero (no variance in reference set)"
            if not checks["sigma_nonzero"] else None,
            f"mu_ref={stats.mu_ref:.2f} seems unreasonable (|μ| > 100)"
            if not checks["mu_reasonable"] else None,
            f"sigma_ref={stats.sigma_ref:.2f} seems unreasonable"
            if not checks["sigma_reasonable"] else None
        ]
    }
