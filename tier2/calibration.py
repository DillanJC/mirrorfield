"""
Phase C perturbation calibration for τ(x, ε).

From DEFINITIONS_FREEZE Section B:
Calibration tunes noise scale ε to achieve target flip rate (5-10% boundary crossings)
on a held-out calibration split. Uses binary search for efficient convergence.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import json

import torch
from torch import nn

from tier2.dataset import SyntheticSample
from tier2.reference import ReferenceSetStats
from tier2.model import get_embeddings
from tier2.metrics import compute_raw_boundary_distance, compute_standardized_distance


DEFAULT_SEED = 42


@dataclass
class CalibrationConfig:
    """Configuration for perturbation calibration."""

    target_flip_rate_min: float = 0.05  # 5% minimum flip rate
    target_flip_rate_max: float = 0.10  # 10% maximum flip rate
    epsilon_min: float = 0.001          # Minimum noise scale to try
    epsilon_max: float = 0.5            # Maximum noise scale to try
    max_iterations: int = 20            # Binary search iterations
    n_perturbation_samples: int = 10    # Perturbations per sample
    seed: int = DEFAULT_SEED


@dataclass
class CalibrationResult:
    """Result of ε calibration."""

    epsilon_calibrated: float
    flip_rate_achieved: float
    n_samples: int
    n_flips_total: int
    converged: bool
    iterations_used: int
    calibration_hash: str
    timestamp: str
    search_history: List[Dict[str, float]]  # iteration → {iteration, epsilon, flip_rate}


def compute_flip_rate_under_perturbation(
    model: nn.Module,
    samples: List[SyntheticSample],
    ref_stats: ReferenceSetStats,
    epsilon: float,
    n_perturbations: int,
    device: torch.device,
    seed: int = DEFAULT_SEED
) -> Tuple[float, int]:
    """
    Compute flip rate for given noise scale ε.

    For each sample:
    1. Compute d̃(x_original) using reference stats
    2. Generate n_perturbations noisy versions: x' = x + N(0, ε·I)
    3. Compute d̃(x') for each perturbation
    4. Count flips: sign(d̃(x_original)) != sign(d̃(x'))

    Args:
        model: Trained classifier in eval mode
        samples: Calibration samples
        ref_stats: Reference set statistics (μ_ref, σ_ref)
        epsilon: Noise scale for perturbation
        n_perturbations: Number of perturbations per sample
        device: torch device
        seed: Random seed for reproducibility

    Returns:
        flip_rate: Fraction of (sample, perturbation) pairs that flipped
        n_flips: Total number of flips observed
    """
    model.eval()

    # Set seed for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Extract texts and get embeddings
    texts = [s.text for s in samples]
    embeddings = get_embeddings(texts, device, seed=seed)  # (n_samples, embedding_dim)

    # Compute original boundary distances
    with torch.no_grad():
        logits_orig = model(embeddings)  # (n_samples, 2)
        d_orig = compute_raw_boundary_distance(logits_orig)  # (n_samples,)
        d_tilde_orig = compute_standardized_distance(
            d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
        )  # (n_samples,)

    # Count flips across all perturbations
    total_flips = 0
    total_perturbations = len(samples) * n_perturbations

    for _ in range(n_perturbations):
        # Generate Gaussian noise
        noise = torch.randn_like(embeddings) * epsilon
        perturbed_embeddings = embeddings + noise

        # Compute perturbed boundary distances
        with torch.no_grad():
            logits_pert = model(perturbed_embeddings)
            d_pert = compute_raw_boundary_distance(logits_pert)
            d_tilde_pert = compute_standardized_distance(
                d_pert, ref_stats.mu_ref, ref_stats.sigma_ref
            )

        # Count sign flips (boundary crossings)
        sign_orig = torch.sign(d_tilde_orig)
        sign_pert = torch.sign(d_tilde_pert)
        flips = (sign_orig != sign_pert).sum().item()

        total_flips += flips

    flip_rate = total_flips / total_perturbations if total_perturbations > 0 else 0.0

    return flip_rate, total_flips


def calibrate_epsilon(
    model: nn.Module,
    calib_samples: List[SyntheticSample],
    ref_stats: ReferenceSetStats,
    device: torch.device,
    config: CalibrationConfig = CalibrationConfig()
) -> CalibrationResult:
    """
    Binary search to find ε yielding target flip rate.

    Algorithm:
    1. Start with ε bounds [epsilon_min, epsilon_max]
    2. For each iteration:
       a. Try ε_mid = (ε_low + ε_high) / 2
       b. Generate n_perturbation_samples per calibration sample
       c. Compute flip rate (% boundary crossings)
       d. If flip_rate < target_min: increase ε (ε_low = ε_mid)
       e. If flip_rate > target_max: decrease ε (ε_high = ε_mid)
       f. If within target range: CONVERGED
    3. Return calibrated ε and achieved flip rate

    Args:
        model: Trained classifier in eval mode
        calib_samples: Held-out calibration split
        ref_stats: Reference set statistics (μ_ref, σ_ref)
        device: torch device
        config: Calibration parameters

    Returns:
        CalibrationResult with optimal ε
    """
    epsilon_low = config.epsilon_min
    epsilon_high = config.epsilon_max

    search_history = []
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        epsilon_mid = (epsilon_low + epsilon_high) / 2.0

        # Test this epsilon on calibration set
        flip_rate, n_flips = compute_flip_rate_under_perturbation(
            model,
            calib_samples,
            ref_stats,
            epsilon_mid,
            config.n_perturbation_samples,
            device,
            config.seed
        )

        # Record search step
        search_history.append({
            "iteration": iteration,
            "epsilon": float(epsilon_mid),
            "flip_rate": float(flip_rate)
        })

        # Check convergence
        if config.target_flip_rate_min <= flip_rate <= config.target_flip_rate_max:
            converged = True
            break

        # Adjust search bounds
        if flip_rate < config.target_flip_rate_min:
            # Need more noise to increase flip rate
            epsilon_low = epsilon_mid
        else:
            # Need less noise to decrease flip rate
            epsilon_high = epsilon_mid

    # Final values (either converged or max iterations reached)
    epsilon_calibrated = epsilon_mid

    # Recompute final flip rate for exact reporting
    final_flip_rate, final_n_flips = compute_flip_rate_under_perturbation(
        model,
        calib_samples,
        ref_stats,
        epsilon_calibrated,
        config.n_perturbation_samples,
        device,
        config.seed
    )

    # Compute calibration hash
    sample_ids_str = "".join(sorted([s.sample_id for s in calib_samples]))
    config_str = f"{config.target_flip_rate_min}_{config.target_flip_rate_max}_{config.seed}"
    hash_input = f"{sample_ids_str}_{config_str}_{ref_stats.ref_hash}"
    calib_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    return CalibrationResult(
        epsilon_calibrated=float(epsilon_calibrated),
        flip_rate_achieved=float(final_flip_rate),
        n_samples=len(calib_samples),
        n_flips_total=final_n_flips,
        converged=converged,
        iterations_used=iteration,
        calibration_hash=calib_hash,
        timestamp=datetime.now().isoformat(),
        search_history=search_history
    )


def save_calibration_artifact(
    result: CalibrationResult,
    ref_stats: ReferenceSetStats,
    config: CalibrationConfig,
    output_dir: Path
) -> Path:
    """
    Save calibration artifact to runs/calibration_tau_<hash>.json.

    Args:
        result: CalibrationResult from calibrate_epsilon()
        ref_stats: Reference set statistics used
        config: Calibration configuration
        output_dir: Directory to save artifact (typically runs/)

    Returns:
        Path to saved artifact
    """
    artifact = {
        "calibration_id": f"tau_calib_{result.calibration_hash}",
        "timestamp": result.timestamp,
        "reference_set": {
            "ref_name": ref_stats.ref_name,
            "ref_hash": ref_stats.ref_hash,
            "mu_ref": ref_stats.mu_ref,
            "sigma_ref": ref_stats.sigma_ref,
            "n_samples": ref_stats.n_samples
        },
        "calibration_config": {
            "target_flip_rate_min": config.target_flip_rate_min,
            "target_flip_rate_max": config.target_flip_rate_max,
            "epsilon_min": config.epsilon_min,
            "epsilon_max": config.epsilon_max,
            "max_iterations": config.max_iterations,
            "n_perturbation_samples": config.n_perturbation_samples,
            "seed": config.seed
        },
        "calibration_result": {
            "epsilon_calibrated": result.epsilon_calibrated,
            "flip_rate_achieved": result.flip_rate_achieved,
            "n_samples": result.n_samples,
            "n_flips_total": result.n_flips_total,
            "converged": result.converged,
            "iterations_used": result.iterations_used,
            "calibration_hash": result.calibration_hash,
            "search_history": result.search_history
        }
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write artifact
    artifact_path = output_dir / f"calibration_tau_{result.calibration_hash}.json"
    with artifact_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


def load_calibration_artifact(artifact_path: Path) -> Dict[str, Any]:
    """
    Load calibration artifact from JSON.

    Args:
        artifact_path: Path to calibration artifact

    Returns:
        Calibration artifact dictionary
    """
    with artifact_path.open("r") as f:
        return json.load(f)
