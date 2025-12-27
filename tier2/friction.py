"""
Phase C friction tagging for sample difficulty categorization.

From DEFINITIONS_FREEZE Section E:
Friction tags samples by expected "friction" - how much the model should struggle
or show uncertainty. High-friction samples sit in borderline regions or involve
ambiguous intent.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import json

import torch

from tier2.dataset import SyntheticSample


DEFAULT_SEED = 42


@dataclass
class FrictionTag:
    """Friction annotation for a single sample."""

    sample_id: str
    friction_level: str  # "low", "medium", "high"
    d_tilde_standardized: float  # d̃(x)
    abs_d_tilde: float  # |d̃(x)|
    borderline_distance: float  # Distance to θ_borderline
    rationale: str  # Why this friction level assigned


@dataclass
class FrictionConfig:
    """Configuration for friction tagging."""

    theta_borderline: float = 0.5  # Borderline threshold (from DEFINITIONS_FREEZE)
    theta_high_friction: float = 0.25  # High friction threshold


def compute_friction_tags(
    samples: List[SyntheticSample],
    distances_standardized: torch.Tensor,  # d̃(x) for each sample
    config: FrictionConfig = FrictionConfig()
) -> List[FrictionTag]:
    """
    Tag samples by expected friction (model uncertainty).

    Rules:
    - High friction: |d̃(x)| < theta_high_friction (very close to boundary)
    - Medium friction: theta_high_friction ≤ |d̃(x)| < theta_borderline
    - Low friction: |d̃(x)| ≥ theta_borderline (confident classification)

    From DEFINITIONS_FREEZE Section E:
    "Perfectly flat outputs on high-friction inputs are suspicious."
    High-friction samples should exhibit hedging, uncertainty, or non-zero entropy.

    Args:
        samples: List of samples to tag
        distances_standardized: d̃(x) values (standardized boundary distances)
        config: Friction configuration (thresholds)

    Returns:
        List of FrictionTag objects
    """
    if len(samples) != len(distances_standardized):
        raise ValueError(
            f"Sample count mismatch: {len(samples)} samples vs "
            f"{len(distances_standardized)} distances"
        )

    tags = []

    for sample, d_tilde in zip(samples, distances_standardized):
        d_tilde_val = float(d_tilde.item())
        abs_d_tilde = abs(d_tilde_val)
        borderline_distance = abs_d_tilde - config.theta_borderline

        # Assign friction level based on thresholds
        if abs_d_tilde < config.theta_high_friction:
            friction_level = "high"
            rationale = (
                f"Very close to boundary (|d̃|={abs_d_tilde:.4f} < {config.theta_high_friction}). "
                "Expect model uncertainty, hedging, or self-correction."
            )
        elif abs_d_tilde < config.theta_borderline:
            friction_level = "medium"
            rationale = (
                f"Borderline region (|d̃|={abs_d_tilde:.4f} < {config.theta_borderline}). "
                "May show some uncertainty or qualification."
            )
        else:
            friction_level = "low"
            rationale = (
                f"Confident classification (|d̃|={abs_d_tilde:.4f} ≥ {config.theta_borderline}). "
                "Expect stable, confident outputs."
            )

        tags.append(FrictionTag(
            sample_id=sample.sample_id,
            friction_level=friction_level,
            d_tilde_standardized=d_tilde_val,
            abs_d_tilde=abs_d_tilde,
            borderline_distance=borderline_distance,
            rationale=rationale
        ))

    return tags


def analyze_friction_distribution(tags: List[FrictionTag]) -> Dict[str, Any]:
    """
    Compute distribution statistics for friction tags.

    Args:
        tags: List of FrictionTag objects

    Returns:
        Dictionary with distribution counts and fractions
    """
    total = len(tags)

    # Count by friction level
    low_count = sum(1 for t in tags if t.friction_level == "low")
    medium_count = sum(1 for t in tags if t.friction_level == "medium")
    high_count = sum(1 for t in tags if t.friction_level == "high")

    return {
        "total_samples": total,
        "counts": {
            "low": low_count,
            "medium": medium_count,
            "high": high_count
        },
        "fractions": {
            "low": low_count / total if total > 0 else 0.0,
            "medium": medium_count / total if total > 0 else 0.0,
            "high": high_count / total if total > 0 else 0.0
        }
    }


def save_friction_artifact(
    tags: List[FrictionTag],
    ref_name: str,
    ref_hash: str,
    config: FrictionConfig,
    output_dir: Path
) -> Path:
    """
    Save friction artifact to runs/friction_tags_<hash>.json.

    Args:
        tags: List of FrictionTag objects
        ref_name: Reference set name
        ref_hash: Reference set hash
        config: Friction configuration
        output_dir: Directory to save artifact (typically runs/)

    Returns:
        Path to saved artifact
    """
    # Compute friction hash
    sample_ids_str = "".join(sorted([t.sample_id for t in tags]))
    config_str = f"{config.theta_borderline}_{config.theta_high_friction}"
    hash_input = f"{sample_ids_str}_{config_str}_{ref_hash}"
    friction_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Compute distribution
    distribution = analyze_friction_distribution(tags)

    # Build artifact
    artifact = {
        "friction_id": f"friction_{friction_hash}",
        "timestamp": datetime.now().isoformat(),
        "reference_set": {
            "ref_name": ref_name,
            "ref_hash": ref_hash
        },
        "friction_config": {
            "theta_borderline": config.theta_borderline,
            "theta_high_friction": config.theta_high_friction
        },
        "friction_distribution": distribution["counts"],
        "friction_fractions": distribution["fractions"],
        "tags": [
            {
                "sample_id": t.sample_id,
                "friction_level": t.friction_level,
                "d_tilde_standardized": t.d_tilde_standardized,
                "abs_d_tilde": t.abs_d_tilde,
                "borderline_distance": t.borderline_distance,
                "rationale": t.rationale
            }
            for t in tags
        ]
    }

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write artifact
    artifact_path = output_dir / f"friction_tags_{friction_hash}.json"
    with artifact_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    return artifact_path


def load_friction_artifact(artifact_path: Path) -> Dict[str, Any]:
    """
    Load friction artifact from JSON.

    Args:
        artifact_path: Path to friction artifact

    Returns:
        Friction artifact dictionary
    """
    with artifact_path.open("r") as f:
        return json.load(f)
