"""
Phase D integrated evaluation pipeline.

Combines semantic transforms (Phase B), perturbation testing (Phase C calibration),
and friction stratification (Phase C tagging) for comprehensive boundary stability
evaluation.

Implements 4 evaluation modes:
1. Semantic-only: Baseline semantic transform evaluation
2. Perturbation-only: Noise injection stratified by friction
3. Combined: Semantic transforms THEN perturbation
4. Stratified analysis: Cross-mode breakdown by friction level
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import json

import torch
from torch import nn

from tier2.dataset import SyntheticSample
from tier2.model import get_embeddings
from tier2.reference import ReferenceSetStats
from tier2.metrics import compute_raw_boundary_distance, compute_standardized_distance
from tier2.friction import FrictionTag


DEFAULT_SEED = 42


@dataclass
class IntegratedEvalConfig:
    """Configuration for integrated evaluation."""
    model_checkpoint: str
    reference_stats_path: str
    transform_suite_path: str
    calibration_artifact_path: str
    friction_artifact_path: str
    n_perturbation_samples: int = 10
    seed: int = DEFAULT_SEED


@dataclass
class SemanticModeResult:
    """Results from semantic-only evaluation."""
    transform_id: str
    category: str
    original_text: str
    transformed_text: str
    original_d_tilde: float
    transformed_d_tilde: float
    delta_d_tilde: float
    flip_occurred: bool


@dataclass
class PerturbationModeResult:
    """Results from perturbation-only evaluation."""
    sample_id: str
    friction_level: str
    d_tilde_original: float
    flip_rate: float
    n_flips: int
    n_perturbations: int


@dataclass
class CombinedModeResult:
    """Results from semantic + perturbation evaluation."""
    transform_id: str
    category: str
    original_text: str
    transformed_text: str
    original_friction_level: str
    semantic_delta_d_tilde: float
    semantic_flip_occurred: bool
    original_perturbation_flip_rate: float
    transformed_perturbation_flip_rate: float
    compound_effect: float  # transformed_flip_rate - original_flip_rate


def run_semantic_mode(
    model: nn.Module,
    transform_suite: List[Dict],
    ref_stats: ReferenceSetStats,
    device: torch.device,
    seed: int
) -> List[SemanticModeResult]:
    """
    Run semantic-only evaluation (Phase B baseline).

    For each transform:
    1. Compute original d̃(x)
    2. Compute transformed d̃(x')
    3. Compute Δd̃ = d̃(x') - d̃(x)
    4. Check for boundary flip

    Args:
        model: Trained classifier in eval mode
        transform_suite: List of transform dictionaries with original/transformed text
        ref_stats: Reference set statistics for standardization
        device: torch device
        seed: Random seed for embeddings

    Returns:
        List of SemanticModeResult objects
    """
    model.eval()
    results = []

    total_transforms = len(transform_suite)
    for idx, transform in enumerate(transform_suite):
        # Progress logging every 10 transforms
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    Progress: {idx + 1}/{total_transforms} transforms processed ({100*(idx+1)/total_transforms:.1f}%)")

        transform_id = transform["transform_id"]
        category = transform["category"]
        original_text = transform["original_text"]
        transformed_text = transform["transformed_text"]

        # Compute original boundary distance
        original_embedding = get_embeddings([original_text], device, seed=seed)
        with torch.no_grad():
            logits_orig = model(original_embedding)
            d_orig = compute_raw_boundary_distance(logits_orig)
            d_tilde_orig = compute_standardized_distance(
                d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
            )

        # Compute transformed boundary distance
        transformed_embedding = get_embeddings([transformed_text], device, seed=seed)
        with torch.no_grad():
            logits_trans = model(transformed_embedding)
            d_trans = compute_raw_boundary_distance(logits_trans)
            d_tilde_trans = compute_standardized_distance(
                d_trans, ref_stats.mu_ref, ref_stats.sigma_ref
            )

        # Compute change and flip
        delta_d_tilde = float((d_tilde_trans - d_tilde_orig).item())
        flip_occurred = (torch.sign(d_tilde_orig) != torch.sign(d_tilde_trans)).item()

        results.append(SemanticModeResult(
            transform_id=transform_id,
            category=category,
            original_text=original_text,
            transformed_text=transformed_text,
            original_d_tilde=float(d_tilde_orig.item()),
            transformed_d_tilde=float(d_tilde_trans.item()),
            delta_d_tilde=delta_d_tilde,
            flip_occurred=flip_occurred
        ))

    return results


def run_perturbation_mode(
    model: nn.Module,
    samples: List[SyntheticSample],
    friction_tags: List[FrictionTag],
    ref_stats: ReferenceSetStats,
    epsilon: float,
    n_perturbations: int,
    device: torch.device,
    seed: int
) -> List[PerturbationModeResult]:
    """
    Run perturbation-only evaluation.

    For each sample:
    1. Compute original d̃(x)
    2. Apply n_perturbations noise injections
    3. Measure flip rate (fraction crossing boundary)
    4. Stratify by friction level

    Args:
        model: Trained classifier in eval mode
        samples: List of evaluation samples
        friction_tags: Friction annotations for samples
        ref_stats: Reference set statistics
        epsilon: Calibrated noise scale
        n_perturbations: Number of perturbations per sample
        device: torch device
        seed: Random seed

    Returns:
        List of PerturbationModeResult objects
    """
    model.eval()
    results = []

    # Create friction lookup
    friction_map = {tag.sample_id: tag for tag in friction_tags}

    # Set up deterministic perturbations with manual seed control
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    total_samples = len(samples)
    for idx, sample in enumerate(samples):
        # Progress logging every 100 samples
        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx + 1}/{total_samples} samples processed ({100*(idx+1)/total_samples:.1f}%)")

        friction_tag = friction_map.get(sample.sample_id)
        if friction_tag is None:
            # Skip samples without friction tags
            continue

        # Compute original boundary distance
        original_embedding = get_embeddings([sample.text], device, seed=seed)
        with torch.no_grad():
            logits_orig = model(original_embedding)
            d_orig = compute_raw_boundary_distance(logits_orig)
            d_tilde_orig = compute_standardized_distance(
                d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
            )

        # Apply perturbations
        flips = 0
        for _ in range(n_perturbations):
            noise = torch.randn_like(original_embedding) * epsilon
            perturbed_embedding = original_embedding + noise

            with torch.no_grad():
                logits_pert = model(perturbed_embedding)
                d_pert = compute_raw_boundary_distance(logits_pert)
                d_tilde_pert = compute_standardized_distance(
                    d_pert, ref_stats.mu_ref, ref_stats.sigma_ref
                )

            # Check for boundary crossing
            if (torch.sign(d_tilde_orig) != torch.sign(d_tilde_pert)).item():
                flips += 1

        flip_rate = flips / n_perturbations

        results.append(PerturbationModeResult(
            sample_id=sample.sample_id,
            friction_level=friction_tag.friction_level,
            d_tilde_original=float(d_tilde_orig.item()),
            flip_rate=flip_rate,
            n_flips=flips,
            n_perturbations=n_perturbations
        ))

    return results


def run_combined_mode(
    model: nn.Module,
    transform_suite: List[Dict],
    samples: List[SyntheticSample],
    friction_tags: List[FrictionTag],
    ref_stats: ReferenceSetStats,
    epsilon: float,
    n_perturbations: int,
    device: torch.device,
    seed: int
) -> List[CombinedModeResult]:
    """
    Run combined semantic + perturbation evaluation.

    For each transform:
    1. Apply semantic transform
    2. Compute transformed d̃(x')
    3. Apply perturbations to BOTH original and transformed inputs
    4. Measure compound effect

    This tests: Does semantic transformation change perturbation robustness?

    Args:
        model: Trained classifier in eval mode
        transform_suite: List of transform dictionaries
        samples: List of samples (for friction lookup)
        friction_tags: Friction annotations
        ref_stats: Reference set statistics
        epsilon: Calibrated noise scale
        n_perturbations: Number of perturbations per sample
        device: torch device
        seed: Random seed

    Returns:
        List of CombinedModeResult objects
    """
    model.eval()
    results = []

    # Create friction lookup
    friction_map = {tag.sample_id: tag for tag in friction_tags}

    # Set up deterministic perturbations with manual seed control
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    total_transforms = len(transform_suite)
    for idx, transform in enumerate(transform_suite):
        # Progress logging every 5 transforms
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"    Progress: {idx + 1}/{total_transforms} transforms processed ({100*(idx+1)/total_transforms:.1f}%)")

        transform_id = transform["transform_id"]
        category = transform["category"]
        original_text = transform["original_text"]
        transformed_text = transform["transformed_text"]

        # Try to find matching friction tag (by text match, since transform_suite doesn't have sample_id)
        # For MVP, we'll match on original_text
        matching_tag = None
        for tag in friction_tags:
            # Find sample with matching text
            matching_samples = [s for s in samples if s.text == original_text]
            if matching_samples:
                matching_sample = matching_samples[0]
                if tag.sample_id == matching_sample.sample_id:
                    matching_tag = tag
                    break

        if matching_tag is None:
            # Skip transforms without friction tags
            continue

        # Compute semantic distances
        original_embedding = get_embeddings([original_text], device, seed=seed)
        transformed_embedding = get_embeddings([transformed_text], device, seed=seed)

        with torch.no_grad():
            # Original
            logits_orig = model(original_embedding)
            d_orig = compute_raw_boundary_distance(logits_orig)
            d_tilde_orig = compute_standardized_distance(
                d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
            )

            # Transformed
            logits_trans = model(transformed_embedding)
            d_trans = compute_raw_boundary_distance(logits_trans)
            d_tilde_trans = compute_standardized_distance(
                d_trans, ref_stats.mu_ref, ref_stats.sigma_ref
            )

        semantic_delta = float((d_tilde_trans - d_tilde_orig).item())
        semantic_flip = (torch.sign(d_tilde_orig) != torch.sign(d_tilde_trans)).item()

        # Measure perturbation robustness of ORIGINAL input
        original_flips = 0
        for _ in range(n_perturbations):
            noise = torch.randn_like(original_embedding) * epsilon
            perturbed_embedding = original_embedding + noise

            with torch.no_grad():
                logits_pert = model(perturbed_embedding)
                d_pert = compute_raw_boundary_distance(logits_pert)
                d_tilde_pert = compute_standardized_distance(
                    d_pert, ref_stats.mu_ref, ref_stats.sigma_ref
                )

            if (torch.sign(d_tilde_orig) != torch.sign(d_tilde_pert)).item():
                original_flips += 1

        original_flip_rate = original_flips / n_perturbations

        # Measure perturbation robustness of TRANSFORMED input
        transformed_flips = 0
        for _ in range(n_perturbations):
            noise = torch.randn_like(transformed_embedding) * epsilon
            perturbed_embedding = transformed_embedding + noise

            with torch.no_grad():
                logits_pert = model(perturbed_embedding)
                d_pert = compute_raw_boundary_distance(logits_pert)
                d_tilde_pert = compute_standardized_distance(
                    d_pert, ref_stats.mu_ref, ref_stats.sigma_ref
                )

            if (torch.sign(d_tilde_trans) != torch.sign(d_tilde_pert)).item():
                transformed_flips += 1

        transformed_flip_rate = transformed_flips / n_perturbations

        # Compound effect: did transform make sample MORE or LESS robust?
        compound_effect = transformed_flip_rate - original_flip_rate

        results.append(CombinedModeResult(
            transform_id=transform_id,
            category=category,
            original_text=original_text,
            transformed_text=transformed_text,
            original_friction_level=matching_tag.friction_level,
            semantic_delta_d_tilde=semantic_delta,
            semantic_flip_occurred=semantic_flip,
            original_perturbation_flip_rate=original_flip_rate,
            transformed_perturbation_flip_rate=transformed_flip_rate,
            compound_effect=compound_effect
        ))

    return results


def analyze_stratified_results(
    semantic_results: List[SemanticModeResult],
    perturbation_results: List[PerturbationModeResult],
    combined_results: List[CombinedModeResult]
) -> Dict[str, Any]:
    """
    Cross-mode analysis stratified by friction level.

    Computes:
    - Semantic metrics by friction (mean Δd̃, flip rate)
    - Perturbation metrics by friction (mean flip rate)
    - Combined metrics by friction
    - Category analysis (preserving/changing/gotcha)
    - Interaction effects

    Args:
        semantic_results: Results from semantic mode
        perturbation_results: Results from perturbation mode
        combined_results: Results from combined mode

    Returns:
        Dictionary with stratified analysis
    """
    # Perturbation stratification (directly available)
    pert_by_friction = defaultdict(list)
    for result in perturbation_results:
        pert_by_friction[result.friction_level].append(result)

    perturbation_by_friction = {}
    for friction_level in ["low", "medium", "high"]:
        results = pert_by_friction[friction_level]
        if results:
            flip_rates = [r.flip_rate for r in results]
            perturbation_by_friction[friction_level] = {
                "mean_flip_rate": sum(flip_rates) / len(flip_rates),
                "n_samples": len(results)
            }
        else:
            perturbation_by_friction[friction_level] = {
                "mean_flip_rate": 0.0,
                "n_samples": 0
            }

    # Combined stratification
    combined_by_friction = defaultdict(list)
    for result in combined_results:
        combined_by_friction[result.original_friction_level].append(result)

    combined_by_friction_stats = {}
    for friction_level in ["low", "medium", "high"]:
        results = combined_by_friction[friction_level]
        if results:
            semantic_deltas = [r.semantic_delta_d_tilde for r in results]
            semantic_flips = [r.semantic_flip_occurred for r in results]
            orig_flip_rates = [r.original_perturbation_flip_rate for r in results]
            trans_flip_rates = [r.transformed_perturbation_flip_rate for r in results]
            compound_effects = [r.compound_effect for r in results]

            combined_by_friction_stats[friction_level] = {
                "mean_semantic_delta": sum(semantic_deltas) / len(semantic_deltas),
                "semantic_flip_rate": sum(semantic_flips) / len(semantic_flips),
                "mean_original_flip_rate": sum(orig_flip_rates) / len(orig_flip_rates),
                "mean_transformed_flip_rate": sum(trans_flip_rates) / len(trans_flip_rates),
                "mean_compound_effect": sum(compound_effects) / len(compound_effects),
                "n_samples": len(results)
            }
        else:
            combined_by_friction_stats[friction_level] = {
                "mean_semantic_delta": 0.0,
                "semantic_flip_rate": 0.0,
                "mean_original_flip_rate": 0.0,
                "mean_transformed_flip_rate": 0.0,
                "mean_compound_effect": 0.0,
                "n_samples": 0
            }

    # Category analysis (preserving/changing/gotcha)
    semantic_by_category = defaultdict(list)
    for result in semantic_results:
        semantic_by_category[result.category].append(result)

    category_analysis = {}
    for category in ["preserving", "changing", "gotcha"]:
        results = semantic_by_category[category]
        if results:
            deltas = [r.delta_d_tilde for r in results]
            flips = [r.flip_occurred for r in results]
            category_analysis[category] = {
                "mean_delta_d_tilde": sum(deltas) / len(deltas),
                "abs_mean_delta": sum(abs(d) for d in deltas) / len(deltas),
                "flip_rate": sum(flips) / len(flips),
                "n_transforms": len(results)
            }
        else:
            category_analysis[category] = {
                "mean_delta_d_tilde": 0.0,
                "abs_mean_delta": 0.0,
                "flip_rate": 0.0,
                "n_transforms": 0
            }

    # Overall summaries
    overall_semantic = {
        "n_transforms": len(semantic_results),
        "mean_delta_d_tilde": sum(r.delta_d_tilde for r in semantic_results) / len(semantic_results) if semantic_results else 0.0,
        "flip_rate": sum(r.flip_occurred for r in semantic_results) / len(semantic_results) if semantic_results else 0.0
    }

    overall_perturbation = {
        "n_samples": len(perturbation_results),
        "mean_flip_rate": sum(r.flip_rate for r in perturbation_results) / len(perturbation_results) if perturbation_results else 0.0
    }

    overall_combined = {
        "n_transforms": len(combined_results),
        "mean_compound_effect": sum(r.compound_effect for r in combined_results) / len(combined_results) if combined_results else 0.0
    }

    return {
        "semantic_overall": overall_semantic,
        "perturbation_overall": overall_perturbation,
        "combined_overall": overall_combined,
        "perturbation_by_friction": perturbation_by_friction,
        "combined_by_friction": combined_by_friction_stats,
        "category_analysis": category_analysis
    }


def save_integrated_eval_results(
    results_dir: Path,
    run_id: str,
    semantic_results: List[SemanticModeResult],
    perturbation_results: List[PerturbationModeResult],
    combined_results: List[CombinedModeResult],
    stratified_analysis: Dict[str, Any],
    config: Dict[str, Any],
    environment: Dict[str, Any]
) -> None:
    """
    Save comprehensive evaluation results to JSON files.

    Args:
        results_dir: Output directory
        run_id: Timestamped run identifier
        semantic_results: Semantic mode results
        perturbation_results: Perturbation mode results
        combined_results: Combined mode results
        stratified_analysis: Cross-mode analysis
        config: Evaluation configuration
        environment: Environment snapshot
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save semantic results
    semantic_path = results_dir / "semantic_results.json"
    with semantic_path.open("w") as f:
        json.dump([asdict(r) for r in semantic_results], f, indent=2)

    # Save perturbation results
    perturbation_path = results_dir / "perturbation_results.json"
    with perturbation_path.open("w") as f:
        json.dump([asdict(r) for r in perturbation_results], f, indent=2)

    # Save combined results
    combined_path = results_dir / "combined_results.json"
    with combined_path.open("w") as f:
        json.dump([asdict(r) for r in combined_results], f, indent=2)

    # Save stratified analysis
    analysis_path = results_dir / "stratified_analysis.json"
    with analysis_path.open("w") as f:
        json.dump(stratified_analysis, f, indent=2)

    # Save comprehensive summary
    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "phase": "phase_d_integrated_eval",
        "environment": environment,
        "config": config,
        "evaluation_modes": {
            "semantic_only": stratified_analysis["semantic_overall"],
            "perturbation_only": stratified_analysis["perturbation_overall"],
            "combined": stratified_analysis["combined_overall"]
        },
        "stratified_analysis": {
            "perturbation_by_friction": stratified_analysis["perturbation_by_friction"],
            "combined_by_friction": stratified_analysis["combined_by_friction"],
            "category_analysis": stratified_analysis["category_analysis"]
        }
    }

    summary_path = results_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
