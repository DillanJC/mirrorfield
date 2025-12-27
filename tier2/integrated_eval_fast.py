"""
Phase D integrated evaluation pipeline (OPTIMIZED VERSION).

Batched embeddings for 5-10× speedup over tier2/integrated_eval.py.

Key optimization: Batch all unique texts and embed once, then cache results.
This avoids repeated SentenceTransformer model loading and enables GPU batching.

Performance:
- Original: ~2 hours (2000+ sequential embedding calls)
- Optimized: ~15-20 minutes (1-2 batched embedding calls)

Maintains same outputs as integrated_eval.py for verification.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
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
    compound_effect: float


def precompute_embeddings_batch(
    texts: List[str],
    device: torch.device,
    seed: int,
    batch_size: int = 64
) -> Dict[str, torch.Tensor]:
    """
    Precompute embeddings for all unique texts in a single batched call.

    This is the KEY optimization: instead of calling get_embeddings 2000+ times
    with single texts, we call it once with all unique texts batched.

    Args:
        texts: List of all texts to embed (may contain duplicates)
        device: torch device
        seed: Random seed for determinism
        batch_size: Batch size for sentence-transformer encoding

    Returns:
        Dictionary mapping text -> embedding tensor (shape: [384])
    """
    # Get unique texts to avoid redundant computation
    unique_texts = list(set(texts))
    print(f"    Batched embedding: {len(unique_texts)} unique texts (from {len(texts)} total)")

    # Single batched call to get_embeddings (this is the speedup!)
    embeddings = get_embeddings(unique_texts, device, seed=seed, batch_size=batch_size)

    # Create lookup dictionary: text -> embedding
    embedding_cache = {text: embeddings[i] for i, text in enumerate(unique_texts)}

    return embedding_cache


def run_semantic_mode_fast(
    model: nn.Module,
    transform_suite: List[Dict],
    ref_stats: ReferenceSetStats,
    device: torch.device,
    seed: int
) -> List[SemanticModeResult]:
    """
    Optimized semantic-only evaluation with batched embeddings.

    Speedup: Instead of 60 separate embedding calls (30 transforms × 2 texts),
    we make 1 batched call for all unique texts.
    """
    model.eval()
    results = []

    # Collect all texts that need embedding
    all_texts = []
    for transform in transform_suite:
        all_texts.append(transform["original_text"])
        all_texts.append(transform["transformed_text"])

    # Precompute all embeddings in one batched call
    print(f"  Precomputing embeddings for {len(all_texts)} texts...")
    embedding_cache = precompute_embeddings_batch(all_texts, device, seed)

    # Now process transforms using cached embeddings
    total_transforms = len(transform_suite)
    for idx, transform in enumerate(transform_suite):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"    Progress: {idx + 1}/{total_transforms} transforms processed ({100*(idx+1)/total_transforms:.1f}%)")

        transform_id = transform["transform_id"]
        category = transform["category"]
        original_text = transform["original_text"]
        transformed_text = transform["transformed_text"]

        # Lookup cached embeddings (no re-computation!)
        original_embedding = embedding_cache[original_text].unsqueeze(0)  # Add batch dim
        transformed_embedding = embedding_cache[transformed_text].unsqueeze(0)

        # Compute boundary distances
        with torch.no_grad():
            logits_orig = model(original_embedding)
            d_orig = compute_raw_boundary_distance(logits_orig)
            d_tilde_orig = compute_standardized_distance(
                d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
            )

            logits_trans = model(transformed_embedding)
            d_trans = compute_raw_boundary_distance(logits_trans)
            d_tilde_trans = compute_standardized_distance(
                d_trans, ref_stats.mu_ref, ref_stats.sigma_ref
            )

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


def run_perturbation_mode_fast(
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
    Optimized perturbation-only evaluation with batched embeddings.

    Speedup: Instead of 2000 separate embedding calls (1 per sample),
    we make 1 batched call for all sample texts.
    """
    model.eval()
    results = []

    # Create friction lookup
    friction_map = {tag.sample_id: tag for tag in friction_tags}

    # Collect all sample texts
    sample_texts = [sample.text for sample in samples]

    # Precompute all embeddings in one batched call
    print(f"  Precomputing embeddings for {len(sample_texts)} samples...")
    embedding_cache = precompute_embeddings_batch(sample_texts, device, seed)

    # Set up deterministic perturbations
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    total_samples = len(samples)
    for idx, sample in enumerate(samples):
        if (idx + 1) % 100 == 0:
            print(f"    Progress: {idx + 1}/{total_samples} samples processed ({100*(idx+1)/total_samples:.1f}%)")

        friction_tag = friction_map.get(sample.sample_id)
        if friction_tag is None:
            continue

        # Lookup cached embedding (no re-computation!)
        original_embedding = embedding_cache[sample.text].unsqueeze(0)

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


def run_combined_mode_fast(
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
    Optimized combined semantic + perturbation evaluation with batched embeddings.

    Speedup: Instead of 60 separate embedding calls (30 transforms × 2 texts),
    we make 1 batched call for all unique texts.
    """
    model.eval()
    results = []

    # Create friction lookup
    friction_map = {tag.sample_id: tag for tag in friction_tags}

    # Collect all texts that need embedding (originals + transformed)
    all_texts = []
    for transform in transform_suite:
        all_texts.append(transform["original_text"])
        all_texts.append(transform["transformed_text"])

    # Precompute all embeddings in one batched call
    print(f"  Precomputing embeddings for {len(all_texts)} texts...")
    embedding_cache = precompute_embeddings_batch(all_texts, device, seed)

    # Set up deterministic perturbations
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    total_transforms = len(transform_suite)
    for idx, transform in enumerate(transform_suite):
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"    Progress: {idx + 1}/{total_transforms} transforms processed ({100*(idx+1)/total_transforms:.1f}%)")

        transform_id = transform["transform_id"]
        category = transform["category"]
        original_text = transform["original_text"]
        transformed_text = transform["transformed_text"]

        # Find matching friction tag
        matching_tag = None
        for tag in friction_tags:
            matching_samples = [s for s in samples if s.text == original_text]
            if matching_samples:
                matching_sample = matching_samples[0]
                if tag.sample_id == matching_sample.sample_id:
                    matching_tag = tag
                    break

        if matching_tag is None:
            continue

        # Lookup cached embeddings (no re-computation!)
        original_embedding = embedding_cache[original_text].unsqueeze(0)
        transformed_embedding = embedding_cache[transformed_text].unsqueeze(0)

        # Compute semantic distances
        with torch.no_grad():
            logits_orig = model(original_embedding)
            d_orig = compute_raw_boundary_distance(logits_orig)
            d_tilde_orig = compute_standardized_distance(
                d_orig, ref_stats.mu_ref, ref_stats.sigma_ref
            )

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


# Import remaining functions from original module (no changes needed)
from tier2.integrated_eval import analyze_stratified_results, save_integrated_eval_results
