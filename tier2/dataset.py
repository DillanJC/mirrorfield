"""
Synthetic sentiment dataset generation for Phase B MVP.

Generates template-based sentiment classification data with deterministic seeding
for reproducibility.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import hashlib
import random

DEFAULT_SEED = 42


@dataclass
class SyntheticSample:
    """A single training sample with text and label."""

    text: str
    label: int  # 0=negative, 1=positive
    sample_id: str  # Unique identifier


def generate_synthetic_dataset(
    n_samples: int = 2000,
    seed: int = DEFAULT_SEED
) -> Tuple[List[SyntheticSample], Dict[str, Any]]:
    """
    Generate synthetic binary sentiment dataset using template-based approach.

    Templates provide simple but diverse positive/negative sentiment examples
    without requiring external LLM calls.

    Args:
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        samples: List of SyntheticSample objects
        metadata: Generation metadata including hash for reproducibility
    """
    random.seed(seed)

    # Template patterns for positive sentiment
    positive_templates = [
        "I love this {product}",
        "I adore this {product}",
        "This {product} is amazing",
        "This {product} is excellent",
        "Great {product}, highly recommend",
        "Fantastic {product}, very satisfied",
        "This {product} exceeded my expectations",
        "Best {product} I've ever used",
        "I'm very happy with this {product}",
        "Absolutely wonderful {product}",
        "This {product} is perfect",
        "Outstanding {product} quality",
        "I enjoy using this {product}",
        "This {product} works wonderfully",
        "Impressive {product} performance",
    ]

    # Template patterns for negative sentiment
    negative_templates = [
        "I hate this {product}",
        "I dislike this {product}",
        "This {product} is terrible",
        "This {product} is awful",
        "Bad {product}, would not recommend",
        "Disappointing {product}, not satisfied",
        "This {product} failed to meet expectations",
        "Worst {product} I've ever used",
        "I'm very unhappy with this {product}",
        "Absolutely horrible {product}",
        "This {product} is defective",
        "Poor {product} quality",
        "I regret buying this {product}",
        "This {product} doesn't work properly",
        "Terrible {product} performance",
    ]

    # Slot fillers for product/service
    products = [
        "product", "service", "experience", "item", "purchase",
        "tool", "device", "software", "app", "feature",
        "system", "solution", "platform", "offering", "package"
    ]

    samples = []

    # Generate balanced dataset (50% positive, 50% negative)
    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    # Generate positive samples
    for i in range(n_positive):
        template = random.choice(positive_templates)
        product = random.choice(products)
        text = template.format(product=product)
        sample_id = f"pos_{i:04d}"

        samples.append(SyntheticSample(
            text=text,
            label=1,
            sample_id=sample_id
        ))

    # Generate negative samples
    for i in range(n_negative):
        template = random.choice(negative_templates)
        product = random.choice(products)
        text = template.format(product=product)
        sample_id = f"neg_{i:04d}"

        samples.append(SyntheticSample(
            text=text,
            label=0,
            sample_id=sample_id
        ))

    # Shuffle to mix positive and negative
    random.shuffle(samples)

    # Compute metadata
    class_counts = {0: n_negative, 1: n_positive}

    # Hash for reproducibility tracking
    content = "".join(sorted([s.sample_id + s.text for s in samples]))
    dataset_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

    metadata = {
        "n_samples": n_samples,
        "class_balance": class_counts,
        "generation_strategy": "template-based",
        "seed": seed,
        "hash": dataset_hash
    }

    return samples, metadata


def split_dataset(
    samples: List[SyntheticSample],
    train_fraction: float = 0.8,
    seed: int = DEFAULT_SEED
) -> Tuple[List[SyntheticSample], List[SyntheticSample], Dict[str, Any]]:
    """
    Split dataset into train and validation sets with stratification.

    Args:
        samples: Full dataset
        train_fraction: Fraction for training (0.8 = 80% train, 20% val)
        seed: Random seed for reproducibility

    Returns:
        train_samples: Training set
        val_samples: Validation set
        split_metadata: Split statistics
    """
    random.seed(seed)

    # Separate by class
    positive_samples = [s for s in samples if s.label == 1]
    negative_samples = [s for s in samples if s.label == 0]

    # Shuffle within each class
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # Split each class
    n_train_pos = int(len(positive_samples) * train_fraction)
    n_train_neg = int(len(negative_samples) * train_fraction)

    train_samples = (
        positive_samples[:n_train_pos] +
        negative_samples[:n_train_neg]
    )
    val_samples = (
        positive_samples[n_train_pos:] +
        negative_samples[n_train_neg:]
    )

    # Shuffle final sets
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    split_metadata = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "train_fraction": train_fraction,
        "class_balance_train": {
            0: sum(1 for s in train_samples if s.label == 0),
            1: sum(1 for s in train_samples if s.label == 1)
        },
        "class_balance_val": {
            0: sum(1 for s in val_samples if s.label == 0),
            1: sum(1 for s in val_samples if s.label == 1)
        },
        "seed": seed
    }

    return train_samples, val_samples, split_metadata


def split_dataset_3way(
    samples: List[SyntheticSample],
    train_fraction: float = 0.6,
    calib_fraction: float = 0.2,
    seed: int = DEFAULT_SEED
) -> Tuple[List[SyntheticSample], List[SyntheticSample], List[SyntheticSample], Dict[str, Any]]:
    """
    Split dataset into train/calibration/evaluation sets with stratification.

    Calibration set is used for tuning perturbation noise ε (Phase C).
    Evaluation set is held-out for final testing.

    Args:
        samples: Full dataset
        train_fraction: Fraction for training (default: 0.6)
        calib_fraction: Fraction for calibration (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        train_samples: Training set
        calib_samples: Calibration set
        eval_samples: Evaluation set
        split_metadata: Split statistics
    """
    random.seed(seed)

    # Compute eval fraction
    eval_fraction = 1.0 - train_fraction - calib_fraction
    if eval_fraction < 0:
        raise ValueError(
            f"Invalid fractions: train={train_fraction}, calib={calib_fraction} "
            f"must sum to ≤ 1.0"
        )

    # Separate by class
    positive_samples = [s for s in samples if s.label == 1]
    negative_samples = [s for s in samples if s.label == 0]

    # Shuffle within each class
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # Split each class into three parts
    n_train_pos = int(len(positive_samples) * train_fraction)
    n_calib_pos = int(len(positive_samples) * calib_fraction)

    n_train_neg = int(len(negative_samples) * train_fraction)
    n_calib_neg = int(len(negative_samples) * calib_fraction)

    # Slice positive samples
    train_pos = positive_samples[:n_train_pos]
    calib_pos = positive_samples[n_train_pos:n_train_pos + n_calib_pos]
    eval_pos = positive_samples[n_train_pos + n_calib_pos:]

    # Slice negative samples
    train_neg = negative_samples[:n_train_neg]
    calib_neg = negative_samples[n_train_neg:n_train_neg + n_calib_neg]
    eval_neg = negative_samples[n_train_neg + n_calib_neg:]

    # Combine and shuffle
    train_samples = train_pos + train_neg
    calib_samples = calib_pos + calib_neg
    eval_samples = eval_pos + eval_neg

    random.shuffle(train_samples)
    random.shuffle(calib_samples)
    random.shuffle(eval_samples)

    split_metadata = {
        "n_train": len(train_samples),
        "n_calib": len(calib_samples),
        "n_eval": len(eval_samples),
        "train_fraction": train_fraction,
        "calib_fraction": calib_fraction,
        "eval_fraction": eval_fraction,
        "class_balance_train": {
            0: sum(1 for s in train_samples if s.label == 0),
            1: sum(1 for s in train_samples if s.label == 1)
        },
        "class_balance_calib": {
            0: sum(1 for s in calib_samples if s.label == 0),
            1: sum(1 for s in calib_samples if s.label == 1)
        },
        "class_balance_eval": {
            0: sum(1 for s in eval_samples if s.label == 0),
            1: sum(1 for s in eval_samples if s.label == 1)
        },
        "seed": seed
    }

    return train_samples, calib_samples, eval_samples, split_metadata
