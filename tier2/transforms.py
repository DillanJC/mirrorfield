"""
Semantic transform dataclasses and validation logic.

Defines the structure for Tier-2 semantic transforms (preserving/changing/gotcha)
and suite-level validation tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import hashlib


@dataclass
class SemanticTransform:
    """A single semantic transform pair with metadata."""

    original_text: str
    transformed_text: str
    category: str  # "preserving", "changing", "gotcha"
    transform_id: str
    description: str
    validated: bool = False
    validator_notes: str = ""

    def __post_init__(self):
        """Validate category is one of the allowed values."""
        allowed_categories = {"preserving", "changing", "gotcha"}
        if self.category not in allowed_categories:
            raise ValueError(
                f"Invalid category '{self.category}'. Must be one of {allowed_categories}"
            )


@dataclass
class TransformSuite:
    """Collection of semantic transforms with generation and validation metadata."""

    transforms: List[SemanticTransform]
    suite_id: str
    llm_model: str = "manual"  # e.g., "gpt-4", "claude-3.5-sonnet", "manual"
    generation_prompt: str = ""
    validation_count: int = 0
    suite_hash: str = ""

    def __post_init__(self):
        """Compute suite hash if not provided."""
        if not self.suite_hash:
            self.suite_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of transform suite."""
        content = "".join(
            sorted([
                f"{t.transform_id}_{t.original_text}_{t.transformed_text}"
                for t in self.transforms
            ])
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def get_validated_transforms(self) -> List[SemanticTransform]:
        """Return only transforms that have been human-validated."""
        return [t for t in self.transforms if t.validated]

    def get_category_counts(self) -> Dict[str, int]:
        """Count transforms by category."""
        counts = {"preserving": 0, "changing": 0, "gotcha": 0}
        for transform in self.transforms:
            counts[transform.category] += 1
        return counts

    def compute_hash(self):
        """Recompute suite hash and update suite_hash attribute."""
        self.suite_hash = self._compute_hash()

    def validate_suite_requirements(self) -> Dict[str, Any]:
        """
        Check if suite meets Phase B validation requirements.

        From DEFINITIONS_FREEZE Section D:
        - Minimum 5 human spot-checks total
        - At least 1 validation per category recommended

        Returns:
            Dict with validation status and breakdown
        """
        validated = self.get_validated_transforms()
        validation_count = len(validated)

        # Count validated transforms per category
        category_validated = {"preserving": 0, "changing": 0, "gotcha": 0}
        for transform in validated:
            category_validated[transform.category] += 1

        meets_requirement = validation_count >= 5

        return {
            "meets_requirement": meets_requirement,
            "total_validated": validation_count,
            "min_required": 5,
            "categories_validated": category_validated,
            "all_categories_covered": all(count > 0 for count in category_validated.values())
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return transform_suite_to_dict(self)


def transform_suite_to_dict(suite: TransformSuite) -> Dict[str, Any]:
    """Convert TransformSuite to JSON-serializable dictionary."""
    validation_report = suite.validate_suite_requirements()

    return {
        "suite_id": suite.suite_id,
        "llm_model": suite.llm_model,
        "generation_prompt": suite.generation_prompt,
        "validation": {
            "total_validated": validation_report["total_validated"],
            "min_required": validation_report["min_required"],
            "meets_requirement": validation_report["meets_requirement"],
            "categories_validated": validation_report["categories_validated"],
            "all_categories_covered": validation_report["all_categories_covered"]
        },
        "suite_hash": suite.suite_hash,
        "transforms": [
            {
                "transform_id": t.transform_id,
                "category": t.category,
                "original_text": t.original_text,
                "transformed_text": t.transformed_text,
                "description": t.description,
                "validated": t.validated,
                "validator_notes": t.validator_notes
            }
            for t in suite.transforms
        ]
    }


def transform_suite_from_dict(data: Dict[str, Any]) -> TransformSuite:
    """Load TransformSuite from dictionary."""
    transforms = [
        SemanticTransform(
            original_text=t["original_text"],
            transformed_text=t["transformed_text"],
            category=t["category"],
            transform_id=t["transform_id"],
            description=t["description"],
            validated=t.get("validated", False),
            validator_notes=t.get("validator_notes", "")
        )
        for t in data["transforms"]
    ]

    return TransformSuite(
        transforms=transforms,
        suite_id=data["suite_id"],
        llm_model=data.get("llm_model", "manual"),
        generation_prompt=data.get("generation_prompt", ""),
        validation_count=data.get("validation", {}).get("total_validated", 0),
        suite_hash=data.get("suite_hash", "")
    )
