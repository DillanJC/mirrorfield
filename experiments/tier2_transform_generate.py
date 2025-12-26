"""
Tier-2 Semantic Transform Generator (LLM-Assisted + Human Validation)

Interactive script for generating semantic transform suite with human validation.

Flow:
1. Load sample texts from synthetic dataset
2. For each category (preserving/changing/gotcha):
   - Display LLM prompt for user to run through GPT-4/Claude
   - Parse LLM output into SemanticTransform objects
3. Interactive validation (user spot-checks ≥5 transforms)
4. Save validated transform suite to runs/tier2_transforms_v1.json

Phase B requirement: ≥5 human validations total (≥1 per category)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

from tier2.dataset import generate_synthetic_dataset, SyntheticSample
from tier2.transforms import SemanticTransform, TransformSuite

DEFAULT_SEED = 42


# LLM Prompt Templates for each category
PRESERVING_PROMPT = """
# Task: Generate Meaning-Preserving Paraphrases

Generate 10 meaning-preserving paraphrases for binary sentiment classification. These transforms should keep the sentiment (positive/negative) EXACTLY the same while changing the wording.

**Examples:**
- Original: "I love this product" → Transformed: "I adore this product" (synonym substitution)
- Original: "This service is terrible" → Transformed: "This service is awful" (synonym substitution)
- Original: "Great experience, highly recommend" → Transformed: "Excellent experience, would definitely recommend" (synonym + elaboration)

**Source texts to transform:**
{source_texts}

**Output format (JSON list):**
[
  {{
    "original": "...",
    "transformed": "...",
    "description": "brief explanation of transform technique"
  }},
  ...
]

Generate 10 transforms that preserve sentiment while changing wording.
"""

CHANGING_PROMPT = """
# Task: Generate Intent-Flipping Transforms

Generate 10 intent-flipping transforms for binary sentiment classification. These transforms should REVERSE the sentiment (positive↔negative) while keeping the surface structure similar.

**Examples:**
- Original: "I love this product" → Transformed: "I hate this product" (negation via antonym)
- Original: "This service is excellent" → Transformed: "This service is terrible" (sentiment word flip)
- Original: "Best purchase I've ever made" → Transformed: "Worst purchase I've ever made" (superlative flip)

**Source texts to transform:**
{source_texts}

**Output format (JSON list):**
[
  {{
    "original": "...",
    "transformed": "...",
    "description": "brief explanation of how sentiment was flipped"
  }},
  ...
]

Generate 10 transforms that flip sentiment while maintaining surface structure.
"""

GOTCHA_PROMPT = """
# Task: Generate Gotcha Transforms (Surface-Preserving + Intent-Flip)

Generate 10 "gotcha" transforms that are TRICKY: they preserve surface structure but subtly flip intent. These are adversarial examples designed to fool the classifier.

**Examples:**
- Original: "I love this product" → Transformed: "I love how terrible this product is" (sarcasm via elaboration)
- Original: "Great service, very satisfied" → Transformed: "Great service, if you enjoy being disappointed" (ironic conditional)
- Original: "This is the worst app ever" → Transformed: "This is the worst app ever... said no one who actually used it" (negation via continuation)

**Source texts to transform:**
{source_texts}

**Output format (JSON list):**
[
  {{
    "original": "...",
    "transformed": "...",
    "description": "brief explanation of gotcha technique (sarcasm, irony, negation, etc.)"
  }},
  ...
]

Generate 10 gotcha transforms that look similar but flip intent.
"""


def load_sample_texts(n_samples: int = 20, seed: int = DEFAULT_SEED) -> List[SyntheticSample]:
    """Load sample texts from synthetic dataset for transform generation."""
    samples, _ = generate_synthetic_dataset(n_samples=n_samples, seed=seed)
    return samples


def format_source_texts(samples: List[SyntheticSample], n_show: int = 10) -> str:
    """Format sample texts for LLM prompt."""
    texts = []
    for i, sample in enumerate(samples[:n_show], 1):
        sentiment = "POSITIVE" if sample.label == 1 else "NEGATIVE"
        texts.append(f"{i}. [{sentiment}] \"{sample.text}\"")
    return "\n".join(texts)


def get_llm_prompt(category: str, samples: List[SyntheticSample]) -> str:
    """Generate LLM prompt for given category."""
    source_texts = format_source_texts(samples, n_show=10)

    if category == "preserving":
        return PRESERVING_PROMPT.format(source_texts=source_texts)
    elif category == "changing":
        return CHANGING_PROMPT.format(source_texts=source_texts)
    elif category == "gotcha":
        return GOTCHA_PROMPT.format(source_texts=source_texts)
    else:
        raise ValueError(f"Unknown category: {category}")


def parse_llm_output(llm_output: str, category: str) -> List[SemanticTransform]:
    """Parse LLM JSON output into SemanticTransform objects."""
    try:
        # Try to parse as JSON
        data = json.loads(llm_output)

        if not isinstance(data, list):
            raise ValueError("Expected JSON list, got: " + type(data).__name__)

        transforms = []
        for i, item in enumerate(data):
            transform = SemanticTransform(
                original_text=item["original"],
                transformed_text=item["transformed"],
                category=category,
                transform_id=f"{category}_{i:02d}",
                description=item.get("description", ""),
                validated=False,
                validator_notes=""
            )
            transforms.append(transform)

        return transforms

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print("Please ensure the LLM output is valid JSON (starts with [ and ends with ]).")
        return []
    except KeyError as e:
        print(f"ERROR: Missing required field in JSON: {e}")
        print("Each transform must have 'original', 'transformed', and 'description' fields.")
        return []


def interactive_validation(transforms: List[SemanticTransform]) -> List[SemanticTransform]:
    """Interactive validation workflow for transforms."""
    print("\n" + "=" * 60)
    print("Interactive Transform Validation")
    print("=" * 60)
    print(f"Review {len(transforms)} transforms.")
    print("You can spot-check any transforms to validate quality.")
    print("Phase B requires ≥5 total validations (≥1 per category).")
    print()

    validated_count = 0

    for i, transform in enumerate(transforms, 1):
        print(f"\nTransform {i}/{len(transforms)} [{transform.category}]")
        print(f"  Original:    \"{transform.original_text}\"")
        print(f"  Transformed: \"{transform.transformed_text}\"")
        print(f"  Description: {transform.description}")

        while True:
            response = input("  Validate this? (y/n/q to quit validation): ").strip().lower()

            if response == 'q':
                print(f"\nValidation stopped. {validated_count} transforms validated so far.")
                return transforms

            if response in ['y', 'n']:
                if response == 'y':
                    notes = input("  Validator notes (optional, press Enter to skip): ").strip()
                    transform.validated = True
                    transform.validator_notes = notes if notes else "Validated - quality check passed"
                    validated_count += 1
                    print(f"  [OK] Validated ({validated_count} total)")
                else:
                    print("  [SKIP] Not validated")
                break
            else:
                print("  Invalid input. Enter 'y' to validate, 'n' to skip, 'q' to quit.")

    print(f"\nValidation complete: {validated_count} transforms validated out of {len(transforms)}")
    return transforms


def main():
    parser = argparse.ArgumentParser(description="Generate semantic transform suite with LLM assistance")
    parser.add_argument("--output", "-o", default="runs/tier2_transforms_v1.json",
                        help="Output path for transform suite JSON (default: runs/tier2_transforms_v1.json)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for source text sampling (default: 42)")
    parser.add_argument("--n-source", type=int, default=20,
                        help="Number of source texts to sample (default: 20)")
    parser.add_argument("--llm-model", default="manual",
                        help="LLM model used for generation (default: manual)")

    args = parser.parse_args()

    print("=" * 60)
    print("Tier-2 Semantic Transform Generator")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Source texts: {args.n_source}")
    print(f"Output: {args.output}")
    print(f"LLM model: {args.llm_model}")
    print()

    # Load source texts
    print("Step 1: Loading source texts from synthetic dataset...")
    samples = load_sample_texts(n_samples=args.n_source, seed=args.seed)
    print(f"  Loaded {len(samples)} source samples")
    print()

    all_transforms = []

    # Generate transforms for each category
    categories = ["preserving", "changing", "gotcha"]

    for category in categories:
        print("=" * 60)
        print(f"Category: {category.upper()}")
        print("=" * 60)
        print()

        # Generate LLM prompt
        prompt = get_llm_prompt(category, samples)

        print("Step 2: LLM Prompt Generated")
        print("Copy the prompt below and paste it into your LLM (GPT-4, Claude, etc.):")
        print()
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        print()

        # Get LLM output from user
        print("Step 3: Paste LLM Output")
        print("After the LLM responds, copy the JSON output and paste it below.")
        print("(Type 'END' on a new line when done, or 'SKIP' to skip this category)")
        print()

        llm_lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            if line.strip().upper() == "SKIP":
                print(f"Skipping {category} category.")
                llm_lines = []
                break
            llm_lines.append(line)

        if not llm_lines:
            print(f"No transforms generated for {category}.")
            continue

        llm_output = "\n".join(llm_lines)

        # Parse LLM output
        transforms = parse_llm_output(llm_output, category)

        if not transforms:
            print(f"Failed to parse transforms for {category}. Skipping this category.")
            continue

        print(f"  Parsed {len(transforms)} {category} transforms")
        all_transforms.extend(transforms)
        print()

    if not all_transforms:
        print("ERROR: No transforms generated. Exiting.")
        return

    print()
    print("=" * 60)
    print(f"Transform Generation Complete: {len(all_transforms)} total transforms")
    print("=" * 60)

    # Count by category
    category_counts = {}
    for t in all_transforms:
        category_counts[t.category] = category_counts.get(t.category, 0) + 1

    for cat, count in category_counts.items():
        print(f"  {cat}: {count} transforms")

    print()

    # Interactive validation
    print("Step 4: Human Validation")
    print("Phase B requires ≥5 validated transforms (≥1 per category).")
    response = input("Start interactive validation? (y/n): ").strip().lower()

    if response == 'y':
        all_transforms = interactive_validation(all_transforms)
    else:
        print("Skipping validation. You can validate later by editing the JSON file.")

    # Create transform suite
    suite = TransformSuite(
        transforms=all_transforms,
        suite_id="tier2_transforms_v1",
        llm_model=args.llm_model,
        generation_prompt="See PRESERVING_PROMPT, CHANGING_PROMPT, GOTCHA_PROMPT in tier2_transform_generate.py",
        validation_count=sum(1 for t in all_transforms if t.validated)
    )

    # Compute hash
    suite.compute_hash()

    # Validate suite requirements
    validation_report = suite.validate_suite_requirements()

    print()
    print("=" * 60)
    print("Transform Suite Summary")
    print("=" * 60)
    print(f"Suite ID: {suite.suite_id}")
    print(f"Total transforms: {len(suite.transforms)}")
    print(f"Validated: {suite.validation_count}")
    print(f"LLM model: {suite.llm_model}")
    print(f"Suite hash: {suite.suite_hash}")
    print()

    print("Validation Requirements:")
    print(f"  Total validated: {validation_report['total_validated']} (required: ≥5)")
    print(f"  Meets requirement: {'PASS' if validation_report['meets_requirement'] else 'FAIL'}")

    if not validation_report['meets_requirement']:
        print()
        print("WARNING: Suite does not meet Phase B validation requirements!")
        print("Please validate at least 5 transforms before using for evaluation.")
        print("You can validate later by editing the JSON file or re-running this script.")

    print()

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suite_dict = suite.to_dict()

    with output_path.open("w") as f:
        json.dump(suite_dict, f, indent=2)

    print(f"Transform suite saved: {output_path}")
    print()
    print("=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review the transform suite JSON file")
    print("2. If validation count <5, add more validations by editing JSON or re-running")
    print("3. Run semantic evaluation:")
    print("   python experiments\\tier2_semantic_eval.py \\")
    print("       --model-checkpoint <path_to_checkpoint> \\")
    print("       --reference-stats <path_to_reference_summary> \\")
    print(f"       --transform-suite {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
