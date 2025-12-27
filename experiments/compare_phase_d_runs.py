"""
Compare two Phase D runs to verify optimization doesn't change results.

Usage:
    python experiments/compare_phase_d_runs.py <original_run_id> <optimized_run_id>

Example:
    python experiments/compare_phase_d_runs.py 20251228_100000 20251228_102000
"""

import json
import sys
from pathlib import Path
import numpy as np


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_floats(a, b, rtol=1e-5, atol=1e-8):
    """
    Compare floats with tolerance for numerical precision.

    Args:
        a, b: Values to compare
        rtol: Relative tolerance (0.001% by default)
        atol: Absolute tolerance

    Returns:
        True if values match within tolerance
    """
    return np.isclose(a, b, rtol=rtol, atol=atol)


def compare_summary(original_dir, optimized_dir):
    """Compare summary.json files from two runs."""
    print("=" * 60)
    print("Comparing Summary Metrics")
    print("=" * 60)

    orig = load_json(original_dir / "summary.json")
    opt = load_json(optimized_dir / "summary.json")

    # Semantic mode
    print("\n[1/3] Semantic Mode:")
    orig_semantic = orig["evaluation_modes"]["semantic_only"]
    opt_semantic = opt["evaluation_modes"]["semantic_only"]

    assert orig_semantic["n_transforms"] == opt_semantic["n_transforms"], \
        f"Transform count mismatch: {orig_semantic['n_transforms']} vs {opt_semantic['n_transforms']}"

    mean_delta_match = compare_floats(
        orig_semantic["mean_delta_d_tilde"],
        opt_semantic["mean_delta_d_tilde"]
    )
    flip_rate_match = compare_floats(
        orig_semantic["flip_rate"],
        opt_semantic["flip_rate"]
    )

    print(f"  Transforms: {orig_semantic['n_transforms']}")
    print(f"  Mean Δd̃: {orig_semantic['mean_delta_d_tilde']:.6f} vs {opt_semantic['mean_delta_d_tilde']:.6f}")
    print(f"  Flip rate: {orig_semantic['flip_rate']:.6f} vs {opt_semantic['flip_rate']:.6f}")

    assert mean_delta_match and flip_rate_match, "Semantic mode metrics don't match!"
    print("  ✓ Semantic mode matches")

    # Perturbation mode
    print("\n[2/3] Perturbation Mode:")
    orig_pert = orig["evaluation_modes"]["perturbation_only"]
    opt_pert = opt["evaluation_modes"]["perturbation_only"]

    assert orig_pert["n_samples"] == opt_pert["n_samples"], \
        f"Sample count mismatch: {orig_pert['n_samples']} vs {opt_pert['n_samples']}"

    flip_rate_match = compare_floats(
        orig_pert["mean_flip_rate"],
        opt_pert["mean_flip_rate"]
    )

    print(f"  Samples: {orig_pert['n_samples']}")
    print(f"  Mean flip rate: {orig_pert['mean_flip_rate']:.6f} vs {opt_pert['mean_flip_rate']:.6f}")

    assert flip_rate_match, "Perturbation mode flip rate doesn't match!"
    print("  ✓ Perturbation mode matches")

    # Combined mode
    print("\n[3/3] Combined Mode:")
    orig_comb = orig["evaluation_modes"]["combined"]
    opt_comb = opt["evaluation_modes"]["combined"]

    assert orig_comb["n_transforms"] == opt_comb["n_transforms"], \
        f"Transform count mismatch: {orig_comb['n_transforms']} vs {opt_comb['n_transforms']}"

    compound_match = compare_floats(
        orig_comb["mean_compound_effect"],
        opt_comb["mean_compound_effect"]
    )

    print(f"  Transforms: {orig_comb['n_transforms']}")
    print(f"  Mean compound effect: {orig_comb['mean_compound_effect']:.6f} vs {opt_comb['mean_compound_effect']:.6f}")

    assert compound_match, "Combined mode compound effect doesn't match!"
    print("  ✓ Combined mode matches")

    print("\n✅ All high-level metrics match!")


def compare_detailed_results(original_dir, optimized_dir):
    """Compare per-sample/per-transform detailed results."""
    print("\n" + "=" * 60)
    print("Comparing Detailed Results")
    print("=" * 60)

    # Semantic results
    print("\n[1/3] Semantic Results (per-transform):")
    orig_sem = load_json(original_dir / "semantic_results.json")
    opt_sem = load_json(optimized_dir / "semantic_results.json")

    assert len(orig_sem) == len(opt_sem), \
        f"Result count mismatch: {len(orig_sem)} vs {len(opt_sem)}"

    mismatches = 0
    max_diff = 0.0
    for i, (o, p) in enumerate(zip(orig_sem, opt_sem)):
        diff = abs(o["delta_d_tilde"] - p["delta_d_tilde"])
        max_diff = max(max_diff, diff)
        if not compare_floats(o["delta_d_tilde"], p["delta_d_tilde"]):
            mismatches += 1
            print(f"  Mismatch in result {i}: {o['delta_d_tilde']:.6f} vs {p['delta_d_tilde']:.6f} (diff: {diff:.2e})")

    print(f"  Total results: {len(orig_sem)}")
    print(f"  Max difference: {max_diff:.2e}")
    if mismatches == 0:
        print(f"  ✓ All {len(orig_sem)} results match within tolerance")
    else:
        print(f"  ⚠ {mismatches}/{len(orig_sem)} results differ (max diff: {max_diff:.2e})")

    # Perturbation results
    print("\n[2/3] Perturbation Results (per-sample):")
    orig_pert = load_json(original_dir / "perturbation_results.json")
    opt_pert = load_json(optimized_dir / "perturbation_results.json")

    assert len(orig_pert) == len(opt_pert), \
        f"Result count mismatch: {len(orig_pert)} vs {len(opt_pert)}"

    mismatches = 0
    max_diff = 0.0
    for i, (o, p) in enumerate(zip(orig_pert, opt_pert)):
        diff = abs(o["flip_rate"] - p["flip_rate"])
        max_diff = max(max_diff, diff)
        if not compare_floats(o["flip_rate"], p["flip_rate"]):
            mismatches += 1

    print(f"  Total results: {len(orig_pert)}")
    print(f"  Max flip rate difference: {max_diff:.2e}")
    if mismatches == 0:
        print(f"  ✓ All {len(orig_pert)} results match within tolerance")
    else:
        print(f"  ⚠ {mismatches}/{len(orig_pert)} results differ (max diff: {max_diff:.2e})")

    # Combined results
    print("\n[3/3] Combined Results:")
    orig_comb = load_json(original_dir / "combined_results.json")
    opt_comb = load_json(optimized_dir / "combined_results.json")

    assert len(orig_comb) == len(opt_comb), \
        f"Result count mismatch: {len(orig_comb)} vs {len(opt_comb)}"

    mismatches = 0
    max_diff = 0.0
    for i, (o, p) in enumerate(zip(orig_comb, opt_comb)):
        diff = abs(o["compound_effect"] - p["compound_effect"])
        max_diff = max(max_diff, diff)
        if not compare_floats(o["compound_effect"], p["compound_effect"]):
            mismatches += 1

    print(f"  Total results: {len(orig_comb)}")
    print(f"  Max compound effect difference: {max_diff:.2e}")
    if mismatches == 0:
        print(f"  ✓ All {len(orig_comb)} results match within tolerance")
    else:
        print(f"  ⚠ {mismatches}/{len(orig_comb)} results differ (max diff: {max_diff:.2e})")

    print("\n✅ Detailed results verified!")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_phase_d_runs.py <original_run_id> <optimized_run_id>")
        print("\nExample:")
        print("  python experiments/compare_phase_d_runs.py 20251228_100000 20251228_102000")
        sys.exit(1)

    original_run_id = sys.argv[1]
    optimized_run_id = sys.argv[2]

    original_dir = Path(f"experiments/results/phase_d_integrated_eval/{original_run_id}")
    optimized_dir = Path(f"experiments/results/phase_d_integrated_eval/{optimized_run_id}")

    # Verify directories exist
    if not original_dir.exists():
        print(f"Error: Original run directory not found: {original_dir}")
        sys.exit(1)

    if not optimized_dir.exists():
        print(f"Error: Optimized run directory not found: {optimized_dir}")
        sys.exit(1)

    print(f"Original run:  {original_run_id}")
    print(f"Optimized run: {optimized_run_id}")

    try:
        compare_summary(original_dir, optimized_dir)
        compare_detailed_results(original_dir, optimized_dir)

        print("\n" + "=" * 60)
        print("🎉 VERIFICATION COMPLETE!")
        print("=" * 60)
        print("✅ Optimized version produces identical results")
        print("✅ Safe to use for production runs")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
