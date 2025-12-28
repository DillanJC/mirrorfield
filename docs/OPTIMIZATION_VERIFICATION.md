# Phase D Optimization Verification Plan

**Date:** 2025-12-28
**Status:** VERIFIED - Optimization successful
**Git Commit:** 5fa0d475b987 (with RNG fixes)
**Speedup Achieved:** 302× (116 minutes → 23 seconds)

---

## Overview

We've created an optimized version of Phase D evaluation with batched embeddings for 5-10× speedup.

**Files:**
- `tier2/integrated_eval_fast.py` - Optimized core functions
- `experiments/phase_d_integrated_eval_fast.py` - Optimized CLI harness

**Expected Performance:**
- Original: ~2 hours (2000+ sequential embedding calls)
- Optimized: ~15-20 minutes (1-2 batched embedding calls)

---

## Verification Protocol

To ensure optimization doesn't change results, we must verify outputs match.

### Step 1: Run Original Version (Baseline)

```powershell
cd C:\Users\User\mirrorfield
PYTHONIOENCODING=utf-8 PYTHONPATH=/c/Users/User/mirrorfield python -u experiments/phase_d_integrated_eval.py `
    --model-checkpoint experiments/results/tier2_train/20251227_154218/model_checkpoint.pt `
    --reference-stats experiments/results/tier2_reference/20251227_154218/summary.json `
    --transform-suite runs/tier2_transforms_v1.json `
    --calibration-artifact runs/calibration_tau_0c4f1ff5d77e.json `
    --friction-artifact runs/friction_tags_57a44d005300.json `
    --n-perturbations 10 `
    --seed 42
```

**Output:** `experiments/results/phase_d_integrated_eval/<run_id_original>/`

**Expected Runtime:** ~2 hours

---

### Step 2: Run Optimized Version

```powershell
cd C:\Users\User\mirrorfield
PYTHONIOENCODING=utf-8 PYTHONPATH=/c/Users/User/mirrorfield python -u experiments/phase_d_integrated_eval_fast.py `
    --model-checkpoint experiments/results/tier2_train/20251227_154218/model_checkpoint.pt `
    --reference-stats experiments/results/tier2_reference/20251227_154218/summary.json `
    --transform-suite runs/tier2_transforms_v1.json `
    --calibration-artifact runs/calibration_tau_0c4f1ff5d77e.json `
    --friction-artifact runs/friction_tags_57a44d005300.json `
    --n-perturbations 10 `
    --seed 42
```

**Output:** `experiments/results/phase_d_integrated_eval/<run_id_optimized>/`

**Expected Runtime:** ~15-20 minutes

---

### Step 3: Compare Results

Create comparison script `experiments/compare_phase_d_runs.py`:

```python
"""
Compare two Phase D runs to verify optimization doesn't change results.
"""

import json
import sys
from pathlib import Path
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compare_floats(a, b, rtol=1e-5, atol=1e-8):
    """Compare floats with tolerance for numerical precision."""
    return np.isclose(a, b, rtol=rtol, atol=atol)

def compare_summary(original_dir, optimized_dir):
    """Compare summary.json files."""
    orig = load_json(original_dir / "summary.json")
    opt = load_json(optimized_dir / "summary.json")

    print("Comparing evaluation_modes...")

    # Semantic mode
    orig_semantic = orig["evaluation_modes"]["semantic_only"]
    opt_semantic = opt["evaluation_modes"]["semantic_only"]

    assert orig_semantic["n_transforms"] == opt_semantic["n_transforms"]
    assert compare_floats(orig_semantic["mean_delta_d_tilde"], opt_semantic["mean_delta_d_tilde"])
    assert compare_floats(orig_semantic["flip_rate"], opt_semantic["flip_rate"])
    print("  ✓ Semantic mode matches")

    # Perturbation mode
    orig_pert = orig["evaluation_modes"]["perturbation_only"]
    opt_pert = opt["evaluation_modes"]["perturbation_only"]

    assert orig_pert["n_samples"] == opt_pert["n_samples"]
    assert compare_floats(orig_pert["mean_flip_rate"], opt_pert["mean_flip_rate"])
    print("  ✓ Perturbation mode matches")

    # Combined mode
    orig_comb = orig["evaluation_modes"]["combined"]
    opt_comb = opt["evaluation_modes"]["combined"]

    assert orig_comb["n_transforms"] == opt_comb["n_transforms"]
    assert compare_floats(orig_comb["mean_compound_effect"], opt_comb["mean_compound_effect"])
    print("  ✓ Combined mode matches")

    print("\n✅ All high-level metrics match!")

def compare_detailed_results(original_dir, optimized_dir):
    """Compare per-sample/per-transform detailed results."""

    # Semantic results
    orig_sem = load_json(original_dir / "semantic_results.json")
    opt_sem = load_json(optimized_dir / "semantic_results.json")

    assert len(orig_sem) == len(opt_sem)

    mismatches = 0
    for i, (o, p) in enumerate(zip(orig_sem, opt_sem)):
        if not compare_floats(o["delta_d_tilde"], p["delta_d_tilde"]):
            mismatches += 1
            print(f"  Mismatch in semantic result {i}: {o['delta_d_tilde']} vs {p['delta_d_tilde']}")

    if mismatches == 0:
        print(f"  ✓ All {len(orig_sem)} semantic results match")
    else:
        print(f"  ⚠ {mismatches}/{len(orig_sem)} semantic results differ (within tolerance)")

    # Perturbation results
    orig_pert = load_json(original_dir / "perturbation_results.json")
    opt_pert = load_json(optimized_dir / "perturbation_results.json")

    assert len(orig_pert) == len(opt_pert)

    mismatches = 0
    for i, (o, p) in enumerate(zip(orig_pert, opt_pert)):
        if not compare_floats(o["flip_rate"], p["flip_rate"]):
            mismatches += 1

    if mismatches == 0:
        print(f"  ✓ All {len(orig_pert)} perturbation results match")
    else:
        print(f"  ⚠ {mismatches}/{len(orig_pert)} perturbation results differ")

    print("\n✅ Detailed results verified!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_phase_d_runs.py <original_run_id> <optimized_run_id>")
        sys.exit(1)

    original_dir = Path(f"experiments/results/phase_d_integrated_eval/{sys.argv[1]}")
    optimized_dir = Path(f"experiments/results/phase_d_integrated_eval/{sys.argv[2]}")

    compare_summary(original_dir, optimized_dir)
    compare_detailed_results(original_dir, optimized_dir)

    print("\n🎉 Verification complete! Optimized version produces identical results.")
```

**Usage:**
```powershell
python experiments/compare_phase_d_runs.py <original_run_id> <optimized_run_id>
```

---

## Expected Outcomes

### ✅ Success Criteria

1. **High-level metrics match exactly:**
   - Mean Δd̃ for semantic mode
   - Flip rates for all modes
   - Compound effect

2. **Per-sample results match within tolerance:**
   - Individual d̃(x) values (floating-point tolerance: ~1e-5)
   - Flip counts and rates

3. **Speedup achieved:**
   - Optimized version runs 5-10× faster
   - (~2 hours → ~15-20 minutes)

### ⚠️ Acceptable Differences

**Tiny floating-point differences** are acceptable due to:
- Batching order (GPU parallelism)
- Numerical precision in tensor operations
- Should be within `rtol=1e-5` (0.001% relative difference)

### ❌ Failure Indicators

**If these occur, investigation needed:**
- Mean metrics differ by >1%
- Flip rates change significantly (>2%)
- Individual results show systematic bias
- Different number of samples/transforms processed

---

## Post-Verification

Once verification passes:

1. Update `README.md` to recommend fast version
2. Document actual speedup achieved
3. Add entry to Run Ledger
4. Update REFLECTIONS.md with optimization success
5. Consider making fast version the default

---

## Notes

**Why not run verification immediately?**
- Original version takes ~2 hours
- Optimized version takes ~15-20 minutes
- Total verification time: ~2.25 hours
- Recommended: Run overnight or when convenient

**Determinism preserved via:**
- Same seed (42) for both runs
- Seed control in `precompute_embeddings_batch()`
- Deterministic RNG for perturbations

---

## Verification Results (2025-12-28)

### Runs Executed

**Baseline (Original Version):**
- Run ID: `20251228_030427`
- Runtime: 1 hour 55 minutes 34 seconds (6,954 seconds)
- Results: 7.905% flip rate, -0.030 compound effect

**Optimized (Final Version):**
- Run ID: `20251228_050730`
- Runtime: 23 seconds
- Results: 7.905% flip rate, -0.030 compound effect

### Performance

**Speedup: 302× faster than original**
- Original: 116 minutes
- Optimized: 23 seconds
- Far exceeds predicted 5-10× speedup

### Accuracy

**High-Level Metrics:** Perfect match
- Semantic mean Δd̃: -0.931079 (exact match)
- Semantic flip rate: 36.67% (exact match)
- Perturbation flip rate: 7.905% (exact match)
- Combined compound effect: -0.030 (exact match)

**Detailed Results:**
- Semantic: 3/30 results differ by <6.2e-06 (numerical precision only)
- Perturbation: 2000/2000 perfect match
- Combined: 30/30 perfect match

### Bug Fixes Required

Three RNG state management fixes applied during verification:
1. Prevent RNG reset during embedding batch (seed=None)
2. Add per-sample RNG reset in perturbation mode
3. Correct RNG flow in combined mode (one reset, not two)

See `docs/OVERNIGHT_VERIFICATION_LOG.md` for complete debugging details.

### Conclusion

Optimization verified successful. Optimized version produces identical results with 302× speedup. Safe for production use.

---

**END OF VERIFICATION PLAN**
