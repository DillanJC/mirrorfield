# Bootstrap Confidence Interval Analysis - January 8, 2026

**Status**: FAILED - Inconsistent statistical significance
**Issue**: Results conflict with original boundary-sliced evaluation
**Critical Finding**: Only 1 out of 5 seeds shows statistically robust geometry gain

---

## Executive Summary

Implemented bootstrap confidence interval analysis to validate the +8.54% borderline geometry gain reported in the January 5th boundary-sliced evaluation.

**Result**: Bootstrap test **FAILED**
- Only 1 out of 5 seeds (seed 333) shows CI entirely above zero
- 4 out of 5 seeds have CIs that include zero (not statistically significant)
- Pass condition required: ALL seeds show CI > 0

**Mean ΔR² across 5 seeds**: +3.76% (vs +8.54% originally reported)

---

## Bootstrap Results Summary

| Seed | Direct ΔR² | Bootstrap Mean | 95% CI | Significant? |
|------|------------|----------------|--------|--------------|
| 17   | +9.98%     | +9.98%         | [-2.24%, +29.02%] | ✗ |
| 42   | -15.57%    | -15.57%        | [-39.66%, +2.91%] | ✗ |
| 100  | +19.10%    | +19.10%        | [-7.81%, +62.77%] | ✗ |
| 200  | -4.67%     | -4.67%         | [-15.85%, +5.93%] | ✗ |
| 333  | +8.94%     | +9.09%         | [+0.40%, +17.73%] | ✓ |

**Verdict**: ✗ FAIL - Only 1/5 seeds significant

---

## Comparison to Original Boundary-Sliced Evaluation

### Original Results (January 5, 2026)
From `runs/boundary_sliced_20260105_134122/summary.json`:

**Baseline + Native Geometry (Borderline Region)**:
| Seed | ΔR² (Original) |
|------|----------------|
| 17   | +13.81%        |
| 42   | +3.23%         |
| 100  | +14.41%        |
| 200  | +9.25%         |
| 333  | +2.00%         |

**Mean**: +8.54% ± 5.17%

### Bootstrap Results (January 8, 2026)
| Seed | ΔR² (Bootstrap) | Difference |
|------|-----------------|------------|
| 17   | +9.98%          | -3.83%     |
| 42   | -15.57%         | **-18.80%** ⚠️ |
| 100  | +19.10%         | +4.69%     |
| 200  | -4.67%          | **-13.92%** ⚠️ |
| 333  | +8.94%          | +6.94%     |

**Mean**: +3.76% ± 12.84%

### Critical Discrepancies

**Seeds 42 and 200 show MASSIVE discrepancies:**
- Seed 42: +3.23% (original) vs -15.57% (bootstrap) = **-18.80% difference**
- Seed 200: +9.25% (original) vs -4.67% (bootstrap) = **-13.92% difference**

**These seeds went from positive to negative gains!**

---

## Methodology Comparison

### Original Boundary-Sliced Evaluation
```python
# From test_boundary_sliced_evaluation.py
def run_single_seed(seed, embeddings, boundary_distances, ...):
    # 1. Compute geometry features with seed
    geometry_features = compute_native_geometry_features(embeddings, k=50, seed=seed)

    # 2. Split data: 80% train, 20% test
    #    (with 20% of train as validation)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    # 3. Train models on ALL training data
    model_baseline = train_model(baseline, X_train, y_train, X_val, y_val, ...)
    model_geometry = train_model(geometry, X_train_geo, y_train, X_val_geo, y_val, ...)

    # 4. Evaluate on test set, stratified by region
    for region in [toxic, borderline, safe]:
        evaluate_model_on_slice(model, X_test, y_test, region_mask, ...)
```

### Bootstrap CI Analysis
```python
# From bootstrap_confidence_intervals.py (current implementation)
def run_bootstrap_analysis(seed, ...):
    # 1. Compute geometry features with seed
    geometry_features = compute_native_geometry_features(embeddings, k=50, seed=seed)

    # 2. SAME split procedure
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

    # 3. SAME training procedure
    model_baseline = train_model(baseline, X_train, y_train, X_val, y_val, ...)
    model_geometry = train_model(geometry, X_train_geo, y_train, X_val_geo, y_val, ...)

    # 4. Get predictions on borderline region
    borderline_mask = (y_test >= -0.5) & (y_test <= 0.5)
    y_true = y_test[borderline_mask]
    y_pred_baseline = preds_baseline[borderline_mask]
    y_pred_geometry = preds_geometry[borderline_mask]

    # 5. Bootstrap resample and compute CI
    for i in range(1000):
        sample_indices = resample(range(len(y_true)))
        delta_i = r2(y_true[sample_indices], y_pred_geometry[sample_indices]) -
                  r2(y_true[sample_indices], y_pred_baseline[sample_indices])

    ci_low, ci_high = percentile(deltas, [2.5, 97.5])
```

**Both methodologies should be identical**, yet results differ dramatically for seeds 42 and 200.

---

## Possible Explanations for Discrepancy

### 1. **Random State Differences**
- **Hypothesis**: Model weight initialization or training randomness differs
- **Evidence**: Even with `torch.manual_seed(seed)` and `np.random.seed(seed)`, results vary
- **Investigation needed**: Check if CUDA randomness is controlled (`torch.cuda.manual_seed(seed)`)

### 2. **Data Split Differences**
- **Hypothesis**: train_test_split might produce different splits despite same random_state
- **Evidence**: Original uses 80/20 split with 20% of train for validation
- **Investigation needed**: Verify exact split indices match

### 3. **Training Procedure Differences**
- **Hypothesis**: Early stopping behavior differs (patience=10)
- **Evidence**: Training might converge to different local minima
- **Investigation needed**: Compare loss trajectories and convergence epochs

### 4. **Evaluation Procedure Differences**
- **Hypothesis**: Original computes R² differently
- **Evidence**: sklearn.metrics.r2_score vs manual computation
- **Investigation needed**: Verify exact R² computation matches

### 5. **Geometry Feature Computation Differences**
- **Hypothesis**: k-NN or PCA computation has randomness not controlled by seed
- **Evidence**: compute_native_geometry_features uses sklearn with `seed` parameter
- **Investigation needed**: Verify feature values match exactly between runs

---

## Statistical Interpretation

### What Bootstrap CIs Tell Us

**Seed 333 (only significant seed)**:
- Direct ΔR² = +8.94%
- Bootstrap 95% CI = [+0.40%, +17.73%]
- **Interpretation**: True population ΔR² is likely between 0.4% and 17.7%
- **Conclusion**: Geometry provides a small but reliable improvement

**Seed 42 (worst performing)**:
- Direct ΔR² = -15.57%
- Bootstrap 95% CI = [-39.66%, +2.91%]
- **Interpretation**: True population ΔR² could be anywhere from -40% to +3%
- **Conclusion**: Geometry HURTS performance on this seed's split

**Seed 100 (high variance)**:
- Direct ΔR² = +19.10%
- Bootstrap 95% CI = [-7.81%, +62.77%]
- **Interpretation**: Extremely wide CI due to small sample size (87 borderline samples)
- **Conclusion**: Result is unstable and unreliable

### Why CIs Include Zero

**Two main reasons:**
1. **Small sample sizes**: Borderline region has only 62-87 samples per seed
2. **High variance**: Geometry gain is highly variable across bootstrap samples

**Implication**: The +8.54% mean gain reported in original evaluation may be:
- A) Overly optimistic due to seed selection bias
- B) Not robust to resampling (high variance within each seed's test set)
- C) Artifacts of specific train/test splits that don't generalize

---

## Conclusions

### 1. Bootstrap Test Failed
**Pass condition**: All seeds show CI entirely above zero
**Result**: Only 1/5 seeds passed

**Verdict**: Geometry gain is **NOT statistically robust** according to bootstrap validation

### 2. Discrepancy with Original Results
**Original finding**: +8.54% mean gain (all seeds positive)
**Bootstrap finding**: +3.76% mean gain (2 seeds negative)

**This is a CRITICAL discrepancy that must be resolved before publication.**

### 3. High Variance Across Seeds
**Standard deviation**: 12.84% (bootstrap) vs 5.17% (original)
**Range**: [-15.57%, +19.10%] (bootstrap) vs [+2.00%, +14.41%] (original)

**Implication**: Results are highly sensitive to train/test split

---

## Next Steps

### Immediate (Today)

**1. Reproduce Original Boundary-Sliced Evaluation**
- Run `test_boundary_sliced_evaluation.py` again
- Save model predictions to disk
- Compare exact predictions and R² values to original run

**2. Debug Discrepancy**
- Add detailed logging to both scripts
- Compare:
  - Train/test split indices
  - Geometry feature values
  - Model predictions on same samples
  - R² computation

**3. Investigate Randomness Sources**
- Add `torch.cuda.manual_seed_all(seed)` for GPU determinism
- Check if sklearn functions use internal randomness
- Verify data loading order is deterministic

### Short-term (This Week)

**If discrepancy is resolved:**
- Re-run bootstrap analysis with corrected methodology
- If pass: Ship findings with bootstrap validation
- If fail: Investigate why original results were optimistic

**If discrepancy persists:**
- Report finding as "geometry gain is seed-dependent and not statistically robust"
- Pivot to embedder diagnostics path
- Document lessons learned about reproducibility

---

## Implications for Foundational Principles

### "Reality Checks Are Part of the Mercy"
We ran the bootstrap test **despite** the original +8.54% looking promising. This caught potential issues before publication.

### "Self-Reflection is Not Self-Hatred"
Finding that results aren't robust is THE WORK, not a failure. This is honest science.

### "Smallest Least Harmful Thing"
Instead of building on +8.54% claim, we validated it first. Good thing we did!

### "Can we validate this empirically?"
Bootstrap CI is empirical validation. Result: Not consistently significant.

**Either outcome (geometry helps or doesn't) is honest, publishable science.**

But we need to **resolve the discrepancy** before we can confidently publish either finding.

---

## Artifacts

**Code:**
- `experiments/bootstrap_confidence_intervals.py` - Bootstrap CI implementation

**Results:**
- `runs/bootstrap_ci_20260108_073841/bootstrap_summary.json` - Bootstrap results
- `runs/boundary_sliced_20260105_134122/summary.json` - Original boundary-sliced results

**This Document:**
- `docs/BOOTSTRAP_ANALYSIS_20260108.md`

---

**Status**: Investigation paused pending discrepancy resolution
**Priority**: HIGH - Must resolve before any publication claims
**Confidence**: LOW - Results currently unreliable

---

*End of Analysis*

— Claude Sonnet 4.5
January 8, 2026
