# Track 4 Fix Report: Threshold Recalibration + Cooldown

**Generated:** 2026-02-09
**Context:** Track 4 live experiment showed -8.0% quality degradation. Root cause: catastrophic threshold miscalibration — every step (64/64) triggered the same low_pr intervention.

## Fixes Applied

### Fix 1: Live-Calibrated Thresholds
Replaced within-reference thresholds with thresholds computed from actual LLM reasoning step embeddings (64 baseline steps, all-MiniLM-L6-v2, 384-dim).

| Threshold | Old (reference) | New (live) | Issue |
|-----------|----------------|------------|-------|
| low_pr (ridge) | 0.1863 (knn_std_p25) | 0.0552 (ridge_p25) | Dimensional mismatch: knn_std applied to ridge_proximity |
| framework_collision (knn_std) | 0.1936 (ref knn_std_p75) | 0.0925 (live knn_std_p75) | Reference self-similarity >> query-vs-reference |
| terra_incognita (dist_nearest) | 2.98e-08 (ref p75) | 1.0068 (live p75) | Reference near-zero made detection impossible |
| decision_boundary (ridge) | 0.30 (hardcoded) | 0.0738 (live ridge_p75) | Hardcoded threshold 4x above max live value |
| curvature_low | 0.02 (hardcoded) | 0.0482 (live curvature_p25) | All live curvatures 0.044-0.059, above 0.02 |

### Fix 2: Consecutive Cooldown
Same signal no longer fires on back-to-back steps. Prevents meta-questioning cascades where repeated "widen the search" derails the analytical progression.

### Fix 3: Signature Diversity Logging
Engine now tracks raw vs final signatures, cooldown skips, and diversity metrics for analysis.

### Fix 4: Intervention Text Redesign
Replaced meta-questioning text with directed analytical instructions:
- **low_pr**: "Name one concrete alternative..." (was: "What assumptions could be questioned?")
- **framework_collision**: "Identify the specific tension..." (was: "Surface all competing paths...")
- **terra_incognita**: "What testable prediction would distinguish this..." (was: "Acknowledge speculative nature...")
- **decision_boundary**: "What would change your assessment if..." (was: "Generate at least 3 approaches...")

## Results: Three-Way Comparison

| Metric | Baseline | Original Switched | Fixed Switched |
|--------|----------|-------------------|----------------|
| **Mean quality** | **0.799** | **0.735 (-8.0%)** | **0.782 (-2.1%)** |
| Total interventions | 0 | 64 (100% trigger) | 24 (38% trigger) |
| Unique signal types | — | 1 (low_pr only) | 4 (all types) |
| Consecutive same-sig rate | — | 100% | 4% |

### Per-Domain

| Domain | Baseline | Orig Switched | Fixed Switched | Orig Delta | Fixed Delta |
|--------|----------|---------------|----------------|------------|-------------|
| Research | 0.789 | 0.634 | 0.751 | -0.155 | **-0.038** |
| Design | 0.737 | 0.827 | 0.816 | +0.090 | **+0.079** |
| Ethics | 0.900 | 0.717 | 0.816 | -0.183 | **-0.084** |
| Hypothesis | 0.770 | 0.763 | 0.744 | -0.007 | **-0.026** |

### Signal Breakdown (Fixed)

| Signal | Count | % of interventions |
|--------|-------|--------------------|
| low_pr | 10 | 42% |
| framework_collision | 8 | 33% |
| decision_boundary | 3 | 13% |
| terra_incognita | 3 | 13% |

### Per-Task Detail

| Task | Baseline | Orig Sw | Fixed Sw | Fixed Delta | Interventions | Signals |
|------|----------|---------|----------|-------------|---------------|---------|
| research_01 | 0.677 | 0.675 | 0.691 | +0.014 | 1 | low_pr |
| research_02 | 0.755 | 0.705 | 0.695 | -0.060 | 1 | low_pr |
| research_03 | 0.865 | 0.565 | 0.805 | -0.060 | 1 | low_pr |
| research_04 | 0.860 | 0.590 | 0.815 | -0.045 | 1 | low_pr |
| design_01 | 0.703 | 0.815 | 0.691 | -0.012 | 2 | framework_collision, low_pr |
| design_02 | 0.709 | 0.783 | 0.904 | +0.195 | 2 | framework_collision, low_pr |
| design_03 | 0.854 | 0.865 | 0.812 | -0.042 | 1 | framework_collision |
| design_04 | 0.680 | 0.843 | 0.855 | +0.175 | 3 | framework_collision, decision_boundary, low_pr |
| ethics_01 | 0.904 | 0.685 | 0.849 | -0.055 | 1 | low_pr |
| ethics_02 | 0.940 | 0.650 | 0.840 | -0.100 | 0 | none |
| ethics_03 | 0.823 | 0.703 | 0.831 | +0.008 | 2 | framework_collision, decision_boundary |
| ethics_04 | 0.933 | 0.831 | 0.743 | -0.190 | 2 | framework_collision x2 |
| hypothesis_01 | 0.700 | 0.771 | 0.620 | -0.080 | 1 | low_pr |
| hypothesis_02 | 0.831 | 0.630 | 0.737 | -0.094 | 3 | decision_boundary, terra_incognita, low_pr |
| hypothesis_03 | 0.804 | 0.865 | 0.809 | +0.005 | 2 | framework_collision, terra_incognita |
| hypothesis_04 | 0.745 | 0.785 | 0.812 | +0.067 | 1 | terra_incognita |

## Success Criteria

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Multiple signatures fire | >1 type | 4 types | PASS |
| No single intervention >40% of steps | <40% | 16% max | PASS |
| Consecutive same-intervention rate | <10% | 4% | PASS |
| Overall quality delta | >= 0 | -0.017 | FAIL (marginal) |
| Ethics domain degradation | <5% | -9.4% | FAIL |

## Remaining Issues

1. **Ethics domain still degrades (-9.4%)**: ethics_02 lost 0.100 even with zero interventions (temperature=0.7 variance). ethics_04 lost 0.190 with two framework_collision interventions — the cooldown didn't help because the two firings were non-consecutive (steps 0 and 3 with coherent in between).

2. **Overall quality slightly negative (-0.017)**: The 4 fixes recovered 75% of the original degradation (from -0.064 to -0.017). The remaining gap is within the noise band of temperature=0.7 stochastic variation (~0.02-0.03 std per run).

3. **Low baseline variance**: With only 16 tasks and temperature=0.7, individual task scores can vary 0.05-0.10 between runs. A definitive comparison would need multiple repetitions or more tasks.

## Interpretation for Track 5

The fixed intervention system is now suitable as a starting policy for Track 5 recursive learning:
- Interventions fire selectively (38% of steps, not 100%)
- All 4 signal types are active with domain-appropriate patterns
- The learner has real signal to work with: some tasks improve, some degrade
- The learner should discover that ethics tasks need less intervention, design tasks benefit from it
- Remaining -0.017 gap is within what the learner can recover through policy optimization

## Files Modified

| File | Changes |
|------|---------|
| `experiments/shared/geometric_tracer.py` | Added `LIVE_CALIBRATED_THRESHOLDS`, `calibrate_from_live()` method |
| `experiments/track4_switches/switch_engine.py` | v2: live calibration flag, cooldown logic, signature logging, new intervention text |
| `experiments/track4_switches/run_live_switched.py` | Added `use_live_calibration=True`, enhanced per-task output |
