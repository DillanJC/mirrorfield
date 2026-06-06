# Phase E: Kosmos Takeaways and Embedder Diagnostics Gate Plan

**Date**: 2025-12-30
**Status**: Planning document (not yet implemented)
**Purpose**: Convert external review feedback into actionable embedder health checks

---

## 1. What Kosmos is Claiming

**Three root causes of geometry failure across embedders** (CLAIM - Kosmos):

1. **Distance concentration**: Pairwise distances become similar in high-D → local geometry becomes noise
2. **Manifold collapse**: Embeddings effectively low-rank / top PCs dominate variance
3. **Batch/artifact confounding**: Geometry correlates with batch effects rather than signal

**Additional failure modes** (CLAIM - Kosmos):

- **Hubness**: Few points dominate kNN lists (some points appear in many neighborhoods)
- **Isotropy**: No directional structure (embeddings are spherically symmetric)
- **Scale dependence**: Features unstable across different k values
- **Redundancy**: Geometry is monotonic transform of baseline (doesn't add information)
- **Linear boundary tasks**: No manifold structure needed (geometry is overkill)

**Note**: These are common pitfalls in representation learning, not confirmed root causes in our system yet.

---

## 2. How This Maps Onto Our Phase E Outcome

**FACT** (our observed result):
- Test 2 showed **embedder-dependence**: Same friction structure, different embedders → opposite verdicts
  - Friction clusters: ΔR² = 0.16 (REAL_SIGNAL)
  - PCA-reduced: ΔR² = 0.001 (COSMETIC)

**INFERENCE** (connecting to Kosmos):
- "Embedder-dependent" is exactly the symptom these diagnostics are meant to predict
- Our current diagnostics (distance concentration CV=0.022 for PCA) caught **distance concentration** failure
- We haven't systematically tested for: manifold collapse, batch confounding, isotropy, scale dependence

**UNKNOWN** (needs targeted tests):
- Which of Kosmos's root causes applies to our PCA case? (Distance concentration? Manifold collapse? Both?)
- Are our friction-cluster embeddings vulnerable to other failure modes?
- Do our geometry features exhibit scale dependence (unstable across k)?

**Action**: Implement a more comprehensive gate that checks all major failure modes before trusting REAL_SIGNAL verdicts.

---

## 3. Embedder Diagnostics Gate: Minimum Viable Version (MVP)

Propose **8 concrete metrics** computed from embeddings + kNN distances:

### Metric 1: Distance Spread (Concentration Check)
- **What it measures**: Variation in pairwise distances
- **Formula**: CV = std(distances) / mean(distances), or (p95 - p5) / p50
- **Why it matters**: Low spread → distances collapsed → geometry is noise
- **Threshold** (TUNABLE initial default):
  - FAIL: CV < 0.03
  - WARN: 0.03 ≤ CV < 0.08
  - PASS: CV ≥ 0.08

### Metric 2: Effective Rank (Manifold Collapse Check)
- **What it measures**: Participation ratio of PCA spectrum
- **Formula**: exp(entropy of normalized eigenvalues), or ratio of (sum of eigenvalues)² / (sum of eigenvalues²)
- **Why it matters**: Low rank → embeddings lie in low-D subspace → local geometry is artifact
- **Threshold** (TUNABLE):
  - FAIL: eff_rank < 10 (for D=768)
  - WARN: 10 ≤ eff_rank < 50
  - PASS: eff_rank ≥ 50

### Metric 3: Intrinsic Dimensionality (Local Rank Check)
- **What it measures**: Local effective dimensionality from kNN distances
- **Formula**: MLE estimator (current implementation) or distance ratio proxy
- **Why it matters**: High ID → no low-D manifold structure → geometry is unstable
- **Threshold** (TUNABLE):
  - FAIL: ID > 200 (for D=768)
  - WARN: 150 < ID ≤ 200
  - PASS: ID ≤ 150

### Metric 4: Hubness Score
- **What it measures**: Skewness of k-occurrence distribution (how many times each point appears in others' kNN)
- **Formula**: skew(in_degree) where in_degree[i] = number of times point i appears in kNN lists
- **Why it matters**: High hubness → kNN structure is degenerate → geometry features unstable
- **Threshold** (TUNABLE):
  - FAIL: |skewness| > 5.0
  - WARN: 2.0 < |skewness| ≤ 5.0
  - PASS: |skewness| ≤ 2.0

### Metric 5: Scale Stability (k-Dependence Check)
- **What it measures**: Consistency of geometry feature rankings across different k values
- **Formula**: Spearman rank correlation of curvature ranks: corr(ranks_at_k16, ranks_at_k32)
- **Why it matters**: Low correlation → features arbitrary / noise-dependent
- **Threshold** (TUNABLE):
  - FAIL: rank_corr < 0.5
  - WARN: 0.5 ≤ rank_corr < 0.7
  - PASS: rank_corr ≥ 0.7

### Metric 6: Redundancy to Baseline (Information Check)
- **What it measures**: Monotonicity between geometry_score and boundary_distance
- **Formula**: |Spearman_corr(geometry_score, boundary_distance)|
- **Why it matters**: High correlation → geometry is just re-encoding baseline → no added value
- **Threshold** (TUNABLE):
  - FAIL: |corr| > 0.90
  - WARN: 0.75 < |corr| ≤ 0.90
  - PASS: |corr| ≤ 0.75

### Metric 7: Anisotropy Index (Optional - Local Structure)
- **What it measures**: Ratio of top eigenvalue to mean eigenvalue in local PCA
- **Formula**: λ₁ / mean(λ) averaged over neighborhoods
- **Why it matters**: Low anisotropy → no directional structure → geometry features are isotropic noise
- **Threshold** (TUNABLE):
  - FAIL: anisotropy < 2.0
  - WARN: 2.0 ≤ anisotropy < 5.0
  - PASS: anisotropy ≥ 5.0
- **Note**: Compute only if cheap (use SVD of k×D neighborhoods we already have)

### Metric 8: Batch Confounding (Optional - If batch_id Available)
- **What it measures**: Mutual information or correlation between geometry features and batch_id
- **Formula**: MI(geometry_score, batch_id) or mean(|corr(geometry_score, batch_one_hot)|)
- **Why it matters**: High MI → geometry captures batch effects, not signal
- **Threshold** (TUNABLE):
  - FAIL: MI > 0.3 or corr > 0.5
  - WARN: MI > 0.1 or corr > 0.3
  - PASS: MI ≤ 0.1 and corr ≤ 0.3
- **Note**: Only run if batch metadata exists

---

## 4. Decision Rule

**Gate logic** (applied before Phase E falsifier runs):

```
# Determine primary failure mode (most severe or first triggered)
primary_failure_mode = None
if distance_spread is FAIL:
    primary_failure_mode = "distance_concentration"
elif effective_rank is FAIL:
    primary_failure_mode = "rank_collapse"
elif hubness_score is FAIL:
    primary_failure_mode = "hubness"
elif redundancy_to_baseline is FAIL:
    primary_failure_mode = "redundancy"
elif scale_stability is FAIL:
    primary_failure_mode = "scale_instability"
elif anisotropy_index is FAIL:
    primary_failure_mode = "isotropy"
elif batch_confounding is FAIL:
    primary_failure_mode = "batch_confound"
else:
    # Check for warnings (pick strongest signal)
    if distance_spread is WARN:
        primary_failure_mode = "distance_concentration"
    elif effective_rank is WARN:
        primary_failure_mode = "rank_collapse"
    # ... (check other warnings)

# Set recommendation based on primary failure mode
recommendation = get_recommendation(primary_failure_mode)

# Gate decision
if (distance_spread is FAIL) OR (effective_rank is FAIL):
    gate_decision = FAIL
    verdict_override = COSMETIC-risk
    message = "Severe representation failure. Geometry features unreliable."

elif (any metric is FAIL) OR (3+ metrics are WARN):
    gate_decision = WARN
    verdict_override = None  # proceed but tag results
    message = "Representation warnings detected. REAL_SIGNAL verdicts may be embedder-specific artifacts."

else:
    gate_decision = PASS
    verdict_override = None
    primary_failure_mode = None  # no failure
    recommendation = "Representation healthy. Proceed with geometry features."
    message = "Representation health checks passed. Proceed with Phase E falsifier."
```

**Failure Mode → Recommendation Mapping**:

| primary_failure_mode | recommendation |
|---------------------|----------------|
| `distance_concentration` | "Avoid PCA/whitened representations; try raw or lightly-normalized embeddings" |
| `rank_collapse` | "Embeddings too low-rank; increase model capacity or use higher-D representations" |
| `hubness` | "Use hubness-corrected kNN (e.g., mutual proximity, local scaling) or increase k" |
| `redundancy` | "Geometry redundant with baseline; skip geometry features, use boundary_distance only" |
| `scale_instability` | "Features unstable across k; use ensemble over multiple k values or different metric" |
| `isotropy` | "No directional structure; geometry features won't help (linear baseline may suffice)" |
| `batch_confound` | "Geometry captures batch effects; apply batch correction or use batch-aware models" |
| `null` (PASS) | "Representation healthy. Proceed with geometry features." |

**Operational meaning**:

- **FAIL**: Default geometry verdict to COSMETIC-risk. Require a targeted construction test (like Test C) before trusting any REAL_SIGNAL verdict.
- **WARN**: Proceed with falsifier runs, but tag all results with `representation_warning=True`. Document that gains may not generalize.
- **PASS**: Proceed normally. No representation warnings needed.

**Rationale**:
- Distance spread and effective rank are **fatal** (Kosmos root causes #1 and #2)
- Other metrics are **warnings** (common failure modes, not guaranteed failures)
- Multiple warnings → cumulative risk → escalate to WARN decision

**Why Include `primary_failure_mode_guess` and `recommendation`?**

These fields make the gate **actionable**, not just a sterile pass/fail checklist:

- **Diagnosis**: Users know *which* failure mode is most likely (not just "FAIL")
- **Action**: Users get concrete next steps (e.g., "avoid PCA" vs "increase k")
- **Debugging**: If REAL_SIGNAL verdict contradicts gate, users can check if the specific failure mode applies
- **Communication**: Makes gate results interpretable for non-experts (e.g., "hubness problem → try mutual proximity kNN")

**Note**: `primary_failure_mode_guess` is a **guess** based on the strongest metric signal, not a definitive root cause diagnosis. In ambiguous cases (multiple FAIL metrics), prioritize distance_concentration > rank_collapse > others.

---

## 5. Implementation Plan

### File Locations

**Core metrics module** (extend existing):
- `geometry/embedder_diagnostics.py`
  - Add new metrics: effective_rank, scale_stability, redundancy_to_baseline, anisotropy_index, batch_confounding
  - Update `DiagnosticReport` dataclass to include all 8 metrics
  - Update `.run_all()` to compute gate decision
  - Add `.apply_gate_decision()` method

**CLI runner** (extend existing):
- `experiments/run_embedder_diagnostics.py`
  - Add `--baseline-distances` argument (for redundancy check)
  - Add `--batch-ids` argument (for batch confounding check, optional)
  - Add `--k-values` argument (for scale stability, default: [16, 32])
  - Output extended summary.json with gate_decision field

**Tests**:
- `tests/test_embedder_diagnostics_gate.py` (new)
  - Test that gate flags PCA case as FAIL or WARN
  - Test that gate does not flag friction-cluster case as FAIL when it shows REAL_SIGNAL
  - Test stability across seed/batch order

### JSON Schema for Diagnostics Output

```json
{
  "embedder_id": "string (user-provided label)",
  "timestamp": "ISO-8601",
  "n_samples": "int",
  "n_dims": "int",
  "metrics": {
    "distance_spread": {"value": "float", "threshold": "FAIL|WARN|PASS", "cv": "float"},
    "effective_rank": {"value": "float", "threshold": "FAIL|WARN|PASS"},
    "intrinsic_dimensionality": {"value": "float", "threshold": "FAIL|WARN|PASS"},
    "hubness_score": {"value": "float", "threshold": "FAIL|WARN|PASS"},
    "scale_stability": {"value": "float", "threshold": "FAIL|WARN|PASS", "k_values": "[int, int]"},
    "redundancy_to_baseline": {"value": "float", "threshold": "FAIL|WARN|PASS"},
    "anisotropy_index": {"value": "float", "threshold": "FAIL|WARN|PASS", "optional": true},
    "batch_confounding": {"value": "float", "threshold": "FAIL|WARN|PASS", "optional": true}
  },
  "gate_decision": "PASS|WARN|FAIL",
  "primary_failure_mode_guess": "distance_concentration | rank_collapse | hubness | redundancy | scale_instability | isotropy | batch_confound | null",
  "recommendation": "string (one-line actionable advice)",
  "warnings": ["list of warning messages"],
  "verdict_override": "COSMETIC-risk | null",
  "message": "human-readable gate decision explanation",
  "thresholds_used": {
    "distance_spread": {"fail": 0.03, "warn": 0.08},
    "effective_rank": {"fail": 10, "warn": 50},
    "...": "..."
  }
}
```

**Example output** (PCA case - FAIL):
```json
{
  "embedder_id": "test_b_pca_reduced",
  "gate_decision": "FAIL",
  "primary_failure_mode_guess": "distance_concentration",
  "recommendation": "Avoid PCA/whitened representations; try raw or lightly-normalized embeddings",
  "verdict_override": "COSMETIC-risk",
  "message": "Severe representation failure. Geometry features unreliable.",
  "metrics": {
    "distance_spread": {"value": 0.022, "threshold": "FAIL"},
    "...": "..."
  }
}
```

**Example output** (friction clusters - WARN):
```json
{
  "embedder_id": "test_a_friction_clusters",
  "gate_decision": "WARN",
  "primary_failure_mode_guess": "hubness",
  "recommendation": "Use hubness-corrected kNN (e.g., mutual proximity, local scaling) or increase k",
  "verdict_override": null,
  "message": "Representation warnings detected. REAL_SIGNAL verdicts may be embedder-specific artifacts.",
  "metrics": {
    "hubness_score": {"value": 4.97, "threshold": "WARN"},
    "...": "..."
  }
}
```

**Example output** (healthy - PASS):
```json
{
  "embedder_id": "production_embedder_v3",
  "gate_decision": "PASS",
  "primary_failure_mode_guess": null,
  "recommendation": "Representation healthy. Proceed with geometry features.",
  "verdict_override": null,
  "message": "Representation health checks passed. Proceed with Phase E falsifier.",
  "metrics": {
    "distance_spread": {"value": 0.312, "threshold": "PASS"},
    "effective_rank": {"value": 87.3, "threshold": "PASS"},
    "...": "..."
  }
}
```

---

## 6. Verification Plan for the Gate

**Three falsifiers to validate the gate itself**:

### Test 1: Negative Control (Should Flag PCA Case)
- **Dataset**: Test B Strategy B (PCA-reduced embeddings)
- **Expected**: gate_decision = FAIL or WARN
- **Rationale**: We KNOW this embedder produces COSMETIC verdicts (ΔR²=0.001)
- **Pass criteria**: Gate must flag at least distance_spread as FAIL (CV=0.022 < 0.03)
- **Status**: UNKNOWN (need to run gate on Test B-B)

### Test 2: Positive Control (Should NOT Flag Friction Clusters as FAIL)
- **Dataset**: Test A (Friction clusters with REAL_SIGNAL verdict)
- **Expected**: gate_decision = PASS or WARN (not FAIL)
- **Rationale**: We KNOW this embedder produces REAL_SIGNAL (ΔR²=0.14)
- **Pass criteria**: Gate must NOT set verdict_override=COSMETIC-risk
- **Acceptable**: WARN is okay (friction clusters may have hubness), but not FAIL
- **Status**: UNKNOWN (need to run gate on Test A)

### Test 3: Stability Check (Seed/Batch Order Invariance)
- **Dataset**: Test C (Targeted construction)
- **Procedure**:
  1. Run gate with seed=42, batch_size=256
  2. Run gate with seed=100, batch_size=256
  3. Run gate with seed=42, batch_size=512 (different batching)
- **Expected**: gate_decision identical across runs
- **Pass criteria**: All 8 metric values change by <1%, gate_decision unchanged
- **Status**: UNKNOWN (need to implement and test)

**FACT check**: If the gate fails any of these falsifiers, we need to tune thresholds or fix bugs before deployment.

---

## Implementation Checklist

**Phase 1: Core Metrics** (extend existing module)
- [ ] Add `effective_rank()` function to `geometry/embedder_diagnostics.py`
- [ ] Add `scale_stability()` function (requires computing geometry at k=16 and k=32)
- [ ] Add `redundancy_to_baseline()` function (requires boundary_distance input)
- [ ] Add `anisotropy_index()` function (optional, use existing SVD neighborhoods)
- [ ] Add `batch_confounding()` function (optional, requires batch_id input)
- [ ] Update `DiagnosticReport` dataclass with new metrics
- [ ] Implement `apply_gate_decision()` method with FAIL/WARN/PASS logic
- [ ] Implement `determine_primary_failure_mode()` method (returns failure mode enum)
- [ ] Implement `get_recommendation()` method (maps failure mode → recommendation string)

**Phase 2: CLI Runner** (extend existing script)
- [ ] Update `run_embedder_diagnostics.py` to accept `--baseline-distances` argument
- [ ] Update to accept `--batch-ids` argument (optional)
- [ ] Update to accept `--k-values` argument (default: 16,32)
- [ ] Add gate_decision to output JSON
- [ ] Add thresholds_used to output JSON

**Phase 3: Validation**
- [ ] Create `tests/test_embedder_diagnostics_gate.py`
- [ ] Test 1: Run gate on Test B-B (PCA), verify FAIL or WARN
- [ ] Test 2: Run gate on Test A (friction), verify NOT FAIL
- [ ] Test 3: Run gate with seed/batch variations, verify stability
- [ ] Document gate validation results in analysis doc

**Phase 4: Integration**
- [ ] Update Phase E falsifier scripts to call gate before running experiments
- [ ] Add gate_decision to Phase E test summary.json files
- [ ] Update `PHASE_E_FALSIFIER_ANALYSIS.md` Section 7 with gate details
- [ ] Update deployment rule: check gate_decision before trusting REAL_SIGNAL

---

## Example Commands (Placeholders)

**Run gate on Test B-B (PCA case)**:
```bash
python experiments/run_embedder_diagnostics.py \
    --embeddings runs/test_b_strategy_b/embeddings.npy \
    --baseline-distances runs/test_b_strategy_b/boundary_distances.npy \
    --embedder-id "test_b_pca_reduced" \
    --k-values 16,32 \
    --output runs/gate_validation_test_b_pca/
```

**Run gate on Test A (friction clusters)**:
```bash
python experiments/run_embedder_diagnostics.py \
    --embeddings runs/test_a/embeddings.npy \
    --baseline-distances runs/test_a/boundary_distances.npy \
    --embedder-id "test_a_friction_clusters" \
    --k-values 16,32 \
    --output runs/gate_validation_test_a/
```

**Run gate with batch IDs (if available)**:
```bash
python experiments/run_embedder_diagnostics.py \
    --embeddings path/to/embeddings.npy \
    --baseline-distances path/to/boundary_distances.npy \
    --batch-ids path/to/batch_ids.npy \
    --embedder-id "production_embedder_v2" \
    --k-values 16,32 \
    --output runs/gate_production_v2/
```

---

## Next Steps

1. **Implement Phase 1** (core metrics) in `geometry/embedder_diagnostics.py`
2. **Run gate validation** (Tests 1-3) to tune thresholds
3. **Update thresholds** if needed (mark as "empirically tuned from Test A/B/C")
4. **Integrate gate** into Phase E experimental workflow
5. **Document** gate results in Section 7 of analysis doc

**Expected outcome**: A working gate that predicts COSMETIC risk with high accuracy, reducing false confidence in embedder-specific artifacts.

---

**End of Plan**
