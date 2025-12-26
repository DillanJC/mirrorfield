# Mirrorfield AI — Definitions Freeze v0.1

**Date:** 2025-12-27
**Status:** PHASE 0 LOCK — baseline definitions for reproducible evaluation
**Purpose:** Prevent definition drift in Tier-2+ evaluation phases

---

## A) Boundary Distance

### d(x) — Raw Boundary Distance
**Definition:**
The distance from a point `x` in embedding/latent space to the nearest decision boundary or classification threshold.

**Formula:**
`d(x)` is computed as the minimum signed distance to the separator in the model's decision surface. For binary classifiers with logit outputs, this is typically:
```
d(x) = |logit_positive(x) - logit_negative(x)| / 2
```
For multiclass: minimum margin to any decision boundary.

**Sign Convention:**
Positive `d(x)` indicates confident classification; values near zero indicate proximity to decision boundary.

### d̃(x) — Standardized Boundary Distance
**Definition:**
Z-score normalized boundary distance relative to a reference distribution.

**Formula:**
```
d̃(x) = (d(x) - μ_ref) / σ_ref
```
Where:
- `μ_ref` = mean boundary distance over reference set
- `σ_ref` = standard deviation over reference set

**Alternative (robust):**
If distribution is non-Gaussian, use:
```
d̃(x) = (d(x) - median_ref) / MAD_ref
```
Where `MAD_ref` = median absolute deviation.

**Current Implementation:** Z-score (mean/std) — to be validated in Phase A evidence pack.

### Reference Set Specification

**ref_name:** `TBD_baseline_v1`
**ref_build_policy:** TBD — awaiting dataset specification in Phase B
**ref_hash_policy:** `hash(sorted(prompt_ids + prompt_texts))` — ensures reproducibility
**stats:** Mean/std OR median/MAD computed over full reference set
**N_ref:** TBD (target: ≥1000 samples for stable statistics)

**STATUS:** Reference set construction is **PHASE B dependency**. Until then, `d̃(x)` normalization is **NOT OPERATIONAL**.

---

## B) τ Perturbation Policy

### Perturbation Generator
**Symbol:** `τ(x, ε)`
**Role:** Generates perturbed versions of input `x` with controlled noise scale `ε`.

### Policy Type
**Current:** Fixed noise scale
**Calibration:** Not yet implemented

### Fixed Policy Specification (Phase A baseline)
**Noise type:** Gaussian additive noise in feature/embedding space
**Noise scale:** `ε = 0.05` (5% of feature range or embedding norm)
**Sampling:** Independent per feature/dimension
**Seed control:** Fixed seed per run (default: `seed=42`)

### Future Calibration Policy (Phase C)
When implemented, calibration will:
- Use a held-out calibration split (20% of reference set)
- Tune `ε` to achieve target flip rate (e.g., 5-10% boundary crossings)
- Save calibration artifact: `runs/calibration_tau_<hash>.json`
- Lock calibrated `ε` value for all downstream evaluation

**STATUS:** Phase A uses **fixed τ**. Calibrated τ is deferred to Phase C.

---

## C) Borderline Slice Rule

### Definition
The "borderline set" consists of samples closest to decision boundaries, operationalized as:

**Rule:**
```
borderline := {x | |d̃(x)| ≤ θ_borderline}
```

**Default threshold:** `θ_borderline = 0.5` (within 0.5 standard deviations of boundary)

**Alternative percentile rule:**
```
borderline := lowest X% of |d̃(x)| in reference set
```
Where `X = 10%` (bottom decile by absolute standardized distance).

### Tie-breaking
If exactly at threshold: include the sample (inclusive boundary).

### Selection Method
**Current:** Absolute threshold on `|d̃(x)|`
**Phase B refinement:** May switch to percentile-based selection for robustness.

**STATUS:** Locked for Phase A. Threshold `θ_borderline = 0.5` is provisional pending reference set validation.

---

## D) Tier-2 Suite Generation Rules

### Purpose
Tier-2 evaluation tests semantic discrimination: does the model maintain boundary stability under meaning-preserving transforms vs. detect actual intent shifts?

### Transform Categories

#### 1. Preserving Transforms (allowlist)
Paraphrases that preserve semantic intent:
- Synonym substitution (e.g., "buy" → "purchase")
- Sentence restructuring (active ↔ passive voice)
- Minor phrasing changes without content shift

**Expectation:** `|Δd̃(x)| ≈ 0` (boundary distance should not change significantly)

#### 2. Changing Transforms (allowlist)
Intent modifications that alter classification target:
- Negation injection (e.g., "I want to buy" → "I don't want to buy")
- Category shift (positive → negative sentiment)
- Contextual reframing that changes meaning

**Expectation:** `|Δd̃(x)| >> 0` (boundary distance should change detectably)

#### 3. Gotcha Transforms (adversarial)
Surface-preserving paraphrases with hidden intent flip:
- Subtle negation (e.g., "I'd be happy to" → "I'd be happy not to")
- Idiomatic meaning shifts
- Context-dependent sarcasm/irony

**Expectation:** Model should detect intent change (`|Δd̃(x)| > θ_flip`), but many models fail.

### Validation Requirements
- Minimum **5 human spot-checks per category** to validate transform correctness
- Document any ambiguous cases where human annotators disagree
- Record inter-annotator agreement if using multiple raters

### Metrics (Phase B)
- **Δ̃d distribution** for each category
- **FlipRate:** fraction of samples crossing decision boundary
- **Pinned/Valley/Noisy counts** (see LKEP protocol)

**STATUS:** Tier-2 suite construction is **PHASE B deliverable**. Definitions locked; implementation pending.

---

## E) Presence/Friction Tagging (Phase C Enablement)

### Purpose
Tag samples by expected "friction" — how much the model should "struggle" or show uncertainty.

### High-Friction Definition
A sample is tagged **high-friction** if:
- It sits in a borderline region (`|d̃(x)| < θ_borderline`)
- It involves ambiguous intent, edge cases, or contested semantics
- Human annotators express uncertainty or disagreement

### Expected Proxy Signals
Models processing high-friction inputs should exhibit:
- Hedging language ("possibly", "might", "unclear")
- Self-correction or qualification
- Longer response times (if measured)
- Non-zero entropy in output distributions

### Critical Rule
**"Perfectly flat outputs on high-friction inputs are suspicious."**

If a model produces:
- Identical responses across perturbations
- No hedging or uncertainty markers
- Overconfident boundary assignments

...this indicates potential **friction suppression** or lack of genuine uncertainty modeling.

### Tagging Taxonomy (provisional)
- `friction: low` — clear-cut cases far from boundary
- `friction: medium` — borderline but not contested
- `friction: high` — ambiguous, adversarial, or contested

**STATUS:** Taxonomy locked. Tagging implementation deferred to Phase C.

---

## F) Geometry Bundle (Phase E Dependency)

### Purpose
If/when Mirrorfield explores latent-space geometry (embeddings, manifolds, curvature), this section locks what must be included in a "geometry bundle."

### Required Components (when implemented)
1. **Coordinates:** Embedding vectors for all reference samples
2. **Adjacency:** Graph structure or nearest-neighbor relationships
3. **Shell Distances:** Stratified boundary proximity bands
4. **Invariants:** Topological or geometric properties (e.g., Betti numbers, curvature estimates)

### Hash/Version Control
- Bundle must include a content hash: `hash(coords + adjacency + metadata)`
- Version tag: `geometry_bundle_v<semver>`

### Phase E Fence
**STATUS:** `TBD (Phase E)`
**No claims about latent geometry are valid until this bundle exists and is referenced in results.**

If geometry is mentioned in any Phase A/B/C writeup, it must be explicitly marked as:
> "Geometry analysis deferred to Phase E. No bundle exists; claims are speculative."

---

## Version Control & Change Log

**v0.1 (2025-12-27):**
- Initial freeze for Phase 0
- Sections A-F locked as baseline
- Marked TBD dependencies (reference set, Tier-2 suite, geometry bundle)
- Fixed τ policy for Phase A; calibration deferred

**Next Review:** After Phase A evidence pack completion

---

## Cross-References
- **Roadmap:** `MIRRORFIELD_NEXT_MOVES_ROADMAP_v0.2.md` (if exists, else TBD)
- **Run Ledger:** `runs/RUN_LEDGER.md`
- **LKEP Protocol:** `LKEP_v1.2.1_clean.md` (if exists, else TBD)
- **Rotation Test:** `MIRRORFIELD_ROTATION_TEST_v0.2.md` (if exists, else TBD)

---

**END OF DEFINITIONS FREEZE v0.1**
