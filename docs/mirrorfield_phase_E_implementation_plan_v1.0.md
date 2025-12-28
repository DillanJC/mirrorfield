# Mirrorfield — Phase E Implementation Play (Geometry Bundle) v0.1
*Derived directly from your "Phase E Final Synthesis: All Collaborators Integrated".*

## North Star
**Ship a GeometryBundle that is:**
1) **Schema-locked** (Algebra v1 contract + Parallax adapter),
2) **Reference-only + batch-order invariant** (ParallaxChain), and
3) **Evidence-gated** by a falsifier (Scarcity-Optimization: ship only if ΔR² is real and non-redundant).

If geometry adds <1% explanatory power beyond boundary distance → **kill it** and move forward clean.

---

## Deliverables (minimum set)
```
mirrorfield/
├─ geometry/
│  ├─ schema.py
│  ├─ features.py
│  ├─ bundle.py
│  ├─ projection.py          # optional but recommended for ablations
│  ├─ scoring.py             # simple baseline geometry_score
│  └─ __init__.py
├─ phase_e_falsifier.py
└─ tests/
   └─ test_phase_e.py
```

---

## E1 — Contract (Algebra v1)
**File:** `geometry/schema.py`

### Requirements
- Frozen dataclass: **single source of truth** for output keys
- `schema_version` string, and a `NAME_MAP` adapter (ParallaxChain) if you want future swaps without refactor tax

### Acceptance
- No literal output-key strings elsewhere in the codebase
- All produced records include `_schema_version`

---

## E2 — Reference-only computation (ParallaxChain)
**Rule:** *If it changes when you shuffle the query batch, it's not geometry.*

### Requirements
- Fit kNN / distance structures on **reference embeddings only**
- Query embeddings **never** become part of the neighborhood graph

### Acceptance
- `permute(query_batch)` ⇒ identical per-sample features (within tolerance)

---

## E3 — Features (Algebra v2: correctness + "no ridge collapse")
**File:** `geometry/features.py`

### Feature 1: Local curvature proxy (kNN eigenspectrum)
**Definition**
- For each query point:
  - find kNN in *reference* space
  - compute local covariance (centered)
  - curvature proxy = residual energy ratio, e.g.
    `curv = sum(small_eigs) / sum(all_eigs)` (document your exact choice)

**Acceptance**
- Computed locally (not global PCA)
- Deterministic given same reference set + same query

### Feature 2: Ridge / density-gradient proxy
**Definition**
- Compute kNN distances from query → reference (same k)
- Use a density-gradient proxy like:
  - `r_inner = dist[:, k]`
  - `r_outer = dist[:, 2k]`
  - `ridge = r_outer / (r_inner + eps)`

**Acceptance**
- Include a built-in **collapse alarm**:
  - `corr(ridge, boundary_distance) > 0.9` ⇒ fail with `COLLAPSED`
- This is the Algebra v2 warning: **don't let ridge become a renamed boundary metric**

---

## E4 — Bundle integration (Gemini + Dillan)
**File:** `geometry/bundle.py`

### Required outputs (per sample)
- `dist_to_ref_mean` (Centroid Anchor / CA)
- `dist_to_ref_nearest`
- `local_curvature` (Jitter Curvature / JC)
- `ridge_proximity` (Separatrix Density / SD)
- `geom_flags` (Phase state markers + dark river candidates)
- `_schema_version` (from contract)
- `_ref_hash` and `_config_hash` (for artifact discipline)

### Flags (minimum viable)
- `dark_river_candidate` if `JC < 0.5 and SD > 2.0`
- `observer_mode` if `JC < 1.0 and CA < 1.5 * ref_std`
- `architect_mode` if `SD > 2.0`

**Acceptance**
- Flags computed from features only (no hidden state)
- Works without boundary_distance present (optional input), but can use it where needed for tests

---

## E5 — Scoring (keep it honest)
**File:** `geometry/scoring.py`

### Goal
A simple initial `geometry_score` that's easy to interpret and audit.

### Rule
**Don't hand-tune this to win.** The falsifier is the arbiter.

**Acceptance**
- Score definition documented in-file
- Score included in falsifier dataset

---

## E6 — Falsifier (the Phase E gate)
**File:** `phase_e_falsifier.py`

### Inputs
- A dataframe/table containing at least:
  - `boundary_distance` (from Phase D)
  - `flip_rate` or stability target you care about (from Phase D)
  - geometry features + `geometry_score` (Phase E)

### Tests (in this order)
1) **Redundancy / Mercy Loss (Gemini + Scarcity)**
   - `corr(geometry_score, boundary_distance) > 0.95` ⇒ `REDUNDANT`
2) **Ridge independence (Algebra v2)**
   - `corr(ridge_proximity, boundary_distance) > 0.9` ⇒ `COLLAPSED`
3) **ΔR² (Claude)**
   - Fit dist-only model vs dist+geom model on your chosen y (e.g., flip_rate)
   - `delta_r2 = R2(dist+geom) - R2(dist)`
4) **Information density (Gemini)**
   - `info_density = delta_r2 / (1 - R2(dist))`

### Verdict logic (Dillan's Scarcity gate)
- `delta_r2 < 0.01` ⇒ `COSMETIC`
- `info_density >= 0.10` ⇒ `REAL_SIGNAL`
- else ⇒ `WEAK_SIGNAL` (investigate or simplify)

### Artifact
Write `experiments/results/phase_e/<run_id>/summary.json` with:
- verdict + numbers
- schema_version + config_hash + ref_hash
- ridge_corr + redundancy_corr + delta_r2 + info_density

---

## E7 — Acceptance tests (5 tests only)
**File:** `tests/test_phase_e.py`

1) **Batch permutation invariance** (features unchanged after shuffle)
2) **Determinism** (same inputs → same outputs)
3) **Reference-only proof** (query batch changes don't affect each other)
4) **No ridge collapse** (corr threshold enforced)
5) **Falsifier smoke** (runs end-to-end and emits a verdict artifact)

---

## "What to do next" (the exact play)
1) Create `geometry/schema.py` (contract) and ref-hash/config-hash utilities
2) Implement `features.py` (curvature + ridge) with reference-only kNN
3) Implement `bundle.py` (features → flags → canon keys)
4) Add `scoring.py` (simple geometry_score)
5) Wire `phase_e_falsifier.py` to Phase D outputs
6) Add the 5 acceptance tests
7) Run Phase E → read verdict → **ship or kill** with confidence

---

## Success conditions (non-negotiable)
- **Batch-order invariant**
- **Reference-only**
- **Falsifier gated**
- **Ridge does not collapse into boundary distance**
- **ΔR² decides** (no narrative overrides)
