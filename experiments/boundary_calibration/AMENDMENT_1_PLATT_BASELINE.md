# Amendment 1 to the §4y pre-registration — Platt-on-margin baseline (locked BEFORE analysis)

*2026-07-04, autonomous session (see `AUTO_LOG.md`). Extends PREREGISTRATION.md
(cc62bd8). CPU-only; reads the already-saved `boundary_rows_{42,1337}.npz`; no
generation; the deployed gate, its calibrator, and its thresholds are untouched.
This is a **diagnostic baseline**, not the intervention-level refit — the refit stays
parked under its own protocol (METHODS_NOTE.md §7). Verdict below will be a
**candidate**; Dillan concludes.*

## Question

Is the torn-region overconfidence (§4y: +0.21/+0.26) specific to the **frozen**
production calibrator (scaler→logistic→isotonic, fit on earlier data), or does a
**minimal single-feature map, freshly fit with torn-region data available**, match tail
accuracy on a disjoint sample? This bears directly on the mechanism HYPOTHESIS
("calibration fit where data is dense, extrapolates into the sparse tail") — the
staleness/sparsity arm of it — without touching deployment.

Honest naming note: the literature baseline is temperature scaling (Guo et al. 2017 —
citation to be verified before any writeup), which requires logits. Logits were not
persisted; the saved scalar raw mean-margin (mm) admits **Platt scaling** (a logistic
map on one scalar), the closest single-parameter-family analog available post-hoc.

## Method (locked)

- Data: `boundary_rows_{seed}.npz` for seeds 42/1337; rows with `correct >= 0`,
  finite `mm` and `p_int` (identical mask to §4y Opt-2).
- Baseline map: logistic regression of `correct` on `mm` alone —
  `sklearn.linear_model.LogisticRegression(C=1e6, solver="lbfgs")` (near-MLE; no
  regularization tuning).
- **Cross-seed held-out, both directions:** fit on seed 42 → evaluate on seed 1337, and
  fit on 1337 → evaluate on 42. No within-sample evaluation is reported.
- Evaluation (identical to §4y Opt-2): quintiles of `mm` computed on the *evaluation*
  sample; per quintile, mean predicted p̂ vs accuracy with Wilson 95% CI;
  `gap = mean p̂ − accuracy`; min-n 30. Side-by-side, the frozen calibrator's gap on the
  same rows (known Q1 values: +0.21/+0.26).
- Primary contrast: **Q1** (the torn quintile).
- One analysis, as specified. No refits, feature additions, or alternative maps after
  seeing results.

## Pre-registered verdicts (locked)

- **FRESH-MAP-CALIBRATED** — in BOTH directions, Q1's Wilson CI contains the fresh
  map's mean p̂. Reading (ceiling): *consistent with* the sparse/stale-tail arm of the
  mechanism hypothesis; NOT a validated fix and NOT the refit — it says only that a
  simple map fit with tail data available can match tail accuracy out-of-sample.
- **FRESH-MAP-ALSO-OVERCONFIDENT** — in BOTH directions, Q1 gap ≥ +0.10 AND the CI
  excludes mean p̂. Reading: the failure is NOT explained by staleness plus a simple
  map; either the correct-vs-mm relation is non-logistic in the tail or something
  deeper is wrong; the mechanism hypothesis as stated needs revision.
- **MIXED / UNDERPOWERED** — anything else (including directions disagreeing).
  Reported, not interpreted.

## How this could fool us (named in advance)

- Seeds 42/1337 are same-distribution samples (same GLUE pools, same model, greedy) —
  "out-of-sample" here means disjoint items, NOT distribution shift; a map could pass
  here and still fail under drift. Scope any conclusion accordingly.
- mm alone omits me/br; if tail calibration *requires* the extra features, the
  single-feature map is handicapped — a FRESH-MAP-ALSO-OVERCONFIDENT verdict partly
  reflects that, which is why its reading stops at "needs revision," not "refuted."
- All of §4y's scope limits apply: one 3B model, one task family, on this model.
