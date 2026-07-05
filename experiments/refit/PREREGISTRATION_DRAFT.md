# PRE-REGISTRATION DRAFT — the calibration refit (NOT LOCKED)

*Drafted 2026-07-04 with frontier-model help so a later session can execute. **Nothing
here is locked**: Dillan must read, edit, and commit-lock this file (that commit = the
lock) BEFORE any data is generated. GPU generation requires Dillan present. This is the
intervention-level refit parked since the §4y result — the twice-paid failure shape
(fixing a gap on the data that defined it) is the entire reason this document is strict.*

## Question

Does refitting the gate's calibration map — with torn-region data now represented in
training — close the near-boundary overconfidence (§4y: +0.21/+0.26) **on data the refit
never saw**, without breaking aggregate calibration or discrimination?

## Data protocol (the load-bearing part)

- **Training data (fixed, already exists):** the 1,000 §4y rows (seeds 42+1337,
  `boundary_rows_{42,1337}.npz`). These rows are hereby *retired as evidence* — they
  become training material only (consistent with the 2026-07-04 analysis stop). No
  claim may cite them post-refit.
- **Evaluation data (does not exist yet — that's the point):** fresh A1-style answer
  passes, **new seeds 7 and 2024**, 500 RTE+QNLI items each, generated AFTER this lock,
  with the refit calibrator already frozen. The evaluation rows never touch any fitting
  step. Persist raw signals + task labels (now the harness default).

## The correction (pre-specified — no post-hoc switching)

- **Primary:** refit the existing pipeline (scaler → logistic → isotonic) on the 1,000
  training rows. Freeze it (commit the fitted parameters) before generating evaluation
  data.
- **Secondary (reported alongside, never promoted):** Platt-on-margin (the Amendment-1
  map) fit on the same training rows — included because it was the diagnostic that
  motivated this, and if the simple map matches the full stack out-of-sample, that is
  worth knowing.
- Decision thresholds (tau_abstain/tau_present) stay UNCHANGED for the primary
  analysis; operating points under the new map are reported, and any threshold change
  is future work with its own lock.

## Pre-registered verdicts (evaluation = fresh rows only; both seeds; min-n 30; Wilson 95%)

- **GAP CLOSES:** torn-quintile (lowest raw-margin fifth) |mean p̂ − accuracy| ≤ 0.05,
  accuracy CI contains mean p̂, in BOTH fresh seeds; AND aggregate calibration error
  ≤ 0.06; AND discrimination (fresh-seed AUC of the refit score) ≥ 0.60.
- **GAP PERSISTS (a real, reportable outcome — not a reason to retune):**
  torn-quintile gap ≥ +0.10 with CI exclusion in both fresh seeds. If this happens it
  is evidence AGAINST the sparse-tail mechanism hypothesis as sufficient explanation,
  and it gets reported with the same prominence a success would.
- **MIXED/UNDERPOWERED:** anything else; reported, not interpreted.

## Kill criterion (pre-committed, per the methods note §7)

If the held-out torn-region calibration error of the refit gate exceeds **0.10** in
both fresh seeds, the gate is declared **not safety-usable in the low-margin region
as-is**, and that statement goes into the root README's status banner — not a footnote.

## How this could fool us (named before data)

- Seeds 7/2024 are still same-distribution GLUE items — closing the gap here says
  nothing about distribution shift; scope any success accordingly ("on this model, on
  this task family, in-distribution").
- The isotonic refit sees the torn tail now, so *in-training* improvement is guaranteed
  and meaningless — only the fresh-seed numbers exist as evidence.
- Two fresh seeds under greedy decoding = two disjoint item samples, not sampling
  variance (same honest framing as §4y).
- Success does NOT validate deployment at large: it validates one recalibration on one
  model. The second-model question is Plan J, separately.

## Effort

Fit + freeze: CPU, one session. Evaluation generation: ~1,000 generations, GPU,
Dillan-gated, ~1–2 hours on the 3060 Ti. Analysis: existing code paths
(`analyze_margin` pattern).
