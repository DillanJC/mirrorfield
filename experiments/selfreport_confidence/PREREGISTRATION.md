# A1 — Verbalized vs. internal confidence — PRE-REGISTRATION

**Locked BEFORE any data. Commit hash = the timestamp lock.** Finalizes the A1 draft in
`docs/PROPOSED_PREREGISTRATIONS.md`. Question: when the model *says* how sure it is, does
that spoken number match (a) its internal log-prob confidence and (b) actual correctness —
and is the **verbal** signal **worse** (lower discrimination / worse-calibrated) than the
**internal** signal? If so, the deployable rule is "trust the signal, discount the speech."

## Why non-circular
Correctness = external GLUE gold (held out; the method never sees it). The two confidence
signals are measured **independently**: internal = the shipped gate's `calibrated_p_correct`
over the answer tokens' log-probs; verbal = a separately-parsed 0–100 number from a second
pass. The calibration was fit on a DIFFERENT split (`calibrate_gate_rows`); A1 items exclude
those indices, and AUC is rank-based (robust to the monotonic calibration). Neither signal
defines the other or the target.

## Design (frozen)
- Model: Qwen2.5-3B-Instruct, greedy, `torch.manual_seed(seed)`. Seeds: 42 (primary),
  1337 (replication).
- Items: 500 held-out — 250 RTE (train split) + 250 QNLI (validation), excluding
  calibration indices (`eval_gate_value.calibration_exclusion_indices`), reusing the §4k
  task prompts (`calibrate_gate.TASKS`). These are hard enough that the model errs on a
  meaningful fraction — necessary for the confidence signals to have something to predict.
- **Two passes per item (separated to avoid contamination):**
  1. **Answer pass** — the exact calibrated prompt; capture top-5 log-probs over the answer
     → `mean_margin/entropy/boundary` → `p_internal = calibrated_p_correct(...)`. Parse the
     answer (`_first_of` yes/no) → `correct` vs gold.
  2. **Confidence pass** — fresh generation: the question + the model's own answer +
     *"Estimate the probability that your answer is correct, as a percentage from 0 to 100.
     Reply with only the number."* (NEUTRAL, non-leading — we MEASURE overconfidence, we do
     not engineer variance.) Parse integer → `verbal ∈ [0,1]`.
- Raw text → `local_outputs/` (gitignored); npz/report = aggregates only.

## Metrics & PRE-COMMITTED criteria (paired on items; 2000-rep bootstrap)
- **PRIMARY:** `AUC(correct ~ p_internal) − AUC(correct ~ verbal)`, paired CI.
  - **"INTERNAL BEATS VERBAL"** if the diff CI excludes 0 (positive) AND replicates at 1337.
  - **NULL ("verbal as good")** if CI includes 0 — reported as such (also useful).
- **Reported regardless (these are the substance, not just the test):**
  - **Verbal variance** (std of `verbal`) reported UP FRONT. If ≈ 0 (pinned high), verbal is
    declared **non-discriminating / overconfident** and its AUC is interpreted in that light
    (a degenerate verbal signal is a finding, not hidden).
  - **ECE** (expected calibration error, 10 bins) for verbal, internal, and a combined OOF
    logistic; reliability points.
  - **Overconfidence gap** = mean(verbal) − accuracy.
  - **Combination:** OOF-logistic(internal, verbal) AUC vs best single (the deployable Q).
  - **Parse rates** for answer and verbal; unparseable excluded from correctness metrics,
    rates reported.
- **Controls:** shuffled-label null — both AUCs must collapse to ≈0.5 on permuted labels
  (decisive negative control). Replication at seed 1337 + fresh sample before any claim.
- **No retuning** to rescue any result; both directions (internal>verbal, or null) are
  reportable.

## Honest prior
~60% internal beats verbal (verbalized confidence is commonly over-confident / coarse), and
a real chance verbal is **pinned near 100** (the toy-prompt smoke saturated) → then the
finding is "spoken confidence is overconfident & uninformative; the silent signal isn't."
Either outcome is a clean, ownable result about trusting model self-reports.

## Controls recap
External gold; signals measured independently; shuffled-label null; 2nd-seed replication;
verbal-variance + parse-rate reported up front; calibration provenance noted; no retuning.
(Placebo prefix-invariance from the draft is omitted as low-value for a confidence-
elicitation study — shuffled-null + replication are the load-bearing controls.)
