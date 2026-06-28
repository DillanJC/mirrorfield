# Boundary-stratified calibration of the log-prob confidence signal — PRE-REGISTRATION

**Locked BEFORE any confidence×correctness analysis. Commit hash = the timestamp lock.**
The live question (the one the v3.0 retraction left standing): on this model, does the
log-prob confidence signal track correctness **near the decision boundary**, where it
matters — not on average. Average-case calibration on A1 was good (ECE 0.03/0.05, §4t); this
asks whether that holds, degrades, or is uninformative *where the model actually lives*.

**Scope rule for this whole task (applies to the writeup):** no sentence in any result or
summary line generalises past this one model — Qwen2.5-3B on RTE+QNLI — unless it says "on
this model" in the sentence itself. A null is a result; an underpowered region is reported as
underpowered, not interpreted; thin bins are not smoothed.

## Data
A1 saved rows (`selfreport_confidence/selfreport_rows_{42,1337}.npz`): per item
`[task, correct, p_int, verbal, ans_ok, conf_ok]`. **Raw token-margin/entropy/boundary were
NOT persisted — only the calibrated `p_int`.** Marginal of the stratifier (already inspected;
NOT crossed with correctness): `p_int ∈ [0.625, 1.0]`, std ≈ 0.047, **~85% of items in the
verify band `[τ_abstain=0.7763, τ_present=0.8684)`**; the confident/torn tails are nearly empty.

---

## OPTION 1 — calibrated `p_int` axis (no generation)

**Metric:** the deployed calibrated confidence `p_int` (justification: raw margin not on disk;
`p_int` is the deployed signal and a monotone summary of certainty). "Near the boundary" =
the gate operating band `[τ_abstain, τ_present)` + the low (abstain-side) tail. **Acknowledged
limitation:** `p_int` is compressed (std 0.047), so the axis has little dynamic range — this is
exactly why a too-wide slope CI must be called *underpowered-by-compression* rather than read
as "flat."

**Bins (fixed, threshold-anchored — NOT fit to the outcome):**
`0.625 | 0.7763 | 0.80 | 0.82 | 0.845 | 0.8684 | 0.95 | 1.001`
(abstain-tail | four verify-band sub-bins | present-side | high-tail.)

**Min-n:** a bin with **n < 30 (per seed)** is reported underpowered (count + Wilson CI only),
**not interpreted.** Both tails are expected to fail this; that is reported as "cannot say."

**Per bin, per seed:** n, mean `p_int`, empirical accuracy, **Wilson 95% CI** on accuracy,
calibration gap (mean `p_int` − accuracy). Reliability points vs the diagonal. No smoothing.

**Two distinct, separately-reported questions:**
1. **GAP (calibration level), per bin:** is mean `p_int` ≈ accuracy?
   - **HOLDS** — every interpretable (n≥30) bin's accuracy CI contains its mean `p_int`.
   - **DEGRADES** — an interpretable bin's gap CI excludes 0 in a consistent direction and
     exceeds the aggregate ECE.
2. **SLOPE (within-band discrimination) — the real near-boundary question (tightening a):**
   does accuracy *rise with* `p_int` **within the verify band**? Estimate
   `s = OLS slope of correct on p_int` over items with `p_int ∈ [τ_abstain, τ_present)`
   (linear-probability units, so the calibration ideal is slope = 1.0); 2000-rep bootstrap CI
   `[s_lo, s_hi]`. "Meaningful positive slope" pre-set at **0.5** (accuracy tracks ≥ half of
   `p_int`'s movement). Verdict:
   - **RISING** — `s_lo > 0` (within-band signal is informative; report `s` vs the ideal 1.0).
   - **FLAT (no within-band slope)** — `s_lo ≤ 0` AND `s_hi < 0.5` (rules out a meaningful
     positive slope; `p_int`'s within-band variation carries no correctness information).
   - **UNDERPOWERED-BY-COMPRESSION (tightening b)** — `s_lo ≤ 0` AND `s_hi ≥ 0.5` (the CI spans
     flat to meaningfully-rising → the compressed axis + finite n cannot separate the two;
     this is an instrument limit, NOT a clean null — Option 2 is the resolution).
   Verdict must replicate (same category both seeds), else "inconclusive across seeds."

**Honest prior:** given the compression, UNDERPOWERED-BY-COMPRESSION on the slope is the
*expected* Option-1 outcome — which is precisely why Option 2 is pre-authorised to run after.

---

## OPTION 2 — raw token-margin axis (pre-authorised; ~one A1 answer-pass re-gen, resumable)

**Why generate:** Option 1's axis is compressed and likely won't resolve the slope. Re-running
the A1 **answer pass only** (greedy → byte-identical answers ⇒ identical `correct`), persisting
the raw `mean_margin (mm)`, `mean_entropy`, `boundary_ratio`, and `p_int`, gives a
higher-dynamic-range boundary axis. ~500 generations/seed (no confidence pass). Raw text →
`local_outputs/` (gitignored); npz = aggregates only.

**Metric:** `mm` = mean token margin (top1−top2 logprob over the answer tokens). Small `mm` =
model torn = nearest its yes/no decision boundary. (Entropy reported as a secondary axis.)

**Bins:** **equal-n quintiles of `mm`** (5 bins ≈ 100 items each — a rule using only the `mm`
marginal, fixed before crossing with correctness). Min-n 30 (quintiles clear it).

**Per bin, per seed:** n, mean `mm`, mean `p_int`, accuracy, Wilson 95% CI, gap (`p_int` −
accuracy). Two questions, mirroring Option 1:
1. **Discrimination near the boundary:** does accuracy rise across `mm`-quintiles — and
   specifically, is the **lowest-margin quintile** (most torn) materially lower-accuracy than
   the highest? Report accuracy(Q5) − accuracy(Q1) with bootstrap CI. (RISING / FLAT /
   UNDERPOWERED by the same CI logic.)
2. **Calibration in the near-boundary (low-`mm`) stratum:** is mean `p_int` ≈ accuracy in the
   lowest-margin quintile (gap CI contains 0 = HOLDS; excludes 0 = DEGRADES/inverts)?
Both seeds; replicate before claiming; no smoothing; tails/thin cells reported as underpowered.

## Controls / honesty
External GLUE gold (never the method's own output); fixed, externally-anchored bins (Opt 1) /
pre-set quintile rule (Opt 2); Wilson CIs on every cell; both seeds with replication required;
explicit underpowered + underpowered-by-compression verdicts; no smoothing, no retuning.
