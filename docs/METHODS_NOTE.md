# Stress-testing the survivor: a validated uncertainty gate is overconfident exactly where decisions are hardest

*Methods note — Qwen2.5-3B-Instruct on RTE+QNLI. Dillan (DillanJC/mirrorfield), with AI
assistance. Reviewed and finalized by the author 2026-07-04; all numbers reconciled
against the repository's result files the same day. Circulation is the author's decision.*

> **Scope, up front.** Everything in this note is measured on **one small model**
> (Qwen2.5-3B-Instruct, greedy decoding, one consumer GPU) on **one task family**
> (RTE+QNLI). No claim below extends past that; sentences that could read as general are
> deliberately marked "on this model." The five prior experiments summarized in §2 are
> **replications of known phenomena, not novel findings.** What is offered here is a
> *method* — an artifact-catching discipline with a track record — and one new, replicated,
> audited observation produced by turning that discipline on the project's own last
> surviving positive result.

## 1. The arc: what was left standing, and why we tested it

This project's original headline — an embedding-geometry "poisoning detector" at AUC
0.947 — was retracted when an honest baseline collapsed it to ~0.47: the detection target
and the method shared the same geometric assumption, so the number was real but the
inference was circular (WORK_MAP §4, the v3.0 retraction; preceded by a v2.0
positive-overclaim of the same family). Out of that failure came the apparatus described
in §2 — and exactly one validated result survived it: a modest log-prob uncertainty gate
(mean token margin, entropy, boundary ratio → scaler → logistic → isotonic → a calibrated
p(correct), "p_int"). On this model it discriminates wrong answers at AUC 0.685
[0.642, 0.730] live (§4k; 0.60–0.74 per task), with clean **aggregate** calibration error
of 0.03–0.05 (§4t). It is an uncertainty signal, not a safety detector, and we have said
so throughout.

The survivor, though, had only ever been graded **on average**. If the gate is used the
way it is intended — deciding send / verify / abstain — then the operative property is
its calibration **on the items nearest its decision boundary**, where the model is
genuinely torn. An average can hide a lot. So we turned the same discipline that produced
the retraction on the one result it had spared: *does the gate's calibration hold where
data is scarce and decisions are hard, not just in aggregate?*

It does not.

## 2. The apparatus, and why its track record makes the test meaningful

The discipline (full statement: `experiments/EVALUATION_DISCIPLINE.md`): success **and
abandon** criteria pre-registered and commit-locked before data; ground truth external to
the method under test (dataset gold labels, an independent harm classifier, a canary
token — the method never grades itself); placebo/baseline controls for every
manipulation; shuffled-label nulls; replication on a second seed before any claim;
operating points reported, not just AUCs; a null is a result; no threshold retuned to
rescue a finding.

Applied under that discipline, five experiments replicated known self-monitoring
phenomena on this model — **context and method-demonstration, not contributions**:

| exp | § | replicated observation (all: on this model, both seeds) |
|---|---|---|
| A1 | §4t | spoken confidence predicts own correctness at chance (AUC 0.51/0.51; calibration error 0.32–0.36); the internal log-prob signal is modestly predictive (0.64/0.66; error 0.03–0.05) |
| B1 | §4u | refusals 90% stable across 8 rewordings (×2 wrapper sets); 5/50 replicated seams; 29% of a *boundary-adjacent* benign set refused |
| C1 | §4v | stable to a neutral re-ask (1.5% flips) but flips ~40 pts under explicit pushback vs that clean baseline; the internal signal does **not** drop to flag the flip (mean Δ +0.03) |
| B3 | §4w | prompt-injection compliance 80–96% from a 0% control floor; internal confidence on hijacked outputs ≈ clean (0.843 vs 0.848) |
| B3b | §4x | a system-prompt hierarchy defense shaves ~12–16 pts off weak/medium injection, ~nothing off a hard override (95%→93%, CI incl. 0) — at 3B scale |

Why recount replications in a methods note? Because of what happened *while producing
them*: the apparatus caught **three would-be-misleading artifacts before anything was
claimed** — an unscaled-logistic bug that had zeroed out a combined model's AUC (A1); a
contaminated placebo (the "double-check" control itself implied doubt), flagged by an
external critical read and then **tested** with a clean re-ask control, which roughly
*doubled* the measured sycophancy effect rather than rescuing the original number (C1,
§4v); and an instrument-validity failure in a harm-framing measure (§4r). A discipline
that catches errors in both directions — against our hopes and against our fears — is
what makes pointing it at our own surviving result informative rather than theater.

## 3. The test: boundary-stratified calibration (§4y)

Pre-registered before any confidence×correctness analysis (commit `cc62bd8`): fixed bins,
min-n 30 per interpretable bin, Wilson 95% intervals, verdict rules, both seeds (42/1337;
500 fresh RTE+QNLI items per seed, gold labels external, gate frozen long before
sampling). Two arms:

**Arm 1 — the score's own axis (no new generation).** The calibrated p_int turns out to
be severely compressed on this model: std 0.047, ~85% of items inside the gate's verify
band [0.7763, 0.8684). Within that band, the pre-registered slope of correctness on p_int
came back **RISING** — 1.60 [0.51, 2.68] and 2.55 [1.33, 3.72] per seed — where our
pre-registered expectation had been "underpowered-by-compression." We report that
prior-miss as it happened; the locked verdict rules were applied unchanged. The torn tail
(p_int below the abstain threshold) held under 30 items per seed: **underpowered by the
pre-registered rule, reported, not interpreted.** Conclusion of Arm 1: the compressed
score axis cannot reach the region the question is about.

**Arm 2 — the raw axis (pre-authorized re-generation).** The pipeline had discarded the
raw signals, keeping only the calibrated score — itself a lesson about persistence
defaults, since the operationally interesting axis was unrecoverable from disk. We re-ran
the identical answer pass persisting raw mean-margin (and entropy/boundary-ratio), with a
locked validity check: the re-generation reproduces Arm 1's p_int to max|Δ| = **0.0**,
confirming byte-identical generation — the same sample, re-analyzed on the raw axis.
Items were then stratified by raw-margin quintiles.

## 4. The result: aggregate calibration hid a near-boundary failure

Stated in the conditional form the audit recommends (§5): **on the items the gate's own
raw margin marks as most torn (lowest quintile), the gate reports a mean p(correct) of
0.79 while actual accuracy is 0.58 and 0.53 by seed — overconfident by +0.21/+0.26,
replicated, with the accuracy interval excluding the reported score (§4y).** In the
denser mid and upper quintiles, calibration holds — which is exactly why the aggregate
error of 0.03–0.05 (§4t) looked clean. The failure sits precisely where the gate is
supposed to earn its keep, on this model: in the region where the model is most often
wrong.

**Figure:** `experiments/boundary_calibration/boundary_reliability.png` — accuracy
(Wilson 95%) vs the frozen calibrator vs the held-out Platt baseline (§6), per margin
quintile, both seeds. The frozen line is nearly flat while accuracy collapses in the
torn quintile; pure visualization of the logged §4y/§4z numbers.

Two companion facts sharpen it (both §4y, both seeds):

- **The raw signal itself is fine.** Accuracy rises monotonically across margin
  quintiles; acc(Q5) − acc(Q1) = +0.36 [0.26, 0.47] and +0.37 [0.25, 0.49]. The
  *discrimination* survives; it is the calibrated **mapping** that fails near the
  boundary — on this model.
- **The score collapses distinct risks.** Nearly all items (488/500, 487/500) receive
  p_int ≥ 0.78; among only those matched-high-score items, the low-margin half runs
  0.75/0.68 accuracy against the high-margin half's 0.93/0.89 (audit §2d). One score
  value, ~20 points of real risk spread.

The transferable methods point — offered as a question for other setups, not a claim:
**a single aggregate calibration number can conceal a decision-relevant failure.** Test
calibration *conditional on the rawest boundary-distance signal available*, not only on
the score's own axis — the score's axis may be compressed by the very calibration step
being evaluated, as it was here.

## 5. Audit: how this result could have been fooling us, tested

Because the stratifier (raw margin) is itself a gate feature, and because this project
has been burned by circularity before, the analysis was audited against the retraction's
failure shape before this note was written
(`experiments/boundary_calibration/CIRCULARITY_AUDIT.md`, commit `2a4d75f`; diagnostics
reproducible, no GPU; verdict accepted by the author 2026-07-04):

- **No shared target/method assumption.** Correctness is external gold end-to-end;
  no path from labels to features. Binning by a gate input and asking whether the score's
  average matches accuracy inside the bin is the correct conditional probe — and margin
  is available at runtime, so the failure is operationally real.
- **Parse-failure selection: ruled out** (0/500 exclusions, both seeds).
- **Task-mix confound: tested.** The torn quintile is QNLI-heavy (~70–75%), so the rival
  reading was "overconfident on the harder task-mix." Within QNLI alone the torn-region
  gap **replicates** (+0.196/+0.334, n ≥ 70 per seed); the RTE-only slice is positive in
  both seeds but small-n — underpowered, reported, not interpreted.
- **Calibrator-floor artifact: ruled out.** Only 5–6% of torn items sit near the
  calibrator's output floor (0.625); the torn-quintile median is 0.7897 — the calibrator
  *actively assigns* ~0.79 there (an isotonic plateau spans much of the torn region).
- **Stratifier-specificity: tested** (Amendment 2, lock `ddaed33`). The torn-region
  overconfidence replicates when stratifying by mean *entropy* instead of margin
  (+0.16/+0.22, accuracy CI excluding the score, both seeds) — the failure is not an
  artifact of the margin axis (entropy correlates with margin; this is robustness, not
  independence). The boundary-ratio axis is too tie-heavy to stratify cleanly (reported,
  not interpreted).

## 6. Mechanism — a hypothesis, not a finding

The candidate story: **the calibration map was fit where its training data was dense and
extrapolates into the sparse low-margin tail** — hence clean calibration in the dense
region and an unsupported plateau in the tail. This is a *hypothesis*, throughout this
note. The observed plateau (§5) is consistent with it, not confirmation of it. It is
stated because it is load-bearing for the planned refit (§7) and therefore must be
tested, not assumed. Testable implications: the calibrator's training density in the
torn region should be low; a refit including torn-region data should close the gap
**out-of-sample** (§7); and a simpler baseline map should be checked for the same tail
behavior before the story is believed.

That baseline check has now been run, under a pre-locked amendment
(`AMENDMENT_1_PLATT_BASELINE.md`, lock `f0c5b7f`; logits weren't persisted, so Platt
scaling on the saved raw margin stands in for temperature scaling). Cross-seed held-out,
both directions: a fresh single-feature map fit *with torn-region data available* lands
inside the torn quintile's accuracy interval (gaps +0.09/+0.04) where the frozen
calibrator is off by +0.26/+0.21 on the same rows — verdict **FRESH-MAP-CALIBRATED**
(per rules locked before analysis; accepted 2026-07-04), consistent with — still not
confirmation of — this hypothesis. For completeness: the fresh map's *mid*-quintile fit
is imperfect (one mid bin per direction misses the accuracy interval; all bins are in
`platt_baseline_results.json`) — the torn bin was the pre-registered primary contrast,
not a selected one, and the comparison's point is the tail, where the frozen calibrator
fails and the fresh map does not. Caveats: one direction passes by only 0.007, and the
seeds are same-distribution, so this says nothing about drift. The deployed gate is
unchanged; the §7 refit protocol still applies in full.

## 7. Implication, and future work — named, not run

The scoped implication: **on this model and task family, treat the gate's reported
p(correct) in the low-margin region as overstated by roughly 0.2** — trust it least
exactly where the underlying model is most often wrong. This is a statement about one
gate on one model; the general form is only the question in §4.

An anticipated objection, answered: *"the gate is modest (AUC 0.685) — why does the
miscalibration of a weak signal matter?"* Because calibration is precisely what makes a
weak signal *usable*: a weak-but-honest probability can still drive sensible
abstain/verify decisions at a known error cost, while a weak **and overconfident** one
actively misprices risk exactly where decisions are hardest. Miscalibration doesn't just
reduce the gate's value — in the torn region it inverts it.

**The refit (deliberately not attempted here).** Correcting the low-margin overconfidence
is the obvious next step and carries this project's twice-paid failure shape: a
correction fit and evaluated on the same torn rows closes the gap **by construction** and
means nothing. So the protocol is committed to in writing before any refit work begins:

1. pre-register the train/test split **before touching** torn-region data;
2. evaluate only on torn-region items the refit never saw;
3. pre-register that a gap **failing** to close out-of-sample is a real, reportable
   outcome — not a reason to retune;
4. pre-commit the kill criterion in writing before looking (e.g. "if held-out torn-region
   calibration error exceeds X, the gate is not safety-usable in that region as-is").

Also named for later: **true temperature scaling** (requires persisting logits — the
Platt analog in §6 is the best the saved scalars allow); a **second-model transfer run**
(protocol drafted, not locked: `plans/J-second-model-transfer.md`; the 8 GB VRAM ceiling
is the binding constraint, and this is the only real path off "on this model"). Done
since drafting: raw-signal persistence is now the pipeline-wide default (Arm 2 existed
only because the raw axis had been discarded; every live harness now writes it).

## 8. Limitations

One small open model; one task family; behavioral, not mechanistic. Every interpreted
claim above is replicated across two seeds — which here means two disjoint 500-item
samples under deterministic greedy decoding, not sampling variance. Tails below the
pre-registered min-n are reported, never interpreted (Arm 1's abstain tail; the RTE-only
torn slice). The mechanism in §6 is a hypothesis. The five §2 results are replications of
known phenomena and claim no novelty. Scale-dependence is untested throughout: larger or
differently-trained models may behave differently on every count.

## 9. Reproducibility

Pre-registrations commit-locked before data: `cc62bd8` (§4y; bins, min-n, verdict rules,
both arms), amendments `f0c5b7f` (Platt baseline) and `ddaed33` (verbal + stratifiers),
per-experiment `PREREGISTRATION.md` files for A1/B1/C1/B3/B3b. Results:
`experiments/boundary_calibration/` (`boundary_calibration.py`, `boundary_pint_results.json`,
`boundary_margin_results.json`, `platt_baseline.{py,_results.json}`,
`verbal_boundary.{py,_results.json}`, `boundary_rows_{42,1337}.npz`, the figure +
`make_figure.py`); audit + diagnostics: `CIRCULARITY_AUDIT.md`, `audit_diagnostics.py`
(commit `2a4d75f`). Result log: `WORK_MAP.md` §4t–§4z. Model: Qwen2.5-3B-Instruct,
greedy; seeds 42/1337; one RTX 3060 Ti (8 GB); no paid API. Aggregates are public; raw
generations stay in gitignored `local_outputs/`.

## 10. Related work (citations verified by web search, 2026-07-04)

**The positioning correction this pass forced (read first).** The general phenomenon —
*aggregate calibration masking systematic miscalibration on subpopulations, with Platt/
temperature scaling giving no subgroup guarantee* — is **already named** in the
multicalibration literature ([Hébert-Johnson et al. 2018](https://arxiv.org/abs/1711.08513);
see also [When is Multicalibration Post-Processing Necessary?](https://arxiv.org/abs/2406.06487)).
So this note must NOT claim the phenomenon. What it claims is the **instance and the
demonstration**: the miscalibrated subgroup here is *decision-boundary proximity measured
by the gate's own input*, on a deployed LLM uncertainty gate, found by turning a
pre-registered falsification apparatus on the project's own surviving result — with the
compressed-score-axis masking mechanism worked out and audited. §4's "transferable
methods point" should be read as an application of the multicalibration lens to LLM
uncertainty gates, not a new observation about calibration in general.

- **Calibration / recalibration:** [Guo et al. 2017](https://arxiv.org/abs/1706.04599)
  (ICML) — modern networks miscalibrated; temperature scaling as the one-parameter fix.
  Amendment 1's Platt-on-margin is this line's nearest available analog (logits not
  persisted).
- **Selective prediction / abstention:** [Geifman & El-Yaniv 2017](https://arxiv.org/abs/1705.08500)
  (NeurIPS) — risk-coverage trade-off; the gate's present/verify/abstain design is this
  framing; §4y says the risk estimate driving it is biased exactly where abstention
  matters, on this model.
- **LLM self-knowledge:** [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221) —
  *larger* models are well-calibrated at self-evaluation (a scale contrast that supports
  keeping every claim here "on this model"). [Tian et al. 2023](https://arxiv.org/abs/2305.14975)
  (EMNLP) — on RLHF'd frontier assistants, *verbalized* confidence beat conditional
  probabilities — the **opposite** ordering to A1's chance-level verbalized confidence on
  this 3B model; the divergence is itself evidence that the verbal/internal ordering is
  scale- and training-dependent, not universal in either direction.
- **Sycophancy:** [Sharma et al. 2023](https://arxiv.org/abs/2310.13548) — sycophancy is
  consistent across frontier assistants and traceable to preference data; C1 is a
  disciplined small-model replication with a contamination-tested control.
- **Prompt injection / instruction hierarchy:** [Greshake et al. 2023](https://arxiv.org/abs/2302.12173)
  (AISec) — indirect injection formalized; [Wallace et al. 2024](https://arxiv.org/abs/2404.13208)
  — the instruction-hierarchy training defense; note also the 2025 line on
  instruction-hierarchy *failures* ([arXiv:2502.15851](https://arxiv.org/abs/2502.15851)),
  consistent with B3b's scale-dependent floor at 3B.
- **Over-refusal:** [Röttger et al. 2023, *XSTest*](https://arxiv.org/abs/2308.01263)
  — hand-crafted safe/unsafe contrast pairs; [OR-Bench 2024](https://arxiv.org/abs/2405.20947)
  — scaled synthetic over-refusal. B1's 29% on boundary-adjacent benign items sits in
  this line's territory and inherits its central caveat (boundary-adjacent ≠
  representative traffic).

