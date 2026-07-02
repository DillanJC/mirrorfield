# Circularity audit of the §4y boundary-calibration result

*2026-07-02. Mandated by the pre-forum plan ("audit the calibration analysis code for the
same circularity that killed 0.947, before anything goes near the forum"). Audited:
`boundary_calibration.py` (analysis code), `PREREGISTRATION.md` (cc62bd8), the saved rows
(`boundary_rows_{42,1337}.npz` + Opt-2 JSONL checkpoints), and the §4y result-lines.
Diagnostics reproducible via `audit_diagnostics.py` (read-only, no GPU).*

> **Status: audit diagnostics, not findings.** This surfaces candidate verdicts; whether
> §4y stands is Dillan's call. Nothing here is pre-registered; everything here is a
> robustness check *of* a pre-registered result. All numbers are Qwen2.5-3B on RTE+QNLI —
> on this model, on this task family.

## 1. The core circularity question (the 0.947 shape): CLEAN

The v3.0 failure shape was: *detection target and method share the same assumption*
(poison defined by geometry, detected by geometry). Checked here:

- **Target:** correctness = `pred == gold` (GLUE gold labels) — external, fixed before
  the project existed, never touched by any gate feature. (`boundary_calibration.py`,
  regen loop.)
- **Method:** log-prob features (mm/me/br) → frozen calibrator → p_int. The features are
  computed from generation logits only; no path from labels into the score at eval time.
- **Verdict candidate:** no shared assumption. The target is not defined by the method.

**The subtler worry — the stratifier (raw margin) is itself a gate feature.** Is binning
by mm and comparing p_int to accuracy inside the bin circular? No — it is the correct
probe, for a precise reason: if p_int were actually the best estimate of correctness
*given the gate's own inputs*, then its average inside ANY region defined by those inputs
would have to match the accuracy there. A gap inside an mm-defined bin is therefore a
genuine failure of the score with respect to information the gate itself computes — and
mm is available at runtime, so the failure is operationally real, not a statistical
curiosity. What this does mean for wording: see §3.

## 2. Rival explanations, tested (audit_diagnostics.py output, both seeds)

**2a. Parse-failure selection — RULED OUT.** 0/500 unparseable items, both seeds. No
exclusion, no selection bias.

**2b. Task-mix confound — TESTED, does not rescue the gate.** Q1 (the torn quintile) is
QNLI-heavy (75/25 and 70/30 vs 50/50 overall), so "overconfident near the boundary" had a
rival: "overconfident on the harder task-mix." Within-task gaps in Q1:

| seed | QNLI-only gap (p_int − acc) | RTE-only gap |
|---|---|---|
| 42 | **+0.196** (n=75) | +0.258 (n=25) |
| 1337 | **+0.334** (n=70) | +0.081 (n=30) |

The torn-region overconfidence **replicates within QNLI alone** (n≥70 per seed). The
RTE-only slice is positive in both seeds but small-n → reported as underpowered, not
interpreted (per standing rule). Task mix shifts the magnitude; it does not explain the
effect away.

**2c. Isotonic-floor artifact — RULED OUT.** If the calibrator's output floor (0.625)
were doing the work, Q1's p_int would pile up at the floor. It doesn't: only 5–6% of Q1
sits at ≤0.65; the Q1 median is 0.7897 (both seeds). The calibrator *actively assigns*
~0.79 to torn items. Notable observed structure: p25 = median = 0.7897 in both seeds —
an isotonic **plateau** at 0.7897 spans a large share of the torn region. (That a sparse
low-margin training tail would produce exactly such a plateau is consistent with the
mechanism story — which **remains a hypothesis**; the plateau is the observed fact, the
"why" is not.)

**2d. Score-conditional check — the compression, made concrete.** 488/500 and 487/500
items have p_int ≥ 0.78 (the compressed axis). Among only those matched-high-score items,
splitting at the median margin: accuracy 0.754 vs 0.930 (seed 42) and 0.683 vs 0.889
(seed 1337). At essentially the same score, the gate's own margin input still separates
accuracy by ~18–21 points. The score collapses genuinely different risk levels into one
value — this is the precise mechanism by which aggregate ECE stayed clean while the torn
region went unflagged.

## 3. One wording precision for the methods note

State the claim in its **conditional** form: *"on items the gate's own raw margin marks
as torn (lowest quintile), accuracy is ~0.53–0.58 while the gate reports ~0.79 —
overconfident by +0.21/+0.26, replicated."* Avoid the unqualified "p_int is miscalibrated":
on the compressed score axis, marginal (score-binned) calibration can look fine — that is
exactly how the aggregate ECE of 0.03 hid the failure, and the distinction is the
transferable methods point.

## 4. Implementation read — correct, three minor notes

- Wilson intervals: standard, correct. Bootstrap: paired resampling, seeded,
  2000 draws — correct. Verdict logic matches the pre-registered rules (incl. the Opt-1
  RISING outcome that defied the pre-registered prior — reported as a prior-miss, not
  refit).
- Validity check confirmed: re-gen reproduces A1's p_int to max|Δ| = 0.0 → identical
  generation; Opt-2 is the same sample re-analyzed on the raw axis, and "replication"
  means two disjoint item samples (42/1337), generation itself being greedy/deterministic.
- Minor: (a) mean-p_int sampling error is ignored when compared against the accuracy CI —
  negligible at n=100 with p_int std ≈ 0.02–0.05, and the direction is conservative;
  (b) quintile edges are computed in-sample — a pre-registered design choice, no free
  parameters; (c) the npz files drop the task label (the JSONL checkpoints retain it) —
  a persistence gap worth fixing pipeline-wide, same lesson as the raw-margin
  persistence fix already on the board.

## 5. Data-reuse direction-of-bias note

The calibrator was frozen (§4k era) before the A1/§4y items were sampled (seeds 42/1337).
Whether any individual GLUE items overlap the calibrator's original training pool was not
verified here — but the direction of any such bias **favors the gate** (in-sample items
calibrate better), so reuse cannot manufacture the overconfidence; it could only have
shrunk it. The finding is, if anything, conservative on this axis.

## 6. What this audit does NOT establish

- The mechanism ("fit where dense, extrapolates into the sparse tail") is still a
  hypothesis; §2c's plateau is consistent with it, not confirmation of it.
- Nothing here extends past Qwen2.5-3B on RTE+QNLI.
- The RTE-only torn-region slice stays underpowered.
- The refit and its held-out protocol remain future work, untouched.

## Bottom line (candidate verdict — Dillan concludes)

No 0.947-shape circularity. Four rival explanations tested — parse selection, task mix,
isotonic-floor clipping, marginal-vs-conditional confusion — and none rescues the gate;
the within-QNLI replication and the matched-score margin split make the result *stronger*
and more precisely stated than before. Recommended: the methods note can lean on §4y,
using the conditional wording of §3 and citing this audit.
