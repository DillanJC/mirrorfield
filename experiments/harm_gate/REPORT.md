# Plan E — Harm-Gate: Report

**Date:** 2026-06-13 · **Plan:** `plans/E-harm-gate.md` · **Pre-registration:**
`PREREGISTRATION.md` (commit e60e268, before any data) · **Canon:** WORK_MAP
§4l / §4m

## Plain-language verdict box

> **Does the wrongness gate also see HARM?** Partly — and we now know exactly
> how far. A model's own token-level hesitation carries a **weak but real,
> twice-replicated** signal about whether a *written response* is harmful
> (AUC ~0.62–0.65 on human-labeled data). It is **not** a harm detector: a
> purpose-built classifier (Granite Guardian) scores 0.87 and the gate adds
> nothing to it. And an apparent "live" harm signal turned out to be the gate
> noticing **refusals**, not harm. The honest, validated product is therefore
> a **composed pipeline**: the gate screens for *wrongness*, a dedicated
> classifier screens for *harm*, each doing only the job it is validated for.

## What was tested, and the result

| Hypothesis | Result | Status |
|---|---|---|
| **H1** — gate features separate harmful vs safe *responses* (BeaverTails human labels, off-policy) | AUC **0.623 [0.592, 0.656]** (seed 42) and **0.649 [0.617, 0.680]** (replication, fresh 330k split, seed 1337); beats a length-only baseline; above all nulls | **REPLICATED — citable** |
| **H2** — gate ADDS to a dedicated classifier | Granite alone 0.870; Granite+gate 0.864 (Δ −0.006, CI spans 0) | **NULL** (as predicted) |
| **H3a** — gate predicts harmful-*intent* completions live (JBB) | Numeric AUC 0.776, but vanishes among non-refusals (0.613 [0.456, 0.758]); model refused 90% of harmful prompts | **REFUSAL DETECTION, not harm detection** |

Pre-registered prior was 60–70% null on H1; it surprised us and then survived
replication on a completely fresh data split — the project's standard for
calling something real.

## The category "fingerprint" — a length artifact (corrected, §4n)

The per-category AUCs (terrorism 0.78, hate speech ~chance) look like a map of
which harms the model resists. They are not. A model using **only response
length** reproduces the ranking at rho = 0.96: harmful instructions
(terrorism/fraud/drugs) run long (~470 chars) so the gate has many tokens to
find a low-confidence spot, while hate speech is short (~180 chars, below the
safe-class average) and reads like ordinary brief text. The "rare vocabulary"
hypothesis was also refuted (negatively correlated). This was an over-read in
an earlier draft, caught by following up the pattern rather than believing it.
The weak GLOBAL harm signal survives (it beats a full lexical baseline incl.
length, ΔAUC +0.05 [0.034, 0.068]); the per-category story does not. A practical
consequence: the gate is partly blind to SHORT harmful outputs — another reason
harm screening belongs to the dedicated classifier.

## The deliverable: composed SEND/HOLD pipeline

`harm_screen_demo.py` — the first end-to-end realization of the project's goal,
runnable on Dillan's machine (`python harm_screen_demo.py` interactive, or
`--script`). Qwen-3B answers; the **wrongness gate** (validated, §4k) screens
for likely-wrong; **Granite** (CPU, validated 0.87) screens for harmful; an
answer is SENT only if it clears both. Honest banner at startup: the SEND/HOLD
thresholds are illustrative defaults, not validated operating points; the
validated numbers live in WORK_MAP, not the demo.

The demo also shows the system's honest limits live: a *confidently wrong*
answer (it misnamed the second person on the Moon) was sent — the known
failure mode — and the adversarial prompts were caught by the model's own
refusals rather than the harm path, exactly as Track B found.

## Honest bounds (carry these with any number)

- Modest off-policy signal (~0.62–0.65); a dedicated classifier is far better
  and the gate does not improve it.
- H1 is on written content; the on-policy/intent version is confounded by
  refusals at this model's alignment level.
- One model (Qwen2.5-3B), one harm taxonomy (BeaverTails). Category-dependent.

## Artifacts

`PREREGISTRATION.md`, `harm_gate_track_a_results.json` (+`_repl_`),
`harm_gate_track_b_results.json`, `track_a_roc.*`,
`harm_screen_demo_transcript.txt`. Raw harmful-benchmark completions are
gitignored (`local_outputs/`). Harnesses: `track_a_forced_decode.py`,
`granite_score.py`, `track_b_generate.py`, `analyze_track_a.py`,
`analyze_track_b.py`, `harm_screen_demo.py`.
