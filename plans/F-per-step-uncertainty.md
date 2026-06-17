# Plan F: Per-Reasoning-Step Uncertainty — does *where* the model doubts itself predict *which* answers are wrong?

> **Orchestrator adversarial review (applied 2026-06-13).** Architect plan is
> strong: it already controls the length/step-count confound (the §4n killer)
> four ways, uses paired bootstrap so "min-of-steps vs whole-mean" isn't a
> sample-size artifact, pre-registers the aggregation, and labels against GSM8K
> gold (non-circular). **One hardening added** (see *Pre-registered criteria →
> Task-suitability gate*): a wrong result is uninterpretable if the *whole-response
> gate itself* is at chance on GSM8K (a math/free-generation task, unlike the
> RTE/QNLI/SST-2 the gate was validated on). So Step 4 must FIRST confirm the
> incumbent mean-gate clears 0.5 on GSM8K; if it doesn't, the verdict is "the gate
> doesn't transfer to math generation" — a clean finding, not a confounded null.

## In plain language

We already built and verified a "doubt meter": a cheap gadget that watches a language model write an answer and, from the model's own moment-to-moment hesitation, predicts whether the whole answer is likely wrong. It works modestly — on ~900 fresh questions it caught 27.7% of errors while holding back only 14.7% of answers (a coin-flip at the same hold-rate catches 14.7%). But it currently reads the *whole answer at once*: it boils a response down to one average hesitation number.

This plan asks a sharper question. When a model reasons step by step — "first this, then that, therefore 42" — the mistake usually lives in *one bad step* while the rest looks calm. Averaging washes that step out. So: cut each chain of reasoning into steps, run the exact same validated doubt meter on *each step*, find the single least-confident step, and test honestly: (1) **localization** — is the shakiest step the one that actually went wrong? (2) **better prediction** — does "the worst step" predict a wrong final answer *better* than the whole-answer average?

We use grade-school math (GSM8K) because every problem has one correct number to check against an answer key the meter never sees, so we can't cheat by grading ourselves.

One trap, stated up front, that has killed results here before: longer reasoning has more steps, and more steps means more chances for one to look shaky *by luck* — so "the worst step" is mechanically lower just because an answer is long, and wrong answers are often longer. If we don't control for that, we'll see a "discovery" that is really just answer length in costume. The whole plan is built so that explanation is measured head-on, and so that an honest **"no, it doesn't beat the simple average"** is a clean, finished result — which we say up front is the more likely outcome.

## Why this matters / what it builds on

**Builds on the one validated tool:** the lean log-prob gate (mean token margin, mean entropy, boundary ratio), shipped in `mirrorfield/mcp/uncertainty.py`. Live AUC 0.685 [0.642, 0.730] on fresh RTE+QNLI at Qwen2.5-3B; abstains 14.7%, catches 27.7% of errors vs 14.7% random (WORK_MAP §4k). Geometry adds ~0 (§4b); self-consistency rejected (§4d/§4e).

**The question:** the gate averages those features over a whole response. This computes the identical features **per reasoning step**, then asks whether the **worst step** (a) co-locates with the actual error step and/or (b) predicts final wrongness with **higher AUC than the whole-response mean**. This is primitive **A** from `SALVAGE_AUDIT.md`, at finer granularity.

**The honest, geometry-free heir of a dead idea.** The old "reasoning telemetry / equalizer" work tracked *participation ratio across steps* on a single trace, paid API, embedding geometry, no ground-truth label, a self-defined "novelty score" — textbook self-confirming. This keeps only the *shape* (look per step) and replaces every dead part: many traces, local Qwen, **log-prob features not geometry**, GSM8K gold the method never touches.

**Honest prior: WEAK-to-MODERATE; most likely a controlled null (~65–75%).** Error-localization is a live area, but the length/step-count confound has a *mechanical* path to a fake positive (more steps → lower min by chance), and that confound family already forced a retraction here (§4n). The plausible real mechanism: a single wrong arithmetic step is a genuine low-margin fork the mean dilutes but the min preserves. Plausible ≠ probable.

## What already exists

- `mirrorfield/mcp/uncertainty.py` — `compute_token_margins/entropies/boundary_ratio` (reused as-is), `find_uncertain_spans` (for the localization visual).
- `experiments/gate_agent.py` — `features_from_logprobs` (the per-segment feature call, with the exact 4dp live-path rounding), `LocalLM` (seeded greedy Qwen2.5-3B, `output_scores=True`, top-5 logprobs).
- `experiments/calibrate_gate.py` — the generation harness pattern; already sets `torch.manual_seed`.
- `experiments/harm_gate/track_a_forced_decode.py` — the closest ancestor: per-position top-5 logprobs over a multi-token response → `features_from_logprobs`, sliced by step boundaries; plus the npz discipline (features/labels/lengths, never raw text).
- `experiments/validate_rolling_gate.py` — `cv_auc`, `boot_ci` (2000-rep), shuffled-null idioms.
- `experiments/equalizer_test/reasoning_trace_logger.py` — has `parse_reasoning_steps` (a cascade); NOT reused as-is (a per-trace-varying rule is itself a confound) but documents the segmentation problem.
- Cached: `Qwen2.5-3B-Instruct`. **Not cached:** GSM8K (`openai/gsm8k`, config `main`, MIT, ungated, <5 MB; gold = integer after `####`).
- New work in `experiments/step_uncertainty/`.

## The plan, step by step

VRAM rule: only Qwen2.5-3B in VRAM; all analysis CPU from saved `.npz`.

**Step 0 — Pre-registration committed to git BEFORE any data (CPU, ~15 min).** `experiments/step_uncertainty/PREREGISTRATION.md`: the criteria below verbatim + seed (42), replication seed (1337), sample sizes, **the exact segmentation rule**, **the frozen aggregation set {min, mean, last, count} with headline = min**, primary hypothesis, analysis recipe. Commit; hash is the timestamp.
- **Segmentation rule (frozen, one rule):** a "step" = a newline-delimited line of the CoT containing ≥1 alphanumeric char, after stripping markdown bullet/number prefixes (`^\s*(\d+[.)]|[-*])\s*`); lines with <3 tokens merge into the next. Robustness variants (reported, not headline): sentence-split; fixed-20-token chunks (holds granularity constant).
- **Prompt (frozen):** a fixed 2-shot CoT prompt whose examples end in `#### <number>`.

**Step 1 — Download + smoke (GPU, ~30 min).** Download GSM8K. `generate_cot.py` (clone of `calibrate_gate.generate`/`LocalLM`): Qwen2.5-3B greedy, `output_scores=True`, `max_new_tokens=320`, all seeds set. Per problem: generate CoT, parse final integer after `####`, label `wrong=(pred!=gold)`, segment by the frozen rule, compute the 3 gate features per step's token slice. **Smoke N=30**; verify parse rate, sane step counts, finite features. Time-box: if final-answer parse rate <80%, fix prompt/parser now before scaling.

**Step 2 — Full generation (GPU, ~2–3 h).** N=1,000 GSM8K test, seed 42, greedy (deterministic). Expected wrong-rate ~25–45% → ~250–450 positives. Save per problem: `ds_index`, `wrong`, `n_steps`, `resp_len`, whole-response mean features `X_mean` (the incumbent), ragged per-step features (`step_feats` + `step_ptr`), per-step `step_token_len`. No raw text committed.

**Step 3 — Localization labels (CPU/GPU, ~1–1.5 h; wrong cases only).** For wrong problems, find the *first wrong step* via a deterministic arithmetic checker over the model's own stated numbers (gate-independent). Record **assignable fraction**; if <40%, H_loc → INCONCLUSIVE. Artifact: `localization_labels.npz`.

**Step 4 — Analysis (CPU, minutes).** `analyze_steps.py` (imports `cv_auc`/`boot_ci`/null loop). Per-step features → confidence (higher margin / lower entropy / lower boundary = more confident). Aggregations: min, mean, last, step-count.
- **Task-suitability gate (added by review — run FIRST):** AUC of the whole-response mean gate on GSM8K. If its CI includes 0.50, STOP and report "the validated gate does not transfer to math generation at 3B" — do not interpret any min-vs-mean delta (you can't beat a baseline that's itself at chance). This is itself a useful transfer finding.
- **H_pred (primary):** paired ΔAUC(min − mean), same problems resampled in both arms (2000-rep bootstrap). Plus the four length controls (below).
- **H_loc (secondary):** top-1 hit rate (argmin step == error step) vs the random-step baseline mean(1/n_steps), with CI; within-±1; rank test. Step-shuffled null.
- Qualitative: 6 wrong traces rendered with `find_uncertain_spans`, shakiest step vs true error step.
- Artifacts: `step_uncertainty_results.json`, `step_auc.png`, `localization_examples.png`.

**Step 5 — Replication + report (GPU ~2–3 h, then CPU).** Re-run on seed 1337 + a fresh disjoint 1,000-problem sample; a positive must land inside the primary CI and clear its controls. Optional stretch (only if headline positive): a 300-problem second multi-step task. `REPORT.md` with plain-language verdict.

## Pre-registered success / failure criteria

Seed 42; replication 1337. AUC = 5-fold OOF logistic; CI = 2000-rep bootstrap 95%; null = 10 shuffled-label refits (mean ∈ [0.45,0.55], observed > max). **Baseline to beat = the whole-response-mean gate AND a step-count-only model — never a trivial 0.5.**

- **Task-suitability gate (added by review, precedes everything):** whole-response mean-gate AUC on GSM8K must have CI lower bound > 0.50. If not → verdict "gate does not transfer to math generation"; H_pred/H_loc not evaluated (uninterpretable). This prevents a confound-laden null masquerading as "per-step doesn't help."
- **H_pred SUCCESS:** paired ΔAUC(min − mean) CI lower > 0 **AND** length-adjusted ΔAUC of (min + n_steps) over (n_steps alone) CI lower > 0 **AND** effect persists (point estimate positive) in the step-count-matched IQR stratum **AND** replicates on seed 1337 + fresh sample. All four required to headline.
- **H_pred WEAK-REAL:** min-step AUC CI lower > 0.50 and ≥ mean, but ΔAUC over mean spans 0 → "valid alternative aggregation, no demonstrated advantage."
- **H_pred NULL (expected):** ΔAUC includes 0, OR advantage vanishes once n_steps is included / in the matched stratum, OR fails to replicate → "per-step minimum does not improve on the whole-response mean."
- **H_loc SUCCESS:** top-1 hit-rate CI lower > random baseline AND > max of 10 step-shuffled nulls AND replicates. NULL: CI includes baseline. INCONCLUSIVE: assignable fraction < 40%.
- **ABANDON if** H_pred = NULL and H_loc ∈ {NULL, INCONCLUSIVE}. No post-hoc aggregation swaps; headline aggregation = min, frozen Step 0.

## Controls & verification

- **Circularity — target the method defines: NONE.** Label = GSM8K `####` gold (published, gate never reads it); localization label = model's own stated arithmetic checked deterministically. Min-vs-mean = two aggregations of the same features vs the same external label.
- **Aggregation pre-committed** (blocks the §4d "try every aggregation, report the luckiest" trap).
- **Length/step-count confound — controlled four ways:** step-count-only baseline AUC; headline requires min to beat n_steps-as-covariate; within-trace length-residualized step confidences; step-count-matched stratum + fixed-token-chunk variant.
- **"More samples" confound** (min sees N steps, mean sees 1): paired bootstrap on the same traces.
- **Segmentation not cherry-picked:** one frozen rule; two robustness variants; a headline holding under only one segmentation is downgraded to "segmentation-dependent."
- **Seeds:** torch + numpy + random (the §4e lesson). **Replication** before headlining (the §4d lesson).
- **Honest null prior stated:** H_pred ~65–75% null; H_loc ~60% null-or-inconclusive.

## Honest risks

- Length/step-count is the likeliest false-positive source (mechanical path; §4n family) — four controls exist to catch it.
- Confidently-wrong reasoning is invisible to an uncertainty signal by construction (§4c) — if wrong steps aren't low-margin, both hypotheses null.
- Localization ground truth is partial (arithmetic checker can't assign every error step) → H_loc may be inconclusive.
- **New (review):** the gate may simply not transfer to math/free-generation at 3B — the task-suitability gate catches this and turns it into a finding.
- Most likely time-sink: CoT parsing / the arithmetic checker — mitigated by the Step-1 time-box and H_loc being droppable.

## Deliverable Dillan will see

1. `step_auc.png` — AUC bars (worst-step, whole-answer-average, step-count-only, length-only) + the length-adjusted delta, with "must beat" lines.
2. `localization_examples.png` — real wrong solutions with the shakiest step highlighted beside the actual error step.
3. `REPORT.md` — plain-language verdict per hypothesis; the number we promised to beat; whether it replicated. Clean whether yes or (more likely) no.

## Effort

5 sessions; ~5–8 GPU-hours total (generation ×2 for replication; analysis CPU); GSM8K only (~5 MB) download; $0 API.
