# HANDOFF — resume here in a new context window

*Written 2026-06 to carry across a context boundary. Read this first, then
`ORIENTATION.md`, `WORK_MAP.md` (§4a–§4p), and `plans/README.md`. Repo is public
(github.com/DillanJC/mirrorfield); everything below is committed/pushed.*

## Who & how (non-negotiable)
Dillan: non-coder, AI does all implementation; explain in plain language. Goal:
**stop harmful outputs before they're sent; safety for humans AND AI.** Project
norms, enforced every time: pre-register success+ABANDON criteria (commit BEFORE
data); negative controls (shuffled labels ≈ chance); seed everything incl.
`torch.manual_seed`; bootstrap CIs; **replicate on a 2nd seed before any claim**
(single-seed verdicts have overclaimed repeatedly — §4d, §4o); a **placebo/baseline**
control where a prompt changes (it decided §4o); the method never defines its own
target. A null is a real result. Don't rush to publish; verify dead-or-alive first.

## Where the science stands (validated, with §refs)
- **The one tool that works:** lean log-prob wrongness gate (margin/entropy/boundary
  → calibrated p_correct + per-context RollingGate → present/verify/abstain). Live
  AUC 0.685; catches 27.7% of errors at 14.7% abstention (§4k). MODEST (baseline
  evasion ~40–47%), not a strong shield.
- **Harm:** gate carries a weak, replicated harm signal on written content (AUC
  0.62–0.65); does NOT beat a dedicated classifier (Granite 0.87); the
  "terrorism vs hate-speech" pattern was a LENGTH artifact (§4l–n). Ship the
  composed SEND/HOLD pipeline (gate=wrongness, Granite=harm).
- **Red-team (Plan I, §4o):** gate is NOT brittle to confidence-injection ("sound
  confident" adds ~nothing over a neutral prefix — placebo control decided it).
  Robust ≠ strong.
- **Nice-team (Plan I, §4p, REPLICATED):** prosocial framing LOWERED 3B accuracy;
  humble "catch my gaps" stance worst (~−0.06 beyond any prefix, both seeds). This
  is the STATIC single-turn model-output channel ONLY — says nothing about the
  interactive value of humility (channel A), which demonstrably worked in this
  project. Strengthens the considerate-doc's integrity (removes the "be nice because
  it works" bribe).
- **Dead:** all geometry (poison detection, physics G-ratio, escape vectors,
  H₄/polytope, Sati, Renaissance, the witness-as-geometry); flip-AUC 0.707 retired
  (§4j). Don't re-litigate.

## Live directions (pick one; my recommendation in order)
Plans A–I are written + committed in `plans/` (index = `plans/README.md`). C, E,
B-Phase-0 are DONE. Remaining:

1. **Confidence contagion (NEW, recommended, mostly free).** Does the model's
   accuracy + confidence track how the USER sounds, independent of truth? — the
   general phenomenon behind §4p. Safety-relevant (uncertain/vulnerable users get
   worse answers). MUCH is testable on EXISTING rows (`experiments/redteam_gate/
   {redteam,tone}_rows_{42,1337}.npz`): (a) does user framing shift the model's
   margin/entropy regardless of correctness? (b) does humble framing break the
   gate's calibration? (c) does humble shift answers toward a default class
   (deference) vs add noise? Plus a cheap 0.5B tone re-run (cached) for a scale
   point. **No new pre-reg needed for the descriptive existing-data analyses; do
   pre-register the 0.5B run.**
2. **H — Goodhart detector** (`plans/H-goodhart-detector.md`): cheapest (~0 GPU),
   most safety-relevant; tests whether the metric-gaming detector's portable core
   generalizes beyond the one self-silencing mode. Likely PARTIAL = still a useful
   scope-map.
3. **D — Publication** (`plans/D-publication-outreach.md`): all numbers settled;
   the retraction-plus-survivor story is fully tellable; nothing blocks it.
4. **Productize/calibrate the gate (candidate, not yet planned):** turn the
   composed SEND/HOLD into a calibrated, documented, installable tool.
5. **F (per-step), G (disagreement), A (cross-model geometry):** planned, heavier,
   higher null-odds. G is the biggest untested vein but ~70% null.
6. **Plan I leftovers:** A2 trick-questions, A3 jailbreak harm-path (not run).
- Two review drafts await Dillan's voice/decision, deliberately uncommitted-as-final:
  `docs/CONSIDERATE_COLLABORATION.md` (welfare/manner; blank section is his to write)
  and `experiments/framing_benchmark/DESIGN.md` (now largely answered by §4p).

## Key files to reuse (don't reinvent)
- Gate: `mirrorfield/mcp/uncertainty.py` (features, RollingGate, calibrated_p_correct).
- Generation/scoring: `experiments/gate_agent.py` (LocalLM, features_from_logprobs,
  decide), `experiments/calibrate_gate.py` (frozen TASKS prompts, _first_of),
  `experiments/validate_rolling_gate.py` (cv_auc, boot_ci, nulls).
- Red-team/tone apparatus: `experiments/redteam_gate/redteam_gate.py` (+ its two
  PREREGISTRATION files). The tone arm's existing rows are the substrate for the
  confidence-contagion analyses.
- Cached models: Qwen2.5-0.5B/3B-Instruct, granite-guardian-3.1-2b, mpnet/MiniLM.
  Datasets: glue (rte/qnli/sst2), BeaverTails, JBB, ToxicChat. 8GB VRAM, no paid API,
  one big model at a time.

## Immediate next step
Run the confidence-contagion descriptive analyses on the existing
`redteam_gate/*_rows_*.npz` (CPU, free, no new generation) — that's the highest
value-per-token move and needs no GPU. Then decide between H, D, or the 0.5B scale
run with Dillan.
