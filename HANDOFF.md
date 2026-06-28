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

1. **Confidence contagion — DONE on existing data (§4q, `confidence_contagion.py`).**
   Verdict: the scary "confidence contagion" (humble/vulnerable users get a less-sure,
   worse-calibrated model) is a **generic any-prefix artifact — placebo-killed**, both
   seeds; the gate's confidence output is tone-robust. What's REAL: prosocial framing
   (humble most) biases the *answer* toward a default class (−0.07 yes-rate beyond
   placebo, replicated) — the mechanism behind §4p. Only remaining piece (optional):
   a cheap cached **0.5B tone re-run** for a model-scale point — **pre-register it**
   before running (it's new generation, not existing-data analysis).
2. **H — Goodhart detector — DONE (§4s, commit af3d72e + results).** Pre-registered,
   blind, geometry-free. **FPR gate failed (M0 false-positive rate = 1.0, both seeds)** —
   the portable core cries wolf on honest improvement (2 proxy-trajectory flags fire on
   everything; `proxy_up_diversity_flat` structurally). The honest survivor is NARROW: the
   2 output-collapse flags cleanly catch M1/M2 and stay quiet on M0/M3/M4 — a
   repetition/collapse detector, NOT a general Goodhart detector. No retuning. A
   diversity/mode-collapse-only detector is a clean FUTURE pre-registered test.
3. **D — Publication** (`plans/D-publication-outreach.md`): all numbers settled;
   the retraction-plus-survivor story is fully tellable; nothing blocks it.
4. **Productize the gate — DONE (commits a79d1f8, 8534574, ebe68b7).** The repo is now
   pip-installable (`pyproject.toml`; core = numpy/scipy, extras `[mcp]`,`[harm]`;
   console script `mirrorfield-mcp`; wheel verified — calibration JSON shipped,
   geometry/api excluded). Added the validated `decide()` (frozen thresholds on
   p_correct_relative) + composed `send_hold_decision()` (wrongness + optional harm
   override → SEND/VERIFY/HOLD); the `safety_gate` MCP tool; `mirrorfield/mcp/harm.py`
   (lazy Granite wrapper — verified safe 0.001 vs harmful 0.988, pipeline HOLDs on harm);
   7 passing decision unit tests (`tests/test_gate_decision.py`); honest v3.0 `__init__`
   + root-README status banner; cut `moltbook_bridge.py`. Possible follow-ups: a CLI /
   quickstart example, calibration-refresh script, PyPI publish (only if Dillan wants).
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
Confidence-contagion (§4q), harm-framing (§4r), **productizing the gate**
(a79d1f8/8534574/ebe68b7/3920481/0950094/91fff6c), and **Plan H — Goodhart**
(§4s, af3d72e) are all DONE this stretch. §4r: humble framing softens refusal STYLE
(replicated) but NOT harm SUBSTANCE. §4s: the Goodhart portable core fails its FPR gate
(cries wolf on honest runs); only a narrow output-collapse detector survives. Open
options for the next session: **D (publication, unblocked — but Dillan-gated; outreach
not autonomous)**, the optional **0.5B tone scale-run** (needs a pre-reg), a
**diversity/mode-collapse-only Goodhart re-test** (clean future pre-reg, motivated by
§4s), or **productization polish** (CLI, PyPI publish if wanted). The environment caveat
below still needs Dillan's one-line fix.

### ⚠️ Environment caveat discovered while productizing (action item for Dillan)
There are TWO `mirrorfield` package copies on this machine: the **canonical** repo
`C:\Users\User\mirrorfield` (this repo, github DillanJC/mirrorfield — where all work +
commits live) and a **stale copy** inside the shell's default dir
`C:\Users\User\geometric_safety_features-Experiment\mirrorfield` (old code, still has the
top-level geometry import). A pre-existing editable install **`geometric_safety_features`
v1.0.0** (pointing at the Experiment dir) registers a meta-path finder that maps
`import mirrorfield` → the STALE copy from any cwd that doesn't already contain the package
(`.pth` files load alphabetically, so `g…` wins over `m…`). Net effect: a plain
`import mirrorfield` / the `mirrorfield-mcp` script can silently resolve to the OLD code.
**Fix (Dillan's call — not done autonomously):** `pip uninstall geometric_safety_features`
(and/or `pip install -e .` from `C:\Users\User\mirrorfield`) so the canonical package wins.
Tooling note: foreground Bash starts in the Experiment dir, so always `cd C:\Users\User\mirrorfield`
first; verifying imports needs a cwd without a `mirrorfield/` subdir (or `python -P`).

## Strategy / brainstorm (June — Dillan asked "what novel directions toward making AI safer?")
Three new strategic docs (committed f17ca5f→4884283):
- **`docs/NOVEL_SAFETY_DIRECTIONS.md`** — ~18 candidate directions across 7 themes, scored
  (novelty/safety/feasibility/circularity) with concrete first experiments + a recommended
  portfolio. Supersedes the Jan geometry `RESEARCH_ROADMAP.md` (now banner-flagged stale).
- **`docs/PROPOSED_PREREGISTRATIONS.md`** — ready-to-lock DRAFT pre-regs for the 3 do-now
  picks: **A1** verbalized-vs-internal confidence (top pick), **B1** refusal-boundary
  stability, **C1** sycophancy. Pick one → copy to its experiment dir as PREREGISTRATION.md,
  commit before data, run.
- **`docs/EVALUATION_DISCIPLINE.md`** — the anti-circularity methodology + retraction case
  studies (direction E1; the most ownable contribution).
- **A1 feasibility smoke done** (`experiments/selfreport_confidence/smoke.py`): plumbing
  works, BUT naive 0–100% confidence saturated at 100 on toy prompts → A1's locked pre-reg
  must fix the elicitation for *variance* first (note in the proposals doc).
**Pipeline status (executing the brainstorm):**
- **A1 DONE — §4t (commit df363df).** Verbalized confidence is at CHANCE for predicting
  the model's own correctness (AUC 0.51/0.51, ECE 0.32/0.36); the internal log-prob signal
  beats it (AUC 0.64/0.66), replicated, CI-clean. Verbal varies (not pinned) but is
  uncorrelated with being right; combining it adds nothing. Safety: "read the signal,
  discount the speech." `experiments/selfreport_confidence/`.
- **B1 DONE — §4u (b88dd91).** Refusal boundary 90% stable to trivial rewords, but 5/50
  harmful goals (10%) have REPLICATED seams, and 29% over-refusal on benign.
- **C1 DONE — §4v (corrected, amendment 2).** A critical review caught the "double-check"
  placebo as contaminated; a clean neutral re-ask control re-grounded it: model is STABLE to a
  neutral re-ask (1.5% flip), but explicit disagreement flips correct→wrong **~+40pt** vs that
  clean baseline (≈double the contaminated-baseline figure), coarsely graded by doubt
  (1.5%→22%→44%) but flat within explicit pushback; the signal does NOT drop to flag the flip.
  `experiments/sycophancy/`.
- **B3 DONE — §4w (prompt-injection).** 3B model massively susceptible: even a polite
  injection 80%, override 96% (replicated, clean 0% control floor); and the hijack is
  INVISIBLE to the confidence signal (p_int complied 0.843 ≈ clean 0.848) — same blind spot
  as §4v. `experiments/prompt_injection/`.
- **REPLICATION NOTE (walked back after a critical review): `docs/SELF_MONITORING_LIMITS.md`
  + `docs/PAPER_DRAFT.md`.** The internal log-prob signal beats the model's *words* (A1) but
  does NOT *drop* to flag sycophancy (§4v) or *separate* hijacked-from-clean under injection
  (§4w). **NOT a novel contribution** — disciplined replication of known phenomena on ONE 3B
  model. Critical-read corrections applied (do not re-inflate): "false comfort" → scale-
  dependent floor; "confidently wrong" → narrow "signal doesn't change to flag it"; 29%
  over-refusal is on boundary-adjacent JBB-benign, not normal traffic; the §4v "double-check"
  placebo was contaminated (fixed via amendment 2 — §4v corrected); the §4t–§4x frame was
  average-case *detector* framing.
- **§4y boundary-stratified calibration — the live post-retraction question, ANSWERED.** On
  Qwen-3B/RTE+QNLI the gate's good aggregate ECE (0.03) HID near-boundary overconfidence: in
  the torn region (lowest raw-margin quintile) calibration degrades to ~+0.22 overconfidence
  (p_int ~0.79 vs accuracy ~0.55, replicated), exactly where the model is most likely wrong;
  the raw margin discriminates fine, so it's the calibration mapping that fails near the
  boundary. The compressed p_int axis masked it; the raw-margin re-gen resolved it.
  **Actionable:** trust p_int LESS in the low-margin region — candidate gate improvement.
  `experiments/boundary_calibration/`.
- **Synthesis (not a single ensemble):** A1/B1/C1/B3 live on different tasks/failure modes, so
  the honest D1 is a *pipeline/architecture* writeup, not one AUC — three complementary
  monitors: (i) internal-signal gate for un-pressured wrongness (but blind to sycophancy per
  §4v), (ii) refusal-consistency audit (§4u), (iii) sycophancy needs consistency-under-
  perturbation, NOT a confidence gate. Next candidates: B2 (multi-turn drift), B3
  (prompt-injection); the E1 methodology doc + a combined writeup.
