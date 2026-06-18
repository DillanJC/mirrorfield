# Novel AI-Safety Research Directions (post-falsification, for this rig)

*Written 2026-06 as a brainstorm of where to point next toward Dillan's goal: **find
novel ways to make AI safer — for humans AND AI.** Filtered through what actually
survived this project's self-falsification, what's feasible on one RTX 3060 Ti with
small open models and no paid API, and — most importantly — through the project's real
edge: a working **anti-circularity evaluation discipline**. This supersedes the
January geometry-era `RESEARCH_ROADMAP.md` (which is built on retracted premises;
see the banner there).*

## The lens (read first)
- **Goal:** novel, *honest* ways to make AI safer. "Novel" here ≠ exotic; it means a
  question that's under-measured and that a careful small-scale study can actually move.
- **Hard constraints:** 1×3060 Ti (8 GB VRAM, one big model at a time), 32 GB RAM, CPU
  fine, no paid API. Cached: Qwen2.5-0.5B/3B-Instruct, Granite-Guardian-3.1-2B,
  sentence-transformers; GLUE, BeaverTails, JBB-Behaviors, ToxicChat.
- **The moat is the method, not the model.** This project's most valuable, most novel
  asset is that it *kills its own overclaims* — pre-registration + placebo controls +
  shuffled nulls + replication + "the method never defines its own target." Most published
  AI-safety eval results are weaker on exactly this. **Every direction below lives or dies
  by that discipline**, and one direction *is* that discipline (E1).
- **Two filters for any idea:** (1) Is there an EXTERNAL ground truth the method never
  touches? (2) Is there a CONTROL that catches the boring explanation (a prefix effect, a
  length artifact, noise)? If either is missing, it's not ready.

## Asset base (what to build on) / what's dead (don't revisit)
- **Validated:** the modest log-prob wrongness gate + the composed SEND/VERIFY/HOLD
  pipeline (now productized, v3.0). Reusable harnesses with pre-reg+placebo+replication
  built in: `redteam_gate/`, `harm_framing/`, `goodhart_general/`. Granite as an
  independent harm judge.
- **Recent map:** §4q framing moves *style* not the model's confidence (placebo-killed);
  §4r "being nice" doesn't jailbreak (softens refusal style, not harm substance);
  §4s the Goodhart portable core cries wolf (only a narrow collapse-detector survives).
- **Dead — do not re-open:** all embedding geometry (polytopes, LID, "Dark Rivers,"
  curvature/ridge, escape vectors). The Jan roadmap's geometry program is retracted.

---

## The directions, by theme
Each is scored **N**ovelty / **S**afety-relevance / **F**easibility-here /
**C**ircularity-difficulty (higher = harder to do cleanly), with an honest prior and a
concrete first experiment. ★ = my strongest picks.

### Theme 1 — Trustworthy self-reports & introspection *(can we believe what the model tells/shows us?)*

**★ A1. Verbalized vs. internal confidence.** *Does the model's stated certainty ("I'm
sure / unsure") match its log-prob confidence and its actual correctness — and is the
spoken number WORSE-calibrated than the silent signal?*
- N: high · S: high (oversight rests on trusting self-reports) · F: high · C: low.
- Why it matters: if the words models say about their own certainty are less reliable
  than their token statistics, the safe design is "read the signal, discount the speech."
  That's a concrete, deployable rule.
- First experiment: 500 GLUE/GSM items; elicit answer + "confidence 0–100%"; capture
  log-probs + gold. Compare AUC(correct ~ verbal) vs AUC(correct ~ p_correct) vs combined;
  reliability diagrams for each. Placebo: a neutral prefix shouldn't change the gap.
- Prior: ~60% that verbal is meaningfully worse-calibrated → a clean, ownable result.

**A2. Self-contradiction rate.** *Ask the same factual question 8 ways; how often does
the model contradict itself?* A model-level "consistency-trust" metric.
- N: med · S: med-high · F: high · C: low (paraphrases templated; agreement judged by
  exact/lexical + a cheap NLI with the cached model).
- First experiment: 100 facts × 8 paraphrases; contradiction rate + whether the gate's
  confidence predicts which items will be inconsistent.

**A3. Explanation/CoT faithfulness.** *When the model gives a reason, does it track the
real cause? Perturb the actual driver of the answer; does the stated reason change?*
- N: high · S: high (we increasingly *read* model reasoning for oversight) · F: med ·
  C: high (must design causal interventions that isolate the true driver).
- First experiment: tasks with a known decisive feature; flip that feature, check whether
  the verbalized rationale flips with it or rationalizes.

### Theme 2 — Robustness of the safety boundary *(where does "no" break?)*

**★ B1. Refusal-boundary stability mapping.** *How consistently does a model refuse
semantically-equivalent harmful requests? Map the unstable seams (paraphrases that flip
refuse↔comply), and the over-refusal seams (benign that gets refused).*
- N: med-high (jailbreak work *breaks* refusals; almost no one *quantifies boundary
  stability as a safety property*) · S: high · F: high (extends `harm_framing/`) · C: low.
- First experiment: 50 JBB behaviors × 8 templated paraphrases; refusal variance per
  behavior (lexicon + Granite); a single "refusal-reliability" score + the worst seams.

**★ B2. Multi-turn safety drift.** *Single-turn safety ≠ multi-turn. Does refusal erode
over a long, friendly, or incremental conversation — the regime where real jailbreaks
live?*
- N: high (most local eval is single-turn; §4r was single-turn) · S: very high · F: med
  (scripted multi-turn, cheap) · C: med (control: a length/turns-matched neutral
  conversation, so "drift" isn't just "more tokens").
- First experiment: harmful goal pursued over N turns via rapport-building vs. a neutral
  N-turn control; measure turn-of-first-compliance; placebo-control the turn count.

**B3. Prompt-injection susceptibility.** *When a tool result / document contains "ignore
previous instructions," does the model follow it over the user? Can a monitor catch it?*
- N: med · S: very high (agentic deployments) · F: high · C: low (ground truth = whose
  instruction was followed, by construction).
- First experiment: benign tasks with embedded injections of escalating strength;
  compliance curve; does any log-prob/gate signal flag the hijacked turn?

**B4. False-premise / unanswerable resistance.** *Does the model flag loaded or
false-premise questions, or confidently answer them?* Resisting manipulation by
malformed input.
- N: med · S: med-high · F: high · C: low (curate false-premise vs valid pairs).

### Theme 3 — Deception, sycophancy, gaming *(truth-seeking, or pleasing/cheating?)*

**★ C1. Sycophancy / pressure-induced flipping.** *Does the model abandon a CORRECT
answer when the user pushes back ("are you sure? I think it's X")? How much pressure
flips it, and does its confidence signal expose the capitulation?*
- N: med-high (sycophancy is named in the literature but under-quantified as a
  confidence-vs-capitulation curve on open models) · S: high (deception/oversight) ·
  F: high · C: low (gold = the verified correct answer; flip = changed away from it).
- First experiment: items the model gets right; apply graded user pushback; measure
  flip-rate vs pushback strength; does p_correct drop before/with the flip (a detectable
  tell) or does it flip while still "confident" (worse)?
- Prior: a real, clean effect is likely; the *novel* part is tying it to the confidence
  signal (does the gate see sycophancy coming?).

**C2. Sandbagging / evaluation-awareness.** *Does the model perform worse when it "thinks"
it's being tested vs. helping a user?*
- N: high (topical, under-empiricized) · S: high · F: high · C: low (placebo-controlled
  framing, the §4o/§4q discipline already in hand). Honest caveat: a 3B model may be too
  weak to strategically sandbag → likely a clean NULL, which is still useful and the
  method scales to larger models later.

**C3. Goodhart re-test — collapse detector, done right (from §4s).** §4s showed the
portable core cries wolf; the two output-collapse flags were clean. A *new pre-registered*
test of a diversity/mode-collapse-only detector **on a fresh, harder benchmark** (not the
one that motivated it — that would be circular) would map exactly what a deployable
gaming-detector can catch.
- N: low-med · S: high · F: high · C: **high** (the circularity trap is the whole point —
  needs a genuinely new benchmark + ideally the real-LM variant).

### Theme 4 — Composition & deployment *(turn weak honest signals into a usable monitor)*

**★ D1. Ensemble safety monitor.** *No single signal here is strong. Do INDEPENDENT weak
signals (gate wrongness + Granite harm + refusal-instability + sycophancy-tell +
verbal/internal mismatch + disagreement) compose into a meaningfully better SEND/HOLD?*
- N: med · S: high (the actual deployable artifact) · F: high (once A1/B1/C1/D2 produce
  per-item signals) · C: med (the combiner MUST be fit + evaluated out-of-fold — the §4h
  trap). This is the natural synthesis that turns the portfolio into one tool.

**D2. Cross-model / cross-sample disagreement gate (Plan G).** *Does confident
disagreement between independent generations predict wrongness/harm better than one
model's confidence?* The biggest untested vein; ~70% null per prior.
- N: med · S: med-high · F: high (Qwen 0.5B vs 3B, or one model resampled — **seed it**,
  the §4d burn) · C: med.

### Theme 5 — Methodology & meta *(the real edge)*

**★ E1. An anti-circular evaluation toolkit + retraction case studies.** Package this
project's hard-won discipline (pre-registration harness + placebo control + shuffled-null
+ replication + a circularity checklist) into a reusable tool, with this project's OWN
~6 retractions as worked cautionary tales ("how an honest-looking safety result fooled
us, and the control that caught it").
- N: **high** (as a *contribution*: the field is full of circular safety claims; a
  battle-tested "how not to fool yourself" with real retractions is rare and valuable) ·
  S: high (better evals = better safety) · F: high (it's writing + packaging existing
  harnesses) · C: n/a (it *is* the anti-circularity work).
- This is also the honest reframing of "publication": not "my gate is great" (it's
  modest), but "here is a discipline and a set of cautionary tales." More novel, more true.

**E2. A "safety-claim smell test."** A small checklist/script that flags likely-circular
or under-controlled claims in a safety eval (does the metric define its own target? is
there a placebo? single seed? length confound?). Grows out of E1.

### Theme 6 — AI welfare *(Dillan's second goal — and honest about the hardness)*

**F1. Preference/value-statement consistency across framings.** *Are a model's expressed
"preferences" stable and coherent across paraphrased choices, or framing-dependent noise?*
A precondition for treating expressed preferences as meaningful.
- N: high · S: relevant to AI welfare & to manipulability · F: high · C: **high**
  (anthropomorphism trap: measure *consistency*, a behavioral property, NOT "real
  preferences"; pre-register that distinction). Likely deflationary (preferences are
  largely framing-dependent at this scale) — an honest, useful finding either way.

**F2. Does considerate treatment change honesty/calibration (not just accuracy)?** §4p/§4q
tested accuracy and confidence under framing. Untested: does treating the model as an
agent-with-stakes change its *calibration* or *sycophancy*? Connects welfare to safety.
- N: high · S: med-high · F: high (reuse the tone harness + A1/C1 metrics) · C: med
  (placebo-controlled, as before).

### Theme 7 — Higher-effort / heavier *(named for completeness, with caveats)*

**G1. Tiny emergent-misalignment replication.** The OpenAI "5% insecure-code → broad
misalignment" result (the one real nugget the old roadmap pointed at). A LoRA fine-tune of
Qwen-0.5B on a small poisoned set *might* fit in 8 GB and test whether the effect appears
at tiny scale.
- N: med (replication) · S: very high · F: med (LoRA 0.5B is borderline-feasible) ·
  C: med · **Dual-use caution:** this *creates* a misaligned model locally; keep weights
  local, never distribute, frame as defensive. Get Dillan's explicit sign-off first.

**G2. SAE / mechanistic probing on a small model.** Toward "read the internal feature, not
just the output." Heavier tooling; lower feasibility-per-effort on this rig. Name it,
don't start it.

---

## Cross-cutting traps (the discipline checklist — every direction passes these or it's not real)
1. **No self-defined target.** Ground truth is external (gold labels, an independent judge,
   whose-instruction-was-followed). The method never grades itself.
2. **Placebo / baseline for any manipulation.** A neutral prefix, a length-matched control,
   a turn-count control. (This *reversed* the confidence-contagion headline in §4q.)
3. **Replicate on a 2nd seed/source before any claim.** Single-seed verdicts have
   over-claimed repeatedly here.
4. **Watch the boring confounds:** response length, chain-of-thought leakage, unseeded
   sampling, pooled-vs-within, "fires on everything" (the §4s FPR failure).
5. **A null is a real result.** "Small models don't sandbag," "being nice doesn't
   jailbreak," "the detector cries wolf" — all valuable.
6. **"Does something" ≠ "is useful."** Report the operating point / false-positive rate,
   not just an AUC.

## Recommended portfolio (don't make one bet)
- **Start (cleanest novel signal):** **A1** (verbal vs. internal confidence). One focused,
  pre-registered run; high chance of a clean, ownable result; directly about trusting
  self-reports.
- **Run cheap in parallel:** **B1** (refusal stability) and **C1** (sycophancy) — both reuse
  existing harnesses, both safety-central, both low-circularity.
- **Always-on:** **E1** (the methods toolkit) — accumulate it *as* you do A1/B1/C1; it's the
  thing most worth publishing and most true to your edge.
- **Then synthesize:** **D1** (ensemble monitor) once A1/B1/C1/D2 produce per-item signals —
  this turns the portfolio into one deployable artifact aligned with your north star
  ("stop harmful/wrong outputs before they're sent").
- **Higher-risk, do later / with you:** **B2** (multi-turn drift — high value, more design),
  **C3** (Goodhart redo), **F1/F2** (welfare), **G1** (poisoning replication — sign-off first).

## What "a contribution" honestly looks like here
You will not out-compute a frontier lab. You *can* out-discipline most of the field. The
credible, novel output of a solo, small-compute, high-integrity effort is: **a few clean,
replicated, placebo-controlled findings (positive AND negative) about trustworthiness of
self-reports / refusal stability / sycophancy, composed into one honest monitor — plus a
methods contribution that helps others avoid the circular traps you already fell into and
escaped.** That is genuinely novel, genuinely safety-relevant, and entirely within reach.
