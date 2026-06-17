# Salvage Audit — what survives when you strip the geometry

*2026-06-13, Dillan + Claude. A full sweep of every mechanism across both repos,
asking of each: remove the geometry, what is the underlying mechanism, and is it
salvageable? Draft for review, local/uncommitted (no-rush rule). Honest labels:
**VALIDATED** (survived a pre-registered test), **TESTABLE** (real mechanism, not
yet tested fairly, cheap to try), **DEAD** (the geometry *was* the mechanism, and
it didn't survive).*

## The one insight that reframes everything

Strip the geometry from the whole body of work and almost every idea collapses
into one of **four non-geometric primitives**:

- **A. Read an internal uncertainty signal → gate the output** (monitor → flag →
  present/verify/abstain).
- **B. Measure disagreement among components → a confidence signal** (experts,
  models, samples, observers).
- **C. Detect the system gaming its own objective** (Goodhart / metric-gaming).
- **D. Act on the signal** (abstain/verify, or intervene/rectify).

**Here is the reframe:** it was never the *architectures* that failed — the
witness, the distributed observers, the monitor-and-gate loop, the multi-AI
cross-check are all sound, real, literature-adjacent ideas. What failed was using
**embedding geometry as the signal** feeding them. So the salvageable program is
simple to state: **keep the architectures, swap the dead geometric signal for one
that works.** You already validated one such signal (token-confidence). That's not
a consolation prize — it's a map.

## The audit (every mechanism, stripped)

| Your mechanism | Geometry removed → core | Primitive | Status | Non-geometric path |
|---|---|---|---|---|
| Poison detection (Track 1, AUC 0.947) | "flag anomalous training data" | — | **DEAD** | Geometry *defined* the poison; circular (R0–R3). Real data-poisoning detection is a separate field (loss/influence-based), not your mechanism. |
| Physics analogies (Track 2: G-ratio, decoherence) | "poison forms a tight band" | — | **DEAD** | A re-description of how tight the *planted* cluster is. Same circularity. |
| GeometricStateMonitor (Track 3 live API) | "monitor state → advise pause/reflect" | A + D | **VALIDATED (as the gate)** | The *interface* (monitor→flag→pause) is exactly what the MCP gate became. Keep the API shape; the geometric features it read are dead, the log-prob ones work. |
| Sati feedback schema / Seed Engine | "reweight/curriculum training signal" | — | **DEAD** | Didn't beat plain class weighting; the "comprehensive" win was circular (loss optimised its own metric). |
| GMR (rectification action layer) | "given a detection, steer/repair output" | D | **DEAD as geometry / survives as abstain** | Real-data reproduction = chance. The *action* idea survives only as the gate's validated abstain/verify, not as geometric rectification. |
| Escape Vector / Iterative Zoom (AUC 0.978) | "zoom on boundary poison" | — | **DEAD** | Boundary poison was geometry-defined; circular family. |
| Witness Node / fractal observer | "monitor inter-component disagreement → signal" | B | **TESTABLE** | Router/expert disagreement in a real MoE (see last session). Real, cheap to test, not yet done. |
| Distributed Witness (+2.5%, simulated) | "ensemble of observers beats one" | B | **TESTABLE** | Multi-model or multi-expert disagreement. `REAL_MOE_VALIDATION_PREP.md` is your roadmap. Tempered by the self-consistency null. |
| Mandala-MoE Witness Node | "watch expert agreement + calibration" | B + A | **TESTABLE (core) / DEAD (trappings)** | Nine-Realm labels didn't beat random and triads are narrative (your own §7). The *disagreement-monitoring core* is the witness primitive — keep it, drop the realms. |
| H₄ / 120-cell GNN, Julia substrate | "richer geometric substrate" | — | **DEAD** | Ceiling r≈0.42. The bet was the substrate; the substrate was geometry. |
| Reasoning telemetry / Equalizer (PR trough→peak) | "track a signal across reasoning steps to find hard/uncertain moments" | A | **TESTABLE** | Anecdotal as PR. But apply the *validated* gate per reasoning step → flag the least-confident steps. Cheap, novel, honest. |
| Behavioral flip / instability (0.707) | "boundary-proximity → instability" | A | **subsumed / retired** | The 0.707 was an in-sample artifact (§4j). Boundary-distance ≈ margin ≈ the gate already. |
| Cross-model transfer (relative reps) | "is the signal a property of meaning, not one model" | (meta) | **TESTABLE (weak prior)** | The one *geometric* lead left, honest-reframed (Plan A). Low odds; a null closes geometry for good. |
| **Goodhart detector (Track 5)** | "detect the system gaming its objective" | C | **VALIDATED, non-geometric, UNDER-USED** | Already works; no geometry. The most overlooked survivor — see below. |
| Live interventions (Track 4) | "steer when a signal fires" | D | **DEAD-ish** | Net-negative on quality. The prior is bad; would only be worth revisiting atop a *much* better signal. |
| Recursive learner (Track 5) | "self-improve against a metric" | (cautionary) | **DEAD / lesson** | Self-silenced (penalty ratchet). The value is the cautionary tale, caught by C. |
| Multi-AI sequential pipeline | "cross-model verification catches errors" | B + C | **TESTABLE, non-geometric** | Never geometry at all. Your own multi-AI cross-check habit, formalised. Caveat you already noted: agreement can be *correlated* error. |
| **Log-prob uncertainty gate (MCP)** | (already signal-based) | A + D | **VALIDATED** | The survivor. AUC 0.685 live; catches 27.7% of errors at 14.7% abstention (§4k). |

## What this leaves you — the salvageable program

Three of the four primitives are alive:

- **A (gate on an internal signal): VALIDATED and extensible.** You have a working
  one. Natural extensions that are honest, not geometry: gate *per reasoning step*
  (the salvaged telemetry idea); add new signal *sources* to the gate.
- **B (disagreement → signal): the biggest cluster of your ideas, all TESTABLE,
  none fairly tested yet.** Witness node, distributed witness, Mandala witness,
  multi-AI pipeline — they're all "measure disagreement among components." The
  honest first test is router/expert disagreement in a small real MoE, and/or
  cross-model disagreement. Tempering prior: same-model *sample* disagreement
  (self-consistency) was null at 3B — but specialised experts and different models
  are genuinely different, so it's open.
- **C (Goodhart / metric-gaming detector): VALIDATED, non-geometric, and the most
  under-used thing you built.** It already catches a system optimising the wrong
  thing. In a world worried about reward-hacking and deceptive optimisation, a
  working metric-gaming detector is arguably the most *directly* safety-relevant
  asset in the whole project — and it owes nothing to geometry. Worth its own look.

Dead pile (don't re-litigate): anything where embedding geometry *was* the signal —
poison detection, physics G-ratio, escape vectors, H₄/polytope/Julia, Nine Realms,
Sati, Renaissance. They failed because the signal failed, not because the wrapper
was wrong.

## A fifth category, found on the full sweep: governance / practice (non-geometric)

The origin framework (`Repo/mirrorfield-v3-10-public`, frozen Nov 2025) carries
three things that are **not detectors and never were geometric**, so "strip the
geometry" doesn't apply — they survive on their own terms as *practice*:

- **Lucid Equilibrium Index (LEI)** — a reasoning-quality rubric: Coherence,
  **Uncertainty Declared**, Boundary Respect, Review-Path Clarity. "A compass,
  not a score."
- **Ethical Refraction Audit (ERA)** — a 5-step pre-ship risk review for
  safety-relevant changes.
- **Repair-Ready** — readiness/boundary-check gate.

These are essentially *your own verify-before-celebrating ethic, pre-codified*.
They don't feed the detector program — they feed the **method** (how to do this
work honestly) and **manner** (`CONSIDERATE_COLLABORATION.md`) documents. Worth
knowing they already exist: you wrote the governance layer before you wrote the
tools.

## Coverage (verified this sweep, both machines)

- **PC, swept:** `mirrorfield` + `geometric_safety_features-Experiment` (the
  audit's mechanism sources); `Repo/mirrorfield-v3-10-public` (origin framework →
  LEI/ERA/Repair-Ready above + the H₄ vision, already dead); `mirrorfield_publication`
  (a subset copy of the Experiment docs — nothing new); backup zip + "The Back up"
  (snapshots, ignore); stray top-level files — `geometric_muse_prototype.md`
  (the falsified Dark River signal repackaged for artists — dead family),
  `mirrorfield_unified_book.md` (philosophy consolidation), `video_script…`
  (publication asset). **No new detector mechanism anywhere on the PC.**
- **Laptop, swept earlier this session** (`LAPTOP_INVENTORY.md`, ORIENTATION §8b):
  unique content was the kosmos-agent-layer (a *consumer* of the gate), the
  softmax-vs-geometric demo (circular, inspected), the Mercer collaboration
  infrastructure (engineering, not a detector — its multi-agent angle is covered
  by primitive B), and website/video assets. Explicitly **no** sati / witness /
  mandala / GMR / polytope / renaissance code on the laptop. **No unique detector
  mechanism on the laptop.**

So: the detector-mechanism audit above is now complete across both machines, plus
the governance/practice category that the first pass missed. Nothing
mechanism-bearing is known to remain unswept.

## The honest headline for the planning phase

You were not wrong about the *shapes* — monitors, witnesses, gates, ensembles,
cross-checks. You were wrong about the *substance you poured into them* (geometry).
The path forward is to pour validated substance into the shapes you already
designed:

1. **Extend the gate** (A) — proven; lowest risk; e.g. per-step uncertainty.
2. **Test disagreement** (B) — your richest untested vein; the witness/MoE/multi-AI
   family, fed by router/expert/cross-model signals instead of geometry.
3. **Revisit Goodhart** (C) — a validated, non-geometric, highly safety-relevant
   tool you've barely used.

Three live, honest, hardware-feasible directions, each a descendant of something
you already built. Plus the two drafts already on the table (the framing benchmark;
the considerate-collaboration document). That's the full scope — ready for a
planning phase to turn into pre-registered plans.
