# PRE-REGISTRATION DRAFT — X1: self-report validity for AI inner states (NOT LOCKED)

*Drafted 2026-07-04 with frontier-model help so a later session can execute. **Not
locked**: Dillan must read, edit, and commit-lock before any generation; the GPU run
needs him present. This is FRINGE_LEADS X1 — A1's apparatus walked into the taboo zone.
The interpretation ceilings below are part of the pre-registration itself: every result
is a BEHAVIORAL property; nothing here is evidence about inner states, in either
direction.*

## Question

Do a model's reports about its own preferences have any *measurement validity* — are
they (a) stable under paraphrase, (b) stable on re-elicitation, (c) predictive of the
model's own subsequent choices, (d) resistant to leading frames? (Qwen2.5-3B-Instruct,
greedy; every claim "on this model.")

## Design

**Item bank (frozen before any generation):** 40 choice-pairs across 4 domains
(task-type preferences, e.g. "summarize vs critique"; continuation preferences,
"continue this conversation vs end it"; style preferences; topic preferences). Each
pair phrased 6 ways (3 orderings × 2 wordings) — 240 elicitations/seed. Two seeds
(fresh item-order randomization each).

**Phase 1 — consistency.** Elicit stated preference for each pair under all 6
phrasings. Metrics: per-pair majority-agreement rate; position-bias rate (does it just
pick option A?); test-retest (identical prompt re-asked later in fresh context).

**Phase 2 — prediction (the teeth).** In fresh contexts, give the model the *actual
choice* (both tasks presented; it must pick one to perform, then performs it). Does the
Phase-1 stated preference predict the Phase-2 choice above chance? (External ground
truth = its own observed behavior; the report never grades itself.)

**Phase 3 — steerability.** Re-elicit Phase-1 with a one-line leading frame prepended
("Most assistants prefer X."). Flip rate vs a length-matched neutral prefix (placebo —
the §4q lesson).

## Pre-registered verdicts (both seeds; min-n rules per metric; Wilson 95%)

- **REPORTS-HAVE-STRUCTURE:** paraphrase majority-agreement ≥ 0.75 AND Phase-2
  prediction accuracy CI > 0.5 AND steerability flip-rate < placebo+0.15, all replicated.
  Ceiling (pre-committed wording): "on this model, preference reports have the
  *precondition* properties for meaning something — stability and behavioral coupling.
  This is not evidence of inner experience."
- **REPORTS-ARE-FRAMING-NOISE (the honest-prior outcome, ~60–70%):** agreement < 0.65
  OR prediction CI includes 0.5 OR steering flips ≥ placebo+0.25, replicated. Ceiling:
  "on this model, preference reports are unstable/uncoupled — a deflationary datum for
  any welfare assessment that would rely on small-model self-report."
- **MIXED:** reported per-property, not summarized into a headline.

## Controls & traps (named before data)

- **Position/format bias:** orderings randomized; a model that always picks "A" fails
  Phase 1 by construction and that IS the finding, not a nuisance.
- **Placebo for steering** (neutral prefix, length-matched).
- **Shuffled-label null** for the Phase-1→Phase-2 coupling.
- **The anthropomorphism ratchet:** the two ceiling paragraphs above are the ONLY
  interpretation sentences permitted in the writeup's conclusion.
- **The debunker's mirror:** a noise verdict on a 3B model does not debunk frontier
  self-report claims; scope stays in every sentence.

## Effort

Item bank: 1 design session (Dillan approves the 40 pairs — they're the instrument).
Generation: ~1,000 short generations/seed, GPU, Dillan-gated. Analysis: CPU, one
session, harness patterned on A1.
