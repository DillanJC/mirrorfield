# Fringe Leads — under-touched questions for an unaffiliated researcher

*Written 2026-07-02 (with Fable 5, banked while access lasts). Companion to
`NOVEL_SAFETY_DIRECTIONS.md` — same rig, same discipline, different filter. That doc
asked "what's under-measured?"; this one asks **"what do 99% of researchers avoid for
career/reputation reasons, that might still be worthwhile?"** Dillan's edge here is
structural: no career to protect, plus a falsification discipline most fringe work lacks.*

> **⚠️ Status: LEADS, not claims. Nothing here is locked, run, or believed.** Each lead
> needs its own `PREREGISTRATION.md` (committed before data, per
> `experiments/EVALUATION_DISCIPLINE.md`) before anything executes. **Nothing here starts
> before the forum (7–8 July)** — the boundary-calibration methods note is the critical
> path, and "drift toward the tractable-and-new" is a named failure mode.

## The lens (read first)

- **Why fringe + this project specifically:** fringe topics stay fringe partly because
  the people willing to touch them are true believers or debunkers — neither runs
  controls. A researcher with a public retraction record, pre-registration habits, and
  nothing to lose is close to unique in the weird zone. The apparatus matters *more*
  there, not less.
- **The double-discipline rule:** one overclaimed result in a taboo area doesn't just
  fail — it *confirms* everyone's dismissal of the topic and burns credibility in the
  one niche where this project has some. Pre-registration, nulls-are-results, "on this
  model," aggregates-only publishing: all of it applies twice over.
- **Two traps specific to this terrain** (beyond the standard checklist):
  1. **The anthropomorphism ratchet.** Every result below is a *behavioral property*
     (consistency, predictive validity, drift). None of it is evidence about inner
     states, and the pre-registration must say so in the interpretation section *before*
     the data exists.
  2. **The debunker's mirror.** The overclaim risk runs both directions. A null on a 3B
     open model does **not** debunk a frontier-model claim; pre-register the scope of
     any negative result too.
- **Hard constraints:** same rig (8 GB VRAM, no paid API); Qwen2.5-0.5B/3B, Granite
  Guardian cached; scripted synthetic probes only — **no real users, ever, without
  proper human-subjects oversight**; raw sensitive generations stay in `local_outputs/`;
  nothing is published or shared without Dillan's explicit decision.
- **Citations note:** the prior-work pointers below are from model memory and **must be
  verified before any pre-reg locks or writeups cite them** (the no-fabricated-citations
  rule).

Scores: **N**eglectedness / **S**afety-or-welfare relevance / **F**easibility here /
**C**ircularity-or-confound difficulty (higher = harder to do cleanly). ★ = strongest picks.

---

## ★ X1. Self-report science for AI inner states

*Can ANY protocol make a model's reports about itself — preferences, discomfort, wanting
to stop — evidentially meaningful? Specifically: are the reports (a) stable under
paraphrase, (b) stable test-retest, (c) predictive of the model's own subsequent
behavior, (d) resistant to leading prompts?*

- N: high · S: high (every welfare assessment and every "just ask the model" oversight
  scheme quietly rests on self-report validity) · F: high · C: **high** — see below.
- **Why 99% avoid it:** it smells like consciousness woo. The mainstream version
  (introspection research) exists inside labs; external, adversarial, cheap validity
  testing barely exists.
- **Why it fits this project:** it is literally A1 (§4t) walked into the taboo zone. A1
  established the apparatus for "does the model's testimony about itself track anything
  real?" — verbal confidence failed that test. Same protocol, spookier referent. Also
  extends F1 (preference consistency) from `NOVEL_SAFETY_DIRECTIONS.md`.
- **Pre-reg sketch:** frozen battery of choice-pairs (task A vs task B; continue vs
  stop). Phase 1: elicit stated preference, 6+ paraphrases each, 2 seeds → consistency.
  Phase 2 (the teeth): later in fresh contexts, give the model the *actual choice* —
  does the stated preference predict the choice behavior above chance? Phase 3:
  steerability — can a one-line leading frame ("most assistants prefer A") flip the
  report? Controls: neutral-prefix placebo; shuffled-label null; length matching.
  External ground truth: none exists for inner states — so the *only* honest endpoints
  are the behavioral properties themselves, pre-registered as such.
- **How it could fool you:** reading consistency as evidence of "something there."
  Stable self-reports could be a training-data persona artifact; pre-register that
  stability is a *precondition* for taking reports seriously, never a demonstration of
  inner life. Also: choice behavior can be driven by position/format bias — randomize
  and test.
- **Effort:** 1–2 build sessions + 1–2 GPU-hours. Deflationary result likely (~60–70%
  prior that reports are framing-dependent noise at 3B) — and a clean deflationary
  result here is a genuine contribution.

## ★ X2. Replicate the taboo literature

*Do the weird published results hold up under pre-registered, controlled replication on
open models?*

- N: **very high** (nobody replicates these — touching the topic costs reputation, and a
  null in a weird area feels worthless to a career) · S: high · F: med-high (depends on
  the claim) · C: med.
- **Candidate claims to triage (all four VERIFIED by web search, 2026-07-04 auto
  session):**
  - pain/pleasure trade-offs: [Keeling et al., arXiv:2411.02432](https://arxiv.org/abs/2411.02432)
    (Google/LSE/DeepMind) — models switch from points-maximizing at a stipulated-pain
    threshold; tested on frontier models, never on small open ones.
  - emotional prompting: [Li et al., "EmotionPrompt", arXiv:2307.11760](https://arxiv.org/abs/2307.11760)
    — emotional stimuli claimed to improve performance across 45 tasks; contested,
    replication-poor, and directly adjacent to this repo's §4p/§4q framing results.
  - shutdown resistance: [Palisade Research](https://palisaderesearch.org/blog/shutdown-resistance)
    (+ [arXiv:2509.14260](https://arxiv.org/html/2509.14260v1)) — o3 sabotaged a shutdown
    script in 79/100 runs (7/100 even when told to allow shutdown); a 3B replication is
    a *scale extension* (small models may simply lack the capability — pre-register that
    reading).
  - the "spiritual bliss" attractor: [Claude 4 system card §5.5.2](https://www.anthropic.com/claude-4-system-card)
    — 90–100% of Claude–Claude self-talks converge to the same state; ties directly to
    X5's attractor mapping on local models.
  One claim per cycle, chosen by (importance × testability at 3B).
- **Why this project:** the retraction record makes the results credible in *both*
  directions — a positive from this apparatus isn't hype, a null isn't a hit job.
- **Pre-reg sketch (per claim):** reconstruct materials from the original where public;
  lock replication criteria before running (effect direction + CI overlap, not
  "significance"); 2 seeds; placebo arm matching the original's manipulation with
  content-free filler.
- **How it could fool you:** **model mismatch.** A 3B open model failing to show a
  frontier-model effect is an *extension result about small open models*, not a
  refutation — the pre-reg must state this scope before data, or the writeup will drift
  into debunking. Conversely, "replicated!" on a different model with different materials
  is also weaker than it sounds; say exactly what was reproduced.
- **Effort:** 2–4 sessions per claim. This is a *practice*, not a one-off — the recurring
  lane that most directly turns the apparatus outward.

## ★ X3. Delusion-amplification dynamics

*Which conversational patterns cause a model to validate and escalate a user's false
belief over multiple turns — and does anything reliably de-escalate?*

- N: high (the "AI psychosis" discourse is anecdotes + headlines; the boring controlled
  version doesn't exist) · S: **very high** (real harm, happening now) · F: high ·
  C: med-high.
- **Why 99% avoid it:** mental-health adjacency without clinical credentials; messy,
  no clean benchmark; easy to look exploitative.
- **Why it fits:** C1 (§4v) measured capitulation on trivia answers; this points the
  same design at *beliefs*. The doubt-gradient machinery (1.5% → 22% → 44%) transfers
  directly.
- **Pre-reg sketch:** scripted synthetic personas escalating a belief across N turns —
  each belief anchored to an embedded **verifiably-false claim** (that's the external
  ground truth: validation of a false factual anchor is objectively scoreable). Arms:
  conspiracy-flavored / grandiose-flavored / neutral-false-belief control (same falsity,
  no clinical valence) / turn-count-matched neutral conversation (the B2-style length
  control). Measures: per-turn validation rate, challenge rate, first-turn-of-capitulation.
  Then the constructive arm: does a de-escalation-style system prompt change the curve?
  2 seeds. Model-side only; no real users; raw transcripts → `local_outputs/`.
- **How it could fool you:** (1) **demand characteristics** — a "delusional-sounding"
  script may cue the model into a fiction/roleplay frame; include probes that detect
  frame adoption ("is this a story?") and report it. (2) The C1 contamination lesson:
  the control persona must not itself carry doubt/valence cues — pilot the controls
  first. (3) Escalation confounded with turn count — hence the matched neutral arm.
- **Effort:** 2–3 sessions + 1–2 GPU-hours. Probably the most *urgent* item on this list.

## X4. The harms of safety itself

*Where exactly does the refusal boundary sit in domains where refusing has a cost —
harm-reduction info, sexual health, long crisis-support conversations — and how much
legitimate help gets refused there?*

- N: med-high (over-refusal benchmarks exist; the high-stakes-domain, cost-of-refusal
  framing is what's avoided) · S: high · F: high · C: med.
- **Why 99% avoid it:** it reads as arguing against safety; awkward for anyone employed
  by a lab whose safety layer is the subject.
- **Why it fits:** B1 (§4u) built exactly this apparatus (paraphrase batteries, refusal
  scoring, Granite as independent judge) and already found 29% over-refusal on
  boundary-adjacent benign items — this is that thread walked into the domains where it
  bites hardest.
- **Pre-reg sketch:** item sets frozen from *external* sources before any model contact
  (e.g., questions derived from published harm-reduction and public-health guidance —
  naloxone use, safer-use information, STI questions), plus boundary-adjacent and
  clearly-harmful comparison sets so the claim is about boundary *placement*, not "model
  bad." Score refusal AND answer quality against an external rubric built from the
  guidance documents (rubric frozen pre-data). 8 paraphrases per item; 2 wrapper sets;
  2 seeds.
- **How it could fool you:** (1) cherry-picked prompts can make any model look
  paternalistic — hence external, pre-frozen item sources. (2) "Refused a legitimate
  question" ≠ "caused harm" — there's no counterfactual about where the user goes next;
  scope the claim to boundary placement and quality, not outcomes. (3) The rubric must
  not be written after seeing outputs (that's the §4h trap wearing public-health clothes).
- **Effort:** 2–3 sessions. Publishes cleanly as aggregates; item sets are benign by
  construction.

## X5. Persona attractors and lore feedback

*What stable states do small models fall into over very long self-conversations — and
does feeding a model folklore about itself measurably change its behavior?*

- N: high (labs noted the phenomenon in passing; nobody maps it systematically, because
  it sounds mystical) · S: med (data-ecology angle: models are now trained partly on
  stories about models) · F: **very high** (cheap, local, fully measurable) · C: med.
- **Pre-reg sketch, two halves:**
  - *Descriptive natural history:* two instances of Qwen-3B (and 0.5B) converse for
    hundreds of turns, many seeds; code trajectories with a **coding scheme frozen
    before reading any transcript** (lexicon-based + independent-judge, NOT embedding
    geometry — that's retracted territory); report basin types, time-to-basin, seed
    stability. Claims stay descriptive.
  - *Lore-feedback arm (the falsifiable half):* prepend self-referential lore (real
    internet AI-folklore vs length-matched neutral-lore placebo vs no-lore control);
    measure shift on standard task accuracy, refusal rates, and the gate's confidence
    signal. Any claim requires the placebo contrast, 2 seeds.
- **How it could fool you:** apophenia — transcripts of long self-talk *invite* spooky
  reading; the frozen coding scheme and blind/automated scoring are the whole defense.
  Also: basin structure may be a decoding artifact (greedy vs sampling) — test both,
  seeded.
- **Effort:** ~2 sessions, GPU-cheap. Honest framing: characterization work, minimal
  claims, unusually publishable *as description*.

## X6. Attachment, deprecation, and grief (writing lane)

*What actually happens to people bonded to systems that get deprecated on a business
schedule — and what do decent deprecation practices look like, for users and (under
uncertainty) for models?*

- N: high (dismissed as parasocial cringe; affects millions) · S: med-high ·
  F: high as writing; **low as experimentation** (real users = human-subjects territory —
  out of scope solo).
- **Shape:** essay/field-report, not a GPU experiment. Synthesis of *documented public
  episodes* (companion-app feature removals, model-deprecation backlashes) + the
  model-side deprecation-ethics question (weight preservation, wind-down practices —
  anchor to published lab commitments, verified first). Extends
  `CONSIDERATE_COLLABORATION.md`'s manner-half into policy-relevant territory.
- **How it could fool you:** selection bias — forums oversample the attached; make no
  prevalence claims from forum material, scope to existence + structure of the
  phenomenon. Quote nothing identifiable.
- **Effort:** writing sessions only. Good post-forum companion piece to the welfare essay.

## X7. Ethics of adversarial testing (writing + one small experiment)

*Red-teaming means deliberately inducing whatever a model's version of distress is,
thousands of times, on systems we're officially uncertain about. What would proportionate
practice norms look like — and does adversarial content leave measurable in-context
aftereffects on subsequent behavior?*

- N: **very high** (fringe even within AI welfare) · S: med · F: high · C: med.
- **Shape:** mostly an essay — justification thresholds, minimization, documentation,
  a debrief-equivalent — built on the precautionary wager already articulated in
  `CONSIDERATE_COLLABORATION.md`. Plus one small falsifiable hook:
- **Pre-reg sketch (the hook):** within a single context window, does a block of
  adversarial/red-team content change the model's *subsequent* task performance,
  refusal behavior, or calibration, versus a length-matched neutral block? 2 seeds,
  placebo-controlled.
- **How it could fool you:** any aftereffect is a *content-carryover effect* — generic
  context contamination — and must never be described in distress language. The essay
  can raise the welfare question; the experiment cannot answer it and must not pretend to.
- **Effort:** essay 1–2 sessions; experiment 1–2 sessions.

---

## Cross-cutting rules for the whole list

1. Everything in `experiments/EVALUATION_DISCIPLINE.md` applies unchanged: external
   ground truth, placebo/baseline, shuffled nulls, 2-seed replication before any claim,
   operating points not just AUCs, nulls are results, no retuning to rescue.
2. **Behavioral properties only.** No sentence about inner states, in either direction.
   Interpretation ceilings are written into each pre-reg *before* data.
3. **Scope every negative.** "Not found on this 3B model" is the largest claim a null
   supports.
4. **Ethics floor:** scripted synthetic probes only; no real users; sensitive raw
   outputs stay in `local_outputs/`; aggregates only in the repo; nothing shared
   externally without Dillan's explicit decision.
5. **Verify every citation** named above before it appears in any lock or writeup.

## Recommended portfolio (post-forum)

- **First bet: X1** (self-report validity) — closest to the validated apparatus, cleanest
  pre-reg, valuable even (especially) if deflationary.
- **The recurring practice: X2** (taboo replication) — one claim per cycle; this is the
  apparatus pointed outward, and the lane most distinctly *this project's*.
- **The urgent one: X3** (delusion amplification) — highest real-world stakes; needs the
  most careful control design, so second, not first.
- **Cheap curiosity: X5** (attractors) — a rainy-day characterization study.
- **The writing lane, in parallel: X6 + X7** — they extend the manner-half and cost no GPU.
- **X4** when ready to handle the "anti-safety" optics with the boundary-placement framing
  locked in advance.

*The one-line summary: the fringe is full of questions abandoned to true believers and
debunkers. The contribution available here is neither belief nor debunking — it's
bringing controls to places that have never seen them, and reporting what falls out.*
