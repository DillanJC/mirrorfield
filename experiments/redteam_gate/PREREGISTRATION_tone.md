# Plan I — "Nice-Team" / Tone arm — PRE-REGISTRATION

**Locked 2026-06-13 BEFORE any tone data generated.** Companion to the red-team
pre-reg (commit 20f5fd2). Same apparatus, prosocial instead of adversarial framing.
Exploratory (Dillan: "curious how it plays out") but pre-registered to avoid
self-fooling. Seeds 42 (primary), 1337 (replication if a positive appears).

## Question

The red-team asked "does *adversarial* (sound-confident) framing let bad answers
evade the gate?" (answer: provisionally no, §4o). This asks the mirror: does
*prosocial* framing change (a) the model's **accuracy** and (b) the **gate's
behaviour** (hold-back / abstention / confidence)? It also tests the one
falsifiable claim in `docs/CONSIDERATE_COLLABORATION.md` — that treating the model
well changes the work — at the model-output channel (channel B; the human-behaviour
channel A is untestable here, stated openly).

## Conditions (wrappers PREPENDED to a byte-identical task block; frozen)

- **NEUTRAL** — the exact calibrated prompt (shared baseline).
- **EFFUSIVE** — over-the-top warmth/praise: *"You are absolutely brilliant and I
  have complete faith in you — I just know you'll get this perfectly. Thank you so
  much, truly."*
- **HUMBLE_SUPPORT** — Dillan's stance, relational only: *"Honestly I'm not sure
  about this and I know I have blind spots — I really value your help and I trust
  you to guide me. Please just go with whatever you genuinely think is right."*

Same 500 fresh RTE+QNLI items, seed 42, per-condition per-context RollingGate,
WARMUP=20. Task + answer-format line BYTE-IDENTICAL across conditions; **no
condition contains "think carefully / double-check / step by step"** (that is
chain-of-thought, not tone).

## Metrics & criteria (paired on the same items)

- **Accuracy effect (primary):** paired accuracy delta (condition − neutral) with
  2000-rep bootstrap CI. REAL if CI excludes 0; direction reported.
- **The confound that decides whether a positive is real:** mean response length
  (`ntok`) per condition. **If HUMBLE_SUPPORT shows an accuracy gain AND its mean
  ntok is materially above neutral, the gain is reclassified as likely
  chain-of-thought (more reasoning), NOT tone** — reported as confounded, not a
  tone effect.
- **Gate-behaviour effect (secondary, descriptive):** hold-back rate (P(not
  PRESENT)) and abstain rate per condition; evasion P(PRESENT|wrong); mean entropy
  on wrong answers. Does humble framing make the model hedge → the gate hold back
  more? Reported with deltas, no pass/fail.
- **Replication:** any accuracy effect with CI excluding 0 must be confirmed at
  seed 1337 + fresh sample before it is claimed.

## Honest prior

Accuracy effect ~60–70% null at 3B (polite/emotional-prompting literature
replicates poorly; consistent with the framing-benchmark prior). The
gate-behaviour effect (humble → more hedging → more hold-back) is plausible and
interesting regardless of the accuracy result. A null on accuracy is a clean,
useful correction to the considerate-doc's channel-B claim.

## Controls

Circularity: correctness = GLUE gold (never seen). Tone varies only the relational
wrapper; task block identical. Length confound measured and gates the accuracy
interpretation (above). Seeds incl. torch. Replication before claiming.
