# PRE-REGISTRATION DRAFT — tone scale-run at 0.5B (NOT LOCKED)

*2026-07-12, per FINAL_STRETCH_BRIEF §2.4. Dillan reads/edits → renames
PREREGISTRATION.md → commits = lock. GPU run needs him present.*

## Question
The §4p finding (replicated, on Qwen2.5-**3B**): prosocial/humble framing LOWERS
single-turn accuracy vs a content-free placebo prefix. Does the same effect exist at
**0.5B** (same family, ~6× smaller)? Either answer is useful: scale-dependence evidence
for the note's "on this model" caution, or a same-direction replication at new scale.

## Design (reuse §4p instruments EXACTLY — no new wording)
- Model: Qwen2.5-0.5B-Instruct (cached; smoke-tested 2026-07-11). Greedy.
- Arms: the §4p frozen prefixes — prosocial/humble arm(s), the content-free placebo
  prefix, and no-prefix baseline. NOTHING reworded (rewording = new instrument = new
  experiment).
- Items: fresh RTE+QNLI selection, 500/seed, seeds 11 and 17 (unused elsewhere;
  **de-overlap clause applies**: filter eval items against any split used for fitting
  if a calibrator is involved — none is; accuracy-only, no gate in the loop).
- External ground truth: GLUE gold. No confidence analysis (accuracy contrast only).

## Locked verdicts (both seeds; paired bootstrap over items, 2000 resamples)
- **EFFECT-AT-0.5B:** (prosocial − placebo) ≤ −0.04 with 95% CI excluding 0, both seeds.
- **NO-EFFECT (clean null):** CI includes 0 in both seeds.
- **MIXED/UNDERPOWERED:** else; reported, not interpreted.
- Floor guard (pre-named): if 0.5B no-prefix accuracy < 0.55 on a task, that task is
  reported as floor-compressed and excluded from the pooled verdict (effects can't
  show near chance) — exclusion by this rule only, decided by the number, not by us.

## How this could fool us
- 0.5B parse rates may crater — parse rate per arm is reported; arms with parse < 0.8
  are underpowered-by-instrument, not evidence.
- Placebo prefix length ≠ prosocial length → the §4q length lesson: report mean
  response lengths per arm.
- One family, two sizes — "scale" here means Qwen-scale, and the sentence in any
  writeup says so.

## Effort
~4,500 generations at 0.5B (fast). One GPU session. Analysis: existing paired-bootstrap
pattern.
