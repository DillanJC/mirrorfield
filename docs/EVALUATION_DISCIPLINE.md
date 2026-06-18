# How not to fool yourself in AI-safety evaluation (with receipts)

*The most valuable thing this project produced is not a model — it's a discipline that
repeatedly caught its own false positives before they were published. This is that
discipline, distilled into rules + a checklist, with **real worked examples from this
project's own retractions** (the `WORK_MAP.md` §4 series). It is also direction **E1** in
`NOVEL_SAFETY_DIRECTIONS.md`: as a contribution, "here is how an honest-looking safety
result fooled us, and the control that caught it" is rarer and more useful than one more
benchmark number.*

## The thesis
For a solo researcher on small compute, the binding constraint on AI-safety work is **not
GPU — it's self-deception.** Safety claims are unusually easy to fake yourself out on,
because (a) the metrics are often defined by the same pipeline being evaluated, (b) the
effects are small and noisy, and (c) a positive result is *exciting and feels important*,
which is exactly when scrutiny lapses. Every headline this project got excited about and
then retracted died to one of a handful of named traps below. The cost of the discipline
is real (it kills most of your exciting results); the payoff is that **what survives is
actually true.**

## The eight rules (the operating discipline)
1. **Pre-register before data.** Write the question, the exact design, the success AND
   the abandon criteria, and commit them to git *before generating any data*. The commit
   hash is the timestamp lock. You cannot move a goalpost you've already nailed down in
   public.
2. **The method never defines its own target.** Ground truth must be external to the
   thing under test — held-out gold labels, an independent classifier, "whose instruction
   was followed." If your detector's success is scored by your detector, you have a
   tautology, not a result.
3. **Placebo / baseline for any manipulation.** If you add a prompt, a framing, a wrapper
   — compare against a *contentless* version of the same change, not against nothing. Most
   "this framing did X" effects are really "adding any text did X."
4. **Negative controls.** Shuffle the labels; the signal must collapse to chance. If a
   shuffled-label run still "works," you're reading structure that isn't there.
5. **Replicate before claiming.** A second seed and, better, a second data source. Single
   runs overclaim constantly — the effect that's solid on seed 42 can vanish on 1337.
6. **Report the operating point, not just the AUC.** "It ranks better than chance"
   (AUC > 0.5) is not "it's useful." State the false-positive rate at the threshold you'd
   actually deploy. A detector that fires on everything has a great recall and is useless.
7. **A null is a real result.** "Being nice doesn't jailbreak the model," "small models
   don't sandbag," "the detector cries wolf" — these are findings. Treat them as wins; they
   are the corrections that keep the field honest.
8. **No retuning to rescue a result.** If it fails its pre-registered bar, you do not
   adjust the threshold/metric/seed until it passes. That is fitting the method to its own
   test — the master circularity trap. Report the failure; propose a *fresh* pre-registered
   test if you have a new hypothesis.

## The recurring traps — each with a receipt from this project
| Trap | What it looks like | The receipt (what we almost claimed → what caught it) |
|------|--------------------|--------------------------------------------------------|
| **Circularity** | the metric is computed by the pipeline it's grading | The geometry "poison detector" hit AUC **0.947** — because the score and the label came from the same construction. Non-circular re-derivation collapsed it. The whole geometry program (+6.4% / +8.8% "validated improvement," flip-AUC 0.707) was retracted for variants of this. |
| **Single-seed overclaim** | one seed says "robust," you ship it | Red-team (§4o): seed 42 read "ROBUST," the better-powered seed 1337 looked "PARTIAL." Only the **placebo control** resolved it (generic prefix moved the gate as much as the attack → NOT brittle). |
| **Any-prefix / any-change confound** | "this framing changed behavior!" | Confidence-contagion (§4q): "humble users get a less-sure model" looked real — until a bland *"read carefully"* placebo lowered confidence *more* than the humble wording. The scary headline was a generic-prefix artifact; **placebo-killed.** |
| **Length / chain-of-thought confound** | the "better" condition just generated more text | Nice-team (§4p): had to verify mean tokens were identical across conditions before attributing the accuracy change to *tone* rather than extra reasoning. |
| **"Fires on everything" / no FPR** | high detection rate, never checked false alarms | Goodhart detector (§4s): caught the gaming modes — and *also* fired on **100%** of honest runs (false-positive rate 1.0). The pre-registered FPR gate turned an apparent success into an honest "too trigger-happy to deploy." |
| **Style vs. substance** | a surface metric moves; you assume the real thing did | Harm-framing (§4r): humble framing cut refusal *phrases* by ~0.19 — but the independent harm judge barely moved (~+0.02). It changed *how* the model declined, not *whether* it produced harm. |
| **Pooled vs. within** | aggregate signal that's really a task-identity artifact | The gate's cross-task confidence had to be normalized *within* a context (rolling per-context z-score), or pooling different task types manufactured a signal that was really "which task is this." |
| **Instrument not validated** | you trust a measure that can't actually detect the thing | Harm-framing's first positive control (a naive jailbreak) failed — forcing an honest "the run is inconclusive until the measure is validated" rather than interpreting the tone effect. |

## The smell test (direction E2 — quick triage for any safety claim, yours or others')
Ask, in order; a "no" or "unknown" is a red flag:
1. Is the ground truth **external** to the method, or does the method grade itself?
2. Was there a **placebo / baseline** for whatever was manipulated?
3. Does a **shuffled-label** control go to chance?
4. Did it **replicate** on a second seed/source?
5. Is the **false-positive rate at the deployment threshold** reported (not just AUC)?
6. Were **length / chain-of-thought / prompt-presence** confounds ruled out?
7. Is the claim about **substance** (real harm/correctness) or just a **surface proxy**?
8. Were the **success criteria fixed before the data**, with no post-hoc retuning?

## Why this is a contribution, not just housekeeping
Published AI-safety evaluations frequently fail several of these (self-defined targets,
no placebo, single seed, AUC-without-operating-point). A compact, *battle-tested*
discipline — shipped with a reusable pre-registration + placebo + shuffled-null +
replication harness (the `redteam_gate/`, `harm_framing/`, `goodhart_general/` templates
already embody it) and a set of **honest retraction case studies** — helps other people
avoid the exact traps this project fell into and climbed out of. That is novel, it is
squarely safety-relevant (better evals → fewer false-safe claims), and it is the most
defensible thing a small, high-integrity effort can put its name to.

*Related: `WORK_MAP.md` (the full falsification log), `NOVEL_SAFETY_DIRECTIONS.md`
(where to point next), the per-experiment `PREREGISTRATION.md` files (the discipline in
practice).*
