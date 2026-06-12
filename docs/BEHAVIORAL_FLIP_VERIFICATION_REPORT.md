# Behavioral-Flip Verification Report — Phase 0 Verdict

**Date:** 2026-06-12 · **Plan:** `plans/B-flip-auc-verification.md` (Phase 0)
**Supersedes:** `BEHAVIORAL_FLIP_UPDATED_FINDINGS.md`

## Plain-language verdict box

> **CLOSED: the 0.707 was a statistical artifact; the last geometric claim is
> retired.** The original analysis graded its predictor on the same examples it
> trained on. Graded honestly — never letting the model see the question it is
> judging — the score falls from 0.707 to 0.49, a coin toss. A shuffle test
> confirms it: random feature assignments score the same (p = 0.48). The number
> may never be cited.

## What was tested

The claim from `BEHAVIORAL_FLIP_UPDATED_FINDINGS.md`: 7 geometric neighborhood
features of a query predict whether a paraphrase of it flips a sentiment
classifier's answer, AUC 0.707 (N=150 paraphrases, 11 flips).

## Findings

1. **Pipeline reproduction: exact.** Recomputed in-sample AUCs match the stored
   results to 4 decimals (boundary 0.5736, geometry 0.7073, combined 0.6027) —
   the re-analysis tested precisely the original construction.
2. **The flaw:** `behavioral_flip_paraphrase_level_analysis.py` (lines 116–132)
   contains no train/test split; every model is fit and scored on all 150 rows.
   Those rows hold only 30 unique feature vectors (each query's geometry copied
   across its 5 paraphrases) and 11 positive events.
3. **Honest evaluation** (leave-one-query-out CV; 2000-resample cluster
   bootstrap by query; 1000 cluster-preserving feature permutations):

   | model | in-sample | honest out-of-fold | 95% CI | permutation p |
   |---|---|---|---|---|
   | geometry (7 features) | 0.707 | **0.492** | [0.32, 0.75] | 0.48 |
   | boundary distance only | 0.574 | 0.551 | [0.39, 0.74] | — |
   | combined | 0.603 | 0.514 | [0.36, 0.73] | — |

4. **Influence check:** dropping the highest-flip query (4 of 11 flips, the
   "root canal" sample the original doc itself flagged as likely mislabeled)
   *raises* the point estimate to 0.69 with CI [0.43, 0.89] — still spanning
   chance, and itself a demonstration that estimates at 11 events are noise.

## Verdict against the pre-registered criteria

Citation required: OOF AUC ≥ 0.65 AND CI lower bound > 0.50 AND survival of the
influence check. Result: **0.492, CI includes 0.50, permutation-indistinguishable
from chance → RETIRED.**

## Status of the underlying hypothesis

This Phase 0 retires the *number*, not the *question*. Whether neighborhood
geometry predicts flips on properly powered fresh data remains testable
(Plan B Phase 1, or Plan A's H-D arm — run one, not both). The prior is now
weaker: every honest evaluation of a geometric claim in this project has landed
at chance.

## Artifacts

- `experiments/flip_verification/phase0_reanalysis.py` (the harness)
- `experiments/flip_verification/phase0_results.json` (full numbers)
- Canon updates: `WORK_MAP.md` §4j; `ORIENTATION.md` checklist.
