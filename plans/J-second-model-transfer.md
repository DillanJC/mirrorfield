# Plan J — Second-model transfer: does the boundary-calibration failure travel?

*DRAFT 2026-07-04 (auto session) — **NOT LOCKED.** Nothing here runs until Dillan
reviews, edits, and commits a final PREREGISTRATION.md; the GPU run itself waits for
his explicit go (standing rule). This is the only real path off "on this model," and
the state doc's own warning applies: verify first it isn't the geometry frame in
disguise — see the circularity check below.*

## Question (falsifiable)

On a **second small model**, does the §4y pattern reproduce: aggregate calibration looks
clean while the calibrated score is overconfident in the lowest raw-margin quintile —
and does the raw margin still discriminate (acc(Q5) > acc(Q1))? Each finding is scoped
to its model; two models showing the same shape upgrades "on this model" to "on both
models tested," nothing more.

## Model candidates (8 GB VRAM ceiling; pick ONE, pre-registered)

1. **Qwen2.5-0.5B-Instruct** — already cached (no download), same family (weakness: a
   family-specific artifact wouldn't be caught; strength: cleanest scale contrast).
2. **Llama-3.2-3B-Instruct** — different family, same scale class (strength: family
   transfer; cost: ~6 GB download, gated license accept).
3. **Phi-3-mini / Gemma-2-2B** — alternates if (2) is blocked.

Recommendation to discuss: run **(1) first** (free, cached, tests scale within family),
name (2) as the follow-up (tests family at matched scale). One model per
pre-registration; no pooling across models.

## Design (mirrors §4y exactly, to keep the comparison honest)

1. **Calibrator fit, per model (the part that needs care):** the 3B gate's calibrator is
   model-specific; reusing it on another model would measure the wrong thing. Protocol:
   fit a fresh scaler→logistic→isotonic on a **separate calibration split** (fresh items,
   its own seed, e.g. 500 items seed 7), freeze it, THEN run the evaluation items. The
   calibration split and evaluation samples never mix (the §4h out-of-fold lesson).
   **De-overlap clause (2026-07-12, from the refit hostile review):** seed sampling
   draws from shared pools, so different seeds OVERLAP (e.g. 42∩1337 = 40/500) — and
   greedy decoding makes overlapping items identical across splits. The lock must
   specify: evaluation items are filtered by (task, ds_index) against the calibration
   split before generation, and the exclusion count is reported.
2. **Evaluation:** the A1 answer pass, unchanged prompts/parsing, on the same item pools:
   500 RTE+QNLI items × seeds 42/1337 (fresh selection for this model is fine; the
   comparison is of *patterns*, not per-item values). Persist raw mm/me/br + task from
   the start (the persistence fix is already in the harnesses).
3. **Analysis (identical, pre-registered):** §4y Opt-2 — raw-margin quintiles, Wilson
   CIs, min-n 30, per-quintile gap, Q5−Q1 discrimination with bootstrap CI; plus the
   audit's two robustness diagnostics (within-task Q1 gap; matched-score margin split)
   run as standard, not post-hoc.

## Draft verdicts (to finalize with Dillan before locking)

- **PATTERN TRANSFERS:** Q1 gap ≥ +0.10 with accuracy CI excluding mean score, both
  seeds, AND discrimination CI > 0 both seeds — on the new model, with its own fresh
  calibrator.
- **PATTERN DOES NOT TRANSFER:** Q1 calib holds (CI contains mean score) both seeds —
  a real, reportable null: the §4y failure is then not a generic property of this
  calibration recipe even across two models.
- **MIXED/UNDERPOWERED:** anything else; reported, not interpreted.

**Power analysis (done 2026-07-08, pre-lock — numbers to lock against):** Wilson 95%
half-widths at torn-quintile accuracy 0.4–0.6: ~0.095 at n=100/quintile (the 3B
design's 500 items/seed) — MARGINAL for CI-excluding the +0.10 threshold gap; ~0.078
at n=150/quintile (750/seed) — adequate; ~0.068 at n=200/quintile (1,000/seed) —
comfortable. **Recommendation to lock: 1,000 items per seed for the 0.5B run** — the
model is fast, the larger sample costs little, and it removes the marginal-power
excuse from every possible verdict.

## Circularity check (the mandatory one)

Target = correctness vs external GLUE gold; method = log-prob features; stratifier = a
method input, which the §4y audit establishes as the correct conditional probe — no
shared assumption, and nothing geometric anywhere in the pipeline. The one new risk is
**recipe circularity**: fitting the fresh calibrator and evaluating it on data from the
same distribution guarantees decent *aggregate* calibration by construction — that is
fine, because the claim under test is about the *tail*, which the recipe does not
optimize for separately. State this in the lock.

## Effort

Calibration split + two eval seeds ≈ 1,500 generations (0.5B: fast; 3B-class: ~as §4y).
One build session (mostly config: model name + calibrator fit script) + one GPU session.
Analysis is the existing code path.
