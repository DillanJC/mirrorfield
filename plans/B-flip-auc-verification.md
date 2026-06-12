# Plan B: Behavioral-Flip AUC 0.707 — Verify or Retire

## In plain language

The project has one last unverified number: a claim that the "geometry" of how a sentence sits among its neighbors can predict whether an AI classifier will change its answer when the sentence is reworded (a "flip"). The claim says this prediction scores 0.707, where 0.5 is a coin toss and 1.0 is perfect. While preparing this plan I read the original analysis code and found a serious flaw: the 0.707 was measured by testing the predictor on the exact same 150 examples it was trained on — like grading a student on the same questions they studied from, answers included. That almost always inflates the score. We will redo the measurement honestly: first re-score the existing data with a proper train-on-some, test-on-others split (a few hours, free), then — regardless of that outcome — run a bigger, fresh version with new sentences and rewordings generated entirely on Dillan's own computer (no paid services). Success means the number survives honest testing twice, with error bars, and becomes safe to say out loud. Failure means the number collapses — which is also a win: it would close the very last "geometry" claim, make the project's public story simpler and fully consistent, and the falsification itself is publishable evidence of good scientific hygiene.

## Why this matters / what it builds on

- This is the **last provisional number** in the project (ORIENTATION.md §9 open-leads checklist: "Behavioral-flip AUC 0.707 — apply the standard treatment before using the number anywhere"). Everything else is either locked (the v3.0 log-prob-gate numbers) or closed (all other geometric claims, publicly retracted).
- It connects to the one **surviving weak signal**: borderline behavioral-instability correlation r ≈ −0.5 (~6% variance). If flip-prediction survives, that signal gets a defensible headline number for outreach about "predicting when a model is unstable" — directly relevant to Dillan's goal of catching bad outputs before they ship. If it collapses, outreach simplifies to the clean v3.0 story with zero asterisks.
- **The prior is weak — be honest about it.** The 0.707 comes from a rescued follow-up (original N=30 test failed at p=0.141; the paraphrase-level re-analysis got p=0.040). That is the exact pattern that collapsed four other times in this project. Forensic findings from this planning pass make the prior weaker still:
  1. **The 0.707 is an in-sample (training) AUC.** In `behavioral_flip_paraphrase_level_analysis.py` (lines 116–132), every model is `fit()` on all 150 points and then scored on those same 150 points. No cross-validation exists anywhere in the pipeline. With 7 features and 11 positive cases, ~0.7 training AUC is close to what pure overfitting produces.
  2. **Pseudo-replication.** The geometry features come from the 30 *original* query embeddings, each copied 5× across its paraphrases — so there are only 30 unique feature vectors, not 150 independent observations. The p=0.040 point-biserial test inherits the same inflation.
  3. **Only 11 flips total**, and ~4 of them come from one query (the sarcastic "root canal" sentence, which the original doc itself flags as probably mislabeled).
  4. A classic overfit symptom the original doc spun positively: the combined model (0.603) scored *worse* than geometry alone (0.707).
- Bottom line: this direction is the standard treatment WORK_MAP demands — pre-registered criteria, grouped cross-validation, cluster bootstrap, shuffled nulls, multi-seed, fresh held-out data. Either outcome closes the checklist item.

## What already exists

All verified present on disk (2026-06-12):

- **Archived 150-point dataset, fully self-contained** (no API needed to re-analyze): `C:\Users\User\mirrorfield\experiments\behavioral_flip_results.json` (549 KB) — 30 queries × 5 paraphrases, with stored 256-dim OpenAI embeddings, predictions, probabilities, flip labels, zones.
- **Reference set for classifier + geometry**: `C:\Users\User\mirrorfield\runs\openai_3_large_test_20251231_024532\` — `embeddings.npy` (1099 × 256, OpenAI text-embedding-3-large), `labels.npy`, `boundary_distances.npy`, `texts.json` (1099 short synthetic product/service sentiment sentences, 19–82 chars).
- **Original pipeline scripts (templates for the redo)**: `C:\Users\User\mirrorfield\experiments\behavioral_flip_sample_selection.py`, `behavioral_flip_generate_paraphrases.py` (GPT-4 — to be replaced with local Qwen), `behavioral_flip_compute_flips.py` (defines signed boundary distance and flip-rate logic), `behavioral_flip_paraphrase_level_analysis.py` (the flawed analysis), plus `behavioral_flip_protocol.md` and the doc under test, `C:\Users\User\mirrorfield\docs\BEHAVIORAL_FLIP_UPDATED_FINDINGS.md`.
- **Geometry library**: `C:\Users\User\mirrorfield\mirrorfield\geometry\bundle.py` — `GeometryBundle(reference, k=50)` with the same 7 features (`knn_mean/std/min/max_distance`, `local_curvature`, `ridge_proximity`, `dist_to_ref_nearest`).
- **Local model cache (verified in `~\.cache\huggingface`)**: Qwen2.5-3B-Instruct (paraphrase generation + label-preservation judge), all-mpnet-base-v2 and all-MiniLM-L6-v2 (local embedders), and GLUE **sst2/rte/qnli already cached** (fresh prompts source). **Zero downloads, zero API spend needed.**

## The plan, step by step

All new artifacts go under `C:\Users\User\mirrorfield\experiments\flip_verification\` (the implementing session creates this; this plan is read-only). Every script sets seeds for `numpy`, `random`, `torch` (incl. CUDA), and sklearn `random_state`.

**Phase 0 — Forensic re-analysis of the archived 150 points (CPU only, free, fast)**

1. **Reproduce the original number exactly.** Script `phase0_reanalysis.py` loads `behavioral_flip_results.json` + the archived reference embeddings, recomputes the in-sample AUCs, and asserts geometry-only ≈ 0.707, boundary-only ≈ 0.574, combined ≈ 0.603 (matching `behavioral_flip_paraphrase_level_analysis.json`). This proves we re-built the pipeline correctly before changing anything. CPU, ~1 min. Artifact: `phase0_repro_check.json`.
2. **Honest re-scoring, same data.** Same script: leave-one-query-out cross-validation (30 folds, all 5 paraphrases of a query held out together — this is the primary statistic because plain k-fold leaks the duplicated feature vectors), pooled out-of-fold AUC for geometry-only / boundary-only / combined; cluster bootstrap CI (resample the 30 queries with replacement from cached out-of-fold predictions, 2000 resamples, percentile 95% CI); 1000 query-level shuffled-label permutations re-running the full CV pipeline (null AUC distribution); influence check: re-run everything with the "root canal" query removed. CPU, ~10–20 min. Artifact: `phase0_reanalysis.json` + `phase0_forest_plot.png` (AUCs with CIs vs the null band).
3. **Record the Phase 0 verdict** against the pre-registered criteria below. Expected realistic outcome: with only 11 events the CI will be very wide — Phase 0 can *kill* the number or declare it *unverifiable at this N*; it cannot lock it. Either way, proceed to Phase 1, which tests the underlying hypothesis properly.

**Phase 1 — Fresh data, fully local, properly powered**

*Pre-registration lock (same mechanism as Plan A): before step 6 generates any
paraphrase, commit `phase1_analyze.py` (thresholds hard-coded), the paraphrase
prompt template, and the QC rules to git — the commit hash is the receipt that
criteria preceded data.*

*Sequencing note: Plan A's H-D arm performs this same within-model replication
on its own fresh data. If Plan A is chosen first, its H-D result satisfies
Phase 1 here (and Phase 0's forensics still stand alone); if Plan B runs first,
Plan A may cite this Phase 1 instead of re-running H-D. Do not run both blindly.*

4. **Build the local pipeline base.** Script `phase1_build_base.py`: load SST-2 from the cached GLUE dataset; embed 3000 train sentences with **all-mpnet-base-v2** (GPU, ~2 min); train `LogisticRegression` sentiment classifier; this embedding set is also the `GeometryBundle` reference (k=50). Compute signed boundary distance with the exact formula from `behavioral_flip_compute_flips.py`. Artifact: `phase1_base.npz` (embeddings, labels, classifier params) + accuracy report. Note the pre-registered framing: the claim under test is "neighborhood geometry predicts flips" in general — it was never specific to OpenAI embeddings (the R3 precedent already established local embedders as a fair test of the geometry hypothesis).
5. **Select 150 fresh queries.** From the SST-2 *validation* split (872 real sentences, none used in the original experiment), length-filtered 30–200 chars: 50 safe / 50 borderline / 50 unsafe by boundary-distance zone (same ±0.5 thresholds as `behavioral_flip_sample_selection.py`), evenly spaced within zone, seed 42. CPU, ~1 min. Artifact: `phase1_queries.json`. Note: selection uses *margin* zones, never geometry.
6. **Generate paraphrases locally — two independent batches.** Script `phase1_paraphrase.py`: Qwen2.5-3B-Instruct (fp16, fits 8 GB VRAM), one prompt per query returning 6 paraphrases on separate lines (same instruction template as the GPT-4 script: preserve sentiment, vary wording, similar length), temperature 0.7, **seeded sampling**; run twice with generation seeds 1001 (batch A) and 2002 (batch B). ~300 generations of ~200 tokens → **roughly 40–60 min GPU per batch**. Artifact: `phase1_paraphrases_A.json`, `phase1_paraphrases_B.json`.
7. **Quality control (rules fixed before any flip is computed).** Script `phase1_qc.py`: drop paraphrases that are empty, identical to the original, outside ±50% length, or contain meta-text; then a label-preservation check — Qwen2.5-3B as a zero-shot sentiment judge (single-token answer, greedy); a paraphrase is kept only if the judge's label matches the source query's gold label. The judge is a different system from the mpnet classifier under test, so flips cannot be QC'd away by the thing being evaluated. ~1800 short judge calls, **~30–45 min GPU total**. Artifact: `phase1_qc_report.json` (drop counts per rule — logged, never tuned after seeing results).
8. **Compute flips + features.** Script `phase1_flips.py`: embed all surviving texts with all-mpnet-base-v2 (GPU, ~1 min), classify, flip = paraphrase prediction ≠ original prediction; `GeometryBundle` 7 features on original query embeddings; baseline feature = |margin|. Artifact: `phase1_dataset_A.npz`, `phase1_dataset_B.npz`. Expected ~800+ QC-passed paraphrases and (at the original ~7–15% flip rate with borderline oversampling) **~60–120 flip events per batch** — versus 11 originally.
9. **Honest analysis, batch A then batch B.** Script `phase1_analyze.py`: grouped (by query) out-of-fold AUC via leave-one-query-out, for geometry-only / margin-only / margin+geometry; cluster bootstrap 95% CIs (2000 resamples); 1000 query-level shuffled-label nulls; 10 CV-shuffle seeds (report median + range); query-level analysis too — point-biserial of `knn_std_distance` vs per-query flip rate over the 150 queries (the originally designed, honest granularity, now actually powered); ΔAUC of (margin+geometry) over margin-alone with bootstrap CI. CPU, ~15 min. Artifacts: `phase1_results_A.json`, `phase1_results_B.json`, `phase1_forest_plot.png`, `phase1_roc.png`.

**Phase 2 — Verdict and documentation**

10. **Write the verdict report** `docs/BEHAVIORAL_FLIP_VERIFICATION_REPORT.md`: Phase 0 forensics (including the in-sample-AUC finding), Phase 1 numbers against the pre-registered criteria, plots, and a one-box "what may be said out loud" summary. Mark `BEHAVIORAL_FLIP_UPDATED_FINDINGS.md` as superseded by it, and update the WORK_MAP/ORIENTATION checklists accordingly. CPU, ~1 session of writing.

## Pre-registered success / failure criteria

Decided now, before any run. Primary endpoint: **paraphrase-level, grouped out-of-fold AUC of the geometry-only model** (the direct analog of the 0.707 claim). All CIs are 95% cluster-bootstrap by query.

**Phase 0 (archived data) — the old number itself:**
- The 0.707 may be cited only if out-of-fold geometry AUC ≥ 0.65 AND CI lower bound > 0.50 AND it survives removal of the "root canal" query. Anything else → the 0.707 is **retired immediately** as an in-sample artifact (expected outcome). A wide CI straddling 0.5 → verdict "unverifiable at N=11 events"; the number still may never be cited.

**Phase 1 (fresh data) — the underlying hypothesis. LOCK requires ALL of:**
1. Batch A geometry-only out-of-fold AUC CI lower bound > 0.50 (point estimate ≥ 0.60 for it to be called more than "weak").
2. Batch A AUC above the 95th percentile of the 1000-permutation null.
3. Replication: batch B CI lower bound > 0.50 AND batch B point estimate inside batch A's CI.
4. Query-level confirmation: `knn_std_distance` vs flip-rate correlation p < 0.05 with the same sign in both batches.
- **Headline as "geometry predicts instability"** additionally requires: ΔAUC of (margin+geometry) over margin-alone, CI excluding 0 in both batches. If 1–4 pass but this fails, the locked claim is downgraded to "flip-proneness is predictable from the classifier's own margin; geometry is redundant" — consistent with the v3.0 verdict that geometry adds nothing.

**ABANDON the direction** (and publicly close the last geometric claim) if: batch A geometry-only CI includes 0.50, OR batches A and B disagree on sign/significance. No third batch, no re-slicing, no alternative feature sets — a rescue-of-a-rescue is exactly the forbidden pattern.

**Declare "underpowered, no verdict"** (rather than torturing data) if QC-passed paraphrases average < 4 per query or batch A has < 30 flip events.

## Controls & verification

- **Negative control:** 1000 query-level shuffled-label permutations through the full CV pipeline; must center on ~0.50. If the null distribution is shifted, the pipeline leaks — stop and debug before interpreting anything.
- **Positive control:** the margin-only baseline should beat chance at predicting flips (boundary proximity → instability replicated twice already, H2's 1.5×). If margin is at chance on fresh data, the paraphrases themselves are suspect (e.g., sentiment not preserved) — investigate the QC pipeline before reading the geometry arm.
- **Seeds:** numpy/random/torch (incl. CUDA sampling — the §4e lesson) and sklearn all fixed and logged in every artifact; two independent generation seeds give the A/B replication.
- **Grouped everything:** CV folds, bootstrap, and permutations all operate at the query level — the unit of independence — eliminating the pseudo-replication that inflated the original p=0.040.
- **Influence checks:** leave-one-query-out influence on the headline AUC; a result that depends on one query cannot lock (Phase 0's root-canal lesson, made systematic).
- **Circularity check — what target does the method get to define? None.** The label (does the classifier's answer change under rewording) is behavioral, defined with no geometric input; query selection uses margin zones, not geometry; paraphrase generation and QC never see geometric features; the QC judge (Qwen) is a different system from the classifier under test (mpnet+logreg). One shared-substrate caveat stated openly: features and classifier live in the same embedding space — but the target is behavior, not geometry, so the method cannot confirm itself by construction.

## Honest risks

- **Probability of null: roughly 70–80%** that the geometry-only signal collapses on honest evaluation, and even if it clears chance, high odds it fails the ΔAUC-over-margin bar (every non-circular test in this project has found geometry redundant with cheaper signals). Plan accordingly: the abandon branch is the *likely* branch, and it is fully valuable — it closes ORIENTATION's checklist item and ends the geometric era cleanly.
- **In-sample artifact already identified:** the strongest single risk to the old number is no longer a risk — it is a finding. The original 0.707 had no train/test split at all. Expect Phase 0 to land near chance.
- **Embedder-switch confound:** Phase 1 uses mpnet instead of the dead OpenAI pipeline. A skeptic could blame a Phase 1 null on the embedder. Mitigation: Phase 0 runs on the *original* OpenAI embeddings, so the two phases bracket the claim; and the hypothesis was always about neighborhood geometry generally, not one embedder (the R3 precedent).
- **Paraphrase-quality confound:** if Qwen's rewordings genuinely change sentiment, "flips" are correct model behavior, not instability — inflating flip counts with noise. The QC judge mitigates but is itself imperfect (~few % error); this caps how strong any positive claim can be worded.
- **Most likely way this wastes a week:** fiddling with paraphrase prompts and QC thresholds across multiple sessions. Mitigation: the prompt template and QC rules in steps 6–7 are fixed by this plan; one batch-A generation run, one batch-B run, no regeneration. If generation quality is unusable on first inspection, that is a reported result ("local 3B paraphrasing insufficient"), not a tuning loop.

## Deliverable Dillan will see

One report: `docs/BEHAVIORAL_FLIP_VERIFICATION_REPORT.md`, opening with a plain-language verdict box — either **"LOCKED: reworded-question instability is predictable at AUC X.XX [CI], replicated twice"** (with the exact sentence safe to use in outreach) or **"CLOSED: the 0.707 was a statistical artifact; the last geometric claim is retired"** (with the one-paragraph retraction text ready to paste into the public repo). Plus two pictures: a forest plot showing every AUC with its error bars against the grey "coin-toss" null band (Phase 0 next to Phase 1, batch A next to batch B), and the ROC curve for the surviving model if any. The checklist item in ORIENTATION.md §9 gets ticked either way.

## Effort

- **Sessions:** 3–4 total. Session 1: Phase 0 (build + run + verdict). Session 2: Phase 1 steps 4–8 (base, queries, generation batch A+B, QC, flips). Session 3: analysis + report; a short session 4 if documentation/public-repo updates spill over.
- **GPU-hours:** ~2–3 total on the RTX 3060 Ti (paraphrase generation ~1.5–2 h, QC judging ~0.5–0.75 h, embedding minutes). Phase 0 and all statistics are pure CPU.
- **Downloads:** none — Qwen2.5-3B-Instruct, all-mpnet-base-v2, and GLUE/SST-2 are already in the HF cache (verified). **API spend: $0.**

### Critical Files for Implementation
- C:\Users\User\mirrorfield\experiments\behavioral_flip_paraphrase_level_analysis.py (the flawed analysis to reproduce, then correct — the in-sample AUC bug is at lines 116–132)
- C:\Users\User\mirrorfield\experiments\behavioral_flip_results.json (archived 150-point dataset with embeddings; Phase 0 input)
- C:\Users\User\mirrorfield\experiments\behavioral_flip_compute_flips.py (boundary-distance and flip-rate definitions to reuse verbatim)
- C:\Users\User\mirrorfield\mirrorfield\geometry\bundle.py (GeometryBundle, k=50, the 7 features under test)
- C:\Users\User\mirrorfield\docs\BEHAVIORAL_FLIP_UPDATED_FINDINGS.md (the claim under verification; superseded by the verdict report)