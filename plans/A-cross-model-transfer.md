# Plan A: Cross-Model Instability Transfer

## In plain language

We have one leftover clue from all the geometry work that was never fully debunked: sentences that sit in a "messy neighborhood" of an embedding map (the model's internal map of meaning) seem slightly more likely to make a classifier change its answer when the sentence is reworded. The question this plan answers: is that messiness a property of the *sentence itself* — something every AI's map would agree on — or just a quirk of the one map we happened to measure? To find out, we build the same map twice with two unrelated embedding models, mark which sentences are "unstable" in each, and check whether the two models point at the same sentences. The decisive test is cross-prediction: messiness measured on map A must predict actual answer-flips measured on pipeline B, which map A never saw. Success looks like a number (a rank correlation and an AUC score) that clears its pre-registered bar in both directions, with controls showing it is not just "long or weird sentences look weird to everyone." Failure looks like chance-level numbers — and that is genuinely useful, because it would close the entire geometric line of this project for good, with a clean pre-registered receipt instead of a lingering "maybe." Either way, the project stops carrying an unresolved question. Everything runs free, on your own machine, in a few sessions.

## Why this matters / what it builds on

**Honest framing first (non-negotiable):** this is NOT an upgrade of the foundation. The verified foundation is the lean log-prob uncertainty gate (per-task wrong-output AUC 0.60–0.74 at 3B; cross-task 0.63 [0.58, 0.69] with unsupervised z-score; unseen-task transfer QNLI 0.72 / RTE 0.63). Geometry added *nothing* to that gate (dAUC ~ 0) and every geometric "win" so far was self-confirming. This plan is a cheap, pre-registered probe of the **one weak survivor**: the borderline behavioral-instability correlation (r ~ -0.5, ~6% variance) and the PROVISIONAL flip-prediction result.

How weak is the prior, exactly? I verified the source numbers directly:

- The "AUC 0.707" in `C:\Users\User\mirrorfield\docs\BEHAVIORAL_FLIP_UPDATED_FINDINGS.md` rests on **11 flips out of 150 paraphrases** (confirmed in `experiments\behavioral_flip_paraphrase_level_analysis.json`: `n_flips: 11`).
- Worse: in `C:\Users\User\mirrorfield\experiments\behavioral_flip_paraphrase_level_analysis.py` (lines 116–132), every logistic regression was **fit and scored on the same data — no cross-validation**. A 7-feature model in-sample on 11 positives is close to guaranteed to inflate. The 0.707 should be expected to shrink, possibly to chance, under honest evaluation.
- The headline correlation was p = 0.040, single run, on paraphrases treated as independent when they are not — the exact pattern that collapsed four times elsewhere in this project.

Why run it anyway? Three reasons. (1) The anchor literature is real: relative representations (Moschella et al., ICLR 2023) show relational geometry is near-invariant across models even when coordinates differ — so *if* any geometric instability signal is real, it should live on the shared semantic structure and transfer; if it doesn't transfer, it was coordinate noise. (2) It simultaneously discharges two open items on ORIENTATION.md's checklist: "cross-model instability transfer, honest reframe only" and "behavioral-flip AUC 0.707 — apply the standard treatment" (the rebuild required here IS that treatment). (3) For Dillan's end goal — stopping harmful outputs before they're sent — a transferable instability flag would be the only model-agnostic pre-send signal the project has ever produced; a null means all pre-send effort goes into the log-prob gate with no nagging geometric what-if. A null closes the geometric line **completely**.

## What already exists

All verified present on disk (this session, read-only):

- **Flip-experiment texts and legacy labels:** `C:\Users\User\mirrorfield\experiments\behavioral_flip_results.json` — 30 queries (10 safe / 10 borderline / 10 unsafe) × 5 paraphrases each, with texts, legacy OpenAI-pipeline predictions, and flip labels (11/150 flips). Also `behavioral_flip_samples_with_paraphrases.json`, `behavioral_flip_samples.json`.
- **Reference corpus with labels:** `C:\Users\User\mirrorfield\runs\openai_3_large_test_20251231_024532\texts.json` — dict with `texts` (1099 sentiment sentences) and `labels`. (The `embeddings.npy` there is OpenAI 256-d — unusable for new work, but the *texts* let us re-embed everything locally for free.)
- **Geometry code:** `C:\Users\User\mirrorfield\mirrorfield\geometry\bundle.py` — `GeometryBundle(reference, k=50)` computes the 7 k-NN features incl. `knn_std_distance`; reference set is immutable, queries never join the graph (`mirrorfield\geometry\features.py`, `compute_knn_features`).
- **Legacy analysis scripts to mirror (and fix):** `experiments\behavioral_flip_compute_flips.py` (pipeline shape: embed → LR classifier → flips), `experiments\behavioral_flip_paraphrase_level_analysis.py` (contains the no-CV bug — do not copy the evaluation).
- **Models cached** (verified in `%USERPROFILE%\.cache\huggingface\hub`): `sentence-transformers/all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `paraphrase-MiniLM-L6-v2`, `Qwen2.5-3B-Instruct`, `Qwen2.5-0.5B-Instruct`. **GLUE sst2/rte/qnli datasets cached** (`datasets--glue` snapshot contains `sst2`).
- **Environment** (verified): sentence-transformers 5.2.0, sklearn 1.8.0, scipy 1.16.3, torch 2.6.0+cu124, CUDA available.
- **Not reusable:** `experiments\analyze_cross_model_geometry.py` compares LLM *reasoning traces* (Claude vs GLM), not embedding spaces — different question; leave it alone.
- **One download needed:** the second embedding model. Recommend `thenlper/gte-small` (~70 MB, different lab/training pipeline from sentence-transformers, no prompt-prefix gotcha; if substituting an E5 model, the `"query: "` prefix is mandatory or results silently degrade).

## The plan, step by step

All new files go in a new folder `C:\Users\User\mirrorfield\experiments\cross_model_transfer\`. Global seed 42 everywhere except paraphrase generation (seeds 1001, 1002). Lesson from §4e baked in: seed `torch.manual_seed` explicitly, not just numpy/random. All embeddings L2-normalized before any geometry or classifier (makes euclidean ≈ angular and the two spaces scale-comparable; features are additionally converted to within-space percentile ranks, which is scale-free).

**Model cast (fixed now):**
- **A** = `all-mpnet-base-v2` (cached, 768-d) — headline model 1.
- **B** = `thenlper/gte-small` (download, 384-d, different family) — headline model 2.
- **S** = `all-MiniLM-L6-v2` (cached) — *selector* model, used only to stratify query selection so neither headline model defines its own test set; also forms the same-family pair A↔S to calibrate how much correspondence is "free."
- **QC** = `paraphrase-MiniLM-L6-v2` (cached) — paraphrase quality filter only.
- **Generator** = `Qwen2.5-3B-Instruct` (cached, GPU) — paraphrase writer.

**Step 1 — Pre-registration BEFORE any data is generated.** Write `experiments\cross_model_transfer\PREREGISTRATION.md` containing verbatim the criteria in the next section, **the exact paraphrase-generation prompt text and QC thresholds** (paraphrase quality is this plan's top risk — the prompt must not drift between seeds or after a weak first batch), plus the full analysis script `analyze_transfer.py` with thresholds hard-coded. Commit both to git *before* Step 4 runs (commit hash is the timestamp lock). CPU, ~1 session including infrastructure scripts. Artifacts: `PREREGISTRATION.md`, `analyze_transfer.py`, `embed_corpus.py`, `select_queries.py`, `generate_paraphrases.py`, `compute_flips.py`.

**Step 2 — Embed the reference corpus in A, B, S.** `embed_corpus.py`: load the 1099 texts+labels from `runs\openai_3_large_test_20251231_024532\texts.json`, encode with each model (GPU, batched; ~2–5 min total), L2-normalize, save `reference_A.npz`, `reference_B.npz`, `reference_S.npz` (embeddings + labels). Train one seeded `LogisticRegression(max_iter=1000, random_state=42)` per space on the full reference; save accuracy sanity numbers (expect ≥0.90 in-space; if any pipeline is below 0.80 stop and investigate before proceeding).

**Step 3 — Correspondence gate (H-A, the Moschella sanity layer).** `correspondence.py`, CPU, ~5 min. Sample 300 anchor texts from the reference corpus (seed 42). Build relative representations (cosine similarity of each of the 1099 points to the 300 anchors) in A, B, S. Compute: (i) linear CKA between relative-rep matrices, (ii) mean Jaccard@10 nearest-neighbor overlap in the raw spaces, (iii) Procrustes disparity on relative reps — each for pairs A↔B (cross-family) and A↔S (same-family), each against a shuffled-point null. Artifact: `correspondence_report.json` + one heatmap/bar plot. **This is a prerequisite, not a finding** — Moschella et al. predict it passes. If it fails (criteria below), STOP the whole direction and write the null report; instability transfer is meaningless without a structural bridge.

**Step 4 — Select 200 fresh queries with the selector model.** `select_queries.py`, CPU+GPU minutes. Load SST-2 *validation* sentences from the local GLUE cache (never the reference corpus — no train contamination). **Contamination guard (mandatory):** drop any candidate whose normalized text (lowercased, whitespace-collapsed) exactly matches, or has QC-model cosine ≥ 0.98 with, any of the 1099 reference texts — the reference corpus is sentiment data and may overlap SST-2; queries must be unseen by the reference-trained classifiers. Then embed with S, score with S's classifier, stratify by S-pipeline boundary distance: 50 safe / 100 borderline / 50 unsafe (borderline oversampled because that is where flips live — power, decided now). Seed 42. Artifact: `queries.json` (texts, gold labels, S-boundary-distance, stratum, dedup-audit count).

**Step 5 — Generate paraphrases locally.** `generate_paraphrases.py`, GPU, the main compute step: Qwen2.5-3B-Instruct writes 8 sentiment-preserving paraphrases per query (one prompt per query, parse 8 lines), at temperature 0.8, **two independent seeds (1001, 1002) → two complete paraphrase sets**. QC filter (pre-registered): keep paraphrases with QC-model cosine similarity to the original in [0.65, 0.95] and length within ±40%; a query survives if ≥6 paraphrases pass; regenerate failures once, then drop. Rough wall-clock: ~200 prompts × ~20 s × 2 seeds ≈ **2–4 GPU-hours** (can run unattended). Artifacts: `paraphrases_seed1001.json`, `paraphrases_seed1002.json`. **Dillan's manual step:** read 20 random paraphrase pairs and confirm they keep the original meaning — a real QC contribution requiring no code.

**Step 6 — Measure flips in BOTH pipelines.** `compute_flips.py`, CPU/GPU minutes. For each pipeline P ∈ {A, B}: embed originals + paraphrases with P, predict with P's reference-trained classifier, flip = paraphrase label ≠ original's predicted label. Artifacts: `flips_A_seed*.json`, `flips_B_seed*.json`. **Power gate (pre-registered, checked blind):** the ONLY thing inspected before analysis is the flip *count*. Need ≥40 flips per pipeline per seed; if short, increase paraphrases/query to 12 or borderline share to 120 and regenerate — this looks at power only, never at correlations.

**Step 7 — Geometry features in both spaces.** `GeometryBundle(reference_P, k=50).compute(query_embeddings_P)` for P ∈ {A, B}; convert each feature to its percentile rank within its own space. Also compute the **surface-feature baseline** per query: character length, word count, mean log word frequency (`wordfreq` pip package, offline; fallback: frequency within the SST-2 corpus), digit/punctuation ratio. CPU, minutes. Artifacts: `geometry_A.npz`, `geometry_B.npz`, `surface.npz`.

**Step 8 — Run the frozen analysis.** `analyze_transfer.py` (written in Step 1, unmodified) computes, per generation seed:
- **H-B (feature transfer):** Spearman rho between `knn_std_distance` percentile in A vs B over the 200 queries; then *partial* Spearman controlling all surface features. 2000-iteration bootstrap CIs; 1000-iteration permutation null.
- **H-C (behavioral cross-prediction — the headline):** AUC of `knn_std_A` percentile predicting paraphrase-level flips in pipeline **B**, and symmetrically `knn_std_B` → flips in **A**. Primary predictor is the single pre-named feature (no fitting → no overfitting). CIs by **cluster bootstrap over queries** (paraphrases of one query are not independent — the legacy analysis ignored this). Baselines: surface-features-only AUC (GroupKFold by query), and ΔAUC of (surface + knn_std) over (surface) — the increment must be real. Secondary: 7-feature LR with GroupKFold (never in-sample).
- **H-D (within-model replication of the provisional 0.707):** same machinery — primary predictor is the single pre-named feature `knn_std_A` percentile (no model fitting → no overfitting), target flips_A, cluster bootstrap; the 7-feature LR with GroupKFold is secondary only. This is the honest re-test of `BEHAVIORAL_FLIP_UPDATED_FINDINGS.md`.
- **Continuity arm (descriptive only):** re-embed the legacy 30 + 150 texts from `behavioral_flip_results.json` in A and B and report the same statistics, labeled "N too small for inference."
Artifact: `transfer_results.json` + plots. CPU, ~10 min.

**Step 9 — Write the report and update the canon.** `experiments\cross_model_transfer\REPORT.md` with the verdict box; one-paragraph entries for `WORK_MAP.md` (new §) and ORIENTATION.md checklist (tick both items). If null: the entry says the geometric line is closed completely, with the pre-registration commit hash as receipt.

## Pre-registered success / failure criteria

Decided now, before any run. "CI" = 95% bootstrap (cluster bootstrap by query wherever flips are involved). Every criterion must hold on **both** generation seeds (1001 and 1002).

- **G1 — Correspondence gate (H-A):** A↔B mean Jaccard@10 ≥ 0.10 (shuffled null ≈ 0.009) AND linear CKA on relative reps ≥ 0.50. *Fail → stop the entire direction*; report "no structural bridge at this scale" (itself a surprising, reportable null).
- **H-B success:** partial Spearman rho (knn_std percentile, A vs B, controlling surface features) ≥ 0.25 with CI excluding 0. **Weak-but-real:** CI excludes 0 but rho < 0.25. **Component dead:** CI includes 0, or the raw correlation is positive but the partial collapses below 0.10 (= it was surface features all along).
- **H-C success (headline, required for ANY positive claim):** knn_std_A → flips_B AUC CI excludes 0.5 **AND** knn_std_B → flips_A AUC CI excludes 0.5 **AND** ΔAUC over the surface baseline has CI excluding 0, in both directions. Point estimates ≥ 0.60 to call it "useful" rather than "detectable."
- **H-D (replication of 0.707):** GroupKFold AUC for geometry → same-pipeline flips, CI excluding 0.5, replicated across both seeds → the provisional number graduates (at whatever honest value it lands). CI includes 0.5 on either seed → the 0.707 is formally retired as an in-sample artifact.
- **ABANDON THIS DIRECTION** if: G1 passes but H-C fails in either direction on either seed. Pre-stated meaning: *geometric instability is a coordinate-level accident, not a property of the shared semantic manifold; the geometric line of this project is closed completely, including the r ~ -0.5 survivor as a deployable signal.* No re-runs, no threshold adjustments, no "rescued" reanalyses — that is precisely the trap this project documents.
- **If H-C passes:** it is still NOT headlined. Pre-registered next gate: replicate on a third embedding model (e.g., `BAAI/bge-small-en-v1.5`, ~130 MB) before any public claim. A positive here would be the first genuinely new geometric finding since the retraction — which is exactly why it gets the strictest treatment.

## Controls & verification

- **Circularity check — what target does the method get to define? None.** Flips are defined by classifier behavior under paraphrase (behavioral, label-space). Paraphrases are written by an LLM, not selected by geometry. Query stratification uses the *selector* model S's boundary distance (a classifier output, not a geometric feature), and S is outside the headline A↔B pair. Geometry appears exclusively as a *predictor*, and in the headline test its target lives in the *other* model's pipeline, which it cannot touch.
- **Negative controls:** (1) query-level label permutation (1000×) for every AUC — must center on 0.5; (2) shuffled-point null for all correspondence metrics; (3) shuffled-anchor relative reps as a second correspondence null.
- **Surface-confound control:** the explicit surface-feature baseline and partial correlations (length, word count, word frequency, punctuation/digit ratio). A "transfer" fully explained by "rare/long sentences look sparse to every model" is pre-classified as a null — that observation is established literature, not a finding (same verdict ORIENTATION §8b gave the laptop's softmax-vs-geometric demo).
- **Same-family calibration:** A↔S (both sentence-transformers) vs A↔B (cross-family) correspondence reported side by side, so cross-family numbers are read against the inflation ceiling.
- **Seeds:** 42 for sampling/LR/bootstrap; torch generation seeds 1001/1002 explicitly set (the §4e unseeded-sampler bug is the project's own cautionary tale). All criteria must replicate across both generation seeds.
- **Dependence handling:** cluster bootstrap by query + GroupKFold everywhere paraphrase-level rows appear (the legacy analysis's independence assumption and in-sample fitting are both explicitly corrected).
- **Pre-registration lock:** thresholds and analysis script committed to git before data generation; the report links the commit hash.

## Honest risks

- **Probability of an overall null (H-C fails): ~70%.** The within-model signal was 11 flips, p = 0.040, in-sample — cross-model transfer can only be weaker. Best guess at outcomes: G1 passes (~95%, literature), H-B raw correlation positive (~70%) but surviving the surface partial only ~40%, H-C clearing both directions ~25–30%.
- **Most likely way this wastes a week:** paraphrase quality. A 3B model writes mediocre paraphrases; if many silently flip sentiment, "flips" measure paraphrase error, not model instability — inflating flip counts in *both* pipelines in a correlated way and faking transfer. Mitigations: QC similarity band, Dillan's manual 20-pair audit, and the surface/permutation controls; residual risk acknowledged. If audit failure rate > 20%, regenerate with stricter prompting before analysis.
- **Too few flips even after stratification** (legacy rate was 7%): the power gate catches this blind and escalates paraphrase count — costs GPU-hours, not validity.
- **QC-model kinship:** the QC filter (paraphrase-MiniLM) shares a family with A and S, possibly biasing kept paraphrases toward what that family considers "similar." Sensitivity run: repeat headline stats on the unfiltered paraphrase set.
- **A positive could still be a confound we didn't name.** That is what the third-model replication gate and the no-headline rule are for.
- **Scope honesty:** even a full success yields a weak, sentence-classifier-level instability flag — useful as a model-agnostic "this input is unstable, route to review" signal, not a harm detector, and several steps from the production gate.

## Deliverable Dillan will see

One page: `experiments\cross_model_transfer\REPORT.md`, opening with a plain-language verdict box ("Does messiness on one AI's map predict answer-flips on another AI's pipeline? YES/NO, and what that closes or opens"), plus a single scatter plot (`transfer_scatter.png`: each dot a sentence, x = instability percentile in model A, y = in model B, flipped sentences highlighted — if the dots hug the diagonal and the flips cluster top-right, the signal transferred; a shapeless cloud means it didn't), an AUC table with confidence intervals against every baseline, and the pre-registration commit hash. Plus the two one-paragraph canon updates (WORK_MAP.md new section; both ORIENTATION checklist items ticked) ready to paste.

## Effort

- **Sessions:** ~4 AI sessions. (1) Pre-registration + all scripts written and committed; (2) embedding, correspondence gate, query selection, kick off paraphrase generation; (3) flips, geometry, frozen analysis on both seeds; (4) report, canon updates, Dillan's paraphrase audit folded in.
- **GPU:** ~2–4 GPU-hours total (paraphrase generation, two seeds, unattended-capable on the 3060 Ti; Qwen2.5-3B fits 8 GB in fp16). Everything else is CPU- or GPU-minutes.
- **Downloads:** `thenlper/gte-small` ~70 MB; `wordfreq` pip package ~10 MB; optional third model for the replication gate `BAAI/bge-small-en-v1.5` ~130 MB only if H-C passes. SST-2 already cached. Zero paid API spend.

### Critical Files for Implementation
- C:\Users\User\mirrorfield\WORK_MAP.md (canonical truth; receives the new section)
- C:\Users\User\mirrorfield\ORIENTATION.md (§4 honest reframe; checklist items this plan discharges)
- C:\Users\User\mirrorfield\mirrorfield\geometry\bundle.py (GeometryBundle, the 7 k-NN features incl. knn_std_distance)
- C:\Users\User\mirrorfield\experiments\behavioral_flip_results.json (legacy 30×5 texts + flip labels for the continuity arm)
- C:\Users\User\mirrorfield\runs\openai_3_large_test_20251231_024532\texts.json (1099-text labeled reference corpus to re-embed locally)