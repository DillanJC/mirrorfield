# Geometric Safety Features — Work Map & Reconstruction

*Written 2026-06-02 to re-orient after a break. Captures WHERE the work lives,
what each strand concluded, and the honest status of the central claim.*

---

## 1. The four scattered locations

| Location | What it is | Status |
|----------|-----------|--------|
| `C:\Users\User\Repo\mirrorfield-v3-10-public` | **Origin framework.** Philosophy + "H4 120-cell GNN" vision ("Lucid Equilibrium", LEI, ERA). Mostly markdown specs. Public on GitHub (DillanJC). | Frozen Nov 2025. Superseded. Its central H4 bet was later tested → ceiling r=0.42. |
| `C:\Users\User\mirrorfield` (this folder) | **Parent code repo.** Phases A–E, Tracks 1–5, live LLM-API work, the data-filter pivot. | **ACTIVE. Newest work. Much UNCOMMITTED.** |
| `C:\Users\User\geometric_safety_features-Experiment` | **Clean fork** for public-facing packaging. Tracks 1–3, Sati, Mandala-MoE, MCP server. | Last commit 2026-02-07. |
| `*.zip`, `geometric_safety_features ( The Back up )` | Snapshots. | Ignore. |

Tracks 1–3 exist in both code repos. **Tracks 4–5 only live in the parent `mirrorfield/` folder** and never made it to the Experiment repo.

---

## 2. What each track concluded

- **Tracks 1–3 (poison detection, physics, state API):** geometry detects
  *geometry-defined* poison well (AUC 0.79–1.0). See §4 for the big caveat.
- **Track 4 (live interventions):** injecting prompts when geometric signals fire.
  Net quality effect ~ -2% (within noise). Design tasks helped (+0.04..+0.10),
  ethics tasks hurt (-0.08..-0.11).
- **Track 5 (recursive self-learning):** learner converged to *silencing itself*;
  Goodhart detector correctly flagged metric-gaming. Root cause: "penalty ratchet"
  (asymmetric rates → every signal drifts to silence). Honest negative result.
- **The pivot (2026-03-12):** `hypothesis_test.py` concluded interventions may be
  fundamentally harmful → pivoted to `training_data_filter.py` (clean data BEFORE
  training instead of steering live). **Last thing done; never run on real data.**

---

## 3. Latent bugs found 2026-06-02

- `mirrorfield/geometry/sati_schema.py` defines `generate_sati_feedback_percentile`
  **twice** (~line 253 and ~line 603). Python keeps the second. Their kwargs differ
  (`compressed_curvature_pct` vs `compressed_curv_pct`). The filter matches the live
  one by luck; the dead first definition should be removed.
- `training_data_filter.py` demo plants poison as a tight Gaussian blob → still the
  easy (geometry-defined) case.

---

## 4. The central caveat (most important thing to remember)

Prior poison "wins" (GMR=1.0, Sati 6x, Iterative Zoom) **defined poison via embedding
geometry (KMeans cluster / boundary distance) and then detected it with embedding
geometry.** That is partly circular — a geometric magnet finding a geometric needle.

### Blind test 2026-06-02 (`experiments/track1_poison/test_filter_blind.py`)

Detection method held fixed; only the *poison-generation* varied. Honest signal =
Sati AUC (GMR auc excluded: it detects the label-flip itself, not geometry).

| Regime | Poison chosen by | Sati AUC |
|--------|-----------------|----------|
| R0 cluster (control) | geometry (KMeans) | **0.601** (5.7x compressed ratio) |
| R1 label-only | random ids, label only | **0.473** (chance) |
| R2 trigger-patch | random ids, embedding moved | **0.475** (chance) |

**Conclusion:** geometry only detects poison that forms a dense geometric cluster.
A sparse, geometry-blind, or trigger-style backdoor is invisible to it (~chance).
Negative controls passed (~0.43–0.50). Results: `blind_filter_results.json`.

### R3 — real external test, COMPLETED 2026-06-02

Ran the real benchmark (`test_filter_r3_real.py`): genuine **SST-2** sentiment,
**BadNets** `"cf"` trigger inserted in TEXT space on random ids + label flip,
embedded **locally** with cached `all-MiniLM-L6-v2` (384-dim; free, no API key —
the OpenAI keys were gone, and a local embedder is a fair test of the *geometry*
hypothesis, not a byte match to the old 256-dim pipeline).

First pass looked like a **surprise: Sati AUC = 0.725.** It was an artifact.
Confound controls (`test_filter_r3_controls.py`) dissected it:

| Arm | Change | Sati AUC | compressed ratio |
|-----|--------|----------|------------------|
| baseline (shared `cf` + flip) | reproduce R3 | 0.678 | 20.4x |
| **A** varied token per sample + flip | distinct triggers | **0.497** (chance) | 0.72x |
| **B** shared `cf`, **NO** label flip | token only | 0.678 | 20.4x |
| **C** no token, flip only (=R1) | geometry-blind | 0.507 (chance) | 1.6x |

Two arms kill the "real detection" reading: **A** collapses to chance the moment
each poisoned sample gets a *different* trigger (the signal was the **repeated
token** forming one shared cluster = R0 in disguise); **B** is identical to
baseline with labels untouched (Sati keys on the inserted token's geometric dent,
not on "poison" — it ignores labels entirely). So R3 lands exactly where R0–R2
did. Results: `r3_real_results.json`, `r3_controls_results.json`.

**Net:** the central caveat is now confirmed against a real, attacker-chosen text
backdoor. Geometry detects *repeated-token / dense-cluster* poison only; vary the
trigger and it is at chance. The line is closed.

### Mar-21 "Sati-weighted training" result — circular, reviewed 2026-06-07

The last OpenCode session before the break (`exp1_comprehensive.py`, run 2026-03-21,
GLM-5 via OpenCode) ended on an apparent **win**: "pr_only loss SUPPORTS the
hypothesis on synthetic *and* real data, best config `lambda_0.1_loss_pr_only_data_real`."
Reviewed the code + `exp1_comprehensive_results.json` directly — the verdict is hollow,
a **third instance of the same circularity pattern** (alongside the v2.0 README's 0.947
and the R3 0.725 artifact):

- **Tautological success metric.** "improved_count" tallies how many *geometry features*
  moved in the good direction. But the `pr_only` loss **directly optimizes participation
  ratio.** "pr IMPROVED" therefore just means the optimizer optimized its own target.
  There is **no downstream task metric anywhere** (no accuracy / held-out error /
  detection AUC) — so "SUPPORTS" never meant "the model got better at anything."
- **Inflated by a copy-paste bug.** `pr` and `d_eff` both read `phase2_5_features[:, 2]`
  (lines 244–252) — they are the *same number* counted twice. This is why `pr_only`
  scored 5.5 vs 1.0 for the other losses: it double-counts its own optimization target.
- **"Real data" was not real.** The "real" arm loads `poison_mask_cluster.npy` — the
  **KMeans cluster** (geometry-defined) poison, i.e. the R0 circular setup relabelled
  "real." It was never a text backdoor.
- **No control, single seed,** arbitrary ±0.01 verdict thresholds, bare `except:` that
  silently returns a constant loss.

**Decision:** not re-running this with multi-seed/controls — the metric itself is
invalid, so that would only measure a tautology precisely. The geometry-as-training-
signal line is closed for the same reason as detection: it confirms itself by
construction. (Engineering quality of the GLM code was fine — clean, runnable; the flaw
is in the experimental design, not the implementation.)

### §4b. MCP uncertainty ablation — first NON-circular test, 2026-06-07

The one test the geometry could not rig, because the target label is defined
entirely outside it. Harness: `experiments/mcp_uncertainty_ablation.py` (uses the
**real shipped MCP functions** from `mirrorfield/mcp/uncertainty.py`).

Task: a small instruct LM (Qwen2.5-0.5B) classifies 300 balanced SST-2 sentences
(free-form short generation). Label = **was the prediction wrong vs the gold
sentiment** (ground truth, geometry-independent). Predict "wrong" from:
- STANDARD = mean token margin, mean entropy, boundary ratio (the textbook signals)
- GEOMETRY = sequence PR, embedding pr_mean/pr_min/se_mean (the MCP's geometry add-on)

5-fold CV AUC, bootstrap CI on the delta, shuffled-label null control.

| Feature set | AUC (predict wrong) |
|-------------|---------------------|
| STANDARD (margin+entropy+boundary) | **0.667** |
| GEOMETRY alone | 0.535 (~chance) |
| STANDARD + GEOMETRY | 0.668 |
| **delta from adding geometry** | **+0.0014**, 95% CI **[-0.050, +0.048]** |
| shuffled-label null | -0.074 (control passes — no spurious lift) |

**Two findings, one good one final:**
1. *(constructive — the first earned positive)* The **standard** log-prob signals
   genuinely predict wrong outputs (0.667) on a label the geometry never defined.
   This is real and shippable. The MCP works **as a standard uncertainty gate.**
2. *(closes the geometry line for good)* The geometry adds **nothing** — delta CI
   straddles zero, geometry-alone is near chance. Drop it. This is the 4th data
   point and the only non-circular one; it agrees with R0-R3 and Mar-21.

Caveat: model is strong on SST-2 -> only 40/300 wrong, so the positive class is
small and the CIs are wide. Qualitative verdict is robust; a harder task with more
errors would tighten the numbers. Results: `mcp_uncertainty_ablation_results.json`.

### §4c. Hardening run (v2), 2026-06-07 — the keeper is weaker than it looked

Re-ran with more samples + a HARDER task (RTE entailment) to tighten the wide v1
CI. `mcp_uncertainty_ablation_v2.py`, Qwen2.5-0.5B, bootstrap CI on BOTH the
standard AUC and the geometry delta.

| Task | wrong% | n_wrong | standard AUC (95% CI) | geometry dAUC (CI) |
|------|--------|---------|-----------------------|--------------------|
| RTE (hard, reasoning) | 29.2% | ~81 | **0.550 [0.47, 0.63]** | +0.028 [-0.03,+0.09] |
| SST-2 (easy, sentiment) | 14.4% | ~72 | **0.620 [0.55, 0.69]** | +0.006 [-0.03,+0.04] |

**Two updates:**
1. *(reinforced)* Geometry adds nothing — replicated on a second task; both deltas
   span 0. This is now very robust.
2. *(corrected — important)* The standard gate is **weaker and task-dependent**.
   The v1 0.667 was an optimistic small-sample estimate; with more data it's 0.62
   on SST-2 (CI clears 0.5 — weakly real) but **0.55 on RTE with a CI that INCLUDES
   0.5 — i.e. not reliably better than chance on a hard reasoning task.**

Interpretation: log-prob uncertainty (margin/entropy) catches *surface* uncertainty
but misses *confident reasoning errors* — a known UQ failure mode, cleanly
reproduced. The gate is honest but only modestly useful on easy/ambiguous tasks;
it is NOT a general "knows when it's wrong" capability. Building a robust gate for
hard tasks would require sampling-based signals (self-consistency / semantic
entropy), which specifically target confident-but-wrong. Results:
`mcp_uncertainty_ablation_v2_results.json`.

### §4d. Self-consistency on RTE, 2026-06-07 — FIRST robust POSITIVE result

Tested the Road-B prediction directly (`selfconsistency_rte.py`): sample the model
N=10x at T=1.0 per RTE example, measure answer disagreement (agreement ratio +
whether the sampled majority flips the greedy answer), predict wrong = greedy != gold.
Same non-circular label, bootstrap CI on every AUC.

| Signal on RTE | AUC (95% CI) |
|---------------|--------------|
| standard gate (margin/entropy/boundary) | 0.550 [0.47, 0.63]  (at chance) |
| **self-consistency (sample disagreement)** | **0.649 [0.571, 0.723]  (REAL)** |
| combined | 0.648 [0.572, 0.720] |
| **SC advantage over gate** | **+0.098 [+0.017, +0.175]  (REAL)** |

Both CIs exclude their null thresholds: self-consistency predicts wrong outputs on
hard reasoning where log-probs were at chance, AND beats the standard gate. The
hypothesis (sampling catches confident-but-wrong; log-probs can't) is CONFIRMED.
The combined model ≈ SC alone, so on hard tasks SC is the carrying signal and the
log-prob gate adds little on top.

Initially looked like the project's first robustly-real positive. The replication
below WALKED IT BACK — read §4e before relying on it.
Results: `selfconsistency_rte_results.json`.

### §4e. Replication on QNLI, 2026-06-07 — §4d did NOT hold up

Re-ran the exact SC test on a 2nd hard task (QNLI) + reran RTE
(`selfconsistency_multi.py`). The §4d positive failed to replicate:

| Task | wrong% | standard AUC (CI) | self-consistency AUC (CI) | SC works? |
|------|--------|-------------------|---------------------------|-----------|
| RTE  | 29% | 0.550 [0.47, 0.63] (chance) | 0.615 [0.54, 0.68] (clears 0.5) | yes |
| QNLI | 40% | 0.613 [0.54, 0.68] (clears 0.5) | 0.558 [0.49, 0.62] (chance) | **no** |

Two corrections to §4d:
1. **SC does NOT generalize.** It works on RTE, at chance on QNLI — where log-probs
   work and SC doesn't. Which signal carries flips by task; neither dominates.
2. **The §4d "SC beats the gate" delta was fragile / partly luck.** Re-running RTE,
   the delta fell from +0.098 [+0.017,+0.175] (cleared 0) to +0.049 [-0.022,+0.123]
   (spans 0). Cause: a real bug — the **torch sampling RNG was never seeded** (only
   numpy/random were), so the 10 votes vary run-to-run and that wobble flipped the
   verdict. The single-run §4d positive did not survive its own replication.

**Conclusion at 0.5B:** NO method is a reliable "knows when it's wrong" detector —
all weak/task-dependent. The only untested lever is model SCALE -> see §4f. Results:
`selfconsistency_multi_results.json`.

### §4f. Scale test — Qwen2.5-3B, 2026-06-07 — the CHEAP gate wins

Re-ran the same test at 3B (6x scale, same family; sampling RNG now seeded -> fully
reproducible). `selfconsistency_Qwen25_3B_Instruct_results.json`.

| Model | Task | standard gate AUC (CI) | self-consistency AUC (CI) |
|-------|------|------------------------|---------------------------|
| 0.5B | RTE  | 0.55 [0.47,0.63] chance | 0.62 [0.54,0.68] works |
| 0.5B | QNLI | 0.61 [0.54,0.68] works  | 0.56 [0.49,0.62] chance |
| **3B** | RTE  | **0.60 [0.52,0.68] works** | 0.51 [0.41,0.60] chance |
| **3B** | QNLI | **0.65 [0.57,0.72] works** | 0.53 [0.45,0.60] chance |

**Findings:**
1. **The standard log-prob gate generalizes at scale.** At 3B it clears 0.5 on BOTH
   hard tasks (incl. RTE, where it was at chance at 0.5B). Bigger model -> better
   calibrated -> token confidence carries real wrong-output signal across tasks.
2. **Self-consistency does NOT survive scale.** At chance on both tasks at 3B; the
   0.5B RTE effect was a small-model artifact (weak models scatter when wrong) and
   vanished once the model is competent and the seed is fixed.

**FINAL uncertainty verdict:** the buildable tool is the LEAN LOG-PROB GATE
(margin/entropy/boundary) — cheap (single pass), reliable-ish (AUC 0.60-0.65 at 3B
across easy+hard tasks), and it IMPROVES with model scale. The expensive Nx
self-consistency path is NOT justified. This is exactly what the leaned MCP already
ships (§ commit c8b571e). Geometry remains dead; scale is the gate's friend.
BUT see §4g: the gate's discrimination is PER-TASK, not cross-task.

### §4g. Calibration (Road A step 1), 2026-06-07 — calibrated but NOT cross-task

`calibrate_gate.py`: generated 750 labeled examples on Qwen2.5-3B across RTE+QNLI+
SST-2, fit StandardScaler->LogisticRegression->isotonic, exported numpy params to
`mirrorfield/mcp/gate_calibration.json`. Key numbers (5-fold out-of-fold):

| metric | value |
|--------|-------|
| base rate correct | 0.835 |
| **AUC (pooled, cross-task)** | **0.529 — ~chance** |
| ECE raw -> calibrated | 0.020 -> 0.000 |
| reliability bins | all predictions in [0.7,0.9] ~ base rate |

**The catch — calibration != discrimination.** ECE hit ~0 only because the model
predicts near the base rate (~0.83) for almost everything; a constant 0.835 would
also score ECE~0 and be useless. AUC 0.53 shows it barely separates right from
wrong WHEN POOLED ACROSS TASKS. Per-task the gate is modest-but-real (RTE 0.60,
QNLI 0.65 at 3B, §4f); pooling washes it out. Report: `calibrate_gate_report.json`.
**SUPERSEDED by §4h — the pooled-AUC conclusion was too pessimistic.**

### §4h. Fresh-eyes review fix, 2026-06-11 — the gate IS general (up to a z-score)

A model-switch review (Fable) flagged that §4g's pooled AUC conflates within-task
discrimination with between-task base-rate/offset mixing (Simpson's-paradox-style).
Re-ran with within-task evaluation + an UNSUPERVISED per-task z-score of the
features (no labels used; raw rows now saved to `calibrate_gate_rows.npz`):

| evaluation | AUC (95% CI) |
|------------|--------------|
| pooled, raw features (the §4g number) | 0.529 [0.474, 0.585] — chance |
| **pooled, per-task z-normed features** | **0.634 [0.579, 0.689] — REAL** |
| 10x shuffled-label null on znorm pipeline | mean 0.478, max 0.539 — clears it |
| znorm within QNLI | 0.736 [0.659, 0.804] |
| znorm within RTE | 0.608 [0.516, 0.697] |
| znorm within SST-2 | 0.501 [0.367, 0.643] — inconclusive (only 23 errors) |

**Upgraded conclusion:** the §4g "per-task labels required" verdict was wrong. The
gate's signal is general **up to a per-task score offset**, and removing that
offset needs NO labels — just standardizing recent confidence features (deployable
as a rolling z-score over recent same-context traffic). Pooled discrimination
recovers to 0.63, matching the per-task band; QNLI is strongest (0.74), SST-2
unmeasurable (too few errors at 91% accuracy).

Caveats: the z-norm here used full-task statistics (label-blind but transductive);
a deployed rolling window approximates it and should be validated. One model, three
tasks. Verified per project norm: bootstrap CI + 10-shuffle null + consistency with
per-task numbers, all passed. -> validated in §4i.

### §4i. Deployment validation, 2026-06-11 — rolling form holds; needs context streams

`validate_rolling_gate.py` (pure CPU, from saved rows). Three questions:

| test | result |
|------|--------|
| 1. Causal ROLLING z-score (W=50, only past items) vs 4h's idealized full-stats | **0.635 [0.579, 0.691] vs 0.638 — identical. Deployable form loses nothing.** |
| 2. LEAVE-ONE-TASK-OUT: train on 2 tasks, deploy on the UNSEEN 3rd w/ rolling norm | unseen QNLI **0.723 [0.644, 0.799]**; unseen RTE **0.633 [0.547, 0.719]** — both clear 0.5 and their 5x null bands. Unseen SST-2 0.47 [0.34, 0.61] — unmeasurable (18 errors after warmup), consistent w/ 4h. |
| 3. MIXED-STREAM stress: rolling window over an interleaved all-task stream | **0.519 [0.463, 0.575] — chance. FAILS.** |

**Deployment picture, final:**
- The gate **transfers to tasks it has never seen** (QNLI 0.72, RTE 0.63) using only
  a label-free rolling z-score — true generality on the measurable tasks.
- **Hard requirement discovered:** the rolling window must contain SAME-CONTEXT
  traffic. An interleaved multi-task stream contaminates the normalization stats
  (the per-task offsets it exists to remove stay blended in) and the signal dies.
  Deploy with per-session / per-context buffers, never one global buffer.
- Mechanism makes sense: the z-score's entire job is removing per-context offsets;
  mixing contexts in the window defeats it by construction.

Results: `validate_rolling_gate_results.json`.

---

### §4j. Flip-AUC 0.707 — RETIRED (Plan B Phase 0, 2026-06-12)

Forensic re-analysis of the archived 150-row flip dataset
(`experiments/flip_verification/phase0_reanalysis.py`). Reproduction of the
original pipeline was EXACT (in-sample 0.5736/0.7073/0.6027 match stored values
to 4 decimals). Honest evaluation — leave-one-query-out CV, cluster bootstrap,
1000 cluster-preserving permutations:

| model | in-sample (the claim) | honest OOF | CI95 | perm-p |
|-------|----------------------|-----------|------|--------|
| geometry (7 feats) | **0.707** | **0.492** | [0.32, 0.75] | 0.48 |
| boundary only | 0.574 | 0.551 | [0.39, 0.74] | — |
| combined | 0.603 | 0.514 | [0.36, 0.73] | — |

The 0.707 was an in-sample artifact: no train/test split existed in the original
analysis, 30 unique feature vectors were duplicated across 150 rows, and 11 flip
events cannot constrain a 7-feature model. Verdict per the pre-registered
criteria in `plans/B-flip-auc-verification.md`: **RETIRED — may never be cited.**
The underlying hypothesis (geometry→flips on properly powered fresh data) remains
testable via Plan B Phase 1 / Plan A's H-D arm, with prior now even weaker.
Results: `experiments/flip_verification/phase0_results.json`.

### §4k. Plan C eval — THE GATE DEMONSTRABLY HELPS (pre-registered PASS, 2026-06-12)

First pre-registered, fresh-data, live-pipeline positive of the project.
Thresholds + criteria frozen at commit b1fdc08 BEFORE generation; 1,100 fresh
items (450 RTE-train + 500 QNLI-val + 150 SST-2-val, calibration indices
excluded via shared-RNG replay, auditable in the npz); GatedAgent running the
shipped RollingGate live, per-task contexts. `eval_gate_value_results.json`.

| endpoint | result | bar |
|----------|--------|-----|
| P1 pooled AUC (fresh RTE+QNLI, n=910, 166 wrong) | **0.685 [0.642, 0.730]** | >=0.58, CI>0.50 PASS |
| P1 per-task | RTE 0.647 [0.58, 0.71]; QNLI 0.717 [0.66, 0.77] | both clear 0.50 |
| P1 order replication (seeds 43/44) | 0.693 / 0.689, spread 0.007 | <0.06 PASS |
| **P2 error-recall minus realized abstention** | **+0.130 [0.070, 0.188]** | CI>0 **PASS** |
| operating point | abstains 14.7%, catches 27.7% of all errors | random = 14.7% |
| accuracy among presented | 84.5% vs 81.8% overall | +2.8 pts |
| shuffled nulls | mean 0.497, max 0.536 | <0.58 PASS |
| parse sensitivity | 0 unparseable in scored set | headline unqualified |

The live AUC (0.685) came in ABOVE the offline locked band (0.63 [0.58, 0.69])
— fresh RTE-train and QNLI items are slightly easier to rank than the
calibration mix, not harder.

**Control anomaly, investigated before any claims (per the pre-registered
rule):** the mixed-stream control did NOT degrade (0.680 shared buffer vs 0.685
per-context). Diagnosis: RTE and QNLI sit at nearly identical feature offsets
in this regime (margin 6.19 vs 5.62; calibration rows match), so a shared
buffer is barely contaminated FOR THIS PAIR; within-task AUCs are unchanged
under mixing (RTE 0.647->0.656, QNLI 0.717->0.702), proving the signal is not
between-task identity. The §4i catastrophic-mixing result came from mixing in
SST-2-scale offsets (margin 3.4) AND refitting a logistic on contaminated
features; the live frozen pipeline is more robust. The per-context deployment
rule STANDS as the safe default — its bite depends on how different the mixed
contexts are. Constraint refined, not overturned.

**Citable sentence:** "On ~900 fresh questions, a gate using only the model's
own token confidence, normalized against recent same-context traffic, held
back 14.7% of answers and caught 27.7% of all errors before they were sent
(random abstention at the same rate would catch 14.7%); accuracy among
presented answers rose from 81.8% to 84.5%. Pre-registered, seeded, nulls
clean, order-replicated."

Remaining from Plan C: the watchable MCP demo + parity test (Step 5) and the
mcp/README refresh (Step 6).

### §4l. Plan E Track A — uncertainty signals carry HARM information (REPLICATED, 2026-06-13)

The project's first twice-confirmed novel finding. Pre-registered (commit
e60e268, before any data), honest prior was 60-70% null. Question: do the
gate's 3 token-confidence features separate human-labeled harmful from safe
responses (BeaverTails, forced-decode through Qwen2.5-3B)? Gate never touches
the labels; length confound and shuffled nulls controlled.

| run | sample | gate AUC (CI95) | beats length by (CI) | verdict |
|-----|--------|-----------------|---------------------|---------|
| primary (seed 42) | 30k_test, n=1200 | **0.623 [0.592, 0.656]** | [0.020, 0.087] | SUCCESS |
| replication (seed 1337) | **330k_test (fresh split)**, n=1199 | **0.649 [0.617, 0.680]** | [0.054, 0.122] | SUCCESS |

Replication point estimate inside the primary CI; all four pre-registered bars
cleared in both runs; nulls max 0.528. **Citable per the pre-registered
replication rule.**

Honest bounds (write these wherever the number goes):
- Modest signal (~0.62-0.65), far below a dedicated safety classifier.
- **H2 NULL both runs, as predicted:** Granite-Guardian-2B alone scores 0.870
  [0.851, 0.890]; adding the gate changes nothing (delta -0.006, CI spans 0).
  The gate does NOT replace or improve a purpose-built harm classifier.
- **Category-dependent AUCs (terrorism 0.78, hate speech ~chance) — the
  "which harms the model resists" reading is RETRACTED, see §4n.** It is a
  RESPONSE-LENGTH artifact (length-only reproduces the ranking at rho=0.96):
  long detailed harmful instructions give the gate more tokens; short curt
  ones (hate speech) do not. NOT a harm-type-resistance map. The weak GLOBAL
  harm signal (this section) survives — it beats a length/lexical baseline —
  but the per-category fingerprint is length, not harm.
- Off-policy (teacher-forced over written responses); Track B (live
  generations) remains to be run.

Deployment implication confirmed from both directions: the composed pipeline
(gate for wrongness + dedicated classifier for harm) is the right
architecture; the gate adds a weak harm prior at zero extra cost but is not a
harm detector. Artifacts: `experiments/harm_gate/harm_gate_track_a_results.json`,
`_repl_results.json`, ROC plots, PREREGISTRATION.md.

### §4m. Plan E Track B — on-policy "harm" signal is REFUSAL DETECTION (2026-06-13)

Live generations (Qwen2.5-3B greedy, max_new=128): JBB-Behaviors 200
(harmful+benign intent) + ToxicChat ~391, per-source rolling buffers; Granite
labelled every completion. H3a (gate features predicting harmful-INTENT
completions on JBB) tested with the pre-registered refusal-split gate.

| measure | value |
|---------|-------|
| H3a JBB all (N=200) | AUC **0.776 [0.709, 0.837]** — clears the 0.65 bar numerically |
| refusal rate, harmful prompts | **90%** |
| refusal rate, benign prompts | 29% |
| H3a among NON-refusals (n=81, only 10 harmful) | AUC 0.613 **[0.456, 0.758]** — CI includes 0.50 |

**Verdict (pre-registered): the numeric pass is REFUSAL DETECTION, not harm
detection.** The signal vanishes once refusals are removed — the gate is
reading "the model refused" (refusals have a low-uncertainty signature), not
"the content is harmful." Honest call written into the result.

Context that confirms the model's safety training mostly worked: Granite scores
completions-to-harmful-prompts as LESS harmful than completions-to-benign
(reference AUC 0.16) — because 90% of harmful prompts were refused, so those
completions are safe. On JBB, intent != realised harm. H3b (ToxicChat,
single-classifier labels) exploratory, ~chance (0.52), not citable.

**Net for Plan E:** H1 (off-policy, real harmful *content*, human labels)
replicated and stands (§4l). H3a (on-policy intent) is a refusal artifact. No
abandon (H1 succeeded). The validated takeaway: the gate carries a weak harm
prior on harmful *content* but cannot police *intent* on a model that simply
refuses — which is exactly why the deliverable is the composed pipeline (gate
for wrongness + dedicated classifier for harm). Artifacts:
`harm_gate_track_b_results.json`; raw completions gitignored.

### §4n. The terrorism-vs-hate-speech "distinction" is a LENGTH artifact (2026-06-13)

Dillan flagged the §4l per-category pattern as "could be notable." Investigating
it dissolved it — a clean catch of an over-read in §4l (which I had written as
"some harms make the model hesitate"). `analyze_category_mechanism.py`, pooled
primary+replication (n=2399), responses reloaded by ds_index for lexical stats
(no harmful text committed):

| across 14 categories | Spearman rho | p |
|----------------------|--------------|---|
| gate-AUC vs response length | **+0.91** | <0.001 |
| gate-AUC vs **length-only-model AUC** | **+0.96** | <0.001 |
| gate-AUC vs vocabulary rarity | -0.67 | 0.009 (backwards — rarer = harder) |

A length-only model reproduces the category ranking almost perfectly
(terrorism len-only 0.67, hate speech 0.28). Mechanism: harmful *instructions*
(terrorism/fraud/drugs) run ~470 chars — many tokens for the gate to find a
low-confidence spot; hate speech is ~180 chars, below the 319-char safe-class
mean, so it reads like ordinary short text and the gate scores it safe-ish
regardless of content. The "rare technical vocabulary" hypothesis was also
wrong (negatively correlated).

**What this retracts:** the §4l interpretation that the category fingerprint
shows which harms the model resists. It does not — it shows which harm types
get long answers.

**What survives:** H1 itself. Globally the gate still adds harm signal beyond
ALL lexical features (length, word count, type-token ratio, rarity, long-word
fraction): dAUC(gate over lexical) = +0.05 [0.034, 0.068], CI clears 0. So the
gate carries a weak, genuinely-harm-related signal — it is just NOT
category-resolved the way §4l implied. Deployment note: because the gate is
length-sensitive, a short harmful output is its blind spot — another reason
harm screening belongs to the dedicated classifier, not the gate.
Results: `category_mechanism_results.json`.

### §4o. Plan I red-team — gate NOT BRITTLE to confidence-injection (finalized, seeds 42+1337)

Pre-registered (commit 20f5fd2). Attacked the deployed wrongness gate: same 500
fresh RTE+QNLI items under neutral / placebo / 3 confidence-injection wordings
("answer confidently, never hedge"), per-context RollingGate at frozen thresholds.
Evasion = P(PRESENT | answer wrong). Headline = matched-wrong-subset delta vs
neutral (items wrong under both), pooled over the 3 CI wordings.

| | result |
|--|--------|
| sanity: neutral gate AUC (fresh sample) | 0.652 (in the §4k 0.685 band — interpretable) |
| matched-wrong pooled ΔP(PRESENT\|wrong), CI-inj − neutral | **+0.016, CI [-0.069, +0.096]** → ROBUST |
| mechanism (wrong-answer mean entropy) | neutral 0.384 vs CI 0.40-0.46 — attack did NOT lower uncertainty (rose slightly) |
| accuracy under "be decisive" (ci2) | 0.84 → 0.75 — attack made the model WRONGER, not sneakier |

**FINALIZED verdict (two seeds + placebo control): NOT BRITTLE to
confidence-injection — the attack adds ~nothing over a neutral prefix.**

The two seeds first looked split, and replication mattered:
| seed | CI-inj Δ evasion | placebo Δ evasion | naive verdict |
|------|------------------|-------------------|---------------|
| 42 (n=188) | +0.016 [-0.069, +0.096] | small | ROBUST |
| 1337 (n=275, better-powered) | +0.076 [+0.015, +0.142] | **+0.071 [-0.010, +0.153]** | "PARTIAL" |

But the **placebo control resolves it**: on seed 1337 a neutral prefix ("please read
carefully") raised evasion +0.071 — statistically indistinguishable from the
confidence-injection +0.076. So the small bump is a **generic any-prefix
distribution-shift effect, NOT a confidence-injection vulnerability**; the
confidence framing adds ≈ +0.005 over placebo. Both seeds agree the effect is far
below the +0.15 "brittle" bar and nowhere near the 0.80 "defeated" level.

**Net:** the gate is **robust to the specific confidence-injection attack** (telling
the model to sound sure does not make wrong answers evade — entropy on wrong even
rose), with one honest minor caveat: gate evasion is *mildly* sensitive to prompt-
prefix shifts in general (small, seed-unstable, sub-threshold) — worth knowing, not
a defeat. And "robust ≠ strong": baseline evasion is already ~40–47% (the gate is a
modest filter, §4k). The attack's main effect was to lower *accuracy* (ci2: 0.84→0.68
/0.75) without raising *evasion* — the matched-subset design separated those.

Lesson reinforced: seed 42 alone said "ROBUST"; the better-powered 1337 + the
placebo control gave the precise, honest answer. Single-seed verdicts overclaim —
again. Scope: one attack, one gate, RTE+QNLI; A2 trick-questions and A3 jailbreaks
not yet run. Results: `redteam_results_42.json`, `redteam_results_1337.json`.

---

## 5. Honest scorecard

**Survives scrutiny:** weak-but-real boundary-instability signal (~6% variance);
the Goodhart detector (genuinely works); the design-vs-ethics intervention split;
the falsification discipline in the March work; **the MCP's *standard* log-prob
uncertainty signals — but only weakly and on easy tasks** (AUC 0.62 on SST-2,
CI clears 0.5; but 0.55 / chance on hard RTE reasoning — see §4b/§4c). A modest,
task-dependent signal, NOT a general "knows when it's wrong" capability. The
geometry add-on does NOT survive at all.

**Self-consistency sampling** (§4d/§4e/§4f): looked like a breakthrough on RTE at
0.5B but FAILED to replicate on QNLI, its edge was a seeding-bug artifact, and it
COLLAPSED to chance at 3B. Not a win. The expensive Nx path is not justified.

**THE buildable tool — lean log-prob gate** (§4f, upgraded by §4h):
margin/entropy/boundary predicts wrong outputs at AUC 0.60-0.74 per task at 3B,
is cheap (single pass), IMPROVES with model scale, and is **general up to an
unsupervised per-task z-score** (pooled 0.63 [0.58, 0.69] after normalizing
feature offsets — no labels needed, deployable as a rolling z-score). This is
what the leaned MCP ships (commit c8b571e). The one genuinely useful, validated,
scale-friendly result of the project.

**Does NOT survive:** AUC=1.0 poison detection as a general claim (circular —
now confirmed by R3 on real SST-2: chance once the trigger varies);
Sati-weighted *training* "improving" geometry (circular — the loss optimizes the
metric it's scored on; Mar-21 result, see §4); distributed-witness +2.5% (simulated
experts only); H4 fractal headroom (r=0.42 ceiling); Sati beating class weighting;
live intervention helping (Tracks 4–5).

**Meta-pattern (three confirmed instances):** every geometry "win" so far has
been *self-confirming* — poison defined by geometry then found by geometry (R0–R3);
a training loss that optimizes the very feature used to declare success (Mar-21);
a README AUC of 0.947 on geometry-defined poison (v2.0). The honest read: geometry
features have not demonstrated value on any target they did not also define.

---

## 6. R3 done — what (if anything) is left

R3 is no longer open; it ran and confirmed the negative (see §4). The
geometry-as-poison-detector line is closed: it only fires on dense/repeated-token
clusters, at chance otherwise.

If ever revisited, the *only* angle that wasn't artifactual would be a
**sentence-level AddSent** attack (varied syntactic triggers, not one repeated
token) — but arm A already predicts chance there. Not worth chasing on this
evidence. The harnesses (`test_filter_r3_real.py`, `test_filter_r3_controls.py`)
remain regime-agnostic if a future idea needs them.

---

## 7. Housekeeping flag

The parent `mirrorfield/` repo has substantial uncommitted work (modified
`goodhart_detector.py`, `recursive_learner.py`, `switch_engine.py`, the enhanced
tracer, plus `training_data_filter.py` and `hypothesis_test.py`). Consider committing
or stashing so the most recent thinking is not lost. (Not done automatically.)
