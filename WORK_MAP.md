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

---

## 5. Honest scorecard

**Survives scrutiny:** weak-but-real boundary-instability signal (~6% variance);
the Goodhart detector (genuinely works); the design-vs-ethics intervention split;
the falsification discipline in the March work; **the MCP's *standard* log-prob
uncertainty signals** (margin+entropy predict wrong outputs at AUC 0.667 on a
non-circular gold-correctness label — see §4b). The geometry add-on does NOT survive.

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
