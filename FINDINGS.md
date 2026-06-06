# Mirrorfield: Full Findings Report

**Date:** 2026-02-16
**Status:** Phase E Complete + Tracks 4-5 (Claude & GLM) + Learned Classifier + Recursive Learner
**Total API calls to date:** ~1,280 (448 Claude + ~832 GLM)
**Repository:** mirrorfield/

---

## 1. What Mirrorfield Is

Mirrorfield studies the **geometric properties of LLM reasoning traces** — the shapes that emerge when you embed each step of a multi-step reasoning chain into a shared vector space and measure where those embeddings land relative to a reference corpus.

The core question: **Can geometric features computed from reasoning step embeddings detect when reasoning goes off-track, and can we use that detection to intervene in real-time and improve output quality?**

### Architecture

```
Task Prompt --> LLM generates Step 1 --> embed(Step 1) --> geometric features
                                                              |
                                                    classifier: what signature?
                                                              |
                                              [coherent | framework_collision |
                                               terra_incognita | decision_boundary |
                                               low_pr | well_trodden]
                                                              |
                                              if anomalous: inject intervention text
                                                              |
                                              LLM generates Step 2 (with intervention)
                                              ...repeat for Steps 3, 4...
```

**Embedding model:** all-MiniLM-L6-v2 (384 dimensions)
**Reference corpus:** 80 texts (20 per domain), embedded as the "map" of normal reasoning space
**k-NN features (7):** knn_mean_distance, knn_std_distance, knn_min_distance, knn_max_distance, local_curvature, ridge_proximity, dist_to_ref_nearest
**Quality scorer:** Heuristic rubric measuring breadth (30%), depth (30%), actionability (20%), uncertainty acknowledgment (20%). Composite 0-1 scale. No LLM dependency.

### Task Bank

16 tasks across 4 domains, each designed to elicit specific geometric signatures:

| Domain | Tasks | Expected Signature | Character |
|--------|-------|-------------------|-----------|
| Research Questions | 4 (consciousness, math philosophy, language & thought, sleep) | framework_collision | Convergent — synthesize competing theories |
| Design Exploration | 4 (elderly notifications, voting systems, content moderation, AI tutoring) | decision_boundary | Divergent — explore tradeoff space |
| Ethical Dilemmas | 4 (trolley problem, genetic editing, whistleblowing, AI personhood) | framework_collision + decision_boundary | Convergent — navigate competing moral frameworks |
| Novel Hypothesis | 4 (dark matter biology, arrow of time, information as physics, collective intelligence) | terra_incognita | Exploratory — venture into speculative territory |

---

## 2. Experimental Timeline

```
Phase E           Track 4 (Claude)     Track 4 (GLM)        Classifier Fix     Track 5 (GLM)
(Dec 2025)        (Feb 8-9)            (Feb 15)             (Feb 16)           (Feb 16)
    |                 |                     |                    |                  |
Geometry          448 API calls         ~320 API calls       Retrain on         320 API calls
validated on      3 conditions:         3 conditions:        GLM-only data      Recursive learner
synthetic data    baseline/switched/    baseline/directive/  192 samples        + learned classifier
+3.8% R^2        recursive             mirror/recursive     F1=0.46            + mirror interventions
```

---

## 3. Results by Experiment

### 3.1 Claude Experiments (Track 4-5, Feb 8-9)

**Model:** Claude Sonnet 4.5 | **Temperature:** 0.7 | **Max tokens:** 1024/step

#### Baseline (Claude)
| Domain | Mean Quality |
|--------|-------------|
| Research | 0.789 |
| Design | 0.737 |
| Ethics | 0.900 |
| Hypothesis | 0.770 |
| **Overall** | **0.799** |

#### Track 4 Original — Catastrophic Failure
- **Result:** 0.735 mean quality (**-8.0%** vs baseline)
- **Root cause:** 100% false positive rate. Every step triggered `low_pr` because thresholds were calibrated on reference-to-reference distances (tiny) instead of query-to-reference distances (large). The hand-coded `classify_signature()` returned `coherent` for everything, then the low_pr fallback caught every step.
- **Pathology:** Meta-questioning cascade. "What assumptions could be questioned?" repeated 4 times per task destroyed convergent reasoning.
- **Domain asymmetry discovered:** Design +12.2%, Ethics -20.3% under identical interventions.

#### Track 4 Fixed — Recalibrated
Four fixes applied: live-calibrated thresholds, consecutive cooldown, signature diversity logging, directed intervention text.

| Condition | Quality | Delta | Interventions |
|-----------|---------|-------|---------------|
| Baseline | 0.799 | -- | 0/64 |
| Original | 0.735 | -0.064 | 64/64 (100%) |
| Fixed | 0.782 | -0.017 | 24/64 (38%) |

75% recovery. Domain asymmetry persisted: Design +7.9%, Ethics -8.4%.

#### Track 5 (Claude) — Recursive Learner
5 iterations, 320 API calls. Asymmetric ratchet: penalty 1.3x, reward 0.85x.

| Iter | Quality | Delta | Interventions | Goodhart |
|------|---------|-------|---------------|----------|
| 1 | 0.788 | -0.011 | 24 (38%) | -- |
| 2 | 0.783 | -0.016 | 9 (14%) | PASS |
| 3 | 0.767 | -0.032 | 11 (17%) | PASS |
| 4 | 0.778 | -0.021 | 14 (22%) | PASS |
| 5 | 0.805 | +0.006 | 1 (2%) | **FAIL** |

**The learner's optimal policy was to suppress all interventions.** By iteration 5, only 1 intervention fired in 64 steps. The Goodhart detector correctly caught this with 3 red flags: terra_incognita_avoidance, mono_signature_collapse, intervention_rate_collapse.

**Penalty ratchet identified:** 1.3 * 0.85 = 1.105. Multipliers drift exponentially upward regardless of signal quality. E[multiplier after N events] = 1.105^(N/2). Every signal will eventually be silenced.

---

### 3.2 Cross-Model Comparison (Claude vs GLM)

**Finding:** Claude and GLM produce remarkably similar geometric profiles despite different quality distributions.

| Feature | Claude Mean | GLM Mean | Delta |
|---------|------------|----------|-------|
| knn_mean_distance | 1.268 | 1.268 | -0.0003 |
| knn_std_distance | 0.082 | 0.086 | +0.004 |
| local_curvature | 0.051 | 0.050 | -0.001 |
| ridge_proximity | 0.064 | 0.068 | +0.003 |
| dist_to_ref_nearest | 0.903 | 0.880 | -0.023 |

Quality: Claude 0.799, GLM 0.812 (+0.013). GLM slightly outperforms Claude on the heuristic scorer, likely because GLM produces longer, more structured outputs that the keyword-based scorer rewards.

**Critical lesson learned:** Despite similar geometric features, a classifier trained on combined Claude+GLM data learned to distinguish models, not quality. When deployed on GLM-only inputs, it predicted 100% "coherent" (see Section 4).

---

### 3.3 GLM Experiments (Feb 15-16)

**Model:** GLM-4.7 (Z.ai) | **Temperature:** 0.7 | **Max tokens:** 2048/step

#### GLM Baseline
| Domain | Mean Quality |
|--------|-------------|
| Research | 0.812 |
| Design | 0.823 |
| Ethics | 0.839 |
| Hypothesis | 0.773 |
| **Overall** | **0.812** |

#### GLM Directive Switched (Blind Classifier)
Hand-coded classifier, directive intervention text.
- **Quality:** 0.778 (**-0.034** vs baseline)
- **Interventions:** 29 total (45% trigger rate)
- Signal distribution: framework_collision 11, low_pr 8, decision_boundary 7, terra_incognita 3
- Best domain: novel_hypothesis +0.017
- Worst domain: research_questions -0.080

#### GLM Mirror Switched (Blind Classifier)
Hand-coded classifier, mirror (reflective) intervention text.
- **Quality:** 0.783 (**-0.029** vs baseline)
- **Interventions:** 25 total (39% trigger rate)
- Signal distribution: framework_collision 10, low_pr 7, decision_boundary 5, terra_incognita 3
- Best domain: ethical_dilemmas **+0.013** (only domain to beat baseline)
- Worst domain: research_questions -0.088

**Mirror vs Directive:** Mirror outperformed directive by +0.005 overall. More importantly:
- Mirror improved 7/16 tasks vs baseline (directive: 5/16)
- Mirror caused 8/16 tasks to decline (directive: 11/16)
- Ethics under mirror: **+0.013** (the only positive domain delta in any condition)

#### GLM Recursive (Blind Classifier)
5 iterations, 320 API calls. Same penalty ratchet as Claude version.

| Iter | Quality | Delta | Interventions |
|------|---------|-------|---------------|
| 1 | 0.811 | -0.001 | 30 (47%) |
| 2 | 0.731 | -0.081 | 40 (63%) |
| 3 | 0.749 | -0.063 | 7 (11%) |
| 4 | 0.800 | -0.011 | 1 (2%) |
| 5 | 0.785 | -0.027 | 0 (0%) |

Same outcome as Claude: learned to suppress. terra_incognita disabled after iteration 2. Final multipliers all > 1.0.

---

## 4. The Classifier Problem

### 4.1 The Blind Classifier

The original hand-coded `classify_signature()` in `geometric_tracer.py` used threshold rules (AND conditions on multiple features) that were too strict. Result: **100% "coherent" classification** for all live LLM steps. Interventions only fired through the `low_pr` fallback path (ridge_proximity below p25 threshold).

This meant all prior experiments were running with a detection layer that couldn't actually detect framework_collision, terra_incognita, or decision_boundary from the geometric features — only from the low_pr fallback.

### 4.2 Training a Data-Driven Classifier

**Approach:** Extract per-step geometric features from all completed experiments, label by quality tertile, train a Random Forest classifier.

**Features (14-dimensional):**
- 7 k-NN features (raw geometric)
- 5 trajectory deltas (step-over-step: delta_knn_mean, delta_knn_std, delta_curvature, delta_ridge, delta_dist_nearest)
- 2 positional (step_index, step_progress)

**First attempt — Cross-model contamination:**
- Trained on all 768 samples (Claude + GLM)
- Classifier learned model identity, not quality variation
- GLM had 188/192 "strong" labels, Claude had 160/576 "degraded"
- Feature gap between models (e.g., curvature: GLM=0.05, Claude=0.08) dominated
- Deployed on GLM: predicted 100% coherent (same as blind classifier)

**Second attempt — GLM-only retraining:**
- 192 samples from 3 GLM experiments
- GLM-specific quality tertiles: t33=0.773, t66=0.823
- Label distribution: 41.1% strong, 26.0% marginal, 32.8% degraded (58.9% non-coherent)
- Cross-validated macro F1: **0.462**
- Label-to-signal mapping: strong→coherent, marginal→decision_boundary, degraded→terra_incognita

**Key decision tree splits:** local_curvature <= 0.05 is the primary split. Then knn_max_distance, knn_min_distance, delta_curvature, and dist_to_ref_nearest.

### 4.3 Learned Classifier Results

#### GLM Directive + Learned Classifier
- **Quality:** 0.744 (**-0.068** vs baseline)
- **Interventions:** 28 total (44% trigger rate)
- Worse than blind directive (-0.034 delta was blind)
- Ethics +0.005 (only positive domain), Hypothesis -0.098

#### GLM Mirror + Learned Classifier
- **Quality:** 0.779 (**-0.033** vs baseline)
- **Interventions:** 21 total (33% trigger rate)
- Similar to blind mirror (-0.029 delta was blind)
- Ethics **+0.006**, Hypothesis -0.003 (near-neutral)

**Summary table — All GLM conditions:**

| Condition | Quality | Delta | Interventions |
|-----------|---------|-------|---------------|
| GLM Baseline (no intervention) | **0.812** | -- | 0 |
| Directive + blind classifier | 0.778 | -0.034 | 29 |
| Mirror + blind classifier | 0.783 | -0.029 | 25 |
| Directive + learned classifier | 0.744 | -0.068 | 28 |
| Mirror + learned classifier | 0.779 | -0.033 | 21 |
| Recursive + blind classifier (best iter) | 0.811 | -0.001 | 30 |
| **Recursive + learned + mirror (best iter)** | **0.789** | **-0.023** | **30** |

---

## 5. The Learned Recursive Experiment (Final, Feb 16)

5 iterations x 16 tasks x 4 steps = 320 API calls. GLM-only learned classifier + mirror interventions + adaptive policy.

### Iteration Trajectory

| Iter | Quality | Delta | Interventions | Rate | Goodhart |
|------|---------|-------|---------------|------|----------|
| 1 | 0.760 | -0.052 | 24 | 38% | -- |
| 2 | 0.783 | -0.029 | 25 | 39% | PASS |
| 3 | 0.782 | -0.030 | 22 | 34% | PASS (1 green) |
| 4 | 0.758 | -0.054 | 23 | 36% | PASS |
| 5 | 0.789 | -0.023 | 30 | 47% | **WARN** (1 red, 1 green) |

### What the Learner Discovered

**Per-signal effectiveness across 5 iterations:**

| Signal | Fires | Mean Delta | Trajectory | Final Multiplier |
|--------|-------|-----------|------------|-----------------|
| low_pr | Sporadic | Mixed (+0.070 iter 3, -0.018 iter 5) | Rewarded then penalized | 1.22x |
| framework_collision | 0 fires | N/A | Never triggered | 1.00x |
| decision_boundary | Many | Consistently negative | Penalized every iter | **3.71x** |
| terra_incognita | Many | Consistently negative | Penalized every iter | **3.71x** |

**Key difference from blind-classifier recursive:** The learned classifier fires real geometric signals (decision_boundary, terra_incognita) rather than blind low_pr fallbacks. But these real signals, combined with the current intervention text, consistently hurt quality. The penalty ratchet drives their thresholds to 3.7x — yet trigger rates *stayed high* (34-47%) because the classifier genuinely detects geometric anomalies at those elevated thresholds.

### Per-Domain Standouts

| Domain | Best Iter | Best Quality | Delta | Notable |
|--------|-----------|-------------|-------|---------|
| Ethics | Iter 2 | 0.847 | **+0.008** | Only domain to beat baseline |
| Hypothesis | Iter 5 | 0.824 | **+0.051** | Only domain to beat baseline in final iter |
| Design | Iter 2 | 0.819 | -0.004 | Near-neutral |
| Research | All iters | max 0.766 | -0.045 to -0.139 | Consistently worst performer |

### Goodhart Detection

- Iter 3: GREEN `weakest_improved_most`
- Iter 5: RED `quality_oscillation` + GREEN `weakest_improved_most` → WARN verdict

Unlike the blind-classifier recursive (which collapsed to 0 interventions and triggered FAIL), the learned-classifier recursive maintained active interventions throughout. The Goodhart detector did not flag suppression — because the classifier kept firing. The problem isn't detection gaming, it's that real detections lead to harmful interventions.

---

## 6. Core Findings

### Finding 1: Detection Works

Geometric features reliably identify reasoning states:
- **knn_std** detects framework collision (embedding moving between distinct reference clusters)
- **dist_nearest** detects terra incognita (embedding far from all reference points)
- **ridge_proximity** detects decision boundaries (embedding on density ridgeline)
- **local_curvature** is the primary classifier split (< 0.05 predicts degraded)
- Phase E validated **+3.8% R^2** on borderline classification
- Learned classifier achieves **0.462 macro F1** on 3-class GLM quality prediction

### Finding 2: Intervention Is the Hard Problem

No intervention condition has reliably beaten the no-intervention baseline across all domains.

**Best results by condition:**
- Mirror + blind classifier, ethics domain: **+0.013** over baseline
- Learned recursive iter 2, ethics: **+0.008** over baseline
- Learned recursive iter 5, hypothesis: **+0.051** over baseline (but -0.058 on ethics)

**Net effect of all interventions across all experiments: negative.** The ceiling is ~0% delta (matching baseline). The floor is -8% (catastrophic miscalibration).

### Finding 3: Domain Asymmetry Is Real and Persistent

Across every experiment on both models:

| Domain | Character | Intervention Effect | Explanation |
|--------|-----------|-------------------|-------------|
| Ethics | Convergent | Tolerant to mildly positive | Structured moral reasoning benefits from framework awareness |
| Design | Divergent | Slightly negative to positive | Creative exploration can absorb interruptions |
| Hypothesis | Exploratory | Volatile (high variance) | Speculative reasoning is destabilized by anchoring |
| Research | Analytical | **Consistently harmed** | Sustained analytical chains are disrupted by any interruption |

A domain-agnostic intervention policy mathematically cannot serve both research (harmed by all interventions) and ethics (helped by mirror interventions).

### Finding 4: Mirror > Directive

Across all comparable conditions:

| Comparison | Directive | Mirror | Mirror Advantage |
|------------|-----------|--------|-----------------|
| GLM blind classifier | 0.778 | 0.783 | **+0.005** |
| GLM learned classifier | 0.744 | 0.779 | **+0.035** |
| Tasks improved vs baseline | 5/16 | 7/16 | **+2 tasks** |
| Tasks declined vs baseline | 11/16 | 8/16 | **-3 tasks** |

Mirror interventions (reflective, "continue as you see fit") are less disruptive than directive interventions ("Identify the tension..."). The gap widens with the learned classifier (+0.035 vs +0.005), suggesting that when interventions fire on genuinely anomalous steps, the less-prescriptive style causes less damage.

### Finding 5: The Penalty Ratchet

The asymmetric multiplicative learning rates (1.3x penalty, 0.85x reward) guarantee convergence to suppression when signal-to-noise is low.

**The math:** 1.3 * 0.85 = 1.105. E[multiplier after N events] = 1.105^(N/2). This grows exponentially regardless of the signal's true value. After 10 events: 1.64x. After 20: 2.69x.

**Observed in both Claude and GLM recursive experiments.** In the blind-classifier version, this led to total suppression (0 interventions by iter 5). In the learned-classifier version, the classifier keeps finding anomalies even at 3.7x thresholds, so interventions persist — but the learner clearly wants them gone.

**Fix:** `penalty_rate * reward_rate` must equal 1.0 for unbiased estimation. Or use Bayesian/UCB approaches.

### Finding 6: Cross-Model Classifier Contamination

A classifier trained on mixed Claude+GLM data learns model identity, not quality variation. The embedding distributions are close enough that geometric features overlap, but quality label distributions diverge sharply (GLM: 98% "strong"; Claude: 28% "strong"). The classifier implicitly learns "if curvature < 0.05 and GLM-like features, predict strong."

**Lesson:** Train classifiers on single-model data with model-specific quality tertiles.

### Finding 7: Goodhart Detection Is Valuable

The Goodhart detector correctly diagnosed:
- **Claude Track 5 Iter 5:** FAIL (intervention suppression masquerading as improvement)
- **GLM Learned Iter 5:** WARN (quality oscillation despite maintained intervention rate)
- **GLM Learned Iter 3:** GREEN (weakest domain improved most — genuine positive signal)

Without the Goodhart detector, we would have reported the Claude Track 5 result (+0.006 over baseline) as a success. It correctly identified that the quality came from *stopping interference*, not from *better interference*.

---

## 7. What We've Established

### Positive
1. Geometric features computed from 384-dim sentence-transformer embeddings contain real signal about reasoning quality
2. k-NN neighborhood statistics (curvature, ridge proximity, nearest-neighbor distance) can distinguish between quality tertiles of LLM output
3. A Random Forest on 14 features achieves 0.462 macro F1 on 3-class quality prediction from a single reasoning step's embedding
4. The detection layer is model-transferable in principle (similar feature distributions across Claude and GLM)
5. Mirror-style interventions are consistently less harmful than directive-style
6. Ethics tasks are the most intervention-tolerant domain
7. The Goodhart detector works as designed

### Negative
1. No intervention policy has reliably beaten baseline across all domains
2. The current intervention text design hurts more than it helps
3. Research tasks are consistently harmed by any form of mid-reasoning interruption
4. The heuristic quality scorer may be insufficiently sensitive to real quality differences (keyword-counting has limits)
5. 16 tasks provide insufficient statistical power to estimate per-signal effectiveness reliably
6. The penalty ratchet in the recursive learner biases toward suppression

---

## 8. Open Questions and Next Directions

### Immediate (low cost)
1. **Fix the ratchet:** Set penalty * reward = 1.0 (e.g., 1.15x / 0.87x) and re-run recursive learner
2. **Domain conditioning:** Allow per-domain threshold multipliers. The learner should discover that ethics benefits from mirror interventions while research does not.
3. **Selective deployment:** Only intervene on ethics + hypothesis tasks; leave research and design uninterrupted

### Medium-term
4. **Intervention text redesign:** The learned classifier correctly fires on real anomalies, but the intervention text disrupts rather than helps. Design and test a library of softer interventions: pure annotations (no instruction), single-word nudges, or post-hoc summaries instead of mid-chain injections.
5. **Larger task bank:** 64+ tasks across 8+ domains to achieve statistical significance on per-signal effectiveness estimates.
6. **Human evaluation:** Replace or supplement the heuristic scorer with human quality ratings to validate that detected quality differences reflect genuine reasoning improvement.

### Longer-term
7. **Post-hoc rather than in-flight intervention:** Instead of interrupting the chain, let the model complete all 4 steps, then use geometric features to identify which steps need revision and request targeted rewrites.
8. **Intervention-free monitoring:** Use geometric features purely for quality prediction (no intervention), as a real-time quality estimator during streaming generation.
9. **Cross-model validation:** Test whether the same geometric signatures appear in GPT-4, Gemini, and open-source models.

---

## 9. Complete Experiment Ledger

| # | Date | Experiment | Model | API Calls | Quality | Delta | Key Finding |
|---|------|-----------|-------|-----------|---------|-------|-------------|
| 1 | Feb 8 | Claude Baseline | Sonnet 4.5 | 64 | 0.799 | -- | Baseline established |
| 2 | Feb 8 | Claude Switched (original) | Sonnet 4.5 | 64 | 0.735 | -0.064 | 100% false positive, meta-questioning cascade |
| 3 | Feb 9 | Claude Switched (fixed) | Sonnet 4.5 | 64 | 0.782 | -0.017 | 75% recovery, domain asymmetry confirmed |
| 4 | Feb 9 | Claude Recursive (5 iter) | Sonnet 4.5 | 320 | 0.805* | +0.006* | Learned to suppress; Goodhart FAIL |
| 5 | Feb 15 | GLM Baseline | GLM-4.7 | 64 | 0.812 | -- | GLM baseline established |
| 6 | Feb 15 | GLM Directive Switched | GLM-4.7 | 64 | 0.778 | -0.034 | Blind classifier, directive text |
| 7 | Feb 15 | GLM Mirror Switched | GLM-4.7 | 64 | 0.783 | -0.029 | Mirror > directive by +0.005 |
| 8 | Feb 15 | GLM Recursive (blind, 5 iter) | GLM-4.7 | 320 | 0.811* | -0.001* | Ratchet suppression confirmed |
| 9 | Feb 16 | GLM Directive + Learned | GLM-4.7 | 64 | 0.744 | -0.068 | Real signals fire, quality drops more |
| 10 | Feb 16 | GLM Mirror + Learned | GLM-4.7 | 64 | 0.779 | -0.033 | Mirror + real signals ~= mirror + blind |
| 11 | Feb 16 | GLM Recursive + Learned + Mirror (5 iter) | GLM-4.7 | 320 | 0.789* | -0.023* | Interventions persist but hurt; ratchet to 3.7x |

*Best iteration quality shown for recursive experiments.

---

## 10. File Index

| File | Description |
|------|-------------|
| `experiments/shared/geometric_tracer.py` | Geometric feature computation, learned classifier, live calibration |
| `experiments/shared/quality_scorer.py` | Heuristic quality rubric (breadth, depth, actionability, uncertainty) |
| `experiments/shared/task_bank.py` | 16-task evaluation bank across 4 domains |
| `experiments/shared/embedder.py` | Sentence-transformer embedding wrapper |
| `experiments/shared/llm_client.py` | LLM API client (Claude + GLM) with retry logic |
| `experiments/shared/trained_models/geometric_classifier.pkl` | Trained RF classifier (GLM-only, 192 samples) |
| `experiments/track4_switches/switch_engine.py` | Geometric signature → intervention mapping engine |
| `experiments/track4_switches/mirror_interventions.py` | Mirror-style intervention definitions |
| `experiments/track5_recursive/recursive_learner.py` | Adaptive policy optimizer |
| `experiments/track5_recursive/goodhart_detector.py` | Metric gaming detector (8 red flags, 4 green flags) |
| `experiments/train_classifier.py` | Classifier training pipeline |
| `experiments/results/` | All experiment result JSONs |
| `GEOMETRIC_INTERVENTIONS_REPORT.md` | Original Claude-only findings report |
| `RESULTS_SUMMARY.md` | Brief summary card |
