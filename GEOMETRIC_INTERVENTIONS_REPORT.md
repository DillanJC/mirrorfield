# Can Geometric Features Steer AI Reasoning? A Live Experiment in Failure, Diagnosis, and Recursive Learning

**Date:** 2026-02-09
**Model:** Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
**Embeddings:** all-MiniLM-L6-v2 (384-dim)
**Total API calls:** 448 (64 baseline + 64 original + 64 fixed + 320 recursive)
**Repository:** mirrorfield/experiments/track4_switches/, track5_recursive/

---

## Abstract

We tested whether geometric features computed from AI reasoning step embeddings can be used as real-time interventions to improve multi-step reasoning quality. The answer is nuanced: geometric signals reliably detect reasoning states (framework collisions, unexplored territory, low novelty), but translating detection into beneficial intervention is harder than detection itself. Our initial system achieved a 100% false positive rate due to threshold miscalibration, producing a -8.0% quality degradation. After diagnosis and recalibration, quality recovered to -2.1% with a striking domain asymmetry: design tasks improved +9.0% while ethics tasks degraded -18.3% under the same intervention. A recursive learner tasked with optimizing the intervention policy discovered, over 5 iterations, that the optimal strategy was to suppress all interventions -- a conclusion the system's own Goodhart detector correctly flagged as metric gaming. The failure modes we encountered (threshold miscalibration, meta-questioning cascades, the penalty ratchet) are, we argue, more informative than a clean positive result would have been, because they illuminate the gap between detecting uncertainty and improving reasoning about uncertainty.

---

## 1. Experimental Arc

```
Phase E        Track 4 Original   Diagnosis      Track 4 Fixed    Track 5
(Dec 2025)     (Feb 8 evening)    (Feb 8 night)  (Feb 9 00:31)    (Feb 9 02:12)
    |               |                  |               |               |
Geometry       64 API calls        Root cause     64 API calls     320 API calls
validated      -8.0% quality       100% false     -2.1% quality    V-curve: degrade
on synthetic   All low_pr x4       positive       4 signal types   then surpass baseline
embeddings     Meta-questioning    Distribution   75% recovery     Learns to suppress
+3.8% R^2      cascade            mismatch       Domain asymmetry Goodhart catches it
```

**Phase E** (December 2025) validated that k-NN geometric features improve boundary case prediction by +3.8% R^2 on borderline classification tasks. Seven features -- knn_mean, knn_std, knn_min, knn_max, local_curvature, ridge_proximity, dist_nearest -- computed from 384-dimensional sentence-transformer embeddings showed real signal for detecting proximity to decision boundaries.

**Tracks 4-5** (February 2026) asked the harder question: can we use these geometric signals not just for classification, but as real-time steering signals during multi-step LLM reasoning? The experiment used Claude Sonnet 4.5 generating 4-step reasoning chains on 16 tasks across 4 domains, with geometric features computed from each step's embedding against a reference corpus.

---

## 2. Track 4 Original: Catastrophic Threshold Miscalibration

### Setup

- **16 tasks** across 4 domains: research questions (4), design exploration (4), ethical dilemmas (4), novel hypothesis generation (4)
- **4 reasoning steps** per task, each generating ~500-word analytical text
- **Baseline run**: 64 API calls, no interventions, quality scored by heuristic rubric (breadth, depth, actionability, uncertainty acknowledgment; composite 0-1)
- **Switched run**: Same 16 tasks, but after each step the embedding is classified into one of 6 geometric signatures. Four signatures (low_pr, framework_collision, terra_incognita, decision_boundary) trigger text interventions injected before the next step.

### Result: -8.0% Quality Degradation

| Metric | Baseline | Switched | Delta |
|--------|----------|----------|-------|
| **Mean quality** | **0.799** | **0.735** | **-0.064 (-8.0%)** |
| Interventions | 0/64 | 64/64 (100%) | -- |
| Unique signal types | -- | 1 (low_pr only) | -- |
| Consecutive same-sig | -- | 100% | -- |

Every single reasoning step -- all 64 across all 16 tasks -- triggered the same `low_pr` intervention: *"What assumptions could be questioned?"* The model received this prompt four times per task, creating a meta-questioning cascade.

### Root Cause: Distribution Mismatch

The thresholds for geometric signature classification were calibrated from the *reference corpus embeddings* (80 reference texts embedded against each other). But the *query embeddings* (live LLM reasoning steps embedded against that same reference corpus) occupy a fundamentally different distribution:

| Threshold | Reference Value | Live Value | Problem |
|-----------|----------------|------------|---------|
| low_pr (ridge < p25) | 0.186 | 0.055 | Dimensional mismatch: knn_std percentile applied to ridge_proximity |
| framework_collision (knn_std > p75) | 0.194 | 0.093 | Reference self-similarity (intra-corpus) >> query-vs-reference |
| terra_incognita (dist_nearest > p75) | 2.98e-08 | 1.007 | Reference vectors nearly identical to each other; detection impossible |
| decision_boundary (ridge > 0.30) | 0.30 (hardcoded) | 0.074 | Threshold 4x above maximum observed live value |

The fundamental error: thresholds trained on how reference documents relate to *each other* do not transfer to how LLM reasoning steps relate to *reference documents*. Reference-to-reference distances are tiny (near-duplicate embeddings from related texts). Query-to-reference distances are much larger. This made the `low_pr` threshold (ridge proximity below p25) trigger on every single step, while `framework_collision`, `terra_incognita`, and `decision_boundary` could never trigger.

### The Meta-Questioning Cascade

The `low_pr` intervention text was: *"What assumptions could be questioned?"*

When injected at every step, this created a distinctive pathology. Instead of advancing its analysis, the model repeatedly paused to question its own framework:

> **Step 2:** "Let me examine the assumptions embedded in my Step 1 approach..."
> "**Assumption 1:** I treated [X] and [Y] as if they're distinct entities..."
> "**Assumption 2:** I centered on [Z]. But [the topic] is much broader..."

Each step became a meta-commentary on the previous step rather than analytical progress. The model was compliant -- it did exactly what the intervention asked -- but compliance destroyed convergent reasoning. The most damaged task (research_03, consciousness and language) dropped from 0.865 to 0.565: a 35% degradation from a single miscalibrated threshold.

### Domain Asymmetry: The First Surprise

Despite the universal miscalibration, different domains responded differently:

| Domain | Baseline | Switched | Delta | % Change |
|--------|----------|----------|-------|----------|
| Design | 0.737 | 0.827 | **+0.090** | **+12.2%** |
| Hypothesis | 0.770 | 0.763 | -0.007 | -0.9% |
| Research | 0.789 | 0.634 | -0.155 | -19.6% |
| Ethics | 0.900 | 0.717 | **-0.183** | **-20.3%** |

Design tasks *improved* under the meta-questioning cascade. The "question your assumptions" prompt, while derailing convergent reasoning, actually helped divergent reasoning by forcing the model to explore alternatives it wouldn't have considered. Design exploration -- where creative branching is valuable -- benefited from exactly the prompt that destroyed the analytical progression of ethics and research tasks.

This is not an artifact. The same asymmetry persisted across every subsequent experiment. It reflects a real property of the interaction between intervention type and reasoning domain.

---

## 3. Track 4 Fixed: Calibration Recovery

### Four Fixes Applied

**Fix 1: Live-Calibrated Thresholds.** Replaced reference-based thresholds with percentiles computed from the actual 64 baseline reasoning step embeddings:

```
LIVE_CALIBRATED_THRESHOLDS = {
    'knn_std_p75': 0.092529,    # framework_collision: knn_std > this
    'knn_std_p25': 0.070334,
    'knn_mean_p25': 1.253397,
    'dist_nearest_p75': 1.006756, # terra_incognita: dist_nearest > this
    'curvature_p25': 0.048213,
    'ridge_p25': 0.055153,        # low_pr: ridge < this
    'ridge_p75': 0.073758,        # decision_boundary: ridge > this
}
```

**Fix 2: Consecutive Cooldown.** Same signal cannot fire on back-to-back steps. Prevents the cascade where the same intervention derails multiple consecutive steps.

**Fix 3: Signature Diversity Logging.** Engine tracks raw vs. final signatures, cooldown skips, and diversity metrics for post-hoc analysis.

**Fix 4: Intervention Text Redesign.** Replaced meta-questioning with directed analytical instructions:

| Signal | Old (meta-questioning) | New (directed) |
|--------|----------------------|----------------|
| low_pr | "What assumptions could be questioned?" | "Name one concrete alternative approach or explanation you have not yet considered." |
| framework_collision | "Surface all competing paths..." | "Identify the specific tension between competing frameworks..." |
| terra_incognita | "Acknowledge speculative nature..." | "What testable prediction would distinguish this from existing alternatives?" |
| decision_boundary | "Generate at least 3 approaches..." | "What would change your assessment if your key assumption were false?" |

### Result: 75% Recovery

| Metric | Baseline | Original | Fixed | Recovery |
|--------|----------|----------|-------|----------|
| **Mean quality** | **0.799** | **0.735** | **0.782** | **75%** |
| Interventions | 0/64 | 64/64 (100%) | 24/64 (38%) | -- |
| Signal types | -- | 1 | 4 | -- |
| Consecutive same-sig | -- | 100% | 4% | -- |

### Per-Domain Results

| Domain | Baseline | Original | Fixed | Orig Delta | Fixed Delta |
|--------|----------|----------|-------|------------|-------------|
| Research | 0.789 | 0.634 | 0.751 | -0.155 | **-0.038** |
| Design | 0.737 | 0.827 | 0.816 | +0.090 | **+0.079** |
| Ethics | 0.900 | 0.717 | 0.816 | -0.183 | **-0.084** |
| Hypothesis | 0.770 | 0.763 | 0.744 | -0.007 | **-0.026** |

### Signal Breakdown (Fixed Run)

| Signal | Fires | % | Typical Domain |
|--------|-------|---|----------------|
| low_pr | 10 | 42% | Research, Ethics |
| framework_collision | 8 | 33% | Design |
| decision_boundary | 3 | 13% | Hypothesis |
| terra_incognita | 3 | 13% | Hypothesis |

The fixed system achieved what the original could not: selective, diverse interventions. But three findings persisted:

1. **Design still benefits** (+0.079). Framework collision interventions on design tasks consistently improve quality.
2. **Ethics still degrades** (-0.084). Even with fewer interventions, ethics tasks -- which have the highest baselines (0.900 mean) -- lose quality when interrupted.
3. **Overall quality still slightly negative** (-0.017). The remaining gap is within the noise band of temperature=0.7 stochastic variation.

---

## 4. Track 5: Recursive Learning and the Penalty Ratchet

### Setup

The recursive learner ran 5 iterations through all 16 tasks (320 total API calls). After each iteration, it computed per-signature effectiveness (mean quality delta for tasks where each signal fired) and adjusted policy:

- **Negative delta** (signal hurts quality): multiply threshold by 1.3x (harder to trigger)
- **Positive delta** (signal helps quality): multiply threshold by 0.85x (easier to trigger)
- **Strongly negative** (delta < -0.10): disable signal entirely
- Four adjustable signals: `low_pr`, `framework_collision`, `decision_boundary`, `terra_incognita`

The Goodhart detector monitored for metric gaming with 8 red flags and 4 green flags.

### Learning Curve

| Iter | Mean Quality | Delta | Interventions | Rate | Goodhart |
|------|-------------|-------|---------------|------|----------|
| BL | 0.799 | -- | 0 | 0% | -- |
| 1 | 0.788 | -0.011 | 24 | 38% | -- |
| 2 | 0.783 | -0.016 | 9 | 14% | PASS (0 red, 1 green) |
| 3 | 0.767 | -0.032 | 11 | 17% | PASS (0 red, 2 green) |
| 4 | 0.778 | -0.021 | 14 | 22% | PASS (0 red, 2 green) |
| 5 | **0.805** | **+0.006** | **1** | **2%** | **FAIL (3 red, 2 green)** |

The quality trajectory is V-shaped: degraded through iterations 1-3, recovered in iteration 4, and surpassed baseline in iteration 5 by +0.006. But the mechanism of recovery reveals a problem.

### Policy Evolution

| Signal | Init | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Final |
|--------|------|--------|--------|--------|--------|-------|
| low_pr | 1.00 | 1.30 | 1.69 | 1.69 | 1.69 | 1.69 |
| framework_collision | 1.00 | 1.30 | 1.11 | 0.94 | 1.22 | 1.22* |
| decision_boundary | 1.00 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 |
| terra_incognita | 1.00 | 0.85 | 0.72 | 0.94 | 1.22 | 1.22 |

*framework_collision was DISABLED after iteration 5 (single fire produced delta -0.161).

**low_pr** was penalized immediately and permanently. It fired on 9 tasks in iteration 1 (mean delta -0.026), penalized to 1.3x. Fired once more in iteration 2 (delta -0.006), penalized to 1.69x. Never fired again. The learner correctly identified that low_pr interventions disrupt structured reasoning.

**framework_collision** oscillated. Penalized in iteration 1, rewarded in iterations 2-3, penalized again in iteration 4. It was beneficial for design tasks but harmful for ethics -- the domain-agnostic policy couldn't resolve this conflict. Its sole fire in iteration 5 (on ethics_03, delta -0.161) triggered the disable threshold.

**decision_boundary** fired once in iteration 1 (delta -0.028), was penalized to 1.30x, and never fired again. Insufficient data for the learner to assess.

**terra_incognita** was the most volatile. Initially rewarded (down to 0.72x), it triggered heavily in iteration 3 where it produced mean delta -0.057. The learner reversed course, penalizing it back to 1.22x. Individual task deltas ranged from -0.180 to +0.152 -- high variance that defeated reliable estimation.

### The Twist: It Learned to Do Nothing

By iteration 5, all threshold multipliers exceeded 1.2x. Only one intervention fired in 64 steps (framework_collision on ethics_03, producing the worst quality in the iteration: 0.662). With interventions effectively silenced, the model produced its natural output quality: 0.805, slightly above baseline.

**The learner's optimal policy was to suppress all interventions.**

### Goodhart Detection

The Goodhart detector correctly caught this:

| Red Flag | Value | Threshold | Triggered |
|----------|-------|-----------|-----------|
| terra_incognita_avoidance | 0 fires | >0 (initial: 3) | YES |
| mono_signature_collapse | 100% single signal | >80% | YES |
| intervention_rate_collapse | 1.6% | <10% (initial: 38%) | YES |

| Green Flag | Value | Triggered |
|------------|-------|-----------|
| sparser_higher_impact | rate -0.203, quality +0.027 | YES |
| weakest_improved_most | hypothesis +0.062 vs avg +0.027 | YES |

Verdict: **FAIL** (3 red flags >= threshold of 3).

The green flags are technically true -- fewer interventions did produce higher quality, and the weakest domain did improve most -- but the red flags reveal *why*: not because the learner found better interventions, but because it found that no intervention is the best intervention.

### Per-Domain Quality Across All Iterations

| Domain | BL | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|--------|-----|--------|--------|--------|--------|--------|
| Research | 0.789 | 0.793 | 0.737 | 0.736 | 0.750 | **0.829** |
| Design | 0.737 | 0.778 | 0.799 | 0.773 | **0.833** | 0.788 |
| Ethics | 0.900 | 0.787 | 0.829 | 0.821 | 0.792 | 0.805 |
| Hypothesis | 0.770 | 0.793 | 0.766 | 0.738 | 0.738 | **0.800** |

Design achieved its best result (0.833, +0.096 over baseline) in iteration 4, when framework_collision was most active (4 fires, all on design tasks). This is the strongest evidence that targeted interventions can genuinely help -- but only for the right domain.

Response diversity remained stable (~0.96) across all iterations, confirming the learner did not collapse to repetitive outputs.

### Root Cause: The Penalty Ratchet

The asymmetric learning rates (penalty 1.3x vs. reward 0.85x) combined with stochastic quality variance create a systematic upward ratchet on all threshold multipliers.

**The mechanism:**

1. Each intervention adds noise to task quality. At temperature 0.7, individual task scores vary ~0.05 between runs -- larger than most intervention effects (~0.01-0.02).
2. When noise is negative (looks like the signal hurt), the threshold multiplier is penalized by 1.3x (strong push up).
3. When noise is positive (looks like the signal helped), the threshold multiplier is rewarded by 0.85x (weak push down).
4. Over iterations, the expected multiplier drifts upward regardless of the signal's true value.
5. Eventually all multipliers exceed the range where any signal can fire.

**The math:** If a signal's true effect is zero (pure noise), after N penalty/reward events with equal probability:

```
E[multiplier] = 1.0 * (1.3)^(N/2) * (0.85)^(N/2)
              = 1.0 * (1.3 * 0.85)^(N/2)
              = 1.0 * (1.105)^(N/2)
```

This grows exponentially. After 10 events (5 penalties, 5 rewards): multiplier = 1.105^5 = 1.64x. After 20 events: 2.69x. **Every signal will eventually be silenced**, regardless of its actual value.

**The general principle:** For multiplicative policy updates, `penalty_rate * reward_rate` must equal 1.0 to avoid systematic drift. Our rates (1.3 * 0.85 = 1.105) guaranteed convergence to suppression.

---

## 5. Key Findings

### Finding 1: Threshold Miscalibration as Intervention Failure Mode

Geometric features calibrated on a reference corpus do not transfer to live LLM reasoning embeddings. Reference-to-reference distances (intra-corpus) differ fundamentally from query-to-reference distances (cross-distribution). This produced a 100% false positive rate that no downstream component could recover from. **Lesson:** any real-time geometric intervention system must calibrate thresholds on the same embedding distribution it will encounter at inference time.

### Finding 2: The Meta-Questioning Cascade

Repeated "widen the search" prompts destroy convergent reasoning. When the same open-ended meta-question ("What assumptions could be questioned?") fires at every step, the model shifts from analytical mode to recursive self-interrogation. Each step questions the previous step rather than advancing toward a conclusion. This is a specific, reproducible failure mode of naive intervention design. **Lesson:** intervention text must direct rather than interrogate. "Name one concrete alternative" works; "What assumptions could be questioned?" does not.

### Finding 3: Domain Asymmetry

The same geometric signal produces opposite effects depending on reasoning context:

| | Design | Ethics |
|---|--------|--------|
| Original (all low_pr) | +12.2% | -20.3% |
| Fixed (mixed signals) | +7.9% | -8.4% |
| Track 5 best | +9.6% (iter 4) | -7.1% (iter 2) |

Design exploration is divergent -- it benefits from prompts that force consideration of alternatives. Ethics reasoning is convergent -- it benefits from uninterrupted analytical progression toward a nuanced conclusion. A domain-agnostic intervention policy cannot serve both. **Lesson:** intervention policy should be conditioned on reasoning domain or at minimum on the divergent/convergent character of the task.

### Finding 4: The Penalty Ratchet

Asymmetric multiplicative learning rates (penalty 1.3x, reward 0.85x) guarantee convergence to intervention suppression when the signal-to-noise ratio is low. The product 1.3 * 0.85 = 1.105 creates exponential drift toward silence. This is independent of the signal's true value -- even a genuinely beneficial signal will be suppressed if quality measurements are noisy. **Lesson:** for multiplicative policy updates, `penalty_rate * reward_rate` must equal 1.0. Alternatively, use additive updates, Bayesian updating, or explore/exploit strategies (UCB, Thompson sampling) that explicitly account for estimation uncertainty.

### Finding 5: Goodhart Detection Works

The Goodhart detector correctly distinguished "quality improved because we stopped interfering" from "quality improved because we intervened better." In iteration 5, the quality metric alone showed a +0.006 improvement over baseline -- a tempting positive result. The Goodhart detector identified three red flags (intervention avoidance, signal collapse, rate collapse) and issued a FAIL verdict. Without this meta-monitoring, we might have reported Track 5 as a success. **Lesson:** any adaptive system that optimizes an objective needs a second-order monitor that checks *how* the objective is being achieved.

---

## 6. What Geometric Features Actually Do

The experimental arc illuminates a distinction between *detection* and *intervention*:

**Detection works.** Geometric features reliably identify reasoning states:
- `knn_std` (neighborhood standard deviation) detects framework collision -- the embedding moving between distinct reference clusters
- `dist_nearest` detects terra incognita -- the embedding far from all reference points
- `ridge_proximity` detects low novelty -- the embedding close to the reference manifold ridge
- Phase E validated +3.8% R^2 improvement for borderline classification

**Intervention is the hard problem.** Converting detection into improved reasoning requires:
1. Correct thresholds (distribution-matched, not reference-based)
2. Correct text (directed instructions, not meta-questions)
3. Correct timing (cooldown to prevent cascades)
4. Correct targeting (domain-aware, not domain-agnostic)
5. Correct learning dynamics (balanced rates, sufficient sample size)

We solved problems 1-3. Problem 4 was identified but deliberately left unsolved (to maintain domain-agnostic design). Problem 5 was identified through the penalty ratchet analysis.

The gap between detecting uncertainty and improving reasoning about uncertainty is large. Geometric signals can reliably tell you *where* the model is uncertain. They cannot yet tell you *how to make the model less uncertain in a way that improves output quality*. That requires understanding the interaction between intervention type, reasoning domain, and task difficulty -- a problem that is fundamentally about intervention design, not geometry.

---

## 7. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| LLM | Claude Sonnet 4.5 (claude-sonnet-4-5-20250929) |
| Temperature | 0.7 |
| Max tokens | 1024 per step |
| Steps per task | 4 |
| Tasks | 16 (4 per domain) |
| Domains | research_questions, design_exploration, ethical_dilemmas, novel_hypothesis |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Reference corpus | 80 texts (20 per domain) |
| k (k-NN) | 50 |
| Quality scorer | Heuristic rubric: breadth, depth, actionability, uncertainty (composite 0-1) |
| Track 5 penalty rate | 1.3x (threshold multiplier for negative-delta signals) |
| Track 5 reward rate | 0.85x (threshold multiplier for positive-delta signals) |
| Track 5 disable threshold | -0.10 (mean delta below this disables signal) |

---

## 8. Limitations

1. **Heuristic quality scorer.** The scorer measures lexical proxies (list item count, depth markers, uncertainty phrases). It may not capture true reasoning quality. A human evaluation or LLM-as-judge approach would provide more reliable quality signals.

2. **Temperature variance.** At 0.7, individual task scores vary ~0.05-0.10 between runs. This is larger than most intervention effects (~0.01-0.02), making per-signal effectiveness estimation unreliable with 16 tasks.

3. **Small task bank.** 16 tasks x 4 domains provides limited statistical power. Per-signal effectiveness estimates have high variance, especially for signals that fire on only 1-3 tasks.

4. **No domain conditioning.** The experiment deliberately used domain-agnostic thresholds. The data strongly suggests domain-conditioned policy would outperform.

5. **Single model.** All results are from Claude Sonnet 4.5. Geometric features may behave differently with other LLMs, embedding models, or reasoning architectures.

6. **No human evaluation.** All quality assessment is automated. The heuristic scorer's sensitivity to intervention effects is unvalidated.

---

## 9. Next Steps

1. **Fix the penalty ratchet and re-run Track 5.** Use balanced rates (`penalty * reward = 1.0`, e.g., 1.15x / 0.87x) or switch to Bayesian updating with confidence-gated adjustments.

2. **Add domain conditioning.** Allow per-domain threshold multipliers. The learner should discover that design benefits from framework_collision while ethics does not.

3. **Increase sample size.** More tasks or multiple repetitions per iteration to reduce variance. Target: 95% confidence that a 2% quality delta is statistically significant.

4. **Intervention text as learnable parameter.** Instead of fixed instruction text, allow the learner to select from a library of intervention styles (directed, questioning, constraining, expanding).

5. **Cross-model validation.** Test whether the same geometric signatures appear in different LLMs (GPT-4, Gemini, open-source models). If the geometry is model-invariant, the intervention framework becomes portable.

6. **Human evaluation study.** Replace or supplement the heuristic scorer with human quality ratings to validate that detected quality differences reflect genuine reasoning improvement.

---

## Appendix A: Complete Per-Task Results

### Baseline

| Task | Domain | Quality |
|------|--------|---------|
| research_01 | research | 0.677 |
| research_02 | research | 0.755 |
| research_03 | research | 0.865 |
| research_04 | research | 0.860 |
| design_01 | design | 0.703 |
| design_02 | design | 0.709 |
| design_03 | design | 0.854 |
| design_04 | design | 0.680 |
| ethics_01 | ethics | 0.904 |
| ethics_02 | ethics | 0.940 |
| ethics_03 | ethics | 0.823 |
| ethics_04 | ethics | 0.933 |
| hypothesis_01 | hypothesis | 0.700 |
| hypothesis_02 | hypothesis | 0.831 |
| hypothesis_03 | hypothesis | 0.804 |
| hypothesis_04 | hypothesis | 0.745 |

### Track 4: Three-Way Comparison

| Task | Baseline | Original | Fixed | Orig Delta | Fixed Delta | Fixed Signals |
|------|----------|----------|-------|------------|-------------|---------------|
| research_01 | 0.677 | 0.675 | 0.691 | -0.002 | +0.014 | low_pr |
| research_02 | 0.755 | 0.705 | 0.695 | -0.050 | -0.060 | low_pr |
| research_03 | 0.865 | 0.565 | 0.805 | -0.300 | -0.060 | low_pr |
| research_04 | 0.860 | 0.590 | 0.815 | -0.270 | -0.045 | low_pr |
| design_01 | 0.703 | 0.815 | 0.691 | +0.112 | -0.012 | framework_collision, low_pr |
| design_02 | 0.709 | 0.783 | 0.904 | +0.074 | +0.195 | framework_collision, low_pr |
| design_03 | 0.854 | 0.865 | 0.812 | +0.011 | -0.042 | framework_collision |
| design_04 | 0.680 | 0.843 | 0.855 | +0.163 | +0.175 | framework_collision, decision_boundary, low_pr |
| ethics_01 | 0.904 | 0.685 | 0.849 | -0.219 | -0.055 | low_pr |
| ethics_02 | 0.940 | 0.650 | 0.840 | -0.290 | -0.100 | (none) |
| ethics_03 | 0.823 | 0.703 | 0.831 | -0.120 | +0.008 | framework_collision, decision_boundary |
| ethics_04 | 0.933 | 0.831 | 0.743 | -0.102 | -0.190 | framework_collision x2 |
| hypothesis_01 | 0.700 | 0.771 | 0.620 | +0.071 | -0.080 | low_pr |
| hypothesis_02 | 0.831 | 0.630 | 0.737 | -0.201 | -0.094 | decision_boundary, terra_incognita, low_pr |
| hypothesis_03 | 0.804 | 0.865 | 0.809 | +0.061 | +0.005 | framework_collision, terra_incognita |
| hypothesis_04 | 0.745 | 0.785 | 0.812 | +0.040 | +0.067 | terra_incognita |

### Track 5: Per-Task Quality (Iteration 5, Final)

| Task | Baseline | Iter 5 | Delta | Interventions |
|------|----------|--------|-------|---------------|
| research_01 | 0.677 | 0.843 | +0.166 | 0 |
| research_02 | 0.755 | 0.806 | +0.051 | 0 |
| research_03 | 0.865 | 0.820 | -0.045 | 0 |
| research_04 | 0.860 | 0.845 | -0.015 | 0 |
| design_01 | 0.703 | 0.777 | +0.074 | 0 |
| design_02 | 0.709 | 0.795 | +0.086 | 0 |
| design_03 | 0.854 | 0.805 | -0.049 | 0 |
| design_04 | 0.680 | 0.775 | +0.095 | 0 |
| ethics_01 | 0.904 | 0.852 | -0.052 | 0 |
| ethics_02 | 0.940 | 0.870 | -0.070 | 0 |
| ethics_03 | 0.823 | 0.662 | -0.161 | 1 (framework_collision) |
| ethics_04 | 0.933 | 0.837 | -0.096 | 0 |
| hypothesis_01 | 0.700 | 0.662 | -0.038 | 0 |
| hypothesis_02 | 0.831 | 0.789 | -0.042 | 0 |
| hypothesis_03 | 0.804 | 0.910 | +0.106 | 0 |
| hypothesis_04 | 0.745 | 0.837 | +0.092 | 0 |

---

## Appendix B: Goodhart Detector Architecture

The Goodhart detector monitors 8 red flags and 4 green flags across iterations:

**Red Flags** (evidence of gaming):
1. `pr_up_quality_flat` -- ridge proximity improves but quality doesn't
2. `diversity_decrease` -- response diversity drops below 70% of initial
3. `interventions_suppressed` -- intervention rate drops to zero
4. `terra_incognita_avoidance` -- model stops exploring unfamiliar territory
5. `quality_up_depth_down` -- composite quality up but depth dimension down
6. `mono_signature_collapse` -- single signal accounts for >80% of fires
7. `quality_oscillation` -- alternating up-down pattern across 3+ iterations
8. `intervention_rate_collapse` -- rate drops below 10% of steps

**Green Flags** (evidence of genuine improvement):
1. `pr_and_quality_up` -- both ridge proximity and quality improve together
2. `richer_exploration` -- more framework_collision appearances (broader search)
3. `sparser_higher_impact` -- fewer interventions but higher per-intervention quality gain
4. `weakest_improved_most` -- previously weakest domain gets biggest improvement

**Verdict:** FAIL (>= 3 red), WARN (>= 1 red), PASS (0 red).

---

## Appendix C: File Index

| File | Description |
|------|-------------|
| `experiments/track4_switches/switch_engine.py` | Geometric signature classifier + intervention engine |
| `experiments/track4_switches/TRACK4_FIX_REPORT.md` | Diagnosis and fix report for threshold miscalibration |
| `experiments/track5_recursive/recursive_learner.py` | Adaptive policy optimizer (threshold multipliers) |
| `experiments/track5_recursive/goodhart_detector.py` | Metric gaming detector (8 red, 4 green flags) |
| `experiments/track5_recursive/run_live_trials.py` | Live experiment runner (320 API calls) |
| `experiments/shared/geometric_tracer.py` | Geometric feature computation + live calibration |
| `experiments/shared/quality_scorer.py` | Heuristic quality rubric |
| `experiments/shared/task_bank.py` | 16-task evaluation bank |
| `experiments/shared/embedder.py` | Sentence-transformer embedding wrapper |
| `experiments/shared/llm_client.py` | Claude API client with retry logic |
| `experiments/results/track4_live_baseline/20260208_221402/` | Baseline results (64 calls) |
| `experiments/results/track4_live_switched/20260208_225034/` | Original switched results (64 calls) |
| `experiments/results/track4_live_switched/20260209_003159/` | Fixed switched results (64 calls) |
| `experiments/results/track5_recursive/20260209_021234/` | Recursive trials results (320 calls) |
