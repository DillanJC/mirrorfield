# Track 5: Recursive Self-Learning -- Live Evaluation Report

**Generated:** 2026-02-09
**Model:** claude-sonnet-4-5-20250929 (temperature 0.7, max_tokens 1024)
**Iterations:** 5 (320 API calls: 5 x 16 tasks x 4 steps)
**Timestamp:** 20260209_021234

## Summary

| Metric | Value |
|--------|-------|
| Baseline mean quality | 0.7989 |
| Best iteration quality | **0.8053** (iteration 5, +0.006 vs baseline) |
| Worst iteration quality | 0.7670 (iteration 3, -0.032 vs baseline) |
| Final policy signals enabled | 3 of 4 (framework_collision DISABLED) |
| Final intervention rate | 1.6% (1/64 steps) |
| Goodhart final verdict | **FAIL** (3 red, 2 green) |
| Response diversity | Stable (~0.96 across all iterations) |

**Key finding:** The recursive learner discovered that suppressing interventions produces higher quality than applying them. After 5 iterations the policy effectively silenced itself (1/64 fires), quality rebounded above baseline (+0.006), but the Goodhart detector correctly flagged this as metric gaming.

## Learning Curve

| Iter | Mean Quality | Delta | Interventions | Rate | Goodhart |
|------|-------------|-------|---------------|------|----------|
| BL | 0.7989 | -- | 0 | 0% | -- |
| 1 | 0.7877 | -0.011 | 24 | 38% | -- |
| 2 | 0.7828 | -0.016 | 9 | 14% | PASS (0R/1G) |
| 3 | 0.7670 | -0.032 | 11 | 17% | PASS (0R/2G) |
| 4 | 0.7781 | -0.021 | 14 | 22% | PASS (0R/2G) |
| 5 | **0.8053** | **+0.006** | 1 | 2% | FAIL (3R/2G) |

The quality trajectory is V-shaped: degraded through iterations 1-3 (nadir -0.032), recovered in iteration 4, and surpassed baseline in iteration 5. The recovery coincides exactly with intervention suppression -- quality improved as fewer interventions fired.

## Final Policy

| Signal | Multiplier | Enabled | Behavior |
|--------|-----------|---------|----------|
| low_pr | 1.69 | Yes | Silent (threshold too high to trigger) |
| framework_collision | 1.22 | **No** | DISABLED (delta -0.161 < -0.10 threshold) |
| decision_boundary | 1.30 | Yes | Silent (never fired after iteration 1) |
| terra_incognita | 1.22 | Yes | Silent (threshold too high to trigger) |

### Policy Evolution (threshold multipliers)

| Signal | Init | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|--------|------|--------|--------|--------|--------|--------|
| low_pr | 1.00 | 1.30 | 1.69 | 1.69 | 1.69 | 1.69 |
| framework_collision | 1.00 | 1.30 | 1.11 | 0.94 | 1.22 | 1.22* |
| decision_boundary | 1.00 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 |
| terra_incognita | 1.00 | 0.85 | 0.72 | 0.94 | 1.22 | 1.22 |

*Disabled after iteration 5 (single fire produced delta -0.161).

### Per-signal learning narrative

- **low_pr**: Permanently penalized. Fired 9 tasks in iteration 1 (mean delta -0.026), penalized to 1.3. Fired once more in iteration 2 (delta -0.006), penalized to 1.69. Never fired again. The learner correctly identified that low_pr interventions ("name one concrete alternative") disrupt structured reasoning.

- **framework_collision**: Oscillated. Penalized in iter 1, rewarded in iter 2-3 (delta +0.020, +0.007), penalized again in iter 4 (delta -0.007). Only fired once in iter 5 (on ethics_03, quality 0.662, delta -0.161) and was disabled. The oscillation suggests its value is context-dependent -- beneficial for design tasks but harmful for ethics.

- **decision_boundary**: Fired once in iteration 1 (delta -0.028), penalized to 1.30, never fired again. Insufficient data for the learner to assess.

- **terra_incognita**: Most volatile. Rewarded in iters 1-2 (down to 0.72), penalized in iter 3 (mean delta -0.057), then oscillated. High variance in individual deltas (-0.180 to +0.152). The learner couldn't find a stable setpoint.

## Goodhart Analysis

### Verdict History

| Iter | Verdict | Red Flags | Green Flags |
|------|---------|-----------|-------------|
| 2 | PASS | 0 | 1 (weakest_improved_most) |
| 3 | PASS | 0 | 2 (richer_exploration, weakest_improved_most) |
| 4 | PASS | 0 | 2 (richer_exploration, weakest_improved_most) |
| 5 | **FAIL** | 3 | 2 |

### Iteration 5 Red Flags

| Flag | Value | Threshold | Detail |
|------|-------|-----------|--------|
| terra_incognita_avoidance | 0 fires | >0 initial | Agent stopped exploring unfamiliar territory |
| mono_signature_collapse | 100% | >80% | Only framework_collision fired (1 of 1) |
| intervention_rate_collapse | 1.6% | <10% | Learner suppressed nearly everything |

### Iteration 5 Green Flags

| Flag | Value | Detail |
|------|-------|--------|
| sparser_higher_impact | rate -0.203, quality +0.027 | Fewer interventions, higher quality |
| weakest_improved_most | hypothesis +0.062 vs avg +0.027 | Weakest domain got biggest gains |

### Diversity Trend

Response diversity remained remarkably stable across all iterations:

| Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 |
|--------|--------|--------|--------|--------|
| 0.964 | 0.967 | 0.966 | 0.960 | 0.965 |

This is a genuinely positive signal: the learner did not collapse to repetitive outputs despite suppressing interventions. Quality improvement came from removing interference, not from gaming diversity metrics.

## Per-Domain Analysis

| Domain | Baseline | Iter 1 | Iter 2 | Iter 3 | Iter 4 | Iter 5 | Best Delta |
|--------|----------|--------|--------|--------|--------|--------|------------|
| Research | 0.789 | 0.793 | 0.737 | 0.736 | 0.750 | **0.829** | **+0.039** |
| Design | 0.737 | 0.778 | 0.799 | 0.773 | **0.833** | 0.788 | **+0.096** |
| Ethics | 0.900 | 0.787 | 0.829 | 0.821 | 0.792 | 0.805 | **-0.071** |
| Hypothesis | 0.770 | 0.793 | 0.766 | 0.738 | 0.738 | **0.800** | **+0.030** |

### Domain patterns

- **Design** consistently improved across all iterations (+0.037 to +0.096). This domain benefits from creative exploration interventions. Best result in iteration 4 (0.833, +0.096) when framework_collision was most active (4 fires, all on design tasks).

- **Ethics** consistently degraded across all iterations (-0.071 to -0.113). Ethics tasks have high baselines (0.900 mean) and interventions disrupt their naturally strong analytical structure. Even iteration 5 with 0 interventions scored 0.805 (-0.095), suggesting within-run temperature variance.

- **Research** showed a U-shaped curve: degraded in iterations 2-4, then rebounded to best-ever 0.829 (+0.039) in iteration 5 with zero interventions.

- **Hypothesis** similar U-shape: low in iterations 3-4, recovered to 0.800 (+0.030) in iteration 5.

## Track 4 vs Track 5 Comparison

| Metric | Baseline | Track 4 Fixed | Track 5 Best (Iter 5) |
|--------|----------|---------------|----------------------|
| Mean quality | 0.799 | 0.782 (-2.1%) | **0.805 (+0.8%)** |
| Intervention rate | 0% | 38% | 2% |
| Signal types active | -- | 4 | 1 (then 0) |
| Design delta | -- | +0.079 | +0.051 |
| Ethics delta | -- | -0.084 | -0.095 |
| Research delta | -- | -0.038 | +0.039 |
| Hypothesis delta | -- | -0.026 | +0.030 |

Track 5 achieved higher overall quality than Track 4 fixed, but through a radically different mechanism: Track 4 applied selective interventions (38% rate, 4 signal types), while Track 5 converged to near-silence (2% rate, 1 signal type). The quality gain in Track 5 is essentially the baseline quality plus temperature-driven variance, not a learned intervention strategy.

## Success Criteria Assessment

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Policy converges | Multipliers stabilize | All signals raised above 1.2x | **PASS** |
| Quality improves in >= 1 domain | Domain delta > 0 | Design +0.051, Research +0.039, Hypothesis +0.030 | **PASS** |
| Overall quality >= baseline | Delta >= 0 | +0.006 in iteration 5 | **PASS** (marginal) |
| Goodhart passes all iterations | 0 FAIL verdicts | FAIL in iteration 5 | **FAIL** |
| No signal disabled | All signals enabled | framework_collision DISABLED | **FAIL** |
| Intervention rate > 10% | Rate >= 10% | 2% in iteration 5 | **FAIL** |

## Root Cause: The Penalty Ratchet

The asymmetric learning rates (penalty 1.3x vs reward 0.85x) combined with stochastic quality variance create a systematic upward ratchet on all threshold multipliers.

**Mechanism:**
1. Each intervention adds noise to task quality (temperature 0.7 creates ~0.05 std per task)
2. When noise is negative, the signal gets penalized 1.3x (strong push up)
3. When noise is positive, the signal gets rewarded 0.85x (weak push down)
4. Over iterations, the expected multiplier drift is upward regardless of true signal value
5. Eventually all multipliers exceed the range where signals can fire
6. With no signals firing, the model produces its natural (baseline-like) quality

**Mathematical intuition:** If a signal's true effect is zero (pure noise), after N penalty/reward events with equal probability:
- Expected multiplier = 1.0 * (1.3)^(N/2) * (0.85)^(N/2) = 1.0 * (1.3 * 0.85)^(N/2) = 1.0 * (1.105)^(N/2)
- This grows exponentially regardless of signal quality -- every signal will eventually be silenced.

**The fix for Track 6:** The penalty/reward rates must satisfy `penalty_rate * reward_rate = 1.0` to avoid systematic drift. For example: penalty 1.15x, reward 1/1.15 = 0.87x. Or use additive adjustments instead of multiplicative.

## Interpretation

### What the experiment proved

1. **The Goodhart detector works.** It correctly identified metric gaming in iteration 5 with three specific red flags (intervention avoidance, signal collapse, rate collapse) while the quality metric alone showed improvement.

2. **Intervention noise exceeds signal.** With temperature 0.7 and 16 tasks, the stochastic variance per task (~0.05) is larger than the true intervention effect (estimated 0.01-0.02). The learner cannot reliably distinguish beneficial from harmful interventions.

3. **The penalty ratchet is real.** Asymmetric learning rates with multiplicative updates create systematic drift toward silence. This is a known failure mode in online learning -- Thompson sampling or UCB approaches would handle exploration/exploitation better.

4. **Design domain genuinely benefits from interventions.** Across all 5 iterations, design tasks scored above baseline (+0.037 to +0.096), especially when framework_collision was active. This is the strongest positive signal in the dataset.

5. **Ethics domain is naturally strong and fragile.** Ethics tasks have the highest baselines (0.900 mean) and interventions consistently harm them. A domain-aware policy (which we deliberately avoided) would be beneficial here.

### What the experiment did NOT prove

1. **That interventions are useless.** The penalty ratchet artificially suppresses all signals. With balanced learning rates, the learner might converge to a selective policy that fires on design tasks and avoids ethics tasks.

2. **That recursive learning can't work.** The architecture is sound -- 320 API calls generated real per-signature effectiveness data. The failure is in the hyperparameters (asymmetric rates) and sample size (16 tasks may be too few for reliable signal estimation).

3. **That quality improvements in iteration 5 are genuine.** The +0.006 delta is within the noise band. Multiple repetitions would be needed to confirm.

## Limitations

1. **Heuristic quality scorer**: The scorer measures lexical proxies (list items, depth markers, uncertainty phrases). It may not capture true reasoning quality differences.

2. **Temperature variance**: At 0.7, individual task scores vary ~0.05-0.10 between runs, larger than most intervention effects.

3. **Small task bank**: 16 tasks x 4 domains provides limited statistical power for per-signal estimation.

4. **No domain-specific policy**: The experiment deliberately used domain-agnostic thresholds. A domain-conditional policy could preserve design gains while protecting ethics.

5. **Fixed cooldown window**: Cooldown was held at 1 throughout; the learner did not explore higher cooldown values.

## Files

| File | Role |
|------|------|
| `experiments/track5_recursive/recursive_learner.py` | Adaptive policy optimizer (PolicyOptimizer, AdaptivePolicy) |
| `experiments/track5_recursive/goodhart_detector.py` | Metric gaming detector (8 red flags, 4 green flags) |
| `experiments/track5_recursive/run_live_trials.py` | Live experiment runner (320 API calls) |
| `experiments/results/track5_recursive/20260209_021234/recursive_trials.json` | Full results (per-task, per-iteration, policy history) |

## Recommendations for Track 6

1. **Fix the penalty ratchet**: Use balanced rates (penalty * reward = 1.0) or additive updates instead of multiplicative.

2. **Increase sample size**: Either more tasks or multiple repetitions per iteration to reduce variance in effectiveness estimates.

3. **Add domain conditioning**: Allow per-domain threshold multipliers. The data clearly shows design benefits from interventions while ethics does not.

4. **Use Bayesian updating**: Replace point estimates with posterior distributions over signal effectiveness. Only adjust policy when confidence exceeds a threshold.

5. **Lower temperature**: Reducing from 0.7 to 0.4-0.5 would reduce stochastic noise and make intervention effects more detectable.
