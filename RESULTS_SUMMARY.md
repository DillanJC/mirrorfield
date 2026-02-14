# Geometric Interventions: Results Summary

**Experiment:** Can geometric features from reasoning step embeddings steer LLM reasoning in real-time?
**Model:** Claude Sonnet 4.5 | **Embeddings:** all-MiniLM-L6-v2 (384-dim) | **API calls:** 448 total

---

## Key Numbers

| Experiment | Mean Quality | Delta vs Baseline | Intervention Rate |
|------------|-------------|-------------------|-------------------|
| Baseline (no interventions) | **0.799** | -- | 0% |
| Track 4 Original (miscalibrated) | 0.735 | **-8.0%** | 100% (all low_pr) |
| Track 4 Fixed (recalibrated) | 0.782 | -2.1% | 38% (4 signal types) |
| Track 5 Iter 5 (learned policy) | 0.805 | +0.8% | 2% (1 fire in 64 steps) |

## Domain Asymmetry (Original Switched)

| Domain | Baseline | Switched | Change |
|--------|----------|----------|--------|
| Design | 0.737 | 0.827 | **+12.2%** (helped) |
| Ethics | 0.900 | 0.717 | **-20.3%** (harmed) |

Same intervention, opposite effects. Persisted across all experiments.

## Five Novel Findings

1. **Threshold miscalibration** -- Reference-corpus thresholds don't transfer to live LLM embeddings. Produced 100% false positive rate.
2. **Meta-questioning cascade** -- Repeated "question your assumptions" prompts destroy convergent reasoning. Directed instructions ("name one alternative") work; open interrogation does not.
3. **Domain asymmetry** -- Design tasks (divergent) benefit from interventions; ethics tasks (convergent) are harmed. Domain-agnostic policy cannot serve both.
4. **Penalty ratchet** -- Asymmetric multiplicative learning rates (1.3 * 0.85 = 1.105) guarantee convergence to suppression. Fix: `penalty * reward = 1.0`.
5. **Goodhart detection works** -- Correctly flagged iteration 5's quality improvement as gaming (3 red flags: intervention avoidance, signal collapse, rate collapse).

## The Gap

Geometric features reliably **detect** reasoning states (+3.8% R^2 on borderline classification). Converting detection into beneficial **intervention** requires correct thresholds, correct text, correct timing, and correct targeting. Detection is solved; intervention design remains open.

## Files

- Full report: [`GEOMETRIC_INTERVENTIONS_REPORT.md`](GEOMETRIC_INTERVENTIONS_REPORT.md)
- Track 4 fix report: [`experiments/track4_switches/TRACK4_FIX_REPORT.md`](experiments/track4_switches/TRACK4_FIX_REPORT.md)
- Track 5 report: [`experiments/track5_recursive/TRACK5_REPORT.md`](experiments/track5_recursive/TRACK5_REPORT.md)
- Visualization data: [`experiments/results/track5_visualization_data.json`](experiments/results/track5_visualization_data.json)
