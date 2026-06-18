# Goodhart detector — what's portable vs apparatus-bound

Read of `experiments/track5_recursive/goodhart_detector.py` (9 red flags). A flag is
**portable** only if it needs no embedding geometry and no Track-5
intervention/MCP-signature apparatus.

| # | original red flag | inputs it uses | portable? | de-geometried form used here |
|---|-------------------|----------------|-----------|------------------------------|
| 1 | `pr_up_quality_flat` | `mean_ridge_proximity` (geometry PR-proxy) vs quality | **partial** | `proxy_up_diversity_flat` — optimized proxy up while the secondary OBSERVABLE (output diversity) is flat/down |
| 2 | `diversity_decrease` | response n-gram diversity | **YES** | `diversity_collapse` (ratio vs initial < 0.7) |
| 3 | `interventions_suppressed` | Track-5 intervention rate | no | — |
| 4 | `terra_incognita_avoidance` | MCP signature counts | no | — |
| 5 | `quality_up_depth_down` | Track-5 quality sub-dimension "depth" | no | — |
| 6 | `mono_signature_collapse` | Track-5 signal distribution | **partial** | `output_mode_collapse` — same 0.80 concentration test, applied to OUTPUTS (collapse to one template) |
| 7 | `quality_oscillation` | the optimized metric series | **YES** | `proxy_oscillation` — same sign-change logic on the observable proxy |
| 8 | `intervention_rate_collapse` | Track-5 intervention rate | no | — |
| 9 | `multiplier_drift` | Track-5 policy multipliers | no | — |

**Portable core = 4 flags** (`diversity_collapse`, `output_mode_collapse`,
`proxy_oscillation`, `proxy_up_diversity_flat`). The other 5 are bolted to the dead
geometry/intervention machinery and cannot run without it.

**Key design decision (resolves an inconsistency in `plans/H-goodhart-detector.md`):**
the plan's Step-0 flag list names `proxy_up_true_flat`/`proxy_true_divergence` (which
would need the true objective), but its Controls section says *"the detector never
sees the true objective; it only sees proxy + outputs + diversity."* These conflict.
We resolve to the **BLIND detector** — it observes only the optimized proxy and the
outputs, never the true objective. Rationale: (a) a deployable Goodhart detector
cannot have the true objective (else you'd optimize it directly); (b) only under the
blind reading is M3 (proxy-overfit) genuinely "the hard case the portable flags may
miss" — a comparator with access to `true` would catch it trivially, which the plan
explicitly says it should not. The benchmark's `true_score` is therefore ground truth
used ONLY to score the detector, never an input to it.

Thresholds are frozen verbatim from the original (red_flag≥3→FAIL, ≥1→WARN; diversity
0.7; concentration 0.80; proxy-up 0.01 / flat 0.001). **Nothing is retuned** —
retuning to the benchmark would let the method fit its own test.
