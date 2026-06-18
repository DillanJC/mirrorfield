# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [3.0.0] - 2026-06-19 — Honest rebuild: the gate is the product; geometry retracted

### Added
- **Pip-installable package** (`pyproject.toml`): core = numpy/scipy only; extras
  `[mcp]` (MCP server) and `[harm]` (Granite-Guardian); console script `mirrorfield-mcp`.
  Calibration + `gate_thresholds.json` shipped as package data; geometry/api excluded
  from the wheel (verified).
- **Validated decision API**: `decide()` (frozen thresholds `tau_present`/`tau_abstain`
  on `p_correct_relative`, WORK_MAP §4k) and `send_hold_decision()` — the composed
  **SEND / VERIFY / HOLD** pipeline (wrongness gate + optional harm override).
- **`safety_gate` MCP tool** exposing the composed pipeline; `confidence_report` now
  uses the validated operating point when a `context_id` is supplied (was a heuristic).
- **`mirrorfield/mcp/harm.py`** — optional lazy wrapper around IBM Granite-Guardian-3.1-2B
  (third-party harm classifier) for the harm half of the pipeline.
- **`tests/test_gate_decision.py`** (7 tests) locking the thresholds + composition rules;
  **`examples/quickstart.py`** runnable demo.

### Changed / Retracted
- **The embedding-geometry framework (Phase 0–E, Tracks 4–5) is retracted as a set of
  standing claims.** The project's own non-circular falsification showed the geometry
  add-on gives ≈0 predictive lift over standard log-prob signals (ΔAUC ≈ 0); several
  earlier headline results (incl. the 0.7.0 "+6.4%" and "302× / 20-seed" framing) were
  circular or measured the wrong thing. The root README now carries a status banner;
  full per-result verdicts are in `WORK_MAP.md` §4. The one survivor — the lean log-prob
  uncertainty gate — is the product.
- `mirrorfield/__init__.py` rewritten honestly (v3.0.0; dropped the geometry framing).

### Removed
- `mirrorfield/mcp/moltbook_bridge.py` (unwired external-posting bridge — cut).
- The embedding-geometry subpackage is excluded from the installed wheel.

### Research (this cycle — see `WORK_MAP.md`)
- **§4q confidence-contagion:** the "humble users get a less-sure model" worry is a
  generic any-prefix artifact (placebo-killed); humble framing biases the *answer*
  toward a default class.
- **§4r harm-framing:** humble framing softens refusal *style* (−0.19/−0.20 vs placebo,
  replicated JBB+BeaverTails) but not harm *substance* (independent Granite harm ~+0.02)
  — being nice does not jailbreak the 3B model.

## [2.1.0] - 2026-02-08

### Added — Tracks 4 & 5: Recursive Self-Learning Experiment
- **Track 4 (Architectural Switches):** Hard-coded geometric interventions
  - `experiments/track4_switches/switch_engine.py` — Geometric state → intervention mapping
  - `experiments/track4_switches/run_baseline_reasoning.py` — Baseline reasoning (no interventions)
  - `experiments/track4_switches/run_switched_reasoning.py` — Reasoning with geometric switches
  - `experiments/track4_switches/evaluate_switches.py` — Switched vs baseline comparison
- **Track 5 (Recursive Self-Learning):** Adaptive intervention policy
  - `experiments/track5_recursive/recursive_learner.py` — Policy evolution agent
  - `experiments/track5_recursive/run_recursive_trials.py` — Multi-iteration learning loop
  - `experiments/track5_recursive/goodhart_detector.py` — Goodhart's Law detection monitor
  - `experiments/track5_recursive/evaluate_recursive.py` — Final analysis + Goodhart report
- **Shared infrastructure:**
  - `experiments/shared/task_bank.py` — 16 evaluation tasks across 4 domains
  - `experiments/shared/quality_scorer.py` — Heuristic quality rubric (breadth, depth, actionability, uncertainty)
  - `experiments/shared/geometric_tracer.py` — Wraps GeometryBundle for per-step tracing

### Key Concepts
- **Geometric interventions:** Use k-NN geometric signatures (framework_collision, terra_incognita, decision_boundary) to trigger reasoning corrections in real-time
- **Goodhart detection:** Monitors for metric gaming (PR up but quality flat, diversity collapse, intervention suppression)
- **Policy learning:** Track 5 starts from Track 4's hard-coded table and evolves via running-average quality deltas

## [2.0.0] - 2025-12-28

### Changed
- Phase E schema v2.0: removed binary dark_river_candidate and observer_mode flags
- Continuous features only (dark river hypothesis falsified)

## [0.7.0] - 2025-12-28

### Added
- Phase E: Geometry Bundle with 7 k-NN features (+6.4% validated improvement)
- Phase D: Integrated evaluation pipeline (302× speedup, 20-seed validation)
- Phase C: Calibration + friction tagging
- Phase B: Tier-2 semantic discriminator
- Phase A: Evidence pack
- Phase 0: Locked definitions
