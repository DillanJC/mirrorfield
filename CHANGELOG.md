# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
