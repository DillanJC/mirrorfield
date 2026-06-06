# Phase 1/2/5 Status — Experiment Tracks

**Last Updated:** 2026-02-08

## Experiment Track Status

| Track | Name | Status | Report |
|-------|------|--------|--------|
| Track 4 | Architectural Switches | IN PROGRESS | [`experiments/track4_switches/TRACK4_REPORT.md`](../experiments/track4_switches/TRACK4_REPORT.md) |
| Track 5 | Recursive Self-Learning | IN PROGRESS | [`experiments/track5_recursive/TRACK5_REPORT.md`](../experiments/track5_recursive/TRACK5_REPORT.md) |

## Track 4: Architectural Switches

**Goal:** Hard-coded geometric interventions as a deterministic baseline.

Maps geometric signatures to reasoning corrections:
- `framework_collision` → Surface competing paths explicitly
- `terra_incognita` → Acknowledge unfamiliar territory, cite prior work
- `decision_boundary` → Generate 3 approaches, then synthesize
- Low PR → Step back and widen exploration

**Files:**
- `experiments/track4_switches/switch_engine.py`
- `experiments/track4_switches/run_baseline_reasoning.py`
- `experiments/track4_switches/run_switched_reasoning.py`
- `experiments/track4_switches/evaluate_switches.py`

## Track 5: Recursive Self-Learning

**Goal:** Learn which interventions work for which geometric signals.

Starts from Track 4's switch table and evolves policy through iteration:
1. Run all tasks with current policy
2. Record (signal, intervention, quality_delta) triples
3. Drop interventions with negative average delta
4. Check for Goodhart effects (metric gaming)

**Key risk:** Goodhart's Law — agent may game metrics rather than genuinely improve.

**Files:**
- `experiments/track5_recursive/recursive_learner.py`
- `experiments/track5_recursive/run_recursive_trials.py`
- `experiments/track5_recursive/goodhart_detector.py`
- `experiments/track5_recursive/evaluate_recursive.py`

## Shared Infrastructure

- `experiments/shared/task_bank.py` — 16 tasks across 4 domains
- `experiments/shared/quality_scorer.py` — Heuristic quality rubric
- `experiments/shared/geometric_tracer.py` — GeometryBundle wrapper for tracing

## Verification Commands

```powershell
# Import check
python -c "from experiments.shared.task_bank import get_task_bank"

# Track 4
python experiments/track4_switches/run_baseline_reasoning.py
python experiments/track4_switches/run_switched_reasoning.py
python experiments/track4_switches/evaluate_switches.py

# Track 5
python experiments/track5_recursive/run_recursive_trials.py
python experiments/track5_recursive/evaluate_recursive.py
```
