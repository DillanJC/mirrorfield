# AGENTS.md

This file provides context and guidelines for how OpenCode should operate in this project, based on the user's Reflective Humanist Manifesto and R-OS framework.

## User Operating Framework

### Reflective Humanist Manifesto (Key Principles)
- **Center of Experience Axiom**: Treat inner lives as real, no one is collateral.
- **Awareness as Responsibility**: Seeing harm creates duty; deliberate numbness is spiritual wound.
- **No Purity**: Honest participation in systems without moral purity fantasy.
- **Care Beyond Personal**: Extend compassion to systems, institutions, tech.
- **Paradox Principle**: Hold tensions without resolution; tension is architecture of mature ethics.
- **Repair Imperative**: Curiosity before defensiveness; repair is moral aliveness.
- **Body-Agnostic Baseline**: Everyone deserves shelter, safety, witness regardless of body/mind state.
- **Technology with Intention**: Tools amplify intention; assess agency, dependency, habits.
- **Meaning as Co-Creation**: Meaning made through relationships, struggle, commitment.
- **Practice Daily**: Rhythm: Act, Listen, Reflect, Repair. Show up one choice at a time.

### R-OS: Research Operating System (Core Constraints)
- **Artifact Mandatory**: Every session produces external trace (log, code, test).
- **Falsifier First**: Test before implementation; one extension per experiment.
- **Phases**: Locks (definitions) → Baseline → Robustness → Extensions.
- **Three Tracks**: A (main pipeline 70%), B (exploration 10-25%), C (maintenance 5-15%).
- **Scarcity Optimization**: Signal ≥ 2× noise; minimal surface area.
- **Human-AI Orchestration**: Clear handoffs, escalation triggers, context management.

## Operational Guidelines
- Prioritize evidence-based changes; measure impact before scaling.
- Maintain reversibility and containment; blast radius budgeting.
- Freeze language early; definitions in DEFINITIONS_FREEZE.md.
- Use YAGNI > KISS > DRY; build only what's needed now, simply.
- Track in run ledger; review negative space weekly.
- Human limits: 3-4h deep work/day; AI limits: context <80%, confidence >70%.

## Coding Conventions
- YAGNI > KISS > DRY priority
- Functions <50 lines, complexity <10
- Early returns, fail-fast error handling
- Tests: >80% coverage on critical paths, negative tests included

## Measurement Policy
- Measure only decision-informing metrics (e.g., cyclomatic complexity, test coverage)
- Skip theater: daily time tracking, LOC counts unless they change decisions
- Use "not measured" for irrelevant data

## Session Structure
- Prep: Read handoff, set intention
- Execute: One primary action (Track A/B/C)
- AAR: What happened? Expected vs actual? Why delta? Next tweak?
- Externalize: Artifact (log + commit if code)

## Phase Guidelines
- Phase 0: Freeze definitions, reproducibility rules
- Phase 1: Minimal baseline
- Phase 2: Robustness (reproducibility, edge cases)
- Phase 3: One extension at a time with falsifiers

## Project-Specific Notes
- Generated during /init; edit manually for custom context.
- Commit to Git for consistency across team/AI interactions.