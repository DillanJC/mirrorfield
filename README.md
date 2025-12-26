# Mirrorfield AI — Evidence-First Boundary Evaluation

**Version:** v0.1 (Phase 0 + Phase A)
**Purpose:** Replayable, auditable AI model boundary stability evaluation
**Platform:** Windows 11 | NVIDIA RTX 3060 Ti | PyTorch 2.5.1 (CUDA 12.4)

---

## Phase 0: Locked Definitions

**Critical Documents:**
- **[Definitions Freeze](docs/DEFINITIONS_FREEZE_v0.1.md)** — Canonical definitions for boundary distance, perturbation policy, borderline rules, and evaluation tiers (prevents definition drift)
- **[Run Ledger](runs/RUN_LEDGER.md)** — Reproducible record of all experimental runs with seeds, artifacts, and environment snapshots

These files are **locked** for Phase A and must not be modified without explicit version bumps.

---

## Quick Start

### Setup
1. Activate the virtual environment: `.\.venv\Scripts\Activate.ps1`
2. Verify GPU: `python tools\gpu_sanity_check.py`
3. Check PyTorch: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### Experiments

#### GPU Playground
Basic training harness to validate GPU pipeline:
```powershell
python experiments\gpu_playground.py
```
Outputs device selection, loss logs every ~20 steps, and final runtime summary.

#### Toy Graph Playground
4×4 grid GNN convergence test:
```powershell
python experiments\toy_graph_playground.py
```
Validates graph neural network implementation (converges to ~0 loss by step ~60).

#### On-Device Jitter Harness v0.1
Trains 4×4 grid GNN, then measures logit stability under noise perturbations:
```powershell
python experiments\jitter_graph_playground.py
```
Emits `experiments/jitter_graph_results_run01.json` with per-node variances, top-k extremes, and timing data.

#### Jitter Analysis
Post-process jitter results:
```powershell
python experiments\analyze_jitter_results.py
```
Prints statistical snapshots and (optionally) plots variance histograms.

---

## Project Status

**Phase 0 (Definitions):** ✅ Complete
- Boundary distance `d(x)`, `d̃(x)` locked
- Perturbation policy `τ(x, ε)` locked (fixed noise for Phase A)
- Borderline slice rule locked
- Tier-2 suite rules defined

**Phase A (Evidence Pack):** 🔄 In Progress
- Artifact structure: `experiments/results/<experiment>/<run_id>/`
- Run ledger seeded with baseline runs
- Next: Timestamped runs with full environment snapshots

**Phase B (Tier-2 Semantic Discriminator):** ⏸️ Blocked (awaits Phase 0+A completion)

---

## Documentation

- **Definitions:** [`docs/DEFINITIONS_FREEZE_v0.1.md`](docs/DEFINITIONS_FREEZE_v0.1.md)
- **Run Ledger:** [`runs/RUN_LEDGER.md`](runs/RUN_LEDGER.md)
- **Tools README:** [`tools/README.md`](tools/README.md)

---

## Engineering Standards

**Reproducibility:**
- All runs use explicit seeds (default: 42)
- Environment snapshots include `torch.__version__`, CUDA version, device name
- Git commit hash + dirty status recorded

**Artifact Discipline:**
- Every experiment emits JSON summary with config + environment + metrics
- Naming: `experiments/results/<experiment_name>/<run_id>/summary.json`
- Run IDs: `YYYYMMDD_HHMMSS` format

**Git Hygiene:**
- Small, reviewable commits
- Prefer adding new files over sweeping refactors
- No uncommitted changes during logged runs

---

**Last Updated:** 2025-12-27
