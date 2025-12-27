# Mirrorfield AI — Evidence-First Boundary Evaluation

**Version:** v0.3 (Phase 0 + Phase A + Phase B Complete)
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

#### Phase B: Tier-2 Semantic Discriminator

**1. Train Binary Sentiment Classifier + Compute Reference Stats**
```powershell
python experiments\tier2_classifier_train.py
```
Trains classifier on 2000 synthetic sentiment samples, computes reference set statistics (μ_ref, σ_ref).
Outputs: `experiments/results/tier2_train/<run_id>/` and `experiments/results/tier2_reference/<run_id>/`

**2. Generate Semantic Transform Suite (Interactive)**
```powershell
python experiments\tier2_transform_generate.py
```
LLM-assisted transform generation with human validation (≥5 spot-checks required).
Displays prompts to copy/paste into GPT-4/Claude, parses JSON output, validates interactively.
Outputs: `runs/tier2_transforms_v1.json`

**3. Run Semantic Evaluation**
```powershell
python experiments\tier2_semantic_eval.py ^
    --model-checkpoint experiments\results\tier2_train\<run_id>\model_checkpoint.pt ^
    --reference-stats experiments\results\tier2_reference\<run_id>\summary.json ^
    --transform-suite runs\tier2_transforms_v1.json
```
Computes boundary distance metrics (d(x), d̃(x), Δd̃, FlipRate) for all transforms.
Outputs: `experiments/results/tier2_semantic_eval/<run_id>/`

**4. Analyze Results**
```powershell
python experiments\analyze_tier2_results.py --input experiments\results\tier2_semantic_eval\<run_id>\summary.json
```
Prints human-readable summary and expectation checking (preserving <0.5, changing >1.5).
Optional: generates Δd̃ histogram by category.

---

## Project Status

**Phase 0 (Definitions):** ✅ Complete
- Boundary distance `d(x)`, `d̃(x)` locked
- Perturbation policy `τ(x, ε)` locked (fixed noise for Phase A)
- Borderline slice rule locked
- Tier-2 suite rules defined

**Phase A (Evidence Pack):** ✅ Complete
- Artifact structure: `experiments/results/<experiment>/<run_id>/`
- All baseline experiments producing timestamped artifacts
- Full environment snapshots (torch, CUDA, device, git)
- Run ledger with 5 documented runs
- See: [`docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md`](docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md)

**Phase B (Tier-2 Semantic Discriminator):** 🚧 In Progress
- MVP implementation complete (train → generate → evaluate → analyze)
- Binary sentiment classifier on synthetic dataset
- Semantic transform suite with LLM-assisted generation
- Boundary distance metrics: d(x), d̃(x), Δd̃, FlipRate
- ✅ End-to-end pipeline validated with 6 spot-checks

---

## Documentation

- **Phase A Summary:** [`docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md`](docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md) — Evidence pack completion report
- **Phase B Transform Validation:** [`docs/TIER2_TRANSFORM_EXAMPLES.md`](docs/TIER2_TRANSFORM_EXAMPLES.md) — Transform validation log template (≥5 spot-checks)
- **Definitions:** [`docs/DEFINITIONS_FREEZE_v0.1.md`](docs/DEFINITIONS_FREEZE_v0.1.md) — Canonical definitions (Phase 0 lock)
- **Run Ledger:** [`runs/RUN_LEDGER.md`](runs/RUN_LEDGER.md) — Reproducible run tracking
- **Tools README:** [`tools/README.md`](tools/README.md) — GPU playground documentation

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
