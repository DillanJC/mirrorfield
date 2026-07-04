# Mirrorfield — an honest, modest uncertainty gate for LLM outputs

> **⚠️ Status (2026-06, v3.0 — please read first).** This project began as the
> embedding-geometry "boundary evaluation" framework documented below (Phase 0–E,
> Tracks 4–5). It then put that framework through its own *non-circular*
> falsification — and **most of the geometry did not survive.** The
> embedding-geometry add-on adds ≈0 predictive lift over standard log-prob signals
> (ΔAUC ≈ 0); several headline results in the legacy sections below were circular
> or measured the wrong thing. Read those numbers with that in mind.
>
> **What survived, and is now the actual tool:** a *modest* log-prob uncertainty
> gate — from token margin/entropy it estimates how likely an answer is wrong
> (live AUC ≈ 0.685; catches ~28% of errors at ~15% abstention), composed with a
> harm classifier into a **SEND / VERIFY / HOLD** decision *before* an answer is
> sent. It is honestly a ranking signal that improves the odds, not an oracle.
>
> - **Use it:** [`mirrorfield/mcp/README.md`](mirrorfield/mcp/README.md) · `pip install -e ".[mcp]"`
> - **The full, honest research log** (everything tried — with what was falsified and why): [`WORK_MAP.md`](WORK_MAP.md)
> - **Where things stand / resume point:** [`HANDOFF.md`](HANDOFF.md)
> - **Current work (July 2026):** the same discipline, turned on the surviving gate —
>   its aggregate calibration hides a near-boundary failure (overconfident ~+0.22 in the
>   torn region, replicated + audited, *on this model*): [`docs/METHODS_NOTE.md`](docs/METHODS_NOTE.md)
>   · methods checklist: [`docs/SAFETY_CLAIM_SMELL_TEST.md`](docs/SAFETY_CLAIM_SMELL_TEST.md)
>
> The geometry-framework documentation below is kept for the record (and for
> reproducing the falsification), **not** as a list of standing claims.

**Platform:** Windows 11 | NVIDIA RTX 3060 Ti | PyTorch (CUDA 12.x)

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

#### Phase D: Integrated Evaluation Pipeline

**Run Comprehensive 4-Mode Evaluation (OPTIMIZED - 302× faster)**
```powershell
python experiments\phase_d_integrated_eval_fast.py ^
    --model-checkpoint experiments\results\tier2_train\<run_id>\model_checkpoint.pt ^
    --reference-stats experiments\results\tier2_reference\<run_id>\summary.json ^
    --transform-suite runs\tier2_transforms_v1.json ^
    --calibration-artifact runs\calibration_tau_*.json ^
    --friction-artifact runs\friction_tags_*.json ^
    --n-perturbations 10 ^
    --seed 42
```
Combines Phases B+C into unified evaluation with 4 modes:
1. Semantic-only: Baseline transform evaluation
2. Perturbation-only: Noise injection stratified by friction
3. Combined: Semantic + perturbation compound effects
4. Stratified analysis: Cross-mode friction/category breakdown

Outputs: `experiments/results/phase_d_integrated_eval/<run_id>/`
- `summary.json` - Comprehensive cross-mode summary
- `semantic_results.json` - Per-transform semantic metrics
- `perturbation_results.json` - Per-sample perturbation metrics
- `combined_results.json` - Compound effect analysis
- `stratified_analysis.json` - Friction/category breakdowns

**Performance:** ~23 seconds per run on RTX 3060 Ti (302× speedup from batched embeddings)

**Validation:** Validated across 20 independent seeds (100% consistency). See `runs/phase_d_validation_manifest.json` and `docs/PHASE_D_DEEP_ANALYSIS_v1.0.md`.

#### Phase E: Geometry Bundle

**Run Geometry Validation (Synthetic Data)**
```powershell
python experiments\validate_phase_e_on_real_data.py --synthetic --n-ref 1000 --n-query 500 --device cpu
```
Tests geometry features (local curvature + ridge proximity) on synthetic embeddings.
Outputs: Falsifier verdict (REDUNDANT/COLLAPSED/COSMETIC/REAL_SIGNAL/WEAK_SIGNAL)

**Run Multi-Seed Validation**
```powershell
python experiments\phase_e_multiseed_validation.py --n-seeds 10 --device cpu
```
Validates geometry bundle across multiple seeds (robustness check).
Outputs: `runs/phase_e_multiseed_validation_<timestamp>.json`

**Benchmark Performance**
```powershell
python experiments\benchmark_phase_e_svd_curvature.py --n-query 2000 --device cpu
```
Measures isolated SVD curvature performance.
Performance: ~21k queries/sec (CPU), ~1k queries/sec (GPU)

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

**Phase B (Tier-2 Semantic Discriminator):** ✅ Complete
- MVP implementation complete (train → generate → evaluate → analyze)
- Binary sentiment classifier on synthetic dataset
- Semantic transform suite with LLM-assisted generation
- Boundary distance metrics: d(x), d̃(x), Δd̃, FlipRate
- End-to-end pipeline validated with 6 spot-checks

**Phase C (Calibration + Friction Tagging):** ✅ Complete
- Perturbation calibration: Binary search finds ε achieving 5-10% flip rate
  - Calibrated ε = 0.0166 (flip rate 8.3%)
  - 3-way dataset split (train/calib/eval: 60/20/20)
- Friction tagging: Categorizes samples by expected difficulty
  - Low friction: 64.3% (|d̃(x)| ≥ 0.5)
  - Medium friction: 18.1% (0.25 ≤ |d̃(x)| < 0.5)
  - High friction: 17.6% (|d̃(x)| < 0.25)
- Canonical artifacts: `runs/calibration_tau_*.json`, `runs/friction_tags_*.json`

**Phase D (Integrated Evaluation):** ✅ Complete + Validated (20 seeds)
- Unified 4-mode evaluation pipeline combining Phases B+C
- **Performance:** 302× speedup via batched embeddings (~23s per run vs 116 minutes)
- **Validation:** 20 independent seeds tested (100% consistency)
  - Mean stratification: 35.2× (low → high friction)
  - Statistical significance: p < 0.0001
  - Bias checks: All passed (random seeds + negative control)
- **Friction stratification (seed 42 baseline):**
  - Low friction: 1.9% flip rate | Medium: 11.8% | High: 25.9%
  - Stratification: 13.6× difference (low → high)
- **Key finding (VALIDATED):** d̃(x) is a robust, seed-independent predictor of perturbation robustness
- **Artifacts:** Validation manifest, 20 run directories, negative control script
- **Documentation:** [`docs/PHASE_D_DEEP_ANALYSIS_v1.0.md`](docs/PHASE_D_DEEP_ANALYSIS_v1.0.md)

**Phase E (Geometry Bundle):** ✅ Complete + Validated (10 seeds)
- Geometry features: Local curvature (GPU-batched low-rank SVD) + ridge proximity (density gradient)
- 5-verdict falsifier: REDUNDANT/COLLAPSED/COSMETIC/REAL_SIGNAL/WEAK_SIGNAL
- **Implementation:** 2,930 lines (production + tests + benchmarks)
- **All acceptance tests passing:**
  - SVD equivalence (math correctness, tolerance 1e-6)
  - Batch independence (reproducibility guarantees)
  - Phase D→E integration (contract validation)
- **Multi-seed validation (10 seeds):** 100% consistency
  - Verdict: 10/10 COSMETIC (on synthetic data)
  - Ridge independence: 10/10 PASS (corr(ridge, bd) = -0.021±0.036)
  - ΔR² = 0.0017±0.0021 (geometry adds 0.17% explanatory power)
  - Geometry features highly stable (curvature std = 0.0003)
- **Performance:** 21k queries/sec (CPU), 0.18s per 500-query validation
- **Scientific finding (on synthetic data):** Geometry does not add meaningful explanatory power beyond boundary distance when no geometric structure exists (COSMETIC verdict). This validates the falsifier is not biased toward false positives.
- **Key validation:** Falsifier verdict is honest - correctly identifies lack of signal on uncorrelated synthetic data
- **Artifacts:** 10-seed validation manifest, test suite, benchmarks
- **Next:** Real Phase D data integration (optional - synthetic validation establishes correctness)

**Track 4 (Architectural Switches):** IN PROGRESS
- Hard-coded geometric interventions mapping signatures to reasoning corrections
- `python experiments\track4_switches\evaluate_switches.py` — Run full evaluation
- Report: [`experiments/track4_switches/TRACK4_REPORT.md`](experiments/track4_switches/TRACK4_REPORT.md)

**Track 5 (Recursive Self-Learning):** IN PROGRESS
- Adaptive intervention policy that learns from quality feedback
- Includes Goodhart's Law detection for metric gaming
- `python experiments\track5_recursive\evaluate_recursive.py` — Run full evaluation
- Report: [`experiments/track5_recursive/TRACK5_REPORT.md`](experiments/track5_recursive/TRACK5_REPORT.md)

---

## Documentation

- **Phase A Summary:** [`docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md`](docs/PHASE_A_COMPLETION_SUMMARY_v1.0.md) — Evidence pack completion report
- **Phase B Transform Validation:** [`docs/TIER2_TRANSFORM_EXAMPLES.md`](docs/TIER2_TRANSFORM_EXAMPLES.md) — Transform validation log template (≥5 spot-checks)
- **Phase D Deep Analysis:** [`docs/PHASE_D_DEEP_ANALYSIS_v1.0.md`](docs/PHASE_D_DEEP_ANALYSIS_v1.0.md) — 20-seed validation, bias checks, stratification analysis
- **Definitions:** [`docs/DEFINITIONS_FREEZE_v0.1.md`](docs/DEFINITIONS_FREEZE_v0.1.md) — Canonical definitions (Phase 0 lock)
- **Run Ledger:** [`runs/RUN_LEDGER.md`](runs/RUN_LEDGER.md) — Reproducible run tracking
- **Validation Manifest:** [`runs/phase_d_validation_manifest.json`](runs/phase_d_validation_manifest.json) — 20-seed validation artifact
- **Tools README:** [`tools/README.md`](tools/README.md) — GPU playground documentation
- **Track 4 Report:** [`experiments/track4_switches/TRACK4_REPORT.md`](experiments/track4_switches/TRACK4_REPORT.md) — Architectural switches evaluation
- **Track 5 Report:** [`experiments/track5_recursive/TRACK5_REPORT.md`](experiments/track5_recursive/TRACK5_REPORT.md) — Recursive self-learning evaluation
- **Track Status:** [`docs/PHASE1_2_5_STATUS.md`](docs/PHASE1_2_5_STATUS.md) — Experiment track status
- **Changelog:** [`CHANGELOG.md`](CHANGELOG.md) — Version history

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

**Last Updated:** 2026-02-08
