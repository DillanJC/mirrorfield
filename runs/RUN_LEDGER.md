# Mirrorfield AI — Run Ledger

**Purpose:** Reproducible record of all experimental runs
**Status:** Active tracking log
**Last Updated:** 2025-12-27

---

## Template

```markdown
### Run: <run_id>
**Date:** YYYY-MM-DD (Sydney)
**Git Commit:** <hash> [clean|dirty]
**Command(s):**
```
<exact command as typed>
```
**Seeds/Determinism:**
- Primary seed: <value>
- torch.manual_seed: <value>
- Determinism flags: <TF32 status, etc.>
**Dataset/Suite:**
- Name: <dataset_id>
- Hash: <content_hash>
- N_samples: <count>
**Thresholds:**
- θ_borderline: <value> (or N/A)
- θ_jitter: <value> (or N/A)
- Other: <list>
**Artifacts Produced:**
- Path 1: `<relative path>`
- Path 2: `<relative path>`
**Headline Results:**
- Metric 1: <value>
- Metric 2: <value>
**Notes:**
<any observations, failures, warnings>
```

---

## Seed Runs (Phase A Baseline)

### Run: gpu_sanity_check_001
**Date:** 2025-12-21 (Sydney)
**Git Commit:** TBD [clean]
**Command(s):**
```powershell
python tools\gpu_sanity_check.py
```
**Seeds/Determinism:**
- N/A (deterministic CUDA matmul)
**Dataset/Suite:**
- N/A (synthetic 4096×4096 matmul)
**Thresholds:**
- N/A
**Artifacts Produced:**
- Console output only (no files)
**Headline Results:**
- CUDA available: True
- Device: NVIDIA GeForce RTX 3060 Ti
- Matmul time: ~0.12s (4096×4096)
**Notes:**
Validates PyTorch CUDA build and GPU accessibility. Baseline sanity check for all experiments.

---

### Run: gpu_playground_001
**Date:** 2025-12-21 (Sydney)
**Git Commit:** TBD [clean]
**Command(s):**
```powershell
python experiments\gpu_playground.py
```
**Seeds/Determinism:**
- Seed control: TBD (check script for hardcoded seed)
- torch.manual_seed: TBD
- Determinism flags: default PyTorch behavior
**Dataset/Suite:**
- Synthetic regression (3-layer MLP)
- N_samples: TBD (generated in-script)
**Thresholds:**
- N/A
**Artifacts Produced:**
- Console output only (no files)
**Headline Results:**
- Device: GPU (CUDA)
- Final loss: TBD (check console logs)
- Runtime: ~0.3s
**Notes:**
Basic GPU training harness. Confirms PyTorch training loop works on device.

---

### Run: toy_graph_playground_001
**Date:** 2025-12-21 (Sydney)
**Git Commit:** TBD [clean]
**Command(s):**
```powershell
python experiments\toy_graph_playground.py
```
**Seeds/Determinism:**
- Seed control: TBD (check script)
- Graph: 4×4 grid (deterministic structure)
**Dataset/Suite:**
- 4×4 grid graph (16 nodes)
- Synthetic node features
**Thresholds:**
- Convergence: loss ~0 by step ~60
**Artifacts Produced:**
- Console output only (no files)
**Headline Results:**
- Convergence step: ~60
- Final loss: ~0
**Notes:**
ToyGNN baseline. Validates graph neural network implementation and convergence.

---

### Run: jitter_graph_001
**Date:** 2025-12-21 (Sydney)
**Git Commit:** TBD [clean]
**Command(s):**
```powershell
python experiments\jitter_graph_playground.py
```
**Seeds/Determinism:**
- Seed control: TBD (check script for noise seed)
- Noise: Gaussian, scale TBD
- N_resamples: 100 (jitter passes)
**Dataset/Suite:**
- 4×4 grid graph (16 nodes)
- Post-training jitter test
**Thresholds:**
- Noise scale (ε): TBD
**Artifacts Produced:**
- `experiments/jitter_graph_results_run01.json`
**Headline Results:**
- Per-node variance: see JSON
- Top-k extremes: see JSON
- Training time: TBD
- Jitter time: TBD
**Notes:**
First artifact-producing run. JSON contains full config + env snapshot + variance stats.

---

### Run: analyze_jitter_001
**Date:** 2025-12-21 (Sydney)
**Git Commit:** TBD [clean]
**Command(s):**
```powershell
python experiments\analyze_jitter_results.py
```
**Seeds/Determinism:**
- N/A (post-processing only)
**Dataset/Suite:**
- Input: `experiments/jitter_graph_results_run01.json`
**Thresholds:**
- N/A
**Artifacts Produced:**
- Console statistics
- (Optional) matplotlib histogram if enabled
**Headline Results:**
- Mean variance: TBD
- Max variance: TBD
- Histogram: TBD
**Notes:**
Analysis script for jitter_graph runs. Validates JSON artifact is parseable and complete.

---

## Phase A Complete Runs (Timestamped with Environment Snapshots)

### Run: jitter_graph_20251227_091225
**Date:** 2025-12-27 (Sydney)
**Git Commit:** unknown [unknown] _(repo not yet initialized)_
**Command(s):**
```powershell
python experiments\jitter_graph_playground.py
```
**Seeds/Determinism:**
- Primary seed: 42
- torch.manual_seed: 42
- Determinism flags: TF32=false
**Dataset/Suite:**
- 4×4 grid graph (16 nodes, synthetic features)
- Graph structure: deterministic (8-feature nodes)
- N_samples: 100 (jitter passes)
**Thresholds:**
- Noise scale (ε): 0.05
- k_extremes: 3
**Artifacts Produced:**
- `experiments/results/jitter_graph/20251227_091225/summary.json`
- `experiments/results/jitter_graph/20251227_091225/analysis.json`
- `experiments/results/jitter_graph/20251227_091225/variance_histogram.png`
**Headline Results:**
- Final training loss: 8.20e-08
- Training time: 0.612 s
- Jitter time: 0.043 s
- Mean logit variance: 1.794
- Variance of variances: 0.774
- Min node variance: 0.355 (node 0)
- Max node variance: 3.566 (node 10)
**Environment:**
- torch: 2.6.0+cu124
- CUDA: 12.4
- Device: NVIDIA GeForce RTX 3060 Ti (8.59 GB)
- TF32: false
**Notes:**
First Phase A run with complete environment snapshot and timestamped artifact structure. All three artifacts generated successfully (summary.json, analysis.json, histogram.png).

---

## Phase B Tier-2 Runs (Future)

_Tier-2 semantic discriminator runs will be logged here after Phase B implementation._

---

## Revision History

**v0.1 (2025-12-27):**
- Initial ledger with seed run placeholders
- Template defined
- Baseline runs documented (gpu_sanity_check, gpu_playground, toy_graph_playground, jitter_graph, analyze_jitter)

**Next Update:**
- Fill TBD values by re-running scripts with explicit logging
- Add Phase A timestamped runs with full env snapshots

---

**END OF RUN LEDGER**
