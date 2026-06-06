# Geometric Safety Features — v3.0

> **Status: honest retraction + one validated keeper.**
> This release supersedes v2.0.0. After four independent tests — including one the
> method could not rig — the central claim (that embedding *geometry* detects unsafe
> or poisoned data) did **not** survive. What did survive is a standard log-prob
> uncertainty gate. This README documents both, because a negative result that is
> rigorously established is worth more than a positive one that isn't.

---

## TL;DR — what changed since v2.0.0

| v2.0.0 claimed | v3.0 verdict | Why |
|----------------|--------------|-----|
| Cluster poison **AUC 0.947** | **Withdrawn** | Poison was *defined by* geometry (KMeans cluster), then "detected" by geometry. Circular. |
| Boundary poison **AUC 0.785** | **Withdrawn** | Same circularity (poison = boundary distance). Also a duplicate-feature bug inflated later runs. |
| Random poison **AUC 0.653** | **Withdrawn** | An honest geometry-blind version drops to **0.47 — chance**. |
| **+3.9% / +3.1% R²** from geometric features | **Within noise** | No CI / multi-seed; effect sits inside run-to-run variance. |
| Behavioral instability **r ≈ −0.5** (borderline PR) | **Kept (weak)** | Real but modest (~6% variance). Not a detector. |
| MCP geometric uncertainty signatures | **Reframed** | Geometry adds nothing over standard signals; the *standard* signals work. See below. |

---

## The one thing that works: a standard uncertainty gate

The original motivation was sound: **can a model flag a likely-wrong output before it
is sent?** The answer is yes — but with textbook log-probability signals, not geometry.

Non-circular test (`mcp_uncertainty_ablation.py`): a small instruct model classifies
300 balanced SST-2 sentences; the label is **ground-truth correctness** (independent of
any geometry). Predicting "this output is wrong":

| Feature set | AUC |
|-------------|-----|
| **Standard** (token margin + entropy + boundary ratio) | **0.667** |
| Geometry alone (sequence PR + embedding PR) | 0.535 (≈ chance) |
| Standard + geometry | 0.668 |
| Δ from adding geometry | **+0.001**, 95% CI **[−0.05, +0.05]** |
| Shuffled-label null control | −0.07 (passes) |

**Conclusion:** ship the gate on margin + entropy + boundary ratio. The geometry layer
is decorative and is being removed. The MCP server remains useful as an honest
uncertainty-awareness tool (present / verify / abstain), minus the geometric claims.

---

## Why the poison-detection results collapsed (the circularity)

Every headline poison "win" in v1.x–v2.0 shared one flaw: the poison was **generated**
using an embedding-geometry rule (a dense cluster, or distance-to-boundary), and then
**detected** with embedding geometry. A geometric magnet was finding a geometric needle
it had placed itself.

A blind test held the detector fixed and varied only how poison was generated:

| Poison defined by | Geometry-honest AUC |
|-------------------|---------------------|
| Geometry (KMeans cluster) — the original setup | 0.60 |
| Random ids, label only (geometry-blind) | 0.47 (chance) |
| Embedding-space trigger patch | 0.48 (chance) |
| **Real text backdoor** (SST-2 + BadNets `cf` trigger) | 0.50 (chance once the trigger varies) |

The real-backdoor case is decisive: a first pass looked promising (AUC 0.72), but
controls showed it was an artifact of a *single repeated trigger token* forming one
shared cluster — i.e. the circular setup in disguise. Give each poisoned sample a
*different* trigger and detection drops to chance.

**Bottom line:** geometric features detect poison only when the poison is itself a dense
geometric cluster. Against a real, attacker-chosen backdoor they are at chance.

---

## What honestly survives

- **A working uncertainty gate** on standard log-prob signals (AUC 0.667, non-circular).
- **A weak behavioral-instability correlation** (borderline participation ratio,
  r ≈ −0.5; ~6% of variance). Real, but not a detector on its own.
- **A Goodhart / metric-gaming detector** built during the intervention experiments,
  which correctly caught a self-silencing failure mode.
- **The methodology itself** — a reusable, regime-agnostic falsification harness for
  testing whether a "detector" is actually circular.

## What does not

- Poison detection by geometry as a general capability (circular; chance on real attacks).
- Geometry-weighted *training* "improving" models (the success metric optimized the very
  feature it was scored on — tautological).
- Distributed-witness ensemble gains (simulated experts only).
- The H4 / 120-cell fractal-geometry bet (ceiling correlation r ≈ 0.42).
- Live geometric interventions improving output quality (net neutral-to-negative).

---

## Reproducing the key results

| Finding | Script |
|---------|--------|
| Blind circularity test (R0–R2) | `experiments/track1_poison/test_filter_blind.py` |
| Real backdoor + confound controls (R3) | `test_filter_r3_real.py`, `test_filter_r3_controls.py` |
| MCP uncertainty ablation (the keeper) | `experiments/mcp_uncertainty_ablation.py` |

All run locally (CPU/GPU), no paid API required. Each writes a JSON results file and
includes a negative control (shuffled labels must give ≈ chance / Δ ≈ 0).

---

## Citation note

If citing v2.0.0's poison-detection AUCs, please cite this v3.0 retraction instead — the
v2.0.0 numbers reflect a circular evaluation and overstate real-world performance.

---

*v3.0 — supersedes v2.0.0. The project's lasting contributions are an honest negative
result on geometric poison detection and a small, validated log-prob uncertainty gate.*
