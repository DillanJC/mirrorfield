# Mirrorfield — Reconciled Orientation Document

*2026-06-11. Merges the chat-history consolidated map with the verified record
(`WORK_MAP.md`). Where they conflict, the verified record wins — every claim below
is marked. This supersedes the chat-only map as the single orientation document.*

**Canonical source order (corrected):**
1. `WORK_MAP.md` (this repo) — the verified falsification + validation record.
2. This document — orientation on top of it.
3. `REAL_MOE_VALIDATION_PREP.md` and `experiments/SUMmary.md` live in the
   **geometric_safety_features-Experiment** repo (not here, as the chat map claimed).

---

## 1. The former "validated core" — status after the falsification campaign

| Chat-map claim | Verified status |
|---|---|
| Cluster poisoning AUC **0.947** via participation_ratio | **RETRACTED (publicly, v3.0 README, 2026-06-11).** Circular: poison was geometry-defined, then geometry-detected. Geometry-blind and real-backdoor variants score at chance (R0–R3, WORK_MAP §4). |
| **+8.8%** borderline improvement | **NO SOURCE EXISTS.** Appears in no document in either repo. Chat-memory artifact — do not use. Documented relatives (+3.9%/+3.1% R²) were within run-to-run noise. |
| Flip prediction **AUC ≈ 0.707** (N=150) | **PROVISIONAL.** Source doc exists (`docs/BEHAVIORAL_FLIP_UPDATED_FINDINGS.md`) but it is a rescued follow-up: original N=30 test failed (p=0.141), paraphrase-level re-analysis got p=0.040. Single marginal result — the exact pattern that collapsed four times elsewhere. Needs multi-seed/replication before locking. |
| participation_ratio = strongest single signal | Only in circular setups. In the honest non-circular ablation, geometric features added **nothing** over standard log-prob signals (ΔAUC ≈ 0). |
| "Measures behavioral instability, not harm" framing | **Survives** — the r ≈ −0.5 borderline instability correlation is real but weak (~6% variance). A diagnostic hint, not a detector. |

## 2. What is ACTUALLY validated now (the real foundation)

The log-prob uncertainty gate — fully characterized 2026-06 (WORK_MAP §4b–4i):

- Per-task wrong-output prediction: **AUC 0.60–0.74** at 3B (CIs clear 0.5); improves with model scale.
- Cross-task: chance raw (0.53), recovers to **0.63 [0.58, 0.69]** via unsupervised per-task z-score.
- **Transfers to unseen task types**: trained on 2 tasks, deployed on a 3rd — QNLI 0.72, RTE 0.63.
- Hard constraint: rolling buffers must hold **same-context traffic** (mixed streams → chance).
- Geometry adds nothing; N× self-consistency sampling rejected (0.5B artifact, chance at 3B).
- Shipped: leaned MCP server with calibrated P(correct) + per-context `RollingGate`.

## 3. Tracks 4–5 — chat map vs verified

- **GMR AUC 1.0**: the map said "treat with leakage suspicion until reproduced on real data."
  **That reproduction is DONE → chance.** GMR keys on the label flip itself; cluster detection
  is circular by construction. Checklist item closed, negatively.
- **Goodhart negative result**: survives. Agreed — best science of the early project.
- **Distributed Witness +2.5%**: simulated experts only; unverified. Real-MoE validation remains
  open *only if* a signal worth distributing emerges (currently none does — see §5).

## 4. Cross-model witness transfer (the map's §4) — honest reframe

The anchor (relative representations, Moschella et al. ICLR 2023) is real, and the
"coordinate-free features" observation is correct. **But** "upgrades the entire
foundation" presumed the witness signals are valuable — the falsification campaign
showed the only surviving geometric signal is the weak instability correlation.

Honest version of the experiment (still weekend-sized, still interesting):
*does flip-proneness / borderline instability correspond across two embedding models?*
Pre-register the success criterion; expect the weak signal to transfer weakly or not
at all; a strong positive would be the first genuinely new geometric finding since
the retraction. Do not headline it in advance.

## 5. Architecture directions — all parked

Mandala-MoE, fractal witness tree, 2-level witness tree, H₄/120-cell GNN, Julia
substrate: every one of these is a way to **distribute the geometric witness signal**.
That signal did not survive testing. Parked until a signal worth distributing exists
(the §4 reframed experiment is the only live path to one). The map's own discipline
note agrees: elegant-recursive ideas have the project's worst track record.
H₄ specifically: r = 0.42 confirmed as ceiling.

## 6. Sati — closed

Matches the verified record: did not beat class weighting; embedding-geometry
difference from RLHF inconclusive. Additionally (WORK_MAP §4, Mar-21 review): the
"comprehensive Sati training" win was circular — the loss optimized the metric it was
scored on, inflated by a pr/d_eff double-count bug. Thread closed.
Renaissance Protocol: `experiments/RENAISSANCE_PROTOCOL.md` exists in the Experiment
repo — confirm fold-into-Sati and close.

## 7. Falsification log (merged, complete)

From the chats (accurate, independently matches the repo):
- Dark River (discrete dangerous regions) — falsified; embeddings too smooth.
- 4D polytope projection — loses ~61% information; native 256-D wins.
- Triad structure as mechanism — narrative, not mechanistic.
- r = 0.42 on GNN embeddings — ceiling, not floor.
- Nine Realms — did not beat random.

From the verified campaign (2026-06, new since the chat map):
- Geometry detects poison — circular (R0–R3 + real BadNets backdoor: chance).
- Geometry-weighted training helps — circular (metric optimized itself).
- Geometry adds to uncertainty estimation — null (ΔAUC ≈ 0, two tasks).
- Self-consistency sampling — 0.5B artifact (unseeded sampler), chance at 3B.
- Raw cross-task gate — chance without per-context normalization.

## 8. Tooling — current state

- **MCP server (leaned)**: geometry tools removed; `confidence_report` emits calibrated
  `p_correct` + per-context `p_correct_relative` (RollingGate); `novelty_map` kept,
  relabeled interpretive. Moltbook unwired.
- **openclaw `novelty_map`**: resolved by laptop inventory (2026-06-11). It WAS
  committed — to `Geometric_Safety_Features-V2.0.0` (commit 24094d7), which is also
  where this repo's copy came from. Not in the openclaw repo itself. The leaned,
  relabeled version in this repo is current. What was never merged anywhere is the
  **agent layer** that consumes it (`novelty_aware_agent.py`,
  `uncertainty_aware_agent.py`, `openclaw_integration.py`) — now preserved in
  `DillanJC/consolidated-experiments` (kosmos-agent-layer/). Checklist item closed.
- **Multi-AI cross-checking**: the method works — a model-switch review (2026-06-11)
  caught a too-pessimistic conclusion (cross-task gate) and two stale-claim issues.
  Standing caveat from the map holds: model agreement can be correlated error.

## 8b. Cross-machine reconciliation (laptop inventory, 2026-06-11)

The laptop (`LapCl`, OpenClaw/Mercer machine) was inventoried read-only; unique
items were consolidated into **`DillanJC/consolidated-experiments`**. Findings:

- **"+8.8%" is now confirmed absent on BOTH machines** — exhaustive search of all
  repos, docs, and ~160 OpenClaw session transcripts found nothing. The number is
  permanently closed: chat-only artifact, never use it.
- **Fractal observer has no code or notes anywhere** — it exists as one line in chat
  history ("it might not be on the laptop" — correct, and not on the PC either).
  The chat map's "most build-ready item" was never started.
- **No sati / witness / mandala / GMR / polytope / renaissance material on the
  laptop** — everything in those threads lives in the PC repos. Nothing was lost.
- **Softmax-vs-geometric comparison** (laptop, untracked, now in
  consolidated-experiments): a synthetic 2D demo (make_moons + planted blobs) where
  geometry "beats" softmax on OOD/poisoned scenarios. Verdict on inspection: the
  anomalies are geometrically planted — the same circular pattern as the retracted
  poison results — and the geometric risk score reads high on EVERY scenario
  including clean (no discrimination). It does not contradict the verified
  non-circular result (geometry adds nothing for wrong-output gating). Its one fair
  point — neighborhood features can flag distribution shift softmax misses — is
  established literature, not a project finding.
- **The unmerged agent layer** (`kosmos-agent-layer/` in consolidated-experiments):
  agent wrappers that consume the MCP uncertainty tools (adaptive epistemic modes,
  terrain navigation). Built for the old 7-tool server; would need adapting to the
  leaned server. **This is the natural starting point for Road A's "wire the gate
  into an agent loop" step.**
- Publication assets (project website + zips, Manim video scripts) and the Mercer
  workspace (execution engine, bridge, skills, daily logs Feb 2026) are preserved
  in consolidated-experiments; laptop also carries a moltbook_bridge credential
  patch (moltbook is unwired here — pull only if moltbook returns).
- Laptop credentials hygiene: multiple key files under `.openclaw\`, `.hermes\`,
  `.config\moltbook\` (paths logged in the laptop inventory); one GLM key known
  stale since April. Rotate on next laptop session.

## 9. Open leads — reconciled checklist

- [ ] Cross-model instability transfer, **honest reframe only** (§4 above) — the one
      live experimental lead.
- [x] Behavioral-flip AUC 0.707 — **RETIRED 2026-06-12** (Plan B Phase 0,
      WORK_MAP §4j): in-sample artifact; honest out-of-fold AUC 0.49 (chance),
      permutation p=0.48. May never be cited. The citable set has no
      provisional numbers left.
- [x] Renaissance Protocol — confirmed and CLOSED 2026-06-12. Read the design
      doc (`experiments/RENAISSANCE_PROTOCOL.md`, Experiment repo): it remaps
      the 14 geometric features to aesthetic functions ("safety as emergent
      beauty") — Sati lineage, built wholly on features the falsification
      campaign showed carry no validated detector signal. No empirical
      foundation to build on; the document stands as a creative artifact only.
- [x] GMR AUC 1.0 real-data reproduction — done, chance; closed.
- [x] novelty_map commit state — resolved (§8); survivor is in this repo; unmerged
      agent layer preserved in consolidated-experiments.
- [x] "+8.8%" — confirmed absent on both machines (§8b); permanently closed.
- [x] Road A integration — **DONE 2026-06-12 (Plan C, WORK_MAP §4k): pre-registered
      PASS.** GatedAgent + eval (live AUC 0.685 [0.64, 0.73]; gate catches 27.7%
      of errors at 14.7% abstention vs 14.7% random; CI clears zero), 25/25
      MCP parity, watchable demo (`experiments/run_gate_demo.py --run`).
      The old kosmos-agent-layer served as pattern donor only, as planned.
- [x] Number-consistency before outreach — **the locked set is now the v3.0 set**:
      per-task gate 0.60–0.74; rolling cross-task 0.63 [0.58, 0.69]; unseen-task
      transfer 0.72/0.63; geometry Δ ≈ 0; borderline instability r ≈ −0.5 (weak).
      The old set (+8.8% / 0.707 / 0.947) is respectively: unsourced / provisional /
      **publicly retracted**. Any outreach using it would contradict the project's
      own published retraction.

---

*Throughline, updated: the validated foundation is no longer the geometric detection
stack — it is the calibrated, context-relative log-prob gate, plus the falsification
methodology itself. The architecture ideas remain parked until the reframed
cross-model experiment (or something else) produces a geometric signal that survives
the project's own standards.*
