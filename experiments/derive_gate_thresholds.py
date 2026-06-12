# -*- coding: utf-8 -*-
"""
Plan C Step 2 — pre-register gate thresholds BEFORE any eval data exists.

Derives tau_abstain / tau_present from the 750 calibration-era rows
(calibrate_gate_rows.npz) streamed through the shipped RollingGate, and writes
gate_thresholds.json containing the thresholds, the order seed, stability
checks (seeds 43/44), and the pre-registered success/failure criteria verbatim.
This file + JSON must be committed to git before eval_gate_value.py generates
anything. CPU-only, seconds. See plans/C-agent-loop-integration.md Step 2.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mirrorfield.mcp.uncertainty import RollingGate  # noqa

ROWS = Path(__file__).parent / "calibrate_gate_rows.npz"
OUT = Path(__file__).parent / "gate_thresholds.json"
WARMUP = 20  # matches validate_rolling_gate.py / WORK_MAP 4i

CRITERIA = {
    "P1": "pooled fresh RTE+QNLI wrong-output AUC >= 0.58 AND 95% bootstrap CI "
          "lower bound > 0.50 (offline locked band 0.63 [0.58,0.69]); per-task "
          "AUCs+CIs co-registered for the disambiguation rule",
    "P2": "at frozen tau_abstain, 95% bootstrap CI of (error-recall among "
          "abstained - REALIZED abstention rate) has lower bound > 0; both "
          "recomputed inside every resample; threshold fixed. Null is exactly "
          "zero regardless of score drift - the gate cannot loosen its own bar",
    "S1": "stretch (MRPC unseen-task): AUC CI lower bound > 0.50; failing S1 "
          "bounds the transfer claim, does not fail the plan",
    "random_gate_verdict": "P1 passes but P2 CI includes 0 -> publish AUC + "
          "risk-coverage curve; demo ships labeled 'modest gate - ranking "
          "signal real, operating-point value not demonstrated'; the phrase "
          "'the gate helps' is NOT used",
    "disambiguation": "if pooled P1 fails but QNLI-alone CI lower bound > 0.50: "
          "verdict 'RTE-train-arm shift - foundation intact, transfer claim "
          "bounded'; abandon rule does NOT fire",
    "abandon": "pooled P1 fails AND QNLI-alone fails AND the debugging pass "
          "reproduces 0.63 on saved calibration rows AND one focused bug-hunt "
          "finds nothing -> foundation in question; write up as such. No "
          "threshold tuning, no metric swaps, no second operating point",
    "controls": "mixed-stream shared-buffer AUC must drop >=0.05 below "
          "per-context (expected ~chance); 10 shuffled-label perms mean in "
          "[0.45,0.55] max < 0.58; P1 must hold across stream orders "
          "123/43/44 with AUC spread < 0.06",
}


def stream_scores(X, task, order_seed):
    """Per-task streams through fresh RollingGates; returns scores + meta."""
    rng = np.random.RandomState(order_seed)
    gate_scores = np.full(len(task), np.nan)
    pos = np.full(len(task), -1)
    for t in sorted(np.unique(task)):           # qnli, rte, sst2 (sorted)
        idx = np.where(task == t)[0]
        order = rng.permutation(idx)            # shared rng, advances per task
        g = RollingGate(window=50, min_history=5)
        for j, i in enumerate(order):
            r = g.score(str(t), *X[i])
            pos[i] = j
            if r["p_correct_relative"] is not None:
                gate_scores[i] = r["p_correct_relative"]
    return gate_scores, pos


def derive(order_seed, X, task):
    scores, pos = stream_scores(X, task, order_seed)
    m = (pos >= WARMUP) & np.isin(task, ["rte", "qnli"]) & ~np.isnan(scores)
    pool = scores[m]
    return (round(float(np.percentile(pool, 20)), 4),
            round(float(np.percentile(pool, 50)), 4), int(m.sum()))


def main():
    d = np.load(ROWS, allow_pickle=True)
    X = np.round(d["X"].astype(np.float64), 4)   # mirror server.py rounding
    task = d["task"]

    tau_a, tau_p, n = derive(42, X, task)
    stab = {s: derive(s, X, task)[:2] for s in (43, 44)}
    print(f"frozen (seed 42): tau_abstain={tau_a}  tau_present={tau_p}  "
          f"(n_pool={n})")
    for s, (a, p) in stab.items():
        print(f"stability seed {s}: tau_abstain={a}  tau_present={p}")

    OUT.write_text(json.dumps({
        "tau_abstain": tau_a, "tau_present": tau_p,
        "order_seed": 42, "warmup": WARMUP,
        "derived_from": "calibrate_gate_rows.npz, post-warmup RTE+QNLI "
                        "p_correct_relative only (SST-2 excluded), features "
                        "rounded to 4dp matching server.py",
        "n_pooled_scores": n,
        "stability_check_seeds_43_44": {str(k): list(v) for k, v in stab.items()},
        "decision_policy": "ABSTAIN if p_rel < tau_abstain; PRESENT if "
                           "p_rel >= tau_present; else VERIFY; "
                           "warming_up -> VERIFY",
        "preregistered_criteria": CRITERIA,
    }, indent=2))
    print(f"saved -> {OUT}\nCommit this BEFORE running eval_gate_value.py.")


if __name__ == "__main__":
    main()
