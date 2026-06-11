# -*- coding: utf-8 -*-
"""
Validate the DEPLOYABLE form of the general gate (WORK_MAP 4h follow-up).

4h showed the gate is general up to a per-task z-score — but used full-task
statistics (transductive: each item normalized with stats that include items
that haven't "arrived" yet). A real deployment only has the PAST. This script
answers, from the saved rows (no model re-run):

  1. ROLLING vs TRANSDUCTIVE — replace full-task stats with a causal rolling
     window (only past items). Does pooled AUC hold near 0.63?
  2. LEAVE-ONE-TASK-OUT (the true generality test) — train the gate on two
     tasks, deploy on the third with rolling normalization. The gate has never
     seen the deployment task. Per-task AUC + bootstrap CI + shuffled nulls.
  3. MIXED-STREAM STRESS — rolling window over an interleaved all-task stream
     (no task separation at all). How much does contamination hurt?

Data: experiments/calibrate_gate_rows.npz (750 rows, Qwen2.5-3B, rte/qnli/sst2).
Pure numpy/sklearn, CPU, runs in seconds.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEED = 42
WINDOW = 50          # rolling window size (recent same-context outputs)
WARMUP = 20          # first items of each stream excluded from scoring
N_BOOT = 2000
ROWS = Path(__file__).parent / "calibrate_gate_rows.npz"
OUT = Path(__file__).parent / "validate_rolling_gate_results.json"

rng = np.random.RandomState(SEED)


def rolling_znorm(Xs, window=WINDOW):
    """Causal z-norm: item i normalized by stats of the previous `window` items.

    Items with fewer than 5 past observations get zeros (excluded by warmup).
    """
    n = len(Xs)
    out = np.zeros_like(Xs, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window)
        past = Xs[lo:i]
        if len(past) < 5:
            continue
        mu, sd = past.mean(axis=0), past.std(axis=0) + 1e-9
        out[i] = (Xs[i] - mu) / sd
    return out


def cv_auc(X, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return oof


def boot_ci(y, p):
    vals = []
    for _ in range(N_BOOT):
        bi = rng.randint(0, len(y), len(y))
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(roc_auc_score(y[bi], p[bi]))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def main():
    d = np.load(ROWS, allow_pickle=True)
    X, y, task = d["X"].astype(np.float64), d["y"], d["task"]
    tasks = sorted(np.unique(task))
    print("=" * 70)
    print("VALIDATE THE DEPLOYABLE GATE — rolling z-score + unseen-task transfer")
    print("=" * 70)
    print(f"rows={len(y)}  tasks={tasks}  window={WINDOW}  warmup={WARMUP}\n")
    results = {}

    # ── 1. rolling vs transductive (pooled CV, per-task streams) ───────────
    Xz_roll = np.zeros_like(X)
    keep = np.zeros(len(y), dtype=bool)
    for t in tasks:
        idx = np.where(task == t)[0]
        order = rng.permutation(idx)              # stream order within task
        Z = rolling_znorm(X[order])
        Xz_roll[order] = Z
        keep[order[WARMUP:]] = True               # exclude warmup from scoring

    # transductive reference on the SAME kept subset for a fair comparison
    Xz_full = X.copy()
    for t in tasks:
        m = task == t
        Xz_full[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)

    oof_roll = cv_auc(Xz_roll[keep], y[keep])
    oof_full = cv_auc(Xz_full[keep], y[keep])
    auc_roll, ci_roll = roc_auc_score(y[keep], oof_roll), boot_ci(y[keep], oof_roll)
    auc_full, ci_full = roc_auc_score(y[keep], oof_full), boot_ci(y[keep], oof_full)
    results["1_rolling_vs_transductive"] = {
        "auc_transductive": round(float(auc_full), 4), "ci_transductive": ci_full,
        "auc_rolling": round(float(auc_roll), 4), "ci_rolling": ci_roll,
        "n_scored": int(keep.sum()),
    }
    print("1) ROLLING vs TRANSDUCTIVE (pooled, per-task streams)")
    print(f"   transductive (4h's idealized form): {auc_full:.3f} {ci_full}")
    print(f"   rolling W={WINDOW} (deployable form): {auc_roll:.3f} {ci_roll}\n")

    # ── 2. leave-one-task-out transfer (gate never saw the deploy task) ────
    print("2) LEAVE-ONE-TASK-OUT — train on 2 tasks, deploy on the unseen 3rd")
    loto = {}
    for held in tasks:
        tr_m = task != held
        te_idx = np.where(task == held)[0]
        # offline training: znorm each training task with its own full stats
        Xtr = X[tr_m].copy()
        ttr = task[tr_m]
        for t in np.unique(ttr):
            m = ttr == t
            Xtr[m] = (Xtr[m] - Xtr[m].mean(0)) / (Xtr[m].std(0) + 1e-9)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), y[tr_m])
        # deployment: causal rolling znorm on the unseen task's stream
        order = rng.permutation(te_idx)
        Zte = rolling_znorm(X[order])
        scored = order[WARMUP:]
        Zs = Zte[WARMUP:]
        p = clf.predict_proba(sc.transform(Zs))[:, 1]
        yv = y[scored]
        if len(np.unique(yv)) < 2:
            loto[held] = {"error": "one class after warmup"}
            continue
        auc = roc_auc_score(yv, p)
        ci = boot_ci(yv, p)
        # shuffled-label nulls (5x)
        nulls = []
        for _ in range(5):
            ysh = yv.copy(); rng.shuffle(ysh)
            nulls.append(round(float(roc_auc_score(ysh, p)), 4))
        loto[held] = {"auc": round(float(auc), 4), "ci": ci,
                      "n": int(len(yv)), "n_wrong": int((1 - yv).sum()),
                      "nulls_5x": nulls}
        print(f"   deploy on unseen {held:5s}: AUC {auc:.3f} {ci}  "
              f"(n_wrong={int((1-yv).sum())}, null max={max(nulls):.3f})")
    results["2_leave_one_task_out"] = loto

    # ── 3. mixed-stream stress (no task separation at all) ─────────────────
    order = rng.permutation(len(y))               # fully interleaved stream
    Zmix = rolling_znorm(X[order])
    scored = order[WARMUP:]
    oof_mix = cv_auc(Zmix[WARMUP:], y[scored])
    auc_mix = roc_auc_score(y[scored], oof_mix)
    ci_mix = boot_ci(y[scored], oof_mix)
    results["3_mixed_stream_stress"] = {
        "auc": round(float(auc_mix), 4), "ci": ci_mix, "n_scored": int(len(scored)),
    }
    print(f"\n3) MIXED-STREAM STRESS (interleaved, window sees all tasks)")
    print(f"   rolling W={WINDOW}, no task separation: {auc_mix:.3f} {ci_mix}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT}")
    print("READ: (1) rolling ~ transductive => the deployable form holds.")
    print("      (2) unseen-task CIs > 0.5  => true generality, no task data needed.")
    print("      (3) mixed >> 0.5           => survives even without task separation.")


if __name__ == "__main__":
    main()
