# -*- coding: utf-8 -*-
"""
Export the RELATIVE gate calibration — the model behind the rolling buffer.

The validated deployable pipeline (WORK_MAP 4h/4i) is:
    rolling z-score of features against recent same-context traffic
    -> logistic trained on z-normed features -> isotonic -> P(correct)

The existing gate_calibration.json was trained on RAW features (the 4g pooled
model, near-base-rate). This script trains the z-normed-feature model the
leave-one-task-out validation actually used, on all 750 saved rows, and exports
it as gate_calibration_relative.json (same numpy-applicable format).

Inputs at runtime must be ROLLING-Z-NORMED features, not raw ones.
CPU-only, seconds, no model re-run (reads calibrate_gate_rows.npz).
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
ROWS = Path(__file__).parent / "calibrate_gate_rows.npz"
EXPORT = Path(__file__).resolve().parent.parent / "mirrorfield" / "mcp" / "gate_calibration_relative.json"


def main():
    d = np.load(ROWS, allow_pickle=True)
    X, y, task = d["X"].astype(np.float64), d["y"], d["task"]

    # per-task z-norm (the validated offline form; rolling approximates it, 4i test 1)
    Xz = X.copy()
    for t in np.unique(task):
        m = task == t
        Xz[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-9)

    # honest out-of-fold check before exporting
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(Xz, y):
        sc = StandardScaler().fit(Xz[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xz[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xz[te]))[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"OOF AUC of the relative model: {auc:.4f} (expect ~0.63, matching 4h)")

    # final fit on all rows for export
    scaler = StandardScaler().fit(Xz)
    logreg = LogisticRegression(max_iter=1000).fit(scaler.transform(Xz), y)
    raw = logreg.predict_proba(scaler.transform(Xz))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    xs = np.linspace(0, 1, 101)

    export = {
        "features": ["z_mean_margin", "z_mean_entropy", "z_boundary_ratio"],
        "input_note": "Inputs MUST be rolling z-scores of the raw features against "
                      "recent SAME-CONTEXT traffic (window ~50, >=5 history). "
                      "Raw features go to gate_calibration.json instead.",
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "logreg_coef": [round(float(v), 6) for v in logreg.coef_[0]],
        "logreg_intercept": round(float(logreg.intercept_[0]), 6),
        "isotonic_x": [round(float(v), 4) for v in xs],
        "isotonic_y": [round(float(v), 4) for v in iso.predict(xs)],
        "trained_on": {"model": str(d["model"][0]), "tasks": sorted(set(task.tolist())),
                       "n": int(len(y)), "oof_auc": round(float(auc), 4)},
        "validation": "WORK_MAP 4h/4i: pooled 0.634 [0.579,0.689]; unseen-task "
                      "transfer QNLI 0.723 / RTE 0.633; mixed-context streams "
                      "break the normalization (use per-context buffers).",
    }
    EXPORT.write_text(json.dumps(export, indent=2))
    print(f"Exported -> {EXPORT}")


if __name__ == "__main__":
    main()
