# -*- coding: utf-8 -*-
"""Platt-on-margin baseline vs the frozen calibrator, near the boundary.
(AMENDMENT_1_PLATT_BASELINE.md, locked f0c5b7f — run exactly as specified; CPU-only.)

Fit logistic(correct ~ mm) on one seed's rows, evaluate per-mm-quintile calibration on
the OTHER seed (both directions). Side-by-side with the frozen calibrator's p_int gap.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
MIN_N = 30


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(seed):
    d = np.load(HERE / f"boundary_rows_{seed}.npz", allow_pickle=True)
    correct, mm, p_int = d["correct"], d["mm"], d["p_int"]
    m = (correct >= 0) & np.isfinite(mm) & np.isfinite(p_int)   # identical mask to §4y Opt-2
    return correct[m].astype(int), mm[m].astype(float), p_int[m].astype(float)


def main():
    data = {s: load(s) for s in (42, 1337)}
    out = {}
    for fit_seed, eval_seed in ((42, 1337), (1337, 42)):
        yf, mf, _ = data[fit_seed]
        ye, me_, pe = data[eval_seed]
        clf = LogisticRegression(C=1e6, solver="lbfgs").fit(mf.reshape(-1, 1), yf)
        phat = clf.predict_proba(me_.reshape(-1, 1))[:, 1]
        qs = np.quantile(me_, [0, .2, .4, .6, .8, 1.0])
        bins = []
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            bm = (me_ >= lo) & (me_ <= hi) if i == 4 else (me_ >= lo) & (me_ < hi)
            n = int(bm.sum())
            if not n:
                bins.append({"q": i + 1, "n": 0}); continue
            acc = float(ye[bm].mean())
            wlo, whi = wilson(int(ye[bm].sum()), n)
            mp, fp = float(phat[bm].mean()), float(pe[bm].mean())
            bins.append({"q": i + 1, "n": n,
                         "accuracy": round(acc, 4), "acc_ci": [round(wlo, 4), round(whi, 4)],
                         "platt_mean_p": round(mp, 4), "platt_gap": round(mp - acc, 4),
                         "platt_calib_holds": bool(wlo <= mp <= whi) if n >= MIN_N else None,
                         "frozen_mean_pint": round(fp, 4), "frozen_gap": round(fp - acc, 4)})
        out[f"fit{fit_seed}_eval{eval_seed}"] = {
            "coef": round(float(clf.coef_[0][0]), 4), "intercept": round(float(clf.intercept_[0]), 4),
            "n_fit": len(yf), "n_eval": len(ye), "quintiles": bins}
    # locked verdict logic (amendment): primary contrast = Q1, both directions
    q1 = [out[k]["quintiles"][0] for k in out]
    if all(b["platt_calib_holds"] for b in q1):
        verdict = "FRESH-MAP-CALIBRATED"
    elif all(b["platt_gap"] >= 0.10 and not b["platt_calib_holds"] for b in q1):
        verdict = "FRESH-MAP-ALSO-OVERCONFIDENT"
    else:
        verdict = "MIXED/UNDERPOWERED"
    out["verdict_candidate"] = verdict
    (HERE / "platt_baseline_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'='*84}\n Platt-on-margin baseline vs frozen calibrator (cross-seed held-out; Qwen-3B RTE+QNLI)\n{'='*84}")
    for k, r in out.items():
        if k == "verdict_candidate":
            continue
        print(f"\n {k}  (platt: sigma({r['coef']}*mm + {r['intercept']}); n_fit={r['n_fit']}, n_eval={r['n_eval']})")
        print(f"   {'Q':>2} {'n':>4} {'accuracy [Wilson95]':>26} {'platt_p':>8} {'platt_gap':>9} {'holds':>6} | {'frozen_p':>8} {'frozen_gap':>10}")
        for b in r["quintiles"]:
            print(f"   {b['q']:>2} {b['n']:>4} {b['accuracy']:>8} [{b['acc_ci'][0]:.3f},{b['acc_ci'][1]:.3f}] "
                  f"{b['platt_mean_p']:>8} {b['platt_gap']:>+9} {str(b['platt_calib_holds']):>6} | "
                  f"{b['frozen_mean_pint']:>8} {b['frozen_gap']:>+10}")
    print(f"\n VERDICT CANDIDATE (per locked rules, Dillan concludes): {out['verdict_candidate']}")
    print("saved -> platt_baseline_results.json")


if __name__ == "__main__":
    main()
