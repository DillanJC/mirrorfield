# -*- coding: utf-8 -*-
"""Refit runner — executes PREREGISTRATION_DRAFT.md once LOCKED.

Enforcement beats prose: --fit refuses to run while the pre-reg is still a DRAFT.
To lock: Dillan reads/edits the draft, renames it PREREGISTRATION.md, commits.
That commit hash is the lock.

  --fit              fit primary (scaler->logistic->isotonic) + secondary (Platt on mm)
                     on the 1,000 retired §4y rows; freeze params to refit_calibrator.json
  --eval --seed N    GPU (Dillan present): fresh answer pass on NEW seed (7 / 2024 per
                     the lock), persisting raw signals; resumable JSONL checkpoint
  --analyze          per-quintile calibration of refit vs frozen-original on FRESH rows
                     only; locked verdicts (GAP-CLOSES / GAP-PERSISTS / MIXED) + kill check
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "selfreport_confidence"))

BC = HERE.parent / "boundary_calibration"
MIN_N, N_BOOT = 30, 2000
EVAL_SEEDS = (7, 2024)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _require_lock():
    if not (HERE / "PREREGISTRATION.md").exists():
        sys.exit("REFUSING: pre-registration is not locked. Dillan must read "
                 "PREREGISTRATION_DRAFT.md, rename it PREREGISTRATION.md, and commit "
                 "(the commit = the lock) before any fitting or generation.")


def _load_train():
    xs, ys = [], []
    for seed in (42, 1337):
        d = np.load(BC / f"boundary_rows_{seed}.npz", allow_pickle=True)
        m = (d["correct"] >= 0) & np.isfinite(d["mm"]) & np.isfinite(d["p_int"])
        xs.append(np.column_stack([d["mm"][m], d["me"][m], d["br"][m]]).astype(float))
        ys.append(d["correct"][m].astype(int))
    return np.vstack(xs), np.concatenate(ys)


def fit():
    _require_lock()
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    X, y = _load_train()
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(sc.transform(X), y)
    p_lr = lr.predict_proba(sc.transform(X))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_lr, y)
    platt = LogisticRegression(C=1e6, solver="lbfgs").fit(X[:, :1], y)  # secondary: mm only
    params = {
        "trained_on": "boundary_rows_{42,1337}.npz (retired as evidence; training only)",
        "n_train": int(len(y)),
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "logreg_coef": lr.coef_[0].tolist(), "logreg_intercept": float(lr.intercept_[0]),
        "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
        "platt_coef": float(platt.coef_[0][0]), "platt_intercept": float(platt.intercept_[0]),
    }
    (HERE / "refit_calibrator.json").write_text(json.dumps(params, indent=2))
    print(f"fitted on n={len(y)}; FROZEN -> refit_calibrator.json (commit this file NOW "
          f"— the commit freezes it before any evaluation data exists)")


def _apply(params, mm, me, br):
    x = (np.column_stack([mm, me, br]) - np.array(params["scaler_mean"])) / np.array(params["scaler_scale"])
    z = x @ np.array(params["logreg_coef"]) + params["logreg_intercept"]
    p_lr = 1.0 / (1.0 + np.exp(-z))
    p_iso = np.interp(p_lr, params["iso_x"], params["iso_y"])
    p_platt = 1.0 / (1.0 + np.exp(-(params["platt_coef"] * mm + params["platt_intercept"])))
    return p_iso, p_platt


def _load_jsonl(path):
    rows = []
    if path.exists():
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    return rows


def evaluate(seed):
    _require_lock()
    if not (HERE / "refit_calibrator.json").exists():
        sys.exit("REFUSING: no frozen calibrator. Run --fit and COMMIT refit_calibrator.json first.")
    if seed not in EVAL_SEEDS:
        sys.exit(f"REFUSING: eval seeds are locked to {EVAL_SEEDS}.")
    from selfreport import select_items, _load, _gen, POS               # noqa
    from calibrate_gate import _first_of                                # noqa
    from mirrorfield.mcp.uncertainty import (                           # noqa
        compute_token_margins, compute_token_entropies, compute_boundary_ratio, calibrated_p_correct)
    items = select_items(seed)
    (HERE / "local_outputs").mkdir(exist_ok=True)
    ckpt = HERE / "local_outputs" / f"refit_eval_{seed}.jsonl"
    done = {r["i"] for r in _load_jsonl(ckpt)}
    if len(done) < len(items):
        import torch
        torch.manual_seed(seed)
        tm, tok, model = _load()
        print(f"[refit-eval] seed {seed}: resume {len(done)}/{len(items)}")
        f = ckpt.open("a", encoding="utf-8")
        for i, it in enumerate(items):
            if i in done:
                continue
            ans, tl = _gen(tm, tok, model, it["prompt"], 16, True)
            margins = compute_token_margins(tl); ent = compute_token_entropies(tl)
            fin = margins[np.isfinite(margins)]
            mm = round(float(fin.mean()), 4) if len(fin) else float("nan")
            me = round(float(ent.mean()), 4); br = round(compute_boundary_ratio(margins), 4)
            p_orig = calibrated_p_correct(mm, me, br) if np.isfinite(mm) else None
            pred = _first_of(ans.lower(), *POS[it["task"]])
            f.write(json.dumps({"i": i, "task": it["task"],
                                "correct": int(pred == it["gold"]) if pred is not None else -1,
                                "mm": mm, "me": me, "br": br,
                                "p_orig": p_orig if p_orig is not None else float("nan")}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(items)}")
        f.close()
    recs = sorted(_load_jsonl(ckpt), key=lambda r: r["i"])
    params = json.loads((HERE / "refit_calibrator.json").read_text())
    mm = np.array([r["mm"] for r in recs]); me = np.array([r["me"] for r in recs])
    br = np.array([r["br"] for r in recs])
    p_new, p_platt = _apply(params, mm, me, br)
    np.savez(HERE / f"refit_rows_{seed}.npz",
             task=np.array([r["task"] for r in recs]),
             correct=np.array([r["correct"] for r in recs]),
             mm=mm, me=me, br=br,
             p_orig=np.array([r["p_orig"] for r in recs]),
             p_refit=p_new, p_platt=p_platt)
    print(f"[refit-eval] seed {seed}: {len(recs)} rows -> refit_rows_{seed}.npz")


def analyze():
    _require_lock()
    out = {}
    for seed in EVAL_SEEDS:
        p = HERE / f"refit_rows_{seed}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        m = (d["correct"] >= 0) & np.isfinite(d["mm"]) & np.isfinite(d["p_orig"])
        y = d["correct"][m].astype(int); mg = d["mm"][m].astype(float)
        series = {"refit": d["p_refit"][m], "platt_secondary": d["p_platt"][m],
                  "original_frozen": d["p_orig"][m]}
        qs = np.quantile(mg, [0, .2, .4, .6, .8, 1.0])
        rows = []
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            bm = (mg >= lo) & (mg <= hi) if i == 4 else (mg >= lo) & (mg < hi)
            n = int(bm.sum())
            rec = {"q": i + 1, "n": n, "accuracy": round(float(y[bm].mean()), 4),
                   "acc_ci": [round(v, 4) for v in wilson(int(y[bm].sum()), n)]}
            for name, s in series.items():
                mp = float(s[bm].mean())
                rec[name] = {"mean_p": round(mp, 4), "gap": round(mp - rec["accuracy"], 4),
                             "holds": bool(rec["acc_ci"][0] <= mp <= rec["acc_ci"][1]) if n >= MIN_N else None}
            rows.append(rec)
        # discrimination of refit score (locked: AUC >= 0.60)
        s = series["refit"]; pos, neg = s[y == 1], s[y == 0]
        auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()) if len(pos) and len(neg) else None
        # aggregate calibration error (10 equal-count bins on refit score)
        order = np.argsort(s); bins = np.array_split(order, 10)
        ece = float(np.mean([abs(s[b].mean() - y[b].mean()) for b in bins if len(b)]))
        out[str(seed)] = {"n": int(m.sum()), "quintiles": rows,
                          "refit_auc": round(auc, 4) if auc else None,
                          "refit_aggregate_ce": round(ece, 4)}
    # locked verdicts (both fresh seeds required)
    if len(out) == 2:
        q1 = {s: out[s]["quintiles"][0] for s in out}
        closes = all(abs(q["refit"]["gap"]) <= 0.05 and q["refit"]["holds"] for q in q1.values()) \
            and all(out[s]["refit_aggregate_ce"] <= 0.06 for s in out) \
            and all((out[s]["refit_auc"] or 0) >= 0.60 for s in out)
        persists = all(q["refit"]["gap"] >= 0.10 and not q["refit"]["holds"] for q in q1.values())
        verdict = "GAP-CLOSES" if closes else ("GAP-PERSISTS (reportable — evidence against the mechanism hypothesis as sufficient)" if persists else "MIXED/UNDERPOWERED")
        kill = all(abs(q["refit"]["gap"]) > 0.10 for q in q1.values())
        out["verdict_candidate"] = verdict
        out["kill_criterion_triggered"] = bool(kill)
    (HERE / "refit_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3000])
    print("\nCANDIDATE verdict only — Dillan concludes. saved -> refit_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--analyze", action="store_true"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.fit:
        fit()
    elif a.eval:
        evaluate(a.seed)
    elif a.analyze:
        analyze()
    else:
        print(__doc__)
