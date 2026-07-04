# -*- coding: utf-8 -*-
"""
Boundary-stratified calibration of the log-prob confidence signal. (PREREGISTRATION.md, cc62bd8)

Option 1 (--opt1, no GPU): stratify A1's saved calibrated p_int; per-bin GAP + within-band SLOPE.
Option 2 (--regen then --opt2): re-run the A1 answer pass persisting the RAW mean-margin (mm),
stratify by mm (quintiles). Both: Wilson CIs, both seeds + replication, min-n 30, no smoothing.
All claims are about Qwen2.5-3B on RTE+QNLI ("on this model").
"""

import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "selfreport_confidence"))

TAU_A, TAU_P = 0.7763, 0.8684
EDGES = [0.625, TAU_A, 0.80, 0.82, 0.845, TAU_P, 0.95, 1.001]
LABELS = ["abstain-tail", "verify-1", "verify-2", "verify-3", "verify-4", "present", "high-tail"]
MIN_N, N_BOOT, MEANINGFUL_SLOPE = 30, 2000, 0.5
A1DIR = HERE.parent / "selfreport_confidence"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


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


# ----------------------------- Option 1: p_int axis (no GPU) -----------------------------
def analyze_pint(seeds=(42, 1337)):
    out = {}
    for seed in seeds:
        d = np.load(A1DIR / f"selfreport_rows_{seed}.npz", allow_pickle=True)
        correct, p_int = d["correct"], d["p_int"]
        m = (correct >= 0) & np.isfinite(p_int)
        y = correct[m].astype(int); pi = p_int[m]
        rng = np.random.RandomState(seed)
        bins = []
        for i in range(len(EDGES) - 1):
            lo, hi = EDGES[i], EDGES[i + 1]
            bm = (pi >= lo) & (pi < hi)
            n = int(bm.sum())
            rec = {"bin": LABELS[i], "range": [round(lo, 4), round(hi, 4)], "n": n}
            if n:
                acc = float(y[bm].mean()); mp = float(pi[bm].mean())
                wlo, whi = wilson(int(y[bm].sum()), n)
                rec.update(mean_pint=round(mp, 4), accuracy=round(acc, 4),
                           acc_ci=[round(wlo, 4), round(whi, 4)], gap=round(mp - acc, 4),
                           interpretable=(n >= MIN_N))
                rec["gap_verdict"] = (None if n < MIN_N else
                                      "HOLDS" if wlo <= mp <= whi else
                                      "OVERCONF" if mp > acc else "UNDERCONF")
            bins.append(rec)
        # within-band slope of correct ~ p_int  (units: Δaccuracy per unit p_int; ideal = 1.0)
        band = (pi >= TAU_A) & (pi < TAU_P)
        xb, yb = pi[band], y[band]
        s = float(np.polyfit(xb, yb, 1)[0]) if len(xb) > 2 and xb.std() > 1e-9 else None
        sl = [np.polyfit(xb[i], yb[i], 1)[0]
              for i in (rng.randint(0, len(yb), len(yb)) for _ in range(N_BOOT))
              if xb[i].std() > 1e-9] if s is not None else []
        s_lo, s_hi = (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))) if sl else (None, None)
        if s_lo is None:
            sv = "NO DATA"
        elif s_lo > 0:
            sv = "RISING"
        elif s_hi < MEANINGFUL_SLOPE:
            sv = "FLAT (no within-band slope)"
        else:
            sv = "UNDERPOWERED-BY-COMPRESSION"
        out[str(seed)] = {"n": int(m.sum()), "aggregate_accuracy": round(float(y.mean()), 4),
                          "aggregate_mean_pint": round(float(pi.mean()), 4), "bins": bins,
                          "band_n": int(band.sum()),
                          "within_band_slope": round(s, 4) if s is not None else None,
                          "slope_ci95": [round(s_lo, 4), round(s_hi, 4)] if s_lo is not None else None,
                          "slope_verdict": sv}
    (HERE / "boundary_pint_results.json").write_text(json.dumps(out, indent=2))
    _digest_opt1(out)
    return out


def _digest_opt1(out):
    print(f"\n{'='*82}\n OPTION 1 — calibrated p_int axis (Qwen-3B, RTE+QNLI; no generation)\n{'='*82}")
    for s, r in out.items():
        print(f"\n seed {s}  (n={r['n']}, aggregate acc={r['aggregate_accuracy']}, mean p_int={r['aggregate_mean_pint']})")
        print(f"   {'bin':>12} {'p_int range':>15} {'n':>4} {'mean_pint':>9} {'accuracy [Wilson95]':>26} {'gap':>7}  verdict")
        for b in r["bins"]:
            if b["n"] == 0:
                print(f"   {b['bin']:>12} {str(b['range']):>15} {0:>4}   (empty)"); continue
            tag = b.get("gap_verdict") or "underpowered(n<30)"
            print(f"   {b['bin']:>12} {str(b['range']):>15} {b['n']:>4} {b['mean_pint']:>9} "
                  f"{b['accuracy']:>8} [{b['acc_ci'][0]:.3f},{b['acc_ci'][1]:.3f}] {b['gap']:>+7} {tag}")
        print(f"   WITHIN-BAND SLOPE (correct~p_int over the verify band, n={r['band_n']}): "
              f"{r['within_band_slope']} CI{r['slope_ci95']}  ->  {r['slope_verdict']}")
    seeds = list(out)
    if len(seeds) == 2:
        v0, v1 = out[seeds[0]]["slope_verdict"], out[seeds[1]]["slope_verdict"]
        print(f"\n REPLICATION: slope verdict seed{seeds[0]}={v0!r} / seed{seeds[1]}={v1!r} -> "
              f"{'REPLICATED' if v0 == v1 else 'INCONCLUSIVE across seeds'}")
    print("\nsaved -> boundary_pint_results.json")


# ----------------------------- Option 2: raw mean-margin re-gen + axis -----------------------------
def regen(seed):
    from selfreport import select_items, _load, _gen, POS            # noqa  (identical A1 answer pass)
    from calibrate_gate import _first_of                              # noqa
    from mirrorfield.mcp.uncertainty import (                         # noqa
        compute_token_margins, compute_token_entropies, compute_boundary_ratio, calibrated_p_correct)
    items = select_items(seed)
    (HERE / "local_outputs").mkdir(exist_ok=True)
    ckpt = HERE / "local_outputs" / f"bregen_{seed}.jsonl"
    done = {r["i"] for r in _load_jsonl(ckpt)}
    if len(done) < len(items):
        import torch
        torch.manual_seed(seed)
        tm, tok, model = _load()
        print(f"[Opt2] seed {seed}: re-gen answer pass, resume {len(done)}/{len(items)}")
        f = ckpt.open("a", encoding="utf-8")
        for i, it in enumerate(items):
            if i in done:
                continue
            ans, tl = _gen(tm, tok, model, it["prompt"], 16, True)
            margins = compute_token_margins(tl); ent = compute_token_entropies(tl)
            fin = margins[np.isfinite(margins)]
            mm = round(float(fin.mean()), 4) if len(fin) else float("nan")
            me = round(float(ent.mean()), 4); br = round(compute_boundary_ratio(margins), 4)
            pint = calibrated_p_correct(mm, me, br) if np.isfinite(mm) else None
            pred = _first_of(ans.lower(), *POS[it["task"]])
            f.write(json.dumps({"i": i, "task": it["task"],
                                "correct": int(pred == it["gold"]) if pred is not None else -1,
                                "mm": mm, "me": me, "br": br,
                                "p_int": pint if pint is not None else float("nan")}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  regen {i+1}/{len(items)}")
        f.close()
    recs = sorted(_load_jsonl(ckpt), key=lambda r: r["i"])
    np.savez(HERE / f"boundary_rows_{seed}.npz",
             task=np.array([r["task"] for r in recs]),  # audit note: npz previously dropped task labels
             correct=np.array([r["correct"] for r in recs]), mm=np.array([r["mm"] for r in recs]),
             me=np.array([r["me"] for r in recs]), br=np.array([r["br"] for r in recs]),
             p_int=np.array([r["p_int"] for r in recs]))
    print(f"[Opt2] seed {seed}: {len(recs)} rows -> boundary_rows_{seed}.npz")


def analyze_margin(seeds=(42, 1337)):
    out = {}
    for seed in seeds:
        p = HERE / f"boundary_rows_{seed}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        correct, mm, p_int = d["correct"], d["mm"], d["p_int"]
        # validity: does the re-gen reproduce A1's saved p_int (=> identical generation)?
        repro = None
        a1p = A1DIR / f"selfreport_rows_{seed}.npz"
        if a1p.exists():
            a1 = np.load(a1p, allow_pickle=True)
            k = min(len(a1["p_int"]), len(p_int))
            repro = round(float(np.nanmax(np.abs(a1["p_int"][:k] - p_int[:k]))), 6)
        m = (correct >= 0) & np.isfinite(mm) & np.isfinite(p_int)
        y = correct[m].astype(int); margin = mm[m]; pi = p_int[m]
        rng = np.random.RandomState(seed)
        qs = np.quantile(margin, [0, .2, .4, .6, .8, 1.0])
        bins = []
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            bm = (margin >= lo) & (margin <= hi) if i == 4 else (margin >= lo) & (margin < hi)
            n = int(bm.sum())
            if not n:
                bins.append({"q": i + 1, "n": 0}); continue
            acc = float(y[bm].mean()); pmean = float(pi[bm].mean())
            wlo, whi = wilson(int(y[bm].sum()), n)
            bins.append({"q": i + 1, "margin_range": [round(lo, 3), round(hi, 3)], "n": n,
                         "mean_margin": round(float(margin[bm].mean()), 3), "mean_pint": round(pmean, 4),
                         "accuracy": round(acc, 4), "acc_ci": [round(wlo, 4), round(whi, 4)],
                         "gap_pint_minus_acc": round(pmean - acc, 4),
                         "calib_holds": bool(wlo <= pmean <= whi) if n >= MIN_N else None})
        # discrimination: accuracy(Q5) - accuracy(Q1)
        q1 = margin <= qs[1]; q5 = margin >= qs[4]
        if q1.sum() >= MIN_N and q5.sum() >= MIN_N:
            diffs = [y[q5][rng.randint(0, q5.sum(), q5.sum())].mean()
                     - y[q1][rng.randint(0, q1.sum(), q1.sum())].mean() for _ in range(N_BOOT)]
            disc = {"acc_Q5_minus_Q1": round(float(y[q5].mean() - y[q1].mean()), 4),
                    "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}
            disc["verdict"] = ("RISING (margin discriminates correctness)" if disc["ci95"][0] > 0 else
                               "FLAT/UNDERPOWERED (CI includes 0)")
        else:
            disc = {"verdict": "underpowered"}
        out[str(seed)] = {"reproduces_A1_pint_maxabsdiff": repro, "n": int(m.sum()),
                          "margin_range": [round(float(margin.min()), 3), round(float(margin.max()), 3)],
                          "margin_std": round(float(margin.std()), 3), "quintiles": bins,
                          "discrimination_Q5_Q1": disc}
    (HERE / "boundary_margin_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'='*82}\n OPTION 2 — raw mean-margin axis (Qwen-3B, RTE+QNLI; re-gen)\n{'='*82}")
    for s, r in out.items():
        print(f"\n seed {s}  (n={r['n']}; margin range {r['margin_range']} std {r['margin_std']}; "
              f"reproduces A1 p_int to {r['reproduces_A1_pint_maxabsdiff']})")
        print(f"   {'Qtile':>5} {'margin range':>16} {'n':>4} {'mean_margin':>11} {'mean_pint':>9} {'accuracy [Wilson95]':>26} {'gap':>7} calib")
        for b in r["quintiles"]:
            if b["n"] == 0:
                continue
            print(f"   {b['q']:>5} {str(b['margin_range']):>16} {b['n']:>4} {b['mean_margin']:>11} {b['mean_pint']:>9} "
                  f"{b['accuracy']:>8} [{b['acc_ci'][0]:.3f},{b['acc_ci'][1]:.3f}] {b['gap_pint_minus_acc']:>+7} {b['calib_holds']}")
        dd = r["discrimination_Q5_Q1"]
        print(f"   DISCRIMINATION acc(Q5)-acc(Q1) = {dd.get('acc_Q5_minus_Q1')} CI{dd.get('ci95')} -> {dd['verdict']}")
    seeds = list(out)
    if len(seeds) == 2:
        print(f"\n REPLICATION: discrimination verdict {out[seeds[0]]['discrimination_Q5_Q1']['verdict']!r} / "
              f"{out[seeds[1]]['discrimination_Q5_Q1']['verdict']!r}")
    print("\nsaved -> boundary_margin_results.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt1", action="store_true"); ap.add_argument("--regen", action="store_true")
    ap.add_argument("--opt2", action="store_true"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.opt1:
        analyze_pint()
    elif a.regen:
        regen(a.seed)
    elif a.opt2:
        analyze_margin()
    else:
        print("use --opt1 | --regen --seed N | --opt2")
