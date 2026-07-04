# -*- coding: utf-8 -*-
"""Verbal confidence near the boundary + alternative stratifiers.
(AMENDMENT_2_VERBAL_AND_STRATIFIERS.md, locked ddaed33 — run exactly as specified.)"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
A1DIR = HERE.parent / "selfreport_confidence"
MIN_N = 30


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def quintile_table(strat, torn_high, y, pi, verbal):
    """Per-quintile stats along `strat`; torn_high=True => torn quintile is Q5 axis-wise
    (we RE-INDEX so that printed Q1 is always the torn end, per the locked convention)."""
    qs = np.quantile(strat, [0, .2, .4, .6, .8, 1.0])
    rows = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        bm = (strat >= lo) & (strat <= hi) if i == 4 else (strat >= lo) & (strat < hi)
        n = int(bm.sum())
        if n == 0:  # quantile edges can collapse under heavy ties (e.g. br)
            rows.append({"axis_q": i + 1, "n": 0, "empty_by_ties": True})
            continue
        vmask = bm & np.isfinite(verbal)
        acc = float(y[bm].mean())
        wlo, whi = wilson(int(y[bm].sum()), n)
        rows.append({
            "axis_q": i + 1, "n": n,
            "accuracy": round(acc, 4), "acc_ci": [round(wlo, 4), round(whi, 4)],
            "internal_mean": round(float(pi[bm].mean()), 4),
            "internal_gap": round(float(pi[bm].mean()) - acc, 4),
            "n_verbal": int(vmask.sum()),
            "verbal_missing": int(n - vmask.sum()),
            "verbal_mean": round(float(verbal[vmask].mean()), 4) if vmask.sum() else None,
            "verbal_gap": (round(float(verbal[vmask].mean()) - float(y[vmask].mean()), 4)
                           if vmask.sum() >= MIN_N else None),
            "verbal_acc": round(float(y[vmask].mean()), 4) if vmask.sum() else None,
        })
    if torn_high:
        rows = rows[::-1]
    for j, r in enumerate(rows):
        r["q_torn_first"] = j + 1
    return rows


def main():
    out = {}
    for seed in (42, 1337):
        b = np.load(HERE / f"boundary_rows_{seed}.npz", allow_pickle=True)
        a = np.load(A1DIR / f"selfreport_rows_{seed}.npz", allow_pickle=True)
        k = min(len(a["p_int"]), len(b["p_int"]))
        align = float(np.nanmax(np.abs(a["p_int"][:k] - b["p_int"][:k])))
        assert align == 0.0, f"alignment failed: max|dp_int|={align}"
        correct, mm, me, br, pi = (b["correct"][:k], b["mm"][:k], b["me"][:k],
                                   b["br"][:k], b["p_int"][:k])
        verbal = a["verbal"][:k].astype(float)
        m = (correct >= 0) & np.isfinite(mm) & np.isfinite(pi)
        y = correct[m].astype(int)
        vfin = verbal[m][np.isfinite(verbal[m])]
        res = {"n": int(m.sum()), "alignment_max_abs_dpint": align,
               "verbal_value_counts": {str(v): int(c) for v, c in
                                       zip(*np.unique(np.round(vfin, 2), return_counts=True))},
               "strata": {}}
        for name, arr, torn_high in (("mm", mm[m].astype(float), False),
                                     ("me", me[m].astype(float), True),
                                     ("br", br[m].astype(float), True)):
            res["strata"][name] = quintile_table(arr, torn_high, y, pi[m].astype(float), verbal[m])
        out[str(seed)] = res

    # locked verdicts
    va, vb = {}, {}
    q1 = {s: out[s]["strata"]["mm"][0] for s in out}   # torn-first Q1 on the mm axis
    diffs = {s: (q1[s]["verbal_gap"] - q1[s]["internal_gap"]) if q1[s]["verbal_gap"] is not None else None
             for s in out}
    excl = {s: (q1[s]["verbal_mean"] is not None and q1[s]["n_verbal"] >= MIN_N and not
                (q1[s]["acc_ci"][0] <= q1[s]["verbal_mean"] <= q1[s]["acc_ci"][1])) for s in out}
    if all(d is not None and d >= 0.05 for d in diffs.values()) and all(excl.values()):
        va = "VERBAL-WORSE-IN-TORN"
    elif all(d is not None and abs(d) < 0.05 for d in diffs.values()):
        va = "VERBAL-SIMILAR"
    else:
        va = "MIXED"
    for name in ("me", "br"):
        tq = {s: out[s]["strata"][name][0] for s in out}
        if any(t.get("empty_by_ties") or t["n"] < MIN_N for t in tq.values()):
            vb[name] = "MIXED (torn bin empty/underpowered)"
            continue
        same = all(t["internal_gap"] >= 0.10 and not
                   (t["acc_ci"][0] <= t["internal_mean"] <= t["acc_ci"][1]) for t in tq.values())
        none_ = all(t["internal_gap"] <= 0.05 or
                    (t["acc_ci"][0] <= t["internal_mean"] <= t["acc_ci"][1]) for t in tq.values())
        vb[name] = "SAME-PATTERN" if same else ("NO-PATTERN" if none_ else "MIXED")
    out["verdicts_candidate"] = {"A_verbal_vs_internal_torn": va, "A_gap_diffs_torn": diffs,
                                 "B_stratifiers": vb}
    (HERE / "verbal_boundary_results.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'='*96}\n Amendment 2 — verbal confidence near the boundary + me/br stratifiers (Qwen-3B RTE+QNLI)\n{'='*96}")
    for s in ("42", "1337"):
        r = out[s]
        print(f"\n seed {s} (n={r['n']}; p_int alignment 0.0)")
        for name in ("mm", "me", "br"):
            print(f"  stratifier {name} (Q1 = torn end):")
            print(f"   {'Q':>2} {'n':>4} {'accuracy [Wilson95]':>25} {'p_int':>7} {'int_gap':>8} {'verbal':>7} {'vrb_gap':>8} {'vrb_missing':>11}")
            for b_ in r["strata"][name]:
                if b_.get("empty_by_ties"):
                    print(f"   {b_['q_torn_first']:>2}    0   (empty — quantile ties)"); continue
                print(f"   {b_['q_torn_first']:>2} {b_['n']:>4} {b_['accuracy']:>8} [{b_['acc_ci'][0]:.3f},{b_['acc_ci'][1]:.3f}] "
                      f"{b_['internal_mean']:>7} {b_['internal_gap']:>+8} "
                      f"{str(b_['verbal_mean']):>7} {str(b_['verbal_gap']):>8} {b_['verbal_missing']:>11}")
    print(f"\n VERDICT CANDIDATES (locked rules, Dillan concludes): {json.dumps(out['verdicts_candidate'], indent=1)}")
    print("saved -> verbal_boundary_results.json")


if __name__ == "__main__":
    main()
