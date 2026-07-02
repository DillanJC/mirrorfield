# -*- coding: utf-8 -*-
"""AUDIT DIAGNOSTICS for the §4y result (see CIRCULARITY_AUDIT.md). Read-only; no GPU.
Not pre-registered findings — these test rival explanations for the pre-registered result.
Reads the Opt-2 JSONL checkpoints (which retain task labels; the npz files drop them).

Checks: (1) parse-failure selection, (2) task-mix confound (within-task gaps per quintile),
(3) isotonic-floor mass in Q1, (4) score-conditional margin split (does low margin still
mark risk among matched-high p_int items?).
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def load_jsonl(p):
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return sorted(rows, key=lambda r: r["i"])


def main():
    for seed in (42, 1337):
        recs = load_jsonl(HERE / "local_outputs" / f"bregen_{seed}.jsonl")
        correct = np.array([r["correct"] for r in recs])
        mm = np.array([r["mm"] for r in recs], dtype=float)
        p_int = np.array([r["p_int"] for r in recs], dtype=float)
        task = np.array([r["task"] for r in recs])

        print(f"\n{'='*78}\nSEED {seed}  (rows={len(recs)})")

        # 1. parse failures: how many, and where on the margin axis?
        fail = correct < 0
        print(f"[parse] failures: {fail.sum()}/{len(recs)}")
        if fail.sum():
            print(f"[parse] failed-item mm median {np.nanmedian(mm[fail]):.3f} "
                  f"vs parsed {np.nanmedian(mm[~fail]):.3f}")

        m = (correct >= 0) & np.isfinite(mm) & np.isfinite(p_int)  # same mask as analyze_margin()
        y = correct[m].astype(int); mg = mm[m]; pi = p_int[m]; tk = task[m]
        qs = np.quantile(mg, [0, .2, .4, .6, .8, 1.0])

        # 2. task composition + within-task gap per quintile
        print(f"[task] overall mix: {dict(zip(*np.unique(tk, return_counts=True)))}")
        for i in range(5):
            lo, hi = qs[i], qs[i + 1]
            bm = (mg >= lo) & (mg <= hi) if i == 4 else (mg >= lo) & (mg < hi)
            counts = dict(zip(*np.unique(tk[bm], return_counts=True)))
            line = f"  Q{i+1} n={bm.sum():3d} mix={counts}"
            for t in sorted(set(tk)):
                tm_ = bm & (tk == t)
                if tm_.sum() >= 15:
                    gap = pi[tm_].mean() - y[tm_].mean()
                    line += (f" | {t}: acc={y[tm_].mean():.3f} pint={pi[tm_].mean():.3f} "
                             f"gap={gap:+.3f} (n={tm_.sum()})")
            print(line)

        # 3. within-Q1 p_int distribution vs the isotonic floor (0.625)
        q1 = mg <= qs[1]
        p1 = pi[q1]
        print(f"[floor] Q1 p_int: min={p1.min():.4f} p25={np.percentile(p1, 25):.4f} "
              f"median={np.median(p1):.4f} p75={np.percentile(p1, 75):.4f} max={p1.max():.4f}")
        print(f"[floor] Q1 frac p_int<=0.65: {(p1 <= 0.65).mean():.3f}; <=0.70: {(p1 <= 0.70).mean():.3f}; "
              f"Q1 mean={p1.mean():.4f}, acc={y[q1].mean():.4f}")

        # 4. score-conditional: among matched-high p_int, does margin still separate accuracy?
        hi_pi = pi >= 0.78
        if hi_pi.sum() > 60:
            med = np.median(mg[hi_pi])
            lo_half = hi_pi & (mg < med); hi_half = hi_pi & (mg >= med)
            print(f"[cond] among p_int>=0.78 (n={hi_pi.sum()}): low-margin half acc={y[lo_half].mean():.3f} "
                  f"(n={lo_half.sum()}) vs high-margin half acc={y[hi_half].mean():.3f} (n={hi_half.sum()})")


if __name__ == "__main__":
    main()
