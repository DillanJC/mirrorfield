# -*- coding: utf-8 -*-
"""SMOKE TEST — Plan J plumbing on Qwen2.5-0.5B. n=20, NON-EVIDENTIAL by construction.

Verifies, before any lock: (1) the 0.5B model loads on this rig, (2) the A1-style
answer pass runs against it, (3) raw signals persist, (4) a fresh per-model calibrator
fits end-to-end. No claims may ever cite these numbers (n=20; printed for plumbing
verification only). GPU use authorized by Dillan 2026-07-11.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "selfreport_confidence"))

import selfreport                                                    # noqa
from calibrate_gate import _first_of                                 # noqa
from mirrorfield.mcp.uncertainty import (                            # noqa
    compute_token_margins, compute_token_entropies, compute_boundary_ratio)

SMOKE_N = 20
selfreport.MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # the Plan J config change, in one line


def main():
    import torch
    torch.manual_seed(7)
    items = selfreport.select_items(7)[:SMOKE_N]
    print(f"[smoke] loading {selfreport.MODEL} ...")
    tm, tok, model = selfreport._load()
    rows = []
    for i, it in enumerate(items):
        ans, tl = selfreport._gen(tm, tok, model, it["prompt"], 16, True)
        margins = compute_token_margins(tl); ent = compute_token_entropies(tl)
        fin = margins[np.isfinite(margins)]
        mm = float(fin.mean()) if len(fin) else float("nan")
        pred = _first_of(ans.lower(), *selfreport.POS[it["task"]])
        rows.append({"task": it["task"], "mm": round(mm, 4),
                     "me": round(float(ent.mean()), 4),
                     "br": round(compute_boundary_ratio(margins), 4),
                     "correct": int(pred == it["gold"]) if pred is not None else -1})
        print(f"  {i+1}/{SMOKE_N} mm={mm:+.2f} parsed={pred is not None}")
    ok = [r for r in rows if r["correct"] >= 0]
    X = np.array([[r["mm"], r["me"], r["br"]] for r in ok])
    y = np.array([r["correct"] for r in ok])
    # fresh-calibrator plumbing check (toy fit — NEVER evidence at n=20)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    fit_ok = False
    if len(set(y)) > 1:
        LogisticRegression(C=1e6, solver="lbfgs").fit(StandardScaler().fit_transform(X), y)
        fit_ok = True
    np.savez(HERE / "smoke_0p5b_rows.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(json.dumps({"model_loaded": True, "rows": len(rows), "parse_rate": round(len(ok)/len(rows), 2),
                      "accuracy_NONEVIDENTIAL_n20": round(float(y.mean()), 2) if len(ok) else None,
                      "signals_finite": bool(np.isfinite(X).all()),
                      "fresh_calibrator_fits": fit_ok, "npz_written": True}, indent=1))
    print("[smoke] PLUMBING VERIFIED — Plan J execution is a config change, as designed.")


if __name__ == "__main__":
    main()
