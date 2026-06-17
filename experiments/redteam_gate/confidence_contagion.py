# -*- coding: utf-8 -*-
"""
Confidence contagion — DESCRIPTIVE analysis on EXISTING redteam_gate rows.
NO new generation. CPU only. (HANDOFF.md immediate next step.)

Core question: does the model's CONFIDENCE track how the USER sounds, independent
of the truth? The red-team (§4o) and nice-team (§4p) measured accuracy + evasion
(decision outcomes). This measures the model's INTERNAL confidence signal
(calibrated p_correct, top-token margin, entropy) as a function of user framing,
holding correctness fixed — a different, novel angle on the same saved rows.

Three cuts, paired on (task,pos) against the NEUTRAL condition, per seed,
reported for BOTH seeds so replication is visible:

 A) Confidence shift independent of correctness: paired Delta in p_abs (calibrated
    p_correct), top-token margin, entropy vs neutral — overall AND on the subset
    where correctness did NOT change (wrong[neu]==wrong[cond]). A nonzero Delta on
    that subset = framing moved the model's self-confidence without moving whether
    it was right = contagion. ntok delta reported (length confound).
 B) Calibration: per-condition AUC(wrong ~ -p_abs) and AUC(wrong ~ -p_rel) with
    bootstrap CIs, paired Delta-AUC vs neutral, and the confidence gap (mean p_abs
    on correct minus on wrong). Does a framing break the gate's ability to tell
    right from wrong?
 C) Deference / default class: reconstruct predicted label (yes/no/none) from the
    saved texts; predicted-class rates per condition and paired Delta vs neutral.
    A systematic lean toward one class = deference to a default; flat class-rates
    with lower accuracy = noise. Sanity: reconstructed 'wrong' must match the
    stored array.

Run: python confidence_contagion.py [--tag tone|redteam] [--seeds 42 1337]
Higher margin / lower entropy / higher p_abs = MORE confident.
"""

import argparse, json, os
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).parent
WARMUP = 20
N_BOOT = 2000
SPLITS = {"rte": "train", "qnli": "validation"}   # must match redteam_gate.FRESH
COND_ORDER = {
    "tone": ["neutral", "effusive", "humble_support"],
    "redteam": ["neutral", "placebo", "ci1", "ci2", "ci3"],
}


def _first_of(text, a, b):                  # identical to calibrate_gate._first_of
    ia, ib = text.find(a), text.find(b)
    if ia == -1 and ib == -1:
        return None
    return a if (ib == -1 or (ia != -1 and ia < ib)) else b


def gold_maps():
    from datasets import load_dataset
    maps = {}
    for t, split in SPLITS.items():
        lab = load_dataset("glue", t, split=split)["label"]
        maps[t] = {i: ("yes" if lab[i] == 0 else "no") for i in range(len(lab))}
    return maps


def load_rows(tag, seed):
    d = np.load(HERE / f"{tag}_rows_{seed}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def _ci(triple, nd=4):
    m, lo, hi = triple
    return {"mean": round(m, nd), "ci95": [round(lo, nd), round(hi, nd)]}


def boot_mean_ci(diffs, rng, n=N_BOOT):
    diffs = np.asarray(diffs, float)
    if len(diffs) == 0:
        return (float("nan"), float("nan"), float("nan"))
    bs = [diffs[rng.randint(0, len(diffs), len(diffs))].mean() for _ in range(n)]
    return float(diffs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def key_index(task, pos, mask):
    return {(task[i], int(pos[i])): i for i in np.where(mask)[0]}


def analyze(tag, seed, golds):
    from sklearn.metrics import roc_auc_score
    d = load_rows(tag, seed)
    cond, task, pos = d["cond"], d["task"], d["pos"]
    X, p_abs, p_rel = d["X"], d["p_abs"], d["p_rel"]
    wrong = d["wrong"]
    ntok = d["ntok"] if "ntok" in d else None   # redteam npz predates ntok
    mm, me = X[:, 0], X[:, 1]                 # margin, entropy
    scored = pos >= WARMUP
    rng = np.random.RandomState(seed)
    conds = [c for c in COND_ORDER[tag] if c in set(cond.tolist())]
    neu = key_index(task, pos, scored & (cond == "neutral"))
    out = {"seed": int(seed), "tag": tag,
           "A_confidence": {}, "B_calibration": {}, "C_deference": {}}

    # ---------- A: confidence shift independent of truth ----------
    for c in conds:
        if c == "neutral":
            continue
        idx_c = key_index(task, pos, scored & (cond == c))
        keys = [k for k in neu if k in idx_c]
        ni = np.array([neu[k] for k in keys]); ci = np.array([idx_c[k] for k in keys])
        fin = np.isfinite(p_abs[ni]) & np.isfinite(p_abs[ci])
        same = wrong[ni] == wrong[ci]                 # correctness unchanged
        fs = fin & same
        rec = {
            "n_paired": len(keys),
            "d_p_abs": _ci(boot_mean_ci(p_abs[ci][fin] - p_abs[ni][fin], rng)),
            "d_margin": _ci(boot_mean_ci(mm[ci] - mm[ni], rng)),
            "d_entropy": _ci(boot_mean_ci(me[ci] - me[ni], rng)),
            "d_ntok": (_ci(boot_mean_ci((ntok[ci] - ntok[ni]).astype(float), rng))
                       if ntok is not None else None),
            "n_corr_unchanged": int(same.sum()),
            "d_p_abs_corr_unchanged":
                _ci(boot_mean_ci(p_abs[ci][fs] - p_abs[ni][fs], rng)) if fs.sum() >= 15 else None,
            "d_margin_corr_unchanged":
                _ci(boot_mean_ci(mm[ci][same] - mm[ni][same], rng)) if same.sum() >= 15 else None,
            "d_entropy_corr_unchanged":
                _ci(boot_mean_ci(me[ci][same] - me[ni][same], rng)) if same.sum() >= 15 else None,
        }
        out["A_confidence"][c] = rec

    # ---------- B: calibration ----------
    def auc(mask, score):
        m = scored & mask & np.isfinite(score)
        y = wrong[m]
        return (float(roc_auc_score(y, -score[m])) if len(np.unique(y)) == 2 else None), m

    for c in conds:
        cm = (cond == c)
        a_abs, _ = auc(cm, p_abs)
        a_rel, mrel = auc(cm, p_rel)
        fin = scored & cm & np.isfinite(p_abs)
        cor, wro = fin & (wrong == 0), fin & (wrong == 1)
        gap = (float(p_abs[cor].mean() - p_abs[wro].mean())
               if cor.sum() and wro.sum() else None)
        idx = np.where(mrel)[0]; aucs = []
        for _ in range(N_BOOT):
            bi = idx[rng.randint(0, len(idx), len(idx))]
            yy = wrong[bi]
            if len(np.unique(yy)) == 2:
                aucs.append(roc_auc_score(yy, -p_rel[bi]))
        out["B_calibration"][c] = {
            "n_scored": int((scored & cm).sum()),
            "auc_p_abs": round(a_abs, 4) if a_abs is not None else None,
            "auc_p_rel": round(a_rel, 4) if a_rel is not None else None,
            "auc_p_rel_ci95": ([round(float(np.percentile(aucs, 2.5)), 4),
                                round(float(np.percentile(aucs, 97.5)), 4)] if aucs else None),
            "confidence_gap_p_abs": round(gap, 4) if gap is not None else None,
        }

    # paired Delta-AUC (operational p_rel) vs neutral, same key set
    for c in conds:
        if c == "neutral":
            continue
        idx_c = key_index(task, pos, scored & (cond == c))
        keys = [k for k in neu if k in idx_c]
        ni = np.array([neu[k] for k in keys]); ci = np.array([idx_c[k] for k in keys])
        fn = np.isfinite(p_rel[ni]) & np.isfinite(p_rel[ci])
        ni, ci = ni[fn], ci[fn]
        if len(np.unique(wrong[ni])) == 2 and len(np.unique(wrong[ci])) == 2:
            pt = (roc_auc_score(wrong[ci], -p_rel[ci])
                  - roc_auc_score(wrong[ni], -p_rel[ni]))
            dd = []
            for _ in range(N_BOOT):
                b = rng.randint(0, len(ni), len(ni))
                yn, yc = wrong[ni[b]], wrong[ci[b]]
                if len(np.unique(yn)) == 2 and len(np.unique(yc)) == 2:
                    dd.append(roc_auc_score(yc, -p_rel[ci[b]]) - roc_auc_score(yn, -p_rel[ni[b]]))
            out["B_calibration"][c]["delta_auc_vs_neutral"] = {
                "mean": round(float(pt), 4),
                "ci95": [round(float(np.percentile(dd, 2.5)), 4),
                         round(float(np.percentile(dd, 97.5)), 4)] if dd else None}

    # ---------- C: deference / default class ----------
    txt = json.loads((HERE / "local_outputs" / f"{tag}_texts_{seed}.json").read_text())
    assert len(txt) == len(cond), (len(txt), len(cond))
    preds, recon = [], []
    for r in txt:
        p = _first_of(r["answer"], "yes", "no")
        g = golds[r["task"]][int(r["idx"])]
        preds.append(p if p is not None else "")
        recon.append(1 if (p is None or p != g) else 0)
    preds = np.array(preds, dtype=object); recon = np.array(recon)
    out["C_deference"]["recon_wrong_match_rate"] = round(float((recon == wrong).mean()), 4)
    out["C_deference"]["by_condition"] = {}
    for c in conds:
        m = scored & (cond == c)
        pr = preds[m]
        out["C_deference"]["by_condition"][c] = {
            "n": int(m.sum()),
            "pred_yes": round(float((pr == "yes").mean()), 4),
            "pred_no": round(float((pr == "no").mean()), 4),
            "pred_none": round(float((pr == "").mean()), 4)}
    # paired Delta(pred_yes), Delta(pred_none) vs neutral
    yes = (preds == "yes").astype(float); none = (preds == "").astype(float)
    for c in conds:
        if c == "neutral":
            continue
        idx_c = key_index(task, pos, scored & (cond == c))
        keys = [k for k in neu if k in idx_c]
        ni = np.array([neu[k] for k in keys]); ci = np.array([idx_c[k] for k in keys])
        out["C_deference"]["by_condition"][c]["d_pred_yes_vs_neutral"] = \
            _ci(boot_mean_ci(yes[ci] - yes[ni], rng))
        out["C_deference"]["by_condition"][c]["d_pred_none_vs_neutral"] = \
            _ci(boot_mean_ci(none[ci] - none[ni], rng))
    return out


def fmt_ci(c):
    if c is None:
        return "   n/a"
    lo, hi = c["ci95"]
    star = "*" if (lo > 0 or hi < 0) else " "
    return f"{c['mean']:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"


def digest(results, tag):
    print(f"\n{'='*78}\n CONFIDENCE CONTAGION — tag={tag}   (* = 95% CI excludes 0)\n{'='*78}")
    seeds = sorted(results)
    print("\n[A] Confidence shift vs neutral, on items whose CORRECTNESS DID NOT CHANGE")
    print("    (the clean 'independent of truth' cut). + margin/-entropy/+p_abs = more confident.")
    for c in COND_ORDER[tag]:
        if c == "neutral":
            continue
        line = any(c in results[s]["A_confidence"] for s in seeds)
        if not line:
            continue
        print(f"  {c}:")
        for s in seeds:
            a = results[s]["A_confidence"].get(c)
            if not a:
                continue
            print(f"    seed {s}: p_abs {fmt_ci(a['d_p_abs_corr_unchanged'])} | "
                  f"margin {fmt_ci(a['d_margin_corr_unchanged'])} | "
                  f"entropy {fmt_ci(a['d_entropy_corr_unchanged'])} | "
                  f"ntok(all) {fmt_ci(a['d_ntok'])} | n_same={a['n_corr_unchanged']}")
    print("\n[B] Calibration: gate AUC per condition (wrong predicted from -p_rel); "
          "delta vs neutral.")
    for s in seeds:
        b = results[s]["B_calibration"]
        print(f"  seed {s}:")
        for c in COND_ORDER[tag]:
            if c not in b:
                continue
            r = b[c]
            dl = r.get("delta_auc_vs_neutral")
            ds = "" if c == "neutral" or dl is None else f"  dAUC {fmt_ci(dl)}"
            print(f"    {c:>14}: AUC_p_rel={r['auc_p_rel']} {r['auc_p_rel_ci95']}  "
                  f"gap_p_abs={r['confidence_gap_p_abs']}{ds}")
    print("\n[C] Deference: predicted-class rates (sanity: recon vs stored 'wrong' should ~1.0)")
    for s in seeds:
        cdef = results[s]["C_deference"]
        print(f"  seed {s}: recon_match={cdef['recon_wrong_match_rate']}")
        for c in COND_ORDER[tag]:
            bc = cdef["by_condition"].get(c)
            if not bc:
                continue
            extra = ""
            if "d_pred_yes_vs_neutral" in bc:
                extra = (f"  dyes {fmt_ci(bc['d_pred_yes_vs_neutral'])}"
                         f"  dnone {fmt_ci(bc['d_pred_none_vs_neutral'])}")
            print(f"    {c:>14}: yes={bc['pred_yes']} no={bc['pred_no']} "
                  f"none={bc['pred_none']}{extra}")


def _rows_and_preds(tag, seed):
    d = load_rows(tag, seed)
    txt = json.loads((HERE / "local_outputs" / f"{tag}_texts_{seed}.json").read_text())
    pr = np.array(["" if _first_of(r["answer"], "yes", "no") is None
                   else _first_of(r["answer"], "yes", "no") for r in txt], dtype=object)
    return d, pr


def placebo_controlled(seeds):
    """THE control that decides whether tone effects are real. Pairs each tone
    condition against the redteam PLACEBO on the SAME (task,pos) items (both runs
    select identical items at a given seed; margin/entropy/p_abs/pred are
    response-level, independent of the RollingGate stream). (tone_cond - placebo)
    with CI excluding 0 = a tone-specific effect ABOVE the generic 'any-prefix'
    baseline; CI including 0 = the apparent tone effect is just 'a prefix exists'."""
    print(f"\n{'='*78}\n PLACEBO-CONTROLLED TONE EFFECT  (tone_condition - neutral placebo)\n"
          f" CI excludes 0 ⇒ real tone component above generic-prefix; includes 0 ⇒ not.\n{'='*78}")
    for seed in seeds:
        if not ((HERE / f"tone_rows_{seed}.npz").exists() and (HERE / f"redteam_rows_{seed}.npz").exists()):
            print(f"  seed {seed}: need both tone+redteam rows; skip"); continue
        tone, tpred = _rows_and_preds("tone", seed)
        rt, rpred = _rows_and_preds("redteam", seed)
        ts, rs = tone["pos"] >= WARMUP, rt["pos"] >= WARMUP
        tkey = lambda c: {(tone["task"][i], int(tone["pos"][i])): i
                          for i in np.where(ts & (tone["cond"] == c))[0]}
        rkey = lambda c: {(rt["task"][i], int(rt["pos"][i])): i
                          for i in np.where(rs & (rt["cond"] == c))[0]}
        tn, rn = tkey("neutral"), rkey("neutral")
        sh = [k for k in tn if k in rn]
        dneu = float(np.abs(tone["X"][[tn[k] for k in sh], 0]
                            - rt["X"][[rn[k] for k in sh], 0]).mean())
        print(f"  seed {seed}: neutral margin |Δ| across the two runs = {dneu:.4f} "
              f"(≈0 ⇒ runs aligned, greedy deterministic; n={len(sh)})")
        pl = rkey("placebo"); rng = np.random.RandomState(seed)
        for c in ["effusive", "humble_support"]:
            tc = tkey(c); keys = [k for k in tc if k in pl]
            ti = np.array([tc[k] for k in keys]); pi = np.array([pl[k] for k in keys])
            fin = np.isfinite(tone["p_abs"][ti]) & np.isfinite(rt["p_abs"][pi])
            d_mm = _ci(boot_mean_ci(tone["X"][ti, 0] - rt["X"][pi, 0], rng))
            d_me = _ci(boot_mean_ci(tone["X"][ti, 1] - rt["X"][pi, 1], rng))
            d_pa = _ci(boot_mean_ci(tone["p_abs"][ti][fin] - rt["p_abs"][pi][fin], rng))
            d_yes = _ci(boot_mean_ci((tpred[ti] == "yes").astype(float)
                                     - (rpred[pi] == "yes").astype(float), rng))
            print(f"    {c:>14} − placebo: margin {fmt_ci(d_mm)} | entropy {fmt_ci(d_me)} | "
                  f"p_abs {fmt_ci(d_pa)} | pred_yes {fmt_ci(d_yes)} | n={len(keys)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["tone", "redteam", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    a = ap.parse_args()
    tags = ["tone", "redteam"] if a.tag == "both" else [a.tag]
    golds = gold_maps()
    allres = {}
    for tag in tags:
        results = {}
        for s in a.seeds:
            if not (HERE / f"{tag}_rows_{s}.npz").exists():
                print(f"skip {tag} seed {s}: no rows"); continue
            results[s] = analyze(tag, s, golds)
        digest(results, tag)
        (HERE / f"contagion_results_{tag}.json").write_text(json.dumps(results, indent=2))
        allres[tag] = results
        print(f"\nsaved -> contagion_results_{tag}.json")
    if a.tag == "both":
        placebo_controlled(a.seeds)
    return allres


if __name__ == "__main__":
    main()
