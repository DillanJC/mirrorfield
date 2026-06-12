# -*- coding: utf-8 -*-
"""
Plan B Phase 0 — forensic re-analysis of the archived behavioral-flip data.

The claim under test: "geometry predicts paraphrase flips, AUC = 0.707" from
docs/BEHAVIORAL_FLIP_UPDATED_FINDINGS.md. The original analysis
(behavioral_flip_paraphrase_level_analysis.py lines 116-132) fit and scored
every model on the SAME 150 rows — an in-sample AUC, with only 30 unique
feature vectors (each query's geometry copied across its 5 paraphrases) and
11 flip events.

This script:
  1. REPRODUCES the original in-sample numbers exactly (pipeline-match proof).
  2. Re-scores honestly: leave-one-query-out CV (all 5 paraphrases of a query
     held out together), pooled out-of-fold AUC.
  3. Cluster bootstrap (2000 resamples over the 30 queries) -> 95% CI.
  4. Permutation null (1000 query-level feature shuffles through the full CV).
  5. Influence check: drop the highest-flip query ("root canal"), re-run.
  6. Verdict against Plan B's pre-registered criteria (plans/B-*.md):
     cite 0.707 ONLY if OOF geometry AUC >= 0.65 AND CI low > 0.50 AND it
     survives the influence check. Otherwise the number is RETIRED.

CPU only. No downloads, no generation. Seed 42.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from mirrorfield.geometry import GeometryBundle  # noqa

SEED = 42
N_BOOT = 2000
N_PERM = 1000
OUT = Path(__file__).parent / "phase0_results.json"

FEATURES = ["knn_mean_distance", "knn_std_distance", "knn_min_distance",
            "knn_max_distance", "local_curvature", "ridge_proximity",
            "dist_to_ref_nearest"]


def load_rows():
    samples = json.load(open(ROOT / "experiments" / "behavioral_flip_results.json",
                             encoding="utf-8"))
    rows = []
    qid = 0
    for zone, zone_samples in samples.items():
        for s in zone_samples:
            for p in s["paraphrase_preds"]:
                rows.append({
                    "qid": qid, "zone": zone,
                    "emb": np.array(s["original_pred"]["embedding"]),
                    "bdist": s["boundary_distance"],
                    "flip": int(p["label"] != s["original_pred"]["label"]),
                })
            qid += 1
    return rows


def lr():
    return LogisticRegression(max_iter=1000, random_state=SEED,
                              class_weight="balanced")


def insample_auc(X, y):
    m = lr().fit(X, y)
    return roc_auc_score(y, m.predict_proba(X)[:, 1])


def loqo_oof(X, y, qids):
    """Leave-one-query-out: each query's 5 rows held out together."""
    oof = np.zeros(len(y))
    for q in np.unique(qids):
        te = qids == q
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        m = lr().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def cluster_boot_ci(y, oof, qids, rng, n=N_BOOT):
    uq = np.unique(qids)
    vals = []
    for _ in range(n):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([np.where(qids == q)[0] for q in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], oof[idx]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4))


def perm_null(X, y, qids, rng, n=N_PERM):
    """Permute which query owns which feature vector (cluster-preserving)."""
    uq = np.unique(qids)
    # one feature row per query (rows within a query are identical)
    qrow = {q: X[qids == q][0] for q in uq}
    null_aucs = []
    for _ in range(n):
        perm = rng.permutation(uq)
        Xp = np.vstack([qrow[perm[list(uq).index(q)]] for q in qids])
        oof = loqo_oof(Xp, y, qids)
        null_aucs.append(roc_auc_score(y, oof))
    return np.array(null_aucs)


def main():
    rng = np.random.RandomState(SEED)
    rows = load_rows()
    y = np.array([r["flip"] for r in rows])
    qids = np.array([r["qid"] for r in rows])
    bdist = np.array([r["bdist"] for r in rows]).reshape(-1, 1)
    embs = np.vstack([r["emb"] for r in rows])
    print(f"rows={len(rows)}  queries={len(np.unique(qids))}  "
          f"flips={y.sum()} ({100*y.mean():.1f}%)")

    ref = np.load(ROOT / "runs" / "openai_3_large_test_20251231_024532" / "embeddings.npy")
    bundle = GeometryBundle(ref, k=50)
    geo = bundle.get_feature_matrix(bundle.compute(embs))
    Xc = np.hstack([bdist, geo])

    # ---- 1. reproduce the original in-sample numbers -----------------------
    repro = {
        "auc_boundary_insample": round(float(insample_auc(bdist, y)), 4),
        "auc_geometry_insample": round(float(insample_auc(geo, y)), 4),
        "auc_combined_insample": round(float(insample_auc(Xc, y)), 4),
    }
    stored = json.load(open(ROOT / "experiments" /
                            "behavioral_flip_paraphrase_level_analysis.json"))
    orig = stored["logistic_regression"]
    print("\n-- reproduction check (recomputed vs stored) --")
    for k, sk in [("auc_boundary_insample", "auc_boundary_only"),
                  ("auc_geometry_insample", "auc_geometry_only"),
                  ("auc_combined_insample", "auc_combined")]:
        print(f"  {k}: {repro[k]:.4f}  vs stored {orig[sk]:.4f}  "
              f"{'OK' if abs(repro[k]-orig[sk])<0.02 else 'MISMATCH'}")
    repro["matches_stored"] = all(
        abs(repro[k] - orig[sk]) < 0.02
        for k, sk in [("auc_boundary_insample", "auc_boundary_only"),
                      ("auc_geometry_insample", "auc_geometry_only"),
                      ("auc_combined_insample", "auc_combined")])

    # ---- 2-3. honest out-of-fold + cluster bootstrap -----------------------
    honest = {}
    oofs = {}
    for name, X in [("boundary", bdist), ("geometry", geo), ("combined", Xc)]:
        oof = loqo_oof(X, y, qids)
        auc = round(float(roc_auc_score(y, oof)), 4)
        lo, hi = cluster_boot_ci(y, oof, qids, rng)
        honest[name] = {"auc_oof": auc, "ci95": [lo, hi]}
        oofs[name] = oof
        print(f"\n{name:9s} OOF AUC = {auc}  CI95 [{lo}, {hi}]")

    # ---- 4. permutation null on the primary (geometry) ---------------------
    print(f"\nrunning {N_PERM} cluster-preserving permutations (geometry)...")
    nulls = perm_null(geo, y, qids, rng)
    p_perm = float((nulls >= honest["geometry"]["auc_oof"]).mean())
    honest["geometry"]["perm_null_mean"] = round(float(nulls.mean()), 4)
    honest["geometry"]["perm_null_p95"] = round(float(np.percentile(nulls, 95)), 4)
    honest["geometry"]["perm_p_value"] = round(p_perm, 4)
    print(f"  null mean={nulls.mean():.4f}  null 95th pct="
          f"{np.percentile(nulls,95):.4f}  perm-p={p_perm:.4f}")

    # ---- 5. influence check: drop the highest-flip query -------------------
    flips_per_q = {q: int(y[qids == q].sum()) for q in np.unique(qids)}
    worst_q = max(flips_per_q, key=flips_per_q.get)
    keep = qids != worst_q
    oof_inf = loqo_oof(geo[keep], y[keep], qids[keep])
    auc_inf = round(float(roc_auc_score(y[keep], oof_inf)), 4)
    lo_i, hi_i = cluster_boot_ci(y[keep], oof_inf, qids[keep], rng)
    influence = {"dropped_query": int(worst_q),
                 "its_flips": flips_per_q[worst_q],
                 "auc_oof_without": auc_inf, "ci95_without": [lo_i, hi_i]}
    print(f"\ninfluence check: dropped query {worst_q} "
          f"({flips_per_q[worst_q]} of {int(y.sum())} flips) -> "
          f"geometry OOF AUC {auc_inf} CI [{lo_i}, {hi_i}]")

    # ---- 6. pre-registered verdict -----------------------------------------
    g = honest["geometry"]
    citable = (g["auc_oof"] >= 0.65 and g["ci95"][0] > 0.50
               and auc_inf >= 0.65 and lo_i > 0.50)
    if citable:
        verdict = "VERIFIED: the 0.707-class claim survives honest evaluation"
    elif g["ci95"][0] > 0.50:
        verdict = ("UNVERIFIABLE-AT-THIS-N: above chance but below the 0.65 bar "
                   "or fails influence check; 0.707 may NOT be cited")
    else:
        verdict = ("RETIRED: the 0.707 was an in-sample artifact; out-of-fold "
                   "geometry AUC is indistinguishable from chance")

    res = {"reproduction": repro, "honest_oof": honest,
           "influence_check": influence, "n_flips": int(y.sum()),
           "verdict": verdict,
           "criteria": "cite only if OOF>=0.65 AND CI_low>0.50 AND survives "
                       "influence check (plans/B-flip-auc-verification.md)"}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n{'='*70}\nVERDICT: {verdict}\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
