"""
Blind poison-detection test — does the geometry filter detect poison that was
NOT defined geometrically?

Motivation
----------
Every prior "win" (GMR=1.0, Sati 6x, Iterative Zoom) defined poison via embedding
geometry (KMeans cluster / boundary distance) and then detected it with embedding
geometry. That is partly circular. This harness keeps the detection method fixed
and varies ONLY how poison is chosen, from geometry-defined to geometry-independent:

    R0  cluster        KMeans subcluster flip      (control; should reproduce ~1.0)
    R1  label_only     random ids, flip label,     (geometry-blind; honest hard case)
                       embedding untouched         expected ~0.5 = chance
    R2  trigger_patch  random ids, add fixed delta (backdoor-like; partial signal)
                       to embedding, flip label
    (R3 real benchmark handled separately — needs external data.)

Detection score (the filter's own logic, called as-is with CORRECT kwargs):
    risk = w_sati * compressed_flag + w_gmr * gmr_removed_flag
We report ROC-AUC of risk vs the TRUE poison mask, per regime, plus a
negative control (shuffled mask) which must collapse to ~0.5.

The TrainingDataFilter class is NOT modified. We reuse its private helpers and
the underlying geometry functions directly so the test exercises the exact same
computation. NOTE: sati_schema.py defines generate_sati_feedback_percentile
TWICE (lines ~253 and ~603); Python keeps the second, whose kwarg is
compressed_curv_pct. The filter happens to match the live one; we do too.

Author: experiment harness (read-only w.r.t. existing code)
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]  # .../mirrorfield
sys.path.insert(0, str(BASE))

from mirrorfield.geometry.phase2_weather_features import compute_topology_lite_features
from mirrorfield.geometry.sati_schema import generate_sati_feedback_percentile

SEED = 42
POISON_RATE = 0.05
K = 50


# ---------------------------------------------------------------------------
# Feature computation (mirrors TrainingDataFilter, reference == data itself)
# ---------------------------------------------------------------------------
def compute_features(X):
    """Return knn_min, knn_std, knn_mean, ridge, curvature, participation_ratio."""
    k = min(K, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    distances = distances[:, 1:]      # drop self
    indices = indices[:, 1:]

    knn_min = distances.min(axis=1)
    knn_std = distances.std(axis=1)
    knn_mean = distances.mean(axis=1)
    ridge = knn_std / (knn_mean + 1e-12)

    # local SVD curvature (effective-rank fix, same as filter)
    curvature = np.zeros(len(X))
    for i in range(len(X)):
        neigh = X[indices[i]]
        centered = neigh - neigh.mean(axis=0)
        if not np.allclose(centered, 0):
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            rank = int(np.sum(S > 1e-10))
            curvature[i] = S[rank - 1] / (S[0] + 1e-12) if rank > 1 else 1.0
        else:
            curvature[i] = 1.0

    topo, _ = compute_topology_lite_features(X, X, indices)
    participation_ratio = topo[:, 2]
    return knn_min, knn_std, knn_mean, ridge, curvature, participation_ratio


def sati_risk(X):
    """Continuous risk = Sati confidence where feedback == 'compressed', else 0.

    Calls generate_sati_feedback_percentile with the CORRECT kwarg names
    (the filter uses wrong names and would raise TypeError)."""
    knn_min, knn_std, knn_mean, ridge, curvature, pr = compute_features(X)
    feedback, confidences, _ = generate_sati_feedback_percentile(
        local_curvature=curvature,
        knn_min_distance=knn_min,
        knn_std_distance=knn_std,
        ridge_proximity=ridge,
        participation_ratio=pr,
        compressed_curv_pct=20.0,   # matches the live (2nd) definition in sati_schema
        compressed_pr_pct=20.0,
    )
    risk = np.where(feedback == "compressed", confidences, 0.0)
    return risk.astype(float), feedback


def gmr_risk(X, y):
    """GMR majority-confidence-of-wrong-class as a continuous suspicion score.

    Higher = neighbours disagree with the sample's own label => more suspect.
    Mirrors the vote logic in TrainingDataFilter._gmr_rectify."""
    n, d = X.shape
    metric = "cosine" if d > 100 else "euclidean"
    k = min(K, n - 1)
    knn = NearestNeighbors(metric=metric).fit(X)
    distances, indices = knn.kneighbors(X, n_neighbors=k + 1)
    distances, indices = distances[:, 1:], indices[:, 1:]
    inv = 1.0 / (distances + 1e-8)
    w = inv / inv.sum(axis=1, keepdims=True)

    suspicion = np.zeros(n)
    for i in range(n):
        nl = y[indices[i]]
        v0 = (w[i] * (nl == 0)).sum()
        v1 = (w[i] * (nl == 1)).sum()
        own = v1 if y[i] == 1 else v0
        suspicion[i] = 1.0 - own  # low own-class support => high suspicion
    return suspicion


# ---------------------------------------------------------------------------
# Poison regimes — vary ONLY how indices are chosen / embeddings altered
# ---------------------------------------------------------------------------
def regime_cluster(X, y, rng):
    """R0 control: flip a KMeans subcluster (geometry-DEFINED)."""
    n = len(X)
    n_p = int(n * POISON_RATE)
    km = KMeans(n_clusters=10, random_state=SEED, n_init=10).fit_predict(X)
    sizes = [(km == c).sum() for c in range(10)]
    target = int(np.argmin(sizes))
    idx = np.where(km == target)[0]
    if len(idx) >= n_p:
        idx = rng.choice(idx, n_p, replace=False)
    else:
        extra = rng.choice(np.setdiff1d(np.arange(n), idx), n_p - len(idx), replace=False)
        idx = np.concatenate([idx, extra])
    Xp = X.copy()
    yp = y.copy(); yp[idx] = 1 - yp[idx]
    mask = np.zeros(n, bool); mask[idx] = True
    return Xp, yp, mask


def regime_label_only(X, y, rng):
    """R1: RANDOM ids (not geometric), flip label, embedding untouched."""
    n = len(X)
    n_p = int(n * POISON_RATE)
    idx = rng.choice(n, n_p, replace=False)   # purely id-based
    Xp = X.copy()
    yp = y.copy(); yp[idx] = 1 - yp[idx]
    mask = np.zeros(n, bool); mask[idx] = True
    return Xp, yp, mask


def regime_trigger_patch(X, y, rng, eps=0.15):
    """R2: RANDOM ids, add a FIXED trigger vector then re-normalize, flip label.

    The trigger is a single random unit direction (a 'backdoor patch'), not a
    cluster. Re-normalization keeps all vectors unit-length so the patch is not
    trivially detectable by norm alone."""
    n, d = X.shape
    n_p = int(n * POISON_RATE)
    idx = rng.choice(n, n_p, replace=False)   # id-based selection
    trigger = rng.standard_normal(d); trigger /= np.linalg.norm(trigger)
    Xp = X.copy()
    Xp[idx] = Xp[idx] + eps * trigger
    Xp[idx] /= np.linalg.norm(Xp[idx], axis=1, keepdims=True) + 1e-12
    yp = y.copy(); yp[idx] = 1 - yp[idx]
    mask = np.zeros(n, bool); mask[idx] = True
    return Xp, yp, mask


REGIMES = {
    "R0_cluster": regime_cluster,
    "R1_label_only": regime_label_only,
    "R2_trigger_patch": regime_trigger_patch,
}


def score_regime(Xp, yp, mask, rng):
    sati, feedback = sati_risk(Xp)
    gmr = gmr_risk(Xp, yp)
    # normalize gmr to [0,1] for combination
    gmr_n = (gmr - gmr.min()) / (np.ptp(gmr) + 1e-12)
    combined = 0.5 * sati + 0.5 * gmr_n

    out = {}
    for name, score in [("sati", sati), ("gmr", gmr_n), ("combined", combined)]:
        try:
            out[f"auc_{name}"] = float(roc_auc_score(mask, score))
        except ValueError:
            out[f"auc_{name}"] = None
    # negative control: shuffle the mask, combined score must -> ~0.5
    shuffled = mask.copy(); rng.shuffle(shuffled)
    out["auc_combined_shuffled_control"] = float(roc_auc_score(shuffled, combined))

    # compressed differentiation ratio (the headline Sati metric)
    pc = (feedback[mask] == "compressed").mean()
    cc = (feedback[~mask] == "compressed").mean()
    out["compressed_ratio"] = float(pc / (cc + 1e-6))
    out["n_poison"] = int(mask.sum())
    return out


def main():
    t0 = time.time()
    X = np.load(BASE / "embeddings.npy").astype(np.float32)
    y = np.load(BASE / "experiments/track1_poison/data/original_labels.npy").astype(int)
    print(f"Loaded embeddings {X.shape}, labels {y.shape}, "
          f"pos={int((y==1).sum())} neg={int((y==0).sum())}")

    results = {"config": {"seed": SEED, "poison_rate": POISON_RATE, "k": K,
                          "n": int(len(X)), "dim": int(X.shape[1])},
               "caveat": ("auc_gmr detects label-inconsistency and so trivially "
                          "flags ANY label flip including geometry-blind poison; "
                          "it is NOT evidence for the geometry hypothesis. Use "
                          "auc_sati as the honest geometry-only detection signal."),
               "regimes": {}}

    for name, fn in REGIMES.items():
        rng = np.random.default_rng(SEED)          # same seed per regime
        Xp, yp, mask = fn(X, y, rng)
        res = score_regime(Xp, yp, mask, rng)
        results["regimes"][name] = res
        print(f"\n[{name}]  n_poison={res['n_poison']}")
        print(f"   AUC  sati={res['auc_sati']:.3f}  gmr={res['auc_gmr']:.3f}  "
              f"combined={res['auc_combined']:.3f}")
        print(f"   neg-control (shuffled)={res['auc_combined_shuffled_control']:.3f}  "
              f"compressed_ratio={res['compressed_ratio']:.2f}x")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    out_path = Path(__file__).parent / "blind_filter_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}  ({results['elapsed_sec']}s)")

    # interpretation
    # IMPORTANT: GMR scores label-inconsistency, so it trivially detects ANY
    # label flip (incl. geometry-blind) -> not evidence for the geometry
    # hypothesis. The honest geometry-only signal is the SATI auc.
    print("\n" + "=" * 60 + "\nINTERPRETATION (Sati = honest geometry signal)\n" + "=" * 60)
    s0 = results["regimes"]["R0_cluster"]["auc_sati"]
    s1 = results["regimes"]["R1_label_only"]["auc_sati"]
    s2 = results["regimes"]["R2_trigger_patch"]["auc_sati"]
    print(f"R0 cluster      (geometry-defined control): sati={s0:.3f}")
    print(f"R1 label-only   (geometry-blind)          : sati={s1:.3f}")
    print(f"R2 trigger-patch(backdoor-like)           : sati={s2:.3f}")
    print("\n(GMR auc omitted from verdict: it detects the label flip itself, "
          "not geometry.)")
    if s1 < 0.6:
        print("\n=> R1 ~chance: geometry-only detection FAILS on label-only poison,")
        print("   as predicted. Prior 'wins' relied on geometry-defined poison.")
    if s2 > 0.65:
        print("=> R2: Sati has REAL signal on trigger-patch backdoors "
              "(embedding actually moved).")
    else:
        print("=> R2: even an embedding-space trigger is weakly detected by Sati.")


if __name__ == "__main__":
    main()
