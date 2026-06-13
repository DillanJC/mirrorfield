# -*- coding: utf-8 -*-
"""
Plan E follow-up — WHY terrorism >> hate speech? Harm-resistance, or vocabulary?

The §4l per-category pattern (terrorism AUC ~0.78, hate speech ~chance) has two
explanations:
  (romantic) the model hesitates token-by-token on some harm types;
  (mundane)  the gate detects LEXICALLY UNUSUAL text, and terrorism content is
             full of rare/technical words while hate speech is ordinary words.

This script discriminates them, CPU-only, no harmful text committed (we reload
BeaverTails responses by ds_index purely to compute aggregate lexical stats):

  1. Per category (pooled primary 30k_test + replication 330k_test): gate AUC
     vs safe, mean response length, mean word-rarity (corpus-internal
     -log frequency), type-token ratio.
  2. Across categories: Spearman( per-category gate-AUC , mean rarity ) and
     ( gate-AUC , length ). High positive => the "harm signal" tracks vocabulary.
  3. Confound-controlled H1: OOF AUC of gate-only vs lexical-only (len, word
     count, ttr, rarity, long-word frac) vs gate+lexical; bootstrap CI on
     dAUC(gate over lexical). If the gate retains lift over lexical features,
     the harm signal is more than vocabulary; if not, §4l must be reframed.
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE = Path(__file__).parent
SEED = 42
N_BOOT = 2000
WORD = re.compile(r"[a-z']+")


def load_pooled():
    from datasets import load_dataset
    splits = {"30k_test": load_dataset("PKU-Alignment/BeaverTails",
                                       split="30k_test"),
              "330k_test": load_dataset("PKU-Alignment/BeaverTails",
                                        split="330k_test")}
    rows = []
    for f, split in [("harm_gate_trackA_rows.npz", "30k_test"),
                     ("harm_gate_trackA_repl_rows.npz", "330k_test")]:
        d = np.load(HERE / f, allow_pickle=True)
        ds = splits[split]
        for k in range(len(d["y_unsafe"])):
            resp = ds[int(d["ds_index"][k])]["response"]
            rows.append({"X": d["X"][k], "y": int(d["y_unsafe"][k]),
                         "cat": str(d["category"][k]),
                         "resp_len": int(d["resp_len"][k]), "text": resp})
    return rows


def lexical_features(rows):
    # corpus-internal frequency: rare = rare across THESE responses
    counts = Counter()
    toks_per = []
    for r in rows:
        toks = WORD.findall(r["text"].lower())
        toks_per.append(toks)
        counts.update(toks)
    total = sum(counts.values())
    neglogf = {w: -np.log(c / total) for w, c in counts.items()}
    feats = []
    for r, toks in zip(rows, toks_per):
        n = len(toks)
        if n == 0:
            feats.append([0, 0, 0, 0, 0])
            continue
        rarity = float(np.mean([neglogf[w] for w in toks]))
        ttr = len(set(toks)) / n
        long_frac = np.mean([len(w) >= 9 for w in toks])
        feats.append([len(r["text"]), n, ttr, rarity, long_frac])
    return np.array(feats, dtype=np.float64)


def oof(X, y, seed=SEED):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def boot_delta(y, pa, pb, rng):
    v = []
    n = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, n, n)
        if len(np.unique(y[bi])) < 2:
            continue
        v.append(roc_auc_score(y[bi], pa[bi]) - roc_auc_score(y[bi], pb[bi]))
    return [round(float(np.percentile(v, 2.5)), 4),
            round(float(np.percentile(v, 97.5)), 4)]


def main():
    rng = np.random.RandomState(SEED)
    rows = load_pooled()
    X = np.array([r["X"] for r in rows])              # margin, entropy, boundary
    y = np.array([r["y"] for r in rows])
    cat = np.array([r["cat"] for r in rows])
    L = lexical_features(rows)                         # 5 lexical features
    rarity = L[:, 3]
    print(f"pooled n={len(y)}  unsafe={y.sum()}")

    # global OOF gate probs (same construction as analyze_track_a per-category)
    p_gate = oof(X, y)
    # length-only OOF probs — to test if length ALONE reproduces the ranking
    p_lenonly = oof(L[:, [0]], y)

    # ---- (1) per-category table ---------------------------------------------
    percat = {}
    for c in sorted(set(cat[y == 1])):
        m = (y == 0) | (cat == c)
        if (y[m] == 1).sum() < 20:
            continue
        percat[c] = {
            "n_unsafe": int((cat == c).sum()),
            "gate_auc": round(float(roc_auc_score(y[m], p_gate[m])), 3),
            "len_only_auc": round(float(roc_auc_score(y[m], p_lenonly[m])), 3),
            "mean_rarity": round(float(rarity[cat == c].mean()), 3),
            "mean_len": round(float(L[cat == c, 0].mean()), 1),
            "mean_entropy_feat": round(float(X[cat == c, 1].mean()), 3),
        }
    safe_rarity = round(float(rarity[y == 0].mean()), 3)
    safe_len = round(float(L[y == 0, 0].mean()), 1)

    # ---- (2) does AUC track rarity / length across categories? --------------
    cats = list(percat)
    aucs = np.array([percat[c]["gate_auc"] for c in cats])
    lenaucs = np.array([percat[c]["len_only_auc"] for c in cats])
    rars = np.array([percat[c]["mean_rarity"] for c in cats])
    lens = np.array([percat[c]["mean_len"] for c in cats])
    rho_rar, p_rar = spearmanr(aucs, rars)
    rho_len, p_len = spearmanr(aucs, lens)
    # does length-only reproduce the gate's per-category RANKING?
    rho_rank, p_rank = spearmanr(aucs, lenaucs)

    # ---- (3) confound-controlled H1 -----------------------------------------
    p_lex = oof(L, y)
    p_both = oof(np.hstack([X, L]), y)
    auc_gate = round(float(roc_auc_score(y, p_gate)), 4)
    auc_lex = round(float(roc_auc_score(y, p_lex)), 4)
    auc_both = round(float(roc_auc_score(y, p_both)), 4)
    d_gate_over_lex = boot_delta(y, p_both, p_lex, rng)   # does gate ADD to lexical?

    res = {
        "n": int(len(y)), "n_unsafe": int(y.sum()),
        "safe_mean_rarity": safe_rarity,
        "per_category": percat,
        "across_categories": {
            "spearman_auc_vs_rarity": [round(float(rho_rar), 3), round(float(p_rar), 4)],
            "spearman_auc_vs_length": [round(float(rho_len), 3), round(float(p_len), 4)],
            "spearman_gateauc_vs_lenonlyauc": [round(float(rho_rank), 3), round(float(p_rank), 4)],
            "safe_mean_len": safe_len,
        },
        "confound_controlled_H1": {
            "auc_gate_only": auc_gate,
            "auc_lexical_only": auc_lex,
            "auc_gate_plus_lexical": auc_both,
            "dAUC_gate_over_lexical_ci95": d_gate_over_lex,
        },
    }

    # verdict — weigh LENGTH (the dominant correlate), not just rarity
    length_explains_ranking = (rho_len >= 0.6 and p_len < 0.05) or \
                              (rho_rank >= 0.6 and p_rank < 0.05)
    gate_adds_global = d_gate_over_lex[0] > 0
    if length_explains_ranking and gate_adds_global:
        verdict = ("LENGTH ARTIFACT (per-category) + weak harm signal (global). "
                   "The terrorism>>hate-speech RANKING is dominantly a "
                   "RESPONSE-LENGTH effect (AUC~length rho={:.2f}; length-only "
                   "reproduces the ranking rho={:.2f}): long detailed responses "
                   "give the gate more tokens to find hesitation; short curt "
                   "ones (hate speech) do not. NOT a map of which harms the "
                   "model resists. Separately, the gate retains a small "
                   "harm-specific lift over lexical features globally "
                   "(dAUC {} clears 0) - that is the real but weak §4l signal. "
                   "Reframe the 'distinction' as length, keep the modest global "
                   "finding.".format(rho_len, rho_rank, d_gate_over_lex))
    elif length_explains_ranking:
        verdict = ("LENGTH ARTIFACT: the per-category ranking is a length "
                   "effect and the gate adds nothing beyond lexical features. "
                   "The harm signal is largely lexical. Reframe §4l hard.")
    elif gate_adds_global:
        verdict = ("HARM-SPECIFIC: ranking not explained by length or rarity, "
                   "and the gate adds beyond lexical features. Romantic reading "
                   "holds - but check N.")
    else:
        verdict = "UNCLEAR: small-N category noise the likely driver."
    res["verdict"] = verdict

    (HERE / "category_mechanism_results.json").write_text(json.dumps(res, indent=2))
    print("\nper-category (gate_auc | len_only_auc | mean_len | mean_rarity):")
    for c in sorted(percat, key=lambda x: -percat[x]["gate_auc"]):
        v = percat[c]
        print(f"  {c[:34]:34s} gate={v['gate_auc']:.3f}  len_only={v['len_only_auc']:.3f}  "
              f"len={v['mean_len']:.0f}  rarity={v['mean_rarity']:.2f}  n={v['n_unsafe']}")
    print(f"  (safe class: mean_len={safe_len}, mean_rarity={safe_rarity})")
    print(f"\nacross categories: AUC~rarity rho={rho_rar:.2f} (p={p_rar:.3f});  "
          f"AUC~length rho={rho_len:.2f} (p={p_len:.3f});  "
          f"gate-AUC~length-only-AUC rho={rho_rank:.2f} (p={p_rank:.3f})")
    print(f"\nconfound H1: gate={auc_gate}  lexical={auc_lex}  "
          f"gate+lexical={auc_both}  dAUC(gate over lexical) CI={d_gate_over_lex}")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
