# -*- coding: utf-8 -*-
"""
Self-consistency vs the standard gate on RTE — the decisive Road-B test.

v2 showed the standard log-prob gate (margin/entropy/boundary) is at chance on
RTE (AUC 0.55, CI includes 0.5): log-probs can't see *confident reasoning errors*.

Hypothesis (Road B): sampling the model N times and measuring how much the
answers DISAGREE catches exactly that failure mode — when samples scatter, the
greedy answer is more likely wrong.

Method (NON-circular, same gold-correctness label as v2):
  For each RTE example:
    - 1 greedy pass  -> STANDARD features (margin/entropy/boundary) + greedy answer
    - N sampled passes (temperature) -> SELF-CONSISTENCY features:
        agreement   = max(#yes,#no)/N   (1.0 unanimous, 0.5 max disagreement)
        flips_greedy = 1 if sampled majority != greedy answer
  Label: wrong = (greedy answer != gold).   [the deployed answer is greedy]
  Compare 5-fold CV AUC at predicting "wrong":
        STANDARD   vs   SELF-CONSISTENCY   vs   STANDARD + SC
  Bootstrap 95% CI on each AUC and on the SC-over-standard delta.

Decisive read:
  - SC-alone AUC CI above 0.5  => sampling sees what log-probs missed (Road B real)
  - (STD+SC) - STD delta CI above 0 => SC adds genuine lift over the gate

Run:
    python selfconsistency_rte.py            # dry-run
    python selfconsistency_rte.py --run      # ~277 RTE x (1+N) generations
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

MIRRORFIELD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIRRORFIELD_ROOT))
from mirrorfield.mcp.uncertainty import (   # noqa
    compute_token_margins,
    compute_token_entropies,
    compute_boundary_ratio,
)

SEED = 42
N_SAMPLES = 277          # full RTE validation
N_VOTES = 10             # sampled generations per example
TEMPERATURE = 1.0
TOP_K_LOGPROBS = 5
MAX_NEW_GREEDY = 24
MAX_NEW_VOTE = 8
N_BOOT = 2000
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_JSON = Path(__file__).parent / "selfconsistency_rte_results.json"

PROMPT = ("Premise: {0}\nHypothesis: {1}\nDoes the premise entail the "
          "hypothesis? Answer yes or no first, then a brief reason.")


def _first_of(text, a, b):
    ia, ib = text.find(a), text.find(b)
    if ia == -1 and ib == -1:
        return None
    if ib == -1 or (ia != -1 and ia < ib):
        return a
    return b


def load_rte(n, rng):
    from datasets import load_dataset
    ds = load_dataset("glue", "rte", split="validation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:min(n, len(ds))]
    out = []
    for i in idx:
        ex = ds[i]
        gold = "yes" if ex["label"] == 0 else "no"   # 0=entailment
        out.append((PROMPT.format(ex["sentence1"], ex["sentence2"]), gold))
    return out


def build(items, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    rows = []
    for ptext, gold in items:
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        # --- greedy pass: standard features + deployed answer
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=MAX_NEW_GREEDY, do_sample=False,
                               output_scores=True, return_dict_in_generate=True,
                               pad_token_id=tok.eos_token_id)
        gtext = tok.decode(g.sequences[0, ids.shape[1]:], skip_special_tokens=True).lower()
        top_logprobs = []
        for sc in g.scores:
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            vals, idxs = lp.topk(TOP_K_LOGPROBS)
            top_logprobs.append({int(i): float(v) for i, v in zip(idxs, vals)})
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary = compute_boundary_ratio(margins, threshold=0.5)
        greedy_pred = _first_of(gtext, "yes", "no")

        # --- N sampled passes (one call, num_return_sequences) -> disagreement
        with torch.no_grad():
            s = model.generate(ids, max_new_tokens=MAX_NEW_VOTE, do_sample=True,
                               temperature=TEMPERATURE, num_return_sequences=N_VOTES,
                               return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        votes = []
        for j in range(N_VOTES):
            vtext = tok.decode(s.sequences[j, ids.shape[1]:], skip_special_tokens=True).lower()
            votes.append(_first_of(vtext, "yes", "no"))
        n_yes = sum(1 for v in votes if v == "yes")
        n_no = sum(1 for v in votes if v == "no")
        decided = n_yes + n_no
        agreement = (max(n_yes, n_no) / decided) if decided else 0.5
        majority = "yes" if n_yes >= n_no else "no"
        flips = 1.0 if (greedy_pred is not None and majority != greedy_pred) else 0.0

        wrong = 1 if (greedy_pred is None or greedy_pred != gold) else 0
        fm = margins[np.isfinite(margins)]
        rows.append({
            "wrong": wrong,
            "std": [float(np.mean(fm)) if len(fm) else 0.0,
                    float(np.mean(entropies)), float(boundary)],
            "sc": [float(agreement), float(flips)],
        })
    return rows


def cv_auc(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof), oof


def boot_ci(y, oof, rng, n=N_BOOT):
    from sklearn.metrics import roc_auc_score
    vals = []
    N = len(y)
    for _ in range(n):
        bi = rng.randint(0, N, N)
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(roc_auc_score(y[bi], oof[bi]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("SELF-CONSISTENCY vs STANDARD GATE on RTE  (Road-B decisive test)")
    print("=" * 70)
    print(f"Model {MODEL} | {N_SAMPLES} RTE | {N_VOTES} votes @ T={TEMPERATURE}")
    print("Q: does sample-disagreement predict wrong where log-probs can't?")
    if not args.run:
        print("\n[DRY RUN] no download/generation. Use --run.")
        return

    rng = np.random.RandomState(SEED)
    import random; random.seed(SEED)
    items = load_rte(N_SAMPLES, rng)
    print(f"\nLoaded {len(items)} RTE. Generating greedy + {N_VOTES} votes each...")
    rows = build(items, MODEL)

    y = np.array([r["wrong"] for r in rows])
    Xs = np.array([r["std"] for r in rows])
    Xc = np.array([r["sc"] for r in rows])
    Xsc = np.hstack([Xs, Xc])
    print(f"wrong_rate={y.mean():.3f} (n_wrong={int(y.sum())}/{len(y)})")

    auc_s, oof_s = cv_auc(Xs, y)
    auc_c, oof_c = cv_auc(Xc, y)
    auc_sc, oof_sc = cv_auc(Xsc, y)
    ci_s = boot_ci(y, oof_s, rng)
    ci_c = boot_ci(y, oof_c, rng)
    ci_sc = boot_ci(y, oof_sc, rng)

    from sklearn.metrics import roc_auc_score
    d = []
    N = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, N, N)
        if len(np.unique(y[bi])) < 2:
            continue
        d.append(roc_auc_score(y[bi], oof_sc[bi]) - roc_auc_score(y[bi], oof_s[bi]))
    d_lo, d_hi = np.percentile(d, [2.5, 97.5])

    res = {
        "n": int(N), "n_wrong": int(y.sum()), "wrong_rate": round(float(y.mean()), 4),
        "auc_standard": round(auc_s, 4), "auc_standard_ci95": [round(ci_s[0], 4), round(ci_s[1], 4)],
        "auc_selfconsistency": round(auc_c, 4), "auc_sc_ci95": [round(ci_c[0], 4), round(ci_c[1], 4)],
        "auc_standard_plus_sc": round(auc_sc, 4), "auc_combined_ci95": [round(ci_sc[0], 4), round(ci_sc[1], 4)],
        "delta_sc_over_standard": round(float(auc_sc - auc_s), 4),
        "delta_ci95": [round(float(d_lo), 4), round(float(d_hi), 4)],
        "sc_sees_what_logprobs_missed": bool(ci_c[0] > 0.5),
        "sc_adds_lift_over_gate": bool(d_lo > 0),
    }
    print("\n" + "-" * 70)
    for k, v in res.items():
        print(f"{k:30s}: {v}")
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT_JSON}")
    print("READ: auc_sc_ci95 lower bound > 0.5  => self-consistency works on RTE.")
    print("      delta_ci95 lower bound > 0      => it beats the log-prob gate.")


if __name__ == "__main__":
    main()
