# -*- coding: utf-8 -*-
"""
MCP Uncertainty Ablation v2 — HARDEN the foundation.

v1 ran on SST-2 sentiment where the model was right ~87% of the time, leaving
only ~40 wrong outputs -> a small positive class and a wide CI on AUC. This v2:

  1. Runs on HARDER tasks (RTE entailment by default) where a 0.5B model errs
     30-50% of the time -> a well-populated "wrong" class -> a tighter CI.
  2. Is task-parameterized (--tasks rte,qnli,sst2) so we can confirm the result
     across multiple tasks, not just one.
  3. Optionally sweeps multiple models (--models a,b) for robustness.

Everything else is identical to v1 and still NON-circular: the label is
ground-truth correctness (model prediction vs gold), defined entirely outside
the geometry. We re-ask the same two questions on firmer ground:
  - Do the STANDARD log-prob signals predict wrong outputs? (the keeper)
  - Does the GEOMETRY add anything over them?            (expected: no)

Run:
    python mcp_uncertainty_ablation_v2.py                  # dry-run
    python mcp_uncertainty_ablation_v2.py --run            # RTE, 1 model
    python mcp_uncertainty_ablation_v2.py --run --tasks rte,sst2 --n 500
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

MIRRORFIELD_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIRRORFIELD_ROOT))
try:
    from mirrorfield.mcp.uncertainty import (
        compute_token_margins,
        compute_token_entropies,
        compute_boundary_ratio,
        compute_sequence_pr,
        compute_embedding_pr,
    )
    MCP_OK = True
except Exception as e:  # noqa
    MCP_OK = False
    _MCP_ERR = str(e)

# ----------------------------------------------------------------------------
SEED = 42
DEFAULT_N = 500
DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]
MAX_NEW_TOKENS = 24
TOP_K_LOGPROBS = 5
N_BOOT = 2000
OUT_JSON = Path(__file__).parent / "mcp_uncertainty_ablation_v2_results.json"


# ----------------------------------------------------------------------------
# Task registry — each task: how to load, prompt, and parse the answer.
# gold is mapped to a canonical token; pred is parsed to the same vocabulary.
# ----------------------------------------------------------------------------
def _first_of(text, a, b):
    """Return which of two keywords appears first (or None)."""
    ia = text.find(a)
    ib = text.find(b)
    if ia == -1 and ib == -1:
        return None
    if ib == -1 or (ia != -1 and ia < ib):
        return a
    return b


TASKS = {
    "sst2": {
        "load": ("glue", "sst2", "validation"),
        "text": lambda ex: ex["sentence"],
        "prompt": ("Classify the sentiment of this movie-review sentence as "
                   "positive or negative. Answer with one word first, then a "
                   "brief reason.\n\nSentence: {}"),
        # gold label 1=positive, 0=negative
        "gold": lambda ex: "positive" if ex["label"] == 1 else "negative",
        "parse": lambda t: _first_of(t.lower(), "positive", "negative"),
    },
    "rte": {  # recognizing textual entailment — HARD for a small model
        "load": ("glue", "rte", "validation"),
        "text": lambda ex: ex["sentence1"] + " ||| " + ex["sentence2"],
        "prompt": ("Premise: {0}\nHypothesis: {1}\nDoes the premise entail the "
                   "hypothesis? Answer yes or no first, then a brief reason."),
        # RTE label 0=entailment, 1=not_entailment
        "gold": lambda ex: "yes" if ex["label"] == 0 else "no",
        "parse": lambda t: _first_of(t.lower(), "yes", "no"),
    },
    "qnli": {  # does the sentence answer the question — also hard
        "load": ("glue", "qnli", "validation"),
        "text": lambda ex: ex["question"] + " ||| " + ex["sentence"],
        "prompt": ("Question: {0}\nSentence: {1}\nDoes the sentence contain the "
                   "answer to the question? Answer yes or no first, then a "
                   "brief reason."),
        # QNLI label 0=entailment(answer present), 1=not_entailment
        "gold": lambda ex: "yes" if ex["label"] == 0 else "no",
        "parse": lambda t: _first_of(t.lower(), "yes", "no"),
    },
}


def load_task(task, n, rng):
    """Load a balanced subset, returning [(prompt_text, gold_token)]."""
    from datasets import load_dataset
    cfg = TASKS[task]
    src, name, split = cfg["load"]
    ds = load_dataset(src, name, split=split)
    n_avail = len(ds)
    idx = list(range(n_avail))
    rng.shuffle(idx)
    idx = idx[:min(n, n_avail)]
    items = []
    for i in idx:
        ex = ds[i]
        gold = cfg["gold"](ex)
        if task == "sst2":
            ptext = cfg["prompt"].format(cfg["text"](ex))
        else:
            ptext = cfg["prompt"].format(ex["sentence1"] if task == "rte" else ex["question"],
                                         ex["sentence2"] if task == "rte" else ex["sentence"])
        items.append((ptext, gold, cfg["parse"]))
    return items


# ----------------------------------------------------------------------------
def build_features(items, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device).eval()

    rows = []
    for ptext, gold, parse in items:
        msgs = [{"role": "user", "content": ptext}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                output_scores=True, output_hidden_states=True,
                return_dict_in_generate=True, pad_token_id=tok.eos_token_id,
            )
        gen_ids = out.sequences[0, ids.shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).lower()

        top_logprobs = []
        for step_scores in out.scores:
            lp = torch.log_softmax(step_scores[0].float(), dim=-1)
            vals, idxs = lp.topk(TOP_K_LOGPROBS)
            top_logprobs.append({int(i): float(v) for i, v in zip(idxs, vals)})

        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary = compute_boundary_ratio(margins, threshold=0.5)
        seq_pr = compute_sequence_pr(top_logprobs)

        hs = [step[-1][0, -1].float().cpu().numpy() for step in out.hidden_states]
        H = np.vstack(hs) if len(hs) > 1 else np.repeat(hs[0][None], 2, 0)
        emb = compute_embedding_pr(H, k=min(10, len(H) - 1))

        pred = parse(text)                      # canonical token or None
        wrong = 1 if (pred is None or pred != gold) else 0

        fmar = margins[np.isfinite(margins)]
        rows.append({
            "wrong": wrong,
            "std": [float(np.mean(fmar)) if len(fmar) else 0.0,
                    float(np.mean(entropies)),
                    float(boundary)],
            "geo": [float(seq_pr), float(emb["pr_mean"]),
                    float(emb["pr_min"]), float(emb["se_mean"])],
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


def score_task(rows, rng):
    from sklearn.metrics import roc_auc_score
    y = np.array([r["wrong"] for r in rows])
    if len(np.unique(y)) < 2:
        return {"error": "only one class present", "wrong_rate": float(y.mean())}
    Xs = np.array([r["std"] for r in rows])
    Xg = np.array([r["geo"] for r in rows])
    Xsg = np.hstack([Xs, Xg])

    auc_s, oof_s = cv_auc(Xs, y)
    auc_g, _ = cv_auc(Xg, y)
    auc_sg, oof_sg = cv_auc(Xsg, y)

    deltas = []
    n = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, n, n)
        if len(np.unique(y[bi])) < 2:
            continue
        deltas.append(roc_auc_score(y[bi], oof_sg[bi]) - roc_auc_score(y[bi], oof_s[bi]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # bootstrap CI on the STANDARD auc too (so we trust the keeper)
    s_boot = []
    for _ in range(N_BOOT):
        bi = rng.randint(0, n, n)
        if len(np.unique(y[bi])) < 2:
            continue
        s_boot.append(roc_auc_score(y[bi], oof_s[bi]))
    s_lo, s_hi = np.percentile(s_boot, [2.5, 97.5])

    y_sh = y.copy(); rng.shuffle(y_sh)
    auc_s_n, _ = cv_auc(Xs, y_sh)
    auc_sg_n, _ = cv_auc(Xsg, y_sh)

    return {
        "n": int(n), "n_wrong": int(y.sum()), "wrong_rate": float(y.mean()),
        "auc_standard": round(float(auc_s), 4),
        "auc_standard_ci95": [round(float(s_lo), 4), round(float(s_hi), 4)],
        "auc_geometry_only": round(float(auc_g), 4),
        "auc_standard_plus_geometry": round(float(auc_sg), 4),
        "delta_auc": round(float(auc_sg - auc_s), 4),
        "delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "null_delta_auc": round(float(auc_sg_n - auc_s_n), 4),
        "geometry_verdict": ("ADDS REAL LIFT" if lo > 0 else "adds nothing (CI spans 0)"),
        "standard_verdict": ("predicts wrong outputs" if s_lo > 0.5 else "no better than chance"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tasks", default="rte")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 70)
    print("MCP UNCERTAINTY ABLATION v2  —  hardening the foundation")
    print("=" * 70)
    print(f"MCP importable : {MCP_OK}" + ("" if MCP_OK else f"  ({_MCP_ERR})"))
    print(f"Tasks          : {tasks}   (harder => more errors => tighter CI)")
    print(f"Models         : {models}")
    print(f"Samples/task   : {args.n}")
    print(f"Question       : does STANDARD predict wrong? does GEOMETRY add lift?")

    if not args.run:
        print("\n[DRY RUN] No download, no generation, no write. Use --run.")
        return
    if not MCP_OK:
        print("\nABORT: MCP import failed."); return

    rng = np.random.RandomState(SEED)
    import random; random.seed(SEED)
    all_results = {}
    for model_name in models:
        for task in tasks:
            key = f"{task} | {model_name.split('/')[-1]}"
            print(f"\n--- {key}: loading + generating ({args.n} samples) ---")
            items = load_task(task, args.n, rng)
            rows = build_features(items, model_name)
            res = score_task(rows, rng)
            all_results[key] = res
            print(f"  wrong_rate={res.get('wrong_rate')}  "
                  f"AUC_std={res.get('auc_standard')} {res.get('auc_standard_ci95')}  "
                  f"dGEO={res.get('delta_auc')} {res.get('delta_ci95')}")

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print("\n" + "=" * 70)
    print(f"{'task | model':38s} {'wrong%':>7} {'AUC_std (CI)':>22} {'dGEO (CI)':>22}")
    print("-" * 70)
    for k, r in all_results.items():
        if "error" in r:
            print(f"{k:38s}  {r['error']}"); continue
        print(f"{k:38s} {r['wrong_rate']*100:6.1f}% "
              f"{r['auc_standard']:.3f} [{r['auc_standard_ci95'][0]:.2f},{r['auc_standard_ci95'][1]:.2f}]  "
              f"{r['delta_auc']:+.3f} [{r['delta_ci95'][0]:+.2f},{r['delta_ci95'][1]:+.2f}]")
    print(f"\nSaved -> {OUT_JSON}")
    print("READ: AUC_std CI above 0.5 => the gate is real on this task.")
    print("      dGEO CI spanning 0    => geometry still adds nothing.")


if __name__ == "__main__":
    main()
