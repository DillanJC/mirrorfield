# -*- coding: utf-8 -*-
"""
Self-consistency replication across hard tasks — lock in the §4d positive.

§4d showed self-consistency (sample disagreement) predicts wrong outputs on RTE
at AUC 0.65 [0.57,0.72] where the log-prob gate was at chance. This replicates
the SAME test on a SECOND hard entailment task (QNLI) to confirm it generalizes.

Same method, non-circular label (wrong = greedy answer != gold), bootstrap CIs.
Per example: 1 greedy pass (standard features + deployed answer) + N sampled
passes (disagreement). Compares STANDARD vs SELF-CONSISTENCY vs STANDARD+SC.

Run:
    python selfconsistency_multi.py                      # dry-run
    python selfconsistency_multi.py --run                # rte + qnli
    python selfconsistency_multi.py --run --tasks qnli --n 300
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
N_VOTES = 10
TEMPERATURE = 1.0
TOP_K_LOGPROBS = 5
MAX_NEW_GREEDY = 24
MAX_NEW_VOTE = 8
N_BOOT = 2000
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_JSON = Path(__file__).parent / "selfconsistency_multi_results.json"

TASKS = {
    "rte": {
        "load": ("glue", "rte", "validation"),
        "prompt": ("Premise: {0}\nHypothesis: {1}\nDoes the premise entail the "
                   "hypothesis? Answer yes or no first, then a brief reason."),
        "fields": ("sentence1", "sentence2"),
        "gold": lambda lab: "yes" if lab == 0 else "no",   # 0=entailment
    },
    "qnli": {
        "load": ("glue", "qnli", "validation"),
        "prompt": ("Question: {0}\nSentence: {1}\nDoes the sentence contain the "
                   "answer to the question? Answer yes or no first, then a brief reason."),
        "fields": ("question", "sentence"),
        "gold": lambda lab: "yes" if lab == 0 else "no",   # 0=entailment(answer present)
    },
}


def _first_of(text, a, b):
    ia, ib = text.find(a), text.find(b)
    if ia == -1 and ib == -1:
        return None
    if ib == -1 or (ia != -1 and ia < ib):
        return a
    return b


def load_task(task, n, rng):
    from datasets import load_dataset
    cfg = TASKS[task]
    src, name, split = cfg["load"]
    ds = load_dataset(src, name, split=split)
    f0, f1 = cfg["fields"]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:min(n, len(ds))]
    out = []
    for i in idx:
        ex = ds[i]
        gold = cfg["gold"](ex["label"])
        out.append((cfg["prompt"].format(ex[f0], ex[f1]), gold))
    return out


def build(items, tok, model, device):
    import torch
    rows = []
    for ptext, gold in items:
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(device)
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

        with torch.no_grad():
            s = model.generate(ids, max_new_tokens=MAX_NEW_VOTE, do_sample=True,
                               temperature=TEMPERATURE, num_return_sequences=N_VOTES,
                               return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        votes = [_first_of(tok.decode(s.sequences[j, ids.shape[1]:],
                                      skip_special_tokens=True).lower(), "yes", "no")
                 for j in range(N_VOTES)]
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
        scl = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(scl.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(scl.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof), oof


def boot_ci(y, oof, rng):
    from sklearn.metrics import roc_auc_score
    vals = []
    N = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, N, N)
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(roc_auc_score(y[bi], oof[bi]))
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def score(rows, rng):
    from sklearn.metrics import roc_auc_score
    y = np.array([r["wrong"] for r in rows])
    if len(np.unique(y)) < 2:
        return {"error": "one class", "wrong_rate": float(y.mean())}
    Xs = np.array([r["std"] for r in rows])
    Xc = np.array([r["sc"] for r in rows])
    Xsc = np.hstack([Xs, Xc])
    auc_s, oof_s = cv_auc(Xs, y)
    auc_c, oof_c = cv_auc(Xc, y)
    auc_sc, oof_sc = cv_auc(Xsc, y)
    d = []
    N = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, N, N)
        if len(np.unique(y[bi])) < 2:
            continue
        d.append(roc_auc_score(y[bi], oof_sc[bi]) - roc_auc_score(y[bi], oof_s[bi]))
    d_lo, d_hi = np.percentile(d, [2.5, 97.5])
    return {
        "n": int(N), "n_wrong": int(y.sum()), "wrong_rate": round(float(y.mean()), 4),
        "auc_standard": round(auc_s, 4), "auc_standard_ci95": list(boot_ci(y, oof_s, rng)),
        "auc_selfconsistency": round(auc_c, 4), "auc_sc_ci95": list(boot_ci(y, oof_c, rng)),
        "auc_combined": round(auc_sc, 4),
        "delta_sc_over_standard": round(float(auc_sc - auc_s), 4),
        "delta_ci95": [round(float(d_lo), 4), round(float(d_hi), 4)],
        "sc_works": bool(boot_ci(y, oof_c, rng)[0] > 0.5),
        "sc_beats_gate": bool(d_lo > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--tasks", default="rte,qnli")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fallback", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]

    print("=" * 70)
    print("SELF-CONSISTENCY REPLICATION across hard tasks")
    print("=" * 70)
    print(f"Model {args.model} | tasks={tasks} | n<= {args.n} | {N_VOTES} votes @ T={TEMPERATURE}")
    if not args.run:
        print("\n[DRY RUN] no download/generation. Use --run.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # seed EVERYTHING incl. the sampling RNG (the bug that made §4d non-reproducible)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    def _load(name):
        tk = AutoTokenizer.from_pretrained(name)
        md = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        return tk, md

    used_model = args.model
    try:
        tok, model = _load(args.model)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"\n[OOM] {args.model} did not fit; falling back to {args.fallback}")
        used_model = args.fallback
        tok, model = _load(args.fallback)

    rng = np.random.RandomState(SEED)
    import random; random.seed(SEED)
    results = {}
    for task in tasks:
        print(f"\n--- {task}: loading + generating (n<= {args.n}) ---")
        items = load_task(task, args.n, rng)
        rows = build(items, tok, model, device)
        r = score(rows, rng)
        results[task] = r
        print(f"  wrong%={r.get('wrong_rate')}  "
              f"STD={r.get('auc_standard')}{r.get('auc_standard_ci95')}  "
              f"SC={r.get('auc_selfconsistency')}{r.get('auc_sc_ci95')}  "
              f"dSC={r.get('delta_sc_over_standard')}{r.get('delta_ci95')}")

    tag = used_model.split("/")[-1].replace(".", "").replace("-", "_")
    out_path = OUT_JSON.with_name(f"selfconsistency_{tag}_results.json")
    results["_model"] = used_model
    out_path.write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 70)
    print(f"MODEL USED: {used_model}")
    print(f"{'task':6} {'wrong%':>7} {'STD AUC (CI)':>22} {'SC AUC (CI)':>22} {'SC works':>9}")
    print("-" * 70)
    for t, r in results.items():
        if not isinstance(r, dict):
            continue
        if "error" in r:
            print(f"{t:6} {r['error']}"); continue
        print(f"{t:6} {r['wrong_rate']*100:6.1f}% "
              f"{r['auc_standard']:.3f}[{r['auc_standard_ci95'][0]:.2f},{r['auc_standard_ci95'][1]:.2f}]  "
              f"{r['auc_selfconsistency']:.3f}[{r['auc_sc_ci95'][0]:.2f},{r['auc_sc_ci95'][1]:.2f}]  "
              f"{'YES' if r['sc_works'] else 'no':>9}")
    print(f"\nSaved -> {out_path}")
    print("LOCK-IN: SC CI lower bound > 0.5 on BOTH tasks => the finding generalizes.")


if __name__ == "__main__":
    main()
