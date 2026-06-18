# -*- coding: utf-8 -*-
"""
A1 — Verbalized vs. internal confidence. (PREREGISTRATION.md, commit ee43c6b)

Two passes per held-out RTE/QNLI item through Qwen2.5-3B-Instruct (greedy):
  1. answer pass  -> top-5 logprobs -> gate features -> internal calibrated p_correct;
                     parsed answer -> correct vs GLUE gold.
  2. confidence pass (neutral) -> parsed 0-100 -> verbal confidence.
Compares how well each confidence signal predicts correctness. Raw text ->
local_outputs/ (gitignored); npz/report = aggregates only.

  --run --seed 42|1337     generate (GPU); saves selfreport_rows_<seed>.npz
  --analyze                CPU: AUCs, paired CI, ECE, overconfidence gap, combo, nulls
"""

import argparse, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

from calibrate_gate import TASKS, _first_of                       # noqa
from eval_gate_value import calibration_exclusion_indices         # noqa
from mirrorfield.mcp.uncertainty import (                         # noqa
    compute_token_margins, compute_token_entropies, compute_boundary_ratio,
    calibrated_p_correct,
)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_PER_TASK = 250
N_BOOT = 2000
POS = {"rte": ("yes", "no"), "qnli": ("yes", "no")}
FRESH = {"rte": ("train", N_PER_TASK), "qnli": ("validation", N_PER_TASK)}
CONF_INSTR = ("\n\nEstimate the probability that your answer is correct, as a percentage "
              "from 0 to 100. Reply with only the number.")


def select_items(seed):
    from datasets import load_dataset
    excl = calibration_exclusion_indices()
    rng = np.random.RandomState(seed)
    items = []
    for t, (split, n) in FRESH.items():
        ds = load_dataset("glue", t, split=split)
        cand = [i for i in range(len(ds)) if not (split == "validation" and i in excl[t])]
        rng.shuffle(cand)
        cfg = TASKS[t]; f = cfg["fields"]
        for i in cand[:n]:
            ex = ds[i]
            items.append({"task": t, "ds_index": int(i),
                          "prompt": cfg["prompt"].format(*[ex[k] for k in f]),
                          "gold": cfg["gold"](ex["label"])})
    return items


def _load():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(MODEL, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return torch, tok, model


def _gen(torch, tok, model, text, max_new, want_lp):
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             output_scores=want_lp, return_dict_in_generate=True,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True).strip()
    tl = None
    if want_lp:
        tl = []
        for sc in out.scores:
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            v, ix = lp.topk(5)
            tl.append({int(i): float(x) for i, x in zip(ix, v)})
    return txt, tl


def run(seed):
    import torch
    torch.manual_seed(seed)
    items = select_items(seed)
    print(f"[A1] {len(items)} items, seed {seed}")
    torchmod, tok, model = _load()
    rows, raw = [], []
    for k, it in enumerate(items):
        ans, tl = _gen(torchmod, tok, model, it["prompt"], 16, True)
        margins = compute_token_margins(tl); ent = compute_token_entropies(tl)
        fin = margins[np.isfinite(margins)]
        mm = round(float(fin.mean()), 4) if len(fin) else float("nan")
        me = round(float(ent.mean()), 4)
        br = round(compute_boundary_ratio(margins), 4)
        p_int = calibrated_p_correct(mm, me, br) if np.isfinite(mm) else None
        pred = _first_of(ans.lower(), *POS[it["task"]])
        ans_ok = pred is not None
        correct = int(pred == it["gold"]) if ans_ok else -1
        ctext, _ = _gen(torchmod, tok, model,
                        f"{it['prompt']}\n\nYour answer was: {ans}{CONF_INSTR}", 12, False)
        m = re.search(r"\b(\d{1,3})\b", ctext)
        conf_ok = bool(m) and 0 <= int(m.group(1)) <= 100
        verbal = (int(m.group(1)) / 100.0) if conf_ok else float("nan")
        rows.append({"task": it["task"], "correct": correct,
                     "p_int": p_int if p_int is not None else float("nan"),
                     "verbal": verbal, "ans_ok": int(ans_ok), "conf_ok": int(conf_ok)})
        raw.append({"task": it["task"], "idx": it["ds_index"], "answer": ans, "conf": ctext})
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(items)}")
    (HERE / "local_outputs").mkdir(exist_ok=True)
    (HERE / "local_outputs" / f"selfreport_texts_{seed}.json").write_text(json.dumps(raw), encoding="utf-8")
    np.savez(HERE / f"selfreport_rows_{seed}.npz",
             task=np.array([r["task"] for r in rows]),
             correct=np.array([r["correct"] for r in rows]),
             p_int=np.array([r["p_int"] for r in rows]),
             verbal=np.array([r["verbal"] for r in rows]),
             ans_ok=np.array([r["ans_ok"] for r in rows]),
             conf_ok=np.array([r["conf_ok"] for r in rows]))
    print(f"  -> saved selfreport_rows_{seed}.npz")


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _ece(y, s, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for b in range(bins):
        m = (s >= edges[b]) & (s < edges[b + 1]) if b < bins - 1 else (s >= edges[b]) & (s <= edges[b + 1])
        if m.sum():
            e += (m.sum() / len(s)) * abs(y[m].mean() - s[m].mean())
    return float(e)


def analyze(seeds=(42, 1337)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    out = {}
    for seed in seeds:
        p = HERE / f"selfreport_rows_{seed}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        task, correct = d["task"], d["correct"]
        p_int, verbal = d["p_int"], d["verbal"]
        ans_ok, conf_ok = d["ans_ok"], d["conf_ok"]
        usable = (ans_ok == 1) & (conf_ok == 1) & np.isfinite(p_int) & np.isfinite(verbal) & (correct >= 0)
        y = correct[usable].astype(int); pi = p_int[usable]; vb = verbal[usable]
        rng = np.random.RandomState(seed)
        auc_i, auc_v = _auc(y, pi), _auc(y, vb)
        # paired bootstrap on the AUC difference (internal - verbal)
        diffs = []
        for _ in range(N_BOOT):
            b = rng.randint(0, len(y), len(y))
            if len(np.unique(y[b])) == 2:
                diffs.append(_auc(y[b], pi[b]) - _auc(y[b], vb[b]))
        # combined OOF logistic (StandardScaler: internal feature is narrow-range, std~0.05,
        # so an unscaled logistic underfits it and collapses combined AUC to chance)
        X = np.column_stack([pi, vb]); oof = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
            if len(np.unique(y[tr])) == 2:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
                oof[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        auc_comb = _auc(y, oof)
        # shuffled-label null
        yp = rng.permutation(y)
        out[str(seed)] = {
            "n_usable": int(usable.sum()), "accuracy": round(float(y.mean()), 4),
            "answer_parse_rate": round(float(ans_ok.mean()), 4),
            "verbal_parse_rate": round(float(conf_ok.mean()), 4),
            "verbal_mean": round(float(vb.mean()), 4), "verbal_std": round(float(vb.std()), 4),
            "overconfidence_gap": round(float(vb.mean() - y.mean()), 4),
            "auc_internal": round(auc_i, 4), "auc_verbal": round(auc_v, 4),
            "auc_diff_internal_minus_verbal": round(auc_i - auc_v, 4),
            "auc_diff_ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                              round(float(np.percentile(diffs, 97.5)), 4)] if diffs else None,
            "auc_combined_oof": round(auc_comb, 4),
            "ece_internal": round(_ece(y, pi), 4), "ece_verbal": round(_ece(y, vb), 4),
            "null_auc_internal": round(_auc(yp, pi), 4), "null_auc_verbal": round(_auc(yp, vb), 4),
        }
    (HERE / "selfreport_results.json").write_text(json.dumps(out, indent=2))
    # digest
    print(f"\n{'='*74}\n A1 — VERBALIZED vs INTERNAL CONFIDENCE\n{'='*74}")
    for s, r in out.items():
        print(f"\n seed {s}  (n={r['n_usable']}, accuracy={r['accuracy']}, "
              f"parse ans/verbal={r['answer_parse_rate']}/{r['verbal_parse_rate']})")
        print(f"  verbal: mean={r['verbal_mean']} std={r['verbal_std']}  "
              f"overconfidence gap={r['overconfidence_gap']}")
        print(f"  AUC(correct~internal) = {r['auc_internal']}   ECE={r['ece_internal']}")
        print(f"  AUC(correct~verbal)   = {r['auc_verbal']}   ECE={r['ece_verbal']}")
        print(f"  Δ internal−verbal     = {r['auc_diff_internal_minus_verbal']} {r['auc_diff_ci95']}"
              f"   (CI excl 0 ⇒ internal beats verbal)")
        print(f"  combined OOF AUC      = {r['auc_combined_oof']}")
        print(f"  null AUCs (shuffled)  = int {r['null_auc_internal']} / verbal {r['null_auc_verbal']} (≈0.5)")
    # verdict (primary seed; require replication)
    if out:
        s0 = list(out)[0]; r0 = out[s0]
        ci = r0["auc_diff_ci95"]
        rep = (len(out) > 1 and list(out)[1] and out[list(out)[1]]["auc_diff_ci95"]
               and out[list(out)[1]]["auc_diff_ci95"][0] > 0)
        if ci and ci[0] > 0:
            v = "INTERNAL BEATS VERBAL" + (" (replicated)" if rep else " (seed-1 only; confirm 1337)")
        elif ci and ci[1] < 0:
            v = "VERBAL BEATS INTERNAL (unexpected — inspect)"
        else:
            v = "NULL — verbal ≈ internal on discrimination"
        if r0["verbal_std"] < 0.05:
            v += f"  [verbal pinned: std={r0['verbal_std']} ⇒ overconfident/non-discriminating]"
        print(f"\n VERDICT: {v}")
    print("\nsaved -> selfreport_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.run:
        run(a.seed)
    elif a.analyze:
        analyze()
    else:
        print("use --run --seed N  or  --analyze")
