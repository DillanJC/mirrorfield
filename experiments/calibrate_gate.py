# -*- coding: utf-8 -*-
"""
Calibrate the lean uncertainty gate -> a TRUE probability of correctness.

The gate's raw confidence score ranks outputs (AUC ~0.6) but is not a calibrated
probability. This trains a calibrated P(correct) from the validated signals
(margin / entropy / boundary) on real labeled generations, reports calibration
quality (ECE, Brier, reliability table), and EXPORTS numpy-only parameters so the
MCP can emit a real probability with no sklearn dependency at runtime.

Pipeline (honest, cross-validated):
  generate labeled data (greedy) -> features + correct/incorrect label
  -> StandardScaler + LogisticRegression  (5-fold out-of-fold probs)
  -> isotonic recalibration on the OOF probs
  -> report AUC / Brier / ECE before vs after
  -> export: scaler mean/scale, logreg coef/intercept, isotonic x/y table
     to  mirrorfield/mcp/gate_calibration.json

Apply at runtime (numpy):
    z = (feat - mean) / scale
    p_raw = sigmoid(coef . z + intercept)
    p_cal = interp(p_raw, iso_x, iso_y)

Run:
    python calibrate_gate.py                 # dry-run
    python calibrate_gate.py --run           # 3B, rte+qnli, generate+fit+export
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
N_VOTES = 0
TOP_K_LOGPROBS = 5
MAX_NEW = 24
MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPORT = MIRRORFIELD_ROOT / "mirrorfield" / "mcp" / "gate_calibration.json"
REPORT = Path(__file__).parent / "calibrate_gate_report.json"

TASKS = {
    "rte": {"load": ("glue", "rte", "validation"),
            "prompt": ("Premise: {0}\nHypothesis: {1}\nDoes the premise entail the "
                       "hypothesis? Answer yes or no first, then a brief reason."),
            "fields": ("sentence1", "sentence2"),
            "gold": lambda lab: "yes" if lab == 0 else "no"},
    "qnli": {"load": ("glue", "qnli", "validation"),
             "prompt": ("Question: {0}\nSentence: {1}\nDoes the sentence contain the "
                        "answer to the question? Answer yes or no first, then a brief reason."),
             "fields": ("question", "sentence"),
             "gold": lambda lab: "yes" if lab == 0 else "no"},
    "sst2": {"load": ("glue", "sst2", "validation"),
             "prompt": ("Classify the sentiment of this sentence as positive or "
                        "negative. Answer with one word first, then a brief reason.\n\nSentence: {0}"),
             "fields": ("sentence",),
             "gold": lambda lab: "positive" if lab == 1 else "negative"},
}


def _first_of(text, a, b):
    ia, ib = text.find(a), text.find(b)
    if ia == -1 and ib == -1:
        return None
    return a if (ib == -1 or (ia != -1 and ia < ib)) else b


def load_task(task, n, rng):
    from datasets import load_dataset
    cfg = TASKS[task]
    src, name, split = cfg["load"]
    ds = load_dataset(src, name, split=split)
    fields = cfg["fields"]
    idx = list(range(len(ds))); rng.shuffle(idx); idx = idx[:min(n, len(ds))]
    pos = ("positive", "negative") if task == "sst2" else ("yes", "no")
    out = []
    for i in idx:
        ex = ds[i]
        gold = cfg["gold"](ex["label"])
        vals = [ex[f] for f in fields]
        out.append((cfg["prompt"].format(*vals), gold, pos))
    return out


def generate(items, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    X, y = [], []
    for ptext, gold, pos in items:
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                               output_scores=True, return_dict_in_generate=True,
                               pad_token_id=tok.eos_token_id)
        txt = tok.decode(g.sequences[0, ids.shape[1]:], skip_special_tokens=True).lower()
        tl = []
        for sc in g.scores:
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            vals, idxs = lp.topk(TOP_K_LOGPROBS)
            tl.append({int(i): float(v) for i, v in zip(idxs, vals)})
        m = compute_token_margins(tl); e = compute_token_entropies(tl)
        b = compute_boundary_ratio(m, threshold=0.5)
        fm = m[np.isfinite(m)]
        pred = _first_of(txt, pos[0], pos[1])
        correct = 1 if (pred is not None and pred == gold) else 0
        X.append([float(np.mean(fm)) if len(fm) else 0.0, float(np.mean(e)), float(b)])
        y.append(correct)
    return np.array(X), np.array(y)


def ece(probs, labels, bins=10):
    """Expected Calibration Error."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean(); acc = labels[m].mean()
        e += (m.sum() / len(probs)) * abs(conf - acc)
    return float(e)


def reliability_table(probs, labels, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum() == 0:
            continue
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
                     "mean_pred": round(float(probs[m].mean()), 3),
                     "actual_acc": round(float(labels[m].mean()), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tasks", default="rte,qnli,sst2")
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASKS]

    print("=" * 70)
    print("CALIBRATE THE LEAN GATE  ->  true P(correct)")
    print("=" * 70)
    print(f"Model {args.model} | tasks={tasks} | n/task={args.n}")
    print("Fits calibrated P(correct) from margin/entropy/boundary; exports numpy params.")
    if not args.run:
        print("\n[DRY RUN] no download/generation/export. Use --run.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    rng = np.random.RandomState(SEED)
    import random; random.seed(SEED)

    Xs, ys, tids = [], [], []
    for t in tasks:
        items = load_task(t, args.n, rng)
        print(f"\n--- {t}: generating {len(items)} ---")
        X, y = generate(items, args.model)
        print(f"    accuracy={y.mean():.3f}")
        Xs.append(X); ys.append(y); tids.extend([t] * len(y))
    X = np.vstack(Xs); y = np.concatenate(ys)
    task_arr = np.array(tids)
    print(f"\nTotal {len(y)} examples, overall accuracy={y.mean():.3f}")

    # save raw rows so future analyses never need to regenerate
    rows_path = Path(__file__).parent / "calibrate_gate_rows.npz"
    np.savez(rows_path, X=X, y=y, task=task_arr, model=np.array([args.model]))
    print(f"Saved raw rows -> {rows_path}")

    def oof_probs(Xin):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        out = np.zeros(len(y))
        for tr, te in skf.split(Xin, y):
            sc = StandardScaler().fit(Xin[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xin[tr]), y[tr])
            out[te] = clf.predict_proba(sc.transform(Xin[te]))[:, 1]
        return out

    # out-of-fold raw probs from logistic on standardized features
    oof = oof_probs(X)

    # --- the decisive checks (review finding 1) -----------------------------
    # (a) WITHIN-task AUC of the pooled model: does pooling merely hide
    #     within-task discrimination behind between-task base-rate mixing?
    within = {}
    for t in tasks:
        m = task_arr == t
        if len(np.unique(y[m])) >= 2:
            within[t] = round(float(roc_auc_score(y[m], oof[m])), 4)

    # (b) pooled AUC after PER-TASK feature standardization — an UNSUPERVISED
    #     offset fix (needs no labels at deploy time, just a rolling z-score).
    #     If this recovers the per-task AUCs, the gate IS general up to offsets.
    Xz = X.copy().astype(np.float64)
    for t in tasks:
        m = task_arr == t
        mu, sd = X[m].mean(axis=0), X[m].std(axis=0) + 1e-9
        Xz[m] = (X[m] - mu) / sd
    oof_z = oof_probs(Xz)
    auc_pooled_znorm = round(float(roc_auc_score(y, oof_z)), 4)

    # isotonic recalibration on the OOF probs
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
    oof_cal = iso.predict(oof)

    metrics = {
        "auc_pooled": round(float(roc_auc_score(y, oof)), 4),
        "auc_within_task": within,
        "auc_pooled_after_pertask_znorm": auc_pooled_znorm,
        "brier_raw": round(float(brier_score_loss(y, oof)), 4),
        "brier_calibrated": round(float(brier_score_loss(y, oof_cal)), 4),
        "ece_raw": round(ece(oof, y), 4),
        "ece_calibrated": round(ece(oof_cal, y), 4),
        "base_rate_correct": round(float(y.mean()), 4),
        "n": int(len(y)),
    }
    print("\n--- calibration quality (out-of-fold) ---")
    for k, v in metrics.items():
        print(f"  {k:32s}: {v}")
    print("\n  READ: if auc_within_task >> auc_pooled, pooling was hiding real")
    print("        discrimination; if znorm recovers it, an unsupervised rolling")
    print("        z-score makes the gate general WITHOUT per-task labels.")
    print("\n  reliability (calibrated):")
    for r in reliability_table(oof_cal, y):
        print(f"    {r['bin']}  n={r['n']:4d}  pred={r['mean_pred']:.2f}  actual={r['actual_acc']:.2f}")

    # final model on ALL data for export
    scaler = StandardScaler().fit(X)
    logreg = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    raw_all = logreg.predict_proba(scaler.transform(X))[:, 1]
    iso_all = IsotonicRegression(out_of_bounds="clip").fit(raw_all, y)
    xs = np.linspace(0, 1, 101)
    export = {
        "features": ["mean_margin", "mean_entropy", "boundary_ratio"],
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "logreg_coef": [round(float(v), 6) for v in logreg.coef_[0]],
        "logreg_intercept": round(float(logreg.intercept_[0]), 6),
        "isotonic_x": [round(float(v), 4) for v in xs],
        "isotonic_y": [round(float(v), 4) for v in iso_all.predict(xs)],
        "trained_on": {"model": args.model, "tasks": tasks, "n": int(len(y))},
        "metrics_oof": metrics,
        "note": "P(correct). Apply: z=(f-mean)/scale; p=sigmoid(coef.z+b); interp(p, iso_x, iso_y)",
    }
    EXPORT.write_text(json.dumps(export, indent=2))
    REPORT.write_text(json.dumps({"metrics": metrics,
                                  "reliability": reliability_table(oof_cal, y)}, indent=2))
    print(f"\nExported calibration -> {EXPORT}")
    print(f"Report -> {REPORT}")
    print("\nREAD: ece_calibrated << ece_raw means the probability is now trustworthy.")


if __name__ == "__main__":
    main()
