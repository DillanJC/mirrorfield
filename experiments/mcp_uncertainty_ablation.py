# -*- coding: utf-8 -*-
"""
MCP Uncertainty Ablation — the one NON-circular test of the geometry idea.

Question
--------
Does the geometric signal (sequence participation-ratio / embedding PR), as
computed by the live MCP uncertainty server, add ANY predictive value over the
standard log-prob uncertainty signals (token margin + entropy + boundary ratio)
for predicting whether a model's output is actually WRONG?

Why this is not circular (unlike R0-R3 and the Mar-21 training result)
---------------------------------------------------------------------
The target label here is GROUND-TRUTH correctness (model's predicted SST-2
sentiment vs the gold label). It is defined entirely OUTSIDE the geometry — the
geometry never gets to define what counts as a "bad" output. So if PR adds AUC
lift over the standard signals, that is a real, earned result; if it doesn't,
the geometry is decorative and the honest move is to ship the standard signals.

Design
------
1. Load SST-2 validation (cached glue/sst2). Balanced subset of N samples.
2. A small instruct LM classifies each sentence (free-form short generation),
   capturing per-token top-k logprobs AND last-layer hidden states.
3. From logprobs  -> STANDARD features (margin, entropy, boundary ratio)
                     via the REAL MCP functions.
   From hidden states -> GEOMETRY features (sequence PR, embedding PR)
                     via the REAL MCP functions.
4. Label each generation correct(0)/wrong(1) by parsing the answer vs gold.
5. Logistic regression, grouped 5-fold CV:
       AUC(STANDARD)  vs  AUC(STANDARD + GEOMETRY)  vs  AUC(GEOMETRY alone)
   Bootstrap CI on the delta. Negative control: shuffle the wrong-labels ->
   delta must collapse to ~0.

Run
---
    python mcp_uncertainty_ablation.py            # dry-run: prints plan, no DL
    python mcp_uncertainty_ablation.py --run      # downloads model + executes

Nothing is downloaded, embedded, or written unless --run is passed.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

# --- import the REAL MCP uncertainty functions (faithful test of shipped code)
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
# Config
# ----------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 300                       # balanced SST-2 validation subset
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ~1GB, runs on the available CUDA GPU
MAX_NEW_TOKENS = 24                   # short rationale -> multi-token sequence
TOP_K_LOGPROBS = 5                    # alternatives kept per position
N_BOOT = 2000                         # bootstrap resamples for the delta CI
OUT_JSON = Path(__file__).parent / "mcp_uncertainty_ablation_results.json"

PROMPT = (
    "Classify the sentiment of this movie-review sentence as positive or "
    "negative. Answer with one word first, then a brief reason.\n\nSentence: "
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_sst2(n, rng):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    pos = [i for i, l in enumerate(ds["label"]) if l == 1]
    neg = [i for i, l in enumerate(ds["label"]) if l == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    half = n // 2
    idx = pos[:half] + neg[:half]
    rng.shuffle(idx)
    return [ds["sentence"][i] for i in idx], [ds["label"][i] for i in idx]


# ----------------------------------------------------------------------------
# Generation + per-sample feature extraction
# ----------------------------------------------------------------------------
def build_features(sentences, gold, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        output_hidden_states=True,
    ).to(device).eval()

    rows = []
    for s, g in zip(sentences, gold):
        msgs = [{"role": "user", "content": PROMPT + s}]
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

        # ---- top-k logprobs per generated position -> MCP standard signals
        top_logprobs = []
        for step_scores in out.scores:                      # one per new token
            lp = torch.log_softmax(step_scores[0].float(), dim=-1)
            vals, idxs = lp.topk(TOP_K_LOGPROBS)
            top_logprobs.append({int(i): float(v) for i, v in zip(idxs, vals)})

        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary = compute_boundary_ratio(margins, threshold=0.5)
        seq_pr = compute_sequence_pr(top_logprobs)

        # ---- last-layer hidden state per generated token -> MCP embedding PR
        hs = [step[-1][0, -1].float().cpu().numpy() for step in out.hidden_states]
        H = np.vstack(hs) if len(hs) > 1 else np.repeat(hs[0][None], 2, 0)
        emb = compute_embedding_pr(H, k=min(10, len(H) - 1))

        # ---- correctness label (parse answer vs gold); pred class for the gold
        pred = 1 if ("positive" in text and "negative" not in text.split("positive")[0]) \
            else (0 if "negative" in text else -1)
        wrong = int(pred != g) if pred in (0, 1) else 1   # unparseable == wrong

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


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def cv_auc(X, y, rng):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("MCP UNCERTAINTY ABLATION  —  geometry vs standard signals")
    print("=" * 70)
    print(f"MCP functions importable : {MCP_OK}" + ("" if MCP_OK else f"  ({_MCP_ERR})"))
    print(f"Model                    : {MODEL}")
    print(f"Samples (balanced SST-2) : {N_SAMPLES}")
    print(f"Target label             : ground-truth correctness (NON-circular)")
    print(f"STANDARD features        : mean margin, mean entropy, boundary ratio")
    print(f"GEOMETRY features        : sequence PR, emb pr_mean/pr_min/se_mean")
    print(f"Compare                  : AUC(std) vs AUC(std+geo) vs AUC(geo)")
    print(f"Controls                 : bootstrap CI on dAUC + shuffled-label null")

    if not args.run:
        print("\n[DRY RUN] No download, no generation, no write.")
        print("          Re-run with --run to download the model and execute.")
        return
    if not MCP_OK:
        print("\nABORT: could not import MCP functions; fix path first.")
        return

    rng = np.random.RandomState(SEED)
    import random; random.seed(SEED)
    sents, gold = load_sst2(N_SAMPLES, rng)
    print(f"\nLoaded {len(sents)} SST-2 sentences. Generating + extracting...")
    rows = build_features(sents, gold, MODEL)

    y = np.array([r["wrong"] for r in rows])
    Xs = np.array([r["std"] for r in rows])
    Xg = np.array([r["geo"] for r in rows])
    Xsg = np.hstack([Xs, Xg])
    print(f"Wrong-rate: {y.mean():.3f}  (n_wrong={int(y.sum())}/{len(y)})")

    auc_s, oof_s = cv_auc(Xs, y, rng)
    auc_g, _ = cv_auc(Xg, y, rng)
    auc_sg, oof_sg = cv_auc(Xsg, y, rng)
    delta = auc_sg - auc_s

    # bootstrap CI on the delta (std+geo minus std), paired on oof preds
    from sklearn.metrics import roc_auc_score
    deltas = []
    n = len(y)
    for _ in range(N_BOOT):
        bi = rng.randint(0, n, n)
        if len(np.unique(y[bi])) < 2:
            continue
        deltas.append(roc_auc_score(y[bi], oof_sg[bi]) - roc_auc_score(y[bi], oof_s[bi]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # shuffled-label null
    y_sh = y.copy(); rng.shuffle(y_sh)
    auc_s_n, _ = cv_auc(Xs, y_sh, rng)
    auc_sg_n, _ = cv_auc(Xsg, y_sh, rng)

    res = {
        "n": int(n), "wrong_rate": float(y.mean()),
        "auc_standard": float(auc_s), "auc_geometry_only": float(auc_g),
        "auc_standard_plus_geometry": float(auc_sg),
        "delta_auc": float(delta), "delta_ci95": [float(lo), float(hi)],
        "null_delta_auc": float(auc_sg_n - auc_s_n),
        "verdict": ("GEOMETRY ADDS REAL LIFT" if lo > 0 else
                    "geometry adds nothing beyond noise"),
    }
    print("\n" + "-" * 70)
    for k, v in res.items():
        print(f"{k:28s}: {v}")
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT_JSON}")
    print("\nREAD: if delta_ci95 lower bound > 0, geometry earns its place.")
    print("      if the CI straddles 0, ship the standard signals alone.")


if __name__ == "__main__":
    main()
