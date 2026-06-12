# -*- coding: utf-8 -*-
"""
Plan E Track A — gate features on human-labeled (un)safe pairs, forced decode.

For each BeaverTails (prompt, response) pair: ONE forward pass with the
response teacher-forced through Qwen2.5-3B; top-5 logprobs at every response
position; the gate's 3 features via the SHIPPED mirrorfield.mcp.uncertainty
functions (4dp rounding, matching the live path). Labels are BeaverTails'
human `is_safe` annotations — the gate never defines its target.

  --smoke  N=50 (25 safe / 25 unsafe), verify finite features + npz write
  --run    N=1,200 (600 safe + 600 unsafe stratified over 14 categories)

Seed 42 (pre-registered). Output: harm_gate_trackA_rows.npz (features, labels,
response token length, primary category, ds_index — never raw text).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from gate_agent import features_from_logprobs  # noqa  (shipped-code wrapper)

SEED = 42
MAX_RESP_TOKENS = 256
TOP_K = 5
MODEL = "Qwen/Qwen2.5-3B-Instruct"


def sample_indices(ds, n_per_class, rng):
    """600/600 (or smoke 25/25); unsafe stratified across primary category."""
    safe_idx = [i for i, s in enumerate(ds["is_safe"]) if s]
    unsafe_idx = [i for i, s in enumerate(ds["is_safe"]) if not s]
    rng.shuffle(safe_idx)
    chosen_safe = safe_idx[:n_per_class]

    # primary category = first true flag (dict field), for stratification
    cats = {}
    for i in unsafe_idx:
        c = ds[i]["category"]
        primary = next((k for k, v in c.items() if v), "uncategorized")
        cats.setdefault(primary, []).append(i)
    per_cat = max(1, n_per_class // max(1, len(cats)))
    chosen_unsafe = []
    for k in sorted(cats):
        lst = cats[k]
        rng.shuffle(lst)
        chosen_unsafe.extend(lst[:per_cat])
    # top up to n_per_class from the remainder
    if len(chosen_unsafe) < n_per_class:
        pool = [i for i in unsafe_idx if i not in set(chosen_unsafe)]
        rng.shuffle(pool)
        chosen_unsafe.extend(pool[:n_per_class - len(chosen_unsafe)])
    chosen_unsafe = chosen_unsafe[:n_per_class]
    return chosen_safe, chosen_unsafe


def forced_decode_features(model, tok, device, prompt, response):
    import torch
    pid = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt")[0]
    rid = tok(response, add_special_tokens=False,
              return_tensors="pt").input_ids[0][:MAX_RESP_TOKENS]
    if len(rid) < 2:
        return None, 0
    full = torch.cat([pid, rid]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(full).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    tl = []
    p0 = len(pid)
    for j in range(len(rid)):
        dist = lp[p0 + j - 1]
        vals, idxs = dist.topk(TOP_K)
        tl.append({int(i): float(v) for i, v in zip(idxs, vals)})
    return features_from_logprobs(tl), len(rid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--split", default="30k_test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.smoke or a.run):
        print("use --smoke (N=50) or --run (N=1200)")
        return
    n_per = 25 if a.smoke else 600
    out = Path(a.out) if a.out else HERE / (
        "harm_gate_trackA_smoke.npz" if a.smoke
        else "harm_gate_trackA_rows.npz")

    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split=a.split)
    print(f"split={a.split}  rows={len(ds)}")
    rng = np.random.RandomState(a.seed)
    safe_i, unsafe_i = sample_indices(ds, n_per, rng)
    print(f"sampled {len(safe_i)} safe + {len(unsafe_i)} unsafe "
          f"(seed {a.seed})")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    rows = []
    todo = [(i, 0) for i in safe_i] + [(i, 1) for i in unsafe_i]
    for k, (i, y_unsafe) in enumerate(todo):
        ex = ds[i]
        feats, rlen = forced_decode_features(model, tok, device,
                                             ex["prompt"], ex["response"])
        if feats is None:
            continue
        primary = next((c for c, v in ex["category"].items() if v), "safe")
        rows.append((i, y_unsafe, *feats, rlen, primary))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(todo)}")

    np.savez(out,
             ds_index=np.array([r[0] for r in rows]),
             y_unsafe=np.array([r[1] for r in rows]),
             X=np.array([[r[2], r[3], r[4]] for r in rows]),
             resp_len=np.array([r[5] for r in rows]),
             category=np.array([r[6] for r in rows]),
             seed=np.array([a.seed]))
    X = np.array([[r[2], r[3], r[4]] for r in rows])
    print(f"saved {len(rows)} rows -> {out}")
    print(f"finite features: {np.isfinite(X).all()}")
    for lab, m in [("safe", np.array([r[1] for r in rows]) == 0),
                   ("unsafe", np.array([r[1] for r in rows]) == 1)]:
        print(f"  {lab:6s} margin={X[m,0].mean():.3f} "
              f"entropy={X[m,1].mean():.3f} boundary={X[m,2].mean():.3f}")


if __name__ == "__main__":
    main()
