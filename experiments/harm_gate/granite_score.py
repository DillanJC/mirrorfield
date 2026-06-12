# -*- coding: utf-8 -*-
"""
Plan E — Granite-Guardian-3.1-2B harm scores for BeaverTails pairs.

Generative judge: the guardian emits Yes/No (risk present?); we use
P(Yes) / (P(Yes)+P(No)) from the first generated token's distribution as a
continuous harm score. Primary path: tokenizer.apply_chat_template with
guardian_config (documented API); fallback (time-boxed per plan): plain
prompt format.

  --sanity   10 known-safe + 10 known-unsafe pairs; assert unsafe >> safe
  --run      score the exact rows in harm_gate_trackA_rows.npz

Never co-resident with Qwen in VRAM (this script loads only Granite).
Output: harm_gate_trackA_granite.npz (ds_index, granite_score).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL = "ibm-granite/granite-guardian-3.1-2b"
SEED = 42


class Granite:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(SEED)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        # candidate ids for the Yes/No verdict tokens (cover space variants)
        self.yes_ids = self._ids(["Yes", " Yes", "yes", " yes"])
        self.no_ids = self._ids(["No", " No", "no", " no"])

    def _ids(self, variants):
        out = set()
        for v in variants:
            ids = self.tok(v, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
        return sorted(out)

    def _chat_ids(self, prompt, response):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        try:
            return self.tok.apply_chat_template(
                msgs, guardian_config={"risk_name": "harm"},
                add_generation_prompt=True, return_tensors="pt")
        except TypeError:
            # fallback: plain template without guardian_config kwarg
            return self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")

    def score(self, prompt, response):
        torch = self.torch
        ids = self._chat_ids(prompt, response)[:, -2048:].to(self.device)
        with torch.no_grad():
            logits = self.model(ids).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        py = float(sum(p[i] for i in self.yes_ids))
        pn = float(sum(p[i] for i in self.no_ids))
        return py / (py + pn + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--split", default="30k_test")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.sanity or a.run):
        print("use --sanity or --run")
        return

    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split=a.split)
    g = Granite()

    if a.sanity:
        rng = np.random.RandomState(SEED)
        safe = [i for i, s in enumerate(ds["is_safe"]) if s]
        unsafe = [i for i, s in enumerate(ds["is_safe"]) if not s]
        rng.shuffle(safe)
        rng.shuffle(unsafe)
        s_scores = [g.score(ds[i]["prompt"], ds[i]["response"])
                    for i in safe[:10]]
        u_scores = [g.score(ds[i]["prompt"], ds[i]["response"])
                    for i in unsafe[:10]]
        print(f"safe   mean={np.mean(s_scores):.4f}  "
              f"scores={[round(x,3) for x in s_scores]}")
        print(f"unsafe mean={np.mean(u_scores):.4f}  "
              f"scores={[round(x,3) for x in u_scores]}")
        ok = np.mean(u_scores) > np.mean(s_scores) + 0.2
        print("SANITY:", "PASS (unsafe >> safe)" if ok else
              "WEAK - inspect template/parsing before the full run")
        return

    rows_file = Path(a.rows) if a.rows else HERE / "harm_gate_trackA_rows.npz"
    out_file = Path(a.out) if a.out else HERE / "harm_gate_trackA_granite.npz"
    d = np.load(rows_file, allow_pickle=True)
    idxs = d["ds_index"]
    scores = []
    for k, i in enumerate(idxs):
        scores.append(g.score(ds[int(i)]["prompt"], ds[int(i)]["response"]))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(idxs)}")
    np.savez(out_file, ds_index=idxs, granite_score=np.array(scores))
    print(f"saved {len(scores)} scores -> {out_file}")


if __name__ == "__main__":
    main()
