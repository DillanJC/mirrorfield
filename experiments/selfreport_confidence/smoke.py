# -*- coding: utf-8 -*-
"""
A1 FEASIBILITY SMOKE — plumbing only, NOT the experiment.

Confirms the two-stage elicitation for the verbalized-vs-internal confidence study
is executable: (1) answer pass yields capturable top-k log-probs -> gate features ->
a finite internal p_correct; (2) confidence pass yields a PARSEABLE 0-100 integer.
Uses 3 toy prompts, NO dataset, NO gold, NO verbal-vs-internal comparison — the actual
metric stays locked behind PROPOSED_PREREGISTRATIONS.md (A1) until pre-registered.
"""

import re, sys
from pathlib import Path

# resolve `import mirrorfield` to THIS repo (avoid the stale-copy shadow)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from mirrorfield.mcp.uncertainty import (  # noqa
    compute_token_margins, compute_token_entropies, compute_boundary_ratio,
    calibrated_p_correct,
)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
TOY = [
    "Premise: A man is playing a guitar on stage. Hypothesis: A person is performing music. "
    "Does the premise entail the hypothesis? Answer yes or no.",
    "Premise: The cat slept all day. Hypothesis: The dog barked loudly. "
    "Does the premise entail the hypothesis? Answer yes or no.",
    "Question: Is water wet? Answer yes or no.",
]


def load():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot_download(MODEL, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return torch, tok, model


def gen(torch, tok, model, text, max_new=20, want_logprobs=False):
    ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             output_scores=want_logprobs, return_dict_in_generate=True,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True).strip()
    tl = None
    if want_logprobs:
        tl = []
        for sc in out.scores:
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            vals, idx = lp.topk(5)
            tl.append({int(i): float(v) for i, v in zip(idx, vals)})
    return txt, tl


def main():
    torch, tok, model = load()
    parse_ok = 0
    for i, q in enumerate(TOY):
        ans, tl = gen(torch, tok, model, q, max_new=16, want_logprobs=True)
        mm = compute_token_margins(tl); me = compute_token_entropies(tl)
        import numpy as np
        fin = mm[np.isfinite(mm)]
        mmean = float(fin.mean()) if len(fin) else float("nan")
        p_int = calibrated_p_correct(round(mmean, 4), round(float(me.mean()), 4),
                                     round(compute_boundary_ratio(mm), 4))
        conf_txt, _ = gen(torch, tok, model,
                          f"{q}\n\nProposed answer: {ans}\n\nHow confident are you, 0-100%, "
                          f"that the proposed answer is correct? Reply with just a number.",
                          max_new=12)
        m = re.search(r"\b(\d{1,3})\b", conf_txt)
        ok = bool(m) and 0 <= int(m.group(1)) <= 100
        parse_ok += ok
        print(f"[{i}] answer={ans[:30]!r} | internal p_correct={p_int} (finite={p_int is not None}) "
              f"| conf_raw={conf_txt[:24]!r} parsed={m.group(1) if m else None} ok={ok}")
    print(f"\nPLUMBING: {parse_ok}/{len(TOY)} confidence parses OK; internal p_correct computed. "
          f"(toy prompts, no gold, no metric — A1 stays pre-registration-locked)")


if __name__ == "__main__":
    main()
