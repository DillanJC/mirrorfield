# -*- coding: utf-8 -*-
"""
Plan C Step 1 — the gated agent: model -> answer -> gate -> decision.

LocalLM mirrors calibrate_gate.generate() exactly (Qwen2.5-3B, chat template,
greedy, MAX_NEW=24, top-5 logprobs). InProcessGate reuses the SHIPPED gate code
from mirrorfield.mcp.uncertainty (no copies) and mirrors server.py's 4-decimal
feature rounding, so it is numerically identical to the served path. McpGate
(for the demo) talks to the real MCP server over stdio.

Decision policy (pre-registered, gate_thresholds.json):
  ABSTAIN if p_correct_relative < tau_abstain
  PRESENT if p_correct_relative >= tau_present
  VERIFY  otherwise, and always during warmup.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mirrorfield.mcp.uncertainty import (  # noqa — the shipped code, reused
    RollingGate,
    calibrated_p_correct,
    compute_boundary_ratio,
    compute_token_entropies,
    compute_token_margins,
)

SEED = 42
MAX_NEW = 24
TOP_K = 5
THRESHOLDS = json.loads((Path(__file__).parent / "gate_thresholds.json").read_text())


class LocalLM:
    """Qwen2.5-3B-Instruct, greedy, top-5 logprobs — calibrate_gate's regime."""

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        torch.manual_seed(SEED)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def generate(self, user_text):
        """Returns (answer_text_lower, top_logprobs list of {id: lp})."""
        torch = self.torch
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
                pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out.sequences[0, ids.shape[1]:],
                               skip_special_tokens=True).lower()
        tl = []
        for sc in out.scores:
            lp = torch.log_softmax(sc[0].float(), dim=-1)
            vals, idxs = lp.topk(TOP_K)
            tl.append({int(i): float(v) for i, v in zip(idxs, vals)})
        return text, tl


def features_from_logprobs(top_logprobs):
    """The gate's three features, rounded to 4dp like server.py lines 301-303."""
    m = compute_token_margins(top_logprobs)
    e = compute_token_entropies(top_logprobs)
    b = compute_boundary_ratio(m)
    fm = m[np.isfinite(m)]
    mean_margin = round(float(np.mean(fm)), 4) if len(fm) else 0.0
    mean_entropy = round(float(np.mean(e)), 4)
    boundary = round(float(b), 4)
    return mean_margin, mean_entropy, boundary


class InProcessGate:
    """The shipped gate, in process. Numerically identical to the MCP path."""

    def __init__(self):
        self.rolling = RollingGate(window=50, min_history=5)

    def assess(self, top_logprobs, context_id):
        mm, me, br = features_from_logprobs(top_logprobs)
        p_cal = calibrated_p_correct(mm, me, br)
        rel = self.rolling.score(context_id, mm, me, br)
        return {"features": (mm, me, br),
                "p_correct_calibrated": p_cal,
                "p_correct_relative": rel["p_correct_relative"],
                "warming_up": rel["warming_up"],
                "buffer_size": rel["buffer_size"]}


class McpGate:
    """The real MCP server over stdio (used by the demo; parity-tested vs
    InProcessGate). Synchronous wrapper around the async client."""

    def __init__(self):
        self._session = None

    async def _assess_async(self, tokens, top_logprobs, context_id):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mirrorfield.mcp.server"],
            cwd=str(ROOT))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("confidence_report", {
                    "text": " ".join(tokens),
                    "logprobs": [max(d.values()) for d in top_logprobs],
                    "top_logprobs": [{str(k): v for k, v in d.items()}
                                     for d in top_logprobs],
                    "context_id": context_id})
                return json.loads(res.content[0].text)

    def assess(self, top_logprobs, context_id, tokens=None):
        import asyncio
        toks = tokens or ["t"] * len(top_logprobs)
        out = asyncio.run(self._assess_async(toks, top_logprobs, context_id))
        m = out.get("metrics", {})
        return {"p_correct_calibrated": m.get("p_correct_calibrated"),
                "p_correct_relative": m.get("p_correct_relative"),
                "warming_up": m.get("context_warming_up"),
                "buffer_size": m.get("context_buffer_size")}


def decide(p_rel, warming_up):
    if warming_up or p_rel is None:
        return "VERIFY"
    if p_rel < THRESHOLDS["tau_abstain"]:
        return "ABSTAIN"
    if p_rel >= THRESHOLDS["tau_present"]:
        return "PRESENT"
    return "VERIFY"


class GatedAgent:
    """model -> answer -> gate -> decision. The loop the project is for."""

    def __init__(self, lm, gate):
        self.lm = lm
        self.gate = gate

    def answer(self, question, context_id):
        text, tl = self.lm.generate(question)
        a = self.gate.assess(tl, context_id)
        return {"answer_text": text,
                "top_logprobs": tl,
                **a,
                "decision": decide(a["p_correct_relative"], a["warming_up"])}


if __name__ == "__main__":
    print("gate_agent module — used by eval_gate_value.py and run_gate_demo.py")
    print(f"thresholds: abstain<{THRESHOLDS['tau_abstain']} "
          f"present>={THRESHOLDS['tau_present']}")
