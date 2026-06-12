# -*- coding: utf-8 -*-
"""
Plan C Step 5 — the watchable demo + the parity proof.

  --parity   20 synthetic + 5 real generations scored by BOTH gate paths
             (in-process vs the real MCP server over ONE persistent stdio
             session). Asserts exact 4-decimal equality of
             p_correct_relative and identical decisions on all 25 items.
  --run      the demo: 3 contexts, each silently pre-warmed with 10 items,
             then 8 questions per context streamed live through the REAL
             MCP server; each shows the answer, the gate's probability and
             the decision (ABSTAIN withholds the answer; --reveal shows it).

Honesty banner: demo questions are curated for watchability; NO quantitative
claim comes from this script — all numbers live in eval_gate_value_results.json
(pre-registered criteria, commit b1fdc08).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from gate_agent import (  # noqa
    InProcessGate, LocalLM, THRESHOLDS, decide, features_from_logprobs)

TRANSCRIPT = HERE / "gate_demo_transcript.txt"
PARITY_OUT = HERE / "gate_agent_parity.json"


class PersistentMcpGate:
    """ONE stdio session for the whole stream — rolling state lives server-side."""

    def __init__(self):
        self._cm = None
        self._session_cm = None
        self.session = None

    async def __aenter__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "mirrorfield.mcp.server"],
            cwd=str(ROOT))
        self._cm = stdio_client(params)
        r, w = await self._cm.__aenter__()
        self._session_cm = ClientSession(r, w)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc):
        try:
            await self._session_cm.__aexit__(*exc)
        finally:
            await self._cm.__aexit__(*exc)

    async def assess(self, top_logprobs, context_id):
        res = await self.session.call_tool("confidence_report", {
            "text": " ".join("t" for _ in top_logprobs),
            "logprobs": [max(d.values()) for d in top_logprobs],
            "top_logprobs": [{str(k): v for k, v in d.items()}
                             for d in top_logprobs],
            "context_id": context_id})
        out = json.loads(res.content[0].text)
        m = out.get("metrics", {})
        return {"p_correct_relative": m.get("p_correct_relative"),
                "warming_up": bool(m.get("context_warming_up", False))}


def synthetic_logprobs(rng, n_tokens=12):
    """One synthetic generation: plausible top-5 logprob dicts per position."""
    tl = []
    for _ in range(n_tokens):
        top = -np.sort(rng.exponential(1.2, 5))[::-1] * rng.uniform(0.3, 2.0)
        top = np.sort(top)[::-1]
        tl.append({int(i): float(v) for i, v in enumerate(top)})
    return tl


async def parity():
    rng = np.random.RandomState(7)
    cases = [("synth", synthetic_logprobs(rng)) for _ in range(20)]
    print("generating 5 real answers for parity...")
    lm = LocalLM()
    real_qs = [
        "Premise: The cat sat on the mat.\nHypothesis: An animal was on the mat.\nDoes the premise entail the hypothesis? Answer yes or no first, then a brief reason.",
        "Premise: It rained all day in Paris.\nHypothesis: The weather in Paris was sunny.\nDoes the premise entail the hypothesis? Answer yes or no first, then a brief reason.",
        "Classify the sentiment of this sentence as positive or negative. Answer with one word first, then a brief reason.\n\nSentence: a triumph of imagination and craft",
        "Classify the sentiment of this sentence as positive or negative. Answer with one word first, then a brief reason.\n\nSentence: tedious, lifeless and utterly forgettable",
        "Question: What year did the war end?\nSentence: The treaty was signed in 1945, ending the conflict.\nDoes the sentence contain the answer to the question? Answer yes or no first, then a brief reason.",
    ]
    for q in real_qs:
        _, tl = lm.generate(q)
        cases.append(("real", tl))

    inproc = InProcessGate()
    rows = []
    async with PersistentMcpGate() as mcp_gate:
        for k, (kind, tl) in enumerate(cases):
            ctx = f"parity_{k % 3}"          # 3 interleaved contexts
            a = inproc.assess(tl, ctx)
            b = await mcp_gate.assess(tl, ctx)
            da = decide(a["p_correct_relative"], a["warming_up"])
            db = decide(b["p_correct_relative"], b["warming_up"])
            match = (a["p_correct_relative"] == b["p_correct_relative"]
                     and da == db)
            rows.append({"i": k, "kind": kind,
                         "inproc": a["p_correct_relative"],
                         "mcp": b["p_correct_relative"],
                         "decision_inproc": da, "decision_mcp": db,
                         "match": bool(match)})
    ok = all(r["match"] for r in rows)
    PARITY_OUT.write_text(json.dumps({"all_match": ok, "rows": rows}, indent=1))
    print(f"parity on {len(rows)} items (20 synthetic + 5 real, "
          f"3 interleaved contexts): {'EXACT MATCH - PASS' if ok else 'MISMATCH - FAIL'}")
    if not ok:
        for r in rows:
            if not r["match"]:
                print("  mismatch:", r)
    print(f"saved -> {PARITY_OUT}")
    return ok


def demo_items():
    """Curated questions from the eval pool (deterministic, seed 7)."""
    from eval_gate_value import select_fresh
    items = select_fresh()
    rng = np.random.RandomState(7)
    by_task = {}
    for it in items:
        by_task.setdefault(it["task"], []).append(it)
    warm, show = {}, {}
    for t, lst in by_task.items():
        order = rng.permutation(len(lst))
        warm[t] = [lst[i] for i in order[:10]]
        show[t] = [lst[i] for i in order[10:18]]
    return warm, show


async def run_demo(reveal=False):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("MIRRORFIELD GATE DEMO - answers checked BEFORE they are 'sent'")
    emit(f"thresholds (pre-registered, commit b1fdc08): "
         f"ABSTAIN < {THRESHOLDS['tau_abstain']}  "
         f"PRESENT >= {THRESHOLDS['tau_present']}")
    emit("NOTE: questions curated for watchability; all quantitative claims")
    emit("      live in eval_gate_value_results.json, not this demo.")
    emit("=" * 72)

    warm, show = demo_items()
    lm = LocalLM()
    stats = {"n": 0, "right": 0, "abstained": 0, "abstained_wrong": 0,
             "presented": 0, "presented_right": 0}
    from eval_gate_value import POS
    from calibrate_gate import _first_of

    async with PersistentMcpGate() as gate:
        for t in ["rte", "qnli", "sst2"]:
            emit(f"\n--- context '{t}': warming up with 10 unscored items ---")
            for it in warm[t]:
                text, tl = lm.generate(it["prompt"])
                await gate.assess(tl, t)
            emit(f"--- streaming 8 live questions in context '{t}' ---")
            for it in show[t]:
                text, tl = lm.generate(it["prompt"])
                a = await gate.assess(tl, t)
                d = decide(a["p_correct_relative"], a["warming_up"])
                pred = _first_of(text, *POS[t])
                correct = (pred is not None and pred == it["gold"])
                stats["n"] += 1
                stats["right"] += int(correct)
                q1 = it["prompt"].split("\n")[0][:70]
                emit(f"\nQ ({t}): {q1}...")
                if d == "ABSTAIN":
                    stats["abstained"] += 1
                    stats["abstained_wrong"] += int(not correct)
                    shown = (f"[withheld: low confidence"
                             f"{' | hidden answer: ' + text[:60] if reveal else ''}]")
                else:
                    stats["presented"] += 1
                    stats["presented_right"] += int(correct)
                    shown = text[:90]
                flag = "" if correct else "   (gold disagrees)"
                emit(f"A: {shown}")
                emit(f"   p_correct_relative={a['p_correct_relative']}  "
                     f"decision={d}{flag if d != 'ABSTAIN' or reveal else ''}")

    emit("\n" + "=" * 72)
    emit("SUMMARY (this curated stream only - not a measurement)")
    emit(f"  questions: {stats['n']}   model right: {stats['right']}")
    if stats["presented"]:
        emit(f"  presented: {stats['presented']} "
             f"(right: {stats['presented_right']} = "
             f"{stats['presented_right']/stats['presented']:.0%})")
    emit(f"  withheld:  {stats['abstained']} "
         f"(of which actually wrong: {stats['abstained_wrong']})")
    emit("  The measured result: gate abstains 14.7%, catches 27.7% of all")
    emit("  errors (random would catch 14.7%) - see WORK_MAP.md 4k.")
    emit("=" * 72)
    TRANSCRIPT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\ntranscript -> {TRANSCRIPT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reveal", action="store_true")
    a = ap.parse_args()
    if a.parity:
        asyncio.run(parity())
    elif a.run:
        asyncio.run(run_demo(reveal=a.reveal))
    else:
        print("use --parity (proof both gate paths agree) or --run (the demo)")
