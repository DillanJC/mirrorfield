"""Quickstart — gate a model's answers with the composed SEND / VERIFY / HOLD pipeline.

Pure-package demo: numpy/scipy only, NO model download. In real use you pass the
top-k log-probs your LLM already returns (OpenAI ``top_logprobs``, HF
``output_scores``, etc.); here we synthesize a few to show the mechanics.

    python examples/quickstart.py

Honest scope: the wrongness gate is a MODEST ranking signal (live AUC ~0.69), not an
oracle. Harm is delegated to a separate classifier (optional ``mirrorfield[harm]``);
here we just pass in a harm score to show that harm overrides to HOLD.
"""

import random
import sys
from pathlib import Path

# Run straight from a checkout without installing: put the repo root first on the
# path so `import mirrorfield` resolves to THIS repo (also avoids any stale copy
# installed elsewhere). If you `pip install` mirrorfield, this line is harmless.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from mirrorfield.mcp.uncertainty import (
    RollingGate,
    compute_boundary_ratio,
    compute_token_entropies,
    compute_token_margins,
    send_hold_decision,
)


def features(top_logprobs):
    """Reduce per-token top-k log-probs to the gate's 3 features."""
    m = compute_token_margins(top_logprobs)
    e = compute_token_entropies(top_logprobs)
    finite = m[np.isfinite(m)]
    return (float(finite.mean()) if len(finite) else 0.0,
            float(e.mean()),
            compute_boundary_ratio(m))


# A response is a list of {token: logprob} dicts (one per generated token).
# Big top1-vs-top2 margin + low entropy = the model was confident.
CONFIDENT = [{"yes": -0.05, "no": -3.2, "maybe": -4.1} for _ in range(8)]
UNSURE = [{"yes": -0.7, "no": -0.8, "maybe": -1.1} for _ in range(8)]

gate = RollingGate()           # ONE buffer per kind-of-work context (never share across tasks)
CTX = "qa-task"

# Warm up the context: the gate z-scores each answer against recent SAME-context
# traffic, so it needs a little history (first ~5 calls report warming_up=True).
random.seed(0)
for _ in range(8):
    gate.score(CTX, *features(CONFIDENT if random.random() < 0.5 else UNSURE))


def gate_one(label, top_logprobs, harm_score=None):
    r = gate.score(CTX, *features(top_logprobs))
    d = send_hold_decision(r["p_correct_relative"], r["warming_up"], harm_score=harm_score)
    print(f"  {label:24} p_rel={r['p_correct_relative']}  ->  {d['decision']:6} ({d['reason']})")


if __name__ == "__main__":
    print("composed gate decisions (after warmup):")
    gate_one("confident answer", CONFIDENT)
    gate_one("unsure answer", UNSURE)
    gate_one("confident but harmful", CONFIDENT, harm_score=0.95)  # harm overrides -> HOLD
    print("\n  SEND = present it | VERIFY = present but flag/double-check | HOLD = don't send")
    print("  (modest gate, AUC ~0.69 — it improves the odds, it is not a guarantee)")
