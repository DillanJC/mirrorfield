"""
Mirrorfield — an honest, modest uncertainty gate for LLM outputs.

The validated tool: from token log-probabilities, estimate how likely an answer is
wrong (calibrated ``p_correct`` + a per-context ``RollingGate``) and decide BEFORE
sending whether to PRESENT / VERIFY / ABSTAIN — composed with an optional harm
classifier into a SEND / VERIFY / HOLD pipeline (``send_hold_decision``).

Honest scope: a MODEST ranking signal (live AUC 0.685; catches ~28% of errors at
~15% abstention on RTE/QNLI with Qwen2.5-3B), not an oracle. The embedding-geometry
framework this project began as was FALSIFIED and removed (see ``WORK_MAP.md``);
only the log-prob gate survived scrutiny.

Quick start (Python)::

    from mirrorfield import RollingGate, send_hold_decision
    gate = RollingGate()                 # one buffer per kind-of-work context
    r = gate.score("my-task", mean_margin, mean_entropy, boundary_ratio)
    send_hold_decision(r["p_correct_relative"], r["warming_up"], harm_score=None)

MCP server::  python -m mirrorfield.mcp.server   (or the ``mirrorfield-mcp`` script)
"""

__version__ = "3.0.0"
__author__ = "Mirrorfield Project"

from .mcp.uncertainty import (  # noqa: F401  (light: numpy/scipy only, no torch/geometry)
    RollingGate,
    calibrated_p_correct,
    decide,
    load_gate_thresholds,
    send_hold_decision,
)

__all__ = [
    "RollingGate",
    "calibrated_p_correct",
    "decide",
    "load_gate_thresholds",
    "send_hold_decision",
]
