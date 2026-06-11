"""
Mirrorfield MCP Server — Uncertainty awareness for AI agents.

LEAN build (2026-06-07). The confidence gate runs purely on validated log-prob
signals (token margin + entropy + boundary ratio). Embedding-geometry tools and
PR fields were REMOVED after a non-circular ablation showed geometry adds no
predictive lift over the standard signals (dAUC +0.001, CI [-0.05,+0.05]); the
standard signals alone reach AUC 0.667 at predicting wrong outputs.

Tools:
  1. analyze_logprobs    – token-level uncertainty from log-probabilities
  2. confidence_report   – high-level confidence assessment + recommendation
  3. compare_responses   – compare uncertainty across candidate responses
  4. novelty_map         – interpretive "epistemic terrain" view (presentation
                           layer over the SAME margin/entropy signals; the
                           terrain categories are interpretive, not calibrated)

Prompts:
  - assess-my-response  – guided workflow for single-response assessment
  - compare-drafts      – guided workflow for multi-response comparison
  - explore-uncertainty  – guided workflow for novelty-oriented thinking

Resources:
  - mirrorfield://calibration – current scoring weights and confidence thresholds
"""

import json
import logging
import sys
import time

from mcp.server.fastmcp import FastMCP

from .uncertainty import (
    RollingGate,
    build_novelty_map,
    calibrated_p_correct,
    classify_confidence,
    compute_boundary_ratio,
    compute_confidence_score,
    compute_self_consistency,
    compute_token_entropies,
    compute_token_margins,
    find_uncertain_spans,
    generate_explanation,
)

import numpy as np

# ── Structured logging to stderr ──────────────────────────────────────────

_log = logging.getLogger("mirrorfield.mcp")
_log.setLevel(logging.DEBUG if "--debug" in sys.argv else logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    json.dumps({
        "ts": "%(asctime)s",
        "level": "%(levelname)s",
        "logger": "%(name)s",
        "msg": "%(message)s",
    })
))
_log.addHandler(_handler)


def _log_tool(name: str, **kwargs):
    """Log a tool invocation with timing context."""
    _log.info(f"tool={name} {' '.join(f'{k}={v}' for k, v in kwargs.items())}")


mcp = FastMCP("mirrorfield", instructions="Uncertainty awareness for AI agents")

# Per-context rolling gate (validated deployable form, WORK_MAP 4h/4i).
# One buffer per context_id; mixing contexts in one buffer breaks the signal.
_ROLLING_GATE = RollingGate(window=50, min_history=5)


# ── Calibration constants (exposed as resource) ──────────────────────────

CALIBRATION = {
    "scoring_weights": {
        "mean_margin": 0.4,
        "mean_entropy": 0.35,
        "boundary_ratio": 0.25,
    },
    "confidence_thresholds": {
        "high": 0.8,
        "moderate": 0.6,
        "low": 0.3,
        "very_low": 0.0,
    },
    "recommendation_thresholds": {
        "proceed": 0.7,
        "verify": 0.4,
        "abstain": 0.0,
    },
    "boundary_margin_threshold": 0.5,
    "span_severity_levels": {
        "critical": {"max_margin": 0.1, "color": "#e53e3e"},
        "high": {"max_margin": 0.25, "color": "#dd6b20"},
        "moderate": {"max_margin": 0.5, "color": "#d69e2e"},
    },
}


# ── Resource ──────────────────────────────────────────────────────────────

@mcp.resource("mirrorfield://calibration")
def get_calibration() -> str:
    """Current scoring weights, confidence thresholds, and severity levels.

    Read this to understand how confidence scores are computed and what
    the threshold boundaries mean. These values are used by all tools.
    """
    return json.dumps(CALIBRATION, indent=2)


# ── Prompts ───────────────────────────────────────────────────────────────

@mcp.prompt()
def assess_my_response(response_text: str) -> str:
    """Guided workflow for assessing uncertainty in a single response.

    Use this when you have generated a response and want to know how
    confident you should be before presenting it.
    """
    return (
        f"I need to assess my confidence in the following response:\n\n"
        f"---\n{response_text}\n---\n\n"
        f"Steps:\n"
        f"1. Call the `confidence_report` tool with the text above and any "
        f"log-prob data you have available.\n"
        f"2. Review the `recommendation` field:\n"
        f"   - \"proceed\": Present the response normally.\n"
        f"   - \"verify\": Present it but flag uncertain spans for the human.\n"
        f"   - \"abstain\": Tell the human you are not confident enough.\n"
        f"3. If there are `uncertain_spans`, check their `display.severity` "
        f"to decide which parts need the most attention.\n"
        f"4. Use the `explanation` field to communicate your uncertainty "
        f"to the human in natural language."
    )


@mcp.prompt()
def compare_drafts(draft_texts: str) -> str:
    """Guided workflow for comparing multiple draft responses.

    Use this when you have generated several candidate responses and
    want to pick the best one or understand where they disagree.
    Pass draft texts separated by |||.
    """
    drafts = [d.strip() for d in draft_texts.split("|||") if d.strip()]
    numbered = "\n".join(f"Draft {i}: {d}" for i, d in enumerate(drafts))
    return (
        f"I have {len(drafts)} draft responses to compare:\n\n"
        f"{numbered}\n\n"
        f"Steps:\n"
        f"1. Call `compare_responses` with each draft as a separate entry.\n"
        f"   Include any log-prob or embedding data you have.\n"
        f"2. Check `self_consistency` in the result — if `agreement` is low\n"
        f"   but confidence is high, the drafts contradict each other.\n"
        f"   This is a strong signal to verify with the human.\n"
        f"3. Check `best_index` to see which draft scored highest.\n"
        f"4. If `score_spread` is large, explain why one draft is better.\n"
        f"5. If all drafts agree and are confident, proceed with the best one."
    )


@mcp.prompt()
def explore_uncertainty(response_text: str) -> str:
    """Guided workflow for novelty-oriented thinking.

    Use this when you want to understand *where* you are extrapolating
    and treat uncertainty as a map of interesting territory rather than
    a warning to retreat.
    """
    return (
        f"I want to map the epistemic terrain of my response:\n\n"
        f"---\n{response_text}\n---\n\n"
        f"Steps:\n"
        f"1. Call `novelty_map` with the text above and any log-prob data.\n"
        f"2. Check the `terrain` field:\n"
        f"   - \"well_trodden\": I'm on solid ground. Nothing novel here.\n"
        f"   - \"frontier\": Mix of known and unknown. The interesting zone.\n"
        f"   - \"uncharted\": I'm extrapolating significantly.\n"
        f"   - \"deep_unknown\": I have very little basis for these claims.\n"
        f"3. Look at each span's `signature`:\n"
        f"   - \"decision_boundary\": Two frameworks compete here. "
        f"Explore both sides.\n"
        f"   - \"terra_incognita\": I'm genuinely guessing. Say so honestly.\n"
        f"   - \"framework_collision\": Multiple coherent but incompatible "
        f"views. Surface the disagreement — don't hide it.\n"
        f"4. Use each span's `action` field to decide how to communicate:\n"
        f"   present known ground confidently, flag frontiers as open "
        f"questions, and be transparent about extrapolation.\n"
        f"5. The `exploration_gradient` (0-1) tells you overall how far "
        f"from training distribution you are."
    )


# ── Tool 1 ────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_logprobs(
    tokens: list[str],
    logprobs: list[float],
    top_logprobs: list[dict] | None = None,
) -> dict:
    """Analyse token-level log-probabilities for uncertainty signals.

    Returns per-token margins/entropies, sequence-level statistics,
    boundary token count, participation ratio, and a confidence label.
    """
    _log_tool("analyze_logprobs", n_tokens=len(tokens),
              has_top_logprobs=top_logprobs is not None)
    t0 = time.monotonic()

    result: dict = {}

    # Per-token breakdown
    per_token: list[dict] = []
    margins = None
    entropies = None
    boundary_ratio = None

    if top_logprobs is not None and len(top_logprobs) == len(tokens):
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)

        finite_margins = margins[np.isfinite(margins)]

        for i, tok in enumerate(tokens):
            entry = {"token": tok, "logprob": logprobs[i]}
            if np.isfinite(margins[i]):
                entry["margin"] = round(float(margins[i]), 4)
            entry["entropy"] = round(float(entropies[i]), 4)
            per_token.append(entry)

        result["mean_margin"] = round(float(np.mean(finite_margins)), 4) if len(finite_margins) else None
        result["min_margin"] = round(float(np.min(finite_margins)), 4) if len(finite_margins) else None
        result["mean_entropy"] = round(float(np.mean(entropies)), 4)
        result["boundary_token_count"] = int(np.sum(finite_margins < 0.5)) if len(finite_margins) else 0
        result["boundary_ratio"] = round(boundary_ratio, 4)
    else:
        for i, tok in enumerate(tokens):
            per_token.append({"token": tok, "logprob": logprobs[i]})

    result["per_token"] = per_token

    # Confidence
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    result["confidence_score"] = score
    result["confidence_label"] = label

    _log.debug(f"analyze_logprobs done in {time.monotonic()-t0:.3f}s score={score}")
    return result


# ── Tool 2 ────────────────────────────────────────────────────────────────

@mcp.tool()
def confidence_report(
    text: str,
    logprobs: list[float] | None = None,
    top_logprobs: list[dict] | None = None,
    num_alternatives: int | None = None,
    context_id: str | None = None,
) -> dict:
    """High-level confidence assessment from available log-prob signals.

    Accepts whatever data is available and degrades gracefully.
    Returns a confidence score, label, uncertain spans, explanation,
    raw metrics, and a recommendation (proceed / verify / abstain).

    context_id (recommended): a stable identifier for the current stream of
    same-kind work (e.g. a session or task-type id). When provided, the report
    also includes p_correct_relative — the response's confidence z-scored
    against the last ~50 outputs in the SAME context, then calibrated. This is
    the validated cross-task form of the gate (it transfers to unseen task
    types). Use one context_id per kind of work; mixing different kinds of work
    under one id degrades the relative signal to chance. The first few calls in
    a context report warming_up=true with no relative probability.
    """
    _log_tool("confidence_report", text_len=len(text),
              has_logprobs=logprobs is not None,
              has_top_logprobs=top_logprobs is not None)

    metrics: dict = {}
    margins = None
    entropies = None
    boundary_ratio = None
    tokens = text.split()  # rough tokenization for span detection

    if top_logprobs is not None:
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)
        finite = margins[np.isfinite(margins)]
        metrics["mean_margin"] = round(float(np.mean(finite)), 4) if len(finite) else None
        metrics["mean_entropy"] = round(float(np.mean(entropies)), 4)
        metrics["boundary_ratio"] = round(boundary_ratio, 4)

        # Calibrated P(correct) — only when a calibration file is installed.
        # Modest discriminator (per-task AUC 0.60-0.74); see calibrated_p_correct.
        if metrics["mean_margin"] is not None:
            p_correct = calibrated_p_correct(
                metrics["mean_margin"], metrics["mean_entropy"],
                metrics["boundary_ratio"],
            )
            if p_correct is not None:
                metrics["p_correct_calibrated"] = p_correct

            # Context-relative P(correct) — the validated cross-task form.
            # z-scores this response against recent SAME-context traffic.
            if context_id is not None:
                rel = _ROLLING_GATE.score(
                    context_id, metrics["mean_margin"],
                    metrics["mean_entropy"], metrics["boundary_ratio"],
                )
                metrics["p_correct_relative"] = rel["p_correct_relative"]
                metrics["context_buffer_size"] = rel["buffer_size"]
                metrics["context_warming_up"] = rel["warming_up"]

    if num_alternatives is not None:
        metrics["num_alternatives"] = num_alternatives

    # Score
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    metrics["confidence_score"] = score
    metrics["confidence_label"] = label

    # Uncertain spans
    uncertain_spans: list[dict] = []
    if margins is not None and len(tokens) == len(margins):
        uncertain_spans = find_uncertain_spans(tokens, margins)
    metrics["uncertain_span_count"] = len(uncertain_spans)

    # Recommendation
    if score >= 0.7:
        recommendation = "proceed"
    elif score >= 0.4:
        recommendation = "verify"
    else:
        recommendation = "abstain"

    explanation = generate_explanation(metrics)

    return {
        "confidence_score": score,
        "confidence_label": label,
        "uncertain_spans": uncertain_spans,
        "explanation": explanation,
        "metrics": metrics,
        "recommendation": recommendation,
    }


# ── Tool 3 ────────────────────────────────────────────────────────────────

@mcp.tool()
def compare_responses(responses: list[dict]) -> dict:
    """Compare uncertainty across multiple candidate responses.

    Each entry should have at least ``text``; optionally ``logprobs`` and
    ``top_logprobs``.

    Includes self-consistency analysis: if responses are confident but
    disagree with each other, that is a strong uncertainty signal.
    """
    _log_tool("compare_responses", n_responses=len(responses))

    per_response: list[dict] = []

    for i, resp in enumerate(responses):
        entry: dict = {"index": i, "text_length": len(resp.get("text", ""))}
        margins = None
        entropies = None
        br = None

        if resp.get("top_logprobs"):
            margins = compute_token_margins(resp["top_logprobs"])
            entropies = compute_token_entropies(resp["top_logprobs"])
            br = compute_boundary_ratio(margins)

        score = compute_confidence_score(margins, entropies, br)
        entry["confidence_score"] = score
        entry["confidence_label"] = classify_confidence(score)
        per_response.append(entry)

    # Self-consistency (text agreement)
    texts = [r.get("text", "") for r in responses if r.get("text")]
    consistency = compute_self_consistency(texts)

    scores = [e["confidence_score"] for e in per_response]
    best_idx = int(np.argmax(scores))
    spread = round(float(max(scores) - min(scores)), 4) if scores else 0.0

    # Build recommendation considering both confidence and agreement
    agreement = consistency.get("agreement", 1.0)
    if agreement < 0.3 and max(scores) > 0.6:
        rec = (
            f"Warning: responses are confident but disagree with each other "
            f"(agreement={agreement:.2f}). Verify before proceeding."
        )
    elif spread < 0.15:
        rec = (
            f"Response {best_idx} has the highest confidence "
            f"({scores[best_idx]:.2f}). Responses largely agree."
        )
    else:
        rec = (
            f"Response {best_idx} has the highest confidence "
            f"({scores[best_idx]:.2f}). "
            f"Significant confidence spread — consider verifying."
        )

    return {
        "per_response": per_response,
        "best_index": best_idx,
        "score_spread": spread,
        "self_consistency": consistency,
        "recommendation": rec,
    }


# ── Tool 4 ────────────────────────────────────────────────────────────────

@mcp.tool()
def novelty_map(
    text: str,
    tokens: list[str] | None = None,
    top_logprobs: list[dict] | None = None,
    consistency_texts: list[str] | None = None,
) -> dict:
    """Map the epistemic terrain of a response — where are you on solid
    ground, where are you at a frontier, and where are you extrapolating?

    INTERPRETIVE layer: the terrain labels and signatures are a human-readable
    reframing computed ENTIRELY from the validated margin/entropy/self-consistency
    signals (no geometry). The category boundaries (well_trodden / frontier /
    uncharted) are heuristic thresholds, NOT a calibrated metric — treat the
    output as a qualitative guide, not a measured probability.

    Instead of treating uncertainty as a warning, this tool reinterprets
    it as a map of interesting territory. Use it when you want to be
    honest about what you know vs. what you're guessing.

    Args:
        text: The response text to analyze.
        tokens: Token list (if available). Falls back to whitespace split.
        top_logprobs: Top-k log-prob dicts per position (optional but
            recommended — without these, classification is coarse).
        consistency_texts: Multiple sampled responses to the same prompt
            (optional). If provided, enables framework collision detection
            via self-consistency analysis.

    Returns:
        Dict with exploration_gradient (0=known, 1=uncharted), terrain
        label, novelty-classified spans with descriptions and recommended
        actions, and a natural-language interpretation.
    """
    _log_tool("novelty_map", text_len=len(text),
              has_top_logprobs=top_logprobs is not None,
              has_consistency=consistency_texts is not None)

    tok = tokens if tokens is not None else text.split()

    # Compute signals from whatever data is available
    if top_logprobs is not None and len(top_logprobs) == len(tok):
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
    else:
        # No log-probs — use neutral values. The map will be coarse
        # but still useful if consistency_texts are provided.
        margins = np.full(len(tok), 1.0)   # neutral margin
        entropies = np.zeros(len(tok))      # neutral entropy

    # Self-consistency for framework collision detection
    consistency = None
    consistency_detail = None
    if consistency_texts is not None and len(consistency_texts) >= 2:
        consistency_detail = compute_self_consistency(consistency_texts)
        consistency = consistency_detail.get("agreement")

    result = build_novelty_map(tok, margins, entropies, consistency)

    if consistency_detail is not None:
        result["self_consistency"] = consistency_detail

    return result


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    _log.info(f"starting mirrorfield MCP server transport={transport}")
    mcp.run(transport=transport)
