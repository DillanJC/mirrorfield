# Mirrorfield MCP Server — an honest uncertainty gate for AI agents

An MCP (Model Context Protocol) server that lets an AI agent measure how
confident it was while writing an answer — and decide, **before the answer is
sent**, whether to present it, present it with a warning, or hold it back.

It works from token log-probabilities only (how torn the model was between
word choices). The geometric/embedding analysis this server once carried was
removed after a non-circular ablation showed it added nothing
(ΔAUC ≈ 0; see `WORK_MAP.md` §4b). What remains is the part that survived
every falsification attempt this project could throw at it.

## What it measures, and how well (validated numbers)

Signals: mean token margin, mean entropy, boundary-token ratio — combined into
a calibrated `p_correct`, plus a context-relative `p_correct_relative` that
z-scores each answer against the last ~50 answers in the SAME context
(`context_id`), using the per-context `RollingGate`.

Live, pre-registered evaluation on ~1,100 fresh items (WORK_MAP §4k, criteria
frozen at commit b1fdc08 before the data existed):

- Wrong-answer ranking: **AUC 0.685 [0.642, 0.730]** (fresh RTE+QNLI, Qwen2.5-3B).
- At the frozen threshold the gate **abstained on 14.7% of answers and caught
  27.7% of all errors** (random abstention at the same rate catches 14.7%);
  accuracy among presented answers rose from 81.8% to 84.5%.
- Shuffled-label nulls clean; replicated across 3 stream orders; the
  in-process path and this server agree EXACTLY (25/25 parity,
  `experiments/gate_agent_parity.json`).

Honest limits: this is a **modest** gate, not an oracle — it improves the odds,
it does not guarantee safety. It detects *likely-wrong* answers, not harmful
ones (that is separate, ongoing work — `plans/E-harm-gate.md`). Signals
strengthen with model scale. **Hard deployment rule:** one `context_id` per
kind of work; never share one rolling buffer across very different tasks
(WORK_MAP §4i/§4k — offset-dependent degradation).

## Tools (4)

1. `analyze_logprobs` — token-level uncertainty signals from log-probs.
2. `confidence_report` — the gate: confidence score + calibrated
   `p_correct` (+ `p_correct_relative` when `context_id` is given) +
   uncertain spans + a proceed / verify / abstain recommendation.
3. `compare_responses` — rank candidate drafts; self-consistency check
   flags confident-but-disagreeing drafts.
4. `novelty_map` — *interpretive* "epistemic terrain" view (well-trodden /
   frontier / uncharted) computed from the same validated signals. The
   terrain categories are heuristic labels, not calibrated probabilities.

Prompts: `assess-my-response`, `compare-drafts`, `explore-uncertainty`.
Resource: `mirrorfield://calibration` (weights + thresholds).

## Quick start

```bash
# from the repo root
python -m mirrorfield.mcp.server          # stdio transport
```

Pass `context_id` (any stable string per kind-of-work) to `confidence_report`
to enable the validated context-relative probability. The first ~5 calls per
context report `warming_up: true`.

A watchable end-to-end demo (local LLM answering questions with the gate
holding back low-confidence answers live):
`python experiments/run_gate_demo.py --run`

## Provenance

Every number above traces to a results JSON in `experiments/` and a
pre-registration commit. The full research log — including everything that
was tried and falsified on the way here — is `WORK_MAP.md`. Calibration files:
`gate_calibration.json` (absolute), `gate_calibration_relative.json`
(context-relative; trained Qwen2.5-3B on RTE/QNLI/SST-2).

Removed in the lean rebuild (2026-06): embedding-geometry tools and Moltbook
posting (`moltbook_bridge.py` remains on disk, unwired).
