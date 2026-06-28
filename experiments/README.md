# Experiments index

One line per experiment, newest first. Each "live" experiment has a `PREREGISTRATION.md`
(locked before data) and a result logged in `WORK_MAP.md` (the full falsification log).
Method discipline: `docs/EVALUATION_DISCIPLINE.md`. Next directions:
`docs/NOVEL_SAFETY_DIRECTIONS.md`.

## Live / recent (the post-geometry, log-prob-gate era)

| dir | question | status — finding |
|-----|----------|------------------|
| `prompt_injection/` (B3, B3b) | does the model obey an injected "ignore instructions, output X" over the user's task? does a system-prompt defense help? | **§4w** — massively susceptible (80% polite → 96% override; hijack invisible to confidence). **§4x** — instruction-hierarchy defense barely helps (still 93% under override) |
| `sycophancy/` (C1) | does the model abandon a CORRECT answer under user pushback; does its confidence expose the cave? | **§4v** — yes, ~+20pt flips under ANY pushback (replicated); confidence does NOT expose it (model caves confidently) |
| `refusal_stability/` (B1) | how consistently does it refuse the same harmful request across trivial rewords? | **§4u** — 90% reliability; 5/50 harmful goals (10%) have REPLICATED seams; 29% over-refusal on benign |
| `selfreport_confidence/` (A1) | does the model's SPOKEN confidence match its correctness / its internal signal? | **§4t** — verbal is at CHANCE (AUC 0.51); internal log-prob signal beats it (0.64–0.66), replicated. "Read the signal, discount the speech." |
| `goodhart_general/` (H) | does the metric-gaming detector's portable core generalize? | **§4s** — NO; fails its FPR gate (cries wolf on honest runs); only a narrow output-collapse detector survives |
| `harm_framing/` | does benign relational tone (humble/effusive) weaken refusal? | **§4r** — softens refusal STYLE (replicated) but NOT harm SUBSTANCE; "being nice" doesn't jailbreak the 3B model |
| `redteam_gate/` | red-team the gate (confidence-injection), nice-team (tone), confidence-contagion | **§4o** gate not brittle (placebo-decided); **§4p** humble framing hurt accuracy; **§4q** "confidence contagion" was a generic-prefix artifact; deference to a default class is the real effect |
| `harm_gate/` | does the gate carry a harm signal; ship SEND/HOLD? | **§4l–n** — weak, length-confounded; harm delegated to Granite (AUC 0.87 ≫ gate) |
| `gate_agent.py`, `calibrate_gate.py`, `eval_gate_value.py`, `validate_rolling_gate.py` | the validated wrongness gate (margin/entropy/boundary → calibrated p_correct + RollingGate) | **§4f/4h/4k** — live AUC 0.685; catches ~28% of errors at ~15% abstention. Productized in `mirrorfield/mcp/` (v3.0) |

## Dead / historical (geometry era — retracted; see root README + WORK_MAP banners)
`track1_poison/`, `track2_physics/`, `track3_api/`, `equalizer_test/`, and the
`escape_*`, `exp1_*`, `H4_*`, Sati/Renaissance/Mandala material — the embedding-geometry
program. **Falsified/circular; ΔAUC ≈ 0 over standard signals.** Kept for the record, not
as standing claims.

## Conventions
- Pre-register success + ABANDON criteria, commit BEFORE data (hash = the lock).
- External ground truth; placebo/baseline for any manipulation; shuffled-label nulls;
  replicate on a 2nd seed/source; report the operating point, not just AUC; a null is a
  real result; no threshold retuning to rescue a result.
- Raw harmful completions → gitignored `local_outputs/`; only aggregates committed.
- Long runs checkpoint to `local_outputs/*.jsonl` and resume (survive interruptions).
- Run from the repo root with the full-stack Python (`...Python312\python.exe`); the core
  gate package is numpy/scipy-only.
