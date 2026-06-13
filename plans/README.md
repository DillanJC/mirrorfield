# Project Plans — the five directions, side by side

*Planning phase completed 2026-06-12. Each plan was drafted grounded in the actual
repos, adversarially critiqued against this project's own falsification history,
and revised. Every plan is standalone: a future AI session can execute it with no
other context. Dillan picks the direction; nothing below runs until he says so.*

**Execution status (2026-06-12):**
- **B Phase 0 — DONE:** the 0.707 RETIRED as an in-sample artifact (honest OOF
  0.49, chance; WORK_MAP §4j). Phase 1 remains optional (overlaps Plan A H-D).
- **C — DONE (pre-registered PASS, WORK_MAP §4k):** live AUC 0.685 [0.64, 0.73];
  gate catches 27.7% of errors at 14.7% abstention (random = 14.7%), CI clears
  zero; 25/25 MCP parity; demo: `experiments/run_gate_demo.py --run`.
- **E — DONE (WORK_MAP §4l/§4m):** H1 the gate carries a weak but REPLICATED
  harm signal on written content (AUC 0.62/0.65, fresh-split replication);
  H2 null (Granite 0.87 unimproved); H3a on-policy = refusal detection. Ships
  the composed SEND/HOLD pipeline `experiments/harm_gate/harm_screen_demo.py`.
- Remaining: **D** (publication) — all numbers now settled; nothing blocks it.
  Optional later science: **A** (cross-model transfer), **B Phase 1** (fresh
  flip data — prior now very weak after §4j).

## The five plans

| Plan | Question it answers | Sessions | GPU | Chance of "null" | What you get either way |
|------|--------------------|----------|-----|------------------|------------------------|
| **[A — Cross-model transfer](A-cross-model-transfer.md)** | Is geometric "instability" a property of meaning itself, or one model's quirk? | ~4 | 2–4 h | ~70% | Success: first new geometric finding since the retraction (strictest gates). Null: the geometric line closes **completely**, with a pre-registered receipt. |
| **[B — Flip-AUC verify-or-retire](B-flip-auc-verification.md)** | Does the last provisional number (0.707) survive honest testing? | 3–4 | 2–3 h | ~70–80% | Lock: a defensible instability headline for outreach. Collapse: the final geometric claim retires; the public story becomes asterisk-free. |
| **[C — Gate-in-agent-loop](C-agent-loop-integration.md)** | Does the validated gate actually *help* when wired into a live answer-then-decide system? | ~4 | 2–3 h | ~30–35% | The first working end-to-end "hold the shaky answers" demo + honest selective-prediction measurement. Highest success odds of any direction. |
| **[D — Publication & outreach](D-publication-outreach.md)** | Can the story go public with every number traceable and nothing inconsistent? | 5–7 | 0 | n/a (process) | Preprint + audited website + Zenodo v3.0 + outreach emails. Failure is only inconsistency, and audit gates make that structurally hard. |
| **[E — Harm-gate](E-harm-gate.md)** | Does the wrongness gate see *harm* — the actual end goal? | 5 (+1) | 5–7 h | H1 ~60–70%, H2 ~85–90% | Even on total null: the first **SEND/HOLD** pipeline (gate for wrongness + dedicated classifier for harm) + a clean publishable answer. |

## How they connect

```
            B(Phase 0 only, ~half a session, CPU)
                      │  settles the 0.707's fate
                      ▼
   C ──────────► E                    D (paper/website/Zenodo)
 (wrongness      (adds harm           needs B-Phase-0's verdict;
  agent loop)     screening on top    strengthened by anything
                  of C's loop)        A/B/C/E produce
                      ▲
   A ◄────────────────┘
 (science gamble; its H-D arm overlaps B Phase 1 —
  run one or the other, not both blindly)
```

- **B Phase 0 is nearly free** (hours, CPU-only, uses archived data) and unblocks D's
  paper regardless of outcome. It can run before or alongside anything.
- **C → E is the end-goal track:** C builds the wrongness loop; E extends it with a
  harm classifier and tests whether the gate sees harm at all.
- **A is the science gamble:** highest novelty if it hits, ~70% it closes the
  geometric era instead (also valuable). Its H-D arm subsumes B Phase 1.
- **D can start any time** but lands best after B Phase 0 (so the paper states the
  0.707's fate as fact, not as "pending").

## Recommended sequence (Claude's view, Dillan decides)

1. **B Phase 0** — half a session, settles the last number, feeds everything.
2. **C** — build the thing the project is *for*; highest success probability;
   the morale and demo payoff are real.
3. **E** — extend C toward the actual goal; ships the SEND/HOLD pipeline even on null.
4. **D** — publish once the numbers are settled and the demo exists (a working
   artifact makes the paper and website far stronger).
5. **A** — the science gamble, whenever curiosity and GPU-hours allow; it is the
   only plan that could reopen geometry, and the only one that can close it forever.

## Standards that apply to every plan (non-negotiable)

Pre-registered success AND abandon criteria before any run · negative controls
(shuffled labels ≈ chance) · seeds fixed incl. torch · bootstrap CIs · replication
before headlining · the method never defines its own target · single-run positives
are provisional · plain-language verdict box in every report.
