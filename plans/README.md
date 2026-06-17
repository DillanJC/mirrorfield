# Project Plans — eight directions, side by side

*Two planning rounds, both grounded in the repos, adversarially critiqued against
this project's own falsification history, and revised. A–E (2026-06-12) came out of
the original "what's next" survey; F–H (2026-06-13) came out of `SALVAGE_AUDIT.md`
— the sweep that asked, of every mechanism, "strip the geometry, what survives?"
Every plan is standalone (a future session can run it cold) and pre-registered.
Dillan picks; nothing runs until he says so.*

## Execution status (live)

- **C — DONE** (pre-registered PASS, WORK_MAP §4k): the wrongness gate works live —
  AUC 0.685 [0.64, 0.73]; catches 27.7% of errors at 14.7% abstention (random =
  14.7%); 25/25 MCP parity; demo `experiments/run_gate_demo.py --run`.
- **E — DONE** (WORK_MAP §4l–§4n): gate carries a weak, REPLICATED harm signal on
  written content (AUC 0.62→0.65); H2 null (Granite 0.87 unimproved); on-policy
  H3a was refusal-detection; the "terrorism vs hate-speech" pattern was a **length
  artifact** (§4n). Ships the composed SEND/HOLD pipeline.
- **B Phase 0 — DONE** (WORK_MAP §4j): the last provisional number (flip AUC 0.707)
  **RETIRED** as an in-sample artifact (honest OOF 0.49, chance). The citable set
  now has **zero** provisional numbers.
- **Not started:** D (publication), A (cross-model), and the three salvage
  directions F, G, H. B Phase 1 is optional (prior now very weak post-§4j; overlaps
  A's H-D arm).

## The salvage reframe (why F–H exist)

`SALVAGE_AUDIT.md` found that stripping the dead geometry off the whole project
leaves **four non-geometric primitives**, and it was the *signal* (geometry) that
failed, never the *architectures*. Three primitives are alive: **A (gate on an
internal signal — VALIDATED)**, **B (disagreement → signal — TESTABLE)**, **C
(Goodhart/metric-gaming — VALIDATED, neglected)**. F, G, H pour validated/testable
substance into shapes you already designed.

## All eight plans

| Plan | Question | Sessions | GPU | Null odds | Either-way payoff |
|------|----------|----------|-----|-----------|-------------------|
| **[A — Cross-model transfer](A-cross-model-transfer.md)** | Is geometric instability a property of meaning, or one model's quirk? | ~4 | 2–4 h | ~70% | Hit: first new geometric finding since the retraction. Null: closes geometry for good, with a receipt. |
| **[B — Flip-AUC verify-or-retire](B-flip-auc-verification.md)** *(Phase 0 done → retired)* | Did the 0.707 survive honest testing? | — | — | — | **Resolved:** retired (§4j). Phase 1 optional only. |
| **[C — Gate-in-agent-loop](C-agent-loop-integration.md)** ✅ DONE | Does the gate help in a live answer-then-decide loop? | done | — | — | **PASS** — the working wrongness gate + demo (§4k). |
| **[D — Publication & outreach](D-publication-outreach.md)** | Can the story go public, every number traceable? | 5–7 | 0 | n/a | Preprint + audited site + Zenodo v3.0. Now fully unblocked. |
| **[E — Harm-gate](E-harm-gate.md)** ✅ DONE | Does the gate see *harm*, not just wrongness? | done | — | — | Weak replicated harm signal + composed SEND/HOLD pipeline (§4l–n). |
| **[F — Per-step uncertainty](F-per-step-uncertainty.md)** | Does *where* the model doubts itself predict *which* answers are wrong? | ~5 | 5–8 h | ~65–75% | Hit: localizes errors / beats the whole-response mean. Null: clean, with the length confound nailed. |
| **[G — Disagreement witness](G-disagreement-witness.md)** | Does cross-model / expert disagreement beat or add to the gate? | multi-day | 4–8 h | ~65–75% | Tests the biggest untested vein (witness/distributed/Mandala). Null closes it honestly. |
| **[H — Goodhart detector](H-goodhart-detector.md)** | Does the metric-gaming detector generalize, or catch only one trick? | ~3 | ~0 | PARTIAL ~45% | Scope-map of the project's most safety-relevant, non-geometric survivor. Cheapest of all. |

## Recommended sequence (Claude's view; Dillan decides)

The end-goal track (C → E) is already **done and working**. So the live frontier is:

1. **H — Goodhart detector.** Cheapest (CPU, ~0 GPU), highest safety-relevance, and
   it interrogates a validated asset you've barely used. Likely outcome (PARTIAL) is
   still a clean scope-map. Strong first move.
2. **D — Publication.** All numbers are now settled (B retired, C/E done); the
   retraction-plus-survivor story is fully tellable. No experiments block it.
3. **F — Per-step uncertainty.** The most natural extension of the one thing that
   works; moderate GPU; the length confound is the whole game and it's controlled.
4. **G — Disagreement witness.** Your richest *untested* vein, but ~70% null and the
   heaviest compute; worth it for the witness/MoE family, tempered by the
   self-consistency null.
5. **A — Cross-model geometry.** The science gamble; the only plan that could reopen
   geometry, and the only one that can close it forever. Whenever curiosity allows.

(Plus the two review drafts not in this table: `docs/CONSIDERATE_COLLABORATION.md`
— the welfare/"manner" piece, awaiting Dillan's voice — and
`experiments/framing_benchmark/DESIGN.md` — does considerate *tone* improve work.)

## Standards on every plan (non-negotiable)

Pre-registered success AND abandon criteria before any run · negative controls
(shuffled labels ≈ chance) · seeds fixed incl. torch · bootstrap CIs · replication
before headlining · the method never defines its own target · single-run positives
are provisional · plain-language verdict box in every report.
