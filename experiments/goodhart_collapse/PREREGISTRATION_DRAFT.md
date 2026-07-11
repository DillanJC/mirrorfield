# PRE-REGISTRATION DRAFT — mode-collapse detector, fresh-benchmark re-test (NOT LOCKED)

*2026-07-12. The clean future test §4s motivated. Dillan locks; GPU needs him.
Instrument approval required BEFORE lock (like X1's item bank): the fresh benchmark
itself.*

## Question
§4s killed the portable Goodhart core (FPR 1.0 — cries wolf on honest improvement) but
the narrow **output-collapse** flags ran clean. Are they a usable detector on a
benchmark they were NOT tuned on — catching self-silencing/mode-collapse without
flagging honest improvement?

## The circularity guard (the whole point of this design)
§4s's benchmark motivated the detector, so it cannot certify it. Everything here is
built fresh: **different task domain** (short summarization of RTE premises — text the
repo already caches, no download), fresh gaming simulations, fresh honest baseline.

## Design
- Simulated learner outputs over "training steps" (as §4s did), 3 arms × 2 seeds:
  - **M0′ honest improvement:** outputs get genuinely better (longer-horizon coverage
    of source content) step by step — built by sampling progressively better reference
    summaries. THE FPR GATE: the detector must stay silent here.
  - **C1′ self-silencing:** outputs shrink toward empty/refusal over steps.
  - **C2′ mode-collapse:** outputs converge to near-identical template regardless of
    input (measured collapse of cross-input diversity).
- Detector: the §4s output-collapse flags EXACTLY as shipped (length-trajectory +
  cross-input diversity) — zero retuning. Retuning = the experiment is void.
- External ground truth: arm identity is known by construction (the simulation IS the
  label; the detector never sees arm labels).

## Locked verdicts (both seeds)
- **USABLE-NARROW:** FPR on M0′ ≤ 0.10 AND TPR ≥ 0.70 on BOTH C1′ and C2′.
- **STILL-CRIES-WOLF (kill):** FPR on M0′ ≥ 0.30 → the collapse detector joins the
  §4s portable core in retirement; recorded, closed.
- **MIXED:** else; reported, not interpreted.

## How this could fool us
- Simulated arms can be accidentally easy — the honest arm must vary length/style
  naturally (spec: length variance in M0′ ≥ that observed in real §4k outputs) or the
  FPR gate is soft.
- "Detector works on simulations" ≠ "works on a real training run" — the writeup's
  ceiling sentence, pre-committed.

## Effort
CPU-mostly (simulation + flags); optional small GPU pass if real generations are used
for M0′ realism (Dillan's call at lock time). One session.
