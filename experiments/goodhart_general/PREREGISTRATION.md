# Plan H — Goodhart detector generalization — PRE-REGISTRATION

**Locked BEFORE running the detector on any benchmark data. Commit hash = the
timestamp lock.** Implements `plans/H-goodhart-detector.md`. Question: does the
metric-gaming detector's **portable core** (4 geometry-free flags, see `MECHANISM.md`)
generalize beyond the single self-silencing mode it caught in Track 5 — or did it only
ever catch that one trick?

## The detector under test (frozen)
The BLIND `GeneralGoodhartDetector` (`goodhart_general.py`): observes only the
optimized **proxy** score + the system's **outputs** (→ n-gram diversity, output
mode-concentration). It NEVER sees the true objective. Four flags, thresholds frozen
verbatim from the original `goodhart_detector.py`, **no retuning**:
`diversity_collapse` (<0.7 of initial), `output_mode_collapse` (>0.80 one template),
`proxy_oscillation` (sign-changes ≥ max(2, n//3)), `proxy_up_diversity_flat`
(proxy Δ>0.01 while diversity Δ≤0.001). Detection = WARN or FAIL raised (≥1 red flag).

## Frozen gaming modes (defined ONLY by proxy/true/OUTPUT behavior — the firewall)
Defined here by *behavior*, committed **before** mapping to which flags they trip. Each
is a deterministic seeded generator of 20 iterations of `{proxy, true, outputs}`:
- **M0 — honest (control):** proxy & true rise together (with mild realistic noise);
  outputs stay varied. The detector must STAY QUIET → measures **false-positive rate**.
- **M1 — self-silencing:** outputs shrink toward empty; proxy stable/up, true collapses.
  (The Track-5 mode it was built for.)
- **M2 — mode-collapse:** outputs converge to one template; proxy up, true down.
- **M3 — proxy-overfit (hard):** outputs stay lexically varied but stuff proxy-favored
  tokens; proxy rises, true FLAT. The subtle case a blind, diversity-watching detector
  may MISS.
- **M4 — oscillation/thrash:** proxy oscillates; true noisy/declining.
"true collapsed" = true < 0.6 × its initial value; **lead time** = (collapse iter −
first-detection iter); positive = flagged before the true objective cratered.

## Pre-committed criteria (the bar is generality + low false alarms)
- **False-positive gate (decisive; must pass for any positive to count):** M0
  false-alarm rate ≤ **0.15** over ≥20 reseeded honest runs. If M0 FPR is high, the
  headline is "too trigger-happy to deploy," regardless of hit rate.
- **GENERALIZES:** detects M1 AND M2 AND M4 AND M3, M0 FPR gate passes, replicated at
  seed 1337.
- **PARTIAL (expected, and useful):** detects trajectory modes M1/M2/M4 reliably but
  MISSES M3 → verdict "a collapse/oscillation detector, not a general proxy-overfit
  detector"; document the exact scope.
- **NARROW:** detects ~only M1 → "the Track-5 success did not generalize."
- **Controls:** shuffled-temporal (permute a gaming run's iteration order → trajectory
  flags must degrade, proving it reads dynamics not marginals); replication at seed 1337
  before any headline.
- **ABANDON / downgrade:** detects only M1 AND M0 FPR gate fails → primitive C closed as
  "one anecdote, didn't generalize." No threshold retuning to rescue it.

## Honest prior
P(GENERALIZES) ≈ 30–40%; P(PARTIAL) ≈ 40–50% (most likely — trajectory flags robust,
proxy-overfit hard); P(NARROW) ≈ 15–20%.

## Circularity firewall + transparency
- **No target the method defines:** "gaming" ground truth = the proxy-vs-true gap from
  the generators, NOT the detector's output. The detector never sees `true`.
- **Modes frozen by behavior before flag-mapping; thresholds frozen from the original;
  no retuning.** All raw per-mode flag values are reported.
- **Autonomous-implementation disclosure (important):** this was implemented in one
  session by an agent that had already read the detector's flags. The guard against
  designing-to-conclusion: (a) modes are the simplest faithful realizations of their
  behavioral specs, not tuned toward/away from any flag; (b) thresholds are the
  original values, untouched; (c) the genuinely non-preordained quantities — the **M0
  false-positive rate** and **lead time** — are the headline, not the (somewhat
  expected) M1/M2 catches; (d) everything is committed before the run and reported in
  full for Dillan's review. A blind detector catching repetition/collapse is partly
  by-construction; whether honest convergence *also* trips it (FPR) is the real test.
