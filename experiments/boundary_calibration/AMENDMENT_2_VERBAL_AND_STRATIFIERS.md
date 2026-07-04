# Amendment 2 to the §4y pre-registration — verbal confidence near the boundary + alternative stratifiers (locked BEFORE analysis)

*2026-07-04, autonomous session (AUTO_LOG U11). CPU-only; joins already-saved
`boundary_rows_{42,1337}.npz` (mm/me/br/p_int/correct) with A1's
`selfreport_rows_{42,1337}.npz` (verbal) — same items, same order; alignment is
asserted by requiring the two files' p_int to match to max|Δ| = 0. No generation;
gate untouched; candidate verdicts — Dillan concludes. One analysis pass, no retuning.*

## Questions

**A. Verbal confidence in the torn region.** A1 (§4t) showed spoken confidence predicts
correctness at chance (AUC 0.51) with aggregate calibration error 0.32–0.36 on this
model. Unasked: how is it distributed relative to the *boundary*? Per mm-quintile:
mean verbal confidence vs accuracy (Wilson 95%), gap = mean verbal − accuracy, next to
the internal gap from §4y. Rows lacking a parsed verbal confidence are excluded and
**counted per bin** (if missingness concentrates in the torn region, that is a selection
caveat and gets reported).

**B. Alternative stratifiers.** §4y stratified by raw mean-margin (mm) only. Same
analysis with the other two raw signals, torn quintile defined per stratifier (locked
now): mm → LOWEST quintile; mean entropy (me) → HIGHEST; boundary-ratio (br) → HIGHEST.
Caveat stated in advance: me/br correlate with mm — B is a robustness/description check,
not an independence claim.

## Locked verdicts

- **A — VERBAL-WORSE-IN-TORN:** verbal Q1(torn) gap exceeds the internal Q1 gap by
  ≥ +0.05 in BOTH seeds, with the accuracy CI excluding mean verbal.
  **VERBAL-SIMILAR:** |verbal gap − internal gap| < 0.05 in both seeds.
  Else **MIXED** (reported, not interpreted). Additionally reported (descriptive, no
  verdict): the trend of mean verbal across quintiles — does spoken confidence track
  boundary distance at all? (Prior from A1's AUC ≈ 0.51: it should be ~flat.)
- **B — SAME-PATTERN** (per stratifier): torn-quintile internal gap ≥ +0.10 with CI
  exclusion, both seeds. **NO-PATTERN:** gap ≤ +0.05 or CI contains the mean, both
  seeds. Else **MIXED**.
- min-n 30 per interpreted bin; Wilson 95%; both seeds; quintiles computed on the
  evaluation sample (same rule as §4y).

## How this could fool us (named in advance)

- Verbal confidences cluster on round numbers (80/90/100) — a coarse scale can make
  bin means unstable; we report the verbal distribution alongside.
- Verbal-missing rows (unparseable confidence) may not be random; per-bin exclusion
  counts are part of the result.
- me/br stratifiers are correlated with mm; B cannot distinguish "same failure seen
  through a correlated lens" from "independent failure" — it is not asked to.
- All §4y scope limits: one 3B model, one task family, on this model.
