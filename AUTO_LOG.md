# AUTO_LOG — decision trail for autonomous sessions

*Rule (state doc §4): every result gets one line on how it could be fooling itself.
No line, no result. Dillan audits per-session, after the run.*

## Session 2026-07-04 (handover: Dillan at work; plan in PLAN.md)

**U1 — auto-mode kit created** (`CLAUDE.md`, `PLAN.md`, this file).
Could-be-fooling-itself: a rules file *prompts* discipline but can't *enforce* it
(Dillan's own enforcement-beats-prose point); treat these as legibility for his audit,
not a guarantee.

**U2 — forum walk-in kit drafted** (`docs/FORUM_KIT.md`): 60s story, orientation,
one-sentence claim, 3 Murfet-question options, checklist. Draft only; Dillan's voice pass
required. Could-be-fooling-itself: a rehearsed story invites overclaiming under social
pressure — every number in it was reconciled to WORK_MAP/METHODS_NOTE before writing.

**U3 — stale-install shadow: diagnosed and killed.** Two of my own first probes were
cwd artifacts (running `python -c "import mirrorfield"` FROM the old repo / from C:\Users\User
puts those dirs on sys.path — false positives, caught before acting). Clean result:
the mirrorfield venv has NO stale install (experiments import the canonical package via
their own path inserts → §4y unaffected); the SYSTEM Python had the real one
(`geometric-safety-features 1.0.0` editable → the old experiment folder). Uninstalled
(pre-authorized, state doc §6/§9; reversible via `pip install -e`). Verified gone.
Remaining hazard, NOT acted on (destructive → Dillan's call): the old
`geometric_safety_features-Experiment` folder itself still exists with uncommitted
geometry-era files; archiving/renaming it is his decision.

**U4 — persistence fix committed (7abd37c):** raw mm/me/br (+ task labels) now persisted
by default in selfreport / sycophancy / prompt_injection / boundary harnesses; assembly
backward-compatible (nan for pre-fix rows). Verified: py_compile + read-only exercise of
the new assembly expressions against all 8 real checkpoints (1000/1000/480×4/500×2 rows).
Could-be-fooling-itself: NOT run-verified end-to-end — no GPU in auto, and the
auto-classifier (correctly, by intent) blocked even the no-GPU npz rebuild from complete
checkpoints. Next real run (or Dillan invoking --run/--regen on the complete checkpoints,
which skips generation) is the true test. Legacy scripts (selfconsistency_*, ablations,
gate_agent, calibrate_gate) NOT touched — not live harnesses.

**U5 — Platt-on-margin baseline (Amendment 1, locked f0c5b7f BEFORE analysis; result
committed after):** cross-seed held-out, verdict per locked rules =
**FRESH-MAP-CALIBRATED (candidate — Dillan concludes)**. Torn-quintile gap for the fresh
single-feature map: +0.088/+0.042 (accuracy CI contains prediction, both directions) vs
the frozen calibrator's +0.258/+0.212 on the same rows. Reading stays at the amendment's
ceiling: consistent with the sparse/stale-tail arm of the mechanism HYPOTHESIS; not a
fix; not the refit; deployed gate untouched.
Could-be-fooling-itself: (1) the fit42→eval1337 pass is THIN — mean p̂ 0.6178 vs CI upper
0.625, a 0.007 margin; a slightly different sample could flip the verdict; (2) seeds are
same-distribution — this says nothing about drift; (3) mm-only map got lucky with a
near-linear tail here; none of this validates deployment. Also observed (not a claim):
the fresh map's torn-region value (~0.62) sits near the frozen calibrator's FLOOR
(0.625) — the isotonic plateau at 0.79, not the floor, is what §4y caught.

**U6 — Plan J drafted** (`plans/J-second-model-transfer.md`, renamed from I — letter
taken by red-team plan §4o): second-model transfer protocol, DRAFT NOT LOCKED; GPU run
and final criteria wait for Dillan; includes fresh-calibrator-per-model protocol, power
check before locking, and the mandatory circularity section.
Could-be-fooling-itself: a transfer test designed by the same process that produced §4y
may inherit its blind spots — Dillan should specifically challenge the "identical
analysis" choice (it maximizes comparability but also copies any §4y flaw forward).

**U7 — consolidation-layer edits while owner away (flagged):** METHODS_NOTE.md §6 got a
clearly-bracketed addendum with the Amendment-1 candidate result ([Dillan: keep or cut]);
HANDOFF.md got a delta block. Both are pointers to logged results, no new claims.
Session end. Nothing pushed to GitHub; nothing public; no GPU used; no verdict concluded.
