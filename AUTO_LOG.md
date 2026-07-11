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

*(Session extended on Dillan's instruction: "keep going till you have ran out of
credits". Continuing under the same hard stops.)*

**U8 — WORK_MAP §4z:** audit + Amendment 1 logged in the canonical record, candidate
framing. Could-be-fooling-itself: same numbers now live in 4 places (results JSON,
audit, WORK_MAP, note) — any future edit must reconcile all four; the repo JSONs win.

**U9 — related work VERIFIED + positioning correction (the big one):** every citation
in METHODS_NOTE §10 now web-search-verified with links (Guo 1706.04599; Hébert-Johnson
1711.08513; Geifman&El-Yaniv 1705.08500; Kadavath 2207.05221; Tian 2305.14975; Sharma
2310.13548; Greshake 2302.12173; Wallace 2404.13208; XSTest 2308.01263; OR-Bench
2405.20947). TWO SUBSTANTIVE FINDS: (1) **multicalibration already names the general
phenomenon** (aggregate calibration masking subgroup failure; no subgroup guarantee from
Platt/temp scaling) → §10 now opens with the forced narrowing: §4y claims the INSTANCE +
demonstration, not the phenomenon — this correction protects the note from its own
biggest overclaim risk; (2) Tian et al. 2023: verbalized confidence BEATS token probs on
RLHF'd frontier models — opposite ordering to A1 on this 3B model → strengthens
"on this model" scoping, cited as such. Also: Sakana Fugu is real (June 2026, RL-trained
Conductor/TRINITY orchestration-as-a-model); the Fugu-vs-Parallax delta paragraph the
state doc asked for is drafted in FORUM_KIT (verifier-inside-the-loop vs human-external
verifier — Fugu has the closed-loop shape this project's retractions came from).
Could-be-fooling-itself: search snippets are secondary sources; one ID (1711.08513) was
initially written from memory then verified; Dillan should spot-check links before
circulation; the Parallax half of the Fugu paragraph is from the state doc's one-line
description, not primary notes.

**U10 — reliability figure** (`boundary_reliability.png` + `make_figure.py`): pure
visualization of logged §4y/§4z numbers, referenced from the note (bracketed).
Could-be-fooling-itself: a figure implies a narrative; axes/annotations were checked
against the JSONs; no smoothing anywhere.

**U11 — Amendment 2 run (locked ddaed33 first): verbal-near-boundary + me/br
stratifiers.** Verdicts (candidates): A = MIXED — verbal is numerically LESS
overconfident in the torn bin (+0.13/+0.19 vs internal +0.21/+0.26) BUT only because
spoken confidence is flat ~0.7 everywhere (underconfident −0.2..−0.28 where the model is
right) — logged with the explicit anti-headline warning (broken-clock effect); B: me →
SAME-PATTERN (+0.157/+0.224, CI-excluded, both seeds — the conditional failure is not
margin-axis-specific), br → MIXED (quantile bins collapse under ties). verbal_missing
0/500 both seeds. Logged in WORK_MAP §4z.
Could-be-fooling-itself: me/br correlate with mm (robustness not independence — stated
in the lock); two empty br bins mean the br verdict is about instrument coarseness, not
the phenomenon; the verbal comparison uses the same rows as §4y, so it inherits any §4y
quirk. One implementation bug (empty-bin crash) was fixed BEFORE any results were seen —
no criteria were touched.

**U12 — field guide DRAFT + smell test:** `docs/FIELD_GUIDE_DRAFT.md` (the outward
artifact from the impact conversation — Dillan's voice sections marked, sharing his
call) + `docs/SAFETY_CLAIM_SMELL_TEST.md` (12 questions, each with its receipt from this
repo). Could-be-fooling-itself: a guide written by the AI it warns about is itself a
consolidation layer — every receipt cites its §, and the guide says plainly that its
drafting AI is inside the loop it describes.

**U13 — tests + index hygiene:** `tests/test_boundary_analysis.py` (10 passing — wilson
incl. the §4y torn-bin known value, `_signals` nan/finite/delegate paths, frozen
tau/edges constants so silent drift fails a test) + `plans/README.md` updated (H/I done,
J added). Could-be-fooling-itself: tests written by the same session that wrote the
code; they encode current behavior, not independent expectations — still better than
prose.

**U14 — FRINGE_LEADS X2 citations verified** (Keeling 2411.02432; EmotionPrompt
2307.11760; Palisade shutdown-resistance blog + 2509.14260; Claude 4 system card
§5.5.2 bliss attractor). Could-be-fooling-itself: verified existence + headline claims
via search snippets, not full-text reads; a replication pre-reg must read the primary
source first.

**U15 — README current-work pointer + HANDOFF delta extended. ANALYSIS STOP declared:**
three pre-registered passes (§4y + Amendments 1–2) over the same 1,000 saved rows is
this dataset's limit — a fourth pass would be the garden of forking paths wearing a
pre-registration costume. Next analyses require fresh data (GPU, Dillan-gated).
**U16 — circularity-auditor one-pager** (`docs/CIRCULARITY_AUDITOR_SKETCH.md`): three
buildable tiers, the auditor's own failure modes named first, the collaboration ask
concrete (specimen corpus + theory + second builder); linked from FORUM_KIT.
Could-be-fooling-itself: a design sketch by the AI whose loop the tool would audit —
tier-2's stake-free constraint is stated but unproven.

Session totals: 13 commits this auto stretch, 0 pushed, 0 GPU, 0 concluded verdicts,
5 candidate verdicts queued for Dillan (audit, Platt, Amendment-2 A/me/br). All 32
repo tests pass. Work remaining is Dillan-gated (verdicts, voice passes, GPU runs,
archive decision, push) — stopping here is the discipline, not fatigue.

## Session 2026-07-11 (last Fable day; forum done; Dillan at work — "do as much as you can")

Budget-constrained (≈10% weekly usage): spent on design-grade artifacts only, banked to
`OneDrive\mirrorfield-pocket\` (synced + nightly-covered): **EXIT_RAMP_ESSAY_DRAFT.md**
(the letter to people mid-AI-spiral — his voice, his publish decision),
**STRUCTURES_OF_MERCY_PATTERNS.md** (12-pattern language spine; the unifying book),
**BAR_FIELD_NOTES_KIT.md** (ethics rules + 3-line capture method),
**NEW_DIRECTIONS_BANKED.md** (nine unpursued directions, first steps sized).
Could-be-fooling-itself: all four are MY framing of HIS assets — each is a draft for
his judgment, none is a commitment; the essay especially must pass his edit before any
eyes see it (it trades on his story). Forum debrief was never captured (his dump never
came) — the template waits; details are decaying.

## Dillan's return (2026-07-04, same day)

**All five candidate verdicts ACCEPTED by Dillan** (audit; Platt FRESH-MAP-CALIBRATED;
Amendment-2 entropy SAME-PATTERN / verbal MIXED / br MIXED) — acceptance recorded in
CIRCULARITY_AUDIT.md, METHODS_NOTE.md §5, WORK_MAP §4z. **npz rebuild authorized and
run** (system Python — the mirrorfield venv lacks `datasets`; note for env hygiene):
all 8 npz rebuilt through the new assembly code, generation skipped as predicted;
regression check = every analysis re-run, ZERO changes in any results JSON. The
persistence fix is run-verified end-to-end. Old-folder archive deferred until after
the forum (his call); push deferred until after his edit pass (his call).
