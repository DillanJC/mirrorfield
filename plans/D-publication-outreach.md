# Plan D: Publication & Outreach Package

## In plain language

We turn the project's story into things other people can read, check, and cite: a
short research paper (preprint), an updated project website, a refreshed permanent
archive (Zenodo), and ready-to-send outreach emails. The story is unusual and
genuinely valuable: *a person who cannot code, working with AI assistants,
published an exciting result, then rigorously tested it, found it was a statistical
mirage, publicly retracted it — and salvaged one small, honestly-measured tool from
the wreckage.* Success looks like: every number in every public artifact traces to
a results file in the repo, nothing contradicts the retraction, and the paper is
live on a preprint server. Failure here isn't a null experiment — it's
*inconsistency*: accidentally publishing a retracted number, or claiming more than
the data supports. So this plan is built around audit gates rather than
experiments. This direction needs more of **your** actions than any other (accounts,
clicking "submit", deciding how your name appears) — the AI sessions prepare
everything, but you press the buttons.

## Why this matters / what it builds on

- It is the direct expression of Dillan's stated purpose: being a positive force in
  how AI development is done. The field has a known shortage of published negative
  results and honest retractions; this project has a complete, receipts-attached
  example of both.
- It builds ONLY on the verified record: the v3.0 retraction README (live),
  `WORK_MAP.md` §4–§4i, `ORIENTATION.md`, and the locked v3.0 numbers (per-task gate
  AUC 0.60–0.74; rolling cross-task 0.63 [0.58, 0.69]; unseen-task transfer
  QNLI 0.72 / RTE 0.63; geometry Δ≈0; borderline instability r≈−0.5, weak).
- Two stories are publishable, and they are the SAME artifacts at different zoom:
  1. **The negative result** (AI-safety audience): "Geometric features do not detect
     data poisoning — a pre-registered, multi-test falsification," with the
     circularity meta-pattern (the method defining its own target) as the
     generalizable warning.
  2. **The methodology** (meta-science / AI-assisted-research audience): the
     falsification-harness pattern, multi-AI cross-checking (a model-switch review
     materially corrected a conclusion — documented in WORK_MAP §4h), and what
     verification standards a non-coder + AI team needs. The AI-assisted authorship
     is disclosed prominently — it is part of the contribution, not a footnote.

## What already exists

Verified, all public except where noted:

- **Live retraction:** v3.0 README at `DillanJC/Geometric_Safety_Features-V2.0.0`
  (commit `be9b358`), incl. citation note redirecting old-number citations.
- **The verified record:** `C:\Users\User\mirrorfield\WORK_MAP.md` (the full
  falsification narrative §4–§4i) and `ORIENTATION.md` (reconciliation incl. the
  outreach-numbers lock) — both public in `DillanJC/mirrorfield`.
- **Every results file:** `experiments/*.json` in mirrorfield (ablations v1/v2,
  self-consistency runs incl. the retracted §4d artifact, calibration, rolling-gate
  validation) plus the R0–R3 results under `experiments/track1_poison/`.
- **Website + video assets:** `DillanJC/consolidated-experiments` →
  `geometric-safety-extras/website/` (static site; **content is v2.0-era and
  contradicts the retraction — currently the biggest consistency liability**) and
  `video_scripts/` (Manim explainer, also v2.0-era).
- **Zenodo record 18290279** — DOI badge on the public README; archives the
  pre-retraction version. Needs a new version, not a new record (Zenodo versioning
  keeps the DOI lineage).
- **Private (NEVER in any repo or artifact):** Dillan's arXiv endorsement code and
  target endorser names, in his private notes. Memory rule already standing.
- **Sequencing dependency:** Plans A/B test the last provisional number (the 0.707).
  The paper must not state that number's fate prematurely — see Step 2.

## The plan, step by step

All drafts live in `C:\Users\User\mirrorfield\paper\` (folder exists, currently
old material — new work goes in `paper\v3\`). No GPU anywhere in this plan.

**Step 1 — The Number Audit (1 CPU session; the foundation gate).**
Build `docs/NUMBER_AUDIT.md`: a table of every quantitative claim that will appear
in ANY public artifact → its exact source (results JSON path + git commit + the
script that produced it) → its allowed phrasing (with CI). Then an automated
consistency check (small script, `audit_check.py`): grep all public-facing files
(README v3.0, website drafts, paper drafts) for the retracted/unsourced set —
`0.947`, `+8.8%`, `0.785`, `0.707` (as fact) — and for any number absent from the
audit table. **Gate: nothing advances to Steps 3–6 until the check passes clean.**
Artifact: the audit table + the checker script (both committed).

**Step 2 — Sequencing decision (10 minutes, Dillan + one AI session).**
The paper's instability section depends on the 0.707's fate. Options, decided now:
(a) run Plan B *Phase 0 only* first (~half a session, CPU, likely retires the
number with receipts) and write the paper with that verdict; or (b) write the paper
now, wording the 0.707 strictly as "provisional, in-sample, under verification."
**Recommendation: (a)** — it is nearly free and makes the paper's story complete.
(Full Plan A/B remain independent later work either way.)

**Step 3 — Paper draft (2–3 CPU sessions).**
`paper\v3\falsification_paper.md` → rendered to PDF (pandoc or typst, local).
Working title: *"When the Detector Defines the Target: a Falsification Case Study
in Geometric AI Safety."* Outline (locked now):
1. Motivation — pre-send output gating; a non-coder + AI team.
2. The seductive result — poison-detection AUC 0.947 and siblings.
3. The circularity meta-pattern — four instances, one mechanism (the method
   defines its own target); the falsification harness (R0–R3 design).
4. What survived honest testing — the log-prob gate, full characterization
   (per-task → rolling z-score → unseen-task transfer → context-separation
   constraint), with the §4d self-consistency retraction as a live example of the
   standard applied to ourselves.
5. Methodology lessons — pre-registration, negative controls, seeded everything,
   multi-AI adversarial review (§4h model-switch catch); a checklist other
   AI-assisted researchers can copy.
6. Limitations — small models, three tasks, modest AUCs; what this does NOT claim.
- Every figure regenerates from a repo results JSON via a committed plot script
  (no hand-drawn numbers). AI-assistance disclosure in the author note, prominent.
- Authorship: Dillan (conception, direction, verification standard) — listed
  author; AI assistants acknowledged per venue policy. **Dillan decides name vs
  pseudonym in Step 6 (privacy: his GitHub is DillanJC; email is private-relay).**

**Step 4 — Adversarial read (1 session, the project norm applied to prose).**
A fresh AI session (ideally a different model, continuing the multi-AI pattern)
red-teams the draft against: the audit table (every number), WORK_MAP (every
claim), and an overclaim hunt ("would a hostile reviewer call this sentence more
than the data shows?"). All critical findings fixed before anything goes public.

**Step 5 — Website + Zenodo + video decision (1 session + Dillan actions).**
- Rewrite `website/` content to the v3.0 story (the retraction arc + the gate +
  links to repos/paper). Static HTML edit; deploy free via GitHub Pages
  (**Dillan: enable Pages in repo settings — one click**). Until rewritten, add a
  banner-redirect to the v3.0 README so the stale v2.0 site never circulates bare.
- Zenodo: create a **new version** of record 18290279 from the current public
  repos; updated description = retraction abstract. (**Dillan: Zenodo login +
  two clicks; AI prepares all text.**)
- Video scripts: **defer** (pre-registered decision) until the paper is stable —
  rewriting Manim scripts before the text settles is the scope-creep trap. Revisit
  only after Steps 3–5 are done.

**Step 6 — Outreach (Dillan's hands only).**
AI drafts: a 150-word abstract-pitch email, a longer endorsement-request email
(numbers strictly from the audit table), and a posting plan (arXiv cs.LG primary;
fallbacks that explicitly welcome negative results / workshops, e.g. the NeurIPS
"I Can't Believe It's Not Better" workshop lineage, SoLaR-style safety workshops,
or non-arXiv preprint servers like OSF if endorsement stalls — final venue list
verified live in this step, not assumed). **Dillan: sends emails from his own
account, using his private endorsement details; nothing private ever enters a
repo.** If endorsement does not materialize in 4 weeks: post to OSF/Zenodo as the
preprint of record and continue — publication does not block on gatekeeping.

## Pre-registered success / failure criteria

This is process, not hypothesis-testing — so the gates are verifiable consistency
conditions, decided now:

- **G1 (audit gate):** 100% of quantitative claims in public artifacts trace to a
  results file + commit in the audit table; `audit_check.py` runs clean (zero
  retracted-number hits) on every artifact at publish time. **Hard blocker.**
- **G2 (consistency gate):** the public website no longer asserts any v2.0-era
  claim (checked by the same script against a phrase blacklist).
- **G3 (review gate):** the adversarial read (Step 4) reports zero unresolved
  critical findings.
- **G4 (publication):** preprint publicly accessible (arXiv if endorsed, otherwise
  OSF/Zenodo) + Zenodo v3.0 version live + website consistent — all three, or the
  direction is not "done."
- **ABANDON/PAUSE condition:** if Plan B Phase 0 or Plan A results land mid-draft
  and change a number's status, the paper PAUSES until the audit table is updated
  and re-checked — never publish-then-amend within days. There is no scientific
  "null" here; the only failure is publishing something inconsistent, and the
  gates exist to make that structurally impossible.

## Controls & verification

- **Circularity check — what target does the method define? None.** Nothing here
  generates numbers; it only transports audited ones. The audit table is the
  single source of truth, and the checker is automated, not vibes.
- **The retracted-number blacklist** (0.947, +8.8%, 0.785, standalone 0.707, the
  old "+12.5% / 4.8× / +23%" framings) is enforced by script on every artifact.
- **Privacy controls:** endorsement code + endorser names never in any file that
  touches git; Dillan's legal-name decision made explicitly before submission;
  private-relay email stays the contact unless he opts otherwise.
- **License hygiene:** the paper/website cite datasets (SST-2/GLUE, BeaverTails
  etc. if Plan E has run) without redistributing them; CC-BY-NC respected.
- **Multi-AI review** (Step 4) continues the project's documented error-catching
  pattern — the same mechanism that produced §4h.

## Honest risks

- **Reception risk:** negative results + AI-assisted authorship may draw dismissal
  or extra scrutiny. Mitigation: total transparency (it's the paper's point) and
  receipts for every claim. This risk is also the reason the work matters.
- **Endorsement stall:** arXiv may simply not happen on schedule. Mitigated by the
  pre-registered OSF/Zenodo fallback — visibility is lower, but the record exists
  and the DOI citation chain works.
- **Identity exposure:** publishing links Dillan's GitHub identity to the work.
  His call, made consciously in Step 6 — the plan surfaces it as a decision, not a
  default.
- **Stale-asset leakage:** the v2.0 website is live-ish in a public repo today;
  until Step 5 lands, anyone finding it sees retracted claims. The banner-redirect
  is deliberately scheduled first within Step 5.
- **Most likely way this wastes a week:** wordsmithing the paper in circles.
  Mitigation: the outline is locked above; Step 4's adversarial read is the single
  revision cycle; further polish only on concrete reviewer findings.

## Deliverable Dillan will see

1. **A PDF** — the preprint, with his name (or chosen pseudonym) on it, every
   number traceable, live at a public URL.
2. **The audit table** — one page proving every public number's provenance.
3. **A consistent public footprint** — README v3.0, website, Zenodo v3.0 all
   telling the same true story.
4. **Two outreach emails, ready to send** — he presses send, on his schedule.

## Effort

- **Sessions:** 5–7 AI sessions (audit 1; paper 2–3; adversarial read 1; web/Zenodo
  1; outreach prep ≤1) + Dillan's button-presses (Zenodo login, Pages enable,
  emails, name decision).
- **GPU-hours:** zero. **Downloads:** none (pandoc/typst if not present, ~50 MB).
- **Paid API spend:** zero.

### Critical Files for Implementation
- C:\Users\User\mirrorfield\WORK_MAP.md (the narrative source of truth)
- C:\Users\User\mirrorfield\ORIENTATION.md (reconciliation + outreach-numbers lock)
- C:\Users\User\mirrorfield\experiments\*.json (audit-table sources)
- consolidated-experiments → geometric-safety-extras/website/ (the stale site to rewrite)
- C:\Users\User\mirrorfield\docs\NUMBER_AUDIT.md (created by Step 1; the single source of truth thereafter)
