# GAP_PLAN — operating through the low-access period

*Written 2026-07-04, in the last days of Fable access. Purpose: Dillan's AI access is
about to drop (frontier-model credits ending; local model + cheaper/free tiers remain).
This file tells ANY future session — cheap model, free tier, local model, or a returned
frontier model — what to do, what NOT to do, and what waits. Read `CLAUDE.md` first;
its hard stops bind every session regardless of model quality.*

## The principle

Frontier-model time was spent on **design**; the gap is for **execution**. Everything
below marked EXECUTE has its criteria already locked or drafted — a modest model
following the spec faithfully beats a clever model improvising. Everything marked WAIT
genuinely needs top-tier judgment; doing it badly is worse than not doing it.

## What ANY session can do (EXECUTE — specs exist)

1. **Run locked pre-registrations exactly as written.** Ready or near-ready:
   `experiments/refit/PREREGISTRATION_DRAFT.md` (the calibration refit — Dillan must
   read + commit-lock before any run) and `experiments/selfreport_validity/`
   (X1, same). Rules: follow the doc to the letter; no criteria changes after data;
   verdicts are candidates until Dillan concludes; GPU runs need Dillan present.
2. **Plan J** (`plans/J-second-model-transfer.md`): finalize with Dillan, lock, run.
   The harness changes are config-level; the analysis code path already exists.
3. **Repo maintenance:** keep tests green (`pytest tests/`), keep `local_outputs/`
   gitignored, reconcile any number that moves between documents (the repo JSONs win).
4. **Curriculum delivery:** teach one lesson at a time from
   `docs/CURRICULUM.md` (Dillan's plain-language course) — the lesson plans specify
   exactly what to cover; don't improvise past them.
5. **Passion of the Fight:** build strictly from the design doc (see the GDD in the
   game's repo/folder when created) — small units, playable artifact every session,
   Dillan approves each mechanic before the next.
6. **Prose/formatting polish** on drafts Dillan asks about; NEVER change a number or a
   scoped claim while polishing (the consolidation-layer failure lives exactly there).

## What WAITS for a frontier model (or Dillan + care)

- New pre-registration **design** (fresh experiments, new verdict rules).
- Audits and hostile reviews (the §4y circularity audit is the quality bar).
- Any consolidation-layer writing: synthesis docs, abstracts, outreach text.
- Anything touching the retraction narrative or public claims.
- The conceptual-circularity auditor build (`docs/CIRCULARITY_AUDITOR_SKETCH.md`) —
  parked for collaboration, not for a cheap model to attempt alone.

## The access ladder (cheapest first)

1. **Local (free, private, always on):** LM Studio is installed; model:
   Qwen2.5-7B-Instruct (Q4_K_M) in `C:\Users\User\.lmstudio\models\`. Good for: daily
   questions, drafts, brainstorming, game writing, rubber-ducking. NOT for: verdicts,
   numbers, safety claims (it's a 7B — treat its outputs like a bright intern's).
   Plain-language guide: `C:\Users\User\Downloads\LOCAL_AI_GUIDE.md`.
2. **Free/cheap Claude tiers + cheaper Claude Code models:** execution work from this
   file, lesson delivery, code from specs.
3. **Frontier access (when it returns):** the WAIT list above, in this order:
   (a) whatever the forum surfaced, (b) lock+run the refit, (c) Plan J, (d) the auditor.

## Standing state (2026-07-04)

Repo pushed and live (`3ed9925`). Forum 7–8 July: kit ready (`docs/FORUM_KIT.md`),
note final (`docs/METHODS_NOTE.md`), Dillan's remaining items: rehearse aloud, Zenodo
DOI (one-line commit when found), depth-map read. Backups: Google Drive zip
(2026-07-04) + push. Old experiment folder: archive AFTER forum, with memory migration.
PC: NVMe firmware update + RAM test still outstanding — do the week after the forum,
not before. Grants: deferred by Dillan's explicit choice — do not re-pitch.
