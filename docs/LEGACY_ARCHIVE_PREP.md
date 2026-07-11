# Legacy folder archive prep — making it a one-word decision

*2026-07-12, per FINAL_STRETCH_BRIEF §2.5. Subject: `C:\Users\User\geometric_safety_features-Experiment`.
Nothing was committed or changed there (per the brief's rule); this is inventory + a
recommendation Dillan can approve with one word.*

## What it is
The pre-retraction (geometry-era) repo. Last commit `bf905fc` (v2.0 era — "recursive
self-learning with geometric feedback"). ~116 MB excluding git. **116 uncommitted
files** in the working tree: the June geometry-era experiment sprawl (escape vectors,
H4 benchmark, mandala-MoE docs, tmp PDF-scraper scripts, result JSONs/PNGs) plus a few
later additions (the redirect `CLAUDE.md`, `.claude/launch.json` from the TORN preview).

## What live work still needs from it
- **Nothing imports from it** (the pip shadow was uninstalled 2026-07-04; mirrorfield
  venv/scripts resolve to the canonical repo).
- **The retraction record does NOT depend on it**: the v3.0 falsification and its
  reproduction live in the canonical repo; this folder holds the *unpublished* tail of
  the geometry era (leads that were never claimed).
- **Two things anchored here operationally**: this Claude Code session's project
  identity (memory was already mirrored to the mirrorfield project path on 07-11 —
  future sessions should open in `C:\Users\User\mirrorfield`), and the nightly backup's
  third robocopy pair (harmless either way; path can stay).

## What the retraction ethic says to KEEP
The uncommitted geometry-era files are part of the honest history (the "everything we
tried" record) even though never published. They cost 116 MB. Keep them, frozen.

## Recommendation (the one-word decision)
**"archive"** = I (any session) will: (1) commit the untracked files in that repo to a
single clearly-labeled `archive: geometry-era working tree, frozen post-retraction`
commit — LOCAL only, never pushed; (2) rename the folder to
`geometric_safety_features-ARCHIVED`; (3) update the nightly-backup path and the
redirect CLAUDE.md; (4) start all future sessions from `C:\Users\User\mirrorfield`.
Rollback: rename it back — nothing is deleted.
**"keep"** = leave everything exactly as is; the redirect CLAUDE.md already guards the
main hazard (sessions anchoring there by accident).

Either word is fine. The dangerous parts were already de-fanged (pip shadow gone,
redirect in place, backups fixed); this is tidiness, not safety.
