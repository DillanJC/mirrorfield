# CLAUDE.md — mirrorfield operating rules (lean; every line here has bitten before)

## Hard stops (auto mode especially)
- **Nothing public.** No outreach, publishing, or sharing. Drafts only; sharing is Dillan's call.
- **Nothing destructive or irreversible.** Raw harmful/sensitive generations only in gitignored `local_outputs/`; aggregates only in git.
- **Never CONCLUDE a result holds.** Surface candidates; Dillan concludes, after the run.
- **No new GPU generation without telling Dillan first.** CPU-only analysis of already-saved data is allowed if pre-registered.
- Pre-regs that require GPU runs are drafted, marked NOT LOCKED, and wait for Dillan.

## The discipline (short form — full: `experiments/EVALUATION_DISCIPLINE.md`)
- Pre-register success AND abandon criteria; commit before data (hash = lock). No retuning to rescue a result.
- External ground truth; the method never grades itself. Every detector claim gets a circularity check: does the target share an assumption with the method? (The 0.947 failure shape.)
- Placebo/baseline beside every manipulation; shuffled-label nulls; replicate (2 seeds) before claiming; report operating points, not just AUCs; a null is a result.
- **"On this model" lives inside every sentence that could generalize.** The recurring failure is narrow→universal at the CONSOLIDATION layer (synthesis docs, tables, summaries), not in the experiments. Reconcile every number to `WORK_MAP.md` / the results JSON before it moves between documents — if two docs disagree, the repo wins; stop and reconcile.
- Mechanism stories stay labeled **hypothesis** until tested. Tails under min-n: reported, not interpreted.

## Auto-mode loop
`PLAN.md` (intent) → small reversible unit → fresh verification → `AUTO_LOG.md` entry **including one line on how the result could be fooling itself** (no line, no result) → commit. Keep units small enough that Dillan can audit the whole session afterward.

## Environment (bitten before)
- 8 GB VRAM ceiling (RTX 3060 Ti): ~3B comfortable, one big model at a time.
- A stale editable install `geometric_safety_features` can shadow `import mirrorfield` with an old copy — `pip uninstall geometric_safety_features` so this repo wins.
- Session teardowns kill long runs: harnesses use resumable JSONL checkpoints; keep it that way.
- The canonical repo is `C:\Users\User\mirrorfield`. The old `geometric_safety_features-Experiment` folder is legacy — never commit there.

## State pointers
Resume: `HANDOFF.md`. Full log: `WORK_MAP.md` (§4a–§4y). Forum artifact: `docs/METHODS_NOTE.md` (draft, Dillan's edit). Leads: `docs/NOVEL_SAFETY_DIRECTIONS.md`, `docs/FRINGE_LEADS.md`.
