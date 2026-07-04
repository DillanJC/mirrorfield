# The conceptual-circularity auditor — one-page design sketch (collaboration invite)

*DRAFT 2026-07-04 (auto session), for the forum conversation. Status: parked build, not
started — this page exists so "would you help poke holes in it?" has something concrete
to point at. The problem statement is paid for: this project lost its headline result
(AUC 0.947 → ~0.47) to exactly the failure class described below.*

## The gap

Code-level leakage tools catch *mechanical* leakage — test rows in the training set,
target columns among the features. Nothing catches **conceptual** circularity: the
target and the method sharing an *assumption*, so the evaluation measures agreement
with a premise rather than performance on the world. Examples of the class:

- poison *defined* by a geometric criterion, then *detected* by geometric features
  (this project's 0.947);
- a training loss that optimizes the very metric used to declare the training a success
  (this project's Sati result);
- a "harmfulness detector" evaluated on data labeled by a rubric derived from the
  detector's own feature set;
- an LLM judge scoring outputs for a property the judge was prompted to define.

The class is invisible to train/test splits — the leak is in the *construct*, upstream
of any data.

## What the tool would actually do (three tiers, buildable in order)

1. **Structured interrogation (cheap, buildable now).** A guided questionnaire over an
   experiment's artifacts (pre-registration, labeling procedure, feature list, judge
   prompts) that forces explicit answers to: *Where does the target's definition come
   from? Write it without using any term from the method's feature vocabulary. Who
   labeled the data, seeing what?* Output: a provenance graph of target ← definitions ←
   assumptions, with shared nodes flagged. The value is the forcing function, not
   intelligence.
2. **LLM-assisted overlap detection (the research bet).** An independent model instance
   reads the target definition and the method description *separately*, extracts each
   one's load-bearing assumptions as structured claims, and flags semantic overlap —
   the automated version of the external cold read that caught this project's
   contaminated control. Key design constraint learned here: the auditor instance must
   be **stake-free** (fresh context, no access to the experiment's hoped-for
   conclusion), or it inherits the loop it audits.
3. **Adversarial construct probes (aspirational).** For a flagged overlap, auto-generate
   the *honest baseline* that breaks the tie — the analog of "vary the poison trigger"
   that collapsed 0.947 — and estimate the claim's survival probability before anyone
   runs it.

## Failure modes of the auditor itself (named before anyone builds it)

- **It grades its own homework** if the auditor shares training/prompting with the
  system under audit — tier-2's stake-free constraint is load-bearing and fragile.
- **Checklist theater:** tier-1 degenerates into box-ticking unless the output is a
  provenance *artifact* someone else can inspect, not a yes/no.
- **False confidence:** a "no circularity found" stamp is worth less than nothing if
  the tool's recall is unknown — it needs a benchmark of *known* circular claims
  (this repo contributes ~6 documented specimens; a community corpus is the real ask).

## The ask (what collaboration looks like)

- More **specimens**: documented cases of conceptually-circular claims (retracted or
  caught pre-publication) to build the recall benchmark.
- Theory help: is there a formal object here (the provenance graph's shared-assumption
  test) that connects to existing work on construct validity / measurement theory —
  or to SLT-style analyses of what an evaluation can even distinguish?
- A second builder. One person built the discipline; a tool needs adversaries.

*Contact: Dillan (DillanJC/mirrorfield). The receipts for every claim above:
`WORK_MAP.md` §4–§4z, `docs/SAFETY_CLAIM_SMELL_TEST.md`.*
