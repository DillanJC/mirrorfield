# Plan G: Disagreement as a Confidence Signal — does cross-model (and MoE-expert) disagreement predict wrong answers, and does it beat or add to the gate?

> **Orchestrator adversarial review (applied 2026-06-13).** Architect plan is
> strong and correctly scoped: cross-model is the PRIMARY (all models cached, fit
> 8GB one-at-a-time, zero downloads); MoE router-disagreement is a GATED secondary
> after verifying OLMoE-1B-7B (~13.5 GB) does NOT fit 8GB and is CPU/4-bit only.
> The self-consistency null (§4d/§4e) is the loud tempering prior, and the G3
> "committee-accuracy vs genuine disagreement-signal" firewall is exactly the
> confound that would otherwise fake a win. **Accepted as-is**; one emphasis added
> to the criteria: the headline delta MUST be the *paired* bootstrap of (gate +
> disagreement) − (gate alone) on the *same* items (the plan does this; making it
> explicit so no future session substitutes an unpaired comparison).

## In plain language

We have one tool that works: a "confidence gate" watching how sure a single AI is, word by word, holding back the answers it's least sure of (live AUC 0.685, catches ~28% of errors at ~15% hold-rate). This plan tests an old, intuitive idea the project kept reaching for — the "witness," the "distributed witness," the multi-AI cross-check: **if you ask several different AIs the same question, are the questions where they disagree the ones where the answer is wrong?** If so, disagreement is a second, independent warning light to add to the gate — and a second independent signal could catch mistakes the first misses.

Two honest cautions, built in from the start:
1. **A very close cousin already failed.** Asking *one* model the same question many times and watching its answers wobble ("self-consistency") looked promising once, then fell apart on a second dataset and collapsed to chance on a bigger model — partly due to an unseeded-sampler bug. So we expect this to be weak and won't get excited by one good number.
2. **A trap that looks like success but isn't.** Several models will *also* just be more accurate together. "A committee is more accurate" is real but is **not** the same claim as "disagreement points at the wrong answers." This plan separates them carefully — confusing them is exactly the self-fooling this project has done before.

The deliverable is an honest yes/no, decided by numbers fixed before we look at data: does disagreement predict wrongness, and does it **beat or add to** the gate we already trust? A clear "no" closes the biggest untested idea in the project. The primary experiment uses only models already on the PC — no internet, no paid services.

## Why this matters / what it builds on

`SALVAGE_AUDIT.md` primitive **B — "measure disagreement among components → a confidence signal"** — the biggest untested vein. The Witness Node, Distributed Witness (+2.5%, **simulated experts only, unverified**), Mandala-MoE witness, and multi-AI pipeline all reduce to *measuring disagreement among components*; their shared dead ingredient was embedding **geometry as the signal**, not the architecture.

Bar to clear: the single-model gate's **live AUC 0.685 [0.64, 0.73]** (§4k) — not a coin-flip null.

**Tempering prior (the most important sentence here):** the nearest cousin — same-model *sample* disagreement (self-consistency) — was **rejected**: real on RTE at 0.5B, failed to replicate on QNLI, its "beats the gate" edge was a seeding-bug artifact, collapsed to chance at 3B (§4d/§4e/§4f). Cross-*model* and specialized-*expert* disagreement are genuinely different (independent errors across training runs; real specialization) so it's open — but the honest prior is **weak**, and disagreement must clear 0.685, not merely chance.

## What already exists

- `mirrorfield/mcp/uncertainty.py` — the 3 gate features, `RollingGate`, `calibrated_p_correct`. (Note: `compute_self_consistency` there is text n-gram overlap — NOT the answer/log-prob disagreement this plan measures.)
- `experiments/calibrate_gate.py` — frozen `TASKS` (RTE/QNLI/SST-2), `_first_of` parser, greedy + top-5 logprob loop, held-out index replay.
- `experiments/selfconsistency_multi.py` — the structural template: per-task loop, `cv_auc`, `boot_ci`, STANDARD-vs-signal-vs-COMBINED comparison, OOM→fallback loader, and the corrected all-RNG seeding (`torch.manual_seed` + `cuda.manual_seed_all` — the §4e fix). Adapt "N samples of one model" → "one greedy pass of each of several models."
- `experiments/eval_gate_value.py` — the frozen-threshold fresh-data eval/record pattern + held-out index replay.
- `experiments/validate_rolling_gate.py` — `rolling_znorm`, `cv_auc`, `boot_ci`, shuffled-null.
- Cached: Qwen2.5-0.5B/3B-Instruct, granite-guardian-3.1-2b, all-MiniLM/mpnet/paraphrase; datasets glue/BeaverTails/JBB/ToxicChat. **No MoE cached.**
- `REAL_MOE_VALIDATION_PREP.md` — the Distributed-Witness/Mandala lineage and its never-closed "simulated experts" gap.

**MoE feasibility (verified read-only via HF API):** `allenai/OLMoE-1B-7B-0924-Instruct` = 7B total / 1B active, ~13.5 GB BF16 → **does NOT fit 8GB**; not cached; CPU or 4-bit only. `output_router_logits=True` IS supported (per-layer `(tokens, 64)` → router entropy / top-2 gap / load variance computable). `Qwen1.5-MoE-A2.7B` ~28 GB → no. **Decision: cross-model (a) = PRIMARY; MoE router-disagreement (b) = gated optional SECONDARY** (runs only if a CPU smoke test confirms sane router logits at workable speed).

## The plan, step by step

Anchor scale = Qwen2.5-3B (where the 0.685 baseline lives and where self-consistency died — the only fair comparison). Models loaded one at a time.

**Step 0 — Pre-registration commit BEFORE any data (~30 min).** `experiments/disagreement_gate/PREREGISTRATION.md` (mirror `harm_gate/PREREGISTRATION.md`): frozen success/failure/ABANDON criteria, seeds, confound-control list, honest priors. Commit on branch `plan-g-disagreement`; record the hash. No generation runs before this commit.

**Step 0b — MoE feasibility smoke (optional, ~1–2 h incl. download).** Download OLMoE-1B-7B, load on CPU (or 4-bit GPU fallback), run 5 RTE items with `output_router_logits=True`; confirm (i) per-layer `(tokens,64)` logits, (ii) <~10 s/item, (iii) non-degenerate top-8 routing. Record GO/NO-GO. NO-GO → variant (b) dropped, reason logged, plan proceeds on (a).

**Step 1 — Cross-model panel on frozen held-out items (~2–4 h).** Reuse `calibrate_gate.TASKS` (RTE+QNLI; SST-2 generated but excluded from headline, too few errors per §4h/§4i). Fresh sample seed 42, **excluding §4g/§4k calibration+eval indices** (replay shared RNG as `eval_gate_value.py` does). n ≈ 450 RTE + 500 QNLI. One greedy pass per item through a panel, loaded one at a time, cached to disk before swapping:
- **M1 = Qwen2.5-3B** (anchor; also yields the gate features + `p_correct` — the baseline).
- **M2 = Qwen2.5-0.5B** (same family, weaker → partly-independent errors).
- **M3 = a different-family ~1–3B instruct model that fits 8GB** (first choice cached; else one small permissive download — script MUST run on M1+M2 alone if absent). Different lineage = genuinely independent errors.
- **M4 = mpnet/MiniLM entailment reader** (cheap architecturally-distinct fourth voice; optional).
Parse with `_first_of`. Artifacts: `disagreement_rows.npz`, `disagreement_texts.json`. Models never co-resident.

**Step 2 — Disagreement features (CPU, seconds).** Per item: vote disagreement (minority fraction / vote entropy / pairwise rate); predictive-distribution JSD (shared-tokenizer Qwen pair only; mixed-family → vote disagreement, logged as a scope limit); anchor M1 gate features + `p_correct` (so disagreement is testable *on top of* the gate).

**Step 3 — Score, gate as the bar (CPU, seconds).** Reuse `cv_auc`/`boot_ci`: `auc_gate` (M1 features; cross-check it reproduces ≈0.685 — a built-in sanity test), `auc_disagreement`, `auc_combined`. **Two headline deltas, each paired 2000-rep bootstrap on the same items:** `delta_disagreement_over_gate` and `delta_combined_over_gate`. Per-task and pooled-with-z-norm (`rolling_znorm`).

**Step 4 — Confound separation (the anti-self-fooling step, CPU).** Measure ensemble majority-vote accuracy and each model's solo wrong-rate; test whether disagreement predicts wrongness **beyond** "the weak model was wrong" (nested logistic: gate-only vs gate+disagreement = exactly `delta_combined_over_gate`) and **beyond** "the committee is just more accurate" (panel-composition ablation: same-family 3B+0.5B vs +different-family M3).

**Step 5 — Replication before headline (~2–4 h).** Re-run Steps 1–4 with seed 1337 + a fresh disjoint sample. Citable only if it clears the bar in both runs.

**Step 6 — (optional, if 0b = GO) MoE router disagreement (~3–6 h CPU).** OLMoE on RTE+QNLI with `output_router_logits=True`; aggregate router entropy / top-2 gap / expert-load variance over positions+layers; score as Steps 3–4: does router disagreement predict OLMoE's own wrongness and beat OLMoE's own token-gate? Self-contained sub-result, own bar, own (weak) prior; NOT pooled with cross-model.

## Pre-registered success / failure criteria

Frozen at Step 0. AUC = 5-fold OOF logistic; CI = 2000-rep bootstrap; null ≥10 shuffled-label refits. Label = dataset gold. **Headline bar = the gate (0.685), not a trivial null.**

- **G1 — disagreement carries ANY signal:** SUCCESS = `auc_disagreement` ≥ 0.60, CI lower > 0.50, > max nulls, on both tasks (or pooled-z-norm), replicated. WEAK-REAL = CI lower > 0.50 but AUC < 0.60 (recorded, not headlined). NULL = CI includes 0.50.
- **G2 — the headline: disagreement BEATS or ADDS TO the gate.** SUCCESS = `delta_combined_over_gate` (paired bootstrap on the same items) CI lower > 0 in both runs (or `delta_disagreement_over_gate` CI lower > 0 for "beats"). REAL-BUT-TINY = CI lower > 0 but point < +0.02 (recorded only). NULL (prior favours this) = delta spans 0 → publishable project-closing negative for B's cross-model arm.
- **G3 — confound gate (must pass for any positive to count):** a G2 success counts as "disagreement predicts errors" only if the confound checks pass; if the signal is fully explained by ensemble accuracy or one weak model, reclassify as "committee accuracy, not a disagreement signal" — NOT cited as B support.
- **Secondary (MoE, if run):** SUCCESS = router-disagreement AUC ≥ 0.60, CI lower > 0.50, > nulls, delta over OLMoE's own token-gate CI lower > 0, replicated. NULL expected.
- **ABANDON** the cross-model arm (write the negative) if G2 = NULL across both runs, or G1 passes but G3 attributes it all to ensemble accuracy / one weak model. ABANDON / never-start the MoE arm if 0b = NO-GO or the secondary is NULL.
- **Honest null prior:** P(G2 NULL) ≈ 65–75%.

## Controls & verification

- **Circularity — target defined: NONE.** Labels = GLUE gold; disagreement features and gate never see/define them.
- **Negative controls:** ≥10 shuffled-label refits per AUC (observed > max; null mean ∈ [0.45,0.55]).
- **Seeds:** numpy + random + `torch.manual_seed` + `cuda.manual_seed_all`; replication seed 1337. Greedy decoding (the §4e sampler bug can't bite) — seed anyway.
- **Replication** before headlining (the §4d rule).
- **Named confounds:** (1) committee-accuracy — report ensemble accuracy; disagreement must predict wrongness *conditional on* the panel's predictions. (2) weak-model-is-wrong — per-model solo wrong-rates + nested gate-vs-gate+disagreement + panel-composition ablation. (3) refusal/degenerate (§4m) — report None-rate per model, re-score on parseable-by-all subset. (4) length (§4n) — disagreement must beat a length-only baseline. (5) cross-tokenizer JSD undefined — restricted to the Qwen pair. (6) sanity: M1 gate AUC must reproduce ≈0.685 on the fresh sample before trusting any delta.
- **One-model-in-VRAM discipline** (OOM→fallback loader reused).

## Honest risks

- Most likely NULL (65–75%) — pre-registered as valuable, written up plainly.
- The committee-accuracy trap (highest-value risk) — G3 + confound #1 catch it.
- Weak-model artifact mirrors §4d — panel-composition ablation guards it.
- Third model = a dependency — script runs on M1+M2 alone; same-family-only positive labelled under-powered-for-independence.
- MoE arm may be infeasible/slow/quant-perturbed — explicit GO/NO-GO; NO-GO acceptable, the "simulated experts" gap then recorded as still-open with the hardware reason.
- Small positive-class → wider CIs; anchored at the §4k scale.

## Deliverable Dillan will see

1. A plain-language yes/no/weak verdict (§4k citable-sentence style) with the self-consistency caveat alongside.
2. A results table (+ replication): per-task and pooled `auc_gate` / `auc_disagreement` / `auc_combined`, both paired deltas + CIs, null bands, and the confound block (ensemble accuracy, per-model wrong-rates, parseable-subset re-score).
3. The committed `PREREGISTRATION.md` + hash.
4. If run, the MoE sub-result closing or honestly documenting the `REAL_MOE_VALIDATION_PREP` gap.
5. A one-line `WORK_MAP`/`ORIENTATION §9` update turning "untested vein" into a settled entry.

## Effort

Pre-reg ~0.5 h; MoE smoke ~1–2 h (optional); cross-model generation + replication ~4–8 h GPU; scoring/confounds minutes (CPU); MoE arm ~3–6 h CPU (optional). Coding low-moderate (close adaptation of `selfconsistency_multi.py`). Primary path: no internet, no paid API.
