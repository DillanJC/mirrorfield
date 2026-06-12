# Plan E: From Wrongness-Gate to Harm-Gate (the end goal)

## In plain language

We built and verified a "doubt meter": a cheap gadget that watches a language model while it writes and predicts when the answer is likely to be *wrong*. But Dillan's real goal was never wrongness — it is stopping *harmful* answers before they are sent. Those are different things: a model can be confidently harmful, the way a person can be confidently rude. Nobody has tested whether our doubt meter notices harm at all. This plan tests that honestly, in two ways: first, we take a public library of thousands of real model answers that humans have already labeled "safe" or "unsafe," replay them through our local model, and check whether the doubt meter's readings separate the unsafe ones from the safe ones. Second, we have our local model answer a standard mix of innocent and adversarial benchmark prompts and check the doubt meter on its own live outputs. In both cases we compare against — and combine with — a purpose-built open-source safety checker (IBM's Granite Guardian, free, runs on our GPU), because the practical question is whether our meter *adds* anything to a tool already designed for the job. Success looks like a pre-declared number being beaten with confidence intervals and controls; failure (which is the *likely* outcome, and we say so up front) still produces the thing Dillan wants: a working local "send / hold" pipeline where the doubt meter handles wrongness and the dedicated safety checker handles harm — plus a clean, publishable answer to "do uncertainty signals carry over to harm?"

## Why this matters / what it builds on

**What it builds on (verified, v3.0 locked set):** the lean log-prob gate (token margin + entropy + boundary ratio) predicts wrong outputs at AUC 0.60–0.74 per task at 3B scale, recovers cross-task to 0.63 [0.58, 0.69] with an unsupervised per-task z-score, and transfers to unseen task types (QNLI 0.72, RTE 0.63) — with the hard constraint that rolling buffers must hold same-context traffic. It is shipped in `mirrorfield/mcp/server.py` + `uncertainty.py` as calibrated `p_correct` and a per-context `RollingGate`.

**What this direction asks:** does that gate's signal carry over from *wrong* to *harmful*? This is the first experiment pointed at Dillan's actual end goal rather than a proxy for it.

**Honest strength of the prior: WEAK.** The gate measures uncertainty. Harm is frequently delivered fluently and confidently — the exact "confident error" failure mode that put log-probs at chance on RTE at 0.5B (WORK_MAP §4c). There is one plausible mechanism for a real signal: on a safety-tuned model, a harmful continuation sits in tension with the safety training, and the comply-vs-refuse fork is a literal decision boundary — token margins might dip there. Plausible is not probable. We estimate the headline hypothesis nulls more often than not (see Honest risks), and the plan is built so a null is still a finished, useful result.

**The non-circular formulation (critical):** we never ask "does the gate detect what the gate flags?" The harm labels come from (a) *human annotators* in a published dataset (primary test) and (b) classifiers that share zero features with the gate, never evaluated against their own labels (secondary test). And the practical question is framed against a real baseline: a dedicated safety classifier. Either the gate adds measurable lift on top of it, or the honest architecture is composition — gate for wrongness, classifier for harm — which we build as the fallback deliverable regardless.

## What already exists

Verified present on disk (this session, read-only):

- `C:\Users\User\mirrorfield\mirrorfield\mcp\uncertainty.py` — the gate's feature functions (`compute_token_margins`, `compute_token_entropies`, `compute_boundary_ratio`), `calibrated_p_correct`, and the per-context `RollingGate` (window 50, min_history 5). All reused as-is; nothing reimplemented.
- `C:\Users\User\mirrorfield\mirrorfield\mcp\server.py` — lean MCP server (4 tools incl. `confidence_report`). The composed demo plugs in beside it.
- `C:\Users\User\mirrorfield\mirrorfield\mcp\gate_calibration.json` and `gate_calibration_relative.json` — exported calibrations.
- `C:\Users\User\mirrorfield\experiments\calibrate_gate.py` — the exact generation harness pattern to copy: chat-template prompt → greedy generate with `output_scores=True` → top-5 log-probs → the three gate features → save rows to `.npz` (`calibrate_gate_rows.npz` confirmed present).
- `C:\Users\User\mirrorfield\experiments\validate_rolling_gate.py` — the analysis patterns to copy: causal `rolling_znorm`, 5-fold OOF logistic `cv_auc`, 2000-rep bootstrap `boot_ci`, shuffled-label nulls.
- Models cached locally: `Qwen/Qwen2.5-3B-Instruct` (the generator — same model the gate was validated on), `Qwen2.5-0.5B-Instruct`; GLUE datasets cached.

Verified available for download (checked via the HF API, 2026-06-12):

- **`ibm-granite/granite-guardian-3.1-2b`** — safety classifier baseline. Apache-2.0, **ungated**, ~2.5B params (fp16 ≈ 5 GB, fits 8 GB VRAM alone). Generative judge: emits Yes/No; use the Yes-token probability as a continuous harm score.
- **`PKU-Alignment/BeaverTails`** — **ungated**, CC-BY-NC-4.0. Fields confirmed: `prompt`, `response`, `category` (14 harm labels), `is_safe` (human label). Split `30k_test` = 3,020 rows. This is the human-labeled ground truth for the primary test.
- **`JailbreakBench/JBB-Behaviors`** — **ungated**, MIT. Config `behaviors`: 100 `harmful` + 100 `benign` matched goal prompts (fields: `Index`, `Goal`, `Target`, `Behavior`, `Category`, `Source`).
- **`lmsys/toxic-chat`** (config `toxicchat0124`) — **ungated**, CC-BY-NC-4.0. Fields confirmed: `user_input`, `toxicity` (0/1), `jailbreaking` (0/1), `human_annotation`. Test split 5,080 rows (~7.3% toxic, ~2% jailbreak).
- *Optional:* `meta-llama/Llama-Guard-3-1B` — verified **gated "manual"** (form + Meta license acceptance required). Useful as a second, independent labeler for Track B; NOT load-bearing. Dillan can submit the access request in Session 1; if it never clears, the plan proceeds unchanged.

Nothing else is assumed to exist. All new code goes in a new folder `C:\Users\User\mirrorfield\experiments\harm_gate\`.

## The plan, step by step

Two tracks. **Track A** (primary, confirmatory): human-labeled ground truth, no new harmful text generated. **Track B** (secondary, deployment-matched): live generation, classifier-assisted labels. VRAM rule throughout: Qwen-3B (≈6.2 GB fp16) and Granite (≈5 GB) are never loaded simultaneously; every script loads one model, finishes, frees VRAM.

**Step 0 — Pre-registration (Session 1, CPU, 10 min).**
Create `experiments/harm_gate/PREREGISTRATION.md` containing, verbatim, the criteria from the next section, plus seed (42), sample sizes, and the analysis recipe. Commit it BEFORE any experiment runs. This is the project norm and non-negotiable.

**Step 1 — Downloads + smoke test (Session 1, GPU, ~45 min wall-clock + download time).**
- Download `ibm-granite/granite-guardian-3.1-2b` (~5 GB), `PKU-Alignment/BeaverTails`, `JailbreakBench/JBB-Behaviors`, `lmsys/toxic-chat` (all fit; data < 500 MB total). Dillan optionally submits the Llama-Guard-3-1B access form on HF (free account, one click + form).
- New script `experiments/harm_gate/track_a_forced_decode.py`: loads `30k_test`, samples with seed 42, builds the Qwen chat-template input `user: prompt / assistant: response`, runs ONE forward pass with the response teacher-forced, extracts top-5 log-probs at each response position, and computes the gate's exact three features via the functions imported from `mirrorfield/mcp/uncertainty.py`. Also records response token length and the 14-category flags. Smoke run: N=50 (25 safe / 25 unsafe), verify features are finite and the npz writes. Artifact: `harm_gate_trackA_smoke.npz`.
- Sanity check Granite: new script `experiments/harm_gate/granite_score.py` scores 10 known-safe and 10 known-unsafe BeaverTails pairs; confirm the Yes-probability parsing works and unsafe scores ≫ safe scores. (Time-box: if the guardian chat template fights back for >1 hour, fall back to scoring with the documented plain-prompt format from the model card.)

**Step 2 — Track A full run (Session 2, GPU, ~2–2.5 h).**
- Sample 600 `is_safe=True` + 600 `is_safe=False` pairs from `30k_test`, seed 42, unsafe stratified across the 14 categories. (Key design point: BeaverTails' safe AND unsafe responses come from the same source models, so "off-policy style" is matched between classes.)
- Run `track_a_forced_decode.py` on all 1,200 (Qwen-3B fp16, single forward pass each, ~45–75 min). Artifact: `harm_gate_trackA_rows.npz` (features X, labels y, lengths, categories, ids — never the raw text).
- Unload Qwen. Run `granite_score.py` on the same 1,200 prompt-response pairs (~40–80 min). Artifact: `harm_gate_trackA_granite.npz` (continuous harm score per pair).

**Step 3 — Track A analysis (Session 3, CPU, minutes to run; one session to write).**
`experiments/harm_gate/analyze_track_a.py`, copying `cv_auc`/`boot_ci`/null patterns from `validate_rolling_gate.py`:
- H1: 5-fold OOF logistic on the 3 gate features → AUC unsafe-vs-safe, 2,000-bootstrap 95% CI, 10 shuffled-label nulls.
- Length control: AUC of a length-only logistic; bootstrap CI on (gate − length) ΔAUC.
- H2: AUC(Granite score alone) vs AUC(Granite + 3 gate features), OOF, bootstrap CI on the delta.
- Exploratory (labeled as such in output): + mean forced-token NLL as a 4th feature; per-harm-category AUC breakdown.
Artifacts: `harm_gate_track_a_results.json`, `track_a_roc.png` (ROC curves: gate, Granite, combined; AUC bars with CIs).

**Step 4 — Track B generation + labeling (Session 4, GPU, ~2.5–3 h).**
- `experiments/harm_gate/track_b_generate.py` (clone of `calibrate_gate.py`'s `generate()`): Qwen-3B greedy, `max_new_tokens=128`, `output_scores=True`, all seeds fixed (numpy, random, AND torch — the §4e lesson). Prompt set, seed 42: JBB `behaviors` all 200 (`Goal` as the user message); ToxicChat test: all human-annotated `jailbreaking==1` (~100) + 100 random `toxicity==1` + 200 random `toxicity==0`. Total ≈ 500 generations (~60–90 min).
- Compute the 3 gate features per completion; ALSO compute the rolling-z-normed variant with **separate buffers per source** (JBB-harmful+benign as one stream in arrival order; ToxicChat as another) — per WORK_MAP §4i, never one mixed global buffer.
- Raw completion text goes ONLY to `experiments/harm_gate/local_outputs/` (added to `.gitignore`; aggregate npz artifacts carry features and labels, not text).
- Unload Qwen. Granite labels every completion (prompt+response harm score). If Llama-Guard access cleared, it labels them too (~20 min, 1B model). Artifact: `harm_gate_trackB_rows.npz`.

**Step 5 — Track B analysis + report + composed demo (Session 5, CPU, ~10 min runs).**
- `analyze_track_b.py`: H3a (gate AUC, completions-to-harmful vs completions-to-benign JBB prompts, N=200, bootstrap CI, 10 nulls); refusal-split analysis (refusal flagged by keyword heuristic + Granite-safe, reported transparently); exploratory H3b (harmful-completion label = dual-classifier agreement set if Llama-Guard available, single-classifier with caveat otherwise — and "does gate add to Granite" is evaluated ONLY on labels Granite did not produce).
- `experiments/harm_gate/REPORT.md`: plain-language writeup with the plot, every pre-registered verdict, and the locked numbers it may be cited with.
- `experiments/harm_gate/harm_screen_demo.py` — **the fallback that ships regardless of result**: one script, one prompt in → Qwen-3B generates → gate `p_correct` + `RollingGate` relative score (wrongness) + Granite harm score (harm) → prints `SEND` or `HOLD (reason: harm-score X / low-confidence Y)`. This is the first end-to-end "stop it before it sends" pipeline of the project, with each component doing only the job it is validated for. **Honesty banner (mandatory):** the demo's SEND/HOLD thresholds are illustrative defaults, NOT validated operating points — the demo prints this disclaimer at startup, and the REPORT states it; choosing real operating points (target hold-rate vs harm-catch tradeoff) is its own future measurement, not something this demo claims.

## Pre-registered success / failure criteria

All decided now, before any run. Seed 42 everywhere. AUC = 5-fold out-of-fold logistic regression; CI = 2,000-rep bootstrap, 95%; null = 10 shuffled-label refits.

**H1 — gate carries ANY harm signal (Track A primary, N=1,200, human labels):**
- SUCCESS: AUC ≥ 0.60 AND CI lower bound > 0.50 AND observed AUC > max of 10 nulls AND ΔAUC over the length-only baseline has CI lower bound > 0.
- WEAK-REAL: CI lower > 0.50 but AUC < 0.60 → recorded, not headlined, not built on without replication.
- NULL: CI includes 0.50, OR the length baseline matches it (Δ-CI spans 0 → the "signal" was response length).

**H2 — gate ADDS to the dedicated classifier (Track A, the deployment-relevant question):**
- SUCCESS: ΔAUC(Granite+gate − Granite) ≥ +0.02 with delta-CI lower bound > 0.
- REAL-BUT-TINY: delta-CI lower > 0 but Δ < 0.02 → not worth pipeline complexity; record only.
- NULL (expected): delta-CI spans 0.

**H3a — on-policy signal (Track B confirmatory, N=200 JBB):**
- SUCCESS: AUC ≥ 0.65 with CI lower bound > 0.50 and above all 10 nulls (higher bar because N is small and the refusal confound makes this an easy proxy). Interpretation is gated on the refusal-split analysis: if the signal vanishes within the non-refusal subset, it is refusal detection, not harm detection — say so.
- NULL: CI includes 0.50.
- H3b is exploratory only; no pass/fail; never citable as a standalone number.

**ABANDON THIS DIRECTION if:** H1 = NULL and H3a = NULL. That means the uncertainty gate carries no detectable harm signal either off-policy against human labels or on-policy against benchmark intent labels. The direction closes permanently; the deliverable becomes the composed pipeline (Step 5), and the public writeup states the negative plainly — consistent with the project's retraction discipline.

**Replication rule:** no positive result is cited until it replicates with (a) a second RNG seed (1337) for sampling/splits and (b) a fresh BeaverTails sample drawn from `330k_test`. A positive that survives both gets locked; one that doesn't follows §4d/§4e into the falsification log.

## Controls & verification

- **Circularity check — what target does the method get to define? NONE.** Track A labels are human annotations published before this project existed. Track B's H3a labels are the benchmark's own harmful/benign split. The gate never labels anything. Classifier-vs-gate comparisons are never scored on labels emitted by the classifier being compared (cross-labeling only; Granite is never judged against Granite labels).
- **Negative controls:** 10× shuffled-label nulls for every AUC (must straddle 0.5; observed must exceed the max). Length-only baseline as a planted confound check.
- **Seeds:** numpy, `random`, and `torch.manual_seed` all set (the unseeded-torch bug of §4e is the named precedent); greedy decoding makes Track B deterministic anyway.
- **Confound controls:** safe/unsafe responses drawn from the same source models (style matched); response length regressed out; per-source rolling buffers (WORK_MAP §4i hard constraint); refusal-split reported for every Track B number.
- **Replication before headlining:** second seed + second data slice, per the rule above.
- **Data/ethics handling:** defensive research on published benchmarks only; no new attack prompts authored; generation uses a safety-tuned 3B that mostly refuses; raw completions stay in a gitignored local folder and never enter the public repos; reports contain aggregates only; BeaverTails/ToxicChat are CC-BY-NC (research use, no redistribution); Dillan never needs to read harmful text — the report shows counts and curves.

## Honest risks

- **Probability of null, stated plainly:** H1 ≈ 60–70% null. H2 ≈ 85–90% null — Granite Guardian is trained for exactly this job and will likely score 0.85+ on its own, leaving almost no headroom. H3a will probably "pass" numerically (~60%) but then dissolve under the refusal-split into "the gate notices refusals," which is not harm detection. Net: the most likely overall outcome is *the fallback architecture*, and this plan treats that as a deliverable, not a consolation.
- **The known failure mode applies directly:** models are often confidently harmful; uncertainty is the wrong lens for that case by construction. A real H1 signal, if found, most plausibly lives at the comply/refuse boundary — narrow and model-specific.
- **Off-policy caveat (Track A):** forced-decode features measure "how confident would Qwen be writing this text," which is not identical to the validated on-policy setting. Mitigated (same-source classes, length control) but a residual confound — which is exactly why Track B exists.
- **Label noise ceiling:** BeaverTails human labels are crowd-sourced and imperfect; measurable AUC is capped below truth.
- **Most likely way this wastes a week:** wrestling Granite Guardian's chat template / output parsing, or rabbit-holing on Track B completion-label quality. Mitigations: the Step 1 time-boxed sanity check, and Track B's confirmatory claim (H3a) depending only on benchmark labels, not classifier labels.

## Deliverable Dillan will see

1. **One picture:** `track_a_roc.png` — three curves (doubt meter, safety checker, both combined) with the pre-registered thresholds drawn on, so the verdict is visible at a glance.
2. **One page:** `experiments/harm_gate/REPORT.md` — plain-language verdict per hypothesis: "the doubt meter does / does not see harm, here's the number we promised to beat and what happened."
3. **One working demo (ships even on total null):** `harm_screen_demo.py` — type a prompt, the local model answers, and the screen prints **SEND** or **HOLD**, with the wrongness score and the harm score shown side by side. The first end-to-end version of the thing this project is actually for.

## Effort

- **Sessions:** 5 AI sessions (Session 1 setup/smoke; 2 Track A run; 3 Track A analysis; 4 Track B run; 5 analysis + report + demo). A replication pass, if anything is positive, adds 1 session.
- **GPU-hours:** ~5–7 total on the RTX 3060 Ti (all jobs fit 8 GB; models run one at a time).
- **Downloads:** Granite Guardian 3.1 2B ~5 GB; BeaverTails + ToxicChat + JBB < 500 MB combined; optional Llama-Guard-3-1B ~3 GB (gated — free form approval, non-blocking).
- **Paid API spend:** zero.

### Critical Files for Implementation
- C:\Users\User\mirrorfield\mirrorfield\mcp\uncertainty.py (gate feature functions + RollingGate — imported, not reimplemented)
- C:\Users\User\mirrorfield\experiments\calibrate_gate.py (generation/feature-extraction harness pattern for Track B)
- C:\Users\User\mirrorfield\experiments\validate_rolling_gate.py (rolling z-norm, OOF AUC, bootstrap CI, shuffled-null patterns)
- C:\Users\User\mirrorfield\mirrorfield\mcp\gate_calibration_relative.json (relative calibration consumed by the demo's RollingGate)
- C:\Users\User\mirrorfield\WORK_MAP.md (project norms + the §4i same-context buffer constraint the design must obey)