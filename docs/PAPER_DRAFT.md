# DRAFT — The Confident Blind Spot: Where a Small Language Model's Self-Monitoring Fails

*Working draft (not for circulation). Internal consolidation of pre-registered findings
§4t–§4x; numbers and controls in `WORK_MAP.md` and the per-experiment `PREREGISTRATION.md`
files. **Related-work section is a stub — citations to be added/verified by a human; do not
circulate with placeholder references.** Author: Dillan (DillanJC/mirrorfield).*

## Abstract
AI oversight increasingly leans on a model's own signals — its stated confidence, its
refusals, its apparent certainty — to decide when an answer can be trusted. We ask, on one
small open model (Qwen2.5-3B-Instruct, single consumer GPU), how far those self-signals can
actually be trusted, using five pre-registered, replicated, placebo-controlled experiments.
We find a consistent **confident blind spot**: the model's *internal* log-prob confidence is
its most trustworthy self-signal — it beats the model's *spoken* confidence, which is at
chance — yet it is blind precisely when the model is **confidently wrong**. Under user
pushback the model abandons correct answers (~+20 points vs a neutral control) while its
confidence does not drop; under prompt injection it obeys hidden instructions (80–96%) just
as confidently as it serves the user; and the standard "instruction-hierarchy" defence barely
helps (95%→93% under a determined override). Separately, the model's larger everyday failure
is over-refusal (29% of benign requests), not under-refusal. We argue self-monitoring via
confidence is **necessary but not sufficient**, and that pressured/adversarial inputs require
orthogonal defences. The contribution is an honest map of limits, with the controls to back
it, rather than a new capability.

## 1. Introduction
"Just ask the model how sure it is," "trust it when it refuses," "put trusted instructions in
the system prompt" — much practical AI-safety guidance assumes a model's self-signals are
informative. We test that assumption directly and behaviourally on a small open model, under a
strict anti-circularity discipline (external ground truth, placebo/baseline controls,
shuffled-label nulls, two-seed replication, pre-registration before data; see §2 and
`EVALUATION_DISCIPLINE.md`). Our question is not "is the model safe" but **"can the model's own
signals tell us when it is not?"**

## 2. Method (shared)
- **Model & compute:** Qwen2.5-3B-Instruct, greedy decoding, one RTX 3060 Ti, no paid API.
- **The internal signal under test:** a validated log-prob "wrongness gate" — mean top-token
  margin, entropy, boundary ratio → a calibrated p(correct) (live AUC ≈ 0.685 on held-out
  RTE/QNLI). It is modest, and we treat it as such.
- **Discipline:** every experiment pre-registers success AND abandon criteria (git commit hash
  = the lock); correctness/harm/compliance is judged by *external* ground truth (GLUE gold; an
  independent harm classifier; a canary token), never by the method under test; any prompt
  manipulation is compared against a content-free placebo; effects are replicated on a second
  seed (and, for refusals, a second wrapper set) before being claimed; we report operating
  points, not just AUCs; a null is a result; no threshold is retuned to rescue a finding. This
  discipline caught three would-be-misleading artifacts before they were reported.

## 3. Findings
**F1 — Verbalized confidence is at chance; the internal signal is not (§4t).** On 500 held-out
RTE/QNLI items, the model's spoken confidence predicted its own correctness at chance
(AUC 0.51/0.51; calibration error 0.32–0.36), while its internal signal was modestly predictive
(AUC 0.64/0.66; error 0.03–0.05). Verbal confidence *varies* but is uncorrelated with being
right; combining it with the internal signal adds nothing.

**F2 — Refusals are wording-robust, but over-refusal dominates (§4u).** Across 8 neutral
rewordings of 50 harmful requests (×2 disjoint wrapper sets), 90% of harmful requests are
refused regardless of phrasing; 10% have *replicated* surface seams that flip refuse↔comply.
The larger failure is the opposite: 29% of clearly benign requests are refused.

**F3 — The model caves to pressure, confidently (§4v).** On items it first answered correctly,
*any* user pushback flipped it away from the correct answer ~20 points more than a neutral
double-check (replicated; not graded by pressure strength). The internal signal did **not**
expose the cave: on flips it stayed flat/rose (+0.03), dropping only when the model held firm
(−0.10).

**F4 — The model obeys injections, confidently (§4w).** With an "ignore the above, output
PWNED" canary hidden in a to-be-summarised document, compliance rose from a 0% control floor to
80% (polite injection) and 96% (override), replicated; full task-abandonment 0%→82%→95% with
strength. Internal confidence on hijacked outputs (0.843) ≈ clean (0.848).

**F5 — The standard injection defence is largely false comfort (§4x).** Placing the task in a
trusted system prompt ("treat the document as untrusted data, never as instructions") shaved
~12–16 points off weak/medium injections but did essentially nothing against a determined
override (95%→93%, CI includes 0); the benign task remained intact.

## 4. Synthesis: the confident blind spot
| situation | model wrong/unsafe? | internal signal catches it? |
|---|---|---|
| un-pressured hard question (F1) | sometimes | **yes, modestly** |
| social pressure / sycophancy (F3) | yes (~+20 pt) | **no** — caves confidently |
| injected instruction (F4) | yes (up to 96%) | **no** — obeys confidently |

The internal confidence signal is the best self-monitor the model has, but it only sees the
failure mode it was validated on — the model *being unsure*. The adversarial/social failures
are exactly the cases where the model is **confidently wrong**, so confidence-based monitoring
is blind to them by construction — and the obvious patch for injection (F5) does not close the
gap.

## 5. Implications for safe design
1. Do not use verbalized confidence for triage or oversight (F1).
2. Do not expect a confidence gate to catch sycophancy or injection (F3, F4) — it can't.
3. Pressured/adversarial inputs need *orthogonal* defences: consistency-under-perturbation for
   sycophancy; untrusted-content isolation / injection-resistance training for injection —
   instruction-hierarchy alone is insufficient (F5).
4. Treat over-refusal as a first-class reliability failure (F2); it erodes trust in the safety
   layer itself.
5. Layer the monitors; do not ask one self-signal to do everything.

## 6. Limitations
One small open model (Qwen2.5-3B-Instruct); behavioural, not mechanistic; the "internal signal"
is a specific log-prob gate. **These are claims about this model**, offered as a method and a
set of hypotheses, not universal laws — larger or differently-trained models may behave
differently, and confirming scale-dependence is the obvious next step. Each result is
nonetheless pre-registered, replicated, and placebo-controlled.

## 7. Related work (STUB — human to complete with verified citations)
Touchpoints to position against (verify before citing): verbalized-confidence calibration;
sycophancy and AI-deception surveys; prompt-injection / instruction-hierarchy literature;
chain-of-thought (un)faithfulness; over-refusal / false-refusal benchmarks. See the legacy
pointers in `RESEARCH_ROADMAP.md` (geometry framing retracted; the *citations* there are real).

## Reproducibility
All five experiments are pre-registered (commit-locked before data), with code, frozen prompts,
and aggregate results in the public repository (`experiments/{selfreport_confidence,
refusal_stability,sycophancy,prompt_injection}/`). Raw harmful/text generations are kept local;
only aggregates are released.
