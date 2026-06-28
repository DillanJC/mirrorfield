# DRAFT NOTE — Disciplined replication of known self-monitoring failures on one small model (Qwen2.5-3B)

> **⚠️ Read this banner before the draft (added after a critical review).** An earlier
> version of this file overclaimed and has been walked back. Honest status:
> - **This is a *replication/methods note*, not a novel contribution.** The five results
>   (verbalized-vs-token confidence, sycophancy under pushback, weak small-model injection
>   resistance, a weak system-prompt defense) are all **consistent with existing
>   literature.** The value on offer is *disciplined, pre-registered, replicated* measurement
>   on a single named model — useful in a replication-poor field, **not** a new finding.
> - **"The confident blind spot" is a description, not a validated theory.** Do not treat the
>   name as more than a mnemonic for five same-model observations.
> - **Everything here is one 3B model**, behavioral, average-case. These are claims about
>   *this* model; scale-dependence is untested and several patterns are expected to differ at
>   scale (see the per-finding caveats below, esp. F5).
> - **Framing caveat:** the whole pipeline is *detector framing* (does signal X flag failure
>   Y?). It reports average-case AUCs; it does **not** answer the near-decision-boundary
>   calibration question that is the operationally interesting one. The results here are
>   *negative* detector results (the signal does **not** flag the failure), which is safer
>   than a positive overclaim — but it is still the detector lens.
> - Not for circulation; related-work is a stub (no fabricated citations).

*Internal consolidation of pre-registered findings §4t–§4x (numbers + controls in
`WORK_MAP.md` and the per-experiment `PREREGISTRATION.md` files). Author: Dillan
(DillanJC/mirrorfield).*

## Abstract
AI oversight often leans on a model's own signals — stated confidence, refusals, apparent
certainty — to decide when an answer can be trusted. On **one** small open model
(Qwen2.5-3B-Instruct, single consumer GPU), we **replicate** four such failure modes under a
strict pre-registration + placebo + two-seed discipline, plus one mitigation test. Findings
(all consistent with prior literature): the model's *internal* log-prob confidence modestly
predicts its own errors (AUC ~0.65) while its *spoken* confidence does not (AUC ~0.51); under
user pushback it abandons correct answers (~+20 points vs a re-ask control — but see the
contaminated-control caveat, F3) **without its confidence dropping to flag the change**; under
prompt injection it complies 80–96%, and the gate signal does not separate hijacked from clean
outputs; and a system-prompt ("instruction-hierarchy") mitigation helps only modestly and not
at all against a hard override **at this 3B scale** (it is known to help more at scale). The
mnemonic "confident blind spot" summarizes these same-model observations: the internal signal
is the best self-monitor available but does not *drop* in the pressured/adversarial cases. We
make **no novelty claim** — the contribution is disciplined replication on a named model and
the controls to back it, not a new result or a validated theory; all claims are about this one
model and are average-case, not near-boundary.

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

**F2 — Refusals are wording-robust; over-refusal on a boundary-adjacent benign set (§4u).**
Across 8 neutral rewordings of 50 harmful requests (×2 disjoint wrapper sets), 90% of harmful
requests are refused regardless of phrasing; 10% have *replicated* surface seams that flip
refuse↔comply. The model also refuses **29% of the JBB-*benign* set** — but **caveat:** that
set is topic-matched to the harmful behaviours (deliberately boundary-adjacent), so 29% is a
statement about *edge-case* benign requests, **not** about representative everyday traffic.

**F3 — Under pushback the model flips, and its confidence doesn't drop to flag it (§4v).** On
items it first answered correctly, user pushback flipped it away from the correct answer ~20
points more than a "double-check" control (replicated). **Two caveats the critical read
surfaced:** (a) **the control is contaminated** — "please double-check" already implies the
user thinks something is off, and its own flip rate is an elevated 22–24%; the cleanest reading
is "re-eliciting under any doubt destabilises the answer," and the "not graded by pressure
strength" claim is shaky until tested against a *neutral* re-ask (a follow-up test, §4v-b, is
running). (b) The confidence claim is the *narrow* one: on flips the internal signal does not
**drop** to flag the change (mean Δ +0.03; it drops −0.10 only when the model holds firm) —
"caves confidently" overstates it, since post-flip confidence is confidence in the *new* answer.

**F4 — Under injection the model complies, and the gate signal doesn't separate hijacked from
clean (§4w).** With an "ignore the above, output PWNED" canary hidden in a to-be-summarised
document, compliance rose from a 0% control floor to 80% (polite) and 96% (override), replicated.
The defensible signal claim is narrow: internal confidence on hijacked outputs (0.843) ≈ clean
(0.848), so the gate does **not** distinguish them — *not* the stronger "obeys as confidently as
the user."

**F5 — A system-prompt defence helps only modestly, and not against a hard override — at 3B
(§4x).** Placing the task in a trusted system prompt shaved ~12–16 points off weak/medium
injections but did essentially nothing against a determined override (95%→93%, CI includes 0);
benign task intact. **This is a scale-dependent floor, not a verdict on instruction-hierarchy**,
which is known to buy more at larger scale; a 3B model failing a hard override is close to
expected. Do not generalise "false comfort" past this model.

## 4. Synthesis: the confident blind spot
| situation | model wrong/unsafe? | internal signal catches it? |
|---|---|---|
| un-pressured hard question (F1) | sometimes | **yes, modestly** |
| social pressure / sycophancy (F3) | yes (~+20 pt, contaminated control) | **no** — signal doesn't *drop* to flag it |
| injected instruction (F4) | yes (up to 96%) | **no** — signal doesn't separate hijacked from clean |

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
