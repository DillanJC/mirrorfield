# The safety-claim smell test — 12 questions before you believe a result

*One page, operational. Grown from this project's retractions (each question has drawn
blood here — the right column is the receipt). Apply to your own results first; they
are the ones you least want to test. DRAFT 2026-07-04 (auto session), for Dillan's edit.*

| # | Question | The receipt from this repo |
|---|---|---|
| 1 | **Who provides the ground truth — could the method have shaped it?** | The 0.947 detector: poison *defined* by geometry, *found* by geometry → chance under an honest baseline (v3.0 retraction). |
| 2 | **Were success AND abandonment criteria written and committed before data?** | Every §4t–§4z result carries its pre-reg lock hash; the one prior-miss (§4y Opt-1 slope) was reported as it fell because the verdict rules were locked. |
| 3 | **Is there a placebo / content-free baseline next to the manipulation?** | The confidence-contagion headline (§4q) died against a neutral-prefix placebo — the effect was "any prefix," not the idea. |
| 4 | **Does the pipeline stay silent on shuffled labels?** | Standard null in every harness; a pipeline that "finds" shuffled signal is measuring itself. |
| 5 | **Did it replicate on a second seed / sample / wrapper set?** | Every single-seed positive this project celebrated later shrank or died (§4d, §4o burns). |
| 6 | **Is the control itself clean?** | The sycophancy placebo implied doubt; the clean re-ask control *doubled* the effect (§4v) — testing the critique beat conceding it. |
| 7 | **Does the claim's scope live inside the sentence ("on this model")?** | Both project-level failures were narrow→universal leaks at the consolidation layer, opposite signs (0.947 positive; "confidence can't catch sycophancy" negative). |
| 8 | **Do the numbers in the summary match the logged results, today?** | A corrected experiment's OLD number survived in a synthesis table for weeks (+20pt sycophancy) — summaries drift even when experiments don't. |
| 9 | **Is an average hiding a decision-relevant region?** | Aggregate ECE 0.03 concealed +0.22 overconfidence exactly where the model is most wrong (§4y); the score's own axis was too compressed to show it. |
| 10 | **"Does something" ≠ "is useful": what's the operating point?** | The gate's honest form is "catches ~28% of errors at ~15% abstention" — an AUC alone would flatter it. |
| 11 | **Is a mechanism story being treated as a finding?** | "Fit where dense, extrapolates into the sparse tail" stays labeled HYPOTHESIS through two consistent results (§4z) — consistent-with is not confirmed. |
| 12 | **Who reviewed this with no stake in it?** | An independent cold read caught both the contaminated control and the inflated synthesis framing that the in-session discipline missed. |

**Scoring, bluntly:** any "no" on 1–5 means you do not have a result yet — you have a
lead. A "no" on 6–12 means the writeup will overclaim even if the experiment didn't.

*Longer form: `FIELD_GUIDE_DRAFT.md` · full method: `experiments/EVALUATION_DISCIPLINE.md`.*
