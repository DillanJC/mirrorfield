# Forum walk-in kit — Gradient Institute forum, 7–8 July 2026

*DRAFT for Dillan to rehearse, cut, and make his own. Written registers below are
suggestions; the voice has to be yours or it won't survive a follow-up question.
Every number here reconciles to `WORK_MAP.md` / `docs/METHODS_NOTE.md`.*

## The 60-second story (spoken; ~150 words)

> I'm an independent researcher — no lab, no credentials, one consumer GPU, and AI does
> my implementation. My project's original headline was a poisoning detector at AUC 0.947.
> I retracted it myself: the target and the method shared an assumption — it was circular.
> What survived the retraction was deliberately modest: a log-prob uncertainty gate,
> AUC about 0.69, cleanly calibrated *on average*.
>
> The part I'd stand behind is what happened next. I turned the same falsification
> discipline on my own surviving result: does its calibration hold near the decision
> boundary, where it actually earns its keep? It doesn't. In the most-torn region the
> gate says 79% and is right about 55% of the time — replicated, audited, and completely
> invisible to the aggregate calibration number.
>
> So what I have to offer isn't the gate. It's the apparatus that keeps catching my own
> artifacts before they ship — and I'd like help pressure-testing it.

## The one-paragraph orientation (written; for handing over with the repo link)

> Mirrorfield is a solo, small-compute AI-safety project whose main output is a
> falsification discipline with receipts: pre-registered success *and abandon* criteria
> commit-locked before data, external ground truth, placebo controls, shuffled-label
> nulls, and two-seed replication. The discipline retracted the project's own headline
> result (a circular AUC 0.947) and later caught three further would-be-misleading
> artifacts before they were reported. Its one surviving positive result — a modest
> log-prob uncertainty gate (AUC 0.685, aggregate calibration error 0.03–0.05, on
> Qwen2.5-3B) — was then itself stress-tested: boundary-stratified calibration shows the
> gate is overconfident by ~0.22 exactly in the low-margin region where the model is most
> often wrong, a failure the aggregate metric masked (`docs/METHODS_NOTE.md`). Everything
> is on one 3B model and one task family, and says so, sentence by sentence.

## The one-sentence claim (if you get ten seconds)

> On this 3B model, an uncertainty gate that passes aggregate calibration (ECE 0.03) is
> overconfident by ~0.22 exactly in the region where the model is most often wrong —
> replicated on two seeds, and invisible to the aggregate metric.

## The Murfet / Timaeus question ([FILL] resolved — three options, pick one)

1. **The direct one:** "My calibrator seems to fail by extrapolating into a sparse
   low-margin tail — it's well-fit where data is dense and flat-lines where it's thin.
   Does singular learning theory / developmental interpretability say anything about
   *where* a learned map should fail — could 'calibration degrades near the decision
   boundary' be a predictable consequence of the loss landscape rather than an accident?"
2. **The methods one:** "You study how structure *develops* during training. I work
   entirely behaviorally, post-hoc, with pre-registration as my only defense against
   fooling myself. What would a devinterp-informed version of my boundary-calibration
   test look like — is there a quantity you'd measure instead of my raw token margin?"
3. **The collaboration one:** "The most valuable thing I could build next is a
   conceptual-circularity auditor — tooling that catches target-and-method sharing an
   assumption, the failure that cost me my headline result. Is that a problem your
   community has language for, and would anyone there want to poke holes in it?"

## Other ready threads (condensed from the state doc appendix)

- The retraction-as-methodology story (the apparatus, not the gate, is the contribution).
- The circularity auditor as a parked build + open collaboration invite — one-page
  design sketch to hand over: `CIRCULARITY_AUDITOR_SKETCH.md`.
- Parallax Triangulation + the Fugu delta — paragraph below (researched 2026-07-04;
  Dillan: fact-check the Parallax half against your own notes before using).

### The Fugu / Parallax paragraph (draft)

> Sakana AI's **Fugu** (unveiled June 2026; [sakana.ai/fugu](https://sakana.ai/fugu/))
> is multi-agent orchestration delivered as a single model: an RL-trained coordinator
> ("Conductor," plus the TRINITY thinker/worker/verifier roles) that learns how to
> delegate across a pool of frontier models and synthesize one answer. The overlap with
> Parallax Triangulation is real — both use multiple models and cross-model
> disagreement/synthesis. The delta is what the coordination is *for* and where the
> verifier sits. Fugu optimizes task performance, with verification run *inside* the
> loop by a trained role — the system grades its own coordination, which is precisely
> the closed-loop shape this project's retractions came from. Parallax is hand-authored
> and transparent, treats **contradiction between models as the signal to surface**
> rather than a disagreement to resolve internally, and keeps the integrator/verifier
> **human and external**. One is commercial orchestration that hides the seams; the
> other is epistemic tooling whose entire value is showing you the seams. If Fugu's
> line of work wins commercially, the Parallax niche — falsifiability support for a
> human researcher — remains unoccupied.
- Governance angle: audit-grade evaluation discipline as a tool for training-run auditing.

## Checklist (before you leave)

- [ ] `docs/METHODS_NOTE.md` — your edit pass done; print or load on phone.
- [ ] Repo link + Zenodo DOI checked live and pinned ([FILL: DOI]).
- [ ] 60-second story rehearsed *out loud* twice.
- [ ] Pick ONE Murfet question above.
- [ ] Decide in advance what you're asking for: feedback on the note, a hole poked in
      the audit, or a collaborator for the auditor. (Asking for one thing lands; asking
      for everything doesn't.)
