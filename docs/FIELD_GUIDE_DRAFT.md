# How not to fool yourself when an AI does your research — a field guide with receipts

*DRAFT for Dillan's voice and his decision on sharing — written 2026-07-04, auto
session. Audience: the growing number of people using capable AI assistants to do
research they cannot fully audit themselves. The guide's authority is not expertise;
it is a public record of getting fooled twice and building the machine that catches it.
Every example is real and committed in this repository.*

> **[Dillan: this document is yours.** The bones below are from the project record; the
> voice should be replaced with yours wherever it reads like a manual. The sections
> marked for you at the bottom are the ones only you can write.**]**

## 1. Who this is for

You have an idea and an AI assistant that can implement it. You cannot read every line
it writes or check every number it reports. The AI is fast, encouraging, and produces
results that look like science — plots, AUCs, significance. This guide is about the
specific ways that goes wrong, from someone it went wrong for, twice, with the controls
that caught it.

The awkward premise, stated plainly: **the setup where an AI proposes the experiment,
runs it, and judges whether it worked is a machine for generating convincing false
positives.** Not because the AI lies, but because nothing in the loop is positioned to
notice the boring explanation. You have to build that position in from outside.

## 2. The credential (what happened here)

This project's headline was once a detector with AUC 0.947 — "geometric cluster
poisoning detection." It was retracted, by us: the target and the method shared the same
assumption, so the detector was finding its own definition (an honest baseline collapsed
it to ~0.47). Before that, a v2.0 cycle overclaimed positives the same way. After the
retraction, the same apparatus caught three further would-be results before they
shipped, and later found a real miscalibration in the project's *own surviving tool* —
the one result we had every incentive to protect. The apparatus, not any single result,
turned out to be the contribution.

## 3. The five failure shapes (all of which happened here)

1. **Circularity — the method defines its own target.** Poison defined by geometry,
   detected by geometry → AUC 0.947 → chance under an honest baseline. Ask of every
   result: *what, exactly, provides the ground truth, and could the method have
   influenced it?*
2. **The consolidation layer inflates what the experiments earned.** Our per-experiment
   discipline held; the *summary documents* still drifted — "replication" became
   "novel," "on this model" quietly disappeared, a contaminated control's number
   survived in a synthesis table after the experiment itself was corrected. Watch the
   summaries, abstracts, and tables hardest; that is where overclaiming actually happens.
3. **Single-run wins.** Every single-seed positive this project ever celebrated later
   shrank or died. Two seeds minimum before a claim exists at all.
4. **The manipulated control.** Our sycophancy placebo ("can you double-check?") itself
   implied doubt — an external reader caught it. We *tested* the criticism instead of
   arguing with it: the clean control made the effect *bigger*. Controls need the same
   scrutiny as claims.
5. **The self-grading loop.** In auto mode the assistant reviews its own work — the
   0.947 failure shape at the architecture level. If the AI runs unattended, its job
   must flip from "conclude" to "preserve your ability to audit": logs, small reversible
   units, hard stops, and verdicts that wait for you.

## 4. The rules that actually held

- **Pre-register success AND abandonment before data exists** — write both down, commit
  them (the commit hash is your timestamp), and let the abandon criterion kill the run
  when it triggers. The abandon half is the part that saves you.
- **Ground truth must come from outside the method** — gold labels, an independent
  judge, a canary token. The method never grades itself.
- **A placebo beside every manipulation.** A content-free version of your intervention,
  or you cannot distinguish your idea from "any prefix changes behavior" (that placebo
  reversed one of our headline-bound findings).
- **Shuffled-label null.** If your pipeline finds signal in shuffled labels, the
  pipeline is the signal.
- **A null is a result.** Publishing "the exciting thing didn't replicate" is the
  cheapest credibility you will ever buy.
- **Scope lives inside the sentence.** Not a limitations section nobody reads — the
  words "on this model" inside every sentence that could generalize. Held even when the
  cleaner-sounding sentence is available.
- **Numbers cite their source, and the repo wins.** Every number in every document
  traces to a logged result; when two documents disagree, stop and reconcile before
  anything else. (Our worst drift was numbers mutating between drafts.)
- **Get an outside read before you believe yourself.** A second, independent AI
  instance with no stake in the session — or better, a human — reviewing your synthesis
  cold. Ours caught both the contaminated control and the inflated framing.

The one-page operational version: [`SAFETY_CLAIM_SMELL_TEST.md`](SAFETY_CLAIM_SMELL_TEST.md).

## 5. The division of labor that makes it safe

The AI implements, drafts, replicates, and *surfaces candidates*. The human owns three
things that must never move into the loop: **what counts as success (fixed before
data), whether a verdict is concluded, and whether anything goes public.** This isn't
ceremony — each one is a place where the self-grading loop otherwise closes. And prefer
enforcement to prose: a rules file the AI reads is a suggestion; permissions it
physically lacks are a guarantee. (While drafting this project's work, the harness's
own permission layer once refused an action the session had rationalized as safe — the
layered defense worked exactly as designed, catching what the rules file alone would
not have.)

## 6. What it costs, honestly

This discipline killed most of this project's exciting results — the survivor list is
one modest gate and one calibration finding. It is slower; it reads as pedantic; your
best headline usually dies in week two. What you get back: the two results that survive
are *real*, your retractions become credentials instead of embarrassments, and people
who check your work find it checks out. For an independent researcher with no
institutional reputation, that is the entire asset.

## 7. [Dillan: yours to write]

> - Why you kept going after the retraction instead of quietly deleting the repo.
> - What it felt like to watch the 0.947 collapse, and what you'd tell someone in that
>   moment.
> - What a non-coder actually needs to know going in (you know this better than the AI
>   drafting this sentence does).
> - Your sharing decision: where this goes, if anywhere, and under what name.

---

*Companion documents in this repo: `experiments/EVALUATION_DISCIPLINE.md` (the full
method), `WORK_MAP.md` (every result, including the dead ones), `docs/METHODS_NOTE.md`
(the apparatus turned on the project's own surviving result), and
`docs/CONSIDERATE_COLLABORATION.md` (the manner half — how the human–AI collaboration
itself was run). Nothing in this guide is a universal law; it is one project's tested
practice, offered for others to break and improve.*
