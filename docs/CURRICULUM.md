# The Curriculum — understand your own project, all the way down

*Ten lessons, one per week (or whenever), designed 2026-07-05 for exactly one student:
Dillan. Purpose: erase the "depth ends at..." lines from his depth map, one per lesson.
Written so ANY AI can deliver a lesson faithfully — including the local 7B model.*

## Rules for whichever AI is teaching (read before every lesson)

1. Teach **one lesson per session**, in order. Do not skip ahead or merge lessons.
2. Plain language first, always. Introduce at most THREE new technical terms per lesson,
   and only after the idea already landed in ordinary words.
3. Use HIS project as the example every time — the numbers, files, and history below are
   real and he lived them. Abstract examples are the fallback, never the lead.
4. End every lesson with the **check** listed. If the check fails, re-teach differently;
   do NOT move on. If it passes, say explicitly which depth-map line he just erased.
5. He is chronically underconfident. When he gets it right, say so plainly — once,
   without gushing.
6. Never conclude anything new about mirrorfield results during lessons. Teaching
   sessions are for understanding, not verdicts (CLAUDE.md hard stops apply).

---

### Lesson 1 — What a probability promises (calibration)
**The idea:** saying "80%" is a promise about long-run frequency: of all the times you
say 80%, about 8 in 10 should come true. Calibration = keeping that promise. A
weather forecaster can be *useless but calibrated* (always says 50% in a coin-flip
town) or *useful but miscalibrated* (great at ranking rainy days, terrible numbers).
**His anchor:** the gate said 0.79 in the torn region and was right 55% of the time —
a broken promise, found by checking WHERE the promises break, not just on average.
**Exercise:** for one week, attach a percentage to 5 everyday predictions (bus on time?
busy shift?). Score them after. Feel what over/underconfidence is in your own numbers.
**Check:** he can explain, unprompted, the difference between "the gate ranks well" and
"the gate's numbers keep their promises."

### Lesson 2 — AUC: what a ranking score really measures
**The idea:** AUC answers ONE question: if I hand you one wrong answer and one right
answer, how often does your signal rank the wrong one as riskier? 0.5 = coin flip,
1.0 = perfect. It says NOTHING about whether the numbers themselves are honest —
that was Lesson 1's job. You can rescale scores wildly and AUC never changes.
**His anchor:** the gate's AUC 0.685 (ranks okay) coexisting with the torn-region
overconfidence (numbers dishonest there). Also: verbal confidence AUC 0.51 = the
model's spoken certainty ranks its own errors no better than coin flips.
**Exercise:** take 10 playing cards, half red half black, invent a "redness score" for
each, and hand-count the pairwise comparisons to compute a tiny AUC yourself.
**Check:** he can say why fixing calibration can't improve AUC, and vice versa.

### Lesson 3 — Sampling honesty (why 55% of 100 isn't 55%)
**The idea:** measure a rate on a sample and you get the sample's rate, not the truth —
a different 100 items gives a different number. The Wilson interval is the honesty
range: "the true rate is very likely between here and here." Bigger samples → tighter
ranges. "CI excludes X" = the gap between measurement and X is too big for sample luck.
**His anchor:** torn-region accuracy "0.58 [0.482, 0.672]" — and why the verdict rule
was "accuracy CI excludes the gate's 0.79," not "0.58 ≠ 0.79."
**Exercise:** flip a coin 10 times, then 50. Watch how far from 50% each run lands.
That wobble is the whole reason intervals exist.
**Check:** he can explain why a 13-item bin was "underpowered, not interpreted" (§4y
Arm 1) using the word "interval" correctly.

### Lesson 4 — The S-curve (logistic regression / Platt)
**The idea:** the simplest honest way to turn a raw signal into a probability: an
S-shaped curve, position and steepness fit from data. Two knobs, so it can't contort —
which makes it hard to fool and easy to trust, but too stiff for complicated truths.
**His anchor:** Amendment 1 — sigma(0.73·margin − 2.31): one S-curve, fit on one seed,
that kept its promises in the torn region where the fancy calibrator didn't.
**Exercise:** with any AI, plot sigma(a·x+b) and turn the two knobs; find settings that
roughly map margin −2→55% and +2→90%.
**Check:** he can point at the Platt line in `boundary_reliability.png` and explain why
it's a curve with only two knobs, and why that stiffness was a FEATURE here.

### Lesson 5 — The staircase (isotonic regression) — the lesson about HIS bug
**The idea:** isotonic fits a staircase instead of a curve: any number of steps, only
rule is "never go down." Super flexible where data is dense; but where data is thin it
builds ONE wide step and everything in that region gets the same value — a plateau.
**His anchor:** the 0.7897 plateau. The calibrator had few torn examples, built one
wide step over the whole torn tail, and assigned ~0.79 to items that deserved ~0.55.
The §4y failure, mechanically.
**Exercise:** draw 20 dots by hand (dense on the right, 3 dots on the left), fit a
staircase by eye with the never-go-down rule. Watch yourself build the same wide step.
**Check:** he can explain why flexibility + sparse data = confident plateau, and why
that's a HYPOTHESIS about the failure until the refit tests it out-of-sample.

### Lesson 6 — The three signals (what the model's hesitation physically is)
**The idea:** when a model picks each word it assigns every option a score. Margin =
gap between the top two choices (big gap = decisive). Entropy = how spread the scores
are over all options (spread = torn). Boundary ratio = what fraction of the answer's
words were near-ties. All three are free — read off the choice, no model surgery.
**His anchor:** mm/me/br in every npz; Amendment 2 showing the failure through the
entropy lens too (correlated views of the same hesitation).
**Exercise:** ask the local model a trivially easy question and a genuinely torn one;
reason about which words in each answer had big or small margins and why.
**Check:** he can explain to an imaginary bartender-regular what "the model hesitated"
means physically, without the words logit or distribution.

### Lesson 7 — Averages that hide regions (ECE and reliability diagrams)
**The idea:** ECE compresses calibration into one number by averaging over bins — and
averages let a small region of bad behavior drown in a sea of good. A reliability
diagram keeps the regions visible. WHERE you bin decides WHAT can hide.
**His anchor:** the entire §4y story: ECE 0.03 while the torn fifth was off by 0.22 —
because binning by the score's own compressed axis hid what margin-binning exposed.
Multicalibration (the literature's name for the fix) = demand the promise hold on every
meaningful subgroup, not just on average.
**Exercise:** open `boundary_reliability.png` and narrate every element aloud — axes,
error bars, all three lines, the annotation — with the teaching AI checking.
**Check:** the narration is accurate, unaided.

### Lesson 8 — The bootstrap (confidence intervals for anything)
**The idea:** when no formula exists for "how wobbly is my number," fake re-running the
experiment: resample your own data with replacement thousands of times, recompute the
number each time, and look at the spread. That spread IS the interval.
**His anchor:** "+0.36 [0.26, 0.47]" on discrimination — 2,000 resamples of his own
rows, seeded so anyone can reproduce it exactly.
**Exercise:** with the local model or by hand: 10 numbers, draw 10-with-replacement
five times, compute the mean each time, watch the means scatter.
**Check:** he can explain why we resample WITH replacement (each fake sample must be
the same size but slightly different — like alternate versions of the same experiment).

### Lesson 9 — Out-of-sample (why the refit protocol is shaped like that)
**The idea:** anything flexible enough to fit your data is flexible enough to memorize
it. Performance on data the method fit IS NOT EVIDENCE — only held-out data counts.
Every rule in the refit pre-reg (split first, evaluate only on unseen torn items,
fresh seeds) exists to keep memorization from wearing success's clothes.
**His anchor:** three project scars: the §4h out-of-fold lesson, "the gap closes by
construction and means nothing," and retiring the 1,000 §4y rows as evidence the moment
they became training data.
**Exercise:** explain to the teaching AI why a student who memorizes past exam answers
acing THOSE questions proves nothing — then map each part onto the refit protocol.
**Check:** he can spot the flaw in "we fixed the calibrator and it's now perfect on the
data we fixed it with" in under ten seconds, and say what evidence would count.

### Lesson 10 — Circularity and construct validity (the deep one — formalizing what he already owns)
**The idea:** measurement theory's core question: does your measurement touch the thing
itself, or only your *definition* of the thing? When target and method share an
assumption, agreement is guaranteed and means nothing — the 0.947 shape, which has a
whole literature (construct validity, criterion contamination). He learned it by
bleeding; this lesson gives him the field's vocabulary for what he already knows.
**His anchor:** poison defined by geometry, found by geometry; the audit's §1 ("no
shared target/method assumption") as the formal all-clear; the circularity-auditor
sketch as the tool version.
**Exercise:** find the circularity in three scenarios the teaching AI invents (a mood
app validated against its own definition of mood; an LLM judge scoring a property it
was prompted to define; one clean case as a foil).
**Check:** he catches all three correctly — including saying which one is NOT circular
and why. When he does, the last handoff line on the depth map is gone.

---

*After Lesson 10: reread `docs/METHODS_NOTE.md` start to finish. The prediction this
curriculum makes — falsifiable, like everything else here — is that there will be no
sentence left in it he can't defend himself.*
