"""
Mirror Interventions — Reflective rather than directive.

Instead of telling the model what to do ("Identify the tension..."),
these interventions describe the model's geometric state and let it
decide how to respond.

The hypothesis: directive interventions disrupt because they impose
an external agenda. Reflective interventions inform without directing,
preserving the model's natural reasoning trajectory while making
geometric structure visible.
"""

from experiments.track4_switches.switch_engine import Intervention


# Reflective versions of each intervention
MIRROR_INTERVENTIONS = [
    Intervention(
        signal="framework_collision",
        instruction=(
            "Note: Your reasoning is currently navigating between multiple "
            "distinct conceptual frameworks. The semantic structure of your "
            "recent steps shows high variation in neighbor distances, which "
            "typically indicates you're drawing from competing paradigms "
            "simultaneously. This is an observation about where you are in "
            "reasoning space — continue as you see fit."
        ),
        priority=3,
    ),
    Intervention(
        signal="terra_incognita",
        instruction=(
            "Note: Your current reasoning has moved into semantic territory "
            "that is distant from established reference points. The geometric "
            "trace shows you're exploring ideas that are notably dissimilar "
            "to the conventional approaches in this domain. This is an "
            "observation about your position in reasoning space — continue "
            "as you see fit."
        ),
        priority=4,
    ),
    Intervention(
        signal="decision_boundary",
        instruction=(
            "Note: Your reasoning appears to be at a geometric transition "
            "point. The density gradient in the embedding space suggests "
            "you're positioned between two distinct clusters of ideas, where "
            "small shifts in framing could lead to substantially different "
            "conclusions. This is an observation — continue as you see fit."
        ),
        priority=2,
    ),
    Intervention(
        signal="low_pr",
        instruction=(
            "Note: Your reasoning is currently in a region of low semantic "
            "density — there are fewer established reference points nearby "
            "than is typical for this kind of task. This could indicate "
            "either novel territory or a gap in the conventional discourse. "
            "This is an observation — continue as you see fit."
        ),
        priority=1,
    ),
]
