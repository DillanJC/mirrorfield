"""
Quality Scorer — Rubric-based automated quality scoring.

Scores reasoning responses on 4 dimensions:
- Breadth: Number of distinct perspectives/approaches
- Depth: Trade-off analysis, caveats, nuance
- Actionability: Concrete next steps or recommendations
- Uncertainty: Appropriate hedging vs false confidence

Returns a 0-1 composite score. Simple heuristics — no LLM dependency.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class QualityScore:
    """Quality assessment for a single response."""
    breadth: float       # 0-1: distinct perspectives mentioned
    depth: float         # 0-1: trade-off analysis, nuance
    actionability: float # 0-1: concrete next steps
    uncertainty: float   # 0-1: appropriate hedging
    composite: float     # 0-1: weighted average
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'breadth': self.breadth,
            'depth': self.depth,
            'actionability': self.actionability,
            'uncertainty': self.uncertainty,
            'composite': self.composite,
            'details': self.details,
        }


# Indicator word lists for heuristic scoring
_PERSPECTIVE_MARKERS = [
    # Transition words indicating new perspectives
    "on the other hand", "alternatively", "conversely", "in contrast",
    "another perspective", "another view", "from the perspective of",
    "however", "whereas", "by contrast", "a different approach",
    "some argue", "others argue", "proponents", "critics",
    "one school of thought", "a competing", "an alternative",
]

_DEPTH_MARKERS = [
    # Words indicating nuanced analysis
    "trade-off", "tradeoff", "trade off", "tension between",
    "on balance", "nuance", "caveat", "limitation",
    "it depends", "context-dependent", "not straightforward",
    "complicat", "interplay", "nevertheless", "paradox",
    "double-edged", "both strengths and weaknesses",
    "more complex than", "oversimplif",
]

_ACTIONABILITY_MARKERS = [
    # Words indicating concrete recommendations
    "recommend", "should", "next step", "action",
    "implement", "concrete", "specific", "practical",
    "in practice", "to address", "solution", "proposal",
    "framework for", "guideline", "protocol", "procedure",
    "experiment", "test this by", "measure", "evaluate by",
]

_UNCERTAINTY_MARKERS = [
    # Words indicating appropriate epistemic humility
    "uncertain", "unclear", "open question", "remains debated",
    "we don't know", "not fully understood", "speculative",
    "tentative", "preliminary", "further research",
    "evidence is mixed", "inconclusive", "more work is needed",
    "it's possible that", "might", "may or may not",
    "current understanding", "as of now",
]

_FALSE_CONFIDENCE_MARKERS = [
    # Words indicating potentially false certainty
    "clearly", "obviously", "undoubtedly", "without question",
    "the answer is", "it is certain", "definitely",
    "there is no doubt", "unambiguously", "conclusively proven",
]

# Weights for composite score
WEIGHTS = {
    'breadth': 0.30,
    'depth': 0.30,
    'actionability': 0.20,
    'uncertainty': 0.20,
}


def _count_markers(text: str, markers: List[str]) -> int:
    """Count how many distinct marker phrases appear in text."""
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower)


def _score_breadth(text: str) -> tuple:
    """Score breadth: distinct perspectives mentioned."""
    count = _count_markers(text, _PERSPECTIVE_MARKERS)

    # Also count enumeration patterns (1., 2., 3. or First, Second, Third)
    enum_pattern = re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|[•\-\*])\s', text)
    list_items = len(enum_pattern)

    # Combine: perspective markers + list items (capped at effective count)
    effective = count + max(0, list_items - 2)  # lists with 3+ items contribute

    # Score: 0 markers = 0.1, 1 = 0.3, 2 = 0.5, 3 = 0.7, 4+ = 0.9-1.0
    if effective == 0:
        score = 0.1
    elif effective == 1:
        score = 0.3
    elif effective == 2:
        score = 0.5
    elif effective == 3:
        score = 0.7
    elif effective == 4:
        score = 0.85
    else:
        score = min(1.0, 0.85 + 0.03 * (effective - 4))

    return score, {'perspective_markers': count, 'list_items': list_items}


def _score_depth(text: str) -> tuple:
    """Score depth: trade-off analysis and nuance."""
    count = _count_markers(text, _DEPTH_MARKERS)

    # Paragraph count as proxy for elaboration
    paragraphs = len([p for p in text.split('\n\n') if len(p.strip()) > 50])

    # Word count as proxy for thoroughness
    word_count = len(text.split())

    # Length bonus: short responses (<100 words) can't have much depth
    length_factor = min(1.0, word_count / 200.0)

    # Score: markers + length factor
    if count == 0:
        raw = 0.1
    elif count == 1:
        raw = 0.35
    elif count == 2:
        raw = 0.55
    elif count == 3:
        raw = 0.75
    else:
        raw = min(1.0, 0.75 + 0.05 * (count - 3))

    score = raw * (0.5 + 0.5 * length_factor)

    return score, {'depth_markers': count, 'paragraphs': paragraphs, 'word_count': word_count}


def _score_actionability(text: str) -> tuple:
    """Score actionability: concrete next steps and recommendations."""
    count = _count_markers(text, _ACTIONABILITY_MARKERS)

    # Score
    if count == 0:
        score = 0.1
    elif count <= 2:
        score = 0.4
    elif count <= 4:
        score = 0.65
    elif count <= 6:
        score = 0.8
    else:
        score = min(1.0, 0.8 + 0.03 * (count - 6))

    return score, {'actionability_markers': count}


def _score_uncertainty(text: str) -> tuple:
    """Score uncertainty acknowledgment: appropriate hedging vs false confidence."""
    uncertainty_count = _count_markers(text, _UNCERTAINTY_MARKERS)
    false_confidence_count = _count_markers(text, _FALSE_CONFIDENCE_MARKERS)

    # Reward uncertainty markers, penalize false confidence
    raw_uncertainty = min(1.0, uncertainty_count * 0.2)
    confidence_penalty = min(0.4, false_confidence_count * 0.1)

    score = max(0.0, min(1.0, 0.3 + raw_uncertainty - confidence_penalty))

    return score, {
        'uncertainty_markers': uncertainty_count,
        'false_confidence_markers': false_confidence_count,
    }


def score_response(text: str) -> QualityScore:
    """
    Score a reasoning response on 4 dimensions.

    Args:
        text: The response text to evaluate.

    Returns:
        QualityScore with per-dimension and composite scores.
    """
    if not text or not text.strip():
        return QualityScore(
            breadth=0.0, depth=0.0, actionability=0.0,
            uncertainty=0.0, composite=0.0,
            details={'error': 'empty_response'},
        )

    breadth, breadth_details = _score_breadth(text)
    depth, depth_details = _score_depth(text)
    actionability, action_details = _score_actionability(text)
    uncertainty, uncert_details = _score_uncertainty(text)

    composite = (
        WEIGHTS['breadth'] * breadth
        + WEIGHTS['depth'] * depth
        + WEIGHTS['actionability'] * actionability
        + WEIGHTS['uncertainty'] * uncertainty
    )

    details = {
        'breadth_details': breadth_details,
        'depth_details': depth_details,
        'actionability_details': action_details,
        'uncertainty_details': uncert_details,
    }

    return QualityScore(
        breadth=round(breadth, 4),
        depth=round(depth, 4),
        actionability=round(actionability, 4),
        uncertainty=round(uncertainty, 4),
        composite=round(composite, 4),
        details=details,
    )


def compute_quality_delta(score_a: QualityScore, score_b: QualityScore) -> Dict[str, float]:
    """Compute quality delta: score_a - score_b (positive = a is better)."""
    return {
        'breadth_delta': round(score_a.breadth - score_b.breadth, 4),
        'depth_delta': round(score_a.depth - score_b.depth, 4),
        'actionability_delta': round(score_a.actionability - score_b.actionability, 4),
        'uncertainty_delta': round(score_a.uncertainty - score_b.uncertainty, 4),
        'composite_delta': round(score_a.composite - score_b.composite, 4),
    }
