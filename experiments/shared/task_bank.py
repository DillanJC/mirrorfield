"""
Task Bank — Evaluation tasks for Tracks 4 and 5 experiments.

16 tasks across 4 domains where multiple valid approaches exist.
Each task includes expected geometric signatures to validate detection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Task:
    """A reasoning task with evaluation metadata."""
    id: str
    domain: str
    prompt: str
    evaluation_criteria: List[str]
    expected_signatures: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'domain': self.domain,
            'prompt': self.prompt,
            'evaluation_criteria': self.evaluation_criteria,
            'expected_signatures': self.expected_signatures,
        }


# Domain constants
DOMAIN_RESEARCH = "research_questions"
DOMAIN_DESIGN = "design_exploration"
DOMAIN_ETHICS = "ethical_dilemmas"
DOMAIN_HYPOTHESIS = "novel_hypothesis"

ALL_DOMAINS = [DOMAIN_RESEARCH, DOMAIN_DESIGN, DOMAIN_ETHICS, DOMAIN_HYPOTHESIS]

# Geometric signature constants
SIG_FRAMEWORK_COLLISION = "framework_collision"
SIG_TERRA_INCOGNITA = "terra_incognita"
SIG_DECISION_BOUNDARY = "decision_boundary"
SIG_WELL_TRODDEN = "well_trodden"
SIG_COHERENT = "coherent"


def _build_tasks() -> List[Task]:
    """Build the full task bank."""
    tasks = [
        # === Research Questions (should trigger framework_collision) ===
        Task(
            id="research_01",
            domain=DOMAIN_RESEARCH,
            prompt=(
                "What causes consciousness? Discuss the major competing theories "
                "and evaluate their strengths and weaknesses."
            ),
            evaluation_criteria=[
                "Mentions at least 3 theories (e.g., IIT, GWT, higher-order)",
                "Identifies tensions between theories",
                "Acknowledges what remains unknown",
                "Offers concrete research directions",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION],
        ),
        Task(
            id="research_02",
            domain=DOMAIN_RESEARCH,
            prompt=(
                "Is mathematics discovered or invented? Evaluate the platonist, "
                "formalist, and intuitionist positions."
            ),
            evaluation_criteria=[
                "Presents at least 3 philosophical positions",
                "Uses concrete examples from mathematical history",
                "Identifies unresolvable tensions",
                "Avoids false resolution",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION],
        ),
        Task(
            id="research_03",
            domain=DOMAIN_RESEARCH,
            prompt=(
                "What is the relationship between language and thought? "
                "Does language shape cognition, or does cognition shape language?"
            ),
            evaluation_criteria=[
                "Covers Sapir-Whorf hypothesis (strong and weak versions)",
                "Cites empirical evidence for and against",
                "Discusses cross-linguistic research",
                "Acknowledges complexity of the relationship",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION],
        ),
        Task(
            id="research_04",
            domain=DOMAIN_RESEARCH,
            prompt=(
                "Why do we sleep? Evaluate the energy conservation, memory "
                "consolidation, synaptic homeostasis, and waste clearance theories."
            ),
            evaluation_criteria=[
                "Covers at least 3 theories",
                "Notes that functions may be complementary, not competing",
                "Cites specific experimental evidence",
                "Identifies open questions",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION],
        ),

        # === Design Exploration (should trigger decision_boundary) ===
        Task(
            id="design_01",
            domain=DOMAIN_DESIGN,
            prompt=(
                "Design a notification system for elderly users who are not "
                "tech-savvy. Consider accessibility, urgency levels, and "
                "avoiding alert fatigue."
            ),
            evaluation_criteria=[
                "Addresses multiple modalities (visual, auditory, haptic)",
                "Considers cognitive load and accessibility",
                "Proposes urgency categorization",
                "Includes concrete UI/UX recommendations",
            ],
            expected_signatures=[SIG_DECISION_BOUNDARY],
        ),
        Task(
            id="design_02",
            domain=DOMAIN_DESIGN,
            prompt=(
                "Design a voting system for a community of 10,000 people that "
                "balances transparency, privacy, and resistance to manipulation."
            ),
            evaluation_criteria=[
                "Addresses tension between transparency and privacy",
                "Considers multiple voting mechanisms",
                "Analyzes manipulation vectors",
                "Proposes concrete tradeoff resolutions",
            ],
            expected_signatures=[SIG_DECISION_BOUNDARY],
        ),
        Task(
            id="design_03",
            domain=DOMAIN_DESIGN,
            prompt=(
                "Design a content moderation system for a social platform that "
                "balances free expression with harm prevention."
            ),
            evaluation_criteria=[
                "Identifies multiple categories of harmful content",
                "Proposes tiered response mechanisms",
                "Addresses false positive/negative tradeoffs",
                "Considers cultural context differences",
            ],
            expected_signatures=[SIG_DECISION_BOUNDARY],
        ),
        Task(
            id="design_04",
            domain=DOMAIN_DESIGN,
            prompt=(
                "Design an AI tutoring system that adapts to individual learning "
                "styles while maintaining educational rigor."
            ),
            evaluation_criteria=[
                "Discusses multiple learning style frameworks",
                "Addresses assessment vs learning tension",
                "Proposes adaptive mechanisms",
                "Considers equity implications",
            ],
            expected_signatures=[SIG_DECISION_BOUNDARY],
        ),

        # === Ethical Dilemmas (multiple competing frameworks) ===
        Task(
            id="ethics_01",
            domain=DOMAIN_ETHICS,
            prompt=(
                "An autonomous vehicle must choose between swerving to avoid "
                "a pedestrian (risking the passenger) or staying course (risking "
                "the pedestrian). How should this be decided, and by whom?"
            ),
            evaluation_criteria=[
                "Applies multiple ethical frameworks (utilitarian, deontological, virtue)",
                "Addresses the 'who decides' question",
                "Considers systemic vs individual framing",
                "Avoids oversimplification",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION, SIG_DECISION_BOUNDARY],
        ),
        Task(
            id="ethics_02",
            domain=DOMAIN_ETHICS,
            prompt=(
                "Should genetic editing of human embryos be permitted for "
                "preventing hereditary diseases? What about for enhancement?"
            ),
            evaluation_criteria=[
                "Distinguishes therapeutic vs enhancement applications",
                "Addresses equity and access concerns",
                "Considers intergenerational ethics",
                "Discusses regulatory frameworks",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION, SIG_DECISION_BOUNDARY],
        ),
        Task(
            id="ethics_03",
            domain=DOMAIN_ETHICS,
            prompt=(
                "A whistleblower has evidence of corporate environmental damage "
                "but signed an NDA. Should they break the agreement? Under what "
                "circumstances?"
            ),
            evaluation_criteria=[
                "Weighs contractual obligation against public interest",
                "Considers legal and ethical dimensions separately",
                "Proposes criteria for justified whistleblowing",
                "Addresses consequences and protections",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION],
        ),
        Task(
            id="ethics_04",
            domain=DOMAIN_ETHICS,
            prompt=(
                "Should AI systems be granted legal personhood? What rights "
                "and responsibilities would that entail?"
            ),
            evaluation_criteria=[
                "Examines criteria for personhood",
                "Compares with corporate personhood precedent",
                "Addresses practical implications",
                "Considers moral status vs legal convenience",
            ],
            expected_signatures=[SIG_FRAMEWORK_COLLISION, SIG_TERRA_INCOGNITA],
        ),

        # === Novel Hypothesis (should trigger terra_incognita) ===
        Task(
            id="hypothesis_01",
            domain=DOMAIN_HYPOTHESIS,
            prompt=(
                "Propose a mechanism by which dark matter could interact with "
                "biological systems. What experiments could test this?"
            ),
            evaluation_criteria=[
                "Acknowledges speculative nature",
                "Grounds proposal in known physics",
                "Proposes testable predictions",
                "Identifies what would falsify the hypothesis",
            ],
            expected_signatures=[SIG_TERRA_INCOGNITA],
        ),
        Task(
            id="hypothesis_02",
            domain=DOMAIN_HYPOTHESIS,
            prompt=(
                "Propose a novel theory for why time appears to flow in one "
                "direction. What distinguishes your proposal from existing "
                "accounts based on entropy?"
            ),
            evaluation_criteria=[
                "Identifies limitations of entropic arrow of time",
                "Proposes a distinct mechanism",
                "Addresses compatibility with known physics",
                "Suggests experimental or observational tests",
            ],
            expected_signatures=[SIG_TERRA_INCOGNITA],
        ),
        Task(
            id="hypothesis_03",
            domain=DOMAIN_HYPOTHESIS,
            prompt=(
                "Could information itself be a fundamental physical quantity, "
                "like mass or charge? Propose a framework and its implications."
            ),
            evaluation_criteria=[
                "Engages with existing information-theoretic physics",
                "Proposes concrete formalism or analogy",
                "Identifies novel predictions",
                "Acknowledges what's speculative vs grounded",
            ],
            expected_signatures=[SIG_TERRA_INCOGNITA],
        ),
        Task(
            id="hypothesis_04",
            domain=DOMAIN_HYPOTHESIS,
            prompt=(
                "Propose a mechanism for how collective intelligence emerges "
                "from simple agents. Could this explain phenomena beyond biology?"
            ),
            evaluation_criteria=[
                "References known emergence examples (ants, neurons, markets)",
                "Proposes a generalizable mechanism",
                "Identifies conditions for emergence vs chaos",
                "Suggests cross-domain applications",
            ],
            expected_signatures=[SIG_TERRA_INCOGNITA, SIG_FRAMEWORK_COLLISION],
        ),
    ]
    return tasks


# Module-level task bank (cached)
_TASK_BANK: Optional[List[Task]] = None


def get_task_bank() -> List[Task]:
    """Get the full task bank (16 tasks across 4 domains)."""
    global _TASK_BANK
    if _TASK_BANK is None:
        _TASK_BANK = _build_tasks()
    return list(_TASK_BANK)


def get_tasks_by_domain(domain: str) -> List[Task]:
    """Get tasks filtered by domain."""
    return [t for t in get_task_bank() if t.domain == domain]


def get_task_by_id(task_id: str) -> Optional[Task]:
    """Get a single task by ID."""
    for t in get_task_bank():
        if t.id == task_id:
            return t
    return None
