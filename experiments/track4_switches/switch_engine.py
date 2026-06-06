"""
Switch Engine — Geometric state -> intervention mapping.

Hard-coded intervention table mapping geometric signatures to
reasoning interventions. Deterministic baseline for Track 4.

Uses GeometricTracer for detection (which wraps GeometryBundle
and compute_knn_features).

v2: Live-calibrated thresholds, cooldown, improved intervention text.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from experiments.shared.geometric_tracer import GeometricTracer, StepTrace


@dataclass
class Intervention:
    """A reasoning intervention triggered by geometric state."""
    signal: str        # geometric signature that triggers this
    instruction: str   # text prepended to next reasoning step
    priority: int = 0  # higher = evaluated first (for ties)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal': self.signal,
            'instruction': self.instruction,
            'priority': self.priority,
        }


@dataclass
class InterventionResult:
    """Result of evaluating a step for interventions."""
    triggered: bool
    intervention: Optional[Intervention] = None
    step_trace: Optional[StepTrace] = None
    reason: str = ""
    skipped_cooldown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'triggered': self.triggered,
            'reason': self.reason,
            'skipped_cooldown': self.skipped_cooldown,
        }
        if self.intervention:
            result['intervention'] = self.intervention.to_dict()
        if self.step_trace:
            result['step_trace'] = self.step_trace.to_dict()
        return result


# Fix 4: Redesigned intervention text — directed rather than meta-questioning.
# Original text triggered cascading self-doubt and philosophical spirals.
# New text directs the model toward specific analytical actions.
DEFAULT_INTERVENTIONS = [
    Intervention(
        signal="framework_collision",
        instruction=(
            "Identify the specific tension between the competing frameworks "
            "you've raised. Which predictions differ? Where do they agree? "
            "Evaluate the strongest evidence for each before proceeding."
        ),
        priority=3,
    ),
    Intervention(
        signal="terra_incognita",
        instruction=(
            "This line of reasoning is entering speculative territory. "
            "What testable prediction would distinguish this explanation "
            "from existing alternatives? Anchor the next step in at least "
            "one concrete example or empirical finding."
        ),
        priority=4,
    ),
    Intervention(
        signal="decision_boundary",
        instruction=(
            "A critical decision point: what would change your assessment "
            "if your key assumption were false? Name the assumption, "
            "briefly test the counterfactual, then commit to a direction "
            "with explicit caveats."
        ),
        priority=2,
    ),
    Intervention(
        signal="low_pr",
        instruction=(
            "Name one concrete alternative approach or explanation you "
            "have not yet considered. Briefly evaluate it against what "
            "you've already covered, then continue your analysis."
        ),
        priority=1,
    ),
]

# Signals that receive no intervention
NO_INTERVENTION_SIGNALS = {"well_trodden", "coherent"}


class SwitchEngine:
    """
    Evaluates geometric state and triggers hard-coded interventions.

    v2 improvements:
    - Live-calibrated thresholds (from actual LLM embedding distributions)
    - Consecutive cooldown (same signal won't fire twice in a row)
    - Signature diversity logging

    Usage:
        engine = SwitchEngine(reference_embeddings, use_live_calibration=True)
        result = engine.evaluate(step_embedding, step_index)
        if result.triggered:
            # prepend result.intervention.instruction to next step
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        k: int = 50,
        interventions: Optional[List[Intervention]] = None,
        low_pr_threshold: Optional[float] = None,
        use_live_calibration: bool = False,
        live_thresholds: Optional[Dict[str, float]] = None,
        tracer=None,
    ):
        """
        Args:
            reference_embeddings: Reference corpus embeddings (N_ref, D)
            k: Number of neighbors for k-NN
            interventions: Override default intervention table
            low_pr_threshold: Override auto-calibrated low PR threshold
            use_live_calibration: If True, use live-calibrated thresholds
            live_thresholds: Custom live thresholds (uses defaults if None)
            tracer: External tracer instance (e.g. EnhancedTracer).
                    If provided, used instead of creating a GeometricTracer.
        """
        if tracer is not None:
            self.tracer = tracer
        else:
            self.tracer = GeometricTracer(reference_embeddings, k=k,
                                           classifier_mode="auto")
        self.interventions = interventions or list(DEFAULT_INTERVENTIONS)
        self.interventions.sort(key=lambda x: x.priority, reverse=True)

        # Fix 1: Apply live-calibrated thresholds
        if use_live_calibration:
            self.tracer.calibrate_from_live(live_thresholds)

        # Build signal -> intervention lookup
        self._signal_map: Dict[str, Intervention] = {}
        for intervention in self.interventions:
            self._signal_map[intervention.signal] = intervention

        # Low PR threshold from live calibration or manual override
        # (not used when EnhancedTracer handles low_pr via state mapping)
        if low_pr_threshold is not None:
            self._low_pr_threshold = low_pr_threshold
        elif use_live_calibration:
            self._low_pr_threshold = getattr(
                self.tracer, '_live_ridge_p25', 0.055
            )
        elif hasattr(self.tracer, '_ref_stats'):
            self._low_pr_threshold = self.tracer._ref_stats.get(
                'knn_std_p25', 0.05
            )
        else:
            self._low_pr_threshold = 0.05  # safe default for external tracers

        # Fix 2: Cooldown tracking
        self._last_signal: Optional[str] = None
        self._cooldown_skips: int = 0

        # State probability thresholds (for EnhancedTracer gating)
        self._state_thresholds: Optional[Dict[str, float]] = None

        # Tracking
        self._history: List[InterventionResult] = []

        # Fix 3: Signature diversity logging
        self._signature_log: List[Dict[str, Any]] = []

    def set_state_thresholds(self, thresholds: Dict[str, float]):
        """
        Set per-signal state probability thresholds for gating.

        When using an EnhancedTracer, the dominant state's probability
        must exceed the threshold for that signal to trigger. Otherwise
        the signal is demoted to 'coherent' (no intervention).

        Args:
            thresholds: Dict mapping signal name → minimum probability
        """
        self._state_thresholds = dict(thresholds)

    def evaluate(
        self,
        step_embedding: np.ndarray,
        step_index: int,
    ) -> InterventionResult:
        """
        Evaluate a reasoning step and return any triggered intervention.

        Includes consecutive cooldown: the same signal won't trigger
        interventions on back-to-back steps.

        Args:
            step_embedding: Embedding for the current step (D,) or (1, D)
            step_index: Index of the step

        Returns:
            InterventionResult with triggered intervention (if any)
        """
        # Get geometric trace for this step
        step_trace = self.tracer.trace_step(step_embedding, step_index)
        signature = step_trace.novelty_signature

        # State probability gating (EnhancedTracer only)
        has_state_scores = hasattr(step_trace, 'state_scores') and step_trace.state_scores
        if has_state_scores and self._state_thresholds:
            # Check if dominant state probability exceeds threshold for this signal
            dominant_prob = max(step_trace.state_scores.values())
            signal_threshold = self._state_thresholds.get(signature, 0.0)
            if signature in self._signal_map and dominant_prob < signal_threshold:
                signature = "coherent"  # demote — not confident enough

        # Check for low PR (ridge proximity below threshold)
        # Skip when using EnhancedTracer — it already maps searching→low_pr
        ridge_proximity = step_trace.features.get('ridge_proximity', 0)
        if not has_state_scores:
            if ridge_proximity < self._low_pr_threshold and signature in NO_INTERVENTION_SIGNALS:
                signature = "low_pr"

        # Fix 3: Log every signature for diversity analysis
        self._signature_log.append({
            'step_index': step_index,
            'raw_signature': step_trace.novelty_signature,
            'final_signature': signature,
            'ridge_proximity': ridge_proximity,
        })

        # Look up intervention
        if signature in NO_INTERVENTION_SIGNALS:
            result = InterventionResult(
                triggered=False,
                step_trace=step_trace,
                reason=f"Signal '{signature}' — no intervention needed",
            )
        elif signature in self._signal_map:
            # Fix 2: Cooldown — skip if same signal as last triggered
            if signature == self._last_signal:
                self._cooldown_skips += 1
                result = InterventionResult(
                    triggered=False,
                    step_trace=step_trace,
                    reason=f"Signal '{signature}' — cooldown (same as previous)",
                    skipped_cooldown=True,
                )
            else:
                result = InterventionResult(
                    triggered=True,
                    intervention=self._signal_map[signature],
                    step_trace=step_trace,
                    reason=f"Signal '{signature}' triggered intervention",
                )
                self._last_signal = signature
        else:
            result = InterventionResult(
                triggered=False,
                step_trace=step_trace,
                reason=f"Signal '{signature}' — no matching intervention",
            )

        # Reset last_signal when no intervention fires (not cooldown)
        if not result.triggered and not result.skipped_cooldown:
            self._last_signal = None

        self._history.append(result)
        return result

    def get_history(self) -> List[Dict[str, Any]]:
        """Get intervention history as list of dicts."""
        return [r.to_dict() for r in self._history]

    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get summary statistics on interventions fired."""
        total = len(self._history)
        triggered = sum(1 for r in self._history if r.triggered)
        cooldown_skips = sum(1 for r in self._history if r.skipped_cooldown)

        signal_counts: Dict[str, int] = {}
        for entry in self._signature_log:
            sig = entry['final_signature']
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

        intervention_counts: Dict[str, int] = {}
        for r in self._history:
            if r.triggered and r.intervention:
                sig = r.intervention.signal
                intervention_counts[sig] = intervention_counts.get(sig, 0) + 1

        # Fix 3: Diversity metrics
        unique_signals = len(set(e['final_signature'] for e in self._signature_log))

        return {
            'total_steps': total,
            'interventions_triggered': triggered,
            'cooldown_skips': cooldown_skips,
            'trigger_rate': triggered / total if total > 0 else 0.0,
            'signal_counts': signal_counts,
            'intervention_counts': intervention_counts,
            'unique_signals_seen': unique_signals,
            'signature_diversity': unique_signals / 6.0,  # 6 possible signals
        }

    def reset_history(self):
        """Clear intervention history for a new task."""
        self._history.clear()
        self._signature_log.clear()
        self._last_signal = None
        self.tracer.reset_trajectory()

    def get_policy_table(self) -> List[Dict[str, Any]]:
        """Get the current intervention table as a list of dicts."""
        return [i.to_dict() for i in self.interventions]

    def get_signature_log(self) -> List[Dict[str, Any]]:
        """Get full signature log for diversity analysis."""
        return list(self._signature_log)
