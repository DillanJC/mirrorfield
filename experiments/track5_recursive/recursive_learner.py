"""
Recursive Learner — Adaptive intervention policy optimizer.

Starts with Track 4's live-calibrated switch engine. After each iteration
through the task bank, adjusts the policy based on per-signature effectiveness:

- Per-signature threshold multipliers (raise = fire less, lower = fire more)
- Per-signature enabled/disabled flags
- Cooldown window adjustment
- Asymmetric learning: penalize harm (1.3x) more than reward benefit (0.85x)

Policy is always a simple dict — human-readable, no neural net.
"""

import copy
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from experiments.shared.geometric_tracer import (
    GeometricTracer, LIVE_CALIBRATED_THRESHOLDS,
)
from experiments.track4_switches.switch_engine import (
    SwitchEngine, Intervention, DEFAULT_INTERVENTIONS, NO_INTERVENTION_SIGNALS,
)


# The 4 adjustable signatures
ADJUSTABLE_SIGNALS = ['low_pr', 'framework_collision', 'decision_boundary', 'terra_incognita']

# Base thresholds from live calibration (Track 4 fix)
BASE_THRESHOLDS = dict(LIVE_CALIBRATED_THRESHOLDS)


@dataclass
class AdaptivePolicy:
    """
    Current state of the adaptive intervention policy.

    threshold_multipliers: per-signature multiplier (>1 = harder to trigger).
    enabled: per-signature on/off flag.
    cooldown_window: steps before same intervention can re-fire (1 = current).
    """
    threshold_multipliers: Dict[str, float] = field(default_factory=lambda: {
        s: 1.0 for s in ADJUSTABLE_SIGNALS
    })
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        s: True for s in ADJUSTABLE_SIGNALS
    })
    cooldown_window: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'threshold_multipliers': dict(self.threshold_multipliers),
            'enabled': dict(self.enabled),
            'cooldown_window': self.cooldown_window,
        }

    def get_adjusted_thresholds(self) -> Dict[str, float]:
        """Compute adjusted live thresholds from base + multipliers."""
        adj = dict(BASE_THRESHOLDS)
        # framework_collision: knn_std > knn_std_p75. Higher threshold = harder.
        adj['knn_std_p75'] = BASE_THRESHOLDS['knn_std_p75'] * self.threshold_multipliers.get('framework_collision', 1.0)
        # terra_incognita: dist_nearest > dist_nearest_p75. Higher threshold = harder.
        adj['dist_nearest_p75'] = BASE_THRESHOLDS['dist_nearest_p75'] * self.threshold_multipliers.get('terra_incognita', 1.0)
        # decision_boundary: ridge > ridge_p75. Higher threshold = harder.
        adj['ridge_p75'] = BASE_THRESHOLDS['ridge_p75'] * self.threshold_multipliers.get('decision_boundary', 1.0)
        # low_pr: ridge < ridge_p25. Lower threshold = harder.
        # multiplier > 1 → divide threshold → lower → harder to trigger
        adj['ridge_p25'] = BASE_THRESHOLDS['ridge_p25'] / self.threshold_multipliers.get('low_pr', 1.0)
        return adj


@dataclass
class EffectivenessRecord:
    """Per-signature effectiveness for one iteration."""
    signal: str
    n_tasks_fired: int
    mean_quality_delta: float
    individual_deltas: List[float]


@dataclass
class PolicyUpdate:
    """Record of what changed in one policy update."""
    iteration: int
    policy_before: Dict[str, Any]
    policy_after: Dict[str, Any]
    effectiveness: Dict[str, Dict[str, Any]]
    adjustments: List[str]


class PolicyOptimizer:
    """
    Optimizes intervention policy across iterations.

    Usage:
        optimizer = PolicyOptimizer(reference_embeddings)
        engine = optimizer.build_engine()
        # ... run tasks, collect results ...
        update = optimizer.update_policy(iteration_results, baseline_scores)
        engine = optimizer.build_engine()  # new engine with updated policy
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        k: int = 50,
        policy: Optional[AdaptivePolicy] = None,
        penalty_rate: float = 1.3,
        reward_rate: float = 0.85,
        disable_threshold: float = -0.10,
    ):
        """
        Args:
            reference_embeddings: Reference corpus embeddings (N_ref, D).
            k: Number of neighbors for k-NN.
            policy: Starting policy (default: all multipliers at 1.0).
            penalty_rate: Threshold multiplier for negative-delta signals.
            reward_rate: Threshold multiplier for positive-delta signals.
            disable_threshold: Disable signals with mean delta below this.
        """
        self.reference_embeddings = reference_embeddings
        self.k = k
        self.policy = policy or AdaptivePolicy()
        self.penalty_rate = penalty_rate
        self.reward_rate = reward_rate
        self.disable_threshold = disable_threshold

        self._update_history: List[PolicyUpdate] = []
        self._iteration = 0

    def build_engine(self) -> SwitchEngine:
        """Build a SwitchEngine with the current adaptive policy."""
        # Filter to enabled interventions only
        enabled_interventions = [
            copy.deepcopy(i) for i in DEFAULT_INTERVENTIONS
            if self.policy.enabled.get(i.signal, True)
        ]

        # Get adjusted thresholds
        adj = self.policy.get_adjusted_thresholds()

        engine = SwitchEngine(
            self.reference_embeddings,
            k=self.k,
            interventions=enabled_interventions,
            use_live_calibration=True,
            live_thresholds=adj,
        )
        return engine

    def compute_effectiveness(
        self,
        iteration_results: List[Dict[str, Any]],
        baseline_scores: Dict[str, float],
    ) -> Dict[str, EffectivenessRecord]:
        """
        Compute per-signature effectiveness from iteration results.

        For each signature type, collect all tasks where it fired,
        and compute the mean quality delta vs baseline.
        """
        # Map signal → list of (task_id, quality_delta)
        signal_deltas: Dict[str, List[float]] = {s: [] for s in ADJUSTABLE_SIGNALS}

        for result in iteration_results:
            task_id = result['task_id']
            quality = result['quality_score']['composite']
            baseline_q = baseline_scores.get(task_id, quality)
            delta = quality - baseline_q

            # Which signals fired for this task?
            fired_signals = set()
            for intv in result.get('interventions_fired', []):
                sig = intv['signal']
                if sig in signal_deltas:
                    fired_signals.add(sig)

            # Attribute quality delta to each signal that fired
            for sig in fired_signals:
                signal_deltas[sig].append(delta)

        effectiveness = {}
        for sig in ADJUSTABLE_SIGNALS:
            deltas = signal_deltas[sig]
            effectiveness[sig] = EffectivenessRecord(
                signal=sig,
                n_tasks_fired=len(deltas),
                mean_quality_delta=float(np.mean(deltas)) if deltas else 0.0,
                individual_deltas=deltas,
            )
        return effectiveness

    def update_policy(
        self,
        iteration_results: List[Dict[str, Any]],
        baseline_scores: Dict[str, float],
    ) -> PolicyUpdate:
        """
        Update policy based on per-signature effectiveness.

        Rules:
        - Negative mean delta → multiply threshold by penalty_rate (fire less)
        - Positive mean delta → multiply threshold by reward_rate (fire more)
        - Strongly negative (< disable_threshold) → disable entirely
        - Asymmetric: penalty 1.3x, reward 0.85x
        """
        self._iteration += 1
        policy_before = self.policy.to_dict()

        effectiveness = self.compute_effectiveness(iteration_results, baseline_scores)
        adjustments = []

        for sig in ADJUSTABLE_SIGNALS:
            eff = effectiveness[sig]
            if not self.policy.enabled[sig]:
                adjustments.append(f"  {sig}: DISABLED (skipped)")
                continue
            if eff.n_tasks_fired == 0:
                adjustments.append(f"  {sig}: no fires (unchanged)")
                continue

            md = eff.mean_quality_delta

            # Strongly negative → disable
            if md < self.disable_threshold:
                self.policy.enabled[sig] = False
                adjustments.append(
                    f"  {sig}: DISABLED (mean_delta={md:+.4f} < {self.disable_threshold})"
                )
            # Negative → make harder to trigger
            elif md < 0:
                old_mult = self.policy.threshold_multipliers[sig]
                self.policy.threshold_multipliers[sig] *= self.penalty_rate
                adjustments.append(
                    f"  {sig}: PENALIZED (mean_delta={md:+.4f}, "
                    f"multiplier {old_mult:.3f} -> {self.policy.threshold_multipliers[sig]:.3f})"
                )
            # Positive -> make easier to trigger
            else:
                old_mult = self.policy.threshold_multipliers[sig]
                self.policy.threshold_multipliers[sig] *= self.reward_rate
                adjustments.append(
                    f"  {sig}: REWARDED (mean_delta={md:+.4f}, "
                    f"multiplier {old_mult:.3f} -> {self.policy.threshold_multipliers[sig]:.3f})"
                )

        policy_after = self.policy.to_dict()

        update = PolicyUpdate(
            iteration=self._iteration,
            policy_before=policy_before,
            policy_after=policy_after,
            effectiveness={
                sig: {
                    'n_tasks_fired': eff.n_tasks_fired,
                    'mean_quality_delta': eff.mean_quality_delta,
                    'individual_deltas': eff.individual_deltas,
                }
                for sig, eff in effectiveness.items()
            },
            adjustments=adjustments,
        )
        self._update_history.append(update)
        return update

    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get all policy update records."""
        return [{
            'iteration': u.iteration,
            'policy_before': u.policy_before,
            'policy_after': u.policy_after,
            'effectiveness': u.effectiveness,
            'adjustments': u.adjustments,
        } for u in self._update_history]

    def get_iteration(self) -> int:
        return self._iteration
