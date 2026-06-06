"""
Continuous Terrain Mapper — Replace discrete classifier with continuous feature analysis.

Five-layer pipeline:
1. BaselineCalibrator: Convert raw features to percentile ranks using baseline distribution
2. PairDivergenceMonitor: Track correlated feature pairs for anomalous divergence
3. TrajectoryTracker: Compute step-over-step deltas for trajectory analysis
4. AnomalyScorer: Weighted combination of extremity, divergence, and acceleration
5. TextComposer: Generate dynamic intervention text from terrain profile

Key insight: normally-correlated feature pairs (knn_std/ridge_prox at r=0.995,
knn_mean/knn_max at r=0.860) should be monitored for divergence. When features
that normally agree start disagreeing, that divergence is a powerful anomaly signal.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from mirrorfield.geometry.features import compute_knn_features, FEATURE_NAMES

logger = logging.getLogger(__name__)

# Feature index constants (matches FEATURE_NAMES order)
IDX_KNN_MEAN = 0
IDX_KNN_STD = 1
IDX_KNN_MIN = 2
IDX_KNN_MAX = 3
IDX_CURVATURE = 4
IDX_RIDGE = 5
IDX_DIST_NEAREST = 6

# Short names for text generation
SHORT_NAMES = {
    'knn_mean_distance': 'neighbor distance (mean)',
    'knn_std_distance': 'neighbor spread',
    'knn_min_distance': 'nearest neighbor distance',
    'knn_max_distance': 'farthest neighbor distance',
    'local_curvature': 'local curvature',
    'ridge_proximity': 'ridge proximity',
    'dist_to_ref_nearest': 'distance to reference',
}

# Correlated pairs for divergence monitoring
# (feature_a_idx, feature_b_idx, description, baseline_r, is_integrity_check)
CORRELATED_PAIRS = [
    (IDX_KNN_STD, IDX_RIDGE, 'knn_std vs ridge_prox', 0.995, False),
    (IDX_KNN_MEAN, IDX_KNN_MAX, 'knn_mean vs knn_max', 0.860, False),
    (IDX_KNN_MIN, IDX_DIST_NEAREST, 'knn_min vs dist_nearest', 1.000, True),
]

# Delta feature indices for trajectory tracking
DELTA_FEATURES = [
    (IDX_KNN_MEAN, 'delta_knn_mean'),
    (IDX_KNN_STD, 'delta_knn_std'),
    (IDX_CURVATURE, 'delta_curvature'),
    (IDX_RIDGE, 'delta_ridge'),
    (IDX_DIST_NEAREST, 'delta_dist_nearest'),
]


@dataclass
class TerrainProfile:
    """Complete terrain analysis for a single reasoning step."""
    raw_features: Dict[str, float]
    percentiles: Dict[str, float]
    pair_divergences: Dict[str, float]       # pair_name -> z-score
    delta_percentiles: Dict[str, float]      # delta_name -> percentile of delta
    anomaly_score: float                     # 0-1 composite
    anomaly_breakdown: Dict[str, float]      # component scores
    intervention_text: str                   # generated text (empty if no intervention)
    intervention_intensity: str              # "none", "observation", "detailed"

    def to_dict(self) -> dict:
        return {
            'raw_features': self.raw_features,
            'percentiles': self.percentiles,
            'pair_divergences': self.pair_divergences,
            'delta_percentiles': self.delta_percentiles,
            'anomaly_score': self.anomaly_score,
            'anomaly_breakdown': self.anomaly_breakdown,
            'intervention_text': self.intervention_text,
            'intervention_intensity': self.intervention_intensity,
        }


class BaselineCalibrator:
    """Layer 1: Convert raw features to percentile ranks using baseline distribution."""

    def __init__(self, baseline_features: np.ndarray):
        """
        Args:
            baseline_features: (N, 7) feature matrix from GLM baseline (64 steps).
        """
        assert baseline_features.ndim == 2 and baseline_features.shape[1] == 7, (
            f"Expected (N, 7) baseline features, got {baseline_features.shape}"
        )
        self.baseline_features = baseline_features
        self.n_baseline = len(baseline_features)
        logger.info("BaselineCalibrator initialized with %d baseline observations", self.n_baseline)

    def to_percentiles(self, features: np.ndarray) -> np.ndarray:
        """Convert a 7-element feature vector to percentile ranks.

        Args:
            features: (7,) raw feature vector.

        Returns:
            (7,) percentile ranks in [0, 100].
        """
        percentiles = np.zeros(7, dtype=np.float64)
        for i in range(7):
            percentiles[i] = scipy_stats.percentileofscore(
                self.baseline_features[:, i], features[i], kind='rank'
            )
        return percentiles


class PairDivergenceMonitor:
    """Layer 2: Monitor correlated feature pairs for anomalous divergence."""

    def __init__(self, baseline_features: np.ndarray):
        """Fit linear regressions on baseline data for each correlated pair.

        Args:
            baseline_features: (N, 7) baseline feature matrix.
        """
        self._models: Dict[str, dict] = {}

        for a_idx, b_idx, name, expected_r, is_integrity in CORRELATED_PAIRS:
            x = baseline_features[:, a_idx]
            y = baseline_features[:, b_idx]

            # Fit linear regression: y = slope * x + intercept
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

            # Compute residual std
            predicted = slope * x + intercept
            residuals = y - predicted
            residual_std = float(np.std(residuals))

            self._models[name] = {
                'a_idx': a_idx,
                'b_idx': b_idx,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'residual_std': max(residual_std, 1e-10),  # prevent div-by-zero
                'is_integrity': is_integrity,
            }

            logger.info(
                "PairDivergenceMonitor: %s — r=%.4f, residual_std=%.6f%s",
                name, r_value, residual_std,
                " [INTEGRITY CHECK]" if is_integrity else "",
            )

    def compute_divergences(self, features: np.ndarray) -> Dict[str, float]:
        """Compute residual z-scores for each pair.

        Args:
            features: (7,) raw feature vector.

        Returns:
            Dict mapping pair name -> z-score (divergence in baseline residual std units).
        """
        divergences = {}
        for name, model in self._models.items():
            x = features[model['a_idx']]
            y = features[model['b_idx']]
            predicted_y = model['slope'] * x + model['intercept']
            residual = y - predicted_y
            z_score = float(residual / model['residual_std'])
            divergences[name] = z_score

            # Integrity check: knn_min vs dist_nearest should never diverge
            if model['is_integrity'] and abs(z_score) > 0.01:
                logger.warning(
                    "INTEGRITY WARNING: %s divergence=%.4f (expected ~0). "
                    "Data integrity issue detected.",
                    name, z_score,
                )

        return divergences

    def get_scoring_divergences(self, divergences: Dict[str, float]) -> Dict[str, float]:
        """Return only non-integrity divergences for anomaly scoring."""
        return {
            name: z for name, z in divergences.items()
            if not self._models[name]['is_integrity']
        }


class TrajectoryTracker:
    """Layer 3: Track step-over-step feature deltas."""

    def __init__(self, baseline_features: np.ndarray, n_steps: int = 4):
        """Build baseline delta distributions from the baseline feature matrix.

        The baseline has 16 tasks x 4 steps. Deltas are computed between
        consecutive steps within each task (3 transitions per task = 48 deltas).

        Args:
            baseline_features: (N, 7) baseline feature matrix (N = n_tasks * n_steps).
            n_steps: Steps per task (default: 4).
        """
        self._prev_features: Optional[np.ndarray] = None

        n_total = len(baseline_features)
        n_tasks = n_total // n_steps

        # Collect baseline deltas
        baseline_deltas = []
        for task_i in range(n_tasks):
            start = task_i * n_steps
            for step_j in range(1, n_steps):
                prev = baseline_features[start + step_j - 1]
                curr = baseline_features[start + step_j]
                delta = curr - prev
                # Extract only the delta features we track
                delta_vec = np.array([delta[idx] for idx, _ in DELTA_FEATURES])
                baseline_deltas.append(delta_vec)

        self._baseline_deltas = np.array(baseline_deltas)  # (48, 5)
        logger.info(
            "TrajectoryTracker: %d baseline delta observations",
            len(self._baseline_deltas),
        )

    def reset(self):
        """Reset trajectory state between tasks."""
        self._prev_features = None

    def compute_deltas(self, features: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        """Compute deltas and their percentile ranks.

        Args:
            features: (7,) current raw feature vector.

        Returns:
            Tuple of:
                - raw deltas (5,) or None if first step
                - dict mapping delta_name -> percentile rank
        """
        if self._prev_features is None:
            self._prev_features = features.copy()
            return None, {}

        raw_deltas = np.array([
            features[idx] - self._prev_features[idx]
            for idx, _ in DELTA_FEATURES
        ])

        delta_percentiles = {}
        for i, (_, name) in enumerate(DELTA_FEATURES):
            delta_percentiles[name] = float(scipy_stats.percentileofscore(
                self._baseline_deltas[:, i], raw_deltas[i], kind='rank'
            ))

        self._prev_features = features.copy()
        return raw_deltas, delta_percentiles


class AnomalyScorer:
    """Layer 4: Weighted anomaly scoring from percentiles, divergences, and deltas."""

    def __init__(
        self,
        weight_percentile: float = 0.40,
        weight_divergence: float = 0.35,
        weight_trajectory: float = 0.25,
        threshold_observation: float = 0.45,
        threshold_detailed: float = 0.65,
    ):
        self.weight_percentile = weight_percentile
        self.weight_divergence = weight_divergence
        self.weight_trajectory = weight_trajectory
        self.threshold_observation = threshold_observation
        self.threshold_detailed = threshold_detailed

    def score(
        self,
        percentiles: np.ndarray,
        scoring_divergences: Dict[str, float],
        delta_percentiles: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted anomaly score.

        Args:
            percentiles: (7,) percentile ranks in [0, 100].
            scoring_divergences: Non-integrity pair z-scores.
            delta_percentiles: Delta percentile ranks in [0, 100].

        Returns:
            Tuple of (anomaly_score, breakdown_dict).
        """
        # Component 1: Percentile extremity
        # mean of |percentile - 50| / 50 across all features
        extremity = float(np.mean(np.abs(percentiles - 50.0) / 50.0))

        # Component 2: Pair divergence — sigmoid of max z-score
        if scoring_divergences:
            max_z = max(abs(z) for z in scoring_divergences.values())
            # Sigmoid centered at 2.5 sigma, steepness 2
            divergence_score = float(1.0 / (1.0 + np.exp(-2.0 * (max_z - 2.5))))
        else:
            max_z = 0.0
            divergence_score = 0.0

        # Component 3: Trajectory acceleration — mean extremity of delta percentiles
        if delta_percentiles:
            delta_extremity = float(np.mean([
                abs(p - 50.0) / 50.0 for p in delta_percentiles.values()
            ]))
        else:
            delta_extremity = 0.0

        # Weighted combination
        anomaly_score = (
            self.weight_percentile * extremity
            + self.weight_divergence * divergence_score
            + self.weight_trajectory * delta_extremity
        )

        breakdown = {
            'percentile_extremity': round(extremity, 4),
            'divergence_score': round(divergence_score, 4),
            'max_pair_z': round(max_z, 4),
            'trajectory_acceleration': round(delta_extremity, 4),
            'weighted_score': round(anomaly_score, 4),
        }

        return float(anomaly_score), breakdown

    def classify_intensity(self, anomaly_score: float) -> str:
        """Map anomaly score to intervention intensity."""
        if anomaly_score >= self.threshold_detailed:
            return "detailed"
        elif anomaly_score >= self.threshold_observation:
            return "observation"
        return "none"


class TextComposer:
    """Layer 5: Generate dynamic intervention text from terrain profile."""

    def compose(
        self,
        intensity: str,
        raw_features: Dict[str, float],
        percentiles: Dict[str, float],
        pair_divergences: Dict[str, float],
        delta_percentiles: Dict[str, float],
        anomaly_breakdown: Dict[str, float],
    ) -> str:
        """Generate intervention text based on terrain analysis.

        Args:
            intensity: "none", "observation", or "detailed".
            raw_features: Named feature values.
            percentiles: Named percentile ranks.
            pair_divergences: Pair name -> z-score.
            delta_percentiles: Delta name -> percentile.
            anomaly_breakdown: Scoring components.

        Returns:
            Intervention text (empty string if intensity is "none").
        """
        if intensity == "none":
            return ""

        if intensity == "observation":
            return self._compose_observation(
                raw_features, percentiles, pair_divergences,
            )
        else:
            return self._compose_detailed(
                raw_features, percentiles, pair_divergences,
                delta_percentiles, anomaly_breakdown,
            )

    def _compose_observation(
        self,
        raw_features: Dict[str, float],
        percentiles: Dict[str, float],
        pair_divergences: Dict[str, float],
    ) -> str:
        """Brief one-sentence observation."""
        parts = []

        # Find most extreme feature
        most_extreme_name = None
        most_extreme_pct = 50.0
        for name, pct in percentiles.items():
            if abs(pct - 50.0) > abs(most_extreme_pct - 50.0):
                most_extreme_name = name
                most_extreme_pct = pct

        if most_extreme_name:
            short = SHORT_NAMES.get(most_extreme_name, most_extreme_name)
            direction = "high" if most_extreme_pct > 50 else "low"
            parts.append(
                f"Your {short} is unusually {direction} "
                f"(percentile {most_extreme_pct:.0f})."
            )

        # Mention pair divergence if significant
        scoring_divs = {
            k: v for k, v in pair_divergences.items()
            if not any(
                p[4] and p[2] == k for p in CORRELATED_PAIRS
            )
        }
        if scoring_divs:
            max_pair = max(scoring_divs, key=lambda k: abs(scoring_divs[k]))
            max_z = abs(scoring_divs[max_pair])
            if max_z > 2.0:
                parts.append(
                    f"Notable divergence in {max_pair} ({max_z:.1f}\u03c3)."
                )

        parts.append("Continue as you see fit.")
        return " ".join(parts)

    def _compose_detailed(
        self,
        raw_features: Dict[str, float],
        percentiles: Dict[str, float],
        pair_divergences: Dict[str, float],
        delta_percentiles: Dict[str, float],
        anomaly_breakdown: Dict[str, float],
    ) -> str:
        """Full terrain description with up to 3 extreme features."""
        sections = []

        # Section 1: Top 3 extreme features
        sorted_features = sorted(
            percentiles.items(),
            key=lambda kv: abs(kv[1] - 50.0),
            reverse=True,
        )
        extreme_lines = []
        for name, pct in sorted_features[:3]:
            short = SHORT_NAMES.get(name, name)
            direction = "high" if pct > 50 else "low"
            extreme_lines.append(f"- {short}: percentile {pct:.0f} ({direction})")

        if extreme_lines:
            sections.append(
                "The geometric terrain of your current reasoning shows:\n"
                + "\n".join(extreme_lines)
            )

        # Section 2: Pair divergences > 2 sigma
        scoring_divs = {
            k: v for k, v in pair_divergences.items()
            if not any(p[4] and p[2] == k for p in CORRELATED_PAIRS)
        }
        significant_pairs = [
            (name, z) for name, z in scoring_divs.items() if abs(z) > 2.0
        ]
        if significant_pairs:
            pair_lines = []
            for name, z in sorted(significant_pairs, key=lambda x: abs(x[1]), reverse=True):
                direction = "above" if z > 0 else "below"
                pair_lines.append(
                    f"- {name}: {abs(z):.1f}\u03c3 {direction} expected relationship"
                )
            sections.append(
                "Feature pairs that normally correlate are diverging:\n"
                + "\n".join(pair_lines)
            )

        # Section 3: Trajectory direction
        if delta_percentiles:
            extreme_deltas = sorted(
                delta_percentiles.items(),
                key=lambda kv: abs(kv[1] - 50.0),
                reverse=True,
            )
            if extreme_deltas:
                top_delta_name, top_delta_pct = extreme_deltas[0]
                if abs(top_delta_pct - 50.0) > 25.0:
                    direction = "accelerating" if top_delta_pct > 50 else "decelerating"
                    clean_name = top_delta_name.replace("delta_", "")
                    sections.append(
                        f"Trajectory note: {clean_name} is {direction} "
                        f"relative to baseline patterns."
                    )

        # Section 4: Interpretive sentence for high pair divergence
        if significant_pairs:
            max_pair_name, max_z = max(significant_pairs, key=lambda x: abs(x[1]))
            if 'knn_std vs ridge_prox' in max_pair_name:
                sections.append(
                    "This suggests knn_mean is behaving unusually relative to "
                    "spread and density gradient — the neighborhood structure "
                    "may be shifting in a way not seen in baseline reasoning."
                )
            elif 'knn_mean vs knn_max' in max_pair_name:
                sections.append(
                    "This suggests asymmetric neighborhood stretch — "
                    "the farthest neighbors are moving differently than the bulk."
                )

        return "\n\n".join(sections) if sections else ""


class TerrainMapper:
    """Orchestrator: owns all 5 layers and manages cooldown/reset."""

    def __init__(
        self,
        baseline_features: np.ndarray,
        reference_embeddings: np.ndarray,
        k: int = 50,
        cooldown_steps: int = 1,
        threshold_observation: float = 0.45,
        threshold_detailed: float = 0.65,
    ):
        """
        Args:
            baseline_features: (N, 7) feature matrix from GLM baseline.
            reference_embeddings: (N_ref, D) reference corpus embeddings.
            k: Number of neighbors for kNN feature extraction.
            cooldown_steps: Minimum steps between interventions.
            threshold_observation: Anomaly score threshold for observation.
            threshold_detailed: Anomaly score threshold for detailed.
        """
        self.reference_embeddings = reference_embeddings
        self.k = min(k, len(reference_embeddings) - 1)
        self.cooldown_steps = cooldown_steps

        # Layer 1
        self.calibrator = BaselineCalibrator(baseline_features)

        # Layer 2
        self.divergence_monitor = PairDivergenceMonitor(baseline_features)

        # Layer 3
        self.trajectory_tracker = TrajectoryTracker(baseline_features)

        # Layer 4
        self.scorer = AnomalyScorer(
            threshold_observation=threshold_observation,
            threshold_detailed=threshold_detailed,
        )

        # Layer 5
        self.composer = TextComposer()

        # State
        self._steps_since_intervention = 999  # allow first step
        self._total_evaluations = 0
        self._total_observations = 0
        self._total_detailed = 0
        self._anomaly_scores: List[float] = []

    def reset(self):
        """Reset between tasks. Clears trajectory state and cooldown."""
        self.trajectory_tracker.reset()
        self._steps_since_intervention = 999

    def evaluate(self, step_embedding: np.ndarray, step_index: int) -> TerrainProfile:
        """Run the full five-layer pipeline on a single step.

        Args:
            step_embedding: (D,) or (1, D) embedding vector for this step.
            step_index: Index of the step in the reasoning sequence.

        Returns:
            TerrainProfile with all analysis results.
        """
        # Extract features
        if step_embedding.ndim == 1:
            step_embedding = step_embedding.reshape(1, -1)

        features_matrix, _ = compute_knn_features(
            step_embedding, self.reference_embeddings, k=self.k,
            check_collapse=False,
        )
        raw_vec = features_matrix[0]  # (7,)

        # Named features
        raw_features = {
            name: float(raw_vec[i]) for i, name in enumerate(FEATURE_NAMES)
        }

        # Layer 1: Percentiles
        percentile_vec = self.calibrator.to_percentiles(raw_vec)
        percentiles = {
            name: float(percentile_vec[i]) for i, name in enumerate(FEATURE_NAMES)
        }

        # Layer 2: Pair divergences
        pair_divergences = self.divergence_monitor.compute_divergences(raw_vec)

        # Layer 3: Trajectory deltas
        _, delta_percentiles = self.trajectory_tracker.compute_deltas(raw_vec)

        # Layer 4: Anomaly scoring
        scoring_divs = self.divergence_monitor.get_scoring_divergences(pair_divergences)
        anomaly_score, anomaly_breakdown = self.scorer.score(
            percentile_vec, scoring_divs, delta_percentiles,
        )

        # Determine intensity (with cooldown)
        raw_intensity = self.scorer.classify_intensity(anomaly_score)
        if raw_intensity != "none" and self._steps_since_intervention < self.cooldown_steps:
            intensity = "none"
            anomaly_breakdown['cooldown_suppressed'] = True
        else:
            intensity = raw_intensity

        # Layer 5: Text composition
        intervention_text = self.composer.compose(
            intensity, raw_features, percentiles,
            pair_divergences, delta_percentiles, anomaly_breakdown,
        )

        # Update state
        self._total_evaluations += 1
        self._anomaly_scores.append(anomaly_score)
        if intensity != "none":
            self._steps_since_intervention = 0
            if intensity == "observation":
                self._total_observations += 1
            else:
                self._total_detailed += 1
        else:
            self._steps_since_intervention += 1

        return TerrainProfile(
            raw_features=raw_features,
            percentiles=percentiles,
            pair_divergences={k: round(v, 4) for k, v in pair_divergences.items()},
            delta_percentiles={k: round(v, 2) for k, v in delta_percentiles.items()},
            anomaly_score=round(anomaly_score, 4),
            anomaly_breakdown=anomaly_breakdown,
            intervention_text=intervention_text,
            intervention_intensity=intensity,
        )

    def get_stats(self) -> dict:
        """Return summary statistics for all evaluations so far."""
        scores = self._anomaly_scores
        return {
            'total_evaluations': self._total_evaluations,
            'total_observations': self._total_observations,
            'total_detailed': self._total_detailed,
            'total_interventions': self._total_observations + self._total_detailed,
            'intervention_rate': (
                (self._total_observations + self._total_detailed) / max(1, self._total_evaluations)
            ),
            'anomaly_score_mean': float(np.mean(scores)) if scores else 0.0,
            'anomaly_score_std': float(np.std(scores)) if scores else 0.0,
            'anomaly_score_min': float(np.min(scores)) if scores else 0.0,
            'anomaly_score_max': float(np.max(scores)) if scores else 0.0,
        }


def extract_baseline_features(baseline_results_path: str) -> np.ndarray:
    """Extract the (N, 7) feature matrix from a GLM baseline results JSON.

    Args:
        baseline_results_path: Path to glm_baseline_results.json.

    Returns:
        (N, 7) numpy array of raw feature vectors (N = n_tasks * steps_per_task).
    """
    import json
    with open(baseline_results_path) as f:
        data = json.load(f)

    features = []
    for result in data['results']:
        trace = result.get('trace', {})
        for step in trace.get('steps', []):
            vec = step.get('raw_feature_vector')
            if vec is not None and len(vec) == 7:
                features.append(vec)

    feature_matrix = np.array(features, dtype=np.float32)
    logger.info("Extracted baseline features: %s", feature_matrix.shape)
    return feature_matrix
