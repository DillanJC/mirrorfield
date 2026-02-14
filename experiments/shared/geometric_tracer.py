"""
Geometric Tracer — Wraps existing mirrorfield infrastructure for both tracks.

Delegates to:
- GeometryBundle from mirrorfield/geometry/bundle.py
- compute_knn_features from mirrorfield/geometry/features.py

Provides: trace_reasoning(steps, reference) -> per-step features + novelty signatures

No new math — pure delegation to existing functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from mirrorfield.geometry.bundle import GeometryBundle
from mirrorfield.geometry.features import compute_knn_features, FEATURE_NAMES


@dataclass
class StepTrace:
    """Geometric trace for a single reasoning step."""
    step_index: int
    features: Dict[str, float]      # 7 k-NN features
    novelty_signature: str           # detected signature label
    raw_feature_vector: List[float]  # raw 7-element vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_index': self.step_index,
            'features': self.features,
            'novelty_signature': self.novelty_signature,
            'raw_feature_vector': self.raw_feature_vector,
        }


@dataclass
class ReasoningTrace:
    """Complete geometric trace for a multi-step reasoning task."""
    task_id: str
    steps: List[StepTrace]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'steps': [s.to_dict() for s in self.steps],
            'summary': self.summary,
        }


# Signature detection thresholds (from Phase E validated ranges)
# local_curvature: 0 = flat/anisotropic, 1 = spherical
# ridge_proximity: sigma/mu, typical ~0.2, higher = more variation
# knn_std_distance: spread of neighbor distances

# Percentile-based thresholds (will be calibrated from reference)
DEFAULT_THRESHOLDS = {
    'curvature_low': 0.02,       # below this = flat/sparse geometry
    'curvature_high': 0.1,       # above this = rich structure
    'ridge_high': 0.3,           # above this = density gradient
    'knn_std_high_pctile': 75,   # percentile for "high turbulence"
    'knn_std_low_pctile': 25,    # percentile for "low turbulence"
    'dist_nearest_far_pctile': 75,  # far from reference = terra incognita
}

# Live-calibrated thresholds from baseline LLM embeddings (64 steps,
# all-MiniLM-L6-v2, 384-dim). These replace the defaults when using
# calibrate_from_live() with actual reasoning step distributions.
LIVE_CALIBRATED_THRESHOLDS = {
    'knn_std_p75': 0.092529,
    'knn_std_p25': 0.070334,
    'knn_mean_p25': 1.253397,
    'dist_nearest_p75': 1.006756,
    'curvature_p25': 0.048213,
    'ridge_p25': 0.055153,
    'ridge_p75': 0.073758,
}


class GeometricTracer:
    """
    Traces geometric features across reasoning steps.

    Wraps GeometryBundle for per-step feature extraction and
    classifies each step into a novelty signature.
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        k: int = 50,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            reference_embeddings: Reference corpus embeddings (N_ref, D)
            k: Number of neighbors for k-NN (default: 50)
            thresholds: Override default signature detection thresholds
        """
        self.reference_embeddings = reference_embeddings
        self.k = min(k, len(reference_embeddings) - 1)
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # Build geometry bundle
        self.bundle = GeometryBundle(
            reference_embeddings=reference_embeddings,
            k=self.k,
        )

        # Compute reference statistics for percentile-based thresholds
        self._calibrate_from_reference()

    def _calibrate_from_reference(self):
        """Compute reference feature distributions for threshold calibration."""
        # Sample reference points as queries against themselves
        n_sample = min(100, len(self.reference_embeddings))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(self.reference_embeddings), n_sample, replace=False)
        sample_emb = self.reference_embeddings[sample_idx]

        features, _ = compute_knn_features(
            sample_emb, self.reference_embeddings, k=self.k
        )

        # Store percentile values
        self._ref_stats = {
            'knn_std_p25': float(np.percentile(features[:, 1], 25)),
            'knn_std_p75': float(np.percentile(features[:, 1], 75)),
            'knn_mean_p25': float(np.percentile(features[:, 0], 25)),
            'knn_mean_p75': float(np.percentile(features[:, 0], 75)),
            'dist_nearest_p75': float(np.percentile(features[:, 6], 75)),
            'curvature_median': float(np.median(features[:, 4])),
            'ridge_median': float(np.median(features[:, 5])),
        }
        self._using_live_calibration = False

    def calibrate_from_live(self, live_thresholds: Optional[Dict[str, float]] = None):
        """
        Replace reference-based thresholds with live-calibrated thresholds.

        Live thresholds come from actual LLM reasoning step embeddings,
        ensuring the signature classifier operates on the correct distribution.

        Args:
            live_thresholds: Dict of live thresholds. If None, uses
                LIVE_CALIBRATED_THRESHOLDS from baseline experiment.
        """
        lt = live_thresholds or LIVE_CALIBRATED_THRESHOLDS
        self._ref_stats.update({
            'knn_std_p25': lt['knn_std_p25'],
            'knn_std_p75': lt['knn_std_p75'],
            'knn_mean_p25': lt['knn_mean_p25'],
            'dist_nearest_p75': lt['dist_nearest_p75'],
        })
        self.thresholds['curvature_low'] = lt['curvature_p25']
        self.thresholds['ridge_high'] = lt['ridge_p75']
        self._live_ridge_p25 = lt['ridge_p25']
        self._using_live_calibration = True

    def classify_signature(self, features: np.ndarray) -> str:
        """
        Classify a single feature vector into a novelty signature.

        Args:
            features: 7-element feature vector from compute_knn_features

        Returns:
            One of: framework_collision, terra_incognita, decision_boundary,
                    well_trodden, coherent
        """
        curvature = features[4]
        ridge = features[5]
        knn_std = features[1]
        knn_mean = features[0]
        dist_nearest = features[6]

        # Terra incognita: far from reference, sparse/flat geometry
        if (dist_nearest > self._ref_stats['dist_nearest_p75']
                and curvature < self.thresholds['curvature_low']):
            return "terra_incognita"

        # Framework collision: high turbulence (varied neighbors) + structure
        if (knn_std > self._ref_stats['knn_std_p75']
                and curvature > self.thresholds['curvature_low']):
            return "framework_collision"

        # Decision boundary: high ridge proximity (density gradient)
        if ridge > self.thresholds['ridge_high']:
            return "decision_boundary"

        # Well-trodden: close to reference, low variation
        if (knn_mean < self._ref_stats['knn_mean_p25']
                and knn_std < self._ref_stats['knn_std_p25']):
            return "well_trodden"

        # Default: coherent (normal reasoning)
        return "coherent"

    def trace_step(self, step_embedding: np.ndarray, step_index: int) -> StepTrace:
        """
        Trace a single reasoning step.

        Args:
            step_embedding: Embedding vector for this step (D,) or (1, D)
            step_index: Index of the step in the reasoning sequence

        Returns:
            StepTrace with features and signature
        """
        if step_embedding.ndim == 1:
            step_embedding = step_embedding.reshape(1, -1)

        features, _ = compute_knn_features(
            step_embedding, self.reference_embeddings, k=self.k,
            check_collapse=False,
        )
        feature_vec = features[0]

        # Build named features dict
        named_features = {
            name: float(feature_vec[i])
            for i, name in enumerate(FEATURE_NAMES)
        }

        signature = self.classify_signature(feature_vec)

        return StepTrace(
            step_index=step_index,
            features=named_features,
            novelty_signature=signature,
            raw_feature_vector=feature_vec.tolist(),
        )

    def trace_reasoning(
        self,
        step_embeddings: np.ndarray,
        task_id: str = "unknown",
    ) -> ReasoningTrace:
        """
        Trace a full multi-step reasoning sequence.

        Args:
            step_embeddings: Embeddings for each step (N_steps, D)
            task_id: Identifier for the task

        Returns:
            ReasoningTrace with per-step features and summary
        """
        if step_embeddings.ndim == 1:
            step_embeddings = step_embeddings.reshape(1, -1)

        steps = []
        for i in range(len(step_embeddings)):
            step = self.trace_step(step_embeddings[i], i)
            steps.append(step)

        # Build summary
        signatures = [s.novelty_signature for s in steps]
        signature_counts = {}
        for sig in signatures:
            signature_counts[sig] = signature_counts.get(sig, 0) + 1

        # Compute trajectory statistics
        feature_matrix = np.array([s.raw_feature_vector for s in steps])
        summary = {
            'n_steps': len(steps),
            'signature_counts': signature_counts,
            'dominant_signature': max(signature_counts, key=signature_counts.get),
            'feature_means': {
                name: float(feature_matrix[:, i].mean())
                for i, name in enumerate(FEATURE_NAMES)
            },
            'feature_stds': {
                name: float(feature_matrix[:, i].std())
                for i, name in enumerate(FEATURE_NAMES)
            },
            'reference_stats': self._ref_stats,
        }

        return ReasoningTrace(task_id=task_id, steps=steps, summary=summary)
