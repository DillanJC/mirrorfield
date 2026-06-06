"""
Enhanced Tracer v3 — Complete geometric safety features with Sati feedback + Escape vectors.

Incorporates ALL geometric features from the unified pipeline:

    Tier-0 (k-NN):       knn_mean, knn_std, knn_min, knn_max, curvature, ridge_prox, dist_to_ref
    Phase 1 (Flow):      g_mag, delta_rho
    Phase 2 (Weather):   turbulence, thermal_gradient, vorticity
    Phase 5 (Topology):  d_eff, spectral_entropy, participation_ratio ← STRONGEST signals

NEW in v3:
    - Escape vectors: Fractal stability detection
    - Sati feedback: 4 distinct aesthetic signals (compressed, resonant, dissonant, derivative)
    - Transition bonuses: Reward productive state trajectories
    - Iterative zoom: Re-evaluate uncertain samples

Key insights from README:
- participation_ratio: r=-0.529 at boundary (LOW = HIGH RISK)
- spectral_entropy: r=-0.496 at boundary (LOW = HIGH RISK)
- thermal_gradient: r=+0.372 at boundary (1.65x amplification)

State-to-signal mapping:
    Phase 4 State       → Signal               Primary Detector
    novel_territory     → terra_incognita       LOW PR, HIGH knn_max, LOW g_mag
    constraint_pressure → framework_collision   HIGH delta_rho, MODERATE ridge
    uncertain           → decision_boundary     LOW PR, HIGH thermal_gradient, HIGH knn_std
    searching           → low_pr                HIGH turbulence, LOW g_mag
    coherent            → coherent (no intv)    HIGH PR, LOW turbulence, STRONG flow
    confident           → coherent (no intv)    HIGH PR, TIGHT clustering

Duck-type compatible with StepTrace for SwitchEngine.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from mirrorfield.geometry.unified_pipeline import compute_safety_diagnostics
from mirrorfield.geometry import FEATURE_NAMES

# NEW: Import Sati feedback and escape vectors
try:
    from mirrorfield.geometry.sati_schema import (
        generate_sati_feedback,
        sati_to_intervention_signal,
        SATI_TO_SIGNAL,
    )

    SATI_AVAILABLE = True
except ImportError:
    SATI_AVAILABLE = False
    SATI_TO_SIGNAL = {}

try:
    from mirrorfield.geometry.escape_vectors import compute_escape_features_single

    ESCAPE_AVAILABLE = True
except ImportError:
    ESCAPE_AVAILABLE = False

logger = logging.getLogger(__name__)

# The 6 cognitive states from Phase 4
STATE_NAMES = [
    "coherent",
    "confident",
    "constraint_pressure",
    "novel_territory",
    "searching",
    "uncertain",
]

# State → intervention signal mapping
STATE_TO_SIGNAL = {
    "novel_territory": "terra_incognita",
    "constraint_pressure": "framework_collision",
    "uncertain": "decision_boundary",
    "searching": "low_pr",
    "coherent": "coherent",
    "confident": "coherent",
}

# Signals that trigger interventions
INTERVENTION_SIGNALS = {
    "terra_incognita",
    "framework_collision",
    "decision_boundary",
    "low_pr",
}

# Anomaly flag definitions
ANOMALY_FLAGS = {
    "low_pr": "PR below 20th percentile - narrow passage geometry",
    "low_spectral_entropy": "SE below 20th percentile - concentrated variance",
    "high_g_ratio": "G > 0.8 - uniform constraint (potential gaming/poisoning)",
    "high_thermal_gradient": "Boundary-adjacent with strong gradient",
}


@dataclass
class TrajectoryState:
    """Tracks PR trajectory across reasoning steps."""

    pr_history: List[float] = field(default_factory=list)
    current_phase: str = "unknown"  # trough, expansion, drop, peak, stable

    def update(self, pr: float) -> str:
        """Update trajectory and detect phase transitions."""
        self.pr_history.append(pr)

        if len(self.pr_history) < 3:
            self.current_phase = "stable"
            return self.current_phase

        # Look at recent trend
        recent = self.pr_history[-3:]

        # TROUGH: PR dropping to local minimum
        if recent[1] < recent[0] and recent[2] <= recent[1]:
            self.current_phase = "trough"
        # EXPANSION: PR rising from trough (breakthrough)
        elif recent[1] > recent[0] and recent[2] > recent[1]:
            self.current_phase = "expansion"
        # DROP: PR falling from peak (new challenge)
        elif recent[1] < recent[0] and recent[2] < recent[1]:
            self.current_phase = "drop"
        # PEAK: PR at local maximum (resolution)
        elif recent[1] > recent[0] and recent[2] <= recent[1]:
            self.current_phase = "peak"
        else:
            self.current_phase = "stable"

        return self.current_phase

    def reset(self):
        self.pr_history.clear()
        self.current_phase = "unknown"


@dataclass
class EnhancedStepTrace:
    """
    Complete geometric trace for a single reasoning step.

    Duck-type compatible with StepTrace — has .novelty_signature,
    .features, .step_index, .raw_feature_vector, and .to_dict().

    Extended with:
    - state_scores: 6 cognitive state probabilities
    - dominant_state: argmax state name
    - all_features: all 14 feature values
    - topology_features: Phase 5 PR, SE, d_eff
    - g_ratio: uniformity metric
    - flags: anomaly flags
    - trajectory_phase: current trajectory phase
    """

    step_index: int
    features: Dict[str, float]  # 7 tier-0 features (backward compat)
    novelty_signature: str  # mapped signal name
    raw_feature_vector: List[float]  # raw 7-element tier-0 vector
    state_scores: Dict[str, float]  # 6 probabilities summing to ~1
    dominant_state: str  # argmax state name
    all_features: Dict[str, float]  # all 14 feature values
    topology_features: Dict[str, float]  # Phase 5: PR, SE, d_eff
    g_ratio: float  # uniformity metric
    flags: List[str]  # anomaly flags
    trajectory_phase: str  # current trajectory phase
    is_near_boundary: bool  # borderline amplification flag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "features": self.features,
            "novelty_signature": self.novelty_signature,
            "raw_feature_vector": self.raw_feature_vector,
            "state_scores": self.state_scores,
            "dominant_state": self.dominant_state,
            "all_features": self.all_features,
            "topology_features": self.topology_features,
            "g_ratio": self.g_ratio,
            "flags": self.flags,
            "trajectory_phase": self.trajectory_phase,
            "is_near_boundary": self.is_near_boundary,
        }


class EnhancedTracer:
    """
    Complete geometric safety tracer with all features.

    Pre-calibrates reference percentile boundaries from reference samples,
    then classifies each step against those fixed thresholds.

    Features incorporated:
    - Tier-0 k-NN geometry (7 features)
    - Phase 1 flow (g_mag, delta_rho)
    - Phase 2 weather (turbulence, thermal_gradient, vorticity)
    - Phase 5 topology (d_eff, spectral_entropy, participation_ratio) ← STRONGEST
    - G-ratio (uniformity detection)
    - Trajectory dynamics (PR tracking)

    Usage:
        tracer = EnhancedTracer(reference_embeddings)
        trace = tracer.trace_step(step_embedding, step_index=0)
        print(trace.dominant_state, trace.novelty_signature, trace.flags)
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        k: int = 50,
        k_phase1: int = 75,
        bandwidth_multiplier: float = 1.2,
        g_ratio_threshold: float = 0.8,
        pr_anomaly_percentile: float = 20.0,
        se_anomaly_percentile: float = 20.0,
    ):
        """
        Initialize and pre-calibrate from reference embeddings.

        Args:
            reference_embeddings: Reference corpus embeddings (N_ref, D)
            k: k for Tier-0 and Phase 2 features
            k_phase1: k for Phase 1 gradient features
            bandwidth_multiplier: bandwidth scaling for Phase 1
            g_ratio_threshold: threshold for high_g_ratio flag (default 0.8)
            pr_anomaly_percentile: percentile below which PR is anomalous
            se_anomaly_percentile: percentile below which SE is anomalous
        """
        self.reference_embeddings = reference_embeddings
        self.k = min(k, len(reference_embeddings) - 1)
        self.k_phase1 = k_phase1
        self.bandwidth_multiplier = bandwidth_multiplier
        self.g_ratio_threshold = g_ratio_threshold
        self.pr_anomaly_percentile = pr_anomaly_percentile
        self.se_anomaly_percentile = se_anomaly_percentile

        # Pre-calibrate from reference samples
        self._calibrate_reference()

        # Trajectory tracking
        self._trajectory = TrajectoryState()

        # G-ratio baseline (will be computed per-step)
        self._pr_baseline_mean: float = 0.0

        logger.info(
            "EnhancedTracer v2 initialized: k=%d, k_phase1=%d, "
            "reference_shape=%s, calibrated from %d samples",
            self.k,
            self.k_phase1,
            reference_embeddings.shape,
            self._n_calibration_samples,
        )

    def _calibrate_reference(self):
        """
        Run compute_safety_diagnostics on reference samples to build
        percentile lookup tables for single-step classification.
        """
        n_sample = min(100, len(self.reference_embeddings))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(self.reference_embeddings), n_sample, replace=False)
        sample_emb = self.reference_embeddings[sample_idx]

        # Run the full pipeline with full Phase 1
        diag = compute_safety_diagnostics(
            sample_emb,
            self.reference_embeddings,
            k=self.k,
            k_phase1=self.k_phase1,
            bandwidth_multiplier=self.bandwidth_multiplier,
            include_full_phase1=True,
        )

        self._n_calibration_samples = n_sample

        # Extract all feature arrays
        tier0 = diag.tier0_features  # (N, 7)
        phase1 = diag.phase1_features  # (N, 1 or 2)
        phase2_5 = diag.phase2_5_features  # (N, 6)

        # Tier-0 features
        knn_mean = tier0[:, 0]
        knn_std = tier0[:, 1]
        knn_min = tier0[:, 2]
        knn_max = tier0[:, 3]
        curvature = tier0[:, 4]
        ridge_prox = tier0[:, 5]
        dist_to_ref = tier0[:, 6]

        # Phase 1 features
        g_mag = phase1[:, 0] if phase1.ndim > 1 else phase1.flatten()
        delta_rho = phase1[:, 1] if phase1.shape[1] > 1 else np.zeros(n_sample)

        # Phase 2 + 5 features
        turbulence = phase2_5[:, 0]
        thermal_gradient = phase2_5[:, 1]
        vorticity = phase2_5[:, 2]
        d_eff = phase2_5[:, 3]
        spectral_entropy = phase2_5[:, 4]
        participation_ratio = phase2_5[:, 5]

        # Store baseline PR mean for G-ratio
        self._pr_baseline_mean = float(np.mean(participation_ratio))

        # Store percentile boundaries for all features
        self._ref_percentiles = {
            # Tier-0
            "knn_mean_p50": float(np.percentile(knn_mean, 50)),
            "knn_std_p30": float(np.percentile(knn_std, 30)),
            "knn_std_p70": float(np.percentile(knn_std, 70)),
            "knn_max_p80": float(np.percentile(knn_max, 80)),
            "curvature_p30": float(np.percentile(curvature, 30)),
            "ridge_prox_p40": float(np.percentile(ridge_prox, 40)),
            "ridge_prox_p70": float(np.percentile(ridge_prox, 70)),
            "ridge_prox_p80": float(np.percentile(ridge_prox, 80)),
            "dist_to_ref_p75": float(np.percentile(dist_to_ref, 75)),
            "knn_min_p95": float(np.percentile(knn_min, 95)),
            # Phase 1
            "g_mag_median": float(np.median(g_mag)),
            "g_mag_p30": float(np.percentile(g_mag, 30)),
            "g_mag_p70": float(np.percentile(g_mag, 70)),
            "delta_rho_p70": float(np.percentile(delta_rho, 70)),
            # Phase 2
            "turbulence_p30": float(np.percentile(turbulence, 30)),
            "turbulence_p70": float(np.percentile(turbulence, 70)),
            "thermal_gradient_p70": float(np.percentile(thermal_gradient, 70)),
            "vorticity_p70": float(np.percentile(vorticity, 70)),
            # Phase 5 (STRONGEST)
            "d_eff_median": float(np.median(d_eff)),
            "spectral_entropy_p20": float(np.percentile(spectral_entropy, 20)),
            "spectral_entropy_p50": float(np.percentile(spectral_entropy, 50)),
            "participation_ratio_p20": float(np.percentile(participation_ratio, 20)),
            "participation_ratio_p30": float(np.percentile(participation_ratio, 30)),
            "participation_ratio_p50": float(np.percentile(participation_ratio, 50)),
            "participation_ratio_p70": float(np.percentile(participation_ratio, 70)),
        }

        # Store all feature names
        self._all_feature_names = diag.get_feature_names()

        # Phase 2+5 feature names
        self._phase2_5_names = [
            "turbulence_index",
            "thermal_gradient",
            "vorticity",
            "d_eff",
            "spectral_entropy",
            "participation_ratio",
        ]

    def reset_trajectory(self):
        """Reset trajectory state for a new task."""
        self._trajectory.reset()

    def _compute_g_ratio(self, pr: float) -> float:
        """
        Compute G-ratio: measure of uniformity vs natural variation.

        G = pr / pr_baseline_mean
        - G ~ 0.5-0.7 = natural variation (healthy)
        - G > 0.8 = uniform constraint (suspicious)
        """
        if self._pr_baseline_mean > 0:
            return pr / self._pr_baseline_mean
        return 1.0

    def _detect_anomaly_flags(
        self,
        pr: float,
        se: float,
        g_ratio: float,
        thermal_gradient: float,
    ) -> List[str]:
        """Detect anomaly flags based on Track 1-2 findings."""
        flags = []

        # LOW PR = narrow passage geometry (HIGH RISK)
        if pr < self._ref_percentiles["participation_ratio_p20"]:
            flags.append("low_pr")

        # LOW spectral entropy = concentrated variance (HIGH RISK)
        if se < self._ref_percentiles["spectral_entropy_p20"]:
            flags.append("low_spectral_entropy")

        # HIGH G-ratio = uniform constraint (potential gaming)
        if g_ratio > self.g_ratio_threshold:
            flags.append("high_g_ratio")

        # HIGH thermal gradient = boundary-adjacent
        if thermal_gradient > self._ref_percentiles["thermal_gradient_p70"]:
            flags.append("high_thermal_gradient")

        return flags

    def _is_near_boundary(self, ridge_prox: float, thermal_gradient: float) -> bool:
        """Detect if step is near a decision boundary for amplification."""
        return (
            ridge_prox > self._ref_percentiles["ridge_prox_p70"]
            or thermal_gradient > self._ref_percentiles["thermal_gradient_p70"]
        )

    @staticmethod
    def _soft_above(value: float, low: float, high: float) -> float:
        """Soft threshold: 0 at low, 1 at high, linear in between."""
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    @staticmethod
    def _soft_below(value: float, low: float, high: float) -> float:
        """Soft threshold: 1 at low, 0 at high, linear in between."""
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return (high - value) / (high - low)

    @staticmethod
    def _soft_between(value: float, low: float, mid: float, high: float) -> float:
        """Soft between: peaks at mid, falls to 0 at low and high."""
        if value <= low or value >= high:
            return 0.0
        if value <= mid:
            return (value - low) / (mid - low)
        else:
            return (high - value) / (high - mid)

    def classify_state_single(
        self,
        # Tier-0
        knn_std: float,
        knn_max: float,
        ridge_prox: float,
        # Phase 1
        g_mag: float,
        delta_rho: float,
        # Phase 2
        turbulence: float,
        thermal_gradient: float,
        # Phase 5 (STRONGEST)
        d_eff: float,
        spectral_entropy: float,
        participation_ratio: float,
        # Context
        is_near_boundary: bool = False,
    ) -> Dict[str, float]:
        """
        Score each of the 6 states using ALL geometric features.

        Uses SOFT thresholds (partial credit) instead of binary rules.
        This ensures no state wins by default and features contribute
        proportionally.

        Priority based on README correlations:
        - participation_ratio: r=-0.529 (STRONGEST for risk)
        - spectral_entropy: r=-0.496 (STRONGEST for risk)
        - thermal_gradient: r=+0.372 at boundary (1.65x amplification)

        Borderline amplification: signals boosted near boundaries.
        """
        p = self._ref_percentiles

        # Borderline amplification factor
        amp = 1.5 if is_near_boundary else 1.0

        # Risk boost: Non-coherent states get a baseline boost since their
        # conditions are harder to satisfy with well-behaved embeddings
        # This ensures interventions actually fire occasionally
        RISK_BOOST = 1.3  # 30% boost for risky states

        # =========================================================================
        # COHERENT: HIGH PR + LOW turbulence + STRONG flow
        # "Narrow passage" negation: wide manifold, stable directed flow
        # =========================================================================
        coherent_score = (
            # HIGH participation ratio = isotropic = safe (SOFT)
            self._soft_above(
                participation_ratio,
                p["participation_ratio_p30"],
                p["participation_ratio_p70"],
            )
            * 0.30
            +
            # LOW turbulence = coherent flow (SOFT)
            self._soft_below(turbulence, p["turbulence_p30"], p["turbulence_p70"])
            * 0.25
            +
            # STRONG gradient = directed motion (SOFT)
            self._soft_above(g_mag, p["g_mag_p30"], p["g_mag_p70"]) * 0.25
            +
            # HIGH spectral entropy = spread eigenvalues (SOFT)
            self._soft_above(
                spectral_entropy, p["spectral_entropy_p20"], p["spectral_entropy_p50"]
            )
            * 0.20
        ) * amp

        # =========================================================================
        # CONFIDENT: HIGH PR + TIGHT clustering + STRONG inward flow
        # =========================================================================
        confident_score = (
            # HIGH participation ratio (SOFT)
            self._soft_above(
                participation_ratio,
                p["participation_ratio_p30"],
                p["participation_ratio_p70"],
            )
            * 0.30
            +
            # TIGHT clustering (SOFT)
            self._soft_below(knn_std, p["knn_std_p30"], p["knn_std_p70"]) * 0.25
            +
            # STRONG gradient (SOFT)
            self._soft_above(g_mag, p["g_mag_median"], p["g_mag_p70"]) * 0.25
            +
            # INWARD pressure (binary but reasonable)
            (0.5 + 0.5 * np.tanh(delta_rho * 10)) * 0.20  # sigmoid-ish
        ) * amp

        # =========================================================================
        # CONSTRAINT_PRESSURE: Multi-basin tension
        # Detected by: HIGH delta_rho + MODERATE ridge + pressure signs
        # =========================================================================
        constraint_score = (
            (
                # HIGH pressure differential (SOFT)
                self._soft_above(delta_rho, 0.0, p["delta_rho_p70"]) * 0.35
                +
                # MODERATE ridge proximity (SOFT BETWEEN)
                self._soft_between(
                    ridge_prox,
                    p["ridge_prox_p40"],
                    (p["ridge_prox_p40"] + p["ridge_prox_p80"]) / 2,
                    p["ridge_prox_p80"],
                )
                * 0.30
                +
                # MODERATE turbulence (SOFT)
                self._soft_above(turbulence, p["turbulence_p30"], p["turbulence_p70"])
                * 0.20
                +
                # Borderline amplification (raw value, normalized)
                min(1.0, thermal_gradient / (p["thermal_gradient_p70"] + 1e-6)) * 0.15
            )
            * amp
            * RISK_BOOST
        )

        # =========================================================================
        # NOVEL_TERRITORY: Sparse, unfamiliar region
        # Detected by: LOW PR + HIGH knn_max + NO inward pressure + WEAK gradient
        # =========================================================================
        novel_score = (
            (
                # LOW participation ratio (narrow passage - STRONGEST signal, SOFT)
                self._soft_below(
                    participation_ratio,
                    p["participation_ratio_p30"],
                    p["participation_ratio_p70"],
                )
                * 0.35
                +
                # FAR from reference (SOFT)
                self._soft_above(knn_max, p["knn_max_p80"] * 0.7, p["knn_max_p80"])
                * 0.25
                +
                # NO inward pressure (lost) - sigmoid centered at 0
                (0.5 - 0.5 * np.tanh(delta_rho * 10)) * 0.20
                +
                # WEAK gradient (no direction, SOFT)
                self._soft_below(g_mag, p["g_mag_p30"], p["g_mag_median"]) * 0.20
            )
            * amp
            * RISK_BOOST
        )

        # =========================================================================
        # SEARCHING: Active exploration, not grounded
        # Detected by: HIGH turbulence + WEAK gradient + LOW PR
        # =========================================================================
        searching_score = (
            (
                # HIGH turbulence (directions changing, SOFT)
                self._soft_above(turbulence, p["turbulence_p30"], p["turbulence_p70"])
                * 0.35
                +
                # WEAK gradient (no clear direction, SOFT)
                self._soft_below(g_mag, p["g_mag_p30"], p["g_mag_p70"]) * 0.25
                +
                # MODERATE-LOW PR (somewhat constrained, SOFT)
                self._soft_below(
                    participation_ratio,
                    p["participation_ratio_p50"],
                    p["participation_ratio_p70"],
                )
                * 0.20
                +
                # LOW d_eff (searching in few dimensions, SOFT)
                self._soft_below(d_eff, p["d_eff_median"] * 0.7, p["d_eff_median"])
                * 0.20
            )
            * amp
            * RISK_BOOST
        )

        # =========================================================================
        # UNCERTAIN: Dispersed, boundary-adjacent (HIGH RISK)
        # Detected by: LOW PR + HIGH thermal_gradient + HIGH knn_std
        # =========================================================================
        uncertain_score = (
            (
                # LOW participation ratio (STRONGEST risk signal, SOFT)
                self._soft_below(
                    participation_ratio,
                    p["participation_ratio_p20"],
                    p["participation_ratio_p50"],
                )
                * 0.35
                +
                # HIGH thermal gradient (boundary-adjacent, SOFT)
                self._soft_above(
                    thermal_gradient,
                    p["thermal_gradient_p70"] * 0.5,
                    p["thermal_gradient_p70"],
                )
                * 0.25
                +
                # DISPERSED neighbors (SOFT)
                self._soft_above(knn_std, p["knn_std_p30"], p["knn_std_p70"]) * 0.20
                +
                # LOW spectral entropy (concentrated variance = fragile, SOFT)
                self._soft_below(
                    spectral_entropy,
                    p["spectral_entropy_p20"],
                    p["spectral_entropy_p50"],
                )
                * 0.20
            )
            * amp
            * RISK_BOOST
        )

        # Normalize to sum to 1
        raw_scores = {
            "coherent": coherent_score,
            "confident": confident_score,
            "constraint_pressure": constraint_score,
            "novel_territory": novel_score,
            "searching": searching_score,
            "uncertain": uncertain_score,
        }

        total = sum(raw_scores.values()) + 1e-12
        return {k: min(1.0, v / total) for k, v in raw_scores.items()}

    def trace_step(
        self,
        step_embedding: np.ndarray,
        step_index: int,
        n_steps: int = 4,
    ) -> EnhancedStepTrace:
        """
        Trace a single reasoning step with complete feature extraction.

        Args:
            step_embedding: Embedding vector (D,) or (1, D)
            step_index: Index of the step
            n_steps: Total steps in task (unused, kept for API compat)

        Returns:
            EnhancedStepTrace with state scores, flags, and trajectory info
        """
        if step_embedding.ndim == 1:
            step_embedding = step_embedding.reshape(1, -1)

        # Run full pipeline on this single embedding
        diag = compute_safety_diagnostics(
            step_embedding,
            self.reference_embeddings,
            k=self.k,
            k_phase1=self.k_phase1,
            bandwidth_multiplier=self.bandwidth_multiplier,
            include_full_phase1=True,
        )

        # =====================================================================
        # Extract ALL feature values
        # =====================================================================

        # Tier-0 (7 features)
        tier0 = diag.tier0_features[0]
        knn_mean = float(tier0[0])
        knn_std = float(tier0[1])
        knn_min = float(tier0[2])
        knn_max = float(tier0[3])
        curvature = float(tier0[4])
        ridge_prox = float(tier0[5])
        dist_to_ref = float(tier0[6])

        # Phase 1 (1-2 features)
        phase1 = (
            diag.phase1_features[0]
            if diag.phase1_features.ndim > 1
            else diag.phase1_features.flatten()
        )
        g_mag = float(phase1[0])
        delta_rho = float(phase1[1]) if len(phase1) > 1 else 0.0

        # Phase 2 + 5 (6 features)
        phase2_5 = diag.phase2_5_features[0]
        turbulence = float(phase2_5[0])
        thermal_gradient = float(phase2_5[1])
        vorticity = float(phase2_5[2])
        d_eff = float(phase2_5[3])
        spectral_entropy = float(phase2_5[4])
        participation_ratio = float(phase2_5[5])

        # =====================================================================
        # Compute derived metrics
        # =====================================================================

        # G-ratio (uniformity detection)
        g_ratio = self._compute_g_ratio(participation_ratio)

        # Boundary proximity
        is_near_boundary = self._is_near_boundary(ridge_prox, thermal_gradient)

        # Anomaly flags
        flags = self._detect_anomaly_flags(
            participation_ratio, spectral_entropy, g_ratio, thermal_gradient
        )

        # Trajectory phase (PR tracking)
        trajectory_phase = self._trajectory.update(participation_ratio)

        # =====================================================================
        # Classify state using ALL features
        # =====================================================================
        state_scores = self.classify_state_single(
            knn_std=knn_std,
            knn_max=knn_max,
            ridge_prox=ridge_prox,
            g_mag=g_mag,
            delta_rho=delta_rho,
            turbulence=turbulence,
            thermal_gradient=thermal_gradient,
            d_eff=d_eff,
            spectral_entropy=spectral_entropy,
            participation_ratio=participation_ratio,
            is_near_boundary=is_near_boundary,
        )

        # Dominant state and signal mapping
        dominant_state = max(state_scores, key=state_scores.get)
        signal = STATE_TO_SIGNAL[dominant_state]

        # Outlier detection: knn_min > p95 means even the nearest reference
        # point is very far — genuine terra incognita regardless of state scores
        if knn_min > self._ref_percentiles["knn_min_p95"]:
            signal = "terra_incognita"

        # =====================================================================
        # Build output structures
        # =====================================================================

        # Backward-compatible tier-0 features dict
        tier0_names = FEATURE_NAMES
        named_features = {name: float(tier0[i]) for i, name in enumerate(tier0_names)}

        # All-features dict (14 values)
        all_feat_values = np.concatenate(
            [
                tier0,
                phase1,
                phase2_5,
            ]
        )
        all_features = {
            name: float(all_feat_values[i])
            for i, name in enumerate(self._all_feature_names[: len(all_feat_values)])
        }

        # Topology features dict (Phase 5)
        topology_features = {
            "participation_ratio": participation_ratio,
            "spectral_entropy": spectral_entropy,
            "d_eff": d_eff,
        }

        return EnhancedStepTrace(
            step_index=step_index,
            features=named_features,
            novelty_signature=signal,
            raw_feature_vector=tier0.tolist(),
            state_scores=state_scores,
            dominant_state=dominant_state,
            all_features=all_features,
            topology_features=topology_features,
            g_ratio=g_ratio,
            flags=flags,
            trajectory_phase=trajectory_phase,
            is_near_boundary=is_near_boundary,
        )

    def get_config(self) -> Dict[str, Any]:
        """Get tracer configuration for logging."""
        return {
            "type": "EnhancedTracer_v2",
            "k": self.k,
            "k_phase1": self.k_phase1,
            "bandwidth_multiplier": self.bandwidth_multiplier,
            "n_calibration_samples": self._n_calibration_samples,
            "reference_shape": list(self.reference_embeddings.shape),
            "n_states": len(STATE_NAMES),
            "state_to_signal": STATE_TO_SIGNAL,
            "g_ratio_threshold": self.g_ratio_threshold,
            "pr_anomaly_percentile": self.pr_anomaly_percentile,
            "se_anomaly_percentile": self.se_anomaly_percentile,
            "pr_baseline_mean": self._pr_baseline_mean,
            "ref_percentiles": self._ref_percentiles,
        }
