"""Phase E Falsifier — Verdict Engine for Geometry Signal Quality

The falsifier is the intellectual heart of Phase E. Its job is to answer one question
truthfully and reproducibly:

    **Does geometry explain variance beyond boundary_distance?**

This module implements a rigorous verdict system with 5 possible outcomes:
- REDUNDANT: Geometry is just re-naming distance
- COLLAPSED: Ridge collapsed into distance proxy
- COSMETIC: Adds <1% explanatory power
- REAL_SIGNAL: Adds meaningful independent signal
- WEAK_SIGNAL: Some signal; investigate

Design principle: The falsifier owns Phase E. If it can't reach a verdict,
Phase E is not done.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Phase E falsifier verdicts."""
    REDUNDANT = "REDUNDANT"      # corr(bd, geom_score) > 0.95
    COLLAPSED = "COLLAPSED"      # corr(ridge, bd) > 0.90
    COSMETIC = "COSMETIC"        # ΔR² < 0.01
    REAL_SIGNAL = "REAL_SIGNAL"  # ΔR² ≥ 0.01 and info_density > 0.10
    WEAK_SIGNAL = "WEAK_SIGNAL"  # else


@dataclass
class FalsifierResult:
    """Result of falsifier evaluation.

    Attributes:
        verdict: The falsifier's verdict
        delta_r2: Incremental R² (dist+geom vs dist-only)
        info_density: ΔR² / (1 - R²_dist_only)
        corr_bd_geom: Correlation between boundary_distance and geometry_score
        corr_ridge_bd: Correlation between ridge_proximity and boundary_distance
        r2_dist_only: R² for boundary_distance alone
        r2_dist_geom: R² for boundary_distance + geometry
        evidence: Dict of supporting evidence
        representation_warning: True if embedder diagnostics suggest COSMETIC risk
        embedder_diagnostics: Optional dict of embedder diagnostic metrics
    """
    verdict: Verdict
    delta_r2: float
    info_density: float
    corr_bd_geom: float
    corr_ridge_bd: float
    r2_dist_only: float
    r2_dist_geom: float
    evidence: Dict[str, Any]
    representation_warning: bool = False
    embedder_diagnostics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "verdict": self.verdict.value,
            "delta_r2": float(self.delta_r2),
            "info_density": float(self.info_density),
            "correlations": {
                "boundary_distance_vs_geometry_score": float(self.corr_bd_geom),
                "ridge_proximity_vs_boundary_distance": float(self.corr_ridge_bd),
            },
            "r_squared": {
                "distance_only": float(self.r2_dist_only),
                "distance_plus_geometry": float(self.r2_dist_geom),
            },
            "evidence": self.evidence,
            "representation_warning": self.representation_warning,
        }

        if self.embedder_diagnostics is not None:
            result["embedder_diagnostics"] = self.embedder_diagnostics

        return result


class PhaseEFalsifier:
    """Falsifier for Phase E geometry signal evaluation.

    The falsifier evaluates whether geometry features provide meaningful
    explanatory power beyond boundary distance alone.
    """

    @staticmethod
    def evaluate(
        boundary_distance: np.ndarray,
        geometry_score: np.ndarray,
        ridge_proximity: np.ndarray,
        target: np.ndarray,
        eps: float = 1e-8,
        embedder_diagnostics: Dict[str, Any] = None,
    ) -> FalsifierResult:
        """Evaluate geometry signal quality and return verdict.

        Args:
            boundary_distance: (N,) boundary distances from Phase D
            geometry_score: (N,) composite geometry score
            ridge_proximity: (N,) ridge proximity values
            target: (N,) target variable for R² computation (e.g., flip outcomes)
            eps: small constant for numerical stability
            embedder_diagnostics: Optional dict from EmbedderDiagnostics.run_all()
                                  Used to set representation_warning flag

        Returns:
            FalsifierResult with verdict and supporting evidence

        Verdict logic:
        1. Check REDUNDANT: corr(bd, geom_score) > 0.95
        2. Check COLLAPSED: corr(ridge, bd) > 0.90
        3. Compute ΔR² = R²(bd+geom) - R²(bd)
        4. If ΔR² < 0.01: COSMETIC
        5. If ΔR² ≥ 0.01 and info_density > 0.10: REAL_SIGNAL
        6. Else: WEAK_SIGNAL

        Representation Warning:
        If embedder_diagnostics provided and indicate unstable/collapsed embeddings:
        - Add warning flag (even if verdict is REAL_SIGNAL)
        - Warn: geometry gains may be embedder-specific artifacts
        """
        # Compute correlations
        corr_bd_geom = PhaseEFalsifier._correlation(boundary_distance, geometry_score)
        corr_ridge_bd = PhaseEFalsifier._correlation(ridge_proximity, boundary_distance)

        # Compute R² values
        r2_dist_only = PhaseEFalsifier._r_squared(
            target=target,
            predictors=boundary_distance.reshape(-1, 1),
        )
        r2_dist_geom = PhaseEFalsifier._r_squared(
            target=target,
            predictors=np.column_stack([boundary_distance, geometry_score]),
        )

        # Compute incremental metrics
        delta_r2 = r2_dist_geom - r2_dist_only
        info_density = delta_r2 / max(1.0 - r2_dist_only, eps)

        # Evidence collection
        evidence = {
            "n_samples": len(boundary_distance),
            "boundary_distance_stats": {
                "mean": float(np.mean(boundary_distance)),
                "std": float(np.std(boundary_distance)),
                "min": float(np.min(boundary_distance)),
                "max": float(np.max(boundary_distance)),
            },
            "geometry_score_stats": {
                "mean": float(np.mean(geometry_score)),
                "std": float(np.std(geometry_score)),
                "min": float(np.min(geometry_score)),
                "max": float(np.max(geometry_score)),
            },
            "ridge_stats": {
                "mean": float(np.mean(ridge_proximity)),
                "std": float(np.std(ridge_proximity)),
                "min": float(np.min(ridge_proximity)),
                "max": float(np.max(ridge_proximity)),
            },
        }

        # Verdict determination
        if corr_bd_geom > 0.95:
            verdict = Verdict.REDUNDANT
            evidence["reason"] = f"Geometry score is {corr_bd_geom:.3f} correlated with boundary distance"
        elif corr_ridge_bd > 0.90:
            verdict = Verdict.COLLAPSED
            evidence["reason"] = f"Ridge proximity is {corr_ridge_bd:.3f} correlated with boundary distance"
        elif delta_r2 < 0.01:
            verdict = Verdict.COSMETIC
            evidence["reason"] = f"Geometry adds only {delta_r2:.4f} R² (<1%)"
        elif delta_r2 >= 0.01 and info_density > 0.10:
            verdict = Verdict.REAL_SIGNAL
            evidence["reason"] = f"Geometry adds {delta_r2:.4f} R² with {info_density:.3f} info density"
        else:
            verdict = Verdict.WEAK_SIGNAL
            evidence["reason"] = f"Geometry adds {delta_r2:.4f} R² but info_density={info_density:.3f} is low"

        # Representation warning (gate 3): Check embedder diagnostics
        representation_warning = False
        embedder_diag_dict = None

        if embedder_diagnostics is not None:
            # Extract key metrics from diagnostics
            # Distance concentration and LID are the strongest predictors
            distance_conc = embedder_diagnostics.get("distance_concentration", 1.0)
            local_id = embedder_diagnostics.get("local_intrinsic_dim", 0.0)

            # Empirical thresholds from Test A/B/C analysis:
            # - Distance concentration < 0.05: collapsed (COSMETIC risk)
            # - Local ID > 150: too high-dimensional (COSMETIC risk)
            if distance_conc < 0.05 or local_id > 150:
                representation_warning = True
                evidence["representation_warning_reason"] = []
                if distance_conc < 0.05:
                    evidence["representation_warning_reason"].append(
                        f"Distance concentration collapsed (CV={distance_conc:.3f} < 0.05)"
                    )
                if local_id > 150:
                    evidence["representation_warning_reason"].append(
                        f"Local intrinsic dimensionality too high (LID={local_id:.1f} > 150)"
                    )
                evidence["representation_warning_reason"] = "; ".join(
                    evidence["representation_warning_reason"]
                )

            # Save diagnostics summary
            embedder_diag_dict = {
                "distance_concentration": float(distance_conc),
                "local_intrinsic_dim": float(local_id),
                "representation_warning": representation_warning,
            }

            # If verdict is REAL_SIGNAL but diagnostics warn, add note
            if representation_warning and verdict == Verdict.REAL_SIGNAL:
                evidence["warning_note"] = (
                    "REAL_SIGNAL verdict but representation diagnostics suggest risk of "
                    "embedder-specific artifact. Gains may not generalize to other embedders."
                )

        return FalsifierResult(
            verdict=verdict,
            delta_r2=delta_r2,
            info_density=info_density,
            corr_bd_geom=corr_bd_geom,
            corr_ridge_bd=corr_ridge_bd,
            r2_dist_only=r2_dist_only,
            r2_dist_geom=r2_dist_geom,
            evidence=evidence,
            representation_warning=representation_warning,
            embedder_diagnostics=embedder_diag_dict,
        )

    @staticmethod
    def _correlation(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
        """Compute Pearson correlation coefficient.

        Args:
            x: (N,) array
            y: (N,) array
            eps: small constant for numerical stability

        Returns:
            correlation: float in [-1, 1]
        """
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)) + eps

        return numerator / denominator

    @staticmethod
    def _r_squared(
        target: np.ndarray,
        predictors: np.ndarray,
        eps: float = 1e-8,
    ) -> float:
        """Compute R² via ordinary least squares.

        Args:
            target: (N,) target variable
            predictors: (N, k) predictor matrix
            eps: small constant for numerical stability

        Returns:
            r_squared: float in [0, 1]
        """
        # Add intercept column
        if predictors.ndim == 1:
            predictors = predictors.reshape(-1, 1)

        X = np.column_stack([np.ones(len(predictors)), predictors])

        # OLS: beta = (X^T X)^{-1} X^T y
        try:
            XtX = X.T @ X
            Xty = X.T @ target
            beta = np.linalg.solve(XtX + eps * np.eye(len(XtX)), Xty)
            predictions = X @ beta
        except np.linalg.LinAlgError:
            # Fallback to lstsq if matrix is singular
            beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
            predictions = X @ beta

        # R² = 1 - SSres / SStot
        ss_res = np.sum((target - predictions)**2)
        ss_tot = np.sum((target - np.mean(target))**2) + eps

        return 1.0 - (ss_res / ss_tot)


def evaluate_phase_e(
    boundary_distance: np.ndarray,
    curvature: np.ndarray,
    ridge_proximity: np.ndarray,
    target: np.ndarray,
    geometry_weights: Dict[str, float] = None,
) -> FalsifierResult:
    """Convenience function for Phase E evaluation.

    Args:
        boundary_distance: (N,) from Phase D
        curvature: (N,) local curvature
        ridge_proximity: (N,) ridge proximity
        target: (N,) target for R² (e.g., flip outcomes, true labels)
        geometry_weights: optional weights for geometry score

    Returns:
        FalsifierResult
    """
    from geometry.scoring import compute_geometry_score

    geometry_score = compute_geometry_score(
        curvature=curvature,
        ridge=ridge_proximity,
        weights=geometry_weights,
    )

    return PhaseEFalsifier.evaluate(
        boundary_distance=boundary_distance,
        geometry_score=geometry_score,
        ridge_proximity=ridge_proximity,
        target=target,
    )
