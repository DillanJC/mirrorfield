"""Phase E Embedder Diagnostics — Representation Health Checks

This module provides diagnostic metrics to predict when geometry features will
add REAL_SIGNAL vs remain COSMETIC. The key insight from Phase E falsifier tests:
geometry only helps when the embedding space has stable geometric structure.

Goal: Find 1-2 metrics that predict REAL_SIGNAL vs COSMETIC verdicts.

Diagnostic Metrics:
1. Neighborhood Stability: Do kNN sets persist under small perturbations?
2. Local Intrinsic Dimensionality: Does the space behave low-rank locally?
3. Anisotropy/Hubness: Does kNN collapse to hubs?
4. Distance Concentration: Are distances nearly constant?
5. Curvature/Ridge Distribution Sanity: Do features have usable variance?

Usage:
    from geometry.embedder_diagnostics import EmbedderDiagnostics

    diagnostics = EmbedderDiagnostics(embeddings, nn_index)
    report = diagnostics.run_all()

    if report["representation_warning"]:
        print("WARNING: Embeddings may not support geometric features")
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiagnosticReport:
    """Embedder diagnostic report with health flags.

    Attributes:
        neighborhood_stability: Mean Jaccard similarity of kNN under perturbation [0,1]
                                Higher = more stable. Threshold: >0.6 for healthy.
        local_intrinsic_dim: Mean local intrinsic dimensionality (MLE estimator)
                             Lower = more low-rank structure. Threshold: <50 for D=768.
        hubness: Skewness of in-degree distribution (kNN reciprocity)
                 Near 0 = healthy, >2 = severe hubness problem.
        distance_concentration: Coefficient of variation (std/mean) of distances
                                Near 0 = distances collapsed. Threshold: >0.1 for healthy.
        curvature_variance: Variance of local curvature values
                           Near 0 = no signal. Threshold: >0.001 for usable.
        ridge_variance: Variance of ridge proximity values
                       Near 0 = no signal. Threshold: >0.001 for usable.
        representation_warning: True if any critical diagnostic fails
        verdict_prediction: "REAL_SIGNAL" or "COSMETIC" based on diagnostics
        confidence: Confidence in prediction [0,1]
    """
    neighborhood_stability: float
    local_intrinsic_dim: float
    hubness: float
    distance_concentration: float
    curvature_variance: float
    ridge_variance: float
    representation_warning: bool
    verdict_prediction: str
    confidence: float
    details: Dict[str, Any]


class EmbedderDiagnostics:
    """Diagnostic suite for embedder representation health.

    This class implements 5 diagnostic metrics to predict whether geometry
    features will add meaningful signal (REAL_SIGNAL) or remain cosmetic (COSMETIC).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        nn_index: Optional[Any] = None,  # sklearn.neighbors.NearestNeighbors
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize diagnostics.

        Args:
            embeddings: (N, D) float32 reference embeddings
            nn_index: Pre-fitted NearestNeighbors index (if None, will create one)
            config: Configuration dict with optional keys:
                - n_neighbors: k for kNN (default: 32)
                - perturbation_scale: epsilon for stability test (default: 0.01)
                - stability_samples: number of samples for stability (default: 200)
        """
        self.embeddings = embeddings
        self.n_samples, self.n_dims = embeddings.shape
        self.config = config or {}

        # Setup kNN index if not provided
        if nn_index is None:
            from sklearn.neighbors import NearestNeighbors
            n_neighbors = self.config.get("n_neighbors", 32)
            self.nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            self.nn_index.fit(embeddings)
        else:
            self.nn_index = nn_index

    def neighborhood_stability(
        self,
        epsilon: Optional[float] = None,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[float, Dict[str, Any]]:
        """Measure kNN persistence under small perturbations.

        Strategy:
        1. Sample random query points
        2. Add Gaussian noise: x' = x + ε * N(0, I)
        3. Compute Jaccard similarity: |neighbors(x) ∩ neighbors(x')| / k
        4. Return mean Jaccard score

        Interpretation:
        - High stability (>0.6): kNN sets are stable → geometry features can work
        - Low stability (<0.4): kNN sets are fragile → geometry will be noisy

        Args:
            epsilon: Perturbation scale (default: 0.01 * std of embeddings)
            n_samples: Number of samples to test (default: min(200, N/10))
            seed: Random seed

        Returns:
            (mean_jaccard, details_dict)
        """
        rng = np.random.RandomState(seed)

        # Set defaults
        if epsilon is None:
            epsilon = float(0.01 * np.std(self.embeddings))
        if n_samples is None:
            n_samples = min(200, self.n_samples // 10)

        # Sample query points
        sample_indices = rng.choice(self.n_samples, size=n_samples, replace=False)
        queries = self.embeddings[sample_indices]

        # Get original neighborhoods
        _, orig_neighbors = self.nn_index.kneighbors(queries)
        k = orig_neighbors.shape[1]

        # Perturb queries
        noise = rng.randn(n_samples, self.n_dims).astype(np.float32)
        perturbed = queries + epsilon * noise

        # Get perturbed neighborhoods
        _, pert_neighbors = self.nn_index.kneighbors(perturbed)

        # Compute Jaccard similarities
        jaccard_scores = []
        for i in range(n_samples):
            orig_set = set(orig_neighbors[i])
            pert_set = set(pert_neighbors[i])
            intersection = len(orig_set & pert_set)
            jaccard = intersection / k
            jaccard_scores.append(jaccard)

        mean_jaccard = float(np.mean(jaccard_scores))

        details = {
            "mean_jaccard": mean_jaccard,
            "std_jaccard": float(np.std(jaccard_scores)),
            "min_jaccard": float(np.min(jaccard_scores)),
            "max_jaccard": float(np.max(jaccard_scores)),
            "epsilon": epsilon,
            "n_samples": n_samples,
            "k_neighbors": k,
        }

        return mean_jaccard, details

    def local_intrinsic_dimensionality(
        self,
        n_samples: Optional[int] = None,
        k: int = 20,
        seed: int = 42,
    ) -> Tuple[float, Dict[str, Any]]:
        """Estimate local intrinsic dimensionality using MLE.

        Strategy:
        1. For each query point, get k nearest neighbors
        2. Estimate local ID using maximum likelihood estimator (Levina & Bickel 2004)
        3. Return mean local ID across samples

        MLE Formula:
            ID(x) = -1 / (1/k * Σ log(r_i / r_k))
        where r_i is distance to i-th nearest neighbor.

        Interpretation:
        - Low ID (<50 for D=768): Space has low-rank structure locally → good for geometry
        - High ID (>100): Space is high-dimensional → geometry may be noisy

        Args:
            n_samples: Number of samples (default: min(500, N/4))
            k: Number of neighbors for ID estimation (default: 20)
            seed: Random seed

        Returns:
            (mean_id, details_dict)
        """
        rng = np.random.RandomState(seed)

        if n_samples is None:
            n_samples = min(500, self.n_samples // 4)

        # Sample query points
        sample_indices = rng.choice(self.n_samples, size=n_samples, replace=False)
        queries = self.embeddings[sample_indices]

        # Get k+1 neighbors (including self)
        distances, _ = self.nn_index.kneighbors(queries, n_neighbors=k+1)

        # Remove self (first neighbor)
        distances = distances[:, 1:]  # (n_samples, k)

        # MLE estimator (Levina & Bickel 2004)
        local_ids = []
        for i in range(n_samples):
            r = distances[i]
            r_k = r[-1]  # distance to k-th neighbor

            # Avoid log(0) and division by zero
            if r_k < 1e-10:
                continue

            log_ratios = np.log(r[:-1] / r_k + 1e-10)
            mean_log_ratio = np.mean(log_ratios)

            if mean_log_ratio < -1e-6:  # negative mean
                local_id = -1.0 / mean_log_ratio
                # Clip to reasonable range
                local_id = min(local_id, self.n_dims)
                local_ids.append(local_id)

        if len(local_ids) == 0:
            # Degenerate case
            mean_id = float(self.n_dims)
        else:
            mean_id = float(np.mean(local_ids))

        details = {
            "mean_id": mean_id,
            "std_id": float(np.std(local_ids)) if len(local_ids) > 0 else 0.0,
            "median_id": float(np.median(local_ids)) if len(local_ids) > 0 else mean_id,
            "n_samples": len(local_ids),
            "k_neighbors": k,
            "ambient_dim": self.n_dims,
        }

        return mean_id, details

    def hubness(
        self,
        n_samples: Optional[int] = None,
        k: int = 32,
        seed: int = 42,
    ) -> Tuple[float, Dict[str, Any]]:
        """Measure anisotropy via hubness (kNN in-degree skewness).

        Strategy:
        1. Compute kNN for sample of points
        2. Count in-degree for each point (how many times it appears in others' kNN)
        3. Compute skewness of in-degree distribution

        Interpretation:
        - Near 0: Symmetric kNN (healthy)
        - >2: Severe hubness (some points dominate kNN lists) → geometry unstable

        Args:
            n_samples: Number of query samples (default: min(1000, N/2))
            k: Number of neighbors (default: 32)
            seed: Random seed

        Returns:
            (skewness, details_dict)
        """
        from scipy.stats import skew

        rng = np.random.RandomState(seed)

        if n_samples is None:
            n_samples = min(1000, self.n_samples // 2)

        # Sample query points
        sample_indices = rng.choice(self.n_samples, size=n_samples, replace=False)
        queries = self.embeddings[sample_indices]

        # Get kNN for queries
        _, neighbors = self.nn_index.kneighbors(queries, n_neighbors=k)

        # Count in-degrees
        in_degree = np.zeros(self.n_samples, dtype=np.int32)
        for i in range(n_samples):
            for j in neighbors[i]:
                in_degree[j] += 1

        # Compute skewness of in-degree distribution
        # Only consider points that were sampled or appeared in kNN
        active_degrees = in_degree[in_degree > 0]

        if len(active_degrees) < 3:
            # Not enough data
            skewness = 0.0
        else:
            skewness = float(skew(active_degrees))

        details = {
            "skewness": skewness,
            "mean_in_degree": float(np.mean(active_degrees)) if len(active_degrees) > 0 else 0.0,
            "std_in_degree": float(np.std(active_degrees)) if len(active_degrees) > 0 else 0.0,
            "max_in_degree": int(np.max(in_degree)),
            "n_hubs": int(np.sum(in_degree > 10 * np.mean(active_degrees))) if len(active_degrees) > 0 else 0,
            "n_samples": n_samples,
            "k_neighbors": k,
        }

        return skewness, details

    def distance_concentration(
        self,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[float, Dict[str, Any]]:
        """Measure distance concentration (curse of dimensionality indicator).

        Strategy:
        1. Sample random pairs of points
        2. Compute pairwise distances
        3. Return coefficient of variation: std(distances) / mean(distances)

        Interpretation:
        - Near 0: Distances collapsed (all points equidistant) → geometry useless
        - >0.1: Distances have variance → geometry can capture structure

        Args:
            n_samples: Number of pairs to sample (default: min(5000, N²/1000))
            seed: Random seed

        Returns:
            (cv, details_dict)
        """
        rng = np.random.RandomState(seed)

        if n_samples is None:
            n_samples = min(5000, self.n_samples * self.n_samples // 1000)

        # Sample random pairs
        i_indices = rng.choice(self.n_samples, size=n_samples, replace=True)
        j_indices = rng.choice(self.n_samples, size=n_samples, replace=True)

        # Compute distances
        distances = []
        for i_idx, j_idx in zip(i_indices, j_indices):
            if i_idx != j_idx:
                dist = np.linalg.norm(self.embeddings[i_idx] - self.embeddings[j_idx])
                distances.append(dist)

        distances = np.array(distances)

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))

        # Coefficient of variation
        cv = std_dist / (mean_dist + 1e-10)

        details = {
            "cv": cv,
            "mean_distance": mean_dist,
            "std_distance": std_dist,
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "n_pairs": len(distances),
        }

        return cv, details

    def feature_variance_sanity(
        self,
        curvature: Optional[np.ndarray] = None,
        ridge: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Check if geometry features have usable variance.

        If features are nearly constant, they can't explain anything.

        Args:
            curvature: (N,) curvature values (if None, will compute)
            ridge: (N,) ridge proximity values (if None, will compute)

        Returns:
            (variances_dict, details_dict)
        """
        # Compute features if not provided
        if curvature is None or ridge is None:
            from geometry.features import local_curvature_svd_gpu, ridge_proximity_from_knn_distances

            # Get kNN
            k = self.config.get("k_curvature", 16)
            _, indices = self.nn_index.kneighbors(self.embeddings, n_neighbors=32)

            if curvature is None:
                curv_idx = indices[:, :k]
                curvature = local_curvature_svd_gpu(
                    ref_points=self.embeddings,
                    neighbor_indices=curv_idx,
                    r=4,
                    batch_size=256,
                    device="cuda" if self.config.get("device") == "cuda" else "cpu",
                )

            if ridge is None:
                distances, _ = self.nn_index.kneighbors(self.embeddings, n_neighbors=32)
                ridge = ridge_proximity_from_knn_distances(distances)

        curv_var = float(np.var(curvature))
        ridge_var = float(np.var(ridge))

        variances = {
            "curvature_variance": curv_var,
            "ridge_variance": ridge_var,
        }

        details = {
            "curvature_mean": float(np.mean(curvature)),
            "curvature_std": float(np.std(curvature)),
            "curvature_range": float(np.max(curvature) - np.min(curvature)),
            "ridge_mean": float(np.mean(ridge)),
            "ridge_std": float(np.std(ridge)),
            "ridge_range": float(np.max(ridge) - np.min(ridge)),
        }

        return variances, details

    def run_all(
        self,
        curvature: Optional[np.ndarray] = None,
        ridge: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> DiagnosticReport:
        """Run all diagnostics and generate health report.

        Args:
            curvature: Pre-computed curvature (optional)
            ridge: Pre-computed ridge (optional)
            seed: Random seed

        Returns:
            DiagnosticReport with all metrics and prediction
        """
        print("Running embedder diagnostics...")

        # 1. Neighborhood stability
        print("  [1/5] Neighborhood stability...")
        stability, stability_details = self.neighborhood_stability(seed=seed)

        # 2. Local intrinsic dimensionality
        print("  [2/5] Local intrinsic dimensionality...")
        lid, lid_details = self.local_intrinsic_dimensionality(seed=seed)

        # 3. Hubness
        print("  [3/5] Anisotropy/hubness...")
        hub, hub_details = self.hubness(seed=seed)

        # 4. Distance concentration
        print("  [4/5] Distance concentration...")
        conc, conc_details = self.distance_concentration(seed=seed)

        # 5. Feature variance
        print("  [5/5] Feature variance sanity...")
        variances, var_details = self.feature_variance_sanity(curvature, ridge)

        # Compute warning flags
        warnings = []

        # Critical thresholds (empirically tuned from Test A/B/C)
        if stability < 0.5:
            warnings.append("low_neighborhood_stability")
        if lid > 100:
            warnings.append("high_intrinsic_dimensionality")
        if abs(hub) > 2.0:
            warnings.append("severe_hubness")
        if conc < 0.05:
            warnings.append("distance_concentration_collapsed")
        if variances["curvature_variance"] < 0.001:
            warnings.append("curvature_no_variance")
        if variances["ridge_variance"] < 0.001:
            warnings.append("ridge_no_variance")

        representation_warning = len(warnings) > 0

        # Predict verdict based on diagnostics
        # Key predictors (from empirical analysis):
        # - Neighborhood stability: most predictive (Test B: 0.6+ → REAL_SIGNAL, <0.4 → COSMETIC)
        # - Distance concentration: secondary (Test B: >0.1 → REAL_SIGNAL)

        confidence = 0.0
        if stability >= 0.6 and conc >= 0.1 and variances["curvature_variance"] >= 0.001:
            verdict_prediction = "REAL_SIGNAL"
            confidence = min(1.0, stability + conc)
        elif stability < 0.4 or conc < 0.05:
            verdict_prediction = "COSMETIC"
            confidence = min(1.0, (1.0 - stability) + (0.1 - conc) if conc < 0.1 else (1.0 - stability))
        else:
            verdict_prediction = "WEAK_SIGNAL"
            confidence = 0.5

        # Compile details
        all_details = {
            "stability": stability_details,
            "local_id": lid_details,
            "hubness": hub_details,
            "concentration": conc_details,
            "variance": var_details,
            "warnings": warnings,
        }

        report = DiagnosticReport(
            neighborhood_stability=stability,
            local_intrinsic_dim=lid,
            hubness=hub,
            distance_concentration=conc,
            curvature_variance=variances["curvature_variance"],
            ridge_variance=variances["ridge_variance"],
            representation_warning=representation_warning,
            verdict_prediction=verdict_prediction,
            confidence=confidence,
            details=all_details,
        )

        print(f"  → Verdict prediction: {verdict_prediction} (confidence: {confidence:.2f})")
        if representation_warning:
            print(f"  ⚠ Representation warnings: {', '.join(warnings)}")

        return report

    def print_report(self, report: DiagnosticReport):
        """Pretty-print diagnostic report.

        Args:
            report: DiagnosticReport from run_all()
        """
        print("\n" + "=" * 80)
        print("EMBEDDER DIAGNOSTIC REPORT")
        print("=" * 80)

        print(f"\n1. Neighborhood Stability: {report.neighborhood_stability:.3f}")
        print(f"   {'✓' if report.neighborhood_stability >= 0.5 else '✗'} "
              f"Threshold: >0.5 for healthy (>0.6 predicts REAL_SIGNAL)")

        print(f"\n2. Local Intrinsic Dimensionality: {report.local_intrinsic_dim:.1f}")
        print(f"   {'✓' if report.local_intrinsic_dim < 100 else '✗'} "
              f"Threshold: <100 for D={self.n_dims} (lower is better)")

        print(f"\n3. Hubness (skewness): {report.hubness:.3f}")
        print(f"   {'✓' if abs(report.hubness) < 2.0 else '✗'} "
              f"Threshold: |skew| < 2.0 (near 0 is ideal)")

        print(f"\n4. Distance Concentration (CV): {report.distance_concentration:.3f}")
        print(f"   {'✓' if report.distance_concentration > 0.05 else '✗'} "
              f"Threshold: >0.05 (>0.1 predicts REAL_SIGNAL)")

        print(f"\n5. Feature Variance Sanity:")
        print(f"   Curvature variance: {report.curvature_variance:.6f} "
              f"{'✓' if report.curvature_variance >= 0.001 else '✗'}")
        print(f"   Ridge variance: {report.ridge_variance:.6f} "
              f"{'✓' if report.ridge_variance >= 0.001 else '✗'}")

        print(f"\n{'─' * 80}")
        print(f"VERDICT PREDICTION: {report.verdict_prediction} (confidence: {report.confidence:.2f})")

        if report.representation_warning:
            print(f"\n⚠ REPRESENTATION WARNING:")
            for warning in report.details["warnings"]:
                print(f"   - {warning.replace('_', ' ')}")
            print(f"\n   Geometry features may be COSMETIC with this embedder.")
        else:
            print(f"\n✓ Representation health: PASS")
            print(f"   Geometry features likely to add meaningful signal.")

        print("=" * 80 + "\n")
