"""
Geometry Bundle — Core API for Geometric Safety Features

Provides a clean interface to compute 7 geometric features from embeddings
for AI safety diagnostics, focusing on uncertainty near decision boundaries.
"""

import numpy as np
from typing import Dict, Optional, Any
from .features import compute_knn_features
from .engines import create_engine, NNEngine


class GeometryBundle:
    """
    Main API for computing geometric safety features on embedding spaces.

    This library provides geometric features for AI safety diagnostics. Rigorous evaluation
    identifies `knn_std_distance` (local neighborhood spread) as the most consistent signal
    for detecting high-uncertainty boundary regions.

    Parameters
    ----------
    reference_embeddings : np.ndarray
        Reference embeddings of shape (n_reference, n_dimensions).
        These form the "known" space against which queries are compared.
    k : int, default=50
        Number of nearest neighbors to consider for feature computation.

    Attributes
    ----------
    features_available : List[str]
        List of feature names that can be computed.

    Examples
    --------
    >>> import numpy as np
    >>> from mirrorfield.geometry import GeometryBundle
    >>>
    >>> # Create sample embeddings
    >>> reference = np.random.randn(1000, 256)
    >>> query = np.random.randn(100, 256)
    >>>
    >>> # Initialize bundle
    >>> bundle = GeometryBundle(reference, k=50)
    >>>
    >>> # Compute features
    >>> features = bundle.compute(query)
    >>> uncertainty_scores = features['knn_std_distance']
    >>> print(f"Computed {len(uncertainty_scores)} uncertainty scores")
    """

    def __init__(
        self, reference_embeddings: np.ndarray, k: int = 50, engine: str = "auto"
    ):
        """
        Initialize the GeometryBundle with reference embeddings.

        Parameters
        ----------
        reference_embeddings : np.ndarray
            Reference embeddings of shape (n_reference, n_dimensions).
        k : int, default=50
            Number of nearest neighbors to use for feature computation.
        engine : str, default="auto"
            NN backend: "sklearn", "faiss", or "auto" (chooses based on data size).
        """
        self.reference_embeddings = reference_embeddings
        self.k = k
        self.engine_name = engine
        self._engine = create_engine(reference_embeddings, engine=engine)
        self.feature_names = [
            "knn_mean_distance",
            "knn_std_distance",
            "knn_min_distance",
            "knn_max_distance",
            "local_curvature",
            "ridge_proximity",
            "dist_to_ref_nearest",
        ]

    def compute(self, query_embeddings) -> Dict[str, np.ndarray]:
        """
        Compute geometric safety features for query embeddings.

        Parameters
        ----------
        query_embeddings : np.ndarray or Dict[str, np.ndarray]
            Query embeddings of shape (n_queries, n_dimensions).
            Alternatively, a dict with 'query' key for query embeddings,
            and optionally 'reference' to override the reference embeddings.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping feature names to computed values.
            Each value is a 1D array of length n_queries.
        """
        if isinstance(query_embeddings, dict):
            query = query_embeddings["query"]
            reference = query_embeddings.get("reference", self.reference_embeddings)
        else:
            query = query_embeddings
            reference = self.reference_embeddings

        # Compute k-NN features using the selected engine
        knn_features, _ = compute_knn_features(
            query, reference, k=self.k, engine=self._engine
        )

        # Extract features into dict
        features = {}
        feature_names = self.feature_names
        for i, name in enumerate(feature_names):
            features[name] = knn_features[:, i]

        return features

    def get_feature_matrix(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract feature matrix from computed features dict.

        Useful for downstream ML models.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Features dict from compute().

        Returns
        -------
        np.ndarray
            (N, 7) array of geometric features.
        """
        N = len(next(iter(features.values())))
        feature_matrix = np.zeros((N, len(self.feature_names)), dtype=np.float32)

        for i, name in enumerate(self.feature_names):
            feature_matrix[:, i] = features[name]

        return feature_matrix

    def summarize(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Summarize computed features.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Features dict from compute().

        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        N = len(next(iter(features.values())))

        summary = {"n_samples": N, "feature_statistics": {}}

        for name in self.feature_names:
            values = features[name]
            summary["feature_statistics"][name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
            }

        return summary
