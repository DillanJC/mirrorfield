"""
Nearest Neighbor Engines for Scalable Geometric Feature Computation.

Provides sklearn and FAISS backends with automatic selection based on data size.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class NNEngine(ABC):
    """Abstract base class for nearest neighbor engines."""

    @abstractmethod
    def kneighbors(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each point in X."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging."""
        pass


class SklearnEngine(NNEngine):
    """Scikit-learn based nearest neighbor engine (default, reliable)."""

    def __init__(self, reference: np.ndarray, metric: str = "euclidean"):
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(
            n_neighbors=reference.shape[0],  # Max possible
            metric=metric,
            algorithm="auto",
        )
        self.nn.fit(reference)
        self._metric = metric

    def kneighbors(self, X, k):
        return self.nn.kneighbors(X, n_neighbors=k)

    @property
    def name(self):
        return f"sklearn_{self._metric}"


class FAISSEngine(NNEngine):
    """FAISS-based nearest neighbor engine (high-performance, optional)."""

    def __init__(self, reference: np.ndarray, metric: str = "euclidean"):
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: pip install faiss-cpu"
            )

        self.reference = reference.astype(np.float32)
        self.metric = metric
        self.index = self._build_index()

    def _build_index(self):
        import faiss

        dim = self.reference.shape[1]

        if self.metric == "euclidean":
            index = faiss.IndexFlatL2(dim)
        elif self.metric == "cosine":
            index = faiss.IndexFlatIP(dim)
            # Normalize for cosine
            faiss.normalize_L2(self.reference)
        else:
            raise ValueError(f"FAISS doesn't support metric: {self.metric}")

        index.add(self.reference)
        return index

    def kneighbors(self, X, k):
        X = X.astype(np.float32)

        if self.metric == "cosine":
            import faiss

            faiss.normalize_L2(X)

        distances, indices = self.index.search(X, k)

        if self.metric == "cosine":
            distances = 1 - distances  # Convert to distance

        return distances, indices

    @property
    def name(self):
        return f"faiss_{self.metric}"


def create_engine(
    reference: np.ndarray, engine: str = "auto", metric: str = "euclidean"
) -> NNEngine:
    """
    Factory function to create appropriate NN engine.

    Parameters
    ----------
    reference : np.ndarray
        Reference embeddings.
    engine : str
        Engine type: "sklearn", "faiss", or "auto".
    metric : str
        Distance metric.

    Returns
    -------
    NNEngine
        Initialized engine.
    """
    if engine == "sklearn":
        return SklearnEngine(reference, metric)

    if engine in ["faiss", "auto"]:
        try:
            return FAISSEngine(reference, metric)
        except ImportError:
            if engine == "auto":
                return SklearnEngine(reference, metric)
            raise

    raise ValueError(f"Unknown engine: {engine}")
