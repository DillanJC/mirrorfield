"""
Sentence-transformer embedder for real geometric structure.

Wraps the all-MiniLM-L6-v2 model (384-dim output) to produce:
- Step embeddings: embed each Claude response for geometric tracing
- Reference corpus: embed task prompts + evaluation criteria for meaningful geometry

384-dim output matches the project EMBEDDING_DIM standard.
"""

import numpy as np
from typing import List


class ResponseEmbedder:
    """
    Embeds text using sentence-transformers for geometric analysis.

    Lazy-loads the model on first use to avoid import overhead.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.

        Args:
            text: Input text to embed.

        Returns:
            (384,) float32 embedding vector (normalized).
        """
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            (N, 384) float32 embedding matrix (normalized).
        """
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings.astype(np.float32)

    def build_reference_corpus(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of reference texts for use as geometric reference.

        Normalizes embeddings to unit sphere for consistent k-NN behavior.

        Args:
            texts: List of reference texts (task prompts, criteria, etc.).

        Returns:
            (N_ref, 384) normalized float32 embeddings.
        """
        embeddings = self.embed_batch(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
