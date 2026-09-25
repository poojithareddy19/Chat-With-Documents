"""Embedding backends.

Vectors are float32 and L2-normalised, so an inner-product search over them
is a cosine-similarity search.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array with unit-length rows."""
        ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model.

    The import is deferred so that the rest of the package, and the test
    suite, do not need torch installed.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")
