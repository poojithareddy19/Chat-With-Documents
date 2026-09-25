"""In-memory FAISS index over chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import faiss
import numpy as np

from chat_with_documents.chunking import Chunk
from chat_with_documents.embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: List[Chunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None

    @classmethod
    def build(cls, chunks: Iterable[Chunk], embedder: Embedder) -> "VectorIndex":
        index = cls(embedder)
        index.add(chunks)
        return index

    def add(self, chunks: Iterable[Chunk]) -> None:
        chunks = list(chunks)
        if not chunks:
            return
        vectors = self._embedder.embed([c.text for c in chunks])
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query: str, k: int) -> List[Hit]:
        if self._index is None or k <= 0:
            return []
        k = min(k, len(self._chunks))
        vector = self._embedder.embed([query])
        scores, ids = self._index.search(np.asarray(vector, dtype="float32"), k)
        return [
            Hit(chunk=self._chunks[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0
        ]

    @property
    def chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)
