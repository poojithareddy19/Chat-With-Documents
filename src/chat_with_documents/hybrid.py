"""Keyword (BM25) retrieval and its fusion with dense retrieval.

Dense embeddings match meaning but blur exact tokens such as POSITION_QC or
section numbers; BM25 matches exact tokens but misses paraphrases. Reciprocal
rank fusion combines the two using ranks only, so the very different score
scales (cosine in [-1, 1], BM25 unbounded) never have to be normalised.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

from rank_bm25 import BM25Okapi

from chat_with_documents.chunking import Chunk
from chat_with_documents.config import RETRIEVAL_MODES
from chat_with_documents.index import Hit, VectorIndex

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens. Underscores are kept, so POSITION_QC stays one token."""
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int) -> List[int]:
        """Indices of the top-k chunks. Chunks sharing no term with the query are left out."""
        if self._bm25 is None or k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in ranked if scores[i] > 0][:k]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int, c: int = 60) -> List[int]:
    """Standard RRF: score = sum over rankers of 1 / (c + rank), rank 1-based.

    Ties keep first-seen order, so the earlier ranking wins a tie.
    """
    score: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            score[idx] = score.get(idx, 0.0) + 1.0 / (c + rank)
    return sorted(score, key=lambda i: -score[i])[:k]


class Retriever:
    """Search a document set in one of three modes: dense, bm25 or hybrid.

    Every returned Hit carries the chunk's cosine similarity to the query,
    whatever the mode, so the UI can keep showing a comparable score.
    """

    def __init__(self, index: VectorIndex, mode: str = "dense", candidates: int = 20):
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {', '.join(RETRIEVAL_MODES)}, got {mode!r}")
        self.index = index
        self.mode = mode
        self.candidates = candidates
        self._chunks = index.chunks
        self._bm25 = BM25Index(self._chunks) if mode != "dense" else None

    def search(self, query: str, k: int) -> List[Hit]:
        if self.mode == "dense" or k <= 0:
            return self.index.search(query, k)
        # A flat index scores every chunk anyway; asking for all of them gives
        # the dense ranking and a cosine score for any chunk BM25 brings in.
        dense = self.index.search(query, len(self._chunks))
        position = {id(c): i for i, c in enumerate(self._chunks)}
        cosine = {position[id(h.chunk)]: h.score for h in dense}
        bm25 = self._bm25.search(query, max(k, self.candidates))
        if self.mode == "bm25":
            chosen = bm25[:k]
        else:
            dense_ids = [position[id(h.chunk)] for h in dense[: max(k, self.candidates)]]
            chosen = reciprocal_rank_fusion([dense_ids, bm25], k)
        return [Hit(chunk=self._chunks[i], score=cosine.get(i, 0.0)) for i in chosen]

    def __len__(self) -> int:
        return len(self._chunks)
