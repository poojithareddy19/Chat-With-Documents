"""Page-level retrieval evaluation.

A question counts as answered at rank r if the r-th retrieved chunk comes
from one of its relevant pages. From that we report recall@k (share of
questions with a hit in the top k) and mean reciprocal rank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from chat_with_documents.index import Hit, VectorIndex


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    relevant_pages: List[int]


@dataclass(frozen=True)
class QuestionResult:
    question: EvalQuestion
    hits: List[Hit]
    first_hit_rank: Optional[int]  # 1-based, None if no relevant page retrieved


@dataclass
class EvalReport:
    results: List[QuestionResult]
    ks: List[int]
    recall_at_k: Dict[int, float] = field(init=False)
    mrr: float = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.results)
        self.recall_at_k = {
            k: sum(1 for r in self.results if r.first_hit_rank and r.first_hit_rank <= k) / n
            for k in self.ks
        }
        self.mrr = sum(1 / r.first_hit_rank for r in self.results if r.first_hit_rank) / n

    def misses(self, k: int) -> List[QuestionResult]:
        return [r for r in self.results if not r.first_hit_rank or r.first_hit_rank > k]

    def to_markdown(self) -> str:
        header = "| Metric | Value |\n| --- | --- |\n"
        rows = "".join(f"| Recall@{k} | {v:.2f} |\n" for k, v in self.recall_at_k.items())
        return header + rows + f"| MRR | {self.mrr:.2f} |\n"


def load_questions(path: Union[str, Path]) -> List[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuestion(q["question"], list(q["relevant_pages"])) for q in data["questions"]]


def first_hit_rank(hits: Sequence[Hit], relevant_pages: Sequence[int]) -> Optional[int]:
    wanted = set(relevant_pages)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.page_number in wanted:
            return rank
    return None


def evaluate(
    index: VectorIndex, questions: Sequence[EvalQuestion], ks: Sequence[int] = (1, 3, 5)
) -> EvalReport:
    if not questions:
        raise ValueError("no questions to evaluate")
    depth = max(ks)
    results = []
    for q in questions:
        hits = index.search(q.question, depth)
        results.append(QuestionResult(q, hits, first_hit_rank(hits, q.relevant_pages)))
    return EvalReport(results, list(ks))
