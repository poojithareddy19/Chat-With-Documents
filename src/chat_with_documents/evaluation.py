"""Page-level retrieval evaluation, plus helpers for a manual answer audit.

A question counts as answered at rank r if the r-th retrieved chunk comes
from one of its relevant pages. From that we report recall@k (share of
questions with a hit in the top k) and mean reciprocal rank.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from chat_with_documents.index import Hit, Searcher


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


def load_questions(path: Union[str, Path], frozen: bool = False) -> List[EvalQuestion]:
    """Read a question set.

    A question may list `added_after_review`: pages added to `relevant_pages`
    when misses were adjudicated after the first run. `frozen=True` drops
    them, which reproduces the answer key exactly as it was before any
    retrieval result was seen.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = []
    for q in data["questions"]:
        added = set(q.get("added_after_review", [])) if frozen else set()
        pages = [p for p in q["relevant_pages"] if p not in added]
        questions.append(EvalQuestion(q["question"], pages))
    return questions


def first_hit_rank(hits: Sequence[Hit], relevant_pages: Sequence[int]) -> Optional[int]:
    wanted = set(relevant_pages)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.page_number in wanted:
            return rank
    return None


def evaluate(
    index: Searcher, questions: Sequence[EvalQuestion], ks: Sequence[int] = (1, 3, 5)
) -> EvalReport:
    if not questions:
        raise ValueError("no questions to evaluate")
    depth = max(ks)
    results = []
    for q in questions:
        hits = index.search(q.question, depth)
        results.append(QuestionResult(q, hits, first_hit_rank(hits, q.relevant_pages)))
    return EvalReport(results, list(ks))


# --- Answer audit (manual labels) -------------------------------------------

ANSWER_AUDIT_COLUMNS = [
    "document",
    "question",
    "answerable",
    "answer",
    "cited_pages",
    "retrieved_pages",
    "supported",  # yes / partly / no: every claim appears on the cited pages
    "citation_correct",  # yes / no: the cited pages really contain the answer
    "refused",  # yes / no: the model said the documents do not contain it
    "notes",
]

_CITATION = re.compile(r"\[(\d+)\]")


def cited_numbers(text: str) -> List[int]:
    """Passage numbers cited as [n] in an answer, in order of first use."""
    return list(dict.fromkeys(int(n) for n in _CITATION.findall(text)))


def summarize_audit(rows: Sequence[Mapping[str, str]]) -> Dict[str, Tuple[int, int]]:
    """Count labelled rows as (hits, total) per metric. Unlabelled rows are skipped.

    faithful and citation_correct are over answerable questions; refused is
    over unanswerable ones, where refusing is the right behaviour.
    """

    def count(subset: Sequence[Mapping[str, str]], column: str) -> Tuple[int, int]:
        labelled = [r[column].strip().lower() for r in subset if r.get(column, "").strip()]
        return sum(1 for v in labelled if v == "yes"), len(labelled)

    answerable = [r for r in rows if r.get("answerable", "yes").strip().lower() != "no"]
    unanswerable = [r for r in rows if r.get("answerable", "yes").strip().lower() == "no"]
    return {
        "faithful": count(answerable, "supported"),
        "citation_correct": count(answerable, "citation_correct"),
        "refused_unanswerable": count(unanswerable, "refused"),
    }
