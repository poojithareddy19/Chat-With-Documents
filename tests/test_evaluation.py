import json

import pytest

from chat_with_documents.chunking import Chunk
from chat_with_documents.evaluation import (
    EvalQuestion,
    evaluate,
    first_hit_rank,
    load_questions,
)
from chat_with_documents.index import Hit, VectorIndex


def _hit(page):
    return Hit(Chunk("doc.pdf", page, f"text on page {page}"), 0.5)


def test_first_hit_rank_is_one_based_and_none_when_absent():
    hits = [_hit(4), _hit(9), _hit(2)]

    assert first_hit_rank(hits, [9]) == 2
    assert first_hit_rank(hits, [2, 4]) == 1
    assert first_hit_rank(hits, [7]) is None


def test_report_metrics(fake_embedder):
    index = VectorIndex.build(
        [
            Chunk("doc.pdf", 1, "cats purr and chase mice"),
            Chunk("doc.pdf", 2, "cars need petrol and tyres"),
            Chunk("doc.pdf", 3, "bread bakes in a hot oven"),
        ],
        fake_embedder,
    )
    questions = [
        EvalQuestion("do cats chase mice", [1]),  # rank 1
        EvalQuestion("petrol for cars", [7]),  # page not in the index: a miss at any k
        EvalQuestion("hot oven bread", [3, 2]),  # rank 1
    ]

    report = evaluate(index, questions, ks=(1, 3))

    assert report.recall_at_k == {1: pytest.approx(2 / 3), 3: pytest.approx(2 / 3)}
    assert report.mrr == pytest.approx(2 / 3)
    assert [r.question.question for r in report.misses(3)] == ["petrol for cars"]
    assert "| Recall@1 | 0.67 |" in report.to_markdown()
    assert "| MRR | 0.67 |" in report.to_markdown()


def test_evaluate_rejects_empty_question_set(fake_embedder):
    with pytest.raises(ValueError):
        evaluate(VectorIndex(fake_embedder), [])


def test_load_questions(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"questions": [{"question": "why?", "relevant_pages": [3, 4]}]}),
        encoding="utf-8",
    )

    assert load_questions(path) == [EvalQuestion("why?", [3, 4])]
