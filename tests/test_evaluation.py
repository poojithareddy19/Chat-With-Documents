import json

import pytest

from chat_with_documents.chunking import Chunk
from chat_with_documents.evaluation import (
    EvalQuestion,
    cited_numbers,
    evaluate,
    first_hit_rank,
    load_questions,
    summarize_audit,
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


def test_frozen_view_drops_pages_added_after_review(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"question": "a?", "relevant_pages": [3, 48], "added_after_review": [48]},
                    {"question": "b?", "relevant_pages": [5]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_questions(path)[0] == EvalQuestion("a?", [3, 48])
    assert load_questions(path, frozen=True) == [EvalQuestion("a?", [3]), EvalQuestion("b?", [5])]


def test_cited_numbers_in_order_of_first_use():
    assert cited_numbers("Two years [2]. Batteries excluded [1][2].") == [2, 1]
    assert cited_numbers("No citation here.") == []


def test_summarize_audit_counts_only_labelled_rows():
    rows = [
        {"answerable": "yes", "supported": "yes", "citation_correct": "yes", "refused": ""},
        {"answerable": "yes", "supported": "partly", "citation_correct": "no", "refused": ""},
        {"answerable": "yes", "supported": "", "citation_correct": "", "refused": ""},
        {"answerable": "no", "supported": "", "citation_correct": "", "refused": "yes"},
        {"answerable": "no", "supported": "", "citation_correct": "", "refused": "No"},
    ]

    assert summarize_audit(rows) == {
        "faithful": (1, 2),
        "citation_correct": (1, 2),
        "refused_unanswerable": (1, 2),
    }
