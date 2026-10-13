import pytest

from chat_with_documents.chunking import Chunk
from chat_with_documents.hybrid import BM25Index, Retriever, reciprocal_rank_fusion, tokenize
from chat_with_documents.index import VectorIndex


def test_tokenize_lowercases_and_keeps_identifiers_whole():
    assert tokenize("POSITION_QC flag 8, see §3.2!") == [
        "position_qc",
        "flag",
        "8",
        "see",
        "3",
        "2",
    ]
    assert tokenize("  ...  ") == []


def test_rrf_orders_by_summed_reciprocal_ranks():
    # c=1: id 0 -> 1/2 + 1/3, id 2 -> 1/4 + 1/2, id 1 -> 1/3, id 3 -> 1/4
    assert reciprocal_rank_fusion([[0, 1, 2], [2, 0, 3]], k=4, c=1) == [0, 2, 1, 3]
    assert reciprocal_rank_fusion([[0, 1, 2], [2, 0, 3]], k=2, c=1) == [0, 2]


def test_rrf_rewards_agreement_over_a_single_top_rank():
    # 7 is second in both lists; 5 and 6 each top only one list.
    assert reciprocal_rank_fusion([[5, 7], [6, 7]], k=1) == [7]


def test_rrf_handles_empty_rankings():
    assert reciprocal_rank_fusion([[], []], k=3) == []
    assert reciprocal_rank_fusion([[4], []], k=3) == [4]


def _chunks():
    return [
        Chunk("doc.pdf", 1, "The position of the float is measured by GPS satellites."),
        Chunk("doc.pdf", 2, "POSITION_QC flag 8 means the position was interpolated."),
        Chunk("doc.pdf", 3, "Salinity and temperature are measured on every profile."),
    ]


def test_bm25_finds_the_exact_identifier_and_skips_non_matching_chunks():
    index = BM25Index(_chunks())

    assert index.search("POSITION_QC", k=3) == [1]
    assert index.search("nothing in common", k=3) == []
    assert BM25Index([]).search("anything", k=3) == []


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_retriever_returns_hits_with_cosine_scores(fake_embedder, mode):
    index = VectorIndex.build(_chunks(), fake_embedder)

    hits = Retriever(index, mode).search("POSITION_QC flag", k=2)

    assert hits[0].chunk.page_number == 2
    assert len(hits) <= 2
    dense_scores = {h.chunk.page_number: h.score for h in index.search("POSITION_QC flag", 3)}
    for hit in hits:
        assert hit.score == pytest.approx(dense_scores[hit.chunk.page_number])


def test_hybrid_keeps_dense_only_matches(fake_embedder):
    index = VectorIndex.build(_chunks(), fake_embedder)

    pages = [h.chunk.page_number for h in Retriever(index, "hybrid").search("salinity", k=3)]

    assert pages[0] == 3
    assert sorted(pages) == [1, 2, 3]  # BM25 matches only page 3; dense fills the rest


def test_retriever_rejects_unknown_mode(fake_embedder):
    with pytest.raises(ValueError):
        Retriever(VectorIndex(fake_embedder), "sparse")


def test_retriever_on_empty_index(fake_embedder):
    assert Retriever(VectorIndex(fake_embedder), "hybrid").search("anything", 3) == []
