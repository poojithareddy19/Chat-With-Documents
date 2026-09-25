from chat_with_documents.chunking import Chunk
from chat_with_documents.index import VectorIndex

CHUNKS = [
    Chunk("cats.pdf", 1, "Cats purr and chase mice around the house."),
    Chunk("cars.pdf", 3, "Cars need petrol, tyres and regular servicing."),
    Chunk("cooking.pdf", 2, "Bake the bread at 220 degrees for thirty minutes."),
]


def test_search_ranks_the_matching_chunk_first(fake_embedder):
    index = VectorIndex.build(CHUNKS, fake_embedder)

    hits = index.search("How often do cars need servicing?", k=2)

    assert hits[0].chunk.source == "cars.pdf"
    assert hits[0].score > hits[1].score
    assert -1.0 <= hits[-1].score <= 1.0


def test_k_is_capped_at_index_size(fake_embedder):
    index = VectorIndex.build(CHUNKS, fake_embedder)

    assert len(index.search("bread", k=50)) == len(CHUNKS)
    assert index.search("bread", k=0) == []


def test_empty_index_returns_nothing(fake_embedder):
    index = VectorIndex(fake_embedder)

    assert index.search("anything", k=3) == []
    assert len(index) == 0


def test_add_appends_to_existing_index(fake_embedder):
    index = VectorIndex.build(CHUNKS[:1], fake_embedder)
    index.add(CHUNKS[1:])

    assert len(index) == 3
    assert index.chunks == CHUNKS
