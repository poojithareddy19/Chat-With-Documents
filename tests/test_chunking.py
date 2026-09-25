import pytest

from chat_with_documents.chunking import chunk_pages, split_text
from chat_with_documents.loader import Page


def test_short_text_is_a_single_chunk():
    assert split_text("just a few words", chunk_size=100, chunk_overlap=10) == ["just a few words"]


def test_chunks_respect_size_and_cover_the_whole_text():
    text = " ".join(f"word{i}" for i in range(400))

    chunks = split_text(text, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert text.startswith(chunks[0])
    assert text.endswith(chunks[-1])


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(300))

    chunks = split_text(text, chunk_size=120, chunk_overlap=40)

    for left, right in zip(chunks, chunks[1:]):
        tail_words = left.split()[-2:]
        assert any(word in right for word in tail_words)


def test_prefers_sentence_boundaries():
    text = ("This is sentence number one. " * 20).strip()

    chunks = split_text(text, chunk_size=150, chunk_overlap=0)

    assert all(c.endswith(".") for c in chunks[:-1])


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, chunk_overlap=10)


def test_chunks_never_cross_pages_and_carry_labels():
    pages = [
        Page("a.pdf", 1, "one " * 300),
        Page("a.pdf", 2, "two " * 50),
        Page("b.pdf", 7, "three"),
    ]

    chunks = chunk_pages(pages, chunk_size=400, chunk_overlap=50)

    labels = {c.label for c in chunks}
    assert labels == {"a.pdf, p.1", "a.pdf, p.2", "b.pdf, p.7"}
    assert all("two" not in c.text for c in chunks if c.page_number == 1)
    assert sum(1 for c in chunks if c.page_number == 1) >= 3
