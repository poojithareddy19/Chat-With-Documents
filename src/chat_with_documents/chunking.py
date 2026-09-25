"""Split page text into overlapping chunks that never cross a page boundary.

Keeping chunks inside a single page means every chunk can be cited with an
exact file name and page number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    source: str
    page_number: int
    text: str

    @property
    def label(self) -> str:
        return f"{self.source}, p.{self.page_number}"


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedy character windows that prefer to end on a natural boundary."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            floor = chunk_size // 2  # never cut a chunk shorter than half size
            for sep in _SEPARATORS:
                cut = window.rfind(sep)
                if cut >= floor:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_pages(pages: Iterable, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, chunk_overlap):
            chunks.append(Chunk(source=page.source, page_number=page.page_number, text=piece))
    return chunks
