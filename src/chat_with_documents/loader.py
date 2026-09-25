"""PDF text extraction that keeps file name and page number with every page."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO, Iterable, List, Optional, Union

from pypdf import PdfReader

PdfSource = Union[str, "os.PathLike[str]", IO[bytes]]


@dataclass(frozen=True)
class Page:
    source: str
    page_number: int  # 1-based, as shown in a PDF viewer
    text: str


def _display_name(file: PdfSource, source: Optional[str]) -> str:
    if source:
        return source
    name = getattr(file, "name", None)
    return os.path.basename(str(name if name else file))


def load_pdf(file: PdfSource, source: Optional[str] = None) -> List[Page]:
    """Return the non-empty pages of one PDF.

    `file` may be a path or a binary file object (for example a Streamlit
    upload). Pages without extractable text, such as scanned images, are
    skipped; callers can compare the result with the input to spot them.
    """
    if hasattr(file, "seek"):
        file.seek(0)  # uploads may already have been read once
    reader = PdfReader(file)
    name = _display_name(file, source)
    pages: List[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(source=name, page_number=number, text=text))
    return pages


def load_pdfs(files: Iterable[PdfSource]) -> List[Page]:
    pages: List[Page] = []
    for file in files:
        pages.extend(load_pdf(file))
    return pages
