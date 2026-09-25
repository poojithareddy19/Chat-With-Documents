import io

from conftest import make_pdf

from chat_with_documents.loader import load_pdf, load_pdfs


def test_pages_keep_source_and_one_based_page_numbers():
    data = make_pdf([["Alpha on page one."], ["Beta on page two."]])
    handle = io.BytesIO(data)
    handle.name = "C:/somewhere/report.pdf"

    pages = load_pdf(handle)

    assert [p.page_number for p in pages] == [1, 2]
    assert {p.source for p in pages} == {"report.pdf"}
    assert "Alpha" in pages[0].text
    assert "Beta" in pages[1].text


def test_pages_without_text_are_skipped_but_numbering_is_preserved():
    data = make_pdf([["Text here"], [], ["Text again"]])

    pages = load_pdf(io.BytesIO(data), source="scan.pdf")

    assert [p.page_number for p in pages] == [1, 3]


def test_load_from_path_uses_file_name(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(make_pdf([["Hello from a file"]]))

    pages = load_pdfs([str(path)])

    assert len(pages) == 1
    assert pages[0].source == "notes.pdf"


def test_multiline_page_text_is_joined_in_order():
    data = make_pdf([["first line", "second line", "third line"]])

    (page,) = load_pdf(io.BytesIO(data), source="x.pdf")

    assert page.text.index("first") < page.text.index("second") < page.text.index("third")
