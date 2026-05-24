import pytest

from rag.ingest import extract_text_from_pdf_file
from rag.utils import EmptyPDFError


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    is_encrypted = False

    def __init__(self, _file_obj):
        self.pages = [FakePage("Revenue was $10 million."), FakePage("Net income improved.")]


class EmptyReader:
    is_encrypted = False

    def __init__(self, _file_obj):
        self.pages = [FakePage("")]


def test_extract_text_from_pdf_file(monkeypatch):
    monkeypatch.setattr("rag.ingest.PdfReader", FakeReader)

    document = extract_text_from_pdf_file(object(), "report.pdf")

    assert document.filename == "report.pdf"
    assert document.page_count == 2
    assert "Revenue was $10 million" in document.text
    assert "[Page 2]" in document.text


def test_extract_text_from_empty_pdf(monkeypatch):
    monkeypatch.setattr("rag.ingest.PdfReader", EmptyReader)

    with pytest.raises(EmptyPDFError) as exc:
        extract_text_from_pdf_file(object(), "scan.pdf")

    assert "OCR is needed" in str(exc.value)

