import pytest
from urllib.error import HTTPError

from rag.ingest import extract_text_from_pdf_file, extract_text_from_url
from rag.utils import EmptyPDFError, WebpageIngestionError


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


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakeResponse:
    headers = FakeHeaders({"Content-Type": "text/html; charset=utf-8"})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _size):
        return b"""
        <html>
            <head><title>Investor Relations</title><script>hidden()</script></head>
            <body>
                <nav>Menu item</nav>
                <main>
                    <h1>Quarterly revenue update</h1>
                    <p>Revenue increased 12 percent as operating margin improved.</p>
                </main>
            </body>
        </html>
        """


def test_extract_text_from_url(monkeypatch):
    monkeypatch.setattr("rag.ingest.urlopen", lambda _request, timeout: FakeResponse())

    document = extract_text_from_url("https://example.com/investors")

    assert document.filename == "Investor Relations (example.com)"
    assert document.page_count == 1
    assert document.metadata["source_type"] == "url"
    assert document.metadata["url"] == "https://example.com/investors"
    assert "Revenue increased 12 percent" in document.text
    assert "hidden()" not in document.text
    assert "Menu item" not in document.text


def test_extract_text_from_url_rejects_invalid_url():
    with pytest.raises(WebpageIngestionError) as exc:
        extract_text_from_url("example.com/investors")

    assert "valid http:// or https:// URL" in str(exc.value)


def test_extract_text_from_url_explains_blocked_sites(monkeypatch):
    def blocked_urlopen(_request, timeout):
        raise HTTPError("https://example.com", 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr("rag.ingest.urlopen", blocked_urlopen)

    with pytest.raises(WebpageIngestionError) as exc:
        extract_text_from_url("https://example.com/investors")

    assert "block automated fetching" in str(exc.value)
    assert "browser challenge" in str(exc.value)
