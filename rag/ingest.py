from __future__ import annotations

import re
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

from rag.utils import EmptyPDFError, SourceDocument, WebpageIngestionError, clean_text


MAX_WEBPAGE_BYTES = 5 * 1024 * 1024


class _ReadableHTMLParser(HTMLParser):
    _skip_tags = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "form",
        "button",
        "select",
        "option",
        "input",
        "textarea",
        "nav",
        "footer",
        "aside",
    }
    _block_tags = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
        "ol",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if self._skip_depth == 0 and tag in self._block_tags:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if self._skip_depth == 0 and tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.parts.append(text)
        self.parts.append(" ")

    @property
    def title(self) -> str:
        return clean_text(" ".join(self.title_parts))

    @property
    def text(self) -> str:
        return clean_text("".join(self.parts))


def extract_text_from_pdf_file(file_obj: BinaryIO, filename: Optional[str] = None) -> SourceDocument:
    """Extract text from a PDF file-like object.

    Raises EmptyPDFError when the PDF is scanned, image-only, encrypted, or
    otherwise contains no extractable text.
    """
    try:
        reader = PdfReader(file_obj)
    except Exception as exc:  # pypdf raises several parser-specific errors.
        raise EmptyPDFError(f"Could not read {filename or 'PDF'} as a valid PDF.") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise EmptyPDFError(
                f"{filename or 'PDF'} is encrypted and cannot be processed without a password."
            ) from exc

    page_text: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = clean_text(text)
        if text:
            page_text.append(f"[Page {page_number}]\n{text}")

    full_text = clean_text("\n\n".join(page_text))
    if not full_text:
        raise EmptyPDFError(
            f"{filename or 'This PDF'} has no extractable text. It may be scanned or image-only; OCR is needed before it can be indexed."
        )

    return SourceDocument(
        filename=filename or "uploaded.pdf",
        text=full_text,
        page_count=len(reader.pages),
    )


def extract_text_from_pdf_path(path: Union[str, Path]) -> SourceDocument:
    pdf_path = Path(path)
    with pdf_path.open("rb") as handle:
        return extract_text_from_pdf_file(handle, filename=pdf_path.name)


def extract_text_from_url(url: str, timeout: int = 15) -> SourceDocument:
    """Fetch a public URL and extract readable text from HTML or PDF content."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WebpageIngestionError("Enter a valid http:// or https:// URL.")

    request = Request(
        parsed.geturl(),
        headers={
            "User-Agent": "FinancialRAGChatbot/1.0 (+https://localhost)",
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(MAX_WEBPAGE_BYTES + 1)
    except HTTPError as exc:
        raise WebpageIngestionError(_http_error_message(exc)) from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise WebpageIngestionError(f"Could not fetch URL: {reason}.") from exc
    except TimeoutError as exc:
        raise WebpageIngestionError("Could not fetch URL: request timed out.") from exc

    if len(body) > MAX_WEBPAGE_BYTES:
        raise WebpageIngestionError("Webpage is too large to index. Try a smaller page or PDF.")

    if "application/pdf" in content_type.lower() or parsed.path.lower().endswith(".pdf"):
        document = extract_text_from_pdf_file(BytesIO(body), filename=_url_filename(parsed.geturl()))
        return SourceDocument(
            filename=document.filename,
            text=document.text,
            page_count=document.page_count,
            metadata={"source_type": "url", "url": parsed.geturl(), "content_type": "pdf"},
        )

    charset = _charset_from_content_type(content_type) or "utf-8"
    html = body.decode(charset, errors="replace")
    parser = _ReadableHTMLParser()
    parser.feed(html)
    text = parser.text
    if not text:
        raise WebpageIngestionError(
            "No readable text was found at that URL. It may require JavaScript, login access, or a different source."
        )

    title = parser.title
    filename = _webpage_display_name(parsed.geturl(), title)
    return SourceDocument(
        filename=filename,
        text=text,
        page_count=1,
        metadata={"source_type": "url", "url": parsed.geturl(), "title": title},
    )


def _charset_from_content_type(content_type: str) -> Optional[str]:
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip('"')


def _http_error_message(exc: HTTPError) -> str:
    if exc.code in {402, 403, 451}:
        return (
            f"Could not fetch URL: HTTP {exc.code}. "
            "The site appears to block automated fetching or require a browser challenge. "
            "Try a direct PDF link, a different public source, or a page that does not require bot protection."
        )
    return f"Could not fetch URL: HTTP {exc.code}."


def _webpage_display_name(url: str, title: str) -> str:
    parsed = urlparse(url)
    if title:
        cleaned_title = re.sub(r"\s+", " ", title).strip()
        if len(cleaned_title) > 80:
            cleaned_title = cleaned_title[:77].rstrip() + "..."
        return f"{cleaned_title} ({parsed.netloc})"
    return _url_filename(url)


def _url_filename(url: str) -> str:
    parsed = urlparse(url)
    path_name = Path(parsed.path).name
    if path_name:
        return f"{path_name} ({parsed.netloc})"
    return parsed.netloc
