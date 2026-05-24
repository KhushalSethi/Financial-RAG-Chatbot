from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union

from pypdf import PdfReader

from rag.utils import EmptyPDFError, SourceDocument, clean_text


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
