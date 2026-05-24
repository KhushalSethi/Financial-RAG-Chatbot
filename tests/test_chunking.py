from rag.chunking import split_document
from rag.utils import SourceDocument


def test_split_document_creates_overlapping_chunks():
    text = "Revenue increased because subscriptions grew. " * 80
    document = SourceDocument(filename="annual.pdf", text=text, page_count=1)

    chunks = split_document(document, chunk_size=180, chunk_overlap=40)

    assert len(chunks) > 1
    assert chunks[0].filename == "annual.pdf"
    assert chunks[0].chunk_number == 1
    assert all(chunk.text for chunk in chunks)


def test_split_document_rejects_invalid_overlap():
    document = SourceDocument(filename="x.pdf", text="hello world", page_count=1)

    try:
        split_document(document, chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

