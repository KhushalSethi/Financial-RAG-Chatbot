from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.utils import Chunk, SourceDocument


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def split_document(
    document: SourceDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(document.text)
    return [
        Chunk(
            text=text.strip(),
            filename=document.filename,
            chunk_number=index,
            metadata={"page_count": document.page_count, **document.metadata},
        )
        for index, text in enumerate(texts, start=1)
        if text.strip()
    ]


def split_documents(
    documents: list[SourceDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(split_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks

