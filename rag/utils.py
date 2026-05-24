from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


class RAGError(Exception):
    """Base exception for user-facing RAG errors."""


class EmptyPDFError(RAGError):
    """Raised when a PDF has no extractable text."""


class MissingAPIKeyError(RAGError):
    """Raised when an LLM provider needs an API key that is not configured."""


@dataclass(frozen=True)
class SourceDocument:
    filename: str
    text: str
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    filename: str
    chunk_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.filename} - chunk {self.chunk_number}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def clean_text(text: str) -> str:
    """Normalize PDF text while preserving paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def short_hash(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:16]


def ensure_directory(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
