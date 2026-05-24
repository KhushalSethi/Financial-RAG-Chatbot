from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable name used to validate cached vector indexes."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return a 2D float32 embedding matrix."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        cache_dir = Path(".cache/huggingface").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir / "sentence-transformers"))

    @property
    def identifier(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic fallback used for tests and offline degraded mode."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @property
    def identifier(self) -> str:
        return f"hashing:{self.dimensions}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row, text in enumerate(texts):
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vectors[row, index] += 1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


def _tokens(text: str) -> Iterable[str]:
    token = []
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            yield "".join(token)
            token = []
    if token:
        yield "".join(token)


def get_default_embedding_provider() -> EmbeddingProvider:
    try:
        return SentenceTransformerEmbeddingProvider()
    except Exception:
        return HashingEmbeddingProvider()
