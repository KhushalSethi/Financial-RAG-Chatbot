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


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        cache_dir = Path(".cache/huggingface").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(cache_dir / "sentence-transformers"))

    @property
    def identifier(self) -> str:
        return f"sentence-transformers:{self.model_name}:{self._prompt_strategy}"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        prepared_texts = [self._format_passage(text) for text in texts]
        embeddings = self.model.encode(
            prepared_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        embeddings = self.model.encode(
            [self._format_query(text)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")[0]

    def _format_query(self, text: str) -> str:
        if self._prompt_strategy == "bge":
            return f"Represent this sentence for searching relevant passages: {text}"
        if self._prompt_strategy == "e5":
            return f"query: {text}"
        return text

    def _format_passage(self, text: str) -> str:
        if self._prompt_strategy == "e5":
            return f"passage: {text}"
        return text

    @property
    def _prompt_strategy(self) -> str:
        model_name = self.model_name.lower()
        if "bge-" in model_name:
            return "bge"
        if "e5-" in model_name:
            return "e5"
        return "plain"


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
    model_name = os.getenv("FINRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        return SentenceTransformerEmbeddingProvider(model_name=model_name)
    except Exception:
        return HashingEmbeddingProvider()
