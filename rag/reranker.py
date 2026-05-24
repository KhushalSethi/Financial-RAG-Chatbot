from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from rag.utils import RetrievedChunk


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(ABC):
    @property
    @abstractmethod
    def identifier(self) -> str:
        """Stable name for UI/debug output."""

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores for query/passage pairs."""


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        cache_dir = Path(".cache/huggingface").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))

        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=512)

    @property
    def identifier(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs)
        return [float(score) for score in scores]


class NoOpReranker(Reranker):
    @property
    def identifier(self) -> str:
        return "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.0 for _ in passages]


def get_default_reranker() -> Reranker:
    model_name = os.getenv("FINRAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    try:
        return CrossEncoderReranker(model_name=model_name)
    except Exception:
        return NoOpReranker()


def rerank_retrieved_chunks(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    reranker: Reranker,
    top_k: int,
) -> list[RetrievedChunk]:
    if not retrieved_chunks or top_k <= 0:
        return []
    if reranker.identifier == "none":
        return retrieved_chunks[:top_k]

    scores = reranker.score(query, [item.chunk.text for item in retrieved_chunks])
    rescored = [
        RetrievedChunk(chunk=item.chunk, score=float(score))
        for item, score in zip(retrieved_chunks, scores)
    ]
    rescored.sort(key=lambda item: item.score, reverse=True)
    return rescored[:top_k]
