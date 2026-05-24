from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

from rag.embeddings import EmbeddingProvider
from rag.utils import Chunk, RetrievedChunk, ensure_directory


class VectorIndex:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunks: Optional[list[Chunk]] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.chunks = chunks or []
        self.vectors = vectors
        self._faiss_index = None
        if self.vectors is not None and len(self.chunks) > 0:
            self._build_faiss_index()

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], embedding_provider: EmbeddingProvider) -> "VectorIndex":
        if not chunks:
            return cls(embedding_provider=embedding_provider)
        vectors = embedding_provider.embed_texts([chunk.text for chunk in chunks])
        index = cls(embedding_provider=embedding_provider, chunks=chunks, vectors=vectors)
        return index

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if self.is_empty or self.vectors is None:
            return []
        query_vector = self.embedding_provider.embed_query(query).astype("float32")
        k = min(k, len(self.chunks))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(np.asarray([query_vector]), k)
            return [
                RetrievedChunk(chunk=self.chunks[int(index)], score=float(score))
                for score, index in zip(scores[0], indices[0])
                if int(index) >= 0
            ]

        scores = self.vectors @ query_vector
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self.chunks[int(index)], score=float(scores[int(index)]))
            for index in top_indices
        ]

    def save(self, directory: Union[str, Path]) -> None:
        directory = ensure_directory(directory)
        payload = {"chunks": self.chunks, "vectors": self.vectors}
        with (directory / "index.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, directory: Union[str, Path], embedding_provider: EmbeddingProvider) -> "VectorIndex":
        path = Path(directory) / "index.pkl"
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return cls(
            embedding_provider=embedding_provider,
            chunks=payload["chunks"],
            vectors=payload["vectors"],
        )

    def _build_faiss_index(self) -> None:
        try:
            import faiss
        except Exception:
            self._faiss_index = None
            return

        if self.vectors is None or self.vectors.size == 0:
            self._faiss_index = None
            return

        dimension = int(self.vectors.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(self.vectors.astype("float32"))
        self._faiss_index = index
