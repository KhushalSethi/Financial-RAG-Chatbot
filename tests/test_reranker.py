from rag.reranker import NoOpReranker, Reranker, rerank_retrieved_chunks
from rag.utils import Chunk, RetrievedChunk


class FakeReranker(Reranker):
    @property
    def identifier(self):
        return "fake"

    def score(self, query, passages):
        return [0.1, 0.9, 0.4]


def test_rerank_retrieved_chunks_orders_by_reranker_score():
    retrieved = [
        RetrievedChunk(Chunk("alpha", "doc.pdf", 1), score=0.8),
        RetrievedChunk(Chunk("beta", "doc.pdf", 2), score=0.2),
        RetrievedChunk(Chunk("gamma", "doc.pdf", 3), score=0.5),
    ]

    reranked = rerank_retrieved_chunks("question", retrieved, FakeReranker(), top_k=2)

    assert [item.chunk.chunk_number for item in reranked] == [2, 3]
    assert [item.score for item in reranked] == [0.9, 0.4]


def test_noop_reranker_keeps_vector_order():
    retrieved = [
        RetrievedChunk(Chunk("alpha", "doc.pdf", 1), score=0.8),
        RetrievedChunk(Chunk("beta", "doc.pdf", 2), score=0.2),
    ]

    reranked = rerank_retrieved_chunks("question", retrieved, NoOpReranker(), top_k=1)

    assert reranked == retrieved[:1]
