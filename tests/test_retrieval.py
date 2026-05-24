from rag.embeddings import HashingEmbeddingProvider
from rag.retriever import VectorIndex
from rag.utils import Chunk


def test_vector_index_returns_relevant_chunk():
    chunks = [
        Chunk(text="Revenue increased by 12 percent due to enterprise subscriptions.", filename="a.pdf", chunk_number=1),
        Chunk(text="The office lease expires in 2030.", filename="a.pdf", chunk_number=2),
        Chunk(text="Cash flow from operations was strong.", filename="b.pdf", chunk_number=1),
    ]
    index = VectorIndex.from_chunks(chunks, HashingEmbeddingProvider(dimensions=128))

    results = index.search("How much did revenue increase?", k=2)

    assert results
    assert results[0].chunk.chunk_number == 1
    assert "Revenue increased" in results[0].chunk.text

