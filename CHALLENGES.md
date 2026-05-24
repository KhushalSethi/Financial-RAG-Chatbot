# Challenges and How They Were Addressed

## 1. Scanned or Empty PDFs

Challenge: Some PDFs are scanned, image-only, encrypted, or malformed, so they do not contain extractable text.

Solution: PDF parsing lives in `rag/ingest.py`. Empty extraction raises `EmptyPDFError`, and the UI shows a clear message explaining that OCR is needed.

## 2. Weak Retrieval From Lightweight Embeddings

Challenge: Lightweight embeddings can retrieve generic chunks that contain similar words but do not answer the question well.

Solution: The default embedding model was upgraded to `BAAI/bge-base-en-v1.5`, with `FINRAG_EMBEDDING_MODEL` available for overrides. A deterministic fallback still keeps tests and degraded local runs working.

## 3. Vector Search Alone Is Not Enough

Challenge: Vector similarity can rank plausible but wrong chunks highly, especially in financial reports with repeated boilerplate.

Solution: Added optional cross-encoder reranking in `rag/reranker.py`. The app retrieves a larger candidate pool, reranks candidates, and only passes the final top-k chunks to the answer step. Reranking is opt-in because cross-encoders can improve some questions and degrade others on specialized documents.

## 4. Broad Questions Need Opening Context

Challenge: Questions like “Which company is this document about?” or “What is the document talking about?” often need title-page context, not only semantic matches.

Solution: The app detects document identity and overview questions and injects opening chunks from each indexed document before answering.

## 5. Repetition From Overlapping Chunks

Challenge: Overlapping chunks can repeat the same sentence or idea in the answer.

Solution: Local answer generation normalizes and deduplicates evidence sentences before producing the final response.

## 6. Stale Vector Indexes

Challenge: Reusing a vector index built with a different embedding model can make retrieval wrong.

Solution: Cached indexes store the embedding provider identifier, and cache paths include the embedding model name. Mismatched indexes are rebuilt.

## 7. Hard-To-Debug Bad Answers

Challenge: Without seeing chunks, it is hard to verify whether the answer is grounded in the uploaded document.

Solution: Added a searchable chunk inspector. A previous retrieval debug chart was removed because it made the UI noisier and did not help ordinary users read answers.

## 8. Local-First vs Answer Quality

Challenge: The app should run without API keys, but local extractive answers are less capable than LLM-generated answers.

Solution: Local mode remains the default and requires no API key. OpenAI mode is optional for higher-quality generation from retrieved context.

## 9. Tests Should Not Depend on Large Models

Challenge: Unit tests should not download embedding or reranking models.

Solution: Tests use deterministic fake or lightweight components: fake PDF readers, `HashingEmbeddingProvider`, and a fake reranker.
