# Financial RAG Chatbot

A local-first Streamlit app for asking questions over uploaded financial PDFs. The app extracts PDF text, chunks it, embeds it, stores it in a vector index, retrieves relevant chunks, and answers with source citations.

For the full project guide, architecture notes, run guide, testing workflow, troubleshooting, and extension ideas, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Features

- Upload one or more PDFs through Streamlit.
- Extract text with `pypdf`.
- Split documents into overlapping chunks with LangChain text splitters.
- Build a FAISS vector index with sentence-transformer embeddings.
- Use stronger configurable BGE embeddings by default.
- Optionally rerank retrieved candidates before answering.
- Inspect retrieval scores, top-k chunks, and indexed chunks in the UI.
- Ask semantic questions over uploaded documents.
- Answer from retrieved context only, with filename and chunk citations.
- View conversation history in the sidebar.
- Generate a document summary.
- Clear chat and reset the cached index.
- Helpful errors for empty, scanned, encrypted, or image-only PDFs.
- Optional OpenAI answer mode when `OPENAI_API_KEY` is configured.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default local embedding model is `BAAI/bge-base-en-v1.5`. It may download on first run. If that model cannot load, the app falls back to a deterministic local embedding provider so the UI can still run in degraded mode.

Optional model overrides:

```bash
export FINRAG_EMBEDDING_MODEL="intfloat/e5-large-v2"
export FINRAG_RERANKER_MODEL="BAAI/bge-reranker-base"
```

For OpenAI-generated answers:

```bash
export OPENAI_API_KEY="your-key"
```

OpenAI mode is optional. Local mode remains the default.

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, upload PDFs, click **Index PDFs**, and ask questions. Use the retrieval debug panel to compare vector results with reranking; reranking is optional because it can help some questions and hurt others.

## Tests

```bash
pytest
```

The tests cover PDF extraction behavior, chunk creation, and retrieval ranking using the deterministic test embedding provider.

## Project Structure

```text
app.py
rag/
  ingest.py
  chunking.py
  embeddings.py
  retriever.py
  reranker.py
  qa.py
  utils.py
tests/
requirements.txt
README.md
```

See [CHALLENGES.md](CHALLENGES.md) for the main implementation challenges and how they were handled.

## Notes

Scanned PDFs usually contain page images rather than embedded text. This app reports those files as requiring OCR before indexing.
