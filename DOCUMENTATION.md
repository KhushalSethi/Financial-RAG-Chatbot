# Financial RAG Chatbot Documentation

This document explains the Financial RAG Chatbot project end to end: what it does, how the code is organized, how the retrieval pipeline works, how to run and test it, and how to extend or troubleshoot it.

## 1. Project Overview

The Financial RAG Chatbot is a local-first document Q&A application. A user uploads one or more PDF files, the app extracts text from those PDFs, splits the text into overlapping chunks, embeds those chunks, stores them in a vector index, retrieves the most relevant chunks for a user question, and answers using only that retrieved context.

The main goal is grounded question answering. The chatbot should not answer from general knowledge. If the uploaded documents do not contain enough information, the answer should say that clearly.

The app is designed for financial documents such as:

- Annual reports
- Quarterly reports
- Investor presentations exported as PDFs
- Financial statements
- Audit reports
- Earnings transcripts
- Internal finance memos

It can work with any text-based PDF, but scanned or image-only PDFs require OCR before indexing.

## 2. Feature Summary

The project includes:

- Streamlit web UI
- PDF upload support
- PDF text extraction with `pypdf`
- Empty, scanned, and encrypted PDF error handling
- Text chunking with overlap using LangChain text splitters
- Local sentence-transformer embeddings by default
- Deterministic fallback embeddings for degraded offline/test mode
- FAISS vector search when `faiss-cpu` is installed
- Numpy similarity search fallback when FAISS is unavailable
- Cached vector indexes under `.cache/vector_index`
- Semantic search over uploaded PDFs
- Context-only answer generation
- Citations with source filename and chunk number
- Conversation history panel
- Document summary feature
- Reset option for chat and index state
- Unit tests for ingestion, chunking, and retrieval

## 3. Tech Stack

| Area | Library |
| --- | --- |
| UI | Streamlit |
| PDF extraction | pypdf |
| Chunking | langchain-text-splitters |
| Embeddings | sentence-transformers |
| Vector search | FAISS, with numpy fallback |
| Optional LLM answers | OpenAI Python SDK |
| Tables/UI metadata | pandas |
| Tests | pytest |

The application targets Python 3.10 or newer.

## 4. File Tree

```text
.
├── app.py
├── DOCUMENTATION.md
├── README.md
├── requirements.txt
├── pytest.ini
├── rag/
│   ├── __init__.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── qa.py
│   ├── retriever.py
│   └── utils.py
└── tests/
    ├── test_chunking.py
    ├── test_ingest.py
    └── test_retrieval.py
```

## 5. Module Responsibilities

### `app.py`

The Streamlit entry point. It handles:

- Page layout and sidebar controls
- PDF upload UI
- Indexing button
- Reset button
- Conversation display
- Chat input
- Document summary button
- Display of indexed source chunks
- Streamlit resource caching

It delegates RAG-specific work to the `rag/` package.

### `rag/ingest.py`

Responsible for reading PDFs and extracting text.

Important functions:

- `extract_text_from_pdf_file(file_obj, filename=None)`
- `extract_text_from_pdf_path(path)`

It raises `EmptyPDFError` when a PDF cannot provide useful text. This includes scanned, image-only, encrypted, corrupted, or otherwise unreadable PDFs.

### `rag/chunking.py`

Responsible for splitting extracted document text into overlapping chunks.

Important functions:

- `split_document(document, chunk_size=900, chunk_overlap=150)`
- `split_documents(documents, chunk_size=900, chunk_overlap=150)`

Each chunk records:

- Chunk text
- Source filename
- Chunk number
- Metadata such as page count

### `rag/embeddings.py`

Responsible for embedding text.

Important classes:

- `EmbeddingProvider`
- `SentenceTransformerEmbeddingProvider`
- `HashingEmbeddingProvider`

`SentenceTransformerEmbeddingProvider` uses `sentence-transformers/all-MiniLM-L6-v2` by default.

`HashingEmbeddingProvider` is a deterministic fallback. It is useful for tests and degraded operation when the sentence-transformer model cannot load.

### `rag/retriever.py`

Responsible for vector indexing and semantic search.

Important class:

- `VectorIndex`

`VectorIndex` can:

- Build an index from chunks
- Search for relevant chunks
- Save an index to disk
- Load an index from disk

It uses FAISS when available. If FAISS is not importable, it falls back to numpy dot-product similarity.

### `rag/qa.py`

Responsible for answer generation and summarization.

Important functions:

- `answer_question(question, retrieved_chunks, mode="local")`
- `summarize_chunks(retrieved_chunks, mode="local")`

There are two answer modes:

- `local`: extracts relevant sentences from retrieved chunks
- `openai`: uses an OpenAI chat model, but only with retrieved context

If `openai` mode is selected without `OPENAI_API_KEY`, the app raises a clear user-facing error.

### `rag/utils.py`

Shared utilities and dataclasses.

Important dataclasses:

- `SourceDocument`
- `Chunk`
- `RetrievedChunk`

Important exceptions:

- `RAGError`
- `EmptyPDFError`
- `MissingAPIKeyError`

## 6. RAG Pipeline

The app follows this retrieval-augmented generation flow:

1. The user uploads one or more PDFs in Streamlit.
2. `rag.ingest` extracts text from each PDF.
3. Empty or scanned PDFs are rejected with a helpful OCR message.
4. `rag.chunking` splits extracted text into overlapping chunks.
5. `rag.embeddings` embeds each chunk.
6. `rag.retriever` stores embeddings and chunk metadata in a vector index.
7. The user asks a question.
8. The question is embedded with the same embedding provider.
9. The vector index retrieves the top matching chunks.
10. `rag.qa` answers using only those retrieved chunks.
11. The answer includes citations like `annual_report.pdf - chunk 4`.

## 7. Why Chunking Matters

PDFs are often too long to pass directly into an LLM or local answer function. Chunking solves this by breaking documents into smaller searchable units.

The default settings are:

```python
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
```

The overlap helps preserve context across chunk boundaries. For example, if one paragraph starts at the end of one chunk and continues into the next, overlap reduces the chance that retrieval loses important context.

## 8. Citation Behavior

Every `Chunk` has a citation string:

```text
filename.pdf - chunk 3
```

When the app answers a question, it appends all retrieved source citations to the answer:

```text
Sources: annual_report.pdf - chunk 2; annual_report.pdf - chunk 5
```

This lets the user inspect the source chunks in the UI and verify where the answer came from.

## 9. Answer Modes

### Local Mode

Local mode does not call an external LLM. It uses the retrieved chunks and selects high-overlap sentences based on the user question.

Benefits:

- No API key required
- Fully local after dependencies/model are installed
- Fast and predictable
- Good for basic extraction-style questions

Limitations:

- Less fluent than an LLM
- May not synthesize complex multi-step answers well
- Best suited for questions where the answer is stated directly in the documents

### OpenAI Mode

OpenAI mode uses the OpenAI Python SDK to produce a more natural answer. It still receives only retrieved context, and the system prompt instructs it not to use outside knowledge.

To use it:

```bash
export OPENAI_API_KEY="your-key"
```

Then select `openai` in the Streamlit sidebar.

If the key is missing, the app shows:

```text
OPENAI_API_KEY is not set. Use local mode or configure the key before using OpenAI answers.
```

## 10. Index Caching

The vector index is cached in two ways:

1. Streamlit resource caching through `@st.cache_resource`
2. Disk persistence under `.cache/vector_index`

The index key is built from the uploaded document filenames and text content. If the same documents are indexed again, the app can reuse the saved vector index instead of rebuilding it unnecessarily.

The reset button clears:

- Uploaded document state
- Chunk state
- Conversation history
- Cached index key
- `.cache/vector_index`
- Streamlit resource cache

## 11. Setup Guide

### 11.1 Prerequisites

Install Python 3.10 or newer.

Check your version:

```bash
python3 --version
```

Expected:

```text
Python 3.10.x
```

or newer.

### 11.2 Create a Virtual Environment

From the project root:

```bash
python3.10 -m venv .venv
```

If your Python 3.10 executable is named differently, use the correct command for your machine:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 11.3 Install Dependencies

```bash
pip install -r requirements.txt
```

The first run may download the default sentence-transformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

That download can take a little time depending on network speed.

## 12. Run Guide

Start the app:

```bash
streamlit run app.py
```

Streamlit will print a local URL similar to:

```text
Local URL: http://localhost:8501
```

Open that URL in your browser.

### Basic User Flow

1. Open the app.
2. Upload one or more PDFs in the sidebar.
3. Click `Index PDFs`.
4. Wait for the success message showing document and chunk counts.
5. Ask a question in the chat input.
6. Read the answer and citations.
7. Expand `Retrieved context` to inspect source chunks.
8. Use `Generate summary` to summarize indexed content.
9. Use `Reset all` to clear chat and index state.

## 13. Testing Guide

Run tests:

```bash
pytest
```

Expected output:

```text
5 passed
```

The test suite covers:

- PDF extraction behavior
- Empty/scanned PDF handling
- Chunk creation
- Invalid chunk overlap validation
- Retrieval ranking with deterministic embeddings

The tests intentionally use lightweight fake PDF reader objects and deterministic embeddings so they are stable and fast.

## 14. Manual QA Checklist

Use this checklist after making changes:

- App starts with `streamlit run app.py`
- Uploading no files and clicking `Index PDFs` shows a warning
- Uploading a valid text PDF creates chunks
- Uploading a scanned/image-only PDF shows the OCR message
- Asking a question before indexing shows a helpful warning
- Asking a question after indexing returns an answer with citations
- Retrieved context expander shows chunk text
- Document summary works after indexing
- `Reset all` clears indexed documents and conversation history
- `local` mode works without `OPENAI_API_KEY`
- `openai` mode shows a helpful error when `OPENAI_API_KEY` is missing

## 15. Troubleshooting

### `ModuleNotFoundError`

Make sure dependencies are installed in the active virtual environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `pytest: command not found`

Install test dependencies:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install pytest
```

### Streamlit command not found

Install dependencies and confirm the venv is active:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Sentence-transformer model download is slow

The default embedding model may download on first run. After it is cached locally, future runs are faster.

If the model cannot load, the app falls back to `HashingEmbeddingProvider`. That fallback keeps the app usable but retrieval quality will be lower.

### FAISS installation issues

The app prefers FAISS but can run without it. If `faiss-cpu` cannot import, `VectorIndex` falls back to numpy similarity search.

For best retrieval performance, install FAISS:

```bash
pip install faiss-cpu
```

### Uploaded PDF has no text

The PDF is probably scanned or image-only. Run OCR before uploading it. Tools that can OCR PDFs include:

- Adobe Acrobat OCR
- OCRmyPDF
- Tesseract-based workflows

After OCR, the PDF should contain selectable text and can be indexed.

### OpenAI mode says the API key is missing

Set the environment variable before launching Streamlit:

```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

## 16. Security and Privacy Notes

By default, local mode does not send document text to an external LLM API. Uploaded PDF content is processed on the machine running Streamlit.

OpenAI mode sends retrieved chunks and the user question to the OpenAI API. Use OpenAI mode only if that is acceptable for the documents being processed.

The app stores vector indexes under:

```text
.cache/vector_index
```

Do not commit cached indexes if the source documents are confidential. The included `.gitignore` excludes `.cache/`.

## 17. Current Limitations

- No OCR is performed inside the app.
- Local answer mode is extractive and simple.
- Chunk citations reference chunk numbers, not exact PDF page ranges.
- The app is intended for local/single-user use, not hardened multi-user deployment.
- The index cache is content-based but not a full document management system.
- Tables inside PDFs are extracted as text by `pypdf`; complex table structure may not be preserved.

## 18. Extension Ideas

### Add OCR

Add an OCR pipeline before `extract_text_from_pdf_file`, or add a separate ingestion path for scanned PDFs.

Possible tools:

- `ocrmypdf`
- `pytesseract`
- Cloud OCR APIs

### Add Page-Level Citations

The current extractor inserts `[Page N]` markers into extracted text. A future chunking strategy could preserve page spans in metadata so citations can include:

```text
annual_report.pdf - pages 12-13 - chunk 5
```

### Add Chroma Support

`VectorIndex` could be expanded or replaced with a Chroma-backed retriever. A clean way to do this would be defining a retriever interface and adding multiple implementations.

### Add Better Local LLM Support

For fully local generative answers, add a local model runner such as:

- Ollama
- llama.cpp
- LM Studio

Then implement a new mode in `rag/qa.py`.

### Add Document Management

For larger workflows, add:

- Saved document collections
- Per-user workspaces
- Delete individual documents
- Re-index changed documents
- Metadata filters

### Improve Financial Analysis

Financial-specific enhancements could include:

- Extracting key financial metrics
- Comparing fiscal years
- Summarizing risk factors
- Detecting revenue, margin, debt, and cash-flow sections
- Table extraction with `pdfplumber`

## 19. Development Notes

The implementation intentionally favors simple, readable code:

- Dataclasses represent documents and chunks.
- Each RAG step lives in its own module.
- The Streamlit app coordinates the workflow but does not own core logic.
- Tests use deterministic components where possible.
- Error messages are written for users, not just developers.

When modifying the project, keep the boundaries clear:

- Put PDF parsing in `rag/ingest.py`
- Put chunking logic in `rag/chunking.py`
- Put embedding providers in `rag/embeddings.py`
- Put indexing and retrieval in `rag/retriever.py`
- Put answering and summarization in `rag/qa.py`
- Keep UI concerns in `app.py`

## 20. Commands Reference

Create environment:

```bash
python3.10 -m venv .venv
```

Activate environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run app:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

Run syntax check:

```bash
python -m py_compile app.py rag/*.py tests/*.py
```

Set OpenAI key:

```bash
export OPENAI_API_KEY="your-key"
```

Clear runtime cache manually:

```bash
rm -rf .cache/vector_index
```

## 21. Expected Successful Run

After starting Streamlit, the app should show:

- A sidebar with PDF upload controls
- Answer mode selector
- Retrieved chunks slider
- `Index PDFs` button
- `Reset all` button
- Conversation history area
- Main chat area
- Document summary panel
- Source chunk panel

After indexing a valid PDF, the UI should show a success message similar to:

```text
Indexed 1 document(s) into 12 chunks.
```

After asking a grounded question, an answer should include:

```text
Sources: filename.pdf - chunk 1; filename.pdf - chunk 3
```

If the answer is not supported by retrieved context, the app should say:

```text
I could not find enough support for that answer in the uploaded documents.
```

