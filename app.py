from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.chunking import split_documents
from rag.embeddings import get_default_embedding_provider
from rag.ingest import extract_text_from_pdf_file, extract_text_from_url
from rag.qa import (
    MissingAPIKeyError,
    answer_question,
    is_document_identity_question,
    is_document_overview_question,
    summarize_chunks,
)
from rag.reranker import get_default_reranker, rerank_retrieved_chunks
from rag.retriever import VectorIndex
from rag.utils import EmptyPDFError, RetrievedChunk, short_hash
from rag.utils import WebpageIngestionError


INDEX_DIR = Path(".cache/vector_index")


st.set_page_config(page_title="Financial RAG Chatbot", page_icon=":page_facing_up:", layout="wide")


@st.cache_resource(show_spinner=False)
def embedding_provider():
    return get_default_embedding_provider()


@st.cache_resource(show_spinner=False)
def reranker_provider():
    return get_default_reranker()


@st.cache_resource(show_spinner=False)
def build_index_cached(cache_key: str, _chunks_tuple: tuple) -> VectorIndex:
    chunks = list(_chunks_tuple)
    provider = embedding_provider()
    index_path = INDEX_DIR / provider.identifier.replace("/", "_").replace(":", "_") / cache_key
    if (index_path / "index.pkl").exists():
        try:
            return VectorIndex.load(index_path, provider)
        except ValueError:
            shutil.rmtree(index_path)
    index = VectorIndex.from_chunks(chunks, provider)
    index.save(index_path)
    return index


def reset_state() -> None:
    st.session_state.pop("documents", None)
    st.session_state.pop("chunks", None)
    st.session_state.pop("index_key", None)
    st.session_state.pop("messages", None)
    st.session_state.pop("last_retrieval", None)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()


def initialize_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("messages", [])


def render_sidebar() -> tuple[str, int, int, bool]:
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        url_text = st.text_area(
            "Website URLs",
            placeholder="https://example.com/annual-report\nhttps://example.com/investors",
            height=90,
        )
        mode = st.radio("Answer mode", ["local", "openai"], horizontal=True)
        top_k = st.slider("Final chunks", min_value=2, max_value=10, value=5)
        candidate_k = st.slider("Candidate pool", min_value=5, max_value=30, value=max(15, top_k * 3))
        use_reranking = st.checkbox("Rerank candidates", value=False)
        provider = embedding_provider()
        if provider.identifier.startswith("hashing"):
            st.warning("Using fallback embeddings. Install sentence-transformers for better retrieval quality.")
        else:
            st.caption(f"Embeddings: {provider.identifier}")
        if use_reranking:
            reranker = reranker_provider()
            if reranker.identifier == "none":
                st.warning("Reranker unavailable. Vector similarity order will be used.")
            else:
                st.caption(f"Reranker: {reranker.identifier}")

        col_a, col_b = st.columns(2)
        with col_a:
            process_clicked = st.button("Index sources", use_container_width=True)
        with col_b:
            if st.button("Reset all", use_container_width=True):
                reset_state()
                st.rerun()

        if process_clicked:
            process_sources(uploaded_files, url_text)

        st.divider()
        st.subheader("Conversation")
        if st.session_state.messages:
            for message in st.session_state.messages[-8:]:
                st.caption(f"{message['role'].title()}: {message['content'][:180]}")
        else:
            st.caption("No messages yet.")

    return mode, top_k, max(candidate_k, top_k), use_reranking


def process_sources(uploaded_files, url_text: str) -> None:
    urls = [line.strip() for line in url_text.splitlines() if line.strip()]
    if not uploaded_files and not urls:
        st.warning("Upload at least one PDF or enter at least one website URL before indexing.")
        return

    documents = []
    errors = []
    for uploaded_file in uploaded_files or []:
        try:
            documents.append(extract_text_from_pdf_file(uploaded_file, uploaded_file.name))
        except EmptyPDFError as exc:
            errors.append(str(exc))
    for url in urls:
        try:
            documents.append(extract_text_from_url(url))
        except (EmptyPDFError, WebpageIngestionError) as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            st.error(error)
    if not documents:
        return

    chunks = split_documents(documents)
    if not chunks:
        st.error("No searchable chunks were created from the provided sources.")
        return

    key = short_hash([doc.filename + doc.text for doc in documents])
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.index_key = key
    build_index_cached(key, tuple(chunks))
    st.success(f"Indexed {len(documents)} source(s) into {len(chunks)} chunks.")


def main() -> None:
    initialize_state()
    mode, top_k, candidate_k, use_reranking = render_sidebar()

    st.title("Financial RAG Chatbot")
    st.caption("Upload financial PDFs or add website URLs, index them locally, and ask questions grounded in retrieved chunks.")

    if st.session_state.documents:
        names = ", ".join(doc.filename for doc in st.session_state.documents)
        st.info(f"Indexed: {names}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "source": doc.filename,
                        "type": doc.metadata.get("source_type", "pdf"),
                        "pages": doc.page_count,
                    }
                    for doc in st.session_state.documents
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    left, right = st.columns([0.68, 0.32], gap="large")

    with right:
        st.subheader("Document Summary")
        if st.button("Generate summary", use_container_width=True, disabled=not st.session_state.chunks):
            index = current_index()
            retrieved = index.search("financial performance revenue profit expenses cash flow risks outlook", k=min(8, len(st.session_state.chunks)))
            retrieved = with_opening_chunks(retrieved)
            try:
                summary = summarize_chunks(retrieved, mode=mode)
                st.write(summary.answer)
            except MissingAPIKeyError as exc:
                st.error(str(exc))

        render_chunk_inspector()

    with left:
        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input("Ask a question about the uploaded documents")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                if not st.session_state.chunks:
                    response_text = "Upload and index at least one PDF or website URL before asking questions."
                    st.warning(response_text)
                else:
                    index = current_index()
                    retrieved = index.search(question, k=candidate_k)
                    if is_document_identity_question(question) or is_document_overview_question(question):
                        retrieved = with_opening_chunks(retrieved)
                    if use_reranking:
                        retrieved = rerank_retrieved_chunks(question, retrieved, reranker_provider(), top_k=top_k)
                    else:
                        retrieved = retrieved[:top_k]
                    try:
                        response = answer_question(question, retrieved, mode=mode)
                        response_text = response.answer
                        st.write(response_text)
                    except MissingAPIKeyError as exc:
                        response_text = str(exc)
                        st.error(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})


def current_index() -> VectorIndex:
    key = st.session_state.get("index_key")
    chunks = tuple(st.session_state.get("chunks", []))
    if not key or not chunks:
        return VectorIndex(embedding_provider())
    return build_index_cached(key, chunks)


def with_opening_chunks(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    combined: list[RetrievedChunk] = []
    seen = set()
    per_file_counts: dict[str, int] = {}
    for chunk in st.session_state.get("chunks", []):
        count = per_file_counts.get(chunk.filename, 0)
        if count >= 4:
            continue
        per_file_counts[chunk.filename] = count + 1
        if chunk.citation not in seen:
            seen.add(chunk.citation)
            combined.append(RetrievedChunk(chunk=chunk, score=1.0))

    for item in retrieved:
        if item.chunk.citation not in seen:
            seen.add(item.chunk.citation)
            combined.append(item)
    return combined


def render_chunk_inspector() -> None:
    st.subheader("Chunk Inspector")
    chunks = st.session_state.get("chunks", [])
    if not chunks:
        st.caption("Index PDFs to inspect chunks.")
        return

    query = st.text_input("Search chunks", placeholder="revenue, risk, sustainability")
    rows = []
    for chunk in chunks:
        if query and query.lower() not in chunk.text.lower() and query.lower() not in chunk.filename.lower():
            continue
        rows.append(
            {
                "filename": chunk.filename,
                "chunk": chunk.chunk_number,
                "citation": chunk.citation,
                "preview": chunk.text[:220].replace("\n", " "),
            }
        )
    st.caption(f"Showing {len(rows)} of {len(chunks)} chunks.")
    st.dataframe(pd.DataFrame(rows[:50]), hide_index=True, use_container_width=True)

    citation_options = [row["citation"] for row in rows[:50]]
    if citation_options:
        selected = st.selectbox("Open chunk", citation_options)
        selected_chunk = next(chunk for chunk in chunks if chunk.citation == selected)
        st.text_area("Chunk text", selected_chunk.text, height=220)


if __name__ == "__main__":
    main()
