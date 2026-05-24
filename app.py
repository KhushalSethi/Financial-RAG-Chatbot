from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from rag.chunking import split_documents
from rag.embeddings import get_default_embedding_provider
from rag.ingest import extract_text_from_pdf_file
from rag.qa import MissingAPIKeyError, answer_question, is_document_identity_question, summarize_chunks
from rag.retriever import VectorIndex
from rag.utils import EmptyPDFError, RetrievedChunk, short_hash


INDEX_DIR = Path(".cache/vector_index")


st.set_page_config(page_title="Financial RAG Chatbot", page_icon=":page_facing_up:", layout="wide")


@st.cache_resource(show_spinner=False)
def embedding_provider():
    return get_default_embedding_provider()


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
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    st.cache_resource.clear()


def initialize_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("messages", [])


def render_sidebar() -> tuple[str, int]:
    with st.sidebar:
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        mode = st.radio("Answer mode", ["local", "openai"], horizontal=True)
        top_k = st.slider("Retrieved chunks", min_value=2, max_value=10, value=5)
        provider = embedding_provider()
        if provider.identifier.startswith("hashing"):
            st.warning("Using fallback embeddings. Install sentence-transformers for better retrieval quality.")
        else:
            st.caption(f"Embeddings: {provider.identifier}")

        col_a, col_b = st.columns(2)
        with col_a:
            process_clicked = st.button("Index PDFs", use_container_width=True)
        with col_b:
            if st.button("Reset all", use_container_width=True):
                reset_state()
                st.rerun()

        if process_clicked:
            process_uploads(uploaded_files)

        st.divider()
        st.subheader("Conversation")
        if st.session_state.messages:
            for message in st.session_state.messages[-8:]:
                st.caption(f"{message['role'].title()}: {message['content'][:180]}")
        else:
            st.caption("No messages yet.")

    return mode, top_k


def process_uploads(uploaded_files) -> None:
    if not uploaded_files:
        st.warning("Upload at least one PDF before indexing.")
        return

    documents = []
    errors = []
    for uploaded_file in uploaded_files:
        try:
            documents.append(extract_text_from_pdf_file(uploaded_file, uploaded_file.name))
        except EmptyPDFError as exc:
            errors.append(str(exc))

    if errors:
        for error in errors:
            st.error(error)
    if not documents:
        return

    chunks = split_documents(documents)
    if not chunks:
        st.error("No searchable chunks were created from the uploaded PDFs.")
        return

    key = short_hash([doc.filename + doc.text for doc in documents])
    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.index_key = key
    build_index_cached(key, tuple(chunks))
    st.success(f"Indexed {len(documents)} document(s) into {len(chunks)} chunks.")


def main() -> None:
    initialize_state()
    mode, top_k = render_sidebar()

    st.title("Financial RAG Chatbot")
    st.caption("Upload financial PDFs, index them locally, and ask questions grounded in retrieved document chunks.")

    if st.session_state.documents:
        names = ", ".join(doc.filename for doc in st.session_state.documents)
        st.info(f"Indexed: {names}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"file": doc.filename, "pages": doc.page_count}
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
            try:
                summary = summarize_chunks(retrieved, mode=mode)
                st.write(summary.answer)
            except MissingAPIKeyError as exc:
                st.error(str(exc))

        st.subheader("Sources")
        for chunk in st.session_state.chunks[:12]:
            with st.expander(chunk.citation):
                st.write(chunk.text[:900])

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
                    response_text = "Upload and index at least one PDF before asking questions."
                    st.warning(response_text)
                else:
                    index = current_index()
                    retrieved = index.search(question, k=top_k)
                    if is_document_identity_question(question):
                        retrieved = with_opening_chunks(retrieved)
                    try:
                        response = answer_question(question, retrieved, mode=mode)
                        response_text = response.answer
                        st.write(response_text)
                        if response.retrieved_chunks:
                            with st.expander("Retrieved context"):
                                for item in response.retrieved_chunks:
                                    st.markdown(f"**{item.chunk.citation}** - score `{item.score:.3f}`")
                                    st.write(item.chunk.text)
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


if __name__ == "__main__":
    main()
