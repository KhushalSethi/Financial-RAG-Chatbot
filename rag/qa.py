from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rag.utils import MissingAPIKeyError, RetrievedChunk


@dataclass(frozen=True)
class QAResponse:
    answer: str
    citations: list[str]
    retrieved_chunks: list[RetrievedChunk]


NOT_FOUND = "I could not find enough support for that answer in the uploaded documents."

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "did",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "more",
    "not",
    "our",
    "the",
    "their",
    "this",
    "that",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "year",
}


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    mode: str = "local",
    openai_model: str = "gpt-4o-mini",
) -> QAResponse:
    if not retrieved_chunks:
        return QAResponse(answer=NOT_FOUND, citations=[], retrieved_chunks=[])

    if mode == "openai":
        answer = _answer_with_openai(question, retrieved_chunks, model=openai_model)
    else:
        answer = _answer_locally(question, retrieved_chunks)

    citations = _citations(retrieved_chunks)
    if citations and answer != NOT_FOUND:
        answer = f"{answer}\n\nSources: " + "; ".join(citations)
    return QAResponse(answer=answer, citations=citations, retrieved_chunks=retrieved_chunks)


def summarize_chunks(
    retrieved_chunks: list[RetrievedChunk],
    mode: str = "local",
    openai_model: str = "gpt-4o-mini",
) -> QAResponse:
    if not retrieved_chunks:
        return QAResponse(answer="No indexed document content is available to summarize.", citations=[], retrieved_chunks=[])
    if mode == "openai":
        question = "Summarize the uploaded financial documents using only the provided context."
        answer = _answer_with_openai(question, retrieved_chunks, model=openai_model)
    else:
        sentences = []
        for item in retrieved_chunks:
            sentences.extend(_sentences(item.chunk.text)[:2])
        answer = " ".join(sentences[:8]).strip() or NOT_FOUND
    citations = _citations(retrieved_chunks)
    return QAResponse(answer=f"{answer}\n\nSources: " + "; ".join(citations), citations=citations, retrieved_chunks=retrieved_chunks)


def _answer_with_openai(question: str, retrieved_chunks: list[RetrievedChunk], model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("OPENAI_API_KEY is not set. Use local mode or configure the key before using OpenAI answers.")

    from openai import OpenAI

    context = _format_context(retrieved_chunks)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer financial document questions using only the supplied context. "
                    "If the context does not contain the answer, say that the documents do not provide enough information. "
                    "Do not use outside knowledge."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    return response.choices[0].message.content.strip()


def _answer_locally(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    useful_chunks = _filter_weak_matches(retrieved_chunks)
    if not useful_chunks:
        return NOT_FOUND

    query_terms = _terms(question)
    if not query_terms:
        return NOT_FOUND

    candidate_sentences: list[tuple[float, str, str]] = []
    for item in useful_chunks:
        for sentence in _sentences(item.chunk.text):
            sentence_terms = _terms(sentence)
            overlap = len(query_terms & sentence_terms)
            if overlap:
                number_bonus = 0.5 if re.search(r"[$%]|\b\d[\d,.]*\b", sentence) else 0.0
                candidate_sentences.append((overlap + number_bonus + item.score, sentence, item.chunk.citation))

    if not candidate_sentences:
        return NOT_FOUND

    candidate_sentences.sort(key=lambda value: value[0], reverse=True)
    unique: list[str] = []
    seen = set()
    for _, sentence, citation in candidate_sentences:
        normalized = _normalize_for_dedupe(sentence)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(f"- {sentence} ({citation})")
        if len(unique) == 3:
            break

    if not unique:
        return NOT_FOUND
    return "Relevant evidence found:\n" + "\n".join(unique)


def _filter_weak_matches(retrieved_chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not retrieved_chunks:
        return []
    best_score = max(item.score for item in retrieved_chunks)
    if best_score < 0.08:
        return []
    cutoff = max(0.08, best_score * 0.45)
    return [item for item in retrieved_chunks if item.score >= cutoff]


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z0-9$%]+", text.lower())
        if len(term) > 2 and term not in STOPWORDS
    }


def _normalize_for_dedupe(text: str) -> str:
    text = re.sub(r"\[page \d+\]", "", text.lower())
    text = re.sub(r"[^a-z0-9$%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", compact)
    return [part.strip() for part in parts if len(part.strip()) > 20]


def _format_context(retrieved_chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{item.chunk.citation}]\n{item.chunk.text}" for item in retrieved_chunks
    )


def _citations(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    citations: list[str] = []
    seen = set()
    for item in retrieved_chunks:
        citation = item.chunk.citation
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
