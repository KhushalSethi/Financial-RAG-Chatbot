from rag.qa import answer_question, is_document_identity_question, is_document_overview_question
from rag.utils import Chunk, RetrievedChunk


def test_document_identity_question_extracts_company_name():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                text="[Page 1]\nAsian Paints Limited\nIntegrated Annual Report 2023-24",
                filename="AP_Synopsis_revised_2023-24.pdf",
                chunk_number=1,
            ),
            score=1.0,
        ),
        RetrievedChunk(
            chunk=Chunk(
                text="The Company has established ethical business practices over 50 years.",
                filename="AP_Synopsis_revised_2023-24.pdf",
                chunk_number=13,
            ),
            score=0.7,
        ),
    ]

    response = answer_question("Which company is this document about?", chunks, mode="local")

    assert "Asian Paints Limited" in response.answer
    assert "Sources: AP_Synopsis_revised_2023-24.pdf - chunk 1" in response.answer


def test_detects_document_identity_question():
    assert is_document_identity_question("which company is this document about")
    assert not is_document_identity_question("What is this report about?")


def test_detects_document_overview_question():
    assert is_document_overview_question("What is the document talking about?")
    assert is_document_overview_question("What is this report about?")


def test_document_overview_question_uses_opening_context():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                text=(
                    "[Page 1]\nAsian Paints Limited\nIntegrated Annual Report 2023-24. "
                    "The report discusses business performance, sustainability, governance, and financial results."
                ),
                filename="AP_Synopsis_revised_2023-24.pdf",
                chunk_number=1,
            ),
            score=1.0,
        ),
        RetrievedChunk(
            chunk=Chunk(
                text="Limited Assurance Conclusion Based on procedures performed, nothing has come to our attention.",
                filename="AP_Synopsis_revised_2023-24.pdf",
                chunk_number=197,
            ),
            score=0.8,
        ),
    ]

    response = answer_question("What is the document talking about?", chunks, mode="local")

    assert "Asian Paints Limited" in response.answer
    assert "business performance" in response.answer
    assert "Limited Assurance Conclusion" not in response.answer


def test_local_answer_cleans_page_markers_and_inline_citations():
    chunks = [
        RetrievedChunk(
            chunk=Chunk(
                text="[Page 13] 13 | P a g e Breaking the mystery- Working capital can be defined as capital that keeps the business working.",
                filename="Decoding-Financial-Jargons.pdf",
                chunk_number=36,
            ),
            score=1.0,
        )
    ]

    response = answer_question("What is working capital?", chunks, mode="local")

    assert "[Page 13]" not in response.answer
    assert "13 | P a g e" not in response.answer
    assert response.answer.count("Sources:") == 1
