"""Unit tests for the answer generator and claim-evidence mapper."""
from __future__ import annotations

import pytest

from vacxis_capstone_fa26.testdata.answer_generator import (
    ExtractiveAnswerGenerator,
    NoOpAnswerGenerator,
    UNANSWERABLE_FALLBACK_ANSWER,
    enrich_dataset,
)
from vacxis_capstone_fa26.testdata.validator import validate

from .conftest import make_chunk


CHUNKS = [
    make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định tiêm cho người lớn.", vaccine="Vax A"),
    make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định phụ nữ có thai.", vaccine="Vax A"),
]


def test_extractive_answer_unanswerable() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q1",
        "QUERY": "Giá vaccine là bao nhiêu?",
        "ANSWERABILITY": "UNANSWERABLE",
        "RELEVANT_CHUNKS": [],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert q["REFERENCE_ANSWER"] == UNANSWERABLE_FALLBACK_ANSWER
    assert q["CLAIM_EVIDENCE_MAP"] == []


def test_extractive_answer_single_hop() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q2",
        "QUERY": "Vaccine Vax A dùng cho ai?",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": 0, "END": len(CHUNKS[0].text)}],
            }
        ],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert q["REFERENCE_ANSWER"] is not None
    assert "[a_c0]" in q["REFERENCE_ANSWER"]
    assert len(q["CLAIM_EVIDENCE_MAP"]) == 1
    assert q["CLAIM_EVIDENCE_MAP"][0]["chunk_id"] == "a_c0"


def test_extractive_answer_multi_hop() -> None:
    gen = ExtractiveAnswerGenerator()
    q = {
        "QUERY_ID": "Q3",
        "QUERY": "Chỉ định và chống chỉ định của Vax A?",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": 0, "END": 10}],
            },
            {
                "CHUNK_ID": "a_c1",
                "TEXT": CHUNKS[1].text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": 0, "END": 10}],
            },
        ],
        "REFERENCE_ANSWER": None,
        "CLAIM_EVIDENCE_MAP": [],
    }
    gen.generate_answers([q])
    assert "[a_c0]" in q["REFERENCE_ANSWER"]
    assert "[a_c1]" in q["REFERENCE_ANSWER"]
    assert len(q["CLAIM_EVIDENCE_MAP"]) == 2


def test_enrich_dataset_respects_existing_answer() -> None:
    existing_answer = "Đã có câu trả lời mẫu [a_c0]."
    q = {
        "QUERY_ID": "Q4",
        "QUERY": "Câu hỏi?",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": 0, "END": 10}],
            }
        ],
        "REFERENCE_ANSWER": existing_answer,
        "CLAIM_EVIDENCE_MAP": [],
    }
    enrich_dataset([q], generator=ExtractiveAnswerGenerator(), overwrite=False)
    assert q["REFERENCE_ANSWER"] == existing_answer


def test_validator_with_reference_answer() -> None:
    q = {
        "QUERY_ID": "Q5",
        "QUERY": "Chỉ định?",
        "QUERY_TYPE": "INDICATION",
        "QUERY_COMPLEXITY": "SINGLE_HOP",
        "ANSWERABILITY": "ANSWERABLE",
        "RELEVANT_DOCS": ["doc_a"],
        "RELEVANT_CHUNKS": [
            {
                "CHUNK_ID": "a_c0",
                "TEXT": CHUNKS[0].text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": 0, "END": 10}],
            }
        ],
        "REFERENCE_ANSWER": "Chỉ định cho người lớn [a_c0].",
        "CLAIM_EVIDENCE_MAP": [{"claim": "Chỉ định người lớn", "chunk_id": "a_c0", "evidence_quote": "..."}],
    }
    rpt = validate([q], CHUNKS, by_category={"chỉ định": [CHUNKS[0]]}, expected_per_type=1, expected_types=["INDICATION"])
    assert rpt.ok, rpt.render()
