"""Offline tests for the query-driven (source) dataset path.

These exercise the rule-based rewriter, grounding, record schema and the
vLLM JSON-parsing helpers *without* a GPU: the real vLLM path is covered by
the marked integration test, which is skipped by default.
"""
from __future__ import annotations

import pytest

from vacxis_capstone_fa26.testdata.generator import (
    RuleBasedRewriter,
    generate_and_validate_from_source,
)
from vacxis_capstone_fa26.testdata.vllm_rewrite import _extract_json, _normalize_result

from .conftest import make_chunk


REQUIRED_KEYS = {
    "QUERY_ID", "QUERY", "QUERY_DIFFICULTY", "QUERY_TYPE", "QUERY_SOURCE",
    "QUERY_COMPLEXITY", "ENTITIES", "CONSTRAINTS", "ANSWERABILITY",
    "RELEVANT_DOCS", "RELEVANT_CHUNKS", "REQUIRED_EVIDENCE_GROUPS",
    "EVIDENCE_COMPLETENESS", "REFERENCE_ANSWER", "CLAIM_EVIDENCE_MAP",
    "SOURCE_VERSION", "VERIFIED", "ANNOTATOR_IDS", "ADJUDICATED",
    "VERIFICATION_NOTES",
}


def _corpus() -> list:
    return [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c1", "doc_a", "cảnh báo và thận trọng", "Thận trọng Vax A.", vaccine="Vax A"),
        make_chunk("b_c0", "doc_b", "chống chỉ định", "Chống chỉ định Vax B.", vaccine="Vax B"),
    ]


def test_rule_based_rewriter_shape():
    import random

    src = [{"id": 1, "query": "răng đau"}, {"id": 2, "query": "sốt"}]
    out = RuleBasedRewriter(_corpus(), random.Random(0)).rewrite(src)
    assert [o["id"] for o in out] == [1, 2]
    for o in out:
        assert o["vaccine"] in {"Vax A", "Vax B"}
        assert o["rewritten_query"] in {"răng đau", "sốt"}
        assert o["query_type"] in {
            "INDICATION", "DOSAGE", "ADMINISTRATION_ROUTE", "CONTRAINDICATION",
            "ADVERSE_EFFECT", "COMPOSITION", "STORAGE", "PRECAUTION",
            "INTERACTION", "SPECIAL_POPULATION",
        }


def test_query_path_records_are_schema_valid_and_grounded():
    import random

    src = [{"id": i, "query": f"question {i}"} for i in range(5)]
    queries, rpt = generate_and_validate_from_source(
        _corpus(),
        src,
        RuleBasedRewriter(_corpus(), random.Random(0)),
        random.Random(1),
        distractors_per_query=1,
    )
    assert len(queries) == 5
    assert rpt.ok, rpt.failures
    chunk_ids = {c.chunk_id for c in _corpus()}
    for q in queries:
        assert REQUIRED_KEYS <= set(q)
        assert q["QUERY_SOURCE"] == "VLLM_QUERY"
        if q["ANSWERABILITY"] != "UNANSWERABLE":
            # Every relevant chunk must point at a real corpus chunk.
            assert q["RELEVANT_CHUNKS"]
            for rc in q["RELEVANT_CHUNKS"]:
                assert rc["CHUNK_ID"] in chunk_ids


def test_extract_json_tolerates_fences_and_trailing_text():
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('prefix {"a": 1} suffix') == {"a": 1}
    assert _extract_json("no json here") is None
    assert _extract_json('[1, 2]') is None  # not a dict


def test_normalize_result_defaults_on_garbage():
    r = _normalize_result({}, fallback_id=7)
    assert r["id"] == 7
    assert r["query_type"] == "PRECAUTION"
    assert r["answerability"] == "ANSWERABLE"
    good = _normalize_result(
        {"vaccine": "X", "query_type": "dosage", "answerability": "UNANSWERABLE"},
        fallback_id=7,
    )
    assert good["query_type"] == "DOSAGE"  # upper-cased
    assert good["answerability"] == "UNANSWERABLE"
    assert good["vaccine"] == "X"


@pytest.mark.integration
def test_vllm_path_smoke(monkeypatch):
    """Exercises the real vLLM classes on a GPU host; skipped by default."""
    from vacxis_capstone_fa26.testdata.vllm_rewrite import QwenRewriter, VLLMClient

    client = VLLMClient(model="Qwen/Qwen2.5-7B-Instruct")
    client.ensure_loaded()
    rw = QwenRewriter(client, vaccine_names=["Vax A"])
    out = rw.rewrite([{"id": 1, "query": "phụ nữ mang thai có tiêm được không"}])
    assert out[0]["vaccine"]
