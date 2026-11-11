"""Tests for the dataset post-processing (Han filter + symptom masking)."""
from __future__ import annotations

import random

from vacxis_capstone_fa26.testdata.postprocess import (
    contains_han,
    filter_no_han,
    find_symptoms,
    mask_symptoms,
    split_and_mask,
)


def _record(query: str, chunk_text: str, chunk_id: str = "c1") -> dict:
    return {
        "QUERY_ID": "VAX_Q_0001",
        "QUERY": query,
        "RELEVANT_CHUNKS": [
            {"CHUNK_ID": chunk_id, "TEXT": chunk_text, "RELEVANCE_SCORE": 3},
            {"CHUNK_ID": "c2", "TEXT": "irrelevant", "RELEVANCE_SCORE": 0},
        ],
    }


def test_contains_han_detects_cjk():
    assert contains_han("hết 苗")
    assert contains_han({"a": "x", "b": ["z 苗"]})
    assert contains_han({"k": "苗"})
    assert not contains_han({"a": "x", "b": ["z"]})
    assert not contains_han("vaccine Verorab")
    assert not contains_han(["a", 3, None])


def test_filter_no_han_drops_cjk_records():
    kept = filter_no_han(
        [
            {"QUERY": "sốt sau tiêm"},
            {"QUERY": "疫苗 A"},
            {"QUERY": "ok", "RELEVANT_CHUNKS": [{"TEXT": "苗"}]},
        ]
    )
    assert [r["QUERY"] for r in kept] == ["sốt sau tiêm"]


def test_find_symptoms_prefers_longest_term():
    text = "chỗ tiêm nổi mề đay và sốt"
    found = find_symptoms(text)
    assert "nổi mề đay" in found
    assert "mề đay" not in found
    assert "sốt" in found


def test_find_symptoms_ignores_substring_matches():
    # "cho" and "khoảng" must not match the generic term "đỏ"/"ho".
    found = find_symptoms("dài khoảng thời gian cho phép")
    assert "ho" not in found
    assert "đỏ" not in found


def test_mask_symptoms_replaces_with_chunk_symptom():
    q = _record(
        "Sau khi tiêm vaccine A, bị đau đầu dữ dội thì phải làm sao?",
        "Các triệu chứng: sốt, đau đầu, mệt mỏi, tiêu chảy nhẹ.",
    )
    updated, report = mask_symptoms(q, random.Random(0))
    # "đau đầu" is the query symptom; it is replaced by a chunk symptom.
    assert report["replaced"] == ["đau đầu"]
    assert report["with"]
    assert "đau đầu" not in updated["QUERY"].lower()
    # The replacement really occurs in the top chunk.
    chunk_text = updated["RELEVANT_CHUNKS"][0]["TEXT"].lower()
    assert report["with"][0].lower() in chunk_text
    # Original record is not mutated.
    assert q["QUERY"] != updated["QUERY"]


def test_mask_symptoms_no_chunk_symptom_leaves_query_unchanged():
    q = _record(
        "Tôi bị đau đầu sau khi tiêm vaccine A?",
        "Bảo quản ở nơi khô ráo, tránh ánh sáng trực tiếp.",
    )
    updated, report = mask_symptoms(q, random.Random(0))
    assert report["replaced"] == []
    assert updated["QUERY"] == q["QUERY"]


def test_mask_symptoms_no_query_symptom_is_noop():
    q = _record(
        "Vaccine A được chỉ định cho ai?",
        "Các triệu chứng: sốt, đau đầu, tiêu chảy.",
    )
    updated, report = mask_symptoms(q, random.Random(0))
    assert report == {"replaced": [], "with": []}


def test_split_and_mask_masks_only_tail():
    records = [
        _record("bị sốt sau tiêm", "Triệu chứng: đau đầu, tiêu chảy."),
        _record("bị đau đầu sau tiêm", "Triệu chứng: sốt, mệt mỏi."),
        _record("không bị gì", "Bảo quản lạnh."),
    ]
    head, tail, stats = split_and_mask(
        records, split_at=1, rng=random.Random(0)
    )
    assert len(head) == 1 and len(tail) == 2
    assert head[0]["QUERY"] == "bị sốt sau tiêm"
    assert stats["split_at"] == 1
    # Tail record 0 has its symptom replaced; record 2 is unchanged.
    assert tail[0]["QUERY"] != "bị đau đầu sau tiêm"
    assert tail[1]["QUERY"] == "không bị gì"


def test_split_at_beyond_length_masks_nothing():
    records = [_record("sốt", "đau đầu")]
    head, tail, stats = split_and_mask(records, split_at=300, rng=random.Random(0))
    assert len(head) == 1 and tail == []
