"""Unit tests for the deterministic evidence-span selector."""
from __future__ import annotations

import random

from vacxis_capstone_fa26.testdata.spans import select_evidence_span

from .conftest import SAMPLE_DOSAGE_TEXT, SAMPLE_INDICATION_TEXT


def test_span_points_at_intent_keyword() -> None:
    text = SAMPLE_DOSAGE_TEXT
    rng = random.Random(0)
    start, end = select_evidence_span(text, ["liều", "phác đồ"], rng)
    sentence = text[start:end]
    # The selector should pick the sentence mentioning both keywords and digits.
    assert "liều" in sentence.lower() or "phác đồ" in sentence.lower()


def test_span_prefers_intent_sentence() -> None:
    # Indication sentence has a number but no keyword; dosage sentence has both.
    text = (
        "Bệnh nhân được tiêm 2 mũi cách nhau 4 tuần.\n"
        "Liều khuyến cáo của vắc xin là 0,5 ml tiêm bắp mỗi liều theo phác đồ."
    )
    rng = random.Random(0)
    start, end = select_evidence_span(text, ["liều", "phác đồ"], rng)
    sentence = text[start:end]
    assert "phác đồ" in sentence.lower() or "liều" in sentence.lower()


def test_span_in_bounds() -> None:
    text = SAMPLE_DOSAGE_TEXT
    rng = random.Random(7)
    start, end = select_evidence_span(text, ["liều"], rng)
    assert 0 <= start < end <= len(text)


def test_span_falls_back_when_no_keywords() -> None:
    text = "Không có từ khóa phù hợp ở đây nhưng vẫn có câu dài đủ để được chọn làm fallback."
    rng = random.Random(0)
    start, end = select_evidence_span(text, ["khongtontai"], rng)
    assert 0 <= start < end <= len(text)


def test_span_fallback_skips_synthetic_header() -> None:
    """Review finding 5c: when the chunk has no scorable sentences (e.g.
    table-only), the fallback must skip the synthetic
    ``[CHỦ ĐỀ: ...] > [MỤC: ...]`` header rather than landing on it.
    """
    text = (
        "[CHỦ ĐỀ: Vax A] > [MỤC: BẢNG]\n"
        "1\n2\n3\n"
    )
    rng = random.Random(0)
    start, end = select_evidence_span(text, ["no-such-keyword"], rng)
    span_text = text[start:end]
    assert "[CHỦ ĐỀ" not in span_text
    assert span_text.startswith("1")
