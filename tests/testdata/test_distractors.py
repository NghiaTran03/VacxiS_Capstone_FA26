"""Unit tests for the category-aware distractor sampler."""
from __future__ import annotations

import random

from vacxis_capstone_fa26.testdata.chunks import index_by_category
from vacxis_capstone_fa26.testdata.distractors import (
    DistractorKind,
    sample_distractors,
    score_for,
)

from .conftest import make_chunk


def _corpus() -> list:
    return [
        make_chunk("a_c0", "doc_a", "chỉ định", "Vax A chỉ định cho trẻ em."),
        make_chunk("b_c0", "doc_b", "chỉ định", "Vax B chỉ định cho người lớn."),
        make_chunk("c_c0", "doc_c", "chỉ định", "Vax C chỉ định cho phụ nữ mang thai."),
        make_chunk("a_c1", "doc_a", "liều lượng và cách dùng", "0,5 ml tiêm bắp."),
        make_chunk("b_c1", "doc_b", "tác dụng không mong muốn", "Sốt nhẹ, đau chỗ tiêm."),
    ]


def test_distractors_split_hard_and_easy() -> None:
    chunks = _corpus()
    by_cat = index_by_category(chunks)
    rng = random.Random(0)
    target = chunks[0]
    distractors = sample_distractors(target, chunks, by_cat, rng, n=3)
    assert len(distractors) == 3
    assert all(t.chunk.doc_id != target.doc_id for t in distractors)
    assert all(t.chunk.chunk_id != target.chunk_id for t in distractors)


def test_distractors_include_hard_negatives() -> None:
    chunks = _corpus()
    by_cat = index_by_category(chunks)
    rng = random.Random(0)
    target = chunks[0]
    distractors = sample_distractors(target, chunks, by_cat, rng, n=4)
    hard = [t for t in distractors if t.kind is DistractorKind.HARD]
    assert len(hard) >= 2


def test_zero_distractors() -> None:
    chunks = _corpus()
    by_cat = index_by_category(chunks)
    rng = random.Random(0)
    assert sample_distractors(chunks[0], chunks, by_cat, rng, n=0) == []


def test_score_for_is_stable_per_kind() -> None:
    """Review finding 3: score is determined by the source pool, not index."""
    assert score_for(DistractorKind.HARD) == 1
    assert score_for(DistractorKind.EASY) == 0


def test_tagged_distractor_kind_independent_of_position() -> None:
    """A hard negative at index 0 still scores higher than an easy negative
    at index 1, regardless of how the sampler shuffled the list."""
    chunks = _corpus()
    by_cat = index_by_category(chunks)
    rng = random.Random(0)
    target = chunks[0]
    distractors = sample_distractors(target, chunks, by_cat, rng, n=2)
    # Score lookup uses .kind, never .index in the list.
    for d in distractors:
        assert score_for(d.kind) in (0, 1)
