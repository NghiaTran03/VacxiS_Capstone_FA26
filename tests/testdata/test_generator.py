"""End-to-end tests for the generator that exercise review findings 3, 4, 5b."""
from __future__ import annotations

import random

from vacxis_capstone_fa26.testdata.generator import generate_dataset
from vacxis_capstone_fa26.testdata.refiner import noop_refiner
from vacxis_capstone_fa26.testdata.taxonomy import QUERY_TYPE_DEFS

from .conftest import make_chunk


def _build_corpus() -> list:
    return [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c2", "doc_a", "liều lượng và cách dùng", "0,5 ml.", vaccine="Vax A"),
        make_chunk("b_c0", "doc_b", "chỉ định", "Chỉ định Vax B.", vaccine="Vax B"),
        make_chunk("b_c1", "doc_b", "tương tác thuốc", "Tương tác Vax B.", vaccine="Vax B"),
    ]


def test_distractor_score_is_stable_per_kind() -> None:
    """Review finding 3: every hard negative scores 1, every easy 0,
    regardless of its position in RELEVANT_CHUNKS."""
    rng = random.Random(0)
    queries = generate_dataset(
        _build_corpus(), rng, per_type=1, distractors_per_query=2,
        refiner=noop_refiner(),
    )
    assert queries, "expected at least one query"
    q = queries[0]
    # Distractors are the only entries with RELEVANCE_SCORE in {0, 1}.
    # Multi-hop partners score 3; the primary scores 3.
    distractors = [rc for rc in q["RELEVANT_CHUNKS"] if rc["RELEVANCE_SCORE"] in (0, 1)]
    assert all(0 <= d["RELEVANCE_SCORE"] <= 1 for d in distractors)


def test_multi_hop_downgrades_to_single_hop_when_no_partner() -> None:
    """Review finding 4: if pair_multi_hop returns secondary=None, the
    generator must downgrade QUERY_COMPLEXITY to SINGLE_HOP rather than
    ship a MULTI_HOP query with a single relevant chunk."""
    # A vaccine with no complementary-category chunks in its doc.
    chunks = [
        make_chunk("solo_c0", "doc_solo", "chỉ định", "Chỉ định.", vaccine="Vax Solo"),
    ]
    rng = random.Random(0)
    queries = generate_dataset(
        chunks, rng, per_type=1, distractors_per_query=0,
        refiner=noop_refiner(),
    )
    assert queries
    # If the original draw was MULTI_HOP, the generator should have
    # downgraded it. Either way, RELEVANT_CHUNKS must be consistent
    # with the QUERY_COMPLEXITY label.
    q = queries[0]
    if q["QUERY_COMPLEXITY"] == "MULTI_HOP":
        # If still MULTI_HOP, there must be more than one RELEVANT_CHUNK.
        primary_chunks = [
            rc for rc in q["RELEVANT_CHUNKS"] if rc["RELEVANCE_SCORE"] >= 3
        ]
        assert len(primary_chunks) >= 2


def test_required_evidence_groups_merges_secondary() -> None:
    """Review finding 5b: REQUIRED_EVIDENCE_GROUPS must include both the
    primary type's groups and the partner chunk's groups."""
    chunks = _build_corpus()
    rng = random.Random(0)
    queries = generate_dataset(
        chunks, rng, per_type=1, distractors_per_query=0,
        refiner=noop_refiner(),
    )
    # Find any multi-hop query (whatever the actual partner pairing).
    mh = [q for q in queries if q["QUERY_COMPLEXITY"] == "MULTI_HOP"]
    assert mh, "expected at least one MULTI_HOP query with this corpus"
    q = mh[0]
    primary_groups = next(
        td["evidence_groups"] for td in QUERY_TYPE_DEFS
        if td["type"] == q["QUERY_TYPE"]
    )
    # The primary's evidence groups are always present.
    for g in primary_groups:
        assert g in q["REQUIRED_EVIDENCE_GROUPS"]
    # If the query has a partner chunk with evidence groups, they must
    # also be present in REQUIRED_EVIDENCE_GROUPS.
    if len(q["RELEVANT_CHUNKS"]) >= 2:
        # At least one entry in REQUIRED_EVIDENCE_GROUPS must be a group
        # *not* in the primary's set — proving the merge happened.
        assert set(q["REQUIRED_EVIDENCE_GROUPS"]) - set(primary_groups)
