"""Unit tests for multi-hop pairing rules."""
from __future__ import annotations

import random

from vacxis_capstone_fa26.testdata.chunks import index_by_vaccine
from vacxis_capstone_fa26.testdata.multi_hop import pair_multi_hop
from vacxis_capstone_fa26.testdata.taxonomy import QUERY_TYPE_DEFS

from .conftest import make_chunk


def _by_type(t: str) -> dict:
    return next(td for td in QUERY_TYPE_DEFS if td["type"] == t)


def test_same_vaccine_multi_hop() -> None:
    chunks = [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c2", "doc_a", "liều lượng và cách dùng", "0,5 ml.", vaccine="Vax A"),
    ]
    by_vaccine = index_by_vaccine(chunks)
    rng = random.Random(0)
    primary = chunks[0]
    res = pair_multi_hop(
        primary, _by_type("INDICATION"),
        by_category={}, by_vaccine=by_vaccine,
        vaccine_mapping={}, used_partner_vaccines=set(), rng=rng,
    )
    assert res.secondary is not None
    assert res.secondary.doc_id == primary.doc_id
    assert res.secondary.chunk_id != primary.chunk_id
    assert res.cross_doc is False


def test_interaction_multi_hop_crosses_vaccines() -> None:
    chunks = [
        make_chunk("a_c0", "doc_a", "tương tác thuốc", "Tương tác Vax A.", vaccine="Vax A"),
        make_chunk("b_c0", "doc_b", "tương tác thuốc", "Tương tác Vax B.", vaccine="Vax B"),
    ]
    by_vaccine = index_by_vaccine(chunks)
    rng = random.Random(0)
    primary = chunks[0]
    res = pair_multi_hop(
        primary, _by_type("INTERACTION"),
        by_category={}, by_vaccine=by_vaccine,
        vaccine_mapping={}, used_partner_vaccines=set(), rng=rng,
    )
    assert res.secondary is not None
    assert res.secondary.vaccine_name != primary.vaccine_name
    assert res.cross_doc is True


def test_secondary_evidence_groups_populated() -> None:
    """Review finding 5b: MultiHopResult.secondary_evidence_groups must
    reflect the partner chunk's category, not be an empty list."""
    chunks = [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
        make_chunk("a_c1", "doc_a", "chống chỉ định", "Chống chỉ định Vax A.", vaccine="Vax A"),
    ]
    by_vaccine = index_by_vaccine(chunks)
    rng = random.Random(0)
    primary = chunks[0]
    res = pair_multi_hop(
        primary, _by_type("INDICATION"),
        by_category={}, by_vaccine=by_vaccine,
        vaccine_mapping={}, used_partner_vaccines=set(), rng=rng,
    )
    assert res.secondary is not None
    assert res.secondary_evidence_groups
    assert "CONTRAINDICATIONS" in res.secondary_evidence_groups


def test_no_partner_returns_none() -> None:
    """When no candidate exists, MultiHopResult.secondary is None so the
    generator can downgrade QUERY_COMPLEXITY (review finding 4)."""
    chunks = [
        make_chunk("a_c0", "doc_a", "chỉ định", "Chỉ định Vax A.", vaccine="Vax A"),
    ]
    by_vaccine = index_by_vaccine(chunks)
    rng = random.Random(0)
    res = pair_multi_hop(
        chunks[0], _by_type("INDICATION"),
        by_category={}, by_vaccine=by_vaccine,
        vaccine_mapping={}, used_partner_vaccines=set(), rng=rng,
    )
    assert res.secondary is None
    assert res.secondary_evidence_groups == []
