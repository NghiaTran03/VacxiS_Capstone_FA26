"""Multi-hop chunk pairing rules.

Two strategies:

* ``same_vaccine_other_category`` (default for every type except INTERACTION):
  find another chunk from the *same* vaccine / doc whose category matches a
  complementary evidence group. This is the realistic case: "I know the dose
  of vaccine A, but is it contraindicated for my condition?" pulls from the
  same product insert.

* ``cross_vaccine`` (INTERACTION only): pick a second vaccine from the
  ``vaccine_name_mapping.json`` and grab its interaction / dosage chunk.
  Real interaction questions do cross products ("can I take A with B?").

When the preferred partner is missing we fall back gracefully and emit a
warning. The generator decides per query whether to apply MULTI_HOP based
on the complexity distribution.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .chunks import Chunk
from .taxonomy import INTERACTION_PARTNER_CATEGORIES


# Complementary categories that pair naturally with each primary type. The
# key is the primary QUERY_TYPE; the value is a tuple of candidate categories
# we look for in the *same* vaccine, ordered by preference.
SAME_VACCINE_PARTNER_CATEGORY: dict[str, tuple[str, ...]] = {
    "INDICATION": ("chống chỉ định", "cảnh báo và thận trọng"),
    "DOSAGE": ("chống chỉ định", "cảnh báo và thận trọng"),
    "ADMINISTRATION_ROUTE": ("cảnh báo và thận trọng", "liều lượng và cách dùng"),
    "CONTRAINDICATION": ("cảnh báo và thận trọng", "thành phần"),
    "ADVERSE_EFFECT": ("cảnh báo và thận trọng", "thành phần"),
    "COMPOSITION": ("tác dụng không mong muốn", "cảnh báo và thận trọng"),
    "STORAGE": ("quy cách đóng gói", "hướng dẫn sử dụng"),
    "PRECAUTION": ("chống chỉ định", "tác dụng không mong muốn"),
    "SPECIAL_POPULATION": ("cảnh báo và thận trọng", "tác dụng không mong muốn"),
}

# Map Vietnamese chunk categories to the SCHEMA_REQ evidence groups they
# satisfy. Used to populate ``secondary_evidence_groups`` so the
# ``REQUIRED_EVIDENCE_GROUPS`` field reflects the full multi-hop requirement
# (review-1 finding 5b).
CATEGORY_TO_EVIDENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "chỉ định": ("INDICATIONS",),
    "liều lượng và cách dùng": (
        "NUMBER_OF_DOSES", "DOSING_INTERVAL", "ADMINISTRATION_ROUTE",
    ),
    "chống chỉ định": ("CONTRAINDICATIONS",),
    "tác dụng không mong muốn": ("ADVERSE_EFFECTS",),
    "thành phần": ("COMPOSITION",),
    "bảo quản": ("STORAGE",),
    "cảnh báo và thận trọng": ("PRECAUTIONS",),
    "tương tác thuốc": ("INTERACTIONS",),
    "thai kỳ và cho con bú": ("SPECIAL_POPULATIONS",),
}


@dataclass
class MultiHopResult:
    """Result of pairing a primary chunk with a multi-hop partner."""

    secondary: Chunk | None
    secondary_evidence_groups: list[str]
    cross_doc: bool


def pair_multi_hop(
    primary: Chunk,
    type_def: dict,
    by_category: dict[str, list[Chunk]],
    by_vaccine: dict[str, list[Chunk]],
    vaccine_mapping: dict[str, str],
    used_partner_vaccines: set[str],
    rng: random.Random,
) -> MultiHopResult:
    """Pair ``primary`` with a multi-hop partner chunk.

    ``vaccine_mapping`` is the ``vaccine_name_mapping.json`` dict (id -> name).
    ``used_partner_vaccines`` is a per-run set the caller maintains so
    INTERACTION multi-hop queries each get a distinct partner vaccine.
    """
    rule = type_def.get("multi_hop_partner", "same_vaccine_other_category")
    qtype = type_def["type"]

    if rule == "cross_vaccine" and qtype == "INTERACTION":
        return _pair_interaction(
            primary, by_vaccine, vaccine_mapping, used_partner_vaccines, rng,
        )

    # Default: same vaccine, different category.
    candidates = [
        c for c in by_vaccine.get(primary.vaccine_name, [])
        if c.chunk_id != primary.chunk_id and c.category != primary.category
    ]
    preferred = SAME_VACCINE_PARTNER_CATEGORY.get(qtype, ())
    for cat in preferred:
        matching = [c for c in candidates if c.category == cat]
        if matching:
            return _result(matching, rng, cross_doc=False)
    if candidates:
        return _result(candidates, rng, cross_doc=False)
    return MultiHopResult(secondary=None, secondary_evidence_groups=[], cross_doc=False)


def _result(
    candidates: list[Chunk],
    rng: random.Random,
    cross_doc: bool,
) -> MultiHopResult:
    """Pick a chunk and derive its evidence groups from the chunk category."""
    secondary = rng.choice(candidates)
    groups = list(CATEGORY_TO_EVIDENCE_GROUPS.get(secondary.category, ()))
    return MultiHopResult(
        secondary=secondary,
        secondary_evidence_groups=groups,
        cross_doc=cross_doc,
    )


def _pair_interaction(
    primary: Chunk,
    by_vaccine: dict[str, list[Chunk]],
    vaccine_mapping: dict[str, str],
    used_partner_vaccines: set[str],
    rng: random.Random,
) -> MultiHopResult:
    available = [
        name for name in by_vaccine
        if name != primary.vaccine_name and name not in used_partner_vaccines
    ]
    if not available:
        # Cycle back through every other vaccine.
        available = [n for n in by_vaccine if n != primary.vaccine_name]
    if not available:
        return MultiHopResult(secondary=None, secondary_evidence_groups=[], cross_doc=True)

    partner_vaccine = rng.choice(available)
    used_partner_vaccines.add(partner_vaccine)

    partner_chunks = by_vaccine.get(partner_vaccine, [])
    for cat in INTERACTION_PARTNER_CATEGORIES:
        matching = [c for c in partner_chunks if c.category == cat]
        if matching:
            return _result(matching, rng, cross_doc=True)
    if partner_chunks:
        return _result(partner_chunks, rng, cross_doc=True)
    return MultiHopResult(secondary=None, secondary_evidence_groups=[], cross_doc=True)
