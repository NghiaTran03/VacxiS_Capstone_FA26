"""Post-processing for generated test datasets.

Two transformations, applied in order by the CLI:

1. **Han filter** -- drop any record that contains a CJK ideograph
   anywhere in its payload (keys, values, nested lists). The rewriter model
   occasionally emits Chinese vaccine names or characters; those records are
   unusable for a Vietnamese evaluation set.

2. **Symptom masking** -- for the records past the split point, replace the
   symptoms mentioned in the query with symptoms that actually occur in the
   record's most relevant chunk. This makes the query answerable from the
   evidence it is grounded on, instead of asking about a symptom the chunk
   never mentions.

The symptom lexicon is Vietnamese and was curated against the chunk corpus,
so every term is one that genuinely occurs in the source documents.
"""
from __future__ import annotations

import random
import re
from typing import Any, Iterable

# CJK ideographs (and compatibility/extension ranges) to filter out.
HAN_PATTERN = re.compile(
    r"[㐀-䶿一-鿿豈-﫿"
    r"\U00020000-\U0002FA1F\U00030000-\U000323AF]"
)

# Vietnamese symptom / adverse-event vocabulary found in the chunk corpus.
# Ordered longest-first at match time so multi-word terms win over their
# substrings ("nổi mề đay" before "mề đay").
SYMPTOM_LEXICON: tuple[str, ...] = (
    "rối loạn tiêu hóa",
    "đau tại chỗ tiêm",
    "viêm mạch máu",
    "đau nửa đầu",
    "nổi mề đay",
    "đau đầu",
    "đau cơ",
    "đau khớp",
    "đau bụng",
    "đau lưng",
    "mệt mỏi",
    "buồn nôn",
    "chóng mặt",
    "phát ban",
    "ban đỏ",
    "phản vệ",
    "dị ứng",
    "mề đay",
    "tiêu chảy",
    "khó thở",
    "sưng hạch",
    "nổi hạch",
    "chán ăn",
    "mất ngủ",
    "sốt cao",
    "sốt",
    "nôn",
    "ngứa",
    "sưng",
    "đỏ",
    "co giật",
    "run rẩy",
    "phù mạch",
    "hắt hơi",
    "sổ mũi",
    "lạnh run",
    "choáng",
    "hét",
)

# Longest-first so "rối loạn tiêu hóa" is preferred over "rối loạn".
_SYMPTOMS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(SYMPTOM_LEXICON, key=len, reverse=True)
)

# Symptom terms that are too generic to be meaningful evidence on their own.
# They are never chosen as a replacement target.
GENERIC_SYMPTOMS: frozenset[str] = frozenset(
    {"đỏ", "ngứa", "sưng", "nôn", "sốt", "hét"}
)


def contains_han(value: Any) -> bool:
    """Return True if ``value`` contains a Han character anywhere."""
    if isinstance(value, str):
        return bool(HAN_PATTERN.search(value))
    if isinstance(value, dict):
        return any(
            contains_han(k) or contains_han(v) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_han(item) for item in value)
    return False


def filter_no_han(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop every record containing a Han character."""
    kept: list[dict[str, Any]] = []
    for record in records:
        if not contains_han(record):
            kept.append(record)
    return kept


def _term_pattern(term: str) -> re.Pattern[str]:
    """Word-boundary matcher for a Vietnamese term."""
    return re.compile(
        r"(?<![a-zà-ỹ])" + re.escape(term) + r"(?![a-zà-ỹ])", re.IGNORECASE
    )


def find_symptoms(text: str) -> list[str]:
    """Return the distinct symptom terms occurring in ``text``.

    Terms are matched longest-first and non-overlapping, so "nổi mề đay" is
    reported instead of the substring "mề đay".
    """
    if not text:
        return []
    lowered = text.lower()
    claimed: list[tuple[int, int]] = []
    found: list[str] = []
    for term in _SYMPTOMS_BY_LENGTH:
        for match in _term_pattern(term).finditer(lowered):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in claimed):
                continue
            claimed.append(span)
            found.append(match.group(0))
    return found


def _primary_chunk_text(record: dict[str, Any]) -> str:
    """Text of the record's most relevant chunk (highest relevance score)."""
    chunks = record.get("RELEVANT_CHUNKS") or []
    if not chunks:
        return ""
    best = max(chunks, key=lambda c: c.get("RELEVANCE_SCORE", 0))
    return best.get("TEXT", "") or ""


def mask_symptoms(
    record: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace the query's symptoms with symptoms present in its top chunk.

    Returns ``(updated_record, report)`` where ``report`` holds the terms
    that were replaced and the terms they were replaced with. The record is
    copied, never mutated in place.
    """
    query = record.get("QUERY", "") or ""
    chunk_text = _primary_chunk_text(record)
    if not query or not chunk_text:
        return record, {"replaced": [], "with": []}

    query_symptoms = find_symptoms(query)
    if not query_symptoms:
        return record, {"replaced": [], "with": []}

    chunk_symptoms = {s.lower() for s in find_symptoms(chunk_text)}
    if not chunk_symptoms:
        return record, {"replaced": [], "with": []}

    # Candidates: symptoms evidenced by the chunk and not already in the
    # query, preferring specific terms over generic ones.
    query_lower = query.lower()
    candidates = [
        s
        for s in chunk_symptoms
        if s not in query_lower and s not in GENERIC_SYMPTOMS
    ]
    if not candidates:
        candidates = [s for s in chunk_symptoms if s not in query_lower]
    if not candidates:
        return record, {"replaced": [], "with": []}

    rng.shuffle(candidates)

    updated = dict(record)
    new_query = query
    replaced: list[str] = []
    used: list[str] = []
    for term in query_symptoms:
        if not candidates:
            break
        target = candidates.pop()
        pattern = _term_pattern(term)
        if not pattern.search(new_query):
            continue
        new_query = pattern.sub(target, new_query, count=1)
        replaced.append(term)
        used.append(target)

    updated["QUERY"] = new_query
    return updated, {"replaced": replaced, "with": used}


def split_and_mask(
    records: list[dict[str, Any]],
    *,
    split_at: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split records and mask symptoms in the tail.

    The first ``split_at`` records are returned untouched; the remainder get
    their symptoms replaced with chunk-grounded ones. Returns
    ``(head, tail, stats)``.
    """
    head = list(records[:split_at])
    tail_records = list(records[split_at:])
    tail: list[dict[str, Any]] = []
    replaced_total = 0
    for record in tail_records:
        updated, report = mask_symptoms(record, rng)
        if report["replaced"]:
            replaced_total += len(report["replaced"])
        tail.append(updated)
    stats = {
        "split_at": split_at,
        "head": len(head),
        "tail": len(tail),
        "symptoms_replaced": replaced_total,
    }
    return head, tail, stats
