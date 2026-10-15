"""Deterministic evidence-span selection.

The original script picked a 140-char random slice inside the chunk. That
frequently missed the actual answer sentence entirely. This module scores
candidate sentences by intent-keyword matches and quantitative content
(so DOSAGE and STORAGE questions prefer sentences that contain numbers or
units), then returns the best one's character offsets into the chunk text.
"""
from __future__ import annotations

import random
import re

# A markdown table row starts with a pipe and a separator row is just pipes
# and dashes.
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^[\s|:-]+$")
_HEADER_LINE_RE = re.compile(r"^\[CHỦ ĐỀ:.*\] > \[MỤC:")

# Sentence splitter. The corpus uses `. `, `? `, `! ` and `.\n` a lot.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\?\!\n])\s+")


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Yield (start, end) spans of sentences in ``text``.

    Skips the synthetic ``[CHỦ ĐỀ: ...] > [MỤC: ...]`` header. Sentences
    shorter than 40 chars and markdown table rows are dropped.
    """
    body_start = text.find("\n")
    if body_start == -1 or body_start > len(text) - 40:
        body_start = 0
    else:
        body_start += 1

    spans: list[tuple[int, int]] = []
    pos = body_start
    for match in _SENTENCE_SPLIT_RE.finditer(text, pos):
        start, end = pos, match.start()
        if end - start >= 40:
            spans.append((start, end))
        pos = match.end()
    if pos < len(text) and len(text) - pos >= 40:
        spans.append((pos, len(text)))

    cleaned: list[tuple[int, int]] = []
    for start, end in spans:
        snippet = text[start:end]
        first_line = snippet.splitlines()[0] if snippet else ""
        if _HEADER_LINE_RE.match(first_line):
            continue
        if _TABLE_LINE_RE.match(first_line) or _TABLE_SEP_RE.match(first_line):
            continue
        cleaned.append((start, end))
    return cleaned


def _score(sentence: str, intent_keywords: list[str], full_text: str) -> int:
    score = 0
    lower = sentence.lower()
    for kw in intent_keywords:
        if kw.lower() in lower:
            score += 3
    if re.search(r"\d", sentence):
        # Quantitative answers are usually the most informative ones for
        # dosage/storage questions.
        score += 1
    if len(sentence) > 200:
        # Prefer non-trivial content over fragments.
        score += 1
    return score


def select_evidence_span(
    text: str,
    intent_keywords: list[str],
    rng: random.Random,
    min_span_len: int = 40,
) -> tuple[int, int]:
    """Pick a ``(start, end)`` substring of ``text`` that answers a query.

    Algorithm:
        1. Split into sentence candidates (skip the synthetic header and
           markdown table rows).
        2. Score each by intent-keyword matches and presence of digits.
        3. Pick the top-scoring sentence; on ties, prefer the longest.
        4. Fall back to the first non-empty sentence if nothing scores.
    """
    candidates = _split_sentences(text)
    if not candidates:
        return 0, min(len(text), min_span_len)

    scored = sorted(
        candidates,
        key=lambda se: (_score(text[se[0]:se[1]], intent_keywords, text), se[1] - se[0]),
        reverse=True,
    )
    start, end = scored[0]
    if end - start < min_span_len and end < len(text):
        end = min(len(text), start + min_span_len)
    return start, end
