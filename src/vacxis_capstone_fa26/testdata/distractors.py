"""Smarter distractor sampling.

We split distractors into two pools:

* hard negatives: same category, different ``doc_id`` (the retriever must
  distinguish vaccines within the same topic — the harder case)
* easy negatives: different category (clearly off-topic)

The 60 / 40 split can be overridden via the ``hard_ratio`` argument. The
final list is shuffled so the order inside ``RELEVANT_CHUNKS`` doesn't leak
which pool each entry came from.

Each distractor is tagged with its source pool (:class:`DistractorKind`) so
the generator can assign a stable ``RELEVANCE_SCORE`` based on the kind
rather than the post-shuffle index (review-1 finding 3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .chunks import Chunk


class DistractorKind(Enum):
    """Origin of a distractor chunk."""

    HARD = "hard"   # same category, different doc — score 1
    EASY = "easy"   # different category — score 0


@dataclass
class TaggedDistractor:
    chunk: Chunk
    kind: DistractorKind


def sample_distractors(
    chunk: Chunk,
    all_chunks: list[Chunk],
    by_category: dict[str, list[Chunk]],
    rng: random.Random,
    n: int,
    hard_ratio: float = 0.6,
) -> list[TaggedDistractor]:
    """Return ``n`` tagged distractor chunks for ``chunk``."""
    if n <= 0:
        return []

    n_hard = max(1, round(n * hard_ratio)) if n >= 2 else n
    n_easy = n - n_hard

    hard_pool = [
        c for c in by_category.get(chunk.category, [])
        if c.doc_id != chunk.doc_id and c.chunk_id != chunk.chunk_id
    ]
    easy_pool = [
        c for c in all_chunks
        if c.category != chunk.category and c.doc_id != chunk.doc_id
    ]

    rng.shuffle(hard_pool)
    rng.shuffle(easy_pool)

    selected: list[TaggedDistractor] = (
        [TaggedDistractor(c, DistractorKind.HARD) for c in hard_pool[:n_hard]]
        + [TaggedDistractor(c, DistractorKind.EASY) for c in easy_pool[:n_easy]]
    )
    # In the rare case one pool is empty, top up from the other.
    if len(selected) < n:
        backup_kind = (
            DistractorKind.EASY if len(hard_pool) < n_hard else DistractorKind.HARD
        )
        backup = easy_pool if len(hard_pool) < n_hard else hard_pool
        for c in backup:
            if len(selected) >= n:
                break
            if not any(t.chunk.chunk_id == c.chunk_id for t in selected):
                selected.append(TaggedDistractor(c, backup_kind))

    rng.shuffle(selected)
    return selected[:n]


def score_for(kind: DistractorKind) -> int:
    """Stable relevance score per distractor kind.

    Review-1 finding 3: the original code used ``max(0, 2 - idx)`` which
    assigned scores by post-shuffle position, so a hard negative at index 2
    could end up with score 0 while an easy negative at index 0 got score 2.
    """
    return 1 if kind is DistractorKind.HARD else 0
