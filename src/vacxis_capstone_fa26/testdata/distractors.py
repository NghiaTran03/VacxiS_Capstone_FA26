"""Smarter distractor sampling.

We split distractors into two pools:

* hard negatives: same category, different ``doc_id`` (the retriever must
  distinguish vaccines within the same topic — the harder case)
* easy negatives: different category (clearly off-topic)

The 60 / 40 split can be overridden via the ``hard_ratio`` argument. The
final list is shuffled so the order inside ``RELEVANT_CHUNKS`` doesn't leak
which pool each entry came from.
"""
from __future__ import annotations

import random

from .chunks import Chunk


def sample_distractors(
    chunk: Chunk,
    all_chunks: list[Chunk],
    by_category: dict[str, list[Chunk]],
    rng: random.Random,
    n: int,
    hard_ratio: float = 0.6,
) -> list[Chunk]:
    """Return ``n`` distractor chunks for ``chunk``."""
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

    selected = hard_pool[:n_hard] + easy_pool[:n_easy]
    # In the rare case one pool is empty, top up from the other.
    if len(selected) < n:
        backup = easy_pool if len(hard_pool) < n_hard else hard_pool
        for c in backup:
            if len(selected) >= n:
                break
            if c not in selected:
                selected.append(c)

    rng.shuffle(selected)
    return selected[:n]
