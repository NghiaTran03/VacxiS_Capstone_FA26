"""Deterministic Information Retrieval (IR) evaluation metrics.

Provides standard rank-based IR metrics (Hit@K, Recall@K, Precision@K,
MRR@K, NDCG@K, MAP@K) for measuring retrieval quality against ground-truth
relevant chunk IDs without requiring an LLM judge.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from vacxis_capstone_fa26.evaluation.schemas import MetricAggregate

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


def _dedup_clean(items: Sequence[Any]) -> list[str]:
    """Deduplicate and normalize string identifiers while preserving order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        cid = str(item).strip()
        if cid and cid not in seen:
            seen.add(cid)
            cleaned.append(cid)
    return cleaned


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Return 1.0 if any relevant chunk is in retrieved[:k], else 0.0."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    top_k = _dedup_clean(retrieved)[:k]
    return 1.0 if any(cid in relevant_set for cid in top_k) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of ground-truth relevant chunks found in retrieved[:k]."""
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    if not relevant_set:
        return 0.0
    top_k = _dedup_clean(retrieved)[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of retrieved[:k] items that are ground-truth relevant."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    top_k = _dedup_clean(retrieved)[:k]
    hits = sum(1 for cid in top_k if cid in relevant_set)
    return hits / float(k)


def mrr_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first relevant chunk in top k."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    top_k = _dedup_clean(retrieved)[:k]
    for idx, cid in enumerate(top_k, start=1):
        if cid in relevant_set:
            return 1.0 / float(idx)
    return 0.0


def map_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Mean Average Precision at k."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    if not relevant_set:
        return 0.0
    top_k = _dedup_clean(retrieved)[:k]
    hits = 0
    sum_precisions = 0.0
    for idx, cid in enumerate(top_k, start=1):
        if cid in relevant_set:
            hits += 1
            sum_precisions += hits / float(idx)
    denominator = len(relevant_set)
    return sum_precisions / float(denominator) if denominator > 0 else 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k with binary relevance."""
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    relevant_set = {str(c).strip() for c in relevant if str(c).strip()}
    if not relevant_set:
        return 0.0
    top_k = _dedup_clean(retrieved)[:k]
    dcg = 0.0
    for idx, cid in enumerate(top_k, start=1):
        if cid in relevant_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_count = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_query_ir_metrics(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Compute all standard IR metrics for a single query across cutoff values."""
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"Hit@{k}"] = hit_at_k(retrieved, relevant, k)
        scores[f"Recall@{k}"] = recall_at_k(retrieved, relevant, k)
        scores[f"Precision@{k}"] = precision_at_k(retrieved, relevant, k)
        scores[f"MRR@{k}"] = mrr_at_k(retrieved, relevant, k)
        scores[f"NDCG@{k}"] = ndcg_at_k(retrieved, relevant, k)
        scores[f"MAP@{k}"] = map_at_k(retrieved, relevant, k)
    return scores


def aggregate_ir_metrics(
    per_query_metrics: Sequence[dict[str, float]],
    metric_names: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, MetricAggregate]:
    """Aggregate per-query IR metrics into MetricAggregate instances."""
    if not per_query_metrics:
        return {}
    if metric_names is None:
        metric_names = list(per_query_metrics[0].keys())

    aggregates: dict[str, MetricAggregate] = {}
    for name in metric_names:
        values: list[float] = []
        passed = 0
        for item in per_query_metrics:
            if name in item:
                val = float(item[name])
                values.append(val)
                if val >= threshold:
                    passed += 1
        count = len(values)
        if count == 0:
            continue
        mean_val = sum(values) / count
        aggregates[name] = MetricAggregate(
            mean=mean_val,
            min=min(values),
            max=max(values),
            count=count,
            pass_rate=passed / count,
        )
    return aggregates
