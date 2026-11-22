"""Jev-backed DeepEval suite for the VacxiS RAG pipeline.

Scoring is delegated to DeepEval's RAG metric family (generator, retriever, and
Jev metrics and a deterministic refusal-correctness check.
"""
from __future__ import annotations

from vacxis_capstone_fa26.evaluation.datasets import (
    GoldenRecord,
    answer_fields,
    load_chunk_texts,
    load_goldens,
    retrieval_context_from_answer,
)
from vacxis_capstone_fa26.evaluation.evaluator import (
    VacxisRAGEvaluator,
    align_results,
    validate_query_ids,
)
from vacxis_capstone_fa26.evaluation.ir_metrics import (
    DEFAULT_K_VALUES,
    aggregate_ir_metrics,
    compute_query_ir_metrics,
    hit_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from vacxis_capstone_fa26.evaluation.jev import require_jev_credentials, resolve_jev_model
from vacxis_capstone_fa26.evaluation.schemas import (
    EvaluationSummary,
    MetricAggregate,
    QueryEvalResult,
)

__all__ = [
    "VacxisRAGEvaluator",
    "align_results",
    "validate_query_ids",
    "resolve_jev_model",
    "require_jev_credentials",
    "GoldenRecord",
    "load_goldens",
    "load_chunk_texts",
    "retrieval_context_from_answer",
    "answer_fields",
    "EvaluationSummary",
    "MetricAggregate",
    "QueryEvalResult",
    "DEFAULT_K_VALUES",
    "hit_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "map_at_k",
    "compute_query_ir_metrics",
    "aggregate_ir_metrics",
]
