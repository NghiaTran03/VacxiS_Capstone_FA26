"""DeepEval-based evaluation suite for the VacxiS RAG pipeline.

Scoring is delegated to DeepEval's RAG metric family (generator, retriever, and
a GEval correctness judge) plus a deterministic refusal-correctness check.
The judge LLM is configurable; by default it runs through the project's
OpenRouter endpoint so no OpenAI key is required.
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
from vacxis_capstone_fa26.evaluation.judge import (
    OpenRouterJudgeLLM,
    require_judge_credentials,
    resolve_judge_model,
)
from vacxis_capstone_fa26.evaluation.schemas import (
    EvaluationSummary,
    MetricAggregate,
    QueryEvalResult,
)

__all__ = [
    "VacxisRAGEvaluator",
    "align_results",
    "validate_query_ids",
    "OpenRouterJudgeLLM",
    "resolve_judge_model",
    "require_judge_credentials",
    "GoldenRecord",
    "load_goldens",
    "load_chunk_texts",
    "retrieval_context_from_answer",
    "answer_fields",
    "EvaluationSummary",
    "MetricAggregate",
    "QueryEvalResult",
]
