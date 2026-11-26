"""End-to-End VacxiS RAG Pipeline coupling Retrieval with Medical Generation."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from itertools import zip_longest
import json
import logging
from time import perf_counter
from types import TracebackType
from typing import Any, Sequence

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer, GenerationConfig
from vacxis_capstone_fa26.query import (
    Clarification, ClarifyPlan, ConversationMemoryConfig, ConversationMessage, QueryPlanner,
)
from vacxis_capstone_fa26.query.memory import limit_history

logger = logging.getLogger(__name__)


def _log_query_plan(
    plan: Any,
    *,
    original_query: str,
    planner_model: Any,
    planner_latency: float,
    history_included: int,
    history_supplied: int,
    clarifications_count: int,
) -> None:
    """Emit a structured debug log for the query planner decision."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    payload: dict[str, Any] = {
        "action": getattr(plan, "action", None),
        "model": planner_model if isinstance(planner_model, str) else None,
        "latency_s": round(planner_latency, 3),
        "history_included": history_included,
        "history_supplied": history_supplied,
        "clarifications_count": clarifications_count,
        "original_query": original_query,
    }
    if isinstance(plan, ClarifyPlan):
        payload["clarification_question"] = plan.clarification_question
    else:
        payload["resolved_query"] = plan.resolved_query
        payload["search_queries"] = list(plan.search_queries)
    logger.debug(
        "QueryPlanner output:\n%s",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _round_score(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 4)
    return None


def _log_retrieved_chunks(chunks: Sequence[Any]) -> None:
    """Emit a structured debug log of the final chunks passed to generation."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    items: list[dict[str, Any]] = []
    for rank, chunk in enumerate(chunks, start=1):
        raw_text = get_chunk_field(chunk, "text", "")
        collapsed = " ".join(str(raw_text).split()) if raw_text else ""
        preview = f"{collapsed[:160]}…" if len(collapsed) > 160 else collapsed
        items.append({
            "rank": rank,
            "chunk_id": get_chunk_field(chunk, "chunk_id", ""),
            "doc_id": get_chunk_field(chunk, "doc_id", ""),
            "doc_title": get_chunk_field(chunk, "doc_title", ""),
            "header": get_chunk_field(chunk, "header", ""),
            "rerank_score": _round_score(get_chunk_field(chunk, "rerank_score", None)),
            "rrf_score": _round_score(get_chunk_field(chunk, "rrf_score", None)),
            "score": _round_score(get_chunk_field(chunk, "score", None)),
            "text_preview": preview,
        })
    logger.debug(
        "Retrieved chunks after rerank (%d):\n%s",
        len(items),
        json.dumps(items, ensure_ascii=False, indent=2),
    )


class VacxisRAGPipeline:
    """End-to-end RAG orchestrator for Vietnamese Vaccine Information."""

    def __init__(
        self,
        retriever: Any | None = None,
        generator: MedicalGenerator | None = None,
        retrieval_config: Any | None = None,
        generation_config: GenerationConfig | None = None,
        *,
        query_planner: QueryPlanner | None = None,
    ) -> None:
        if retriever is None:
            # Lazy import so generation does not require torch unless retriever is initialized
            from vacxis_capstone_fa26.retrieval.factory import build_retriever
            self.retriever = build_retriever(retrieval_config)
        else:
            self.retriever = retriever

        self.generator = generator or MedicalGenerator(config=generation_config)
        self.query_planner = query_planner

    def __enter__(self) -> VacxisRAGPipeline:
        if hasattr(self.retriever, "__enter__"):
            self.retriever.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if hasattr(self.retriever, "__exit__"):
                self.retriever.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if hasattr(self.generator, "close"):
                self.generator.close()

    def run_turn(
        self,
        original_query: str,
        *,
        clarifications: Sequence[Clarification] = (),
        history: Sequence[ConversationMessage] = (),
        use_memory: bool = True,
        top_k: int | None = None,
        skip_rerank: bool | None = None,
        on_progress: Callable[[str, int | None], None] | None = None,
    ) -> GeneratedAnswer | ClarifyPlan:
        """Plan and execute one turn, or return a question for the caller to ask."""
        if self.query_planner is None:
            raise ValueError("run_turn requires a query planner.")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer.")
        if type(use_memory) is not bool:
            raise TypeError("use_memory must be a boolean.")
        memory_config = getattr(self.query_planner, "memory_config", None)
        if not isinstance(memory_config, ConversationMemoryConfig):
            memory_config = ConversationMemoryConfig()
        selected = limit_history(history, memory_config) if use_memory else ()

        if on_progress is not None:
            on_progress("planning", None)
        started = perf_counter()
        plan = self.query_planner.plan(original_query, clarifications, history=selected)
        planner_latency = perf_counter() - started
        _log_query_plan(
            plan,
            original_query=original_query,
            planner_model=getattr(self.query_planner, "model", None),
            planner_latency=planner_latency,
            history_included=len(selected),
            history_supplied=len(history),
            clarifications_count=len(clarifications),
        )
        if isinstance(plan, ClarifyPlan):
            return plan

        retrieval_options = {}
        if on_progress is not None:
            on_progress("searching", None)
            retrieval_options["on_rerank"] = lambda: on_progress("reranking", None)

        if len(plan.search_queries) == 1:
            chunks = self.retriever.retrieve(
                query=plan.resolved_query, top_k=top_k, skip_rerank=skip_rerank, **retrieval_options,
            )
        else:
            limit = top_k if top_k is not None else self.retriever.config.top_k
            groups = self.retriever.batch_retrieve(
                queries=plan.search_queries, top_k=limit, skip_rerank=skip_rerank, **retrieval_options,
            )
            if len(groups) != len(plan.search_queries):
                raise ValueError("Expected one retrieval group per search query.")
            chunks = self._merge_query_results(groups, limit)

        _log_retrieved_chunks(chunks)
        if on_progress is not None:
            on_progress("generating", len(chunks))
        answer = self.generator.generate(query=plan.resolved_query, chunks=chunks)
        return replace(answer, query=original_query, query_context={
            "resolved_query": plan.resolved_query,
            "search_queries": [plan.resolved_query] if len(plan.search_queries) == 1 else list(plan.search_queries),
            "use_memory": use_memory,
            "history_messages_supplied": len(history),
            "history_messages_included": len(selected),
            "history_reduced": use_memory and len(selected) < len(history),
            "memory_limits": asdict(memory_config),
            "planner_model": self.query_planner.model,
            "planner_latency_seconds": planner_latency,
        })

    @staticmethod
    def _merge_query_results(groups: Sequence[Sequence[Any]], limit: int) -> list[Any]:
        chunks: dict[str, Any] = {}
        for rank in zip_longest(*groups):
            for chunk in rank:
                if chunk is not None:
                    chunks.setdefault(str(get_chunk_field(chunk, "chunk_id", "")), chunk)
                    if len(chunks) >= limit:
                        return list(chunks.values())
        return list(chunks.values())

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        override_chunks: Sequence[Any] | None = None,
        skip_rerank: bool | None = None,
    ) -> GeneratedAnswer:
        """Answer a single user query.

        Args:
            query: Medical question text in Vietnamese.
            top_k: Optional number of context chunks to retrieve.
            override_chunks: Optional pre-retrieved chunks (bypasses retrieval).
            skip_rerank: If True, bypass cross-encoder reranking during retrieval.

        Returns:
            GeneratedAnswer with synthesized text, verified citations, and metadata.
        """
        if override_chunks is not None:
            chunks = list(override_chunks)
        else:
            chunks = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                skip_rerank=skip_rerank,
            )

        return self.generator.generate(query=query, chunks=chunks)

    def answer_batch(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        override_chunks_per_query: Sequence[Sequence[Any]] | None = None,
        skip_rerank: bool | None = None,
    ) -> list[GeneratedAnswer]:
        """Answer a batch of queries efficiently using batched retrieval and generation."""
        if override_chunks_per_query is not None:
            chunks_batch = list(override_chunks_per_query)
        else:
            chunks_batch = self.retriever.batch_retrieve(
                queries=queries,
                top_k=top_k,
                skip_rerank=skip_rerank,
            )

        if len(chunks_batch) != len(queries):
            raise ValueError("Expected one chunk group per query.")
        answers = self.generator.generate_batch(queries=queries, chunks_per_query=chunks_batch)
        if len(answers) != len(queries):
            raise ValueError("Expected one generated answer per query.")
        return answers
