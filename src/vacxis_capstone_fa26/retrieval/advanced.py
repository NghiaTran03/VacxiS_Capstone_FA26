"""Evaluation-only multi-query BM25/dense fusion followed by original-query reranking."""
from __future__ import annotations

from typing import Any

from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query.schemas import ClarifyPlan, RetrievePlan

from .schemas import RetrievalConfig, RetrievedChunk


class AdvancedRAGRetriever:
    """Expand searches without changing the question sent to the answer generator."""

    def __init__(
        self,
        retriever: Any,
        planner: Any,
        *,
        config: RetrievalConfig | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        reranker: Any = None,
        owns_retriever: bool = False,
    ) -> None:
        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")
        self.config = config or RetrievalConfig()
        self.retriever = retriever
        self.planner = planner
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._reranker = reranker
        self._owns_retriever = owns_retriever
        self.last_trace: dict[str, Any] = {}

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Keep the original search, fuse expansion candidates, and rerank once."""
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit < 1:
            raise ValueError("top_k must be a positive integer.")
        if not query.strip():
            self.last_trace = {"planner_status": "empty_query", "search_queries": []}
            return []
        queries = [query]
        status = "original_only"
        try:
            plan = self.planner.plan(query)
            if isinstance(plan, ClarifyPlan):
                status = "clarification_fallback"
            elif isinstance(plan, RetrievePlan):
                seen = {query.strip()}
                for text in plan.search_queries:
                    if text.strip() and text.strip() not in seen:
                        queries.append(text.strip())
                        seen.add(text.strip())
                        if len(queries) == 3:
                            break
                if len(queries) > 1:
                    status = "expanded"
            else:
                raise ValueError("Unsupported planner result.")
        except LLMError:
            status = "planner_error_fallback"

        depth = max(self.candidate_k, limit)
        self.last_trace = {"planner_status": status, "search_queries": queries,
                           "candidate_depth": depth, "reranked": not self.config.skip_rerank}
        scores: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}
        for text in queries:
            ranking = self.retriever.retrieve(query=text, top_k=depth)
            seen_ids: set[str] = set()
            for row in ranking:
                cid = str(row["chunk_id"])
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                chunks.setdefault(cid, row)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + len(seen_ids))
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:depth]
        pool = [{**chunks[cid], "rrf_score": scores[cid], "score": scores[cid]} for cid in ordered]
        self.last_trace.update({"unique_candidate_count": len(scores), "rerank_pool_count": len(pool)})
        if self.config.skip_rerank or not pool:
            result = pool[:limit]
        else:
            if self._reranker is None:
                from .reranker import BGEReranker

                self._reranker = BGEReranker(model_path=self.config.reranker_model_path)
            candidates = [
                RetrievedChunk(
                    chunk_id=row["chunk_id"], text=row["text"],
                    doc_id=row.get("doc_id") or "", doc_title=row.get("doc_title") or "",
                    category=row.get("category") or "", header=row.get("header") or "",
                    parent_header=row.get("parent_header") or "",
                    content_type=row.get("content_type") or "",
                    score=row["score"], rrf_score=row["rrf_score"],
                    metadata=row.get("metadata") or {},
                )
                for row in pool
            ]
            ranked = self._reranker.rerank(query=query, candidates=candidates, top_k=limit)
            result = [row.to_dict() for row in ranked]
        self.last_trace["retrieved_chunk_ids"] = [row["chunk_id"] for row in result]
        return result

    def close(self) -> None:
        if self._owns_retriever:
            self.retriever.close()

    def __enter__(self) -> AdvancedRAGRetriever:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
