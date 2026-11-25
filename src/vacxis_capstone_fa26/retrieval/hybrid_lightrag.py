"""Hybrid retriever combining VacxisRetriever with LightRAG Knowledge Graph context."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import logging
import math
from typing import Any

from vacxis_capstone_fa26.retrieval.lightrag_retriever import LightRAGRetriever
from vacxis_capstone_fa26.retrieval.reranker import BGEReranker
from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk

logger = logging.getLogger("vacxis.retrieval.hybrid_lightrag")

VALID_LIGHTRAG_FUSIONS = ("context_append", "rrf_rerank")


def _fuse_hybrid_and_lightrag_candidates(
    hybrid_candidates: Sequence[RetrievedChunk],
    lightrag_chunks: Sequence[RetrievedChunk],
    *,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse Hybrid RRF candidates with LightRAG document chunks using Reciprocal Rank Fusion.

    When a LightRAG chunk matches either a hybrid ``chunk_id`` directly or an ID inside
    a hybrid chunk's ``metadata["merged_chunk_ids"]``, the hybrid chunk payload is kept
    (preserving table-aware dual text) and receives the combined RRF score.
    """
    rrf_scores: dict[str, float] = {}
    chosen_chunks: dict[str, RetrievedChunk] = {}
    merged_id_to_canonical: dict[str, str] = {}
    boosted_by_lightrag: set[str] = set()

    seen_hybrid: set[str] = set()
    for rank, chunk in enumerate(hybrid_candidates, start=1):
        cid = chunk.chunk_id
        if cid in seen_hybrid:
            continue
        seen_hybrid.add(cid)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chosen_chunks[cid] = chunk
        merged_id_to_canonical[cid] = cid
        for merged_id in (chunk.metadata or {}).get("merged_chunk_ids") or []:
            merged_id_to_canonical[str(merged_id)] = cid

    seen_lightrag: set[str] = set()
    for rank, chunk in enumerate(lightrag_chunks, start=1):
        canonical_id = merged_id_to_canonical.get(chunk.chunk_id, chunk.chunk_id)
        if canonical_id in seen_lightrag:
            continue
        seen_lightrag.add(canonical_id)
        rrf_scores[canonical_id] = rrf_scores.get(canonical_id, 0.0) + 1.0 / (rrf_k + rank)
        if canonical_id in chosen_chunks:
            boosted_by_lightrag.add(canonical_id)
        else:
            chosen_chunks[canonical_id] = chunk

    ordered_ids = sorted(
        chosen_chunks.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )

    fused: list[RetrievedChunk] = []
    for cid in ordered_ids:
        chunk = chosen_chunks[cid]
        score = rrf_scores[cid]
        meta = dict(chunk.metadata or {})
        if cid in boosted_by_lightrag:
            meta["lightrag_fused"] = True
        fused.append(
            replace(
                chunk,
                score=score,
                rrf_score=score,
                rerank_score=None,
                metadata=meta,
            )
        )
    return fused


def _safe_sigmoid(value: float) -> float:
    """Compute a numerically stable sigmoid mapping any finite logit into (0, 1)."""
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def _blend_rerank_and_rrf(
    ranked: Sequence[RetrievedChunk],
    *,
    rerank_weight: float,
    top_k: int,
) -> list[RetrievedChunk]:
    """Blend cross-encoder rerank scores with normalized RRF consensus scores.

    Preserves BGEReranker's [0, 1] sigmoid-normalized scores without double-sigmoiding,
    falling back to a numerically stable sigmoid only when raw logits outside [0, 1]
    are encountered.
    """
    if not ranked or top_k <= 0:
        return []
    if rerank_weight >= 1.0:
        return list(ranked[:top_k])

    raw_reranks = [
        float(c.rerank_score if c.rerank_score is not None else c.score)
        for c in ranked
    ]
    use_sigmoid = any(score < 0.0 or score > 1.0 for score in raw_reranks)
    max_rrf = max((max(0.0, float(c.rrf_score or 0.0)) for c in ranked), default=0.0)
    rrf_weight = 1.0 - rerank_weight

    blended: list[RetrievedChunk] = []
    for chunk, raw_rerank in zip(ranked, raw_reranks, strict=True):
        norm_rerank = _safe_sigmoid(raw_rerank) if use_sigmoid else raw_rerank
        rrf_val = max(0.0, float(chunk.rrf_score or 0.0))
        norm_rrf = (rrf_val / max_rrf) if max_rrf > 0.0 else 0.0
        final_score = rerank_weight * norm_rerank + rrf_weight * norm_rrf
        meta = dict(chunk.metadata or {})
        meta["rerank_rrf_weight"] = rerank_weight
        blended.append(replace(chunk, score=final_score, metadata=meta))

    blended.sort(
        key=lambda c: (
            c.score,
            float(c.rerank_score if c.rerank_score is not None else c.score),
            float(c.rrf_score or 0.0),
        ),
        reverse=True,
    )
    return blended[:top_k]


class HybridLightRAGRetriever:
    """Combines VacxisRetriever (Hybrid + Table-Aware) and LightRAGRetriever with one BGE-M3."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        hybrid_retriever: VacxisRetriever | None = None,
        lightrag_retriever: LightRAGRetriever | None = None,
        embed_model: Any | None = None,
        reranker: BGEReranker | None = None,
        lazy_load: bool = True,
    ) -> None:
        self.config = config or RetrievalConfig()
        if self.config.lightrag_fusion not in VALID_LIGHTRAG_FUSIONS:
            raise ValueError(
                f"Unsupported lightrag_fusion: {self.config.lightrag_fusion!r}. "
                f"Expected one of {VALID_LIGHTRAG_FUSIONS}."
            )
        self._hybrid_retriever = hybrid_retriever
        self._lightrag_retriever = lightrag_retriever
        self._embed_model = embed_model
        self._reranker = reranker

        if not lazy_load:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        if self._embed_model is None:
            if (
                self._hybrid_retriever is not None
                and getattr(self._hybrid_retriever, "_embed_model", None) is not None
            ):
                self._embed_model = self._hybrid_retriever._embed_model
            elif (
                self._lightrag_retriever is not None
                and getattr(self._lightrag_retriever, "_embed_model", None) is not None
            ):
                self._embed_model = self._lightrag_retriever._embed_model
            elif self._hybrid_retriever is None or self._lightrag_retriever is None:
                import torch
                from FlagEmbedding import BGEM3FlagModel

                use_fp16 = torch.cuda.is_available()
                logger.info(
                    "Loading shared BGE-M3 embedding model from '%s' (fp16=%s)...",
                    self.config.embed_model_path,
                    use_fp16,
                )
                self._embed_model = BGEM3FlagModel(
                    str(self.config.embed_model_path),
                    use_fp16=use_fp16,
                )

        if self._hybrid_retriever is None:
            self._hybrid_retriever = VacxisRetriever(
                config=self.config,
                embed_model=self._embed_model,
                reranker=self._reranker,
                lazy_load=True,
            )

        if self._lightrag_retriever is None:
            self._lightrag_retriever = LightRAGRetriever(
                config=self.config,
                embed_model=self._embed_model,
                lazy_load=True,
            )

    @property
    def hybrid_retriever(self) -> VacxisRetriever:
        """Return or lazily create the Hybrid VacxisRetriever branch."""
        self._ensure_initialized()
        assert self._hybrid_retriever is not None
        return self._hybrid_retriever

    @property
    def lightrag_retriever(self) -> LightRAGRetriever:
        """Return or lazily create the LightRAG Knowledge Graph branch."""
        self._ensure_initialized()
        assert self._lightrag_retriever is not None
        return self._lightrag_retriever

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_k: int | None = None,
        skip_rerank: bool | None = None,
        query_max_length: int = 512,
        *,
        merge_table_chunks: bool | None = None,
        on_rerank: Callable[[], None] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve combined Hybrid/Table-Aware chunks and LightRAG Knowledge Graph context."""
        if not query or not query.strip():
            logger.warning("Empty or whitespace-only query provided to HybridLightRAGRetriever.")
            return []

        k = top_k if top_k is not None else self.config.top_k
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank

        if self.config.lightrag_fusion == "context_append":
            hybrid_chunks = self.hybrid_retriever.retrieve(
                query=query,
                top_k=k,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                rrf_top_k=k_rrf,
                skip_rerank=should_skip_rerank,
                query_max_length=query_max_length,
                merge_table_chunks=merge_table_chunks,
                on_rerank=on_rerank,
            )
            _, kg_chunk = self.lightrag_retriever.retrieve_components(
                query=query,
                include_chunks=False,
            )
            return [*hybrid_chunks, kg_chunk] if kg_chunk is not None else list(hybrid_chunks)

        # rrf_rerank fusion
        effective_rrf_top_k = max(k_rrf, k)
        hybrid_candidates = self.hybrid_retriever.retrieve(
            query=query,
            top_k=effective_rrf_top_k,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rrf_top_k=effective_rrf_top_k,
            skip_rerank=True,
            query_max_length=query_max_length,
            merge_table_chunks=merge_table_chunks,
        )
        lr_doc_chunks, kg_chunk = self.lightrag_retriever.retrieve_components(
            query=query,
            top_k=self.config.lightrag_chunk_top_k,
            include_chunks=self.config.lightrag_include_chunks,
        )
        fused = _fuse_hybrid_and_lightrag_candidates(
            hybrid_candidates,
            lr_doc_chunks,
            rrf_k=self.config.rrf_k,
        )

        alpha = self.config.lightrag_rerank_weight
        if not fused:
            ranked: list[RetrievedChunk] = []
        elif not should_skip_rerank:
            if on_rerank is not None:
                on_rerank()
            rerank_top_k = len(fused) if alpha < 1.0 else k
            ranked = self.hybrid_retriever.reranker.rerank(
                query=query,
                candidates=fused,
                top_k=rerank_top_k,
            )
            if alpha < 1.0:
                ranked = _blend_rerank_and_rrf(
                    ranked,
                    rerank_weight=alpha,
                    top_k=k,
                )
        else:
            ranked = fused[:k]

        return [*ranked, kg_chunk] if kg_chunk is not None else list(ranked)

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_k: int | None = None,
        skip_rerank: bool | None = None,
        batch_size: int = 16,
        query_max_length: int = 512,
        *,
        merge_table_chunks: bool | None = None,
        on_rerank: Callable[[], None] | None = None,
    ) -> list[list[RetrievedChunk]]:
        """Batch retrieve Hybrid/Table-Aware chunks and LightRAG KG context across queries."""
        if not queries:
            return []

        query_list = list(queries)
        if any(not isinstance(q, str) or not q.strip() for q in query_list):
            raise ValueError("Batch queries must be nonblank strings.")

        k = top_k if top_k is not None else self.config.top_k
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank

        if self.config.lightrag_fusion == "context_append":
            hybrid_groups = self.hybrid_retriever.batch_retrieve(
                queries=query_list,
                top_k=k,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                rrf_top_k=k_rrf,
                skip_rerank=should_skip_rerank,
                batch_size=batch_size,
                query_max_length=query_max_length,
                merge_table_chunks=merge_table_chunks,
                on_rerank=on_rerank,
            )
            results: list[list[RetrievedChunk]] = []
            for q, hybrid_chunks in zip(query_list, hybrid_groups, strict=True):
                _, kg_chunk = self.lightrag_retriever.retrieve_components(
                    query=q,
                    include_chunks=False,
                )
                if kg_chunk is not None:
                    results.append([*hybrid_chunks, kg_chunk])
                else:
                    results.append(list(hybrid_chunks))
            return results

        # rrf_rerank fusion in batch
        effective_rrf_top_k = max(k_rrf, k)
        hybrid_candidate_groups = self.hybrid_retriever.batch_retrieve(
            queries=query_list,
            top_k=effective_rrf_top_k,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rrf_top_k=effective_rrf_top_k,
            skip_rerank=True,
            batch_size=batch_size,
            query_max_length=query_max_length,
            merge_table_chunks=merge_table_chunks,
        )

        fused_groups: list[list[RetrievedChunk]] = []
        kg_chunks: list[RetrievedChunk | None] = []
        for q, hybrid_cands in zip(query_list, hybrid_candidate_groups, strict=True):
            lr_doc_chunks, kg_chunk = self.lightrag_retriever.retrieve_components(
                query=q,
                top_k=self.config.lightrag_chunk_top_k,
                include_chunks=self.config.lightrag_include_chunks,
            )
            fused_groups.append(
                _fuse_hybrid_and_lightrag_candidates(
                    hybrid_cands,
                    lr_doc_chunks,
                    rrf_k=self.config.rrf_k,
                )
            )
            kg_chunks.append(kg_chunk)

        alpha = self.config.lightrag_rerank_weight
        if not should_skip_rerank and any(fused_groups):
            if on_rerank is not None:
                on_rerank()
            rerank_top_k = (
                max(len(group) for group in fused_groups)
                if alpha < 1.0
                else k
            )
            ranked_groups = self.hybrid_retriever.reranker.batch_rerank(
                queries=query_list,
                candidates_list=fused_groups,
                top_k=rerank_top_k,
            )
            if alpha < 1.0:
                ranked_groups = [
                    _blend_rerank_and_rrf(group, rerank_weight=alpha, top_k=k)
                    for group in ranked_groups
                ]
        else:
            ranked_groups = [group[:k] for group in fused_groups]

        return [
            [*ranked, kg_chunk] if kg_chunk is not None else list(ranked)
            for ranked, kg_chunk in zip(ranked_groups, kg_chunks, strict=True)
        ]

    def close(self) -> None:
        """Release both hybrid and LightRAG resources safely inside try/finally."""
        try:
            if self._hybrid_retriever is not None:
                self._hybrid_retriever.close()
        finally:
            if self._lightrag_retriever is not None:
                self._lightrag_retriever.close()

    def __enter__(self) -> HybridLightRAGRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
