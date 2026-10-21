"""End-to-end Retrieval Pipeline for VacxiS Capstone."""
from __future__ import annotations

import logging
from typing import Sequence

import torch
from FlagEmbedding import BGEM3FlagModel

from vacxis_capstone_fa26.retrieval.hybrid_searcher import HybridSearcher
from vacxis_capstone_fa26.retrieval.reranker import BGEReranker
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk

logger = logging.getLogger("vacxis.retrieval.retriever")


class VacxisRetriever:
    """End-to-end retriever orchestrating BGE-M3 encoding, Hybrid Qdrant search, RRF, and Reranking."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        embed_model: BGEM3FlagModel | None = None,
        reranker: BGEReranker | None = None,
        hybrid_searcher: HybridSearcher | None = None,
        lazy_load: bool = False,
    ) -> None:
        """Initialize retrieval pipeline components.

        Args:
            config: Pipeline configuration settings.
            embed_model: Optional pre-loaded BGE-M3 model.
            reranker: Optional pre-loaded BGE-reranker.
            hybrid_searcher: Optional pre-configured HybridSearcher.
            lazy_load: If True, models are loaded upon first call to retrieve.
        """
        self.config = config or RetrievalConfig()
        self._embed_model = embed_model
        self._reranker = reranker
        self._searcher = hybrid_searcher

        if not lazy_load:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure all required components and models are loaded."""
        if self._embed_model is None:
            use_fp16 = torch.cuda.is_available()
            logger.info(
                "Loading BGE-M3 embedding model from '%s' (fp16=%s)...",
                self.config.embed_model_path,
                use_fp16,
            )
            self._embed_model = BGEM3FlagModel(
                str(self.config.embed_model_path),
                use_fp16=use_fp16,
            )
            logger.info("BGE-M3 model loaded successfully.")

        if self._searcher is None:
            self._searcher = HybridSearcher(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                collection_name=self.config.collection_name,
                rrf_k=self.config.rrf_k,
            )

        if self._reranker is None and not self.config.skip_rerank:
            self._reranker = BGEReranker(
                model_path=self.config.reranker_model_path,
            )

    @property
    def embed_model(self) -> BGEM3FlagModel:
        """Get or initialize embedding model."""
        self._ensure_initialized()
        assert self._embed_model is not None
        return self._embed_model

    @property
    def searcher(self) -> HybridSearcher:
        """Get or initialize hybrid searcher."""
        self._ensure_initialized()
        assert self._searcher is not None
        return self._searcher

    @property
    def reranker(self) -> BGEReranker:
        """Get or initialize reranker."""
        self._ensure_initialized()
        if self._reranker is None:
            self._reranker = BGEReranker(model_path=self.config.reranker_model_path)
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_top_k: int | None = None,
        skip_rerank: bool | None = None,
        query_max_length: int = 512,
    ) -> list[RetrievedChunk]:
        """Execute hybrid search, RRF fusion, and reranking for a single query.

        Args:
            query: User question or search text.
            top_k: Number of final results to return.
            top_k_dense: Dense search candidate count.
            top_k_sparse: Sparse search candidate count.
            rrf_top_k: RRF candidate count.
            skip_rerank: If True, bypass cross-encoder reranking.
            query_max_length: Token length truncation for query encoding.

        Returns:
            List of top-k RetrievedChunk items.
        """
        k = top_k if top_k is not None else self.config.top_k
        k_dense = top_k_dense if top_k_dense is not None else self.config.top_k_dense
        k_sparse = top_k_sparse if top_k_sparse is not None else self.config.top_k_sparse
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank

        # 1. Encode query with BGE-M3
        embed_output = self.embed_model.encode(
            [query],
            batch_size=1,
            max_length=query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vec = embed_output["dense_vecs"][0].tolist()
        sparse_weights = embed_output["lexical_weights"][0]

        # 2. Hybrid search on Qdrant + RRF
        candidates = self.searcher.search(
            dense_vec=dense_vec,
            sparse_weights=sparse_weights,
            top_k_dense=k_dense,
            top_k_sparse=k_sparse,
            rrf_top_k=k_rrf,
        )

        if not candidates:
            return []

        # 3. Rerank
        if not should_skip_rerank:
            return self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=k,
            )

        # If skipping rerank, take top-k from RRF directly
        return candidates[:k]

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
    ) -> list[list[RetrievedChunk]]:
        """Execute hybrid search, RRF, and reranking across multiple queries.

        Args:
            queries: Sequence of query strings.
            top_k: Number of final results to return per query.
            top_k_dense: Dense search candidate count.
            top_k_sparse: Sparse search candidate count.
            rrf_top_k: RRF candidate count.
            skip_rerank: If True, bypass cross-encoder reranking.
            batch_size: Batch size for BGE-M3 query encoding.
            query_max_length: Token length truncation for query encoding.

        Returns:
            List of top-k RetrievedChunk lists corresponding to input queries.
        """
        if not queries:
            return []

        query_list = list(queries)
        embed_output = self.embed_model.encode(
            query_list,
            batch_size=batch_size,
            max_length=query_max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = embed_output["dense_vecs"]
        sparse_weights_list = embed_output["lexical_weights"]

        results: list[list[RetrievedChunk]] = []
        k = top_k if top_k is not None else self.config.top_k
        k_dense = top_k_dense if top_k_dense is not None else self.config.top_k_dense
        k_sparse = top_k_sparse if top_k_sparse is not None else self.config.top_k_sparse
        k_rrf = rrf_top_k if rrf_top_k is not None else self.config.rrf_top_k
        should_skip_rerank = skip_rerank if skip_rerank is not None else self.config.skip_rerank

        for i, q_text in enumerate(query_list):
            candidates = self.searcher.search(
                dense_vec=dense_vecs[i].tolist(),
                sparse_weights=sparse_weights_list[i],
                top_k_dense=k_dense,
                top_k_sparse=k_sparse,
                rrf_top_k=k_rrf,
            )
            if not candidates:
                results.append([])
                continue

            if not should_skip_rerank:
                top_chunks = self.reranker.rerank(
                    query=q_text,
                    candidates=candidates,
                    top_k=k,
                )
            else:
                top_chunks = candidates[:k]

            results.append(top_chunks)

        return results

    def close(self) -> None:
        """Release resources and connections."""
        if self._searcher is not None:
            self._searcher.close()

    def __enter__(self) -> VacxisRetriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
