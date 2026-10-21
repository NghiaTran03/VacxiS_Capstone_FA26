"""Qdrant Hybrid Searcher (Dense + Sparse) with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import logging
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from vacxis_capstone_fa26.indexing.indexer import convert_sparse_to_qdrant
from vacxis_capstone_fa26.retrieval.schemas import RetrievedChunk

logger = logging.getLogger("vacxis.retrieval.hybrid_searcher")

DEFAULT_COLLECTION = "VacxiS_Capstone_FA26"
DEFAULT_RRF_K = 60


class HybridSearcher:
    """Performs hybrid vector search over Qdrant using Dense + Sparse with RRF."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = DEFAULT_COLLECTION,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        """Initialize hybrid searcher client.

        Args:
            client: Optional pre-configured QdrantClient.
            host: Qdrant host if client not provided.
            port: Qdrant port if client not provided.
            collection_name: Default Qdrant collection name.
            rrf_k: Smoothing constant for RRF formula (default 60).
        """
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            self.client = QdrantClient(host=host, port=port)
            self._owns_client = True

    def search(
        self,
        dense_vec: list[float] | None = None,
        sparse_weights: dict[str, float] | dict[int, float] | None = None,
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        rrf_top_k: int = 30,
        collection_name: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[RetrievedChunk]:
        """Perform dense and sparse search and merge results using RRF.

        Args:
            dense_vec: Query dense vector (1024-dim).
            sparse_weights: Query lexical weights dict from BGE-M3.
            top_k_dense: Number of dense search results.
            top_k_sparse: Number of sparse search results.
            rrf_top_k: Number of candidates to retain after RRF fusion.
            collection_name: Optional override for collection name.
            dense_weight: Weight multiplier for dense ranks in RRF.
            sparse_weight: Weight multiplier for sparse ranks in RRF.

        Returns:
            List of RetrievedChunk sorted by RRF score descending.
        """
        results = self.batch_search(
            dense_vecs=[dense_vec],
            sparse_weights_list=[sparse_weights],
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rrf_top_k=rrf_top_k,
            collection_name=collection_name,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        return results[0] if results else []

    def batch_search(
        self,
        dense_vecs: Sequence[list[float] | None],
        sparse_weights_list: Sequence[dict[str, float] | dict[int, float] | None],
        top_k_dense: int = 50,
        top_k_sparse: int = 50,
        rrf_top_k: int = 30,
        collection_name: str | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[list[RetrievedChunk]]:
        """Perform hybrid search and RRF fusion across multiple queries in a single Qdrant batch.

        Args:
            dense_vecs: Sequence of dense vectors (each 1024-dim or None).
            sparse_weights_list: Sequence of lexical weights dicts from BGE-M3.
            top_k_dense: Number of dense candidates per query.
            top_k_sparse: Number of sparse candidates per query.
            rrf_top_k: Candidates to retain per query after RRF fusion.
            collection_name: Optional override for collection name.
            dense_weight: Weight multiplier for dense ranks in RRF.
            sparse_weight: Weight multiplier for sparse ranks in RRF.

        Returns:
            List of candidate chunk lists, one per input query.
        """
        num_queries = len(dense_vecs)
        if num_queries == 0:
            return []

        target_collection = collection_name or self.collection_name
        requests: list[models.QueryRequest] = []
        # Mapping from query_idx -> list of (weight, batch_request_index)
        query_streams: list[list[tuple[float, int]]] = [[] for _ in range(num_queries)]

        for q_idx in range(num_queries):
            dense_vec = dense_vecs[q_idx] if q_idx < len(dense_vecs) else None
            sparse_weights = sparse_weights_list[q_idx] if q_idx < len(sparse_weights_list) else None

            if dense_vec is not None and len(dense_vec) > 0 and top_k_dense > 0:
                batch_req_idx = len(requests)
                requests.append(
                    models.QueryRequest(
                        query=dense_vec,
                        using="dense",
                        limit=top_k_dense,
                        with_payload=True,
                    )
                )
                query_streams[q_idx].append((dense_weight, batch_req_idx))

            if sparse_weights and top_k_sparse > 0:
                sparse_vec = convert_sparse_to_qdrant(sparse_weights)
                if sparse_vec.indices:
                    batch_req_idx = len(requests)
                    requests.append(
                        models.QueryRequest(
                            query=sparse_vec,
                            using="sparse",
                            limit=top_k_sparse,
                            with_payload=True,
                        )
                    )
                    query_streams[q_idx].append((sparse_weight, batch_req_idx))

        if not requests:
            return [[] for _ in range(num_queries)]

        batch_results = self.client.query_batch_points(
            collection_name=target_collection,
            requests=requests,
        )

        all_candidates: list[list[RetrievedChunk]] = []
        for q_idx in range(num_queries):
            streams = query_streams[q_idx]
            if not streams:
                all_candidates.append([])
                continue

            rrf_scores: dict[int | str, dict[str, Any]] = {}
            for weight, req_idx in streams:
                result = batch_results[req_idx]
                for rank, point in enumerate(result.points, start=1):
                    pid = point.id
                    score = weight * (1.0 / (self.rrf_k + rank))
                    if pid not in rrf_scores:
                        rrf_scores[pid] = {
                            "score": 0.0,
                            "payload": point.payload or {},
                        }
                    rrf_scores[pid]["score"] += score

            sorted_results = sorted(
                rrf_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True,
            )[:rrf_top_k]

            candidates: list[RetrievedChunk] = []
            for pid, data in sorted_results:
                payload = data["payload"]
                raw_meta = payload.get("metadata")
                meta_dict = dict(raw_meta) if isinstance(raw_meta, dict) else {}
                chunk = RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    doc_id=str(payload.get("doc_id", "")),
                    doc_title=str(payload.get("doc_title") or payload.get("title", "")),
                    category=str(payload.get("category", "")),
                    parent_header=str(payload.get("parent_header", "")),
                    header=str(payload.get("header", "")),
                    content_type=str(payload.get("content_type", "")),
                    text=str(payload.get("text", "")),
                    score=float(data["score"]),
                    rrf_score=float(data["score"]),
                    point_id=pid,
                    metadata=meta_dict,
                )
                candidates.append(chunk)

            all_candidates.append(candidates)

        return all_candidates

    def close(self) -> None:
        """Close Qdrant client connection if owned."""
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self) -> HybridSearcher:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
