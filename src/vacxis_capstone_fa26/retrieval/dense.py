"""Single-query dense retrieval: embed the question and return vector-search top-k."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .batching import run_with_oom_backoff
from .schemas import RetrievalConfig


class DenseRetriever:
    """Search the existing Qdrant dense index without planning, fusion, or reranking."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        embed_model: Any = None,
        client: Any = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embed_model = embed_model
        self._client = client
        self._owns_client = client is None
        self.last_trace: dict[str, Any] = {}

    def retrieve(
        self, query: str, top_k: int | None = None, *, skip_rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Encode the original query once and preserve search order and similarity scores."""
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit < 1:
            raise ValueError("top_k must be a positive integer.")
        self.last_trace = {"search_queries": [], "top_k": limit, "retrieved_chunk_ids": []}
        if not query.strip():
            return []

        if self._embed_model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            self._embed_model = BGEM3FlagModel(
                str(self.config.embed_model_path), use_fp16=torch.cuda.is_available(),
            )
        encoded = self._embed_model.encode(
            [query], batch_size=1, max_length=512,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        vector = encoded["dense_vecs"][0].tolist()

        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                host=self.config.qdrant_host, port=self.config.qdrant_port,
            )
        self.last_trace["search_queries"] = [query]
        response = self._client.query_points(
            collection_name=self.config.collection_name,
            query=vector, using="dense", limit=limit, with_payload=True,
        )
        chunks = []
        for point in response.points[:limit]:
            payload = point.payload or {}
            chunks.append({
                **payload,
                "chunk_id": str(payload.get("chunk_id") or ""),
                "doc_title": payload.get("doc_title") or payload.get("title", ""),
                "score": float(point.score),
                "point_id": point.id,
            })
        self.last_trace["retrieved_chunk_ids"] = [row["chunk_id"] for row in chunks]
        return chunks

    def batch_retrieve(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
        *,
        skip_rerank: bool | None = None,
        batch_size: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Execute batched dense encoding and vector retrieval for a sequence of queries."""
        del skip_rerank
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit < 1:
            raise ValueError("top_k must be a positive integer.")
        if not queries:
            return []

        query_list = list(queries)
        non_empty_indices = [idx for idx, q in enumerate(query_list) if q.strip()]
        if not non_empty_indices:
            self.last_trace = {"search_queries": [], "top_k": limit, "retrieved_chunk_ids": []}
            return [[] for _ in query_list]

        if self._embed_model is None:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            self._embed_model = BGEM3FlagModel(
                str(self.config.embed_model_path), use_fp16=torch.cuda.is_available(),
            )

        effective_batch_size = (
            self.config.embed_batch_size if batch_size is None else batch_size
        )
        non_empty_queries = [query_list[idx] for idx in non_empty_indices]
        encoded = run_with_oom_backoff(
            lambda bs: self._embed_model.encode(
                non_empty_queries,
                batch_size=bs,
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            ),
            batch_size=effective_batch_size,
            operation="DenseRetriever.batch_retrieve",
        )
        dense_vecs = encoded["dense_vecs"]
        if len(dense_vecs) != len(non_empty_queries):
            raise ValueError("Expected one embedding per query.")

        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                host=self.config.qdrant_host, port=self.config.qdrant_port,
            )

        results: list[list[dict[str, Any]]] = [[] for _ in query_list]
        for idx, query_text, dense_vec in zip(
            non_empty_indices, non_empty_queries, dense_vecs, strict=True
        ):
            response = self._client.query_points(
                collection_name=self.config.collection_name,
                query=dense_vec.tolist(),
                using="dense",
                limit=limit,
                with_payload=True,
            )
            chunks = []
            for point in response.points[:limit]:
                payload = point.payload or {}
                chunks.append({
                    **payload,
                    "chunk_id": str(payload.get("chunk_id") or ""),
                    "doc_title": payload.get("doc_title") or payload.get("title", ""),
                    "score": float(point.score),
                    "point_id": point.id,
                })
            results[idx] = chunks
            self.last_trace = {
                "search_queries": [query_text],
                "top_k": limit,
                "retrieved_chunk_ids": [row["chunk_id"] for row in chunks],
            }

        return results

    def close(self) -> None:
        """Release only a client created by this retriever."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> DenseRetriever:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
