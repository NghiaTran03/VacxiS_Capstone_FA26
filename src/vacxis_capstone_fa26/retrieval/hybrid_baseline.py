"""Separate BM25 + dense RRF baseline, without learned sparse search or reranking."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .bm25 import BM25Retriever
from .schemas import RetrievalConfig


class HybridBaselineRetriever:
    """Fuse lexical and dense rankings by chunk ID while preserving corpus metadata."""

    def __init__(
        self,
        corpus: dict[str, dict[str, Any]],
        *,
        config: RetrievalConfig | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        dense_retriever: Any = None,
    ) -> None:
        if type(candidate_k) is not int or candidate_k < 1:
            raise ValueError("candidate_k must be a positive integer.")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer.")
        self.config = replace(config or RetrievalConfig(), skip_rerank=True, top_k_sparse=0)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self._corpus = corpus
        self._bm25 = BM25Retriever({cid: row["text"] for cid, row in corpus.items()})
        self._dense_retriever = dense_retriever
        self._owns_dense = dense_retriever is None

    @classmethod
    def from_parquet(cls, path: str | Path, **kwargs: Any) -> HybridBaselineRetriever:
        """Load the same corpus used for BM25, including generator context metadata."""
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        table = table.filter(pc.and_(pc.is_valid(table["chunk_id"]), pc.is_valid(table["text"])))
        corpus: dict[str, dict[str, Any]] = {}
        for row in table.to_pylist():
            cid, text = str(row["chunk_id"]).strip(), str(row["text"]).strip()
            if not cid or not text:
                continue
            if cid in corpus:
                raise ValueError(f"Duplicate corpus chunk ID: {cid}")
            corpus[cid] = {**row, "chunk_id": cid, "text": text}
        return cls(corpus, **kwargs)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Fuse up to candidate_k items per branch, then select the final top_k."""
        limit = self.config.top_k if top_k is None else top_k
        if type(limit) is not int or limit < 1:
            raise ValueError("top_k must be a positive integer.")
        if not query.strip():
            return []
        depth = max(self.candidate_k, limit)
        if self._dense_retriever is None:
            # Keep torch, FlagEmbedding and Qdrant out of offline imports.
            from .retriever import VacxisRetriever

            self._dense_retriever = VacxisRetriever(config=self.config)
        dense = self._dense_retriever.retrieve(
            query=query, top_k=depth, top_k_dense=depth,
            top_k_sparse=0, rrf_top_k=depth, skip_rerank=True,
        )
        lexical = self._bm25.retrieve(query, top_k=depth)
        scores: dict[str, float] = {}
        for ranking in (lexical, dense):
            seen: set[str] = set()
            for chunk in ranking:
                cid = chunk.get("chunk_id", "") if isinstance(chunk, dict) else chunk.chunk_id
                if cid not in self._corpus:
                    raise ValueError(f"Dense chunk {cid!r} is absent from --chunks; check corpus/collection alignment.")
                if cid in seen:
                    continue
                seen.add(cid)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + len(seen))
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
        return [
            {**self._corpus[cid], "chunk_id": cid, "score": scores[cid], "rrf_score": scores[cid]}
            for cid in ranked
        ]

    def close(self) -> None:
        """Close only a dense retriever created by this baseline."""
        if self._owns_dense and self._dense_retriever is not None:
            self._dense_retriever.close()
            self._dense_retriever = None

    def __enter__(self) -> HybridBaselineRetriever:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
