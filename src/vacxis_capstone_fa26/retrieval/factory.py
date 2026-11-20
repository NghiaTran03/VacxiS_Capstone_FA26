"""Factory for instantiating configured VacxiS retrievers lazily."""
from __future__ import annotations

from typing import Any

from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig


def build_retriever(
    config: RetrievalConfig | None = None,
    *,
    chunk_texts: dict[str, str] | None = None,
    **kwargs: Any,
) -> Any:
    """Build a retriever instance according to ``config.mode``.

    Supported modes:
    - ``naive``: lexical BM25 over ``chunk_texts`` (or ``config.chunks_parquet_path``).
    - ``advanced``: single-query BGE-M3 dense search (``DenseRetriever``).
    - ``hybrid``: BGE-M3 dense+sparse hybrid search + RRF + BGE reranker (``VacxisRetriever``).
    - ``lightrag`` / ``hybrid_lightrag``: reserved for Wave 2 Knowledge Graph integration.
    """
    cfg = config or RetrievalConfig()
    mode = cfg.mode

    if mode == "naive":
        from vacxis_capstone_fa26.evaluation.datasets import load_chunk_texts
        from vacxis_capstone_fa26.retrieval import bm25 as bm25_mod

        texts = (
            chunk_texts
            if chunk_texts is not None
            else load_chunk_texts(cfg.chunks_parquet_path)
        )
        return bm25_mod.BM25Retriever(texts)

    if mode == "advanced":
        from vacxis_capstone_fa26.retrieval import dense as dense_mod

        return dense_mod.DenseRetriever(config=cfg, **kwargs)

    if mode == "hybrid":
        from vacxis_capstone_fa26.retrieval import retriever as retriever_mod

        return retriever_mod.VacxisRetriever(config=cfg, **kwargs)

    if mode in ("lightrag", "hybrid_lightrag"):
        raise NotImplementedError(
            f"Retriever mode {mode!r} is scheduled for Wave 2 (+graph lightrag)."
        )

    raise ValueError(
        f"Unsupported retriever mode: {mode!r}. "
        "Expected one of: 'naive', 'advanced', 'hybrid', 'lightrag', 'hybrid_lightrag'."
    )
