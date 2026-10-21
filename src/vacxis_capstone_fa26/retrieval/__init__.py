"""Retrieval package for VacxiS Capstone.

Provides Hybrid Search (Dense + Sparse), Reciprocal Rank Fusion (RRF),
and Cross-Encoder Reranking using BGE-M3 and BGE-reranker-v2-m3.
"""
from __future__ import annotations

from vacxis_capstone_fa26.retrieval.hybrid_searcher import (
    HybridSearcher,
    convert_sparse_to_qdrant,
)
from vacxis_capstone_fa26.retrieval.reranker import BGEReranker
from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
from vacxis_capstone_fa26.retrieval.schemas import (
    RetrievalConfig,
    RetrievedChunk,
)

__all__ = [
    "BGEReranker",
    "HybridSearcher",
    "RetrievalConfig",
    "RetrievedChunk",
    "VacxisRetriever",
    "convert_sparse_to_qdrant",
]
