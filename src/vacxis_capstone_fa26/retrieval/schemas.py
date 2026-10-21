"""Schema definitions for retrieval pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    parent_header: str
    header: str
    content_type: str
    text: str
    score: float
    rrf_score: float
    rerank_score: float | None = None
    point_id: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for easy serialization."""
        return asdict(self)


@dataclass
class RetrievalConfig:
    """Configuration options for retrieval and reranking pipeline."""

    collection_name: str = "VacxiS_Capstone_FA26"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embed_model_path: str | Path = "models/bge-m3"
    reranker_model_path: str | Path = "models/bge-reranker"
    top_k_dense: int = 50
    top_k_sparse: int = 50
    rrf_k: int = 60
    rrf_top_k: int = 30
    top_k: int = 5
    skip_rerank: bool = False
