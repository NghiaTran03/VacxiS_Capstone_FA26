"""Schema definitions for retrieval pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Literal

RetrieverMode = Literal["naive", "advanced", "hybrid", "lightrag", "hybrid_lightrag"]
LightRAGQueryMode = Literal["local", "global", "hybrid", "mix", "naive"]
LightRAGFusionMode = Literal["context_append", "rrf_rerank"]

# Standalone LightRAG defaults aligned with LightRAG/lightrag/constants.py & QueryParam
LIGHTRAG_DEFAULT_MODE: LightRAGQueryMode = "mix"
LIGHTRAG_DEFAULT_TOP_K = 40
LIGHTRAG_DEFAULT_CHUNK_TOP_K = 20
LIGHTRAG_DEFAULT_MAX_ENTITY_TOKENS = 6000
LIGHTRAG_DEFAULT_MAX_RELATION_TOKENS = 8000
LIGHTRAG_DEFAULT_MAX_TOTAL_TOKENS = 30000

# Compact LightRAG defaults for hybrid_lightrag alongside table-aware dual chunks
HYBRID_LIGHTRAG_DEFAULT_MODE: LightRAGQueryMode = "hybrid"
HYBRID_LIGHTRAG_DEFAULT_TOP_K = 20
HYBRID_LIGHTRAG_DEFAULT_CHUNK_TOP_K = 5
HYBRID_LIGHTRAG_DEFAULT_MAX_ENTITY_TOKENS = 1500
HYBRID_LIGHTRAG_DEFAULT_MAX_RELATION_TOKENS = 1500
HYBRID_LIGHTRAG_DEFAULT_MAX_TOTAL_TOKENS = 30000


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

    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "VacxiS_Capstone_FA26")
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
    mode: RetrieverMode = "hybrid"
    merge_table_chunks: bool = False

    # LightRAG Knowledge Graph settings (None resolves per mode in __post_init__)
    lightrag_working_dir: str | Path = "lightrag_storage"
    lightrag_workspace: str = "vacxis"
    lightrag_mode: LightRAGQueryMode | None = None
    lightrag_top_k: int | None = None
    lightrag_chunk_top_k: int | None = None
    lightrag_max_entity_tokens: int | None = None
    lightrag_max_relation_tokens: int | None = None
    lightrag_max_total_tokens: int | None = None
    lightrag_include_chunks: bool = True
    lightrag_fusion: LightRAGFusionMode = "context_append"
    chunks_parquet_path: str | Path = "chunks_storage/chunks.parquet"

    def __post_init__(self) -> None:
        is_standalone = self.mode == "lightrag"
        if self.lightrag_mode is None:
            self.lightrag_mode = (
                LIGHTRAG_DEFAULT_MODE if is_standalone else HYBRID_LIGHTRAG_DEFAULT_MODE
            )
        if self.lightrag_top_k is None:
            self.lightrag_top_k = (
                LIGHTRAG_DEFAULT_TOP_K if is_standalone else HYBRID_LIGHTRAG_DEFAULT_TOP_K
            )
        if self.lightrag_chunk_top_k is None:
            self.lightrag_chunk_top_k = (
                LIGHTRAG_DEFAULT_CHUNK_TOP_K
                if is_standalone
                else HYBRID_LIGHTRAG_DEFAULT_CHUNK_TOP_K
            )
        if self.lightrag_max_entity_tokens is None:
            self.lightrag_max_entity_tokens = (
                LIGHTRAG_DEFAULT_MAX_ENTITY_TOKENS
                if is_standalone
                else HYBRID_LIGHTRAG_DEFAULT_MAX_ENTITY_TOKENS
            )
        if self.lightrag_max_relation_tokens is None:
            self.lightrag_max_relation_tokens = (
                LIGHTRAG_DEFAULT_MAX_RELATION_TOKENS
                if is_standalone
                else HYBRID_LIGHTRAG_DEFAULT_MAX_RELATION_TOKENS
            )
        if self.lightrag_max_total_tokens is None:
            self.lightrag_max_total_tokens = (
                LIGHTRAG_DEFAULT_MAX_TOTAL_TOKENS
                if is_standalone
                else HYBRID_LIGHTRAG_DEFAULT_MAX_TOTAL_TOKENS
            )
        for field_name in (
            "lightrag_top_k",
            "lightrag_chunk_top_k",
            "lightrag_max_entity_tokens",
            "lightrag_max_relation_tokens",
            "lightrag_max_total_tokens",
        ):
            value = getattr(self, field_name)
            if value < 1:
                raise ValueError(f"{field_name} must be positive.")
