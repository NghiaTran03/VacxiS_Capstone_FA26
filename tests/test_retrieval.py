"""Tests for the retrieval package (Hybrid Search, RRF, Reranker)."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from qdrant_client import models

from vacxis_capstone_fa26.retrieval.hybrid_searcher import (
    HybridSearcher,
    convert_sparse_to_qdrant,
)
from vacxis_capstone_fa26.retrieval.reranker import BGEReranker
from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig, RetrievedChunk


def test_convert_sparse_to_qdrant():
    """Verify conversion from BGE-M3 lexical weights to Qdrant SparseVector."""
    # Empty case
    empty_vec = convert_sparse_to_qdrant(None)
    assert empty_vec.indices == []
    assert empty_vec.values == []

    # Non-empty case
    lexical_weights = {"101": 0.5, "2048": 1.25}
    sparse_vec = convert_sparse_to_qdrant(lexical_weights)
    assert sparse_vec.indices == [101, 2048]
    assert pytest.approx(sparse_vec.values) == [0.5, 1.25]


def test_retrieved_chunk_to_dict():
    """Verify RetrievedChunk serialization to dict."""
    chunk = RetrievedChunk(
        chunk_id="test_doc_c_0001",
        doc_id="test_doc",
        doc_title="Vaccine Test HDSD",
        category="chỉ định",
        parent_header="CHỈ ĐỊNH",
        header="Chỉ định tiêm phòng",
        content_type="text",
        text="Vắc-xin phòng ngừa viêm não Nhật Bản.",
        score=0.92,
        rrf_score=0.032,
        rerank_score=0.92,
        point_id=12,
        metadata={"has_table": False},
    )
    d = chunk.to_dict()
    assert d["chunk_id"] == "test_doc_c_0001"
    assert d["doc_title"] == "Vaccine Test HDSD"
    assert d["score"] == 0.92
    assert d["point_id"] == 12
    assert d["metadata"]["has_table"] is False


def test_hybrid_searcher_rrf_logic():
    """Verify RRF fusion with mocked Qdrant query_batch_points."""
    mock_client = MagicMock()

    # Create mock search results for dense stream (points 1, 2)
    dense_p1 = MagicMock()
    dense_p1.id = 1
    dense_p1.payload = {
        "chunk_id": "c1",
        "doc_id": "d1",
        "doc_title": "Doc 1",
        "category": "cat1",
        "parent_header": "ph1",
        "header": "h1",
        "content_type": "text",
        "text": "Text 1",
    }

    dense_p2 = MagicMock()
    dense_p2.id = 2
    dense_p2.payload = {
        "chunk_id": "c2",
        "doc_id": "d2",
        "doc_title": "Doc 2",
        "category": "cat2",
        "parent_header": "ph2",
        "header": "h2",
        "content_type": "text",
        "text": "Text 2",
    }

    # Create mock search results for sparse stream (points 2, 3)
    sparse_p2 = MagicMock()
    sparse_p2.id = 2
    sparse_p2.payload = dense_p2.payload

    sparse_p3 = MagicMock()
    sparse_p3.id = 3
    sparse_p3.payload = {
        "chunk_id": "c3",
        "doc_id": "d3",
        "doc_title": "Doc 3",
        "category": "cat3",
        "parent_header": "ph3",
        "header": "h3",
        "content_type": "text",
        "text": "Text 3",
    }

    res_dense = MagicMock()
    res_dense.points = [dense_p1, dense_p2]

    res_sparse = MagicMock()
    res_sparse.points = [sparse_p2, sparse_p3]

    mock_client.query_batch_points.return_value = [res_dense, res_sparse]

    searcher = HybridSearcher(client=mock_client, rrf_k=60)
    results = searcher.search(
        dense_vec=[0.1] * 1024,
        sparse_weights={"10": 0.5},
        top_k_dense=2,
        top_k_sparse=2,
        rrf_top_k=5,
    )

    # Point 2 appeared in both dense (rank 2) and sparse (rank 1):
    # RRF score = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.03252
    # Point 1 appeared only in dense (rank 1): 1/61 = ~0.016393
    # Point 3 appeared only in sparse (rank 2): 1/62 = ~0.016129
    assert len(results) == 3
    assert results[0].point_id == 2
    assert results[0].chunk_id == "c2"
    expected_p2_score = (1.0 / 62.0) + (1.0 / 61.0)
    assert pytest.approx(results[0].rrf_score, rel=1e-3) == expected_p2_score
    assert results[1].point_id == 1
    assert results[2].point_id == 3


def test_reranker_sorting():
    """Verify cross-encoder reranker sorts by score."""
    mock_model = MagicMock()
    # Mock compute_score returning scores for 2 candidates
    mock_model.compute_score.return_value = [0.25, 0.88]

    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    candidates = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            doc_title="Doc 1",
            category="chỉ định",
            parent_header="",
            header="",
            content_type="text",
            text="Passage 1",
            score=0.01,
            rrf_score=0.01,
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d2",
            doc_title="Doc 2",
            category="liều dùng",
            parent_header="",
            header="",
            content_type="text",
            text="Passage 2",
            score=0.02,
            rrf_score=0.02,
        ),
    ]

    reranked = reranker.rerank(query="test query", candidates=candidates, top_k=2)
    assert len(reranked) == 2
    # c2 had higher score (0.88) so should be ranked first
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].score == 0.88
    assert reranked[1].chunk_id == "c1"
    assert reranked[1].score == 0.25


@pytest.mark.integration
def test_full_pipeline_integration():
    """End-to-end integration test with live Qdrant and local models."""
    config = RetrievalConfig(
        collection_name="VacxiS_Capstone_FA26",
        qdrant_host="localhost",
        qdrant_port=6333,
        embed_model_path="models/bge-m3",
        reranker_model_path="models/bge-reranker",
        top_k_dense=20,
        top_k_sparse=20,
        rrf_top_k=10,
        top_k=3,
    )

    with VacxisRetriever(config=config) as retriever:
        query = "Nếu trước đây tôi từng bị dị ứng sau khi tiêm một loại vaccine khác thì có tiêm được vaccine Verorab không?"
        results = retriever.retrieve(query=query, top_k=3)

        assert len(results) == 3
        # Ensure all chunks have expected attributes
        for chunk in results:
            assert chunk.chunk_id
            assert chunk.doc_title
            assert chunk.text
            assert chunk.score > 0
            assert chunk.rrf_score > 0
            assert chunk.rerank_score is not None

        # The top result should be relevant to verorab
        assert any("verorab" in c.doc_id.lower() or "verorab" in c.text.lower() for c in results)
