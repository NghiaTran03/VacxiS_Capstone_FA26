"""Tests for the retrieval package (Hybrid Search, RRF, Reranker)."""
from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
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


def test_reranker_immutability():
    """Verify reranker does not mutate original candidate chunk objects."""
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.99]

    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    original_chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Doc 1",
        category="chỉ định",
        parent_header="",
        header="",
        content_type="text",
        text="Passage 1",
        score=0.015,
        rrf_score=0.015,
        rerank_score=None,
    )
    candidates = [original_chunk]
    reranked = reranker.rerank(query="query", candidates=candidates, top_k=1)

    assert reranked[0].score == 0.99
    assert reranked[0].rerank_score == 0.99
    # Original chunk must remain unmodified
    assert original_chunk.score == 0.015
    assert original_chunk.rerank_score is None


def test_hybrid_searcher_empty_sparse_skips_sparse_query():
    """Verify empty sparse dict or empty dense vector does not create invalid QueryRequest."""
    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.points = []
    mock_client.query_batch_points.return_value = [mock_res]

    searcher = HybridSearcher(client=mock_client)

    # Empty sparse_weights should only issue dense query
    searcher.search(
        dense_vec=[0.1] * 1024,
        sparse_weights={},
        top_k_dense=10,
        top_k_sparse=10,
    )
    assert mock_client.query_batch_points.call_count == 1
    requests = mock_client.query_batch_points.call_args[1]["requests"]
    assert len(requests) == 1
    assert requests[0].using == "dense"

    # Both empty should return [] immediately without calling client
    mock_client.reset_mock()
    res = searcher.search(dense_vec=[], sparse_weights={}, top_k_dense=10, top_k_sparse=10)
    assert res == []
    mock_client.query_batch_points.assert_not_called()


def test_retriever_auto_adjusts_rrf_top_k():
    """Verify retriever automatically elevates rrf_top_k if top_k > rrf_top_k."""
    mock_embed = MagicMock()
    mock_embed.encode.return_value = {
        "dense_vecs": MagicMock(tolist=lambda: [[0.1] * 1024]),
        "lexical_weights": [{"1": 0.5}],
    }
    # For single query, embed_output["dense_vecs"][0].tolist()
    dense_mock = MagicMock()
    dense_mock.tolist.return_value = [0.1] * 1024
    mock_embed.encode.return_value = {
        "dense_vecs": [dense_mock],
        "lexical_weights": [{"1": 0.5}],
    }

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = []

    retriever = VacxisRetriever.__new__(VacxisRetriever)
    retriever.config = RetrievalConfig(top_k=5, rrf_top_k=3)
    retriever._embed_model = mock_embed
    retriever._searcher = mock_searcher
    retriever._reranker = MagicMock()

    retriever.retrieve(query="test", top_k=10, rrf_top_k=5)

    # Searcher must have been called with effective rrf_top_k = 10
    mock_searcher.search.assert_called_once()
    call_kwargs = mock_searcher.search.call_args[1]
    assert call_kwargs["rrf_top_k"] == 10


def test_reranker_handles_numpy_scalar_and_0d_array():
    """Verify reranker handles np.float32 and 0-d numpy arrays without error."""
    mock_model = MagicMock()
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    cand = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Doc 1",
        category="cat",
        parent_header="",
        header="",
        content_type="text",
        text="Sample text",
        score=0.1,
        rrf_score=0.1,
    )

    # 1. np.float32 scalar
    mock_model.compute_score.return_value = np.float32(0.77)
    res = reranker.rerank("q", [cand], top_k=1)
    assert len(res) == 1
    assert pytest.approx(res[0].score, rel=1e-3) == 0.77

    # 2. 0-d numpy array
    mock_model.compute_score.return_value = np.array(0.88)
    res = reranker.rerank("q", [cand], top_k=1)
    assert len(res) == 1
    assert pytest.approx(res[0].score, rel=1e-3) == 0.88


def test_batch_search_and_batch_rerank():
    """Verify batched search and batched rerank logic across multiple queries."""
    # Test batch_search on HybridSearcher
    mock_client = MagicMock()
    p1 = MagicMock(id=1, payload={"chunk_id": "c1", "text": "t1", "doc_title": "d1"})
    p2 = MagicMock(id=2, payload={"chunk_id": "c2", "text": "t2", "doc_title": "d2"})

    res1 = MagicMock(points=[p1])
    res2 = MagicMock(points=[p2])
    mock_client.query_batch_points.return_value = [res1, res2]

    searcher = HybridSearcher(client=mock_client)
    batch_results = searcher.batch_search(
        dense_vecs=[[0.1] * 1024, [0.2] * 1024],
        sparse_weights_list=[None, None],
        top_k_dense=5,
        top_k_sparse=0,
        rrf_top_k=5,
    )
    assert len(batch_results) == 2
    assert batch_results[0][0].chunk_id == "c1"
    assert batch_results[1][0].chunk_id == "c2"

    # Test batch_rerank on BGEReranker
    mock_model = MagicMock()
    # compute_score receives 2 pairs total across 2 queries
    mock_model.compute_score.return_value = [0.35, 0.95]
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    reranked_batch = reranker.batch_rerank(
        queries=["q1", "q2"],
        candidates_list=[batch_results[0], batch_results[1]],
        top_k=1,
    )
    assert len(reranked_batch) == 2
    assert reranked_batch[0][0].score == 0.35
    assert reranked_batch[1][0].score == 0.95


def test_retriever_empty_query():
    """Verify retrieve() returns empty list immediately for blank/empty query."""
    retriever = VacxisRetriever(lazy_load=True)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_lazy_load_default():
    """Verify VacxisRetriever defaults to lazy_load=True and does not eagerly load models."""
    retriever = VacxisRetriever()
    assert retriever._embed_model is None
    assert retriever._searcher is None
    assert retriever._reranker is None


def test_retriever_merge_table_chunks_before_rerank():
    """Verify VacxisRetriever merges split table candidates before reranking when enabled."""
    dense_mock = MagicMock()
    dense_mock.tolist.return_value = [0.1] * 1024
    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kw: {
        "dense_vecs": [dense_mock for _ in texts],
        "lexical_weights": [{"1": 0.5} for _ in texts],
    }

    t1 = RetrievedChunk(
        chunk_id="vax_c_0001",
        doc_id="vax",
        doc_title="Vax",
        category="liều dùng",
        parent_header="",
        header="Liều dùng",
        content_type="table",
        text="[CHỦ ĐỀ: Vax] > [MỤC: Liều dùng]\n<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>",
        score=0.8,
        rrf_score=0.8,
        metadata={"has_table": True},
    )
    t2 = RetrievedChunk(
        chunk_id="vax_c_0002",
        doc_id="vax",
        doc_title="Vax",
        category="liều dùng",
        parent_header="",
        header="Liều dùng",
        content_type="table",
        text="[CHỦ ĐỀ: Vax] > [MỤC: Liều dùng]\n<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody></table>",
        score=0.7,
        rrf_score=0.7,
        metadata={"has_table": True},
    )
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [t1, t2]
    mock_searcher.batch_search.return_value = [[t1, t2]]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query, candidates, top_k: candidates[:top_k]
    mock_reranker.batch_rerank.side_effect = (
        lambda queries, candidates_list, top_k: [c[:top_k] for c in candidates_list]
    )

    retriever = VacxisRetriever(
        config=RetrievalConfig(top_k=5, merge_table_chunks=True),
        embed_model=mock_embed,
        reranker=mock_reranker,
        hybrid_searcher=mock_searcher,
    )

    res = retriever.retrieve("liều dùng")
    assert len(res) == 1
    assert res[0].chunk_id == "vax_c_0001"
    assert "<tr><td>2</td></tr>" in res[0].text
    assert res[0].metadata["merged_chunk_ids"] == ["vax_c_0001", "vax_c_0002"]
    # Original candidate metadata must remain unmutated
    assert "merged_chunk_ids" not in t1.metadata
    assert len(mock_reranker.rerank.call_args.kwargs["candidates"]) == 1

    batch_res = retriever.batch_retrieve(["liều dùng"])
    assert len(batch_res) == 1
    assert len(batch_res[0]) == 1
    assert batch_res[0][0].metadata["merged_chunk_ids"] == ["vax_c_0001", "vax_c_0002"]
    assert len(mock_reranker.batch_rerank.call_args.kwargs["candidates_list"][0]) == 1


def test_run_with_oom_backoff_halves_batch_size_and_reraises_at_min():
    """Verify run_with_oom_backoff halves batch_size on CUDA OOM and re-raises non-OOM or min-batch OOM."""
    import torch
    from vacxis_capstone_fa26.retrieval.batching import run_with_oom_backoff

    attempted: list[int] = []

    def flaky_fn(bs: int) -> str:
        attempted.append(bs)
        if bs > 16:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return f"ok-{bs}"

    assert run_with_oom_backoff(flaky_fn, batch_size=64, operation="test") == "ok-16"
    assert attempted == [64, 32, 16]

    # Re-raises at min_batch_size=1
    attempted_min: list[int] = []

    def always_oom(bs: int) -> None:
        attempted_min.append(bs)
        raise RuntimeError("CUDA out of memory while allocating")

    with pytest.raises(RuntimeError, match="out of memory"):
        run_with_oom_backoff(always_oom, batch_size=4, operation="test-min")
    assert attempted_min == [4, 2, 1]

    # Non-OOM RuntimeError is propagated immediately without retry
    attempted_other: list[int] = []

    def other_error(bs: int) -> None:
        attempted_other.append(bs)
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        run_with_oom_backoff(other_error, batch_size=64, operation="test-other")
    assert attempted_other == [64]

    for bad in (0, -4, True):
        with pytest.raises(ValueError, match="positive integer"):
            run_with_oom_backoff(lambda bs: bs, batch_size=bad)


def test_retrieval_config_batch_size_defaults_and_validation():
    """Verify RetrievalConfig defaults embed/rerank batch sizes to 64 and rejects invalid values."""
    cfg = RetrievalConfig()
    assert cfg.embed_batch_size == 64
    assert cfg.rerank_batch_size == 64

    for field_name in ("embed_batch_size", "rerank_batch_size"):
        for bad in (0, -1, True):
            with pytest.raises(ValueError, match=field_name):
                RetrievalConfig(**{field_name: bad})


def test_reranker_default_and_override_batch_size_with_oom_backoff():
    """Verify BGEReranker defaults to 64, respects overrides, and backs off on CUDA OOM."""
    import torch

    seen_batch_sizes: list[int] = []

    def compute_score(pairs, *, batch_size, normalize):
        seen_batch_sizes.append(batch_size)
        if batch_size == 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return [0.9 for _ in pairs]

    mock_model = MagicMock()
    mock_model.compute_score.side_effect = compute_score

    # Compatible with BGEReranker.__new__ without self.batch_size set
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    cand = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Doc 1",
        category="cat",
        parent_header="",
        header="",
        content_type="text",
        text="Passage 1",
        score=0.1,
        rrf_score=0.1,
    )

    ranked = reranker.rerank("q", [cand], top_k=1)
    assert len(ranked) == 1
    assert seen_batch_sizes == [64, 32]

    seen_batch_sizes.clear()
    reranker.batch_size = 16
    reranker.rerank("q", [cand], top_k=1)
    assert seen_batch_sizes == [16]

    seen_batch_sizes.clear()
    reranker.rerank("q", [cand], top_k=1, batch_size=8)
    assert seen_batch_sizes == [8]

    with pytest.raises(ValueError, match="positive integer"):
        reranker.rerank("q", [cand], top_k=1, batch_size=0)


def test_retriever_batch_retrieve_uses_config_batch_sizes_and_oom_backoff():
    """Verify VacxisRetriever.batch_retrieve uses RetrievalConfig.embed_batch_size and backs off on OOM."""
    import torch

    encode_batch_sizes: list[int] = []
    dense_mock = MagicMock()
    dense_mock.tolist.return_value = [0.1] * 1024

    def fake_encode(texts, *, batch_size, **kwargs):
        encode_batch_sizes.append(batch_size)
        if batch_size == 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {
            "dense_vecs": [dense_mock for _ in texts],
            "lexical_weights": [{"1": 0.5} for _ in texts],
        }

    mock_embed = MagicMock()
    mock_embed.encode.side_effect = fake_encode

    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Doc 1",
        category="cat",
        parent_header="",
        header="",
        content_type="text",
        text="Passage 1",
        score=0.5,
        rrf_score=0.5,
    )
    mock_searcher = MagicMock()
    mock_searcher.batch_search.return_value = [[chunk], [chunk]]

    mock_reranker = MagicMock()
    mock_reranker.batch_rerank.return_value = [[chunk], [chunk]]

    retriever = VacxisRetriever(
        config=RetrievalConfig(top_k=1),
        embed_model=mock_embed,
        reranker=mock_reranker,
        hybrid_searcher=mock_searcher,
    )

    results = retriever.batch_retrieve(["q1", "q2"], rerank_batch_size=48)
    assert len(results) == 2
    assert encode_batch_sizes == [64, 32]
    assert mock_reranker.batch_rerank.call_args.kwargs["batch_size"] == 48


def test_retriever_and_reranker_log_inference_timing(caplog):
    """Verify BGE-M3 encoding and BGE-Reranker scoring emit INFO timing logs."""
    import logging

    dense_mock = MagicMock()
    dense_mock.tolist.return_value = [0.1] * 1024
    mock_embed = MagicMock()
    mock_embed.encode.side_effect = lambda texts, **kw: {
        "dense_vecs": [dense_mock for _ in texts],
        "lexical_weights": [{"1": 0.5} for _ in texts],
    }
    cand = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        doc_title="Doc 1",
        category="cat",
        parent_header="",
        header="H1",
        content_type="text",
        text="Sample text",
        score=0.5,
        rrf_score=0.5,
    )
    mock_searcher = MagicMock()
    mock_searcher.search.return_value = [cand]
    mock_searcher.batch_search.return_value = [[cand], [cand]]

    mock_reranker_model = MagicMock()
    mock_reranker_model.compute_score.side_effect = lambda pairs, **kw: [0.9 for _ in pairs]
    reranker = BGEReranker.__new__(BGEReranker)
    reranker.model = mock_reranker_model
    reranker.use_fp16 = False
    reranker.model_path = "mock-reranker"

    retriever = VacxisRetriever(
        config=RetrievalConfig(top_k=5),
        embed_model=mock_embed,
        reranker=reranker,
        hybrid_searcher=mock_searcher,
    )

    with caplog.at_level(logging.INFO):
        retriever.retrieve("q1", top_k=5)
        retriever.batch_retrieve(["q1", "q2"], top_k=5)

    assert "BGE-M3 encoded 1 query in " in caplog.text
    assert "BGE-M3 encoded 2 queries in " in caplog.text
    assert "BGE-Reranker scored 1 candidate pair(s) across 1 query(ies)" in caplog.text
    assert "BGE-Reranker scored 2 candidate pair(s) across 2 query(ies)" in caplog.text


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
