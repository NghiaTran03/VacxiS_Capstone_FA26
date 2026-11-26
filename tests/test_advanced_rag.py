"""Offline advanced RAG contracts for original-query dense search and top-k generation."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.retrieval.dense import DenseRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig


def make_retriever(top_k=2):
    model, client = Mock(), Mock()
    model.encode.return_value = {"dense_vecs": [SimpleNamespace(tolist=lambda: [0.1, 0.9])]}
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(id=1, score=0.95, payload={
            "chunk_id": "a", "text": "Text a", "doc_title": "Title a",
            "header": "Indications", "metadata": {"source": "insert"},
        }),
        SimpleNamespace(id=2, score=0.8, payload={"chunk_id": "b", "text": "Text b"}),
    ])
    retriever = DenseRetriever(
        RetrievalConfig(top_k=top_k, collection_name="test-collection"),
        embed_model=model, client=client,
    )
    return retriever, model, client


def test_embeds_original_query_once_and_returns_dense_top_k_with_native_scores():
    retriever, model, client = make_retriever()
    result = retriever.retrieve(" original question ", top_k=1)
    model.encode.assert_called_once_with(
        [" original question "], batch_size=1, max_length=512,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )
    client.query_points.assert_called_once_with(
        collection_name="test-collection", query=[0.1, 0.9], using="dense",
        limit=1, with_payload=True,
    )
    assert len(result) == 1
    assert result[0]["chunk_id"] == "a"
    assert result[0]["score"] == 0.95
    assert result[0]["doc_title"] == "Title a"
    assert result[0]["header"] == "Indications"
    assert result[0]["metadata"] == {"source": "insert"}
    assert "rrf_score" not in result[0]
    assert "rerank_score" not in result[0]
    assert retriever.last_trace == {
        "search_queries": [" original question "], "top_k": 1, "retrieved_chunk_ids": ["a"],
    }


def test_default_depth_and_empty_results_preserve_search_order():
    retriever, _, client = make_retriever()
    result = retriever.retrieve("question")
    assert [row["chunk_id"] for row in result] == ["a", "b"]
    assert [row["score"] for row in result] == [0.95, 0.8]
    assert client.query_points.call_args.kwargs["limit"] == 2
    client.query_points.return_value.points = []
    assert retriever.retrieve("missing") == []
    assert retriever.last_trace["retrieved_chunk_ids"] == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_depth_is_rejected_before_model_or_search(top_k):
    retriever, model, client = make_retriever()
    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("question", top_k=top_k)
    model.encode.assert_not_called()
    client.query_points.assert_not_called()


def test_blank_query_resets_trace_without_model_or_search():
    retriever, model, client = make_retriever()
    retriever.last_trace = {"retrieved_chunk_ids": ["stale"]}
    assert retriever.retrieve(" \n ") == []
    assert retriever.last_trace == {"search_queries": [], "top_k": 2, "retrieved_chunk_ids": []}
    model.encode.assert_not_called()
    client.query_points.assert_not_called()


def test_model_and_store_errors_propagate_and_injected_client_is_not_closed():
    retriever, model, client = make_retriever()
    model.encode.side_effect = RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        retriever.retrieve("question")
    client.query_points.assert_not_called()
    model.encode.side_effect = None
    client.query_points.side_effect = RuntimeError("store unavailable")
    with pytest.raises(RuntimeError, match="store unavailable"):
        with retriever:
            retriever.retrieve("question")
    client.close.assert_not_called()


def test_owned_client_is_lazy_and_closed_on_failure(monkeypatch):
    _, model, client = make_retriever()
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=factory))
    retriever = DenseRetriever(
        RetrievalConfig(qdrant_host="store", qdrant_port=6334), embed_model=model,
    )
    factory.assert_not_called()
    client.query_points.side_effect = RuntimeError("store unavailable")
    with pytest.raises(RuntimeError, match="store unavailable"):
        with retriever:
            retriever.retrieve("question")
    factory.assert_called_once_with(host="store", port=6334)
    client.close.assert_called_once()
    retriever.close()
    client.close.assert_called_once()


def test_advanced_retrieval_cli_and_metadata_need_no_planner_or_corpus(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

    base = ["vacxis-eval", "--advanced-rag", "--chunks", str(tmp_path / "absent.parquet")]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", base + ["--retrieval-only"])
    args = cli.parse_args()
    assert args.advanced_rag
    retriever = cli._build_advanced_retriever(args)
    assert isinstance(retriever, DenseRetriever)
    summary = EvaluationSummary(per_query_results=[{"query_id": "Q1", "metadata": {}}])
    cli._record_retrieval_run(summary, args)
    run = summary.metadata["run"]
    assert run["retriever"] == "advanced_dense"
    assert run["dense_only"] is True
    assert run["skip_rerank"] is True
    assert "planner_model" not in run
    assert "reranker_path" not in run
    trace = {"search_queries": ["question"], "top_k": 1, "retrieved_chunk_ids": ["a"]}
    cli._attach_retrieval_traces(summary, {"Q1": trace})
    assert summary.per_query_results[0]["metadata"]["retrieval_trace"] == trace
    assert "planner_status_counts" not in summary.metadata
    for extra in (["--baseline"], ["--hybrid-baseline"], ["--results", "saved.json"],
                  ["--top-k", "0"], ["--planner-model", "unused"]):
        monkeypatch.setattr(sys, "argv", base + ["--retrieval-only"] + extra)
        with pytest.raises(SystemExit):
            cli.parse_args()


def test_full_evaluation_requires_corpus_for_judge_context(monkeypatch, tmp_path, capsys):
    from vacxis_capstone_fa26.evaluation.cli import parse_args

    chunks = tmp_path / "chunks.parquet"
    monkeypatch.setattr(sys, "argv", ["vacxis-eval", "--advanced-rag", "--chunks", str(chunks)])
    with pytest.raises(SystemExit):
        parse_args()
    assert "requires a chunk corpus" in capsys.readouterr().err
    chunks.touch()
    assert parse_args().advanced_rag


def test_dense_top_k_flows_directly_to_generator(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.generation import generator

    retriever, model, client = make_retriever()
    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    monkeypatch.setattr(sys, "argv", [
        "vacxis-eval", "--advanced-rag", "--top-k", "1", "--chunks", str(chunks),
    ])
    monkeypatch.setattr(cli, "_build_advanced_retriever", Mock(return_value=retriever))
    gen = Mock()
    gen.generate.return_value.to_dict.return_value = {"answer": "answer"}
    monkeypatch.setattr(generator, "MedicalGenerator", Mock(return_value=gen))
    pipeline, answer_fn = cli._build_answer_fn(cli.parse_args(), {})
    with pipeline:
        assert answer_fn("original question", {"query_id": "Q1"}) == {"answer": "answer"}
    model.encode.assert_called_once()
    client.query_points.assert_called_once()
    gen.generate.assert_called_once()
    sent = gen.generate.call_args.kwargs
    assert sent["query"] == "original question"
    assert [row["chunk_id"] for row in sent["chunks"]] == ["a"]
    assert sent["chunks"][0]["score"] == 0.95
    assert answer_fn.retrieval_traces["Q1"]["retrieved_chunk_ids"] == ["a"]
    gen.close.assert_called_once()


def test_advanced_import_uses_dense_retrieval():
    from vacxis_capstone_fa26.retrieval import DenseRetriever as PublicRetriever
    from vacxis_capstone_fa26.retrieval.advanced import AdvancedRAGRetriever

    assert AdvancedRAGRetriever is PublicRetriever is DenseRetriever


def test_advanced_generation_and_retrieval_evaluation_share_factory(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.generation import generator

    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--advanced-rag", "--chunks", str(chunks)])
    args = cli.parse_args()
    retriever = Mock()
    retriever.__enter__ = Mock(return_value=retriever)
    retriever.__exit__ = Mock(return_value=False)
    ranked = [{"chunk_id": "c1", "text": "context", "doc_title": "Title"}]
    retriever.retrieve.return_value = ranked
    retriever.last_trace = {"search_queries": ["question"], "retrieved_chunk_ids": ["c1"]}
    factory = Mock(return_value=retriever)
    monkeypatch.setattr(cli, "_build_advanced_retriever", factory)
    gen = Mock()
    gen.generate.return_value.to_dict.return_value = {"answer": "answer", "retrieved_chunk_ids": ["c1"]}
    monkeypatch.setattr(generator, "MedicalGenerator", Mock(return_value=gen))
    pipeline, answer_fn = cli._build_answer_fn(args, {})
    with pipeline:
        assert answer_fn("question", {"query_id": "Q1"})["retrieved_chunk_ids"] == ["c1"]
    gen.generate.assert_called_once_with(query="question", chunks=ranked)
    gen.close.assert_called_once()
    evaluator = Mock()
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary
    evaluator.evaluate_retrieval.return_value = EvaluationSummary(per_query_results=[{"query_id": "Q1", "metadata": {}}])
    dataset = [{"QUERY_ID": "Q1", "QUERY": "question"}]
    cli._evaluate_live_retrieval(evaluator, dataset, args)
    evaluated = evaluator.evaluate_retrieval.call_args.args[1]
    assert evaluated[0]["GENERATED_ANSWER"]["retrieved_chunk_ids"] == ["c1"]
    assert factory.call_count == 2
    assert retriever.__exit__.call_count == 2


def test_dense_batch_retrieve_encodes_in_single_batch_with_config_batch_size_and_oom_backoff():
    import torch

    retriever, model, client = make_retriever(top_k=1)
    seen_batch_sizes = []

    def fake_encode(texts, *, batch_size, **kwargs):
        seen_batch_sizes.append(batch_size)
        if batch_size == 64:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {
            "dense_vecs": [SimpleNamespace(tolist=lambda: [0.1, 0.9]) for _ in texts]
        }

    model.encode.side_effect = fake_encode

    results = retriever.batch_retrieve(["q1", "   ", "q2"])
    assert len(results) == 3
    assert len(results[0]) == 1
    assert results[1] == []
    assert len(results[2]) == 1
    assert seen_batch_sizes == [64, 32]
    assert model.encode.call_args.args[0] == ["q1", "q2"]
    assert client.query_points.call_count == 2
