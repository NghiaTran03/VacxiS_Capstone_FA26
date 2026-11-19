"""Offline contract tests; no embedding models, Qdrant, or live retrieval."""
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vacxis_capstone_fa26.retrieval.hybrid_baseline import HybridBaselineRetriever


def make_retriever():
    corpus = {
        cid: {"chunk_id": cid, "text": "vaccine", "doc_title": f"Title {cid}", "header": "Indications"}
        for cid in ("a", "b", "shared")
    }
    dense = Mock()
    dense.retrieve.return_value = [SimpleNamespace(chunk_id="b"), SimpleNamespace(chunk_id="shared")]
    retriever = HybridBaselineRetriever(corpus, dense_retriever=dense, candidate_k=20)
    retriever._bm25 = Mock()
    retriever._bm25.retrieve.return_value = [{"chunk_id": "a"}, {"chunk_id": "shared"}]
    return retriever, dense


def test_rrf_rewards_agreement_and_preserves_metadata():
    retriever, dense = make_retriever()
    chunks = retriever.retrieve("question", top_k=3)
    assert [c["chunk_id"] for c in chunks] == ["shared", "a", "b"]
    assert chunks[0]["score"] == pytest.approx(2 / 62)
    assert chunks[0]["doc_title"] == "Title shared"
    assert chunks[0]["header"] == "Indications"
    dense.retrieve.assert_called_once_with(
        query="question", top_k=20, top_k_dense=20,
        top_k_sparse=0, rrf_top_k=20, skip_rerank=True,
    )
    assert retriever.config.skip_rerank is True
    assert retriever.config.top_k_sparse == 0


def test_duplicate_ids_do_not_gain_extra_votes_and_depth_expands():
    retriever, dense = make_retriever()
    dense.retrieve.return_value = [{"chunk_id": "b"}, {"chunk_id": "b"}, {"chunk_id": "shared"}]
    chunks = retriever.retrieve("question", top_k=30)
    assert len(chunks) == 3
    assert chunks[0]["score"] == pytest.approx(2 / 62)
    assert dense.retrieve.call_args.kwargs["top_k_dense"] == 30
    retriever.close()
    dense.close.assert_not_called()


def test_rejects_dense_results_from_another_corpus():
    retriever, dense = make_retriever()
    dense.retrieve.return_value = [{"chunk_id": "unknown"}]
    with pytest.raises(ValueError, match="corpus/collection alignment"):
        retriever.retrieve("question")


def test_blank_query_and_invalid_top_k_do_not_call_dense():
    retriever, dense = make_retriever()
    assert retriever.retrieve("   ") == []
    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("q", top_k=0)
    dense.retrieve.assert_not_called()


def test_owned_dense_resource_is_closed_on_failure():
    retriever, dense = make_retriever()
    retriever._owns_dense = True
    with pytest.raises(RuntimeError):
        with retriever:
            raise RuntimeError("evaluation failed")
    dense.close.assert_called_once()
    assert retriever._dense_retriever is None


def test_parquet_keeps_context_fields_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "chunks.parquet"
    row = {"chunk_id": "c1", "text": "vaccine", "doc_title": "Title", "category": "cat", "header": "header"}
    pq.write_table(pa.Table.from_pylist([row]), path)
    retriever = HybridBaselineRetriever.from_parquet(path, dense_retriever=Mock())
    assert retriever._corpus["c1"] == row
    pq.write_table(pa.Table.from_pylist([row, row]), path)
    with pytest.raises(ValueError, match="Duplicate corpus"):
        HybridBaselineRetriever.from_parquet(path)


def test_hybrid_cli_selection_and_incompatible_results(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation.cli import parse_args

    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    base = ["vacxis-eval", "--hybrid-baseline", "--chunks", str(chunks)]
    monkeypatch.setattr("sys.argv", base + ["--retrieval-only", "--top-k", "10"])
    args = parse_args()
    assert args.hybrid_baseline and not args.baseline
    assert args.baseline_candidate_k == 50
    assert args.baseline_rrf_k == 60
    assert args.ir_k_values == [1, 3, 5, 10]
    for extra in (["--baseline"], ["--results", "saved.json"], ["--baseline-candidate-k", "0"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit):
            parse_args()


def test_hybrid_generation_and_retrieval_evaluation_share_factory(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.generation import generator

    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--hybrid-baseline", "--chunks", str(chunks)])
    args = cli.parse_args()
    retriever = Mock()
    retriever.__enter__ = Mock(return_value=retriever)
    retriever.__exit__ = Mock(return_value=False)
    ranked = [{"chunk_id": "c1", "text": "context", "doc_title": "Title"}]
    retriever.retrieve.return_value = ranked
    factory = Mock(return_value=retriever)
    monkeypatch.setattr(cli, "_build_hybrid_baseline", factory)
    gen = Mock()
    gen.generate.return_value.to_dict.return_value = {"answer": "answer", "retrieved_chunk_ids": ["c1"]}
    monkeypatch.setattr(generator, "MedicalGenerator", Mock(return_value=gen))
    pipeline, answer_fn = cli._build_answer_fn(args, {})
    with pipeline:
        assert answer_fn("question", {})["retrieved_chunk_ids"] == ["c1"]
    gen.generate.assert_called_once_with(query="question", chunks=ranked)
    gen.close.assert_called_once()
    evaluator = Mock()
    dataset = [{"QUERY_ID": "Q1", "QUERY": "question"}]
    cli._evaluate_live_retrieval(evaluator, dataset, args)
    evaluated = evaluator.evaluate_retrieval.call_args.args[1]
    assert evaluated[0]["GENERATED_ANSWER"]["retrieved_chunk_ids"] == ["c1"]
    assert factory.call_count == 2
    assert retriever.__exit__.call_count == 2


def test_run_metadata_identifies_distinct_baseline(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--hybrid-baseline", "--chunks", str(chunks)])
    summary = EvaluationSummary()
    cli._record_retrieval_run(summary, cli.parse_args())
    run = summary.metadata["run"]
    assert run["retriever"] == "bm25_dense_rrf"
    assert run["skip_rerank"] is True
    assert run["candidate_k"] == 50
    assert "reranker_path" not in run


def test_full_evaluation_records_unanswerable_dataset(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--unanswerable", str(tmp_path / "unanswerable.json")])
    args = cli.parse_args()
    summaries = [EvaluationSummary(metadata={"run": {}}), EvaluationSummary(metadata={"run": {}})]
    monkeypatch.setattr(cli, "_evaluate_live", Mock(side_effect=summaries))
    monkeypatch.setattr(cli, "_load_dataset_items", Mock(return_value=[{}]))
    monkeypatch.setattr(cli, "_load_flat_dataset", Mock(return_value=[{}]))
    monkeypatch.setattr(cli, "_print_summary", Mock())
    monkeypatch.setattr(cli, "_maybe_export", Mock())
    cli._run_live(Mock(), args, {})
    assert summaries[1].metadata["run"]["dataset"] == str(args.unanswerable)
