"""Offline advanced RAG tests with injected planner, search, and reranker stubs."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query.schemas import ClarifyPlan, RetrievePlan
from vacxis_capstone_fa26.retrieval.advanced import AdvancedRAGRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig


def row(cid):
    return {"chunk_id": cid, "text": f"Text {cid}", "doc_title": f"Title {cid}"}


def test_expands_fuses_and_reranks_against_original_question():
    search = Mock()
    search.retrieve.side_effect = [[row("a"), row("shared")], [row("b"), row("shared")]]
    planner = Mock()
    planner.plan.return_value = RetrievePlan("ignored", ("question", "expansion"))
    reranker = Mock()
    reranker.rerank.side_effect = lambda query, candidates, top_k: [replace(candidates[1], score=0.9, rerank_score=0.9)]
    advanced = AdvancedRAGRetriever(search, planner, candidate_k=10, reranker=reranker)
    result = advanced.retrieve("question", top_k=1)
    assert [c.kwargs["query"] for c in search.retrieve.call_args_list] == ["question", "expansion"]
    call = reranker.rerank.call_args.kwargs
    assert call["query"] == "question"
    assert [c.chunk_id for c in call["candidates"]] == ["shared", "a", "b"]
    assert call["candidates"][0].rrf_score == pytest.approx(2 / 62)
    assert result[0]["doc_title"] == "Title a"
    assert result[0]["rerank_score"] == 0.9
    assert advanced.last_trace["planner_status"] == "expanded"
    assert advanced.last_trace["unique_candidate_count"] == 3
    assert advanced.last_trace["retrieved_chunk_ids"] == ["a"]


@pytest.mark.parametrize("plan,status", [
    (ClarifyPlan("Which vaccine?"), "clarification_fallback"),
    (LLMError("unavailable"), "planner_error_fallback"),
    (RetrievePlan("question", ("question",)), "original_only"),
])
def test_planner_fallback_keeps_original_search(plan, status):
    search, planner = Mock(), Mock()
    search.retrieve.return_value = [row("a")]
    if isinstance(plan, Exception):
        planner.plan.side_effect = plan
    else:
        planner.plan.return_value = plan
    advanced = AdvancedRAGRetriever(search, planner, config=RetrievalConfig(skip_rerank=True))
    assert advanced.retrieve("question", top_k=1)[0]["chunk_id"] == "a"
    search.retrieve.assert_called_once_with(query="question", top_k=50)
    assert advanced.last_trace["planner_status"] == status
    assert advanced.last_trace["search_queries"] == ["question"]


def test_query_and_candidate_budgets_and_deduplication():
    search, planner = Mock(), Mock()
    search.retrieve.return_value = [row("a"), row("a"), row("b")]
    planner.plan.return_value = RetrievePlan("question", (" question ", "one", " one ", "two", "three"))
    advanced = AdvancedRAGRetriever(search, planner, config=RetrievalConfig(skip_rerank=True), candidate_k=1)
    result = advanced.retrieve("question", top_k=2)
    assert len(search.retrieve.call_args_list) == 3
    assert [c.kwargs["query"] for c in search.retrieve.call_args_list] == ["question", "one", "two"]
    assert all(c.kwargs["top_k"] == 2 for c in search.retrieve.call_args_list)
    assert [r["chunk_id"] for r in result] == ["a", "b"]
    assert result[1]["rrf_score"] == pytest.approx(3 / 62)


def test_search_errors_propagate_and_owned_resources_close():
    search, planner = Mock(), Mock()
    planner.plan.return_value = RetrievePlan("q", ("q",))
    search.retrieve.side_effect = RuntimeError("store unavailable")
    advanced = AdvancedRAGRetriever(search, planner, owns_retriever=True)
    with pytest.raises(RuntimeError, match="store unavailable"):
        with advanced:
            advanced.retrieve("q")
    search.close.assert_called_once()


def test_empty_query_and_invalid_top_k_do_not_invoke_models():
    search, planner = Mock(), Mock()
    advanced = AdvancedRAGRetriever(search, planner)
    assert advanced.retrieve(" ") == []
    with pytest.raises(ValueError, match="positive integer"):
        advanced.retrieve("q", top_k=0)
    planner.plan.assert_not_called()
    search.retrieve.assert_not_called()


def test_advanced_cli_and_report_traces(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

    chunks = tmp_path / "chunks.parquet"
    chunks.touch()
    base = ["vacxis-eval", "--advanced-rag", "--chunks", str(chunks), "--planner-model", "test-planner"]
    monkeypatch.setattr("sys.argv", base)
    args = cli.parse_args()
    summary = EvaluationSummary(per_query_results=[{"query_id": "Q1", "metadata": {}}])
    cli._record_retrieval_run(summary, args)
    assert summary.metadata["run"]["retriever"] == "advanced_bm25_dense"
    assert summary.metadata["run"]["planner_model"] == "test-planner"
    trace = {"planner_status": "clarification_fallback", "search_queries": ["q"]}
    cli._attach_advanced_traces(summary, {"Q1": trace})
    assert summary.per_query_results[0]["metadata"]["retrieval_trace"] == trace
    assert summary.metadata["planner_status_counts"] == {"clarification_fallback": 1}
    for extra in (["--baseline"], ["--hybrid-baseline"], ["--results", "saved.json"],
                  ["--advanced-candidate-k", "0"], ["--planner-timeout", "121"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit):
            cli.parse_args()
