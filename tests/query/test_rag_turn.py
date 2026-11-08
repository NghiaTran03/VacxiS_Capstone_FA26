import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.cli import main
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, QueryPlanner, RetrievePlan


def test_clarify_skips_retrieval_and_generation():
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.plan.return_value = ClarifyPlan("Bạn đang hỏi về vắc xin nào?")
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)

    assert pipeline.run_turn("Bé tiêm loại này được không?") == planner.plan.return_value
    assert retriever.mock_calls == []
    assert generator.mock_calls == []


def test_provider_error_stops_retrieval_and_generation():
    from vacxis_capstone_fa26.llm import LLMError
    retriever, generator = Mock(), Mock()
    planner = QueryPlanner("fixture", completion=Mock(side_effect=TimeoutError))
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    with pytest.raises(LLMError):
        pipeline.run_turn("Verorab là gì?")
    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()


def test_batch_searches_original_and_merges_unique_chunks_round_robin():
    retriever, generator, planner = Mock(), Mock(), Mock()
    original = "Bé 14 tháng tiêm loại này được không và có phản ứng phụ gì?"
    clarifications = (Clarification("Bạn hỏi vắc xin nào?", "Verorab"),)
    queries = (original, "Verorab cho trẻ 14 tháng", "Phản ứng phụ của Verorab ở trẻ 14 tháng")
    planner.plan.return_value = RetrievePlan(original, queries)
    original_chunk, first, second, extra = (SimpleNamespace(chunk_id=str(i)) for i in range(4))
    retriever.batch_retrieve.return_value = [
        [original_chunk, extra], [first], [SimpleNamespace(chunk_id=original_chunk.chunk_id), second],
    ]
    retriever.config.top_k = 4
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)

    pipeline.run_turn(original, clarifications=clarifications, skip_rerank=True)
    planner.plan.assert_called_once_with(original, clarifications)
    retriever.batch_retrieve.assert_called_once_with(queries=queries, top_k=4, skip_rerank=True)
    retriever.retrieve.assert_not_called()
    generator.generate.assert_called_once_with(
        query=f"{original}\n\nCâu hỏi làm rõ: Bạn hỏi vắc xin nào?\nNgười dùng trả lời: Verorab",
        chunks=[original_chunk, first, extra, second],
    )


def test_batch_handles_empty_groups_and_caps_final_chunks():
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.plan.return_value = RetrievePlan("Original", ("Original", "Variant"))
    chunk = SimpleNamespace(chunk_id="first")
    retriever.batch_retrieve.return_value = [[], [chunk, SimpleNamespace(chunk_id="second")]]

    VacxisRAGPipeline(retriever, generator, query_planner=planner).run_turn("Original", top_k=1)
    generator.generate.assert_called_once_with(query="Original", chunks=[chunk])


def test_cli_ignores_planner_env_for_online_queries(monkeypatch, capsys):
    from vacxis_capstone_fa26.generation import cli
    from vacxis_capstone_fa26.retrieval import retriever
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "generation-from-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", "--query", "Verorab là gì?", "--json"])
    monkeypatch.setattr(retriever, "VacxisRetriever", Mock(return_value=Mock(
        __enter__=Mock(), __exit__=Mock(return_value=False),
    )))
    pipeline = Mock()
    pipeline.answer.return_value = GeneratedAnswer(query="Verorab là gì?", answer="Answer")
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "VacxisRAGPipeline", factory)
    main()
    assert "query_planner" not in factory.call_args.kwargs
    assert factory.call_args.kwargs["generator"].config.model == "generation-from-env"
    pipeline.run_turn.assert_not_called()
    assert json.loads(capsys.readouterr().out)["answer"] == "Answer"


@pytest.mark.parametrize("arguments", [
    ["--query-planner-model", "fixture", "--dry-run"],
    ["--query-planner-model", "fixture", "--input-file", "test_dataset.json"],
    ["--clarification", "question", "answer"],
])
def test_cli_rejects_incompatible_modes(monkeypatch, arguments):
    monkeypatch.delenv("VACXIS_QUERY_PLANNER_MODEL", raising=False)
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])
    with pytest.raises(SystemExit, match="2"):
        main()


@pytest.mark.parametrize("arguments", [
    ["--dry-run", "--query", "Verorab là gì?"],
    ["--input-file", "test_dataset.json"],
])
def test_cli_ignores_planner_env_outside_single_online_mode(monkeypatch, arguments):
    from vacxis_capstone_fa26.generation.cli import parse_args

    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])

    assert not hasattr(parse_args(), "query_planner_model")
