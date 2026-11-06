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


def test_fallback_retrieves_only_the_original():
    retriever, generator = Mock(), Mock()
    complete = Mock(side_effect=TimeoutError)
    planner = QueryPlanner("fixture", completion=complete)
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)

    original = "  Verorab là gì?  "
    assert pipeline.run_turn(original) is generator.generate.return_value
    retriever.retrieve.assert_called_once_with(query=original, top_k=None, skip_rerank=None)
    retriever.batch_retrieve.assert_not_called()
    generator.generate.assert_called_once_with(query=original, chunks=retriever.retrieve.return_value)


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


@pytest.mark.parametrize(("model_flags", "planner_model", "generation_model"), [
    ([], "planner-from-env", "generation-from-env"),
    (["--query-planner-model", "planner-from-cli", "--model", "generation-from-cli"],
     "planner-from-cli", "generation-from-cli"),
])
def test_cli_outputs_clarify_and_passes_verbatim_query(
    monkeypatch, capsys, model_flags, planner_model, generation_model,
):
    from vacxis_capstone_fa26.generation import cli
    from vacxis_capstone_fa26.retrieval import retriever

    original = "  Bé 14 tháng tiêm loại này được không?  "
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setenv("VACXIS_LLM_MODEL", "generation-from-env")
    monkeypatch.setattr("sys.argv", [
        "vacxis-generation", "--query", original, "--json", *model_flags,
        "--clarification", "Bạn đã xác định được sản phẩm chưa?", "Chưa",
    ])
    monkeypatch.setattr(retriever, "VacxisRetriever", Mock(return_value=Mock(
        __enter__=Mock(), __exit__=Mock(return_value=False),
    )))
    pipeline = Mock()
    pipeline.run_turn.return_value = ClarifyPlan("Bạn đang hỏi về vắc xin nào?")
    build_pipeline = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "VacxisRAGPipeline", build_pipeline)

    main()

    assert build_pipeline.call_args.kwargs["query_planner"].model == planner_model
    assert build_pipeline.call_args.kwargs["generator"].config.model == generation_model
    pipeline.run_turn.assert_called_once_with(
        original_query=original,
        clarifications=(Clarification("Bạn đã xác định được sản phẩm chưa?", "Chưa"),),
        top_k=4, skip_rerank=False,
    )
    assert json.loads(capsys.readouterr().out) == {
        "action": "clarify", "clarification_question": "Bạn đang hỏi về vắc xin nào?",
    }


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

    assert parse_args().query_planner_model is None
