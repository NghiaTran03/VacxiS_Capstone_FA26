import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
from vacxis_capstone_fa26.query import (
    Clarification, ClarifyPlan, ConversationMessage, QueryPlanner, RetrievePlan,
)


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


def test_batch_searches_resolved_and_merges_unique_chunks_round_robin():
    retriever, generator, planner = Mock(), Mock(), Mock()
    original = "Bé 14 tháng tiêm loại này được không và có phản ứng phụ gì?"
    clarifications = (Clarification("Bạn hỏi vắc xin nào?", "Verorab"),)
    resolved = "Verorab cho trẻ 14 tháng có phản ứng phụ gì và tiêm được không?"
    queries = (resolved, "Verorab cho trẻ 14 tháng", "Phản ứng phụ của Verorab ở trẻ 14 tháng")
    planner.plan.return_value = RetrievePlan(original, queries, resolved)
    generator.generate.return_value = GeneratedAnswer(resolved, "Answer")
    original_chunk, first, second, extra = (SimpleNamespace(chunk_id=str(i)) for i in range(4))
    retriever.batch_retrieve.return_value = [
        [original_chunk, extra], [first], [SimpleNamespace(chunk_id=original_chunk.chunk_id), second],
    ]
    retriever.config.top_k = 4
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)

    pipeline.run_turn(original, clarifications=clarifications, skip_rerank=True)
    planner.plan.assert_called_once_with(original, clarifications, history=())
    retriever.batch_retrieve.assert_called_once_with(queries=queries, top_k=4, skip_rerank=True)
    retriever.retrieve.assert_not_called()
    generator.generate.assert_called_once_with(
        query=resolved,
        chunks=[original_chunk, first, extra, second],
    )


def test_batch_handles_empty_groups_and_caps_final_chunks():
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.plan.return_value = RetrievePlan("Original", ("Original", "Variant"))
    generator.generate.return_value = GeneratedAnswer("Original", "Answer")
    chunk = SimpleNamespace(chunk_id="first")
    retriever.batch_retrieve.return_value = [[], [chunk, SimpleNamespace(chunk_id="second")]]

    VacxisRAGPipeline(retriever, generator, query_planner=planner).run_turn("Original", top_k=1)
    generator.generate.assert_called_once_with(query="Original", chunks=[chunk])


def test_run_turn_logs_retrieve_plan_at_debug(caplog):
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.model = "test/planner-model"
    original = "Bé 14 tháng tiêm loại này được không?"
    resolved = "Hướng dẫn sử dụng Verorab cho trẻ 14 tháng."
    queries = (resolved, "Liều dùng Verorab cho trẻ nhỏ")
    planner.plan.return_value = RetrievePlan(original, queries, resolved)
    retriever.batch_retrieve.return_value = [[SimpleNamespace(chunk_id="c1")], [SimpleNamespace(chunk_id="c2")]]
    retriever.config.top_k = 2
    generator.generate.return_value = GeneratedAnswer(resolved, "Answer")

    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    with caplog.at_level(logging.DEBUG, logger="vacxis_capstone_fa26.generation.pipeline"):
        pipeline.run_turn(
            original,
            clarifications=(Clarification("Bạn hỏi vắc xin nào?", "Verorab"),),
            history=(ConversationMessage("user", "Tôi muốn hỏi về Verorab"),),
        )

    plan_records = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("QueryPlanner output:")
    ]
    assert len(plan_records) == 1
    payload = json.loads(plan_records[0].split("QueryPlanner output:\n", 1)[1])
    assert payload["action"] == "retrieve"
    assert payload["model"] == "test/planner-model"
    assert payload["original_query"] == original
    assert payload["resolved_query"] == resolved
    assert payload["search_queries"] == list(queries)
    assert payload["history_included"] == 1
    assert payload["history_supplied"] == 1
    assert payload["clarifications_count"] == 1
    assert isinstance(payload["latency_s"], float)


def test_run_turn_logs_clarify_plan_at_debug(caplog):
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.model = "test/planner-model"
    question = "Bạn đang hỏi về vắc xin nào?"
    planner.plan.return_value = ClarifyPlan(question)

    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    with caplog.at_level(logging.DEBUG, logger="vacxis_capstone_fa26.generation.pipeline"):
        pipeline.run_turn("Bé tiêm loại này được không?")

    plan_records = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("QueryPlanner output:")
    ]
    assert len(plan_records) == 1
    payload = json.loads(plan_records[0].split("QueryPlanner output:\n", 1)[1])
    assert payload["action"] == "clarify"
    assert payload["clarification_question"] == question
    assert "resolved_query" not in payload


def test_run_turn_does_not_log_plan_at_info(caplog):
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.model = "test/planner-model"
    planner.plan.return_value = RetrievePlan("Verorab là gì?", ("Verorab là gì?",), "Verorab là gì?")
    retriever.retrieve.return_value = [SimpleNamespace(chunk_id="c1", text="Sample")]
    generator.generate.return_value = GeneratedAnswer("Verorab là gì?", "Answer")

    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    with caplog.at_level(logging.INFO, logger="vacxis_capstone_fa26.generation.pipeline"):
        pipeline.run_turn("Verorab là gì?")

    assert "QueryPlanner output:" not in caplog.text
    assert "Retrieved chunks after rerank" not in caplog.text


def test_run_turn_logs_retrieved_chunks_at_debug(caplog):
    retriever, generator, planner = Mock(), Mock(), Mock()
    planner.model = "test/planner-model"
    planner.plan.return_value = RetrievePlan("Verorab là gì?", ("Verorab là gì?",), "Verorab là gì?")
    long_text = "Nội dung chi tiết về vắc xin Verorab. " * 10
    chunks = [
        SimpleNamespace(
            chunk_id="verorab_c_0001",
            doc_id="verorab_001",
            doc_title="Hướng dẫn sử dụng Verorab",
            header="Chỉ định",
            rerank_score=0.98765,
            rrf_score=0.0321,
            score=0.98765,
            text=long_text,
        ),
        SimpleNamespace(
            chunk_id="verorab_c_0002",
            doc_id="verorab_001",
            doc_title="Hướng dẫn sử dụng Verorab",
            header="Liều dùng",
            rerank_score=0.81234,
            rrf_score=0.021,
            score=0.81234,
            text="Tiêm bắp 0,5 ml.",
        ),
    ]
    retriever.retrieve.return_value = chunks
    generator.generate.return_value = GeneratedAnswer("Verorab là gì?", "Answer")

    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    with caplog.at_level(logging.DEBUG, logger="vacxis_capstone_fa26.generation.pipeline"):
        pipeline.run_turn("Verorab là gì?")

    chunk_records = [
        rec.getMessage()
        for rec in caplog.records
        if rec.getMessage().startswith("Retrieved chunks after rerank")
    ]
    assert len(chunk_records) == 1
    payload = json.loads(chunk_records[0].split("\n", 1)[1])
    assert len(payload) == 2
    assert payload[0]["rank"] == 1
    assert payload[0]["chunk_id"] == "verorab_c_0001"
    assert payload[0]["rerank_score"] == 0.9877
    assert payload[0]["text_preview"].endswith("…")
    assert payload[1]["rank"] == 2
    assert payload[1]["text_preview"] == "Tiêm bắp 0,5 ml."
