import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, QueryPlanner, RetrievePlan
from vacxis_capstone_fa26.query.planner import QUERY_PLAN_RESPONSE_FORMAT


def response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


def test_resolved_is_first_and_deduplicated():
    original = "  Tôi không bị dị ứng trứng, có thể tiêm Verorab không?  "
    rewrite = "Hướng dẫn tiêm Verorab trong trường hợp không bị dị ứng trứng."
    complete = Mock(return_value=response(json.dumps({
        "action": "retrieve",
        "resolved_query": rewrite,
        "additional_queries": [rewrite, rewrite],
        "clarification_question": None,
    })))

    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(
        original, (rewrite,), rewrite,
    )
    complete.assert_called_once()
    request = complete.call_args.kwargs
    assert "num_retries" not in request
    assert request["response_format"] == QUERY_PLAN_RESPONSE_FORMAT
    assert request["extra_body"] == {"provider": {"require_parameters": True}}
    assert json.loads(request["messages"][1]["content"]) == {
        "original_query": original, "clarifications": [], "history": [],
    }


@pytest.mark.parametrize("original", [
    "Vắc xin cúm có phản ứng phụ gì?",
    "Tôi có thể tiêm Verorab nếu tôi không sốt và không có dị ứng trứng không?",
    "Nếu đã tiêm Verorab thì tôi có bị dị ứng trứng không?",
    "vaxigrip( 0.75 ml ) có tiêm được không và có phản ứng phụ nào?",
    "vaxigrip( 0.25 ml ) có tiêm được không và có phản ứng phụ nào?",
    "Verorab (0,5 ml) có tiêm được không và có phản ứng phụ nào?",
])
def test_retrieve_without_variants_keeps_query(original):
    complete = Mock(return_value=response(json.dumps({
        "action": "retrieve", "resolved_query": original, "additional_queries": [],
    })))
    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(original, (original,))
    complete.assert_called_once()


def test_clarification_then_retrieval_with_resolved_query():
    original = "Bé 14 tháng tiêm loại này được không?"
    question = "Bạn đang hỏi về vắc xin nào cho bé 14 tháng?"
    resolved = "Hướng dẫn sử dụng Verorab cho trẻ 14 tháng."
    complete = Mock(side_effect=[
        response(json.dumps({
            "action": "clarify",
            "resolved_query": None,
            "additional_queries": None,
            "clarification_question": question,
        })),
        response(json.dumps({
            "action": "retrieve",
            "resolved_query": resolved,
            "additional_queries": [],
            "clarification_question": None,
        })),
    ])
    planner = QueryPlanner("fixture", completion=complete)

    assert planner.plan(original) == ClarifyPlan(question)
    assert planner.plan(original, (Clarification(question, "Verorab"),)) == RetrievePlan(
        original, (resolved,), resolved,
    )
    assert complete.call_count == 2
    assert json.loads(complete.call_args.kwargs["messages"][1]["content"]) == {
        "original_query": original,
        "clarifications": [{"question": question, "answer": "Verorab"}],
        "history": [],
    }


@pytest.mark.parametrize("output", [
    response("invalid JSON"),
    response("[]"),
    response('{"action":"clarify","clarification_question":" "}'),
    response('{"action":"clarify","clarification_question":"Tên?","additional_queries":[]}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[" "]}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[12]}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":["a","b","c"]}'),
    response('{"action":"retrieve","additional_queries":[]}'),
    response('{"action":"retrieve","resolved_query":" ","additional_queries":[]}'),
    response('{"action":"retrieve","resolved_query":12,"additional_queries":[]}'),
    response('{"action":"retrieve","additional_queries":[],"original_query":"changed"}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[],"clarification_question":"?"}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[],"clarification_question":""}'),
    response('{"action":"clarify","resolved_query":"Q","additional_queries":null,"clarification_question":"Tên?"}'),
    response('{"action":"retrieve","resolved_query":null,"additional_queries":[],"clarification_question":null}'),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[]}', finish_reason="length"),
    response('{"action":"retrieve","resolved_query":"Q","additional_queries":[]}', finish_reason="content_filter"),
    SimpleNamespace(choices=[]),
    SimpleNamespace(choices={}),
])
def test_invalid_response_raises_once(output, caplog):
    complete = Mock(return_value=output)
    original = "  Câu hỏi gốc  "
    with pytest.raises(LLMError):
        QueryPlanner("fixture", completion=complete).plan(original)
    complete.assert_called_once()
    assert original not in caplog.text


def test_provider_failure_raises_without_logging_provider_data(caplog):
    complete = Mock(side_effect=TimeoutError("private provider data"))
    original = "Câu hỏi gốc"
    with pytest.raises(LLMError, match="request failed"):
        QueryPlanner("fixture", completion=complete).plan(original)
    complete.assert_called_once()
    assert "private provider data" not in caplog.text


@pytest.mark.parametrize(("original", "error"), [(" \t", ValueError), (None, TypeError)])
def test_invalid_input_does_not_call_provider(original, error):
    complete = Mock()
    with pytest.raises(error):
        QueryPlanner("fixture", completion=complete).plan(original)
    complete.assert_not_called()
