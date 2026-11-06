import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, QueryPlanner, RetrievePlan


def response(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


def test_original_is_verbatim_first_and_deduplicated():
    original = "  Tôi không bị dị ứng trứng, có thể tiêm Verorab không?  "
    rewrite = "Hướng dẫn tiêm Verorab trong trường hợp không bị dị ứng trứng."
    complete = Mock(return_value=response(json.dumps({
        "action": "retrieve", "additional_queries": [original.strip(), rewrite],
    })))

    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(
        original, (original, rewrite),
    )
    complete.assert_called_once()
    request = complete.call_args.kwargs
    assert request["num_retries"] == 0
    assert request["response_format"] == {"type": "json_object"}
    assert json.loads(request["messages"][1]["content"]) == {
        "original_query": original, "clarifications": [],
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
    complete = Mock(return_value=response('{"action":"retrieve","additional_queries":[]}'))
    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(original, (original,))
    complete.assert_called_once()


def test_clarification_then_retrieval_with_original_and_resolved_query():
    original = "Bé 14 tháng tiêm loại này được không?"
    question = "Bạn đang hỏi về vắc xin nào cho bé 14 tháng?"
    resolved = "Hướng dẫn sử dụng Verorab cho trẻ 14 tháng."
    complete = Mock(side_effect=[
        response(json.dumps({"action": "clarify", "clarification_question": question})),
        response(json.dumps({"action": "retrieve", "additional_queries": [resolved]})),
    ])
    planner = QueryPlanner("fixture", completion=complete)

    assert planner.plan(original) == ClarifyPlan(question)
    assert planner.plan(original, (Clarification(question, "Verorab"),)) == RetrievePlan(
        original, (original, resolved),
    )
    assert complete.call_count == 2
    assert json.loads(complete.call_args.kwargs["messages"][1]["content"]) == {
        "original_query": original,
        "clarifications": [{"question": question, "answer": "Verorab"}],
    }


@pytest.mark.parametrize("output", [
    response("invalid JSON"),
    response("[]"),
    response('{"action":"clarify","clarification_question":" "}'),
    response('{"action":"clarify","clarification_question":"Tên?","additional_queries":[]}'),
    response('{"action":"retrieve","additional_queries":[" "]}'),
    response('{"action":"retrieve","additional_queries":[12]}'),
    response('{"action":"retrieve","additional_queries":["a","b","c"]}'),
    response('{"action":"retrieve","additional_queries":[],"original_query":"changed"}'),
    response('{"action":"retrieve","additional_queries":[]}', finish_reason="length"),
    SimpleNamespace(choices=[]),
    SimpleNamespace(choices={}),
])
def test_invalid_response_falls_back_once(output, caplog):
    complete = Mock(return_value=output)
    original = "  Câu hỏi gốc  "
    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(original, (original,))
    complete.assert_called_once()
    assert "query_planner_invalid_response" in caplog.text
    assert original not in caplog.text


def test_provider_failure_falls_back_once_without_logging_provider_data(caplog):
    complete = Mock(side_effect=TimeoutError("private provider data"))
    original = "Câu hỏi gốc"
    assert QueryPlanner("fixture", completion=complete).plan(original) == RetrievePlan(original, (original,))
    complete.assert_called_once()
    assert "query_planner_provider_error" in caplog.text
    assert "private provider data" not in caplog.text


@pytest.mark.parametrize(("original", "error"), [(" \t", ValueError), (None, TypeError)])
def test_invalid_input_does_not_call_provider(original, error):
    complete = Mock()
    with pytest.raises(error):
        QueryPlanner("fixture", completion=complete).plan(original)
    complete.assert_not_called()
