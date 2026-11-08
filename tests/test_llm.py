"""OpenRouter wire contract and strict online failure tests."""
import json
from unittest.mock import Mock

import httpx
from openai import OpenAI
import pytest

from vacxis_capstone_fa26 import llm
from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.query import QueryPlanner
from vacxis_capstone_fa26.testdata.answer_generator import OpenRouterAnswerGenerator
from vacxis_capstone_fa26.testdata.refiner import OpenRouterRefiner


def chat_response(content="Answer [c1].", finish_reason="stop"):
    return {
        "id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def router(monkeypatch):
    """Exercise the real OpenAI SDK against an in-process HTTP transport."""
    requests, clients = [], []
    result = {"status": 200, "body": chat_response()}
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "fixture/generation")
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "fixture/planner")

    def handle(request):
        requests.append(request)
        return httpx.Response(result["status"], json=result["body"])

    def factory(**kwargs):
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert kwargs["max_retries"] == 0
        client = httpx.Client(transport=httpx.MockTransport(handle))
        clients.append(client)
        return OpenAI(**kwargs, http_client=client)

    monkeypatch.setattr(llm, "OpenAI", factory)
    return result, requests, clients


def test_transport_and_generator_wire_contract(router):
    _, requests, clients = router
    answer = MedicalGenerator().generate("Question", [{"chunk_id": "c1", "text": "Evidence"}])
    assert answer.citations == ["c1"]
    assert answer.model == "fixture/generation"
    assert answer.usage["total_tokens"] == 15
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer fixture-secret"
    body = json.loads(request.content)
    assert body["model"] == "fixture/generation"
    assert body["messages"][1]["role"] == "user"
    assert body["max_tokens"] == 1024
    assert clients[0].is_closed


@pytest.mark.parametrize("field", ["OPENROUTER_API_KEY", "VACXIS_GENERATION_MODEL"])
def test_missing_settings_fail_before_client_creation(monkeypatch, field):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "fixture/model")
    monkeypatch.delenv(field)
    factory = Mock()
    monkeypatch.setattr(llm, "OpenAI", factory)
    with pytest.raises(llm.LLMError, match=field):
        MedicalGenerator().generate("Question", [])
    factory.assert_not_called()


def test_model_override_and_separate_planner_model(monkeypatch):
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "generation")
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner")
    assert llm.resolve_model("override") == "override"
    assert QueryPlanner().model == "planner"
    with pytest.raises(llm.LLMError):
        llm.resolve_model(" ")


@pytest.mark.parametrize("status,body", [
    (401, {"error": {"message": "private fixture-secret", "type": "invalid_request_error"}}),
    (429, {"error": {"message": "private fixture-secret"}}),
    (500, {"error": {"message": "private fixture-secret"}}),
    (200, chat_response("")),
    (200, chat_response(None)),
    (200, chat_response("truncated", "length")),
    (200, {**chat_response(), "choices": []}),
])
def test_errors_are_safe_and_close_client(router, status, body, caplog):
    result, requests, clients = router
    result.update(status=status, body=body)
    with pytest.raises(llm.LLMError) as exc:
        llm.completion(model="fixture", messages=[])
    assert "fixture-secret" not in str(exc.value) + caplog.text
    assert "private" not in str(exc.value) + caplog.text
    assert len(requests) == 1
    assert clients[0].is_closed


def test_timeout_is_safe_and_closes_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("private"))
    ))
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: OpenAI(**kwargs, http_client=client))
    with pytest.raises(llm.LLMError, match="request failed"):
        llm.completion(model="fixture", messages=[], timeout=0.1)
    assert client.is_closed


def test_dry_run_never_creates_client(monkeypatch):
    factory = Mock(side_effect=AssertionError("Unexpected client"))
    monkeypatch.setattr(llm, "OpenAI", factory)
    result = MedicalGenerator(GenerationConfig(dry_run=True)).generate("Question", [])
    assert result.is_refusal
    factory.assert_not_called()


def test_planner_uses_shared_transport(router):
    result, requests, _ = router
    result["body"] = chat_response('{"action":"clarify","clarification_question":"Tên vắc xin?"}')
    assert QueryPlanner().plan("Loại này?").clarification_question == "Tên vắc xin?"
    body = json.loads(requests[0].content)
    assert body["model"] == "fixture/planner"
    assert body["response_format"] == {"type": "json_object"}


def test_reference_answer_and_span_refiner_use_shared_transport(router):
    result, requests, _ = router
    item = {"QUERY": "Question", "ANSWERABILITY": "ANSWERABLE", "RELEVANT_CHUNKS": [
        {"CHUNK_ID": "c1", "TEXT": "Evidence.", "RELEVANCE_SCORE": 3},
    ]}
    result["body"] = chat_response(json.dumps({
        "reference_answer": "Evidence [c1].", "claim_evidence_map": [
            {"claim": "Evidence", "chunk_id": "c1", "evidence_quote": "Evidence."},
        ],
    }))
    OpenRouterAnswerGenerator().generate_answers([item])
    assert item["REFERENCE_ANSWER"] == "Evidence [c1]."
    result["body"] = chat_response('[{"start":0,"end":8}]')
    no_chunks = {"QUERY": "Unanswerable", "RELEVANT_CHUNKS": []}
    OpenRouterRefiner().refine([no_chunks, item])
    assert item["RELEVANT_CHUNKS"][0]["EVIDENCE_SPANS"] == [{"START": 0, "END": 8}]
    assert no_chunks["RELEVANT_CHUNKS"] == []
    assert all(json.loads(r.content)["model"] == "fixture/generation" for r in requests)


@pytest.mark.parametrize("content", [
    "[]", '{"reference_answer":"","claim_evidence_map":[]}',
    '{"reference_answer":"Answer [bad]","claim_evidence_map":[]}',
])
def test_reference_answer_rejects_invalid_schema(router, content):
    result, _, _ = router
    result["body"] = chat_response(content)
    item = {"QUERY": "Question", "RELEVANT_CHUNKS": [
        {"CHUNK_ID": "c1", "TEXT": "Evidence.", "RELEVANCE_SCORE": 3},
    ]}
    with pytest.raises(llm.LLMError):
        OpenRouterAnswerGenerator().generate_answers([item])
    assert "REFERENCE_ANSWER" not in item


@pytest.mark.integration
def test_live_openrouter():
    """Explicit opt-in smoke check using process environment, never reading .env."""
    response = llm.completion(model=llm.resolve_model(), messages=[
        {"role": "user", "content": "Reply with the word OK."},
    ], max_tokens=16)
    assert response.choices[0].message.content.strip()
