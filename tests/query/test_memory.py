"""Mocked memory contracts and orchestration; these do not score planner quality."""

from dataclasses import FrozenInstanceError, asdict
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query import (
    Clarification, ClarifyPlan, ConversationMemoryConfig, ConversationMessage, QueryPlanner, RetrievePlan,
)
from vacxis_capstone_fa26.query.memory import limit_history


def completion_for(decision):
    return Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(decision, ensure_ascii=False)), finish_reason="stop",
    )]))


def retrieve_decision(resolved, expansions=()):
    return {"action": "retrieve", "resolved_query": resolved, "additional_queries": list(expansions)}


def request_payload(complete):
    return json.loads(complete.call_args.kwargs["messages"][1]["content"])


@pytest.mark.parametrize(("history", "question", "resolved"), [
    ([], "Verorab là gì?", "Verorab là gì?"),
    ([('user', 'Tôi hỏi về Verorab.'), ('assistant', 'Bạn muốn biết gì về Verorab?')],
     'Loại này có tác dụng phụ gì?', 'Verorab có tác dụng phụ gì?'),
    ([('user', 'Bé 14 tháng đang hỏi Verorab.')],
     'Bé tiêm được không?', 'Trẻ 14 tháng có thể tiêm Verorab không?'),
    ([('user', 'Bé 14 tháng, mẹ 30 tuổi. Tôi hỏi Gardasil 9 cho mẹ.')],
     'Sửa lại: mẹ 31 tuổi. Lịch tiêm của mẹ thế nào?', 'Lịch tiêm Gardasil 9 cho mẹ 31 tuổi thế nào?'),
    ([('user', 'Bé dùng Vaxigrip 0,25 ml, mẹ dùng Vaxigrip 0,5 ml ngày 10/11/2026.')],
     'Mẹ cần liều tiếp theo khi nào?', 'Mẹ tiêm Vaxigrip 0,5 ml ngày 10/11/2026 cần liều tiếp theo khi nào?'),
    ([('user', 'Nếu tôi dị ứng trứng thì sao?'), ('assistant', 'Bạn đang hỏi vắc xin nào?')],
     'Tôi không dị ứng trứng, hỏi về Verorab.', 'Tôi không dị ứng trứng, có thể tiêm Verorab không?'),
    ([('user', 'Bé 14 tháng hỏi Verorab.')],
     'Gardasil 9 bảo quản thế nào?', 'Gardasil 9 bảo quản thế nào?'),
])
def test_resolution_contract_preserves_history_and_uses_one_call(history, question, resolved):
    messages = tuple(ConversationMessage(*message) for message in history)
    complete = completion_for(retrieve_decision(resolved))
    plan = QueryPlanner('fixture', completion=complete).plan(question, history=messages)
    assert plan == RetrievePlan(question, (resolved,), resolved)
    assert request_payload(complete) == {
        'original_query': question, 'clarifications': [], 'history': [asdict(message) for message in messages],
    }
    complete.assert_called_once()


def test_ambiguous_reference_clarifies_without_downstream_calls():
    complete = completion_for({'action': 'clarify', 'clarification_question': 'Bạn hỏi Verorab hay Gardasil 9?'})
    retriever, generator = Mock(), Mock()
    planner = QueryPlanner('fixture', completion=complete)
    result = VacxisRAGPipeline(retriever, generator, query_planner=planner).run_turn(
        'Loại này thì sao?', history=[ConversationMessage('user', 'So sánh Verorab và Gardasil 9.')],
    )
    assert isinstance(result, ClarifyPlan)
    assert retriever.mock_calls == generator.mock_calls == []


@pytest.mark.parametrize('bad', [0, -1, True, 1.5, '12', None])
@pytest.mark.parametrize('field', ['max_messages', 'max_characters'])
def test_limits_must_be_positive_integers(field, bad):
    with pytest.raises(ValueError):
        ConversationMemoryConfig(**{field: bad})


@pytest.mark.parametrize(('role', 'content', 'error'), [
    ('system', 'text', ValueError), ('tool', 'text', ValueError),
    ('user', '', ValueError), ('assistant', ' \t', ValueError), ('user', None, TypeError),
])
def test_invalid_message(role, content, error):
    with pytest.raises(error):
        ConversationMessage(role, content)


def test_contracts_are_immutable_and_old_retrieve_plan_is_supported():
    with pytest.raises(FrozenInstanceError):
        ConversationMessage('user', 'Q').content = 'Changed'
    with pytest.raises(FrozenInstanceError):
        ConversationMemoryConfig().max_messages = 5
    assert RetrievePlan('Original', ('Original',)).resolved_query == 'Original'


@pytest.mark.parametrize(('contents', 'config', 'expected'), [
    (['aa', 'bb', 'cc', 'dd'], ConversationMemoryConfig(max_messages=3), ['cc', 'dd']),
    (['a', 'b', 'cc', 'dd'], ConversationMemoryConfig(max_characters=4), ['cc', 'dd']),
    (['a', 'b', 'oversized', 'd'], ConversationMemoryConfig(max_characters=5), []),
    (['a', 'b', 'c', 'oversized'], ConversationMemoryConfig(max_characters=5), []),
    (['aa', 'bb'], ConversationMemoryConfig(max_characters=4), ['aa', 'bb']),
])
def test_history_is_contiguous_whole_message_suffix(contents, config, expected):
    history = tuple(ConversationMessage('user' if i % 2 == 0 else 'assistant', text)
                    for i, text in enumerate(contents))
    selected = limit_history(history, config)
    assert [message.content for message in selected] == expected
    assert [message.content for message in history] == contents


def test_default_limits_and_leading_assistant_removal():
    assert ConversationMemoryConfig() == ConversationMemoryConfig(12, 12_000)
    history = tuple(ConversationMessage('user' if i % 2 == 0 else 'assistant', str(i)) for i in range(16))
    assert limit_history(history, ConversationMemoryConfig()) == history[-12:]
    assert limit_history([ConversationMessage('assistant', 'orphan')], ConversationMemoryConfig()) == ()
    assert limit_history([ConversationMessage('user', 'x' * 12_001)], ConversationMemoryConfig()) == ()


def test_invalid_history_is_rejected_before_provider_even_if_outside_budget():
    complete = Mock()
    planner = QueryPlanner('fixture', completion=complete, memory_config=ConversationMemoryConfig(max_messages=1))
    with pytest.raises(TypeError, match='ConversationMessage'):
        planner.plan('Q', history=[{'role': 'user', 'content': 'bad type'}, ConversationMessage('user', 'Q')])
    complete.assert_not_called()


def test_planner_applies_limits_before_request_and_does_not_retain_history():
    complete = completion_for(retrieve_decision('Q'))
    planner = QueryPlanner('fixture', completion=complete, memory_config=ConversationMemoryConfig(max_messages=2))
    history = [ConversationMessage('user', 'Old'), ConversationMessage('assistant', 'Reply'),
               ConversationMessage('user', 'Recent'), ConversationMessage('assistant', 'Recent reply')]
    planner.plan('Q', history=history)
    assert request_payload(complete)['history'] == [asdict(message) for message in history[-2:]]
    planner.plan('Q')
    assert request_payload(complete)['history'] == []


def test_resolved_expansions_are_unique_and_never_add_unresolved_original():
    complete = completion_for(retrieve_decision('  Verorab là gì? ', [' Verorab là gì? ', ' Hướng dẫn Verorab ']))
    plan = QueryPlanner('fixture', completion=complete).plan('Loại này là gì?')
    assert plan.search_queries == ('Verorab là gì?', 'Hướng dẫn Verorab')


def test_resolved_question_reaches_retrieval_generation_and_metadata():
    resolved = 'Verorab có phản ứng phụ gì?'
    complete = completion_for(retrieve_decision(resolved))
    config = ConversationMemoryConfig(max_messages=2)
    planner = QueryPlanner('fixture', completion=complete, memory_config=config)
    retriever, generator = Mock(), Mock()
    retriever.retrieve.return_value = ['fresh evidence']
    generated = GeneratedAnswer(resolved, 'Answer')
    generator.generate.return_value = generated
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    history = [ConversationMessage('user', 'Private old text'), ConversationMessage('assistant', 'Old'),
               ConversationMessage('user', 'Verorab'), ConversationMessage('assistant', 'Private reply')]
    result = pipeline.run_turn('Loại này có phản ứng phụ gì?', history=history)
    retriever.retrieve.assert_called_once_with(query=resolved, top_k=None, skip_rerank=None)
    generator.generate.assert_called_once_with(query=resolved, chunks=['fresh evidence'])
    assert result.query == 'Loại này có phản ứng phụ gì?'
    assert generated.query == resolved and generated.query_context is None
    context = result.to_dict()['query_context']
    assert context == {
        'resolved_query': resolved, 'search_queries': [resolved], 'use_memory': True,
        'history_messages_supplied': 4, 'history_messages_included': 2, 'history_reduced': True,
        'memory_limits': asdict(config), 'planner_model': 'fixture',
        'planner_latency_seconds': context['planner_latency_seconds'],
    }
    assert context['planner_latency_seconds'] >= 0
    assert 'Private' not in json.dumps(context)
    pipeline.run_turn('Loại này có phản ứng phụ gì?', history=history)
    assert retriever.retrieve.call_count == 2


def test_memory_off_keeps_current_clarifications_and_clears_history():
    complete = completion_for(retrieve_decision('Verorab cho trẻ 14 tháng?'))
    retriever, generator = Mock(), Mock()
    generator.generate.return_value = GeneratedAnswer('Resolved', 'Answer')
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=QueryPlanner('fixture', completion=complete))
    clarification = Clarification('Vắc xin nào?', 'Verorab')
    result = pipeline.run_turn('Bé 14 tháng tiêm loại này được không?', history=[ConversationMessage('user', 'Old')],
                               use_memory=False, clarifications=[clarification])
    payload = request_payload(complete)
    assert payload['history'] == []
    assert payload['clarifications'] == [asdict(clarification)]
    assert result.query_context['use_memory'] is False
    assert result.query_context['history_messages_supplied'] == 1
    assert result.query_context['history_messages_included'] == 0
    assert result.query_context['history_reduced'] is False
    assert generator.generate.call_args.kwargs['query'] == 'Verorab cho trẻ 14 tháng?'


@pytest.mark.parametrize('bad', ['not JSON', '{}', '{"action":"retrieve","resolved_query":" ","additional_queries":[]}'])
def test_parsing_failure_prevents_downstream_calls(bad):
    complete = completion_for({})
    complete.return_value.choices[0].message.content = bad
    retriever, generator = Mock(), Mock()
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=QueryPlanner('fixture', completion=complete))
    with pytest.raises(LLMError):
        pipeline.run_turn('Q')
    assert retriever.mock_calls == generator.mock_calls == []


@pytest.mark.parametrize(('finish_reason', 'refusal'), [
    ('length', None), ('content_filter', None), (None, None), ('stop', 'Refused'),
])
def test_incomplete_or_refused_valid_json_plan_prevents_downstream_calls(finish_reason, refusal):
    complete = completion_for(retrieve_decision('Verorab là gì?'))
    choice = complete.return_value.choices[0]
    choice.finish_reason = finish_reason
    choice.message.refusal = refusal
    retriever, generator = Mock(), Mock()
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=QueryPlanner('fixture', completion=complete))
    with pytest.raises(LLMError, match='invalid plan'):
        pipeline.run_turn('Loại này là gì?', history=[ConversationMessage('user', 'Verorab')])
    complete.assert_called_once()
    assert retriever.mock_calls == generator.mock_calls == []


def test_single_turn_answer_and_batch_remain_planner_free_without_query_context():
    retriever, generator, planner = Mock(), Mock(), Mock()
    generator.generate.return_value = GeneratedAnswer('Q', 'Answer')
    generator.generate_batch.return_value = [GeneratedAnswer('Q', 'Answer')]
    pipeline = VacxisRAGPipeline(retriever, generator, query_planner=planner)
    assert 'query_context' not in pipeline.answer('Q', override_chunks=[]).to_dict()
    assert 'query_context' not in pipeline.answer_batch(['Q'], override_chunks_per_query=[[]])[0].to_dict()
    planner.plan.assert_not_called()


def test_empty_query_context_is_not_serialized():
    assert 'query_context' not in GeneratedAnswer('Q', 'Answer', query_context={}).to_dict()
