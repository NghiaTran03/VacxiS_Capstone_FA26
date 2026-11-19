"""Exercise the web submission flow with mocked pipeline calls and browser state."""

from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
from vacxis_capstone_fa26.llm import LLMError
from vacxis_capstone_fa26.query import Clarification, ClarifyPlan, ConversationMessage
from vacxis_capstone_fa26.web import EMPTY_HISTORY, _active, _conversation_history, _progress, _submit, build_app


def saved_conversation():
    return {
        'active': 'selected',
        'conversations': {
            'selected': {'title': 'Verorab', 'pending': None, 'messages': [
                {'role': 'user', 'content': 'Bé 14 tháng hỏi về Verorab.'},
                _progress([('planning', None), ('generating', 1)], 'answered'),
                {'role': 'assistant', 'content': 'Thông tin Verorab.', 'sources': [
                    {'doc_id': 'doc', 'doc_title': 'Private source title', 'text': 'Private source content'},
                ]},
            ]},
            'other': {'title': 'Gardasil 9', 'pending': None, 'messages': [
                {'role': 'user', 'content': 'Mẹ 30 tuổi hỏi Gardasil 9.'},
                {'role': 'assistant', 'content': 'Thông tin Gardasil 9.'},
            ]},
        },
    }


def submit(text, saved, run_turn):
    return list(_submit(text, saved, run_turn))[-1]


def answer():
    return GeneratedAnswer('Resolved', 'Answer')


def test_snapshot_contains_only_selected_previous_text_without_sources_or_progress():
    saved = saved_conversation()
    original = deepcopy(saved)
    run_turn = Mock(return_value=answer())
    output = submit('Loại này có tác dụng phụ gì?', saved, run_turn)
    assert run_turn.call_args.args == ('Loại này có tác dụng phụ gì?',)
    assert run_turn.call_args.kwargs['history'] == (
        ConversationMessage('user', 'Bé 14 tháng hỏi về Verorab.'),
        ConversationMessage('assistant', 'Thông tin Verorab.'),
    )
    assert run_turn.call_args.kwargs['clarifications'] == []
    assert output[0] == output[1]
    assert _active(output[0])['pending'] is None
    assert saved == original
    assert output[0]['conversations']['other'] == saved['conversations']['other']


def test_conversation_selection_and_new_conversation_isolate_context():
    saved = saved_conversation()
    saved['active'] = 'other'
    run_turn = Mock(return_value=answer())
    submit('Lịch tiêm thế nào?', saved, run_turn)
    assert run_turn.call_args.kwargs['history'] == (
        ConversationMessage('user', 'Mẹ 30 tuổi hỏi Gardasil 9.'),
        ConversationMessage('assistant', 'Thông tin Gardasil 9.'),
    )
    saved['active'] = None
    output = submit('Câu hỏi mới', saved, run_turn)
    assert run_turn.call_args.kwargs['history'] == ()
    assert output[0]['active'] not in {'selected', 'other', None}
    assert _active(output[0])['messages'][0]['content'] == 'Câu hỏi mới'
    assert len(output[0]['conversations']) == 3


def test_clarification_reuses_pre_question_snapshot_until_answer():
    saved = saved_conversation()
    snapshot = _conversation_history(_active(saved)['messages'])
    run_turn = Mock(side_effect=[ClarifyPlan('Bạn hỏi bé hay mẹ?'), ClarifyPlan('Bạn hỏi liều nào?'), answer()])
    first = submit('Tiêm được không?', saved, run_turn)[0]
    pending = _active(first)['pending']
    assert pending == {
        'query': 'Tiêm được không?', 'question': 'Bạn hỏi bé hay mẹ?', 'clarifications': [],
        'history': [asdict(message) for message in snapshot],
    }
    second = submit('Bé', first, run_turn)[0]
    assert run_turn.call_args.args == ('Tiêm được không?',)
    assert run_turn.call_args.kwargs['history'] == snapshot
    assert run_turn.call_args.kwargs['clarifications'] == [Clarification('Bạn hỏi bé hay mẹ?', 'Bé')]
    assert _active(second)['pending']['history'] == pending['history']
    third = submit('Liều đầu', second, run_turn)[0]
    assert run_turn.call_args.kwargs['history'] == snapshot
    assert run_turn.call_args.kwargs['clarifications'] == [
        Clarification('Bạn hỏi bé hay mẹ?', 'Bé'), Clarification('Bạn hỏi liều nào?', 'Liều đầu'),
    ]
    assert _active(third)['pending'] is None
    assert run_turn.call_count == 3


def test_legacy_pending_without_snapshot_preserves_question_and_clarifications():
    saved = saved_conversation()
    _active(saved)['pending'] = {
        'query': 'Bé tiêm loại này được không?', 'question': 'Vắc xin nào?',
        'clarifications': [{'question': 'Bé bao nhiêu tuổi?', 'answer': '14 tháng'}],
    }
    run_turn = Mock(return_value=answer())
    output = submit('Verorab', saved, run_turn)
    assert run_turn.call_args.args == ('Bé tiêm loại này được không?',)
    assert run_turn.call_args.kwargs['history'] == ()
    assert run_turn.call_args.kwargs['clarifications'] == [
        Clarification('Bé bao nhiêu tuổi?', '14 tháng'), Clarification('Vắc xin nào?', 'Verorab'),
    ]
    assert _active(output[0])['pending'] is None


@pytest.mark.parametrize('error', [LLMError('provider failed'), RuntimeError('retrieval failed')])
def test_failed_request_does_not_commit_and_retry_does_not_duplicate(monkeypatch, error):
    monkeypatch.setattr('vacxis_capstone_fa26.web.gr.Warning', Mock())
    saved = saved_conversation()
    original = deepcopy(saved)

    def fail(query, **options):
        options['on_progress']('planning', None)
        options['on_progress']('generating', 1)
        raise error

    failed = submit('Follow-up', saved, fail)
    assert saved == original
    assert all(item == {'__type__': 'update'} for item in failed[:3])
    assert failed[4]['value'] == 'Follow-up'
    assert not any(message['content'] == 'Follow-up' for message in failed[3])
    assert not any(message.get('sources') for message in failed[3])
    run_turn = Mock(return_value=answer())
    succeeded = submit('Follow-up', saved, run_turn)[0]
    assert sum(message['content'] == 'Follow-up' for message in _active(succeeded)['messages']) == 1
    assert sum(message['content'] == 'Answer' for message in _active(succeeded)['messages']) == 1
    assert run_turn.call_args.kwargs['history'] == _conversation_history(_active(original)['messages'])


def test_failed_clarification_retry_retains_original_pending_snapshot(monkeypatch):
    monkeypatch.setattr('vacxis_capstone_fa26.web.gr.Warning', Mock())
    saved = submit('Tiêm được không?', saved_conversation(), Mock(return_value=ClarifyPlan('Vắc xin nào?')))[0]
    original = deepcopy(saved)
    failed = submit('Verorab', saved, Mock(side_effect=LLMError('failed')))
    assert saved == original
    assert failed[4]['value'] == 'Verorab'
    run_turn = Mock(return_value=answer())
    result = submit('Verorab', saved, run_turn)[0]
    assert run_turn.call_args.args == ('Tiêm được không?',)
    assert run_turn.call_args.kwargs['clarifications'] == [Clarification('Vắc xin nào?', 'Verorab')]
    assert _active(result)['pending'] is None
    assert sum(message['content'] == 'Verorab' for message in _active(result)['messages']) == 1


def test_new_chat_failure_does_not_persist_empty_conversation(monkeypatch):
    monkeypatch.setattr('vacxis_capstone_fa26.web.gr.Warning', Mock())
    saved = deepcopy(EMPTY_HISTORY)
    failed = submit('Q', saved, Mock(side_effect=RuntimeError('failed')))
    assert saved == EMPTY_HISTORY
    assert failed[4]['value'] == 'Q'
    result = submit('Q', saved, Mock(return_value=answer()))[0]
    assert len(result['conversations']) == 1
    assert len([message for message in _active(result)['messages'] if message['role'] == 'user']) == 1


def test_history_snapshot_has_no_ui_budget_or_metadata():
    messages = [{'role': 'user', 'content': 'Q' * 13_000}, {'role': 'assistant', 'content': 'A', 'sources': []}]
    assert _conversation_history(messages) == (ConversationMessage('user', 'Q' * 13_000), ConversationMessage('assistant', 'A'))


def test_app_still_builds_with_existing_browser_storage(monkeypatch):
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED', 'False')
    documents = Mock(choices=[], ids=set())
    documents.render.return_value = '<p>Choose a document</p>'
    app = build_app(Mock(return_value=answer()), documents)
    assert app is not None
