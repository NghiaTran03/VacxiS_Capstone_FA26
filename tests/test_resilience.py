"""Offline retry and checkpoint behavior, without retrieval models."""
import asyncio
from unittest.mock import Mock
import httpx
import pytest
from openai import APIConnectionError, PermissionDeniedError
from vacxis_capstone_fa26.resilience import RetryPolicy, atomic_json


def test_transient_retry_and_exhaustion(monkeypatch):
    waits = []
    monkeypatch.setattr('vacxis_capstone_fa26.resilience.time.sleep', waits.append)
    error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
    call = Mock(side_effect=[error, error, 'ok'])
    assert RetryPolicy(2, 1, 2).run(call) == 'ok'
    assert len(waits) == 2
    assert 0.8 <= waits[0] <= 1
    assert 1.6 <= waits[1] <= 2
    with pytest.raises(APIConnectionError):
        RetryPolicy(0).run(Mock(side_effect=error))


def test_permanent_error_not_retried():
    response = httpx.Response(403, request=httpx.Request('POST', 'https://example.com'))
    call = Mock(side_effect=PermissionDeniedError('blocked', response=response, body=None))
    with pytest.raises(PermissionDeniedError):
        RetryPolicy().run(call)
    assert call.call_count == 1


def test_async_retry(monkeypatch):
    from unittest.mock import AsyncMock
    sleep = AsyncMock()
    monkeypatch.setattr('vacxis_capstone_fa26.resilience.asyncio.sleep', sleep)
    error = APIConnectionError(request=httpx.Request('POST', 'https://example.com'))
    call = AsyncMock(side_effect=[error, 'ok'])
    assert asyncio.run(RetryPolicy(1).arun(call)) == 'ok'
    sleep.assert_awaited_once()


def test_atomic_checkpoint(tmp_path):
    target = tmp_path / 'nested' / 'report.json'
    atomic_json(target, {'complete': False})
    atomic_json(target, {'complete': True})
    assert 'true' in target.read_text()
    assert not target.with_suffix('.json.tmp').exists()
