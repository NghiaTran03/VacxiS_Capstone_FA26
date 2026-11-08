"""Shared OpenRouter chat transport; configuration comes from the process environment."""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

BASE_URL = "https://openrouter.ai/api/v1"


class LLMError(RuntimeError):
    """A safe error that never contains provider payloads or credentials."""


def resolve_model(model: str | None = None, *, planner: bool = False) -> str:
    """Resolve an explicit override before the component's environment variable."""
    variable = "VACXIS_QUERY_PLANNER_MODEL" if planner else "VACXIS_GENERATION_MODEL"
    value = model if model is not None else os.getenv(variable, "")
    if not isinstance(value, str) or not value.strip():
        raise LLMError(f"Set {variable} or pass an explicit model.")
    return value.strip()


def validate_config(model: str | None = None, *, planner: bool = False) -> str:
    """Validate online settings without creating a network client."""
    resolved = resolve_model(model, planner=planner)
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise LLMError("Set OPENROUTER_API_KEY for online LLM calls.")
    return resolved


def validate_response(response: Any) -> Any:
    """Reject absent, empty, refused, or truncated chat completions."""
    try:
        choice = response.choices[0]
        content = choice.message.content
        valid = (
            choice.finish_reason in (None, "stop")
            and isinstance(content, str) and bool(content.strip())
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        valid = False
    if not valid:
        raise LLMError("OpenRouter returned an invalid or incomplete response.")
    return response


def completion(*, model: str, messages: list[dict[str, str]], timeout: float = 30.0,
               **kwargs: Any) -> Any:
    """Make one bounded SDK request and close the client on every exit path."""
    resolved = validate_config(model)
    try:
        with OpenAI(base_url=BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"),
                    timeout=timeout, max_retries=0) as client:
            response = client.chat.completions.create(
                model=resolved, messages=messages, extra_body={}, **kwargs,
            )
    except Exception:
        raise LLMError("OpenRouter request failed; check credentials, service and timeout.") from None
    return validate_response(response)
