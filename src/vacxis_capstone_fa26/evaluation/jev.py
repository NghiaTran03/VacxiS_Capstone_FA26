"""Configuration helpers for Jev-based DeepEval metrics."""
from __future__ import annotations

import os

JEV_API_KEY_ENV = "OPENROUTER_API_KEY"
JEV_MODEL_ENV = "VACXIS_JEV_MODEL"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
TYPESAFE_BASE_URL_ENV = "TYPESAFE_BASE_URL"
_OPENROUTER_SYSTEM_ONE_BASE_URL = "https://openrouter.ai/api"
_DEFAULT_JEV_MODEL = "~typesafe/jev-latest"


def resolve_jev_model(model: str | None = None) -> str:
    """Resolve the Jev model from an argument, environment, or default."""
    explicit = (model or "").strip()
    if explicit:
        return explicit
    configured = os.getenv(JEV_MODEL_ENV, "").strip()
    return configured or _DEFAULT_JEV_MODEL


def configure_jev_openrouter() -> None:
    """Configure DeepEval's System One client to use Jev through OpenRouter."""
    api_key = os.getenv(JEV_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"Set {JEV_API_KEY_ENV} to run Jev evaluation through OpenRouter."
        )
    os.environ[TYPESAFE_API_KEY_ENV] = api_key
    os.environ[TYPESAFE_BASE_URL_ENV] = _OPENROUTER_SYSTEM_ONE_BASE_URL


def require_jev_credentials() -> None:
    """Compatibility alias that validates and configures OpenRouter access."""
    configure_jev_openrouter()
