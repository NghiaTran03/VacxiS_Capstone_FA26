"""Contracts shared by the query planner and its callers."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ConversationMessage:
    """Plain conversation text used only to resolve the current question."""

    role: Literal["user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError("Conversation role must be user or assistant.")
        if not isinstance(self.content, str):
            raise TypeError("Conversation content must be a string.")
        if not self.content.strip():
            raise ValueError("Conversation content must not be blank.")


@dataclass(frozen=True)
class ConversationMemoryConfig:
    """Whole-message limits for the recent conversation suffix."""

    max_messages: int = 12
    max_characters: int = 12_000

    def __post_init__(self) -> None:
        for value in (self.max_messages, self.max_characters):
            if type(value) is not int or value < 1:
                raise ValueError("Memory limits must be positive integers.")


@dataclass(frozen=True)
class Clarification:
    question: str
    answer: str


@dataclass(frozen=True)
class RetrievePlan:
    original_query: str
    search_queries: tuple[str, ...]
    resolved_query: str | None = None
    action: Literal["retrieve"] = field(default="retrieve", init=False)

    def __post_init__(self) -> None:
        if self.resolved_query is None:
            object.__setattr__(self, "resolved_query", self.original_query)
        if not isinstance(self.resolved_query, str) or not self.resolved_query.strip():
            raise ValueError("Resolved query must not be blank.")


@dataclass(frozen=True)
class ClarifyPlan:
    clarification_question: str
    action: Literal["clarify"] = field(default="clarify", init=False)


type QueryPlan = RetrievePlan | ClarifyPlan
