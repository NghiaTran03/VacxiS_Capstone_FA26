"""Contracts shared by the query planner and its callers."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Clarification:
    question: str
    answer: str


@dataclass(frozen=True)
class RetrievePlan:
    original_query: str
    search_queries: tuple[str, ...]
    action: Literal["retrieve"] = field(default="retrieve", init=False)


@dataclass(frozen=True)
class ClarifyPlan:
    clarification_question: str
    action: Literal["clarify"] = field(default="clarify", init=False)


type QueryPlan = RetrievePlan | ClarifyPlan
