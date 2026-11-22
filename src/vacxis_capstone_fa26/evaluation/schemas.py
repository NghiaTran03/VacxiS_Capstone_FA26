"""Result schemas for the Jev-backed DeepEval evaluation module.

The module delegates decision scoring to Jev through DeepEval. These
dataclasses wrap DeepEval's ``EvaluationResult`` in project-native shapes: a per-query record
that carries every metric's score/reason/success flag, and an aggregated
summary with per-metric statistics and breakdowns by stratification axis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryEvalResult:
    """Per-query evaluation outcome carrying every metric's result.

    ``metric_scores`` maps a metric display name (e.g. ``"Answer Relevancy"``)
    to its 0-1 score. ``metric_results`` maps the same name to the full
    DeepEval ``MetricData`` dict (score, success, reason, error, ...).
    """

    query_id: str
    query: str
    actual_output: str = ""
    expected_output: str = ""
    retrieval_context: list[str] = field(default_factory=list)
    is_refusal: bool = False
    answerability: str = ""
    metric_scores: dict[str, float] = field(default_factory=dict)
    metric_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the per-query result to a plain dictionary."""
        return {
            "query_id": self.query_id,
            "query": self.query,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "retrieval_context": self.retrieval_context,
            "is_refusal": self.is_refusal,
            "answerability": self.answerability,
            "metric_scores": self.metric_scores,
            "metric_results": self.metric_results,
            "metadata": self.metadata,
        }


@dataclass
class MetricAggregate:
    """Mean/min/max/count over a metric's scores across evaluated queries."""

    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0
    pass_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize the aggregate to a rounded dictionary."""
        return {
            "mean": round(self.mean, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "count": self.count,
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class EvaluationSummary:
    """Aggregated DeepEval report across all queries and subsets.

    ``metric_aggregates`` holds overall statistics keyed by metric name;
    ``by_*`` breakdowns hold the mean score per metric for each value of a
    stratification axis (answerability, complexity, query_style, difficulty).
    """

    task: str = "rag"
    num_queries: int = 0
    metric_names: list[str] = field(default_factory=list)
    overall_metrics: dict[str, float] = field(default_factory=dict)
    metric_aggregates: dict[str, MetricAggregate] = field(default_factory=dict)
    by_answerability: dict[str, dict[str, float]] = field(default_factory=dict)
    by_complexity: dict[str, dict[str, float]] = field(default_factory=dict)
    by_query_style: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete summary to a JSON-safe dictionary."""
        return {
            "task": self.task,
            "num_queries": self.num_queries,
            "metric_names": self.metric_names,
            "overall_metrics": {k: round(v, 4) for k, v in self.overall_metrics.items()},
            "metric_aggregates": {k: v.to_dict() for k, v in self.metric_aggregates.items()},
            "by_answerability": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_answerability.items()
            },
            "by_complexity": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_complexity.items()
            },
            "by_query_style": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_query_style.items()
            },
            "by_difficulty": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in self.by_difficulty.items()
            },
            "metadata": self.metadata,
            "per_query_results": self.per_query_results,
        }

    def render_markdown(self) -> str:
        """Render a clean Markdown report with per-metric stats and breakdowns."""
        lines: list[str] = [
            f"# VacxiS DeepEval Report: {self.task.upper()}",
            f"Total evaluated queries: {self.num_queries}",
            "",
            "## Overall Metrics",
            "| Metric | Mean | Min | Max | Pass Rate | Count |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for name in self.metric_names:
            agg = self.metric_aggregates.get(name)
            if agg is None:
                continue
            lines.append(
                f"| {name} | {agg.mean:.4f} | {agg.min:.4f} | {agg.max:.4f} "
                f"| {agg.pass_rate:.2%} | {agg.count} |"
            )

        if self.task == "retrieval":
            lines += ["", "IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.",
                      f"Pass rate is the fraction of scored queries meeting the configured threshold ({self.metadata.get('threshold', 0.5)}); it is not a retrieval accuracy measure."]
            if "scored_queries" in self.metadata:
                lines.append(f"Scored queries: {self.metadata['scored_queries']}; excluded without relevance labels: {self.metadata['unscored_queries']}.")
            for warning in self.metadata.get("warnings", []):
                lines.append(f"Warning: {warning}")
            if "run" in self.metadata:
                lines += ["", "## Run Settings"]
                for key, value in self.metadata["run"].items():
                    lines.append(f"- {key}: {value}")

        breakdown_sections = [
            ("Breakdown by Answerability", self.by_answerability, "Answerability"),
            ("Breakdown by Complexity", self.by_complexity, "Complexity"),
            ("Breakdown by Query Style", self.by_query_style, "Query Style"),
            ("Breakdown by Difficulty", self.by_difficulty, "Difficulty"),
        ]
        for heading, table, label in breakdown_sections:
            if not table:
                continue
            lines += ["", f"## {heading}", self._breakdown_table(label, table)]
        return "\n".join(lines)

    @staticmethod
    def _breakdown_table(
        label: str,
        table: dict[str, dict[str, float]],
    ) -> str:
        """Render one breakdown axis as a Markdown table row per group value."""
        metric_names: list[str] = []
        for values in table.values():
            for key in values:
                if key != "count" and key not in metric_names:
                    metric_names.append(key)
        header = [label, "Count", *metric_names]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join([" :--- "] * len(header)) + " |",
        ]
        for group, values in sorted(table.items()):
            row = [str(group), str(int(values.get("count", 0)))]
            row += [f"{values.get(m, 0.0):.4f}" for m in metric_names]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
