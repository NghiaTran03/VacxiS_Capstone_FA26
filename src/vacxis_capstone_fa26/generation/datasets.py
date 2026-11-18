from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class DatasetItem:
    query: str
    expected_output: str = ""
    chunks: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    original: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthChunk:
    chunk_id: str
    text: str
    doc_title: str = ""
    category: str = ""
    header: str = ""
    score: float = 0.0


def load_dataset(
    path: Path,
    dataset_format: str = "deepeval",
    limit: int | None = None,
) -> list[DatasetItem]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if dataset_format == "deepeval":
        if not isinstance(data, dict):
            raise ValueError(
                "DeepEval dataset must be a JSON object."
            )

        records = data.get("goldens", [])

        if not isinstance(records, list):
            raise ValueError("'goldens' must be a list.")

        items = [
            DatasetItem(
                query=record.get("input", ""),
                expected_output=record.get("expected_output", ""),
                chunks=[
                    GroundTruthChunk(
                        chunk_id=(
                            record.get("additional_metadata", {})
                            .get("source_chunk_ids", [])[i]
                            if i < len(
                                record.get("additional_metadata", {})
                                .get("source_chunk_ids", [])
                            )
                            else ""
                        ),
                        text=context,
                        score=0.0,
                    )
                    for i, context in enumerate(
                        record.get("context") or []
                    )
                    if isinstance(context, str)
                ],
                metadata=record.get("additional_metadata") or {},
                original=record,
            )
            for record in records
        ]

    elif dataset_format == "legacy":
        if not isinstance(data, list):
            raise ValueError(
                "Legacy dataset must be a JSON list."
            )

        items = [
            DatasetItem(
                query=record.get("QUERY", ""),
                expected_output=record.get("EXPECTED_OUTPUT", ""),
                chunks=[
                    GroundTruthChunk(
                        chunk_id=c.get("CHUNK_ID", ""),
                        text=c.get("TEXT", ""),
                        score=float(c.get("RELEVANCE_SCORE") or 0.0),
                    )
                    for c in record.get("RELEVANT_CHUNKS", [])
                    if isinstance(c, dict)
                ],
                metadata=record.get("ENTITIES") or {},
                original=record,
            )
            for record in data
        ]

    else:
        raise ValueError(
            f"Unsupported dataset format: {dataset_format}"
        )

    if limit is not None:
        items = items[:limit]

    return items
