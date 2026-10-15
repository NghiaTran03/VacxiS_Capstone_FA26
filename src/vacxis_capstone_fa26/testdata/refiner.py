"""Optional LLM span refiner.

When ``--refine-spans`` is passed, every generated query's primary chunk
gets its EVIDENCE_SPAN overwritten with the span an LLM picks. Falls back
to the deterministic span on any per-query error and logs it.

The refiner talks to a provider-agnostic endpoint via LiteLLM. The default
model is whatever ``VACXIS_LLM_MODEL`` says (typically ``claude-haiku-4-5``),
and ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_API_KEY`` are read so a custom
gateway works out of the box. To switch providers, set ``VACXIS_LLM_MODEL``
to a LiteLLM-compatible model id (e.g. ``gpt-4o-mini``, ``ollama/llama3``).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterable, Protocol

import litellm


SYSTEM_PROMPT = (
    "Bạn là một trợ lý trích xuất bằng chứng. Với mỗi câu hỏi và đoạn văn, "
    "hãy trả về JSON hợp lệ duy nhất với khóa 'start' (int) và 'end' (int) "
    "là chỉ số ký tự trong `text` của câu trả lời trực tiếp nhất cho câu hỏi. "
    "Khoảng [start, end) phải là một câu con của `text` và phải nằm hoàn toàn "
    "trong `text`. Chỉ trả về JSON, không có văn bản nào khác."
)


class SpanRefiner(Protocol):
    def refine(self, queries: list[dict]) -> list[dict]: ...


class _NoOpRefiner:
    """Used when ``--refine-spans`` is not set."""

    def refine(self, queries: list[dict]) -> list[dict]:
        return queries


def noop_refiner() -> _NoOpRefiner:
    return _NoOpRefiner()


class LiteLLMRefiner:
    """Provider-agnostic span refiner.

    Reads configuration from env:

    * ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_BASE_URL`` (so a custom gateway
      works with the default Anthropic model)
    * ``VACXIS_LLM_MODEL`` overrides the default ``claude-haiku-4-5``
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 8,
        max_retries: int = 1,
    ) -> None:
        self.model = model or os.environ.get("VACXIS_LLM_MODEL", "claude-haiku-4-5")
        self.batch_size = batch_size
        self.max_retries = max_retries
        # Make litellm transparent about the configured gateway.
        self._print_config()

    def _print_config(self) -> None:
        base = os.environ.get("ANTHROPIC_BASE_URL", "<default>")
        print(
            f"[refiner] model={self.model} base_url={base}",
            file=sys.stderr,
        )

    def refine(self, queries: list[dict]) -> list[dict]:
        if not queries:
            return queries
        for batch in _chunks(queries, self.batch_size):
            self._refine_batch(batch)
        return queries

    def _refine_batch(self, batch: list[dict]) -> None:
        payload = [
            {
                "query": q["QUERY"],
                "chunk_id": q["RELEVANT_CHUNKS"][0]["CHUNK_ID"],
                "text": q["RELEVANT_CHUNKS"][0]["TEXT"],
            }
            for q in batch
            if q.get("RELEVANT_CHUNKS")
        ]
        if not payload:
            return
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Trả về một mảng JSON, một phần tử cho mỗi mục dưới đây, "
                            "theo đúng thứ tự. Mỗi phần tử là {\"start\": int, \"end\": int}.\n\n"
                            + "\n\n---\n\n".join(
                                f"CÂU HỎI: {p['query']}\nCHUNK_ID: {p['chunk_id']}\n"
                                f"TEXT:\n{p['text']}"
                                for p in payload
                            )
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            spans = data.get("spans") or data.get("results") or data
            if isinstance(spans, dict):
                spans = list(spans.values())
            for q, p, sp in zip(batch, payload, spans):
                if not isinstance(sp, dict):
                    continue
                start, end = sp.get("start"), sp.get("end")
                if not _valid_span(start, end, p["text"]):
                    print(
                        f"[refiner] dropping invalid span for {p['chunk_id']}: {sp}",
                        file=sys.stderr,
                    )
                    continue
                q["RELEVANT_CHUNKS"][0]["EVIDENCE_SPANS"] = [{"START": start, "END": end}]
        except Exception as exc:  # noqa: BLE001
            print(f"[refiner] batch failed, keeping deterministic spans: {exc}", file=sys.stderr)


def _valid_span(start: object, end: object, text: str) -> bool:
    return (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    )


def _chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
