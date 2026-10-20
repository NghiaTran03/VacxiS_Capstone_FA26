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
import re
import sys
from typing import Iterable, Protocol

import litellm


SYSTEM_PROMPT = (
    "Bạn là một trợ lý trích xuất bằng chứng. Với mỗi mục được cung cấp, hãy trả về "
    "CHÍNH XÁC một phần tử JSON trong một mảng, theo đúng thứ tự. Mỗi phần tử có dạng "
    '{"start": int, "end": int} với [start, end) là chỉ số ký tự trong `text` của câu '
    "trả lời trực tiếp nhất cho câu hỏi. Khoảng phải là một câu con hợp lệ của `text`, "
    "không được trả về 0, không được trả về số âm. Nếu không tìm thấy, hãy chọn câu "
    "gần nhất. Trả về JSON hợp lệ, không kèm văn bản nào khác, không dùng markdown fence.\n\n"
    "Ví dụ:\n"
    "CÂU HỎI: Vaccine Gardasil được chỉ định cho những đối tượng nào?\n"
    "TEXT:\n[CHỦ ĐỀ: Gardasil] > [MỤC: CHỈ ĐỊNH]\nGARDASIL là vắc-xin được chỉ định cho "
    "bé gái và phụ nữ từ 9 đến 26 tuổi để phòng ngừa ung thư cổ tử cung.\n"
    "Đáp án đúng: [37, 130)  (khoảng này bao gồm cả tiêu đề MỤC: CHỈ ĐỊNH, miễn là "
    "nó nằm trong `text`)\n"
    'Trả về: [{"start": 37, "end": 130}]'
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
        batch_size: int = 4,
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
            )
            content = response.choices[0].message.content
            data = _parse_json_content(content)
            if isinstance(data, list):
                spans = data
            else:
                spans = data.get("spans") or data.get("results") or data
            if not isinstance(spans, list):
                raise ValueError(f"refiner: expected list, got {type(spans).__name__}")
            if len(spans) != len(payload):
                raise ValueError(
                    f"refiner: got {len(spans)} spans for {len(payload)} queries"
                )
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
    if not (isinstance(start, int) and isinstance(end, int)):
        return False
    if not (0 <= start < end <= len(text)):
        return False
    # Skip the synthetic ``[CHỦ ĐỀ: ...] > [MỤC: ...]`` header that every
    # chunk starts with. A span that begins at offset 0 (or right at the
    # first newline) almost always lands on this header rather than the
    # actual answer sentence.
    snippet = text[start:end]
    first_line = snippet.split("\n", 1)[0]
    if first_line.startswith("[CHỦ ĐỀ:") and "MỤC:" in first_line:
        return False
    return True


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_content(content: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = _FENCE_RE.sub("", content).strip()
    # Some gateways wrap the whole array without an outer key. Try the object
    # form first, then fall back to a list wrapper.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"spans": json.loads(text)}


def _chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
