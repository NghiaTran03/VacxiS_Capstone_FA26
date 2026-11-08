"""Optional OpenRouter evidence span refinement with strict online errors."""
from __future__ import annotations

import json
import re
from typing import Iterable, Protocol

from vacxis_capstone_fa26.llm import LLMError, completion, resolve_model


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


class OpenRouterRefiner:
    """Refine evidence spans with the shared generation model."""

    def __init__(self, model: str | None = None, batch_size: int = 4) -> None:
        self.model = resolve_model(model)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.batch_size = batch_size

    def refine(self, queries: list[dict]) -> list[dict]:
        if not queries:
            return queries
        for batch in _chunks(queries, self.batch_size):
            self._refine_batch(batch)
        return queries

    def _refine_batch(self, batch: list[dict]) -> None:
        eligible = [q for q in batch if q.get("RELEVANT_CHUNKS")]
        payload = [
            {
                "query": q["QUERY"],
                "chunk_id": q["RELEVANT_CHUNKS"][0]["CHUNK_ID"],
                "text": q["RELEVANT_CHUNKS"][0]["TEXT"],
            }
            for q in eligible
        ]
        if not payload:
            return
        response = completion(
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
        try:
            data = _parse_json_content(content)
        except (ValueError, TypeError, AttributeError):
            raise LLMError("Span refiner returned invalid JSON.") from None
        if isinstance(data, list):
            spans = data
        elif isinstance(data, dict):
            spans = data.get("spans") or data.get("results") or data
        else:
            raise LLMError("Span response must contain an array.")
        if not isinstance(spans, list):
            raise ValueError(f"refiner: expected list, got {type(spans).__name__}")
        if len(spans) != len(payload):
            raise ValueError(
                f"refiner: got {len(spans)} spans for {len(payload)} queries"
            )
        for q, p, sp in zip(eligible, payload, spans):
            if not isinstance(sp, dict):
                raise LLMError("Span response must contain objects.")
            start, end = sp.get("start"), sp.get("end")
            if not _valid_span(start, end, p["text"]):
                raise LLMError("Span response contains invalid evidence offsets.")
            q["RELEVANT_CHUNKS"][0]["EVIDENCE_SPANS"] = [{"START": start, "END": end}]


def _valid_span(start: object, end: object, text: str) -> bool:
    if not (type(start) is int and type(end) is int):
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
