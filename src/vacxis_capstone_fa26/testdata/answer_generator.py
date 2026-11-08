"""Reference answer generator and claim-evidence mapper for Vaccine RAG.

Supports two operational modes:
1. LLM-based generation via OpenRouter (explicit online mode)
2. Extractive grounded fallback (offline, deterministic, dry-run friendly)

Both modes generate:
- ``REFERENCE_ANSWER``: concise, factual Vietnamese medical answers with inline
  bracketed citations e.g. ``[CHUNK_ID]``.
- ``CLAIM_EVIDENCE_MAP``: atomic claims mapped to the specific chunk ID and quote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol

from vacxis_capstone_fa26.llm import LLMError, completion, resolve_model


UNANSWERABLE_FALLBACK_ANSWER = (
    "Tài liệu hướng dẫn sử dụng không cung cấp thông tin cho câu hỏi này. "
    "Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại các cơ sở tiêm chủng để được tư vấn chính xác."
)


REFERENCE_ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reference_answer", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "reference_answer": {"type": "string", "description": "Vietnamese answer grounded in provided chunks, with [CHUNK_ID] citations."},
                "claim_evidence_map": {
                    "type": "array", "description": "Claims and their source evidence; empty for refusal.",
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "properties": {
                            "claim": {"type": "string", "description": "One factual claim in the answer."},
                            "chunk_id": {"type": "string", "description": "Source chunk identifier."},
                            "evidence_quote": {"type": "string", "description": "Verbatim evidence from the source chunk."},
                        },
                        "required": ["claim", "chunk_id", "evidence_quote"],
                    },
                },
            },
            "required": ["reference_answer", "claim_evidence_map"],
        },
    },
}

SYSTEM_PROMPT = """Bạn tổng hợp câu trả lời tham chiếu tiếng Việt từ CHUNKS được cung cấp.
Chỉ sử dụng thông tin trong CHUNKS, không tự suy diễn. Giữ nguyên phạm vi câu hỏi.
Sau mỗi khẳng định, chèn [CHUNK_ID] và cung cấp claim cùng evidence_quote nguyên văn.
ANSWERABILITY là nhãn đầu vào, không được thay đổi. Với PARTIALLY_ANSWERABLE,
chỉ trả lời phần có bằng chứng và nêu rõ phần thiếu. Nếu không thể trả lời,
trả lời rằng tài liệu không cung cấp thông tin và để claim_evidence_map rỗng.
Trả JSON theo schema được cung cấp, không dùng Markdown."""


class AnswerGenerator(Protocol):
    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class NoOpAnswerGenerator:
    """Pass-through generator that leaves REFERENCE_ANSWER unchanged."""

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return queries


class ExtractiveAnswerGenerator:
    """Offline deterministic answer generator using evidence spans."""

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for q in queries:
            self._generate_single(q)
        return queries

    def _generate_single(self, q: dict[str, Any]) -> None:
        answerability = q.get("ANSWERABILITY")
        if answerability == "UNANSWERABLE":
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        relevant = [
            c for c in q.get("RELEVANT_CHUNKS", [])
            if c.get("RELEVANCE_SCORE", 0) >= 3
        ]
        if not relevant:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        claims: list[dict[str, str]] = []
        answer_sentences: list[str] = []

        for c in relevant:
            cid = c.get("CHUNK_ID", "")
            text = c.get("TEXT", "")
            spans = c.get("EVIDENCE_SPANS") or []

            evidence_text = ""
            if spans:
                s, e = spans[0].get("START", 0), spans[0].get("END", 0)
                if 0 <= s < e <= len(text):
                    evidence_text = text[s:e].strip()

            if not evidence_text:
                # Fallback to the first non-header line
                lines = [
                    ln.strip() for ln in text.splitlines()
                    if ln.strip() and not ln.strip().startswith("[CHỦ ĐỀ:")
                ]
                evidence_text = lines[0] if lines else text[:120].strip()

            # Clean leading/trailing artifacts
            evidence_text = evidence_text.lstrip("\n- *").strip()
            if evidence_text:
                answer_sentences.append(f"{evidence_text} [{cid}]")
                claims.append({
                    "claim": evidence_text,
                    "chunk_id": cid,
                    "evidence_quote": evidence_text,
                })

        if answer_sentences:
            prefix = "Theo hướng dẫn sử dụng: "
            q["REFERENCE_ANSWER"] = prefix + " ".join(answer_sentences)
            q["CLAIM_EVIDENCE_MAP"] = claims
        else:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []


class OpenRouterAnswerGenerator:
    """Generates golden reference answers with citations and claim maps via OpenRouter."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.model = resolve_model(model)
        self.batch_size = batch_size
        self.temperature = temperature

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for q in queries:
            try:
                self._generate_single(q)
            except LLMError as exc:
                raise LLMError(f"reference_answer QUERY_ID={q.get('QUERY_ID', '<missing>')} model={self.model}: {exc}") from None
            except (ValueError, TypeError, KeyError, AttributeError):
                raise LLMError(f"reference_answer QUERY_ID={q.get('QUERY_ID', '<missing>')} model={self.model}: invalid structured response.") from None
        return queries

    def _generate_single(self, q: dict[str, Any]) -> None:
        answerability = q.get("ANSWERABILITY")
        if answerability == "UNANSWERABLE":
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        relevant = [
            c for c in q.get("RELEVANT_CHUNKS", [])
            if c.get("RELEVANCE_SCORE", 0) >= 3
        ]
        if not relevant:
            q["REFERENCE_ANSWER"] = UNANSWERABLE_FALLBACK_ANSWER
            q["CLAIM_EVIDENCE_MAP"] = []
            return

        chunks_context = "\n\n---\n\n".join(
            f"CHUNK_ID: {c.get('CHUNK_ID')}\nNỘI DUNG:\n{c.get('TEXT')}"
            for c in relevant
        )

        user_content = (
            f"ANSWERABILITY: {answerability}\nCÂU HỎI: {q.get('QUERY')}\n\n"
            f"CÁC ĐOẠN CHUNKS CUNG CẤP:\n{chunks_context}\n\n"
            "Hãy tổng hợp câu trả lời y tế chuẩn xác (có chèn [CHUNK_ID]) "
            "và phân rã claim_evidence_map dưới dạng JSON."
        )

        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
            response_format=REFERENCE_ANSWER_FORMAT,
            extra_body={"provider": {"require_parameters": True}},
        )

        data = json.loads(response.choices[0].message.content)
        answer, claims = data["reference_answer"], data["claim_evidence_map"]
        q["REFERENCE_ANSWER"] = answer
        q["CLAIM_EVIDENCE_MAP"] = claims


def enrich_dataset(
    queries: list[dict[str, Any]],
    generator: AnswerGenerator | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Enrich an existing list of queries with reference answers and claim maps."""
    if generator is None:
        generator = OpenRouterAnswerGenerator()

    to_process: list[dict[str, Any]] = []
    for q in queries:
        if overwrite or not q.get("REFERENCE_ANSWER"):
            to_process.append(q)

    if to_process:
        generator.generate_answers(to_process)

    return queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enrich dataset with REFERENCE_ANSWER and CLAIM_EVIDENCE_MAP",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("test_dataset.json"),
        help="Input test dataset JSON path",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: overwrite input)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter model override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use extractive offline generator without calling external LLM APIs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing REFERENCE_ANSWER even if already present",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to enrich",
    )

    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[ERROR] Expected JSON list of queries in {args.input}", file=sys.stderr)
        return 1

    target_queries = data[:args.limit] if args.limit else data

    if args.dry_run:
        generator = ExtractiveAnswerGenerator()
    else:
        generator = OpenRouterAnswerGenerator(model=args.model)

    print(f"Enriching {len(target_queries)} queries ...")
    enrich_dataset(target_queries, generator=generator, overwrite=args.overwrite)

    out_path = args.out or args.input
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved enriched dataset to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
