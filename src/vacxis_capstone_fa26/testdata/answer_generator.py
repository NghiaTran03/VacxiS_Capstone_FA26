"""Reference answer generator and claim-evidence mapper for Vaccine RAG.

Supports two operational modes:
1. LLM-based generation via LiteLLM (when credentials / network are available)
2. Extractive grounded fallback (offline, deterministic, dry-run friendly)

Both modes generate:
- ``REFERENCE_ANSWER``: concise, factual Vietnamese medical answers with inline
  bracketed citations e.g. ``[CHUNK_ID]``.
- ``CLAIM_EVIDENCE_MAP``: atomic claims mapped to the specific chunk ID and quote.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol

import litellm


UNANSWERABLE_FALLBACK_ANSWER = (
    "Tài liệu hướng dẫn sử dụng không cung cấp thông tin cho câu hỏi này. "
    "Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại các cơ sở tiêm chủng để được tư vấn chính xác."
)


SYSTEM_PROMPT = """Bạn là một chuyên gia y tế tổng hợp thông tin vắc xin dựa trên tờ Hướng dẫn sử dụng (Product Insert).
Nhiệm vụ: Dựa DUY NHẤT vào các đoạn văn bản (CHUNKS) được cung cấp dưới đây, hãy soạn thảo câu trả lời tham chiếu (REFERENCE_ANSWER) chính xác, ngắn gọn và trung thực.

Quy tắc bắt buộc:
1. Tuyệt đối không suy diễn hoặc đưa thông tin nằm ngoài các đoạn CHUNKS được cung cấp.
2. Trích dẫn nguồn: Sau mỗi khẳng định hoặc câu trong câu trả lời, BẮT BUỘC chèn mã định danh chunk dạng `[CHUNK_ID]`.
3. Phân rã khẳng định (CLAIM_EVIDENCE_MAP): Liệt kê từng luận điểm/khẳng định nhỏ (claim), mã chunk tương ứng (chunk_id), và đoạn trích dẫn nguyên văn từ chunk làm bằng chứng (evidence_quote).
4. Nếu câu hỏi không thể trả lời từ các chunks (hoặc chunks rỗng), hãy trả lời:
   "Tài liệu hướng dẫn sử dụng không cung cấp thông tin cho câu hỏi này. Bạn nên tham khảo ý kiến bác sĩ hoặc chuyên viên y tế tại các cơ sở tiêm chủng để được tư vấn chính xác."
   và để claim_evidence_map là [].
5. Trả về ĐÚNG định dạng JSON theo cấu trúc sau, không kèm bất kỳ giải thích nào khác ngoài JSON:

{
  "reference_answer": "Câu trả lời hoàn chỉnh kèm [CHUNK_ID] trích dẫn sau mỗi câu.",
  "claim_evidence_map": [
    {
      "claim": "Khẳng định y khoa 1",
      "chunk_id": "ID_CỦA_CHUNK",
      "evidence_quote": "Đoạn văn trích nguyên văn làm bằng chứng"
    }
  ]
}
"""


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
            spans = c.get("EVIDENCE_SPANS", [])

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


class LiteLLMAnswerGenerator:
    """Generates golden reference answers with citations and claim maps via LiteLLM."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("VACXIS_LLM_MODEL", "gpt-4o-mini")
        self.batch_size = batch_size
        self.temperature = temperature
        self._fallback = ExtractiveAnswerGenerator()

    def generate_answers(self, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, q in enumerate(queries):
            try:
                self._generate_single(q)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] LiteLLMAnswerGenerator failed on query {q.get('QUERY_ID', i)}: {exc}. "
                    "Falling back to extractive generator.",
                    file=sys.stderr,
                )
                self._fallback._generate_single(q)
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
            f"CÂU HỎI: {q.get('QUERY')}\n\n"
            f"CÁC ĐOẠN CHUNKS CUNG CẤP:\n{chunks_context}\n\n"
            "Hãy tổng hợp câu trả lời y tế chuẩn xác (có chèn [CHUNK_ID]) "
            "và phân rã claim_evidence_map dưới dạng JSON."
        )

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=self.temperature,
        )

        raw_content = response.choices[0].message.content
        data = _parse_json(raw_content)

        ref_answer = data.get("reference_answer", "").strip()
        claims = data.get("claim_evidence_map", [])

        if not ref_answer:
            self._fallback._generate_single(q)
            return

        # Ensure all citations correspond to actual chunk IDs
        valid_cids = {c.get("CHUNK_ID") for c in relevant}
        q["REFERENCE_ANSWER"] = ref_answer
        q["CLAIM_EVIDENCE_MAP"] = [
            cl for cl in claims
            if isinstance(cl, dict) and cl.get("chunk_id") in valid_cids
        ] or claims


def _parse_json(raw: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the outermost braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def enrich_dataset(
    queries: list[dict[str, Any]],
    generator: AnswerGenerator | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Enrich an existing list of queries with reference answers and claim maps."""
    if generator is None:
        has_key = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        generator = LiteLLMAnswerGenerator() if has_key else ExtractiveAnswerGenerator()

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
        help="Model name for LiteLLM (e.g. gpt-4o-mini, gemini/gemini-1.5-flash)",
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
        has_key = bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if has_key:
            generator = LiteLLMAnswerGenerator(model=args.model)
        else:
            print("[INFO] No LLM API key detected in environment. Using extractive offline generator.")
            generator = ExtractiveAnswerGenerator()

    print(f"Enriching {len(target_queries)} queries ...")
    enrich_dataset(target_queries, generator=generator, overwrite=args.overwrite)

    out_path = args.out or args.input
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved enriched dataset to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
