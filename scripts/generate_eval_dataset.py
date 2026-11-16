"""Generate a 3-axis stratified evaluation dataset from chunks.parquet using DeepEval Synthesizer."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any

from dotenv import load_dotenv
import json_repair
from openai import AsyncOpenAI, OpenAI
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import Evolution, EvolutionConfig, StylingConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "eval_dataset.json"

ANSWERABILITY_SHARES: dict[str, float] = {
    "ANSWERABLE": 0.70,
    "PARTIALLY_ANSWERABLE": 0.30,
}

COMPLEXITY_SHARES: dict[str, float] = {
    "SINGLE_HOP": 0.50,
    "MULTI_HOP": 0.30,
    "TABLE_REASONING": 0.20,
}

QUERY_STYLE_SHARES: dict[str, float] = {
    "PATIENT_STYLE": 0.60,
    "CLINICAL_STYLE": 0.40,
}

CLINICAL_MULTI_HOP_PAIRS: tuple[tuple[str, str], ...] = (
    ("chỉ định", "chống chỉ định"),
    ("chỉ định", "liều lượng và cách dùng"),
    ("liều lượng và cách dùng", "cảnh báo và thận trọng"),
    ("liều lượng và cách dùng", "tác dụng không mong muốn"),
    ("chỉ định", "thai kỳ và cho con bú"),
    ("cảnh báo và thận trọng", "thai kỳ và cho con bú"),
    ("tương tác thuốc", "cảnh báo và thận trọng"),
    ("liều lượng và cách dùng", "tương tác thuốc"),
    ("thành phần", "chống chỉ định"),
    ("bảo quản", "hướng dẫn sử dụng"),
    ("cảnh báo và thận trọng", "quá liều"),
)

COMPARABLE_VACCINE_CATEGORIES: frozenset[str] = frozenset(
    {
        "liều lượng và cách dùng",
        "chỉ định",
        "chống chỉ định",
        "tác dụng không mong muốn",
        "cảnh báo và thận trọng",
        "bảo quản",
        "thai kỳ và cho con bú",
        "tương tác thuốc",
    }
)

_OVERDOSE_STUB_MARKERS: tuple[str, ...] = (
    "chưa có trường hợp",
    "không có dữ liệu",
    "chưa có dữ liệu",
    "chưa có báo cáo",
)


@dataclass(frozen=True)
class ChunkRecord:
    """Normalized chunk row loaded from chunks.parquet."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    header: str
    content_type: str
    has_table: bool
    text: str

    def format_context(self) -> str:
        """Render chunk text with document and section headers for synthesis."""
        parts = [p for p in (self.doc_title, self.category, self.header) if p]
        prefix = " | ".join(parts)
        return f"[{prefix}]\n{self.text}" if prefix else self.text


@dataclass(frozen=True)
class ContextGroup:
    """A unit of 1+ chunks passed as one context list to DeepEval Synthesizer."""

    chunks: tuple[ChunkRecord, ...]
    pair_type: str = "single"

    @property
    def texts(self) -> list[str]:
        return [c.format_context() for c in self.chunks]

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.doc_id for c in self.chunks))

    @property
    def doc_titles(self) -> list[str]:
        return list(dict.fromkeys(c.doc_title for c in self.chunks))


@dataclass(frozen=True)
class LightRAGConfig:
    """OpenAI-compatible endpoint settings sourced strictly from LIGHTRAG_* env vars."""

    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> LightRAGConfig:
        """Load required LIGHTRAG_API_KEY, LIGHTRAG_BASE_URL, and LIGHTRAG_LLM."""
        api_key = os.getenv("LIGHTRAG_API_KEY", "").strip()
        base_url = os.getenv("LIGHTRAG_BASE_URL", "").strip()
        model = os.getenv("LIGHTRAG_LLM", "").strip()

        missing = [
            name
            for name, val in (
                ("LIGHTRAG_API_KEY", api_key),
                ("LIGHTRAG_BASE_URL", base_url),
                ("LIGHTRAG_LLM", model),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )
        return cls(api_key=api_key, base_url=base_url, model=model)


class LightRAGDeepEvalLLM(DeepEvalBaseLLM):
    """DeepEvalBaseLLM adapter for OpenAI-compatible endpoints (e.g., OpenRouter)."""

    def __init__(self, config: LightRAGConfig, temperature: float = 0.2) -> None:
        self.config = config
        self.temperature = temperature
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        super().__init__(model=config.model)

    def load_model(self) -> OpenAI:
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._sync_client

    def _load_async_model(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._async_client

    def get_model_name(self) -> str:
        return self.config.model

    @staticmethod
    def _parse_schema(raw: str, schema: type[BaseModel]) -> BaseModel:
        parsed = json_repair.loads(raw)
        # DeepEval's Synthesizer._a_generate passes plain-text prompts with
        # schema=Response(response: str); wrap plain text directly when not JSON.
        if (
            not isinstance(parsed, dict)
            and list(schema.model_fields.keys()) == ["response"]
        ):
            return schema.model_validate({"response": raw.strip()})
        return schema.model_validate(parsed)

    def _build_request_kwargs(self, prompt: str) -> dict[str, Any]:
        # Omit response_format={"type": "json_object"} because Harmony-format
        # endpoints (e.g., gpt-oss-120b) wrap/corrupt JSON output when forced.
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

    def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        client = self.load_model()
        response = client.chat.completions.create(**self._build_request_kwargs(prompt))
        content = response.choices[0].message.content or ""
        if schema is not None:
            return self._parse_schema(content, schema)
        return content

    async def a_generate(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        client = self._load_async_model()
        response = await client.chat.completions.create(
            **self._build_request_kwargs(prompt)
        )
        content = response.choices[0].message.content or ""
        if schema is not None:
            return self._parse_schema(content, schema)
        return content


def _round_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Allocate integer counts summing to total using largest-remainder rounding."""
    if total <= 0:
        return {key: 0 for key in shares}
    raw = {key: total * weight for key, weight in shares.items()}
    floored = {key: int(val) for key, val in raw.items()}
    remainder = total - sum(floored.values())
    ranked = sorted(
        shares.keys(),
        key=lambda k: (raw[k] - floored[k], shares[k]),
        reverse=True,
    )
    for key in ranked[:remainder]:
        floored[key] += 1
    return floored


def allocate_strata(limit: int) -> dict[tuple[str, str, str], int]:
    """Compute exact golden counts across (answerability, complexity, query_style).

    Allocates each axis via largest-remainder rounding to match target shares
    while ensuring the sum of all cells equals ``limit``.
    """
    if limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")
    if limit == 0:
        return {}

    ans_counts = _round_counts(limit, ANSWERABILITY_SHARES)
    comp_counts = _round_counts(limit, COMPLEXITY_SHARES)
    style_counts = _round_counts(limit, QUERY_STYLE_SHARES)

    # Build joint cell targets proportional to outer product, then adjust to
    # preserve exact marginal totals across all three axes.
    joint_shares = {
        (ans, comp, style): (
            ANSWERABILITY_SHARES[ans]
            * COMPLEXITY_SHARES[comp]
            * QUERY_STYLE_SHARES[style]
        )
        for ans in ANSWERABILITY_SHARES
        for comp in COMPLEXITY_SHARES
        for style in QUERY_STYLE_SHARES
    }
    cells = _round_counts(limit, joint_shares)

    # Reconcile marginals only when limit is large enough to absorb rounding.
    if limit >= 20:
        _reconcile_marginals(cells, ans_counts, comp_counts, style_counts)

    return {k: v for k, v in cells.items() if v > 0}


def _reconcile_marginals(
    cells: dict[tuple[str, str, str], int],
    ans_target: dict[str, int],
    comp_target: dict[str, int],
    style_target: dict[str, int],
) -> None:
    """Adjust cell counts minimally so marginal sums match target axis counts."""
    # Reconcile answerability first by shifting within same (comp, style).
    for _ in range(10):
        ans_curr = {
            a: sum(v for (ka, _, _), v in cells.items() if ka == a)
            for a in ans_target
        }
        over = [a for a, c in ans_curr.items() if c > ans_target[a]]
        under = [a for a, c in ans_curr.items() if c < ans_target[a]]
        if not over or not under:
            break
        src_a, dst_a = over[0], under[0]
        candidate = max(
            ((k, v) for k, v in cells.items() if k[0] == src_a and v > 0),
            key=lambda item: item[1],
            default=None,
        )
        if candidate is None:
            break
        (_, c, s), _ = candidate
        cells[(src_a, c, s)] -= 1
        cells[(dst_a, c, s)] += 1

    # Reconcile complexity within same (ans, style).
    for _ in range(10):
        comp_curr = {
            c: sum(v for (_, kc, _), v in cells.items() if kc == c)
            for c in comp_target
        }
        over = [c for c, cnt in comp_curr.items() if cnt > comp_target[c]]
        under = [c for c, cnt in comp_curr.items() if cnt < comp_target[c]]
        if not over or not under:
            break
        src_c, dst_c = over[0], under[0]
        candidate = max(
            ((k, v) for k, v in cells.items() if k[1] == src_c and v > 0),
            key=lambda item: item[1],
            default=None,
        )
        if candidate is None:
            break
        (a, _, s), _ = candidate
        cells[(a, src_c, s)] -= 1
        cells[(a, dst_c, s)] += 1

    # Reconcile query_style within same (ans, comp).
    for _ in range(10):
        style_curr = {
            s: sum(v for (_, _, ks), v in cells.items() if ks == s)
            for s in style_target
        }
        over = [s for s, cnt in style_curr.items() if cnt > style_target[s]]
        under = [s for s, cnt in style_curr.items() if cnt < style_target[s]]
        if not over or not under:
            break
        src_s, dst_s = over[0], under[0]
        candidate = max(
            ((k, v) for k, v in cells.items() if k[2] == src_s and v > 0),
            key=lambda item: item[1],
            default=None,
        )
        if candidate is None:
            break
        (a, c, _), _ = candidate
        cells[(a, c, src_s)] -= 1
        cells[(a, c, dst_s)] += 1


def load_chunks_from_parquet(parquet_path: Path) -> list[ChunkRecord]:
    """Read valid chunk records from chunks.parquet using PyArrow."""
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(
        str(parquet_path),
        columns=[
            "chunk_id",
            "doc_id",
            "doc_title",
            "category",
            "header",
            "content_type",
            "text",
            "metadata",
        ],
    )
    valid_mask = pc.is_valid(table["text"])
    table = table.filter(valid_mask)

    records: list[ChunkRecord] = []
    for row in table.to_pylist():
        text = (row.get("text") or "").strip()
        if not text:
            continue
        meta = row.get("metadata") or {}
        content_type = str(row.get("content_type") or "text")
        has_table = bool(meta.get("has_table", False)) or content_type in ("table", "mixed")
        records.append(
            ChunkRecord(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                doc_title=str(row.get("doc_title") or ""),
                category=str(row.get("category") or ""),
                header=str(row.get("header") or ""),
                content_type=content_type,
                has_table=has_table,
                text=text,
            )
        )
    return records


def _is_informative_overdose_chunk(chunk: ChunkRecord) -> bool:
    """Filter out mislabeled or stub 'quá liều' chunks that lack overdose guidance."""
    if chunk.category != "quá liều":
        return True
    header_lower = chunk.header.lower()
    if "quá liều" not in header_lower and "overdose" not in header_lower:
        return False
    text_stripped = chunk.text.strip()
    if len(text_stripped) < 200:
        return False
    text_lower = text_stripped.lower()
    return not any(marker in text_lower for marker in _OVERDOSE_STUB_MARKERS)


def _build_same_doc_pairs(
    by_doc: dict[str, list[ChunkRecord]],
    rng: random.Random,
) -> list[ContextGroup]:
    """Build clinically complementary cross-section pairs within each vaccine document."""
    pairs: list[ContextGroup] = []
    for doc_chunks in by_doc.values():
        eligible = [c for c in doc_chunks if _is_informative_overdose_chunk(c)]
        by_cat: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in eligible:
            by_cat[chunk.category].append(chunk)
        for cat_list in by_cat.values():
            rng.shuffle(cat_list)

        doc_pairs: list[ContextGroup] = []
        used_ids: set[str] = set()
        for cat_a, cat_b in CLINICAL_MULTI_HOP_PAIRS:
            cand_a = [c for c in by_cat.get(cat_a, ()) if c.chunk_id not in used_ids]
            cand_b = [c for c in by_cat.get(cat_b, ()) if c.chunk_id not in used_ids]
            if cand_a and cand_b:
                c1, c2 = cand_a[0], cand_b[0]
                used_ids.add(c1.chunk_id)
                used_ids.add(c2.chunk_id)
                doc_pairs.append(ContextGroup((c1, c2), pair_type="same_doc"))

        if not doc_pairs:
            # Fallback to any distinct comparable clinical categories in the same doc.
            comparable = [c for c in eligible if c.category in COMPARABLE_VACCINE_CATEGORIES]
            rng.shuffle(comparable)
            for i in range(len(comparable)):
                if comparable[i].chunk_id in used_ids:
                    continue
                for j in range(i + 1, len(comparable)):
                    if (
                        comparable[j].chunk_id not in used_ids
                        and comparable[i].category != comparable[j].category
                    ):
                        used_ids.update((comparable[i].chunk_id, comparable[j].chunk_id))
                        doc_pairs.append(
                            ContextGroup((comparable[i], comparable[j]), pair_type="same_doc")
                        )
                        break

        pairs.extend(doc_pairs)
    return pairs


def _build_cross_vaccine_pairs(
    records: list[ChunkRecord],
    rng: random.Random,
) -> list[ContextGroup]:
    """Build cross-vaccine comparison pairs sharing a comparable clinical category."""
    by_cat: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in records:
        if chunk.category in COMPARABLE_VACCINE_CATEGORIES:
            by_cat[chunk.category].append(chunk)

    pairs: list[ContextGroup] = []
    for cat_chunks in by_cat.values():
        if len(cat_chunks) < 2:
            continue
        shuffled = list(cat_chunks)
        rng.shuffle(shuffled)
        used_ids: set[str] = set()
        for i in range(len(shuffled)):
            if shuffled[i].chunk_id in used_ids:
                continue
            for j in range(i + 1, len(shuffled)):
                if (
                    shuffled[j].chunk_id not in used_ids
                    and shuffled[i].doc_id != shuffled[j].doc_id
                ):
                    used_ids.update((shuffled[i].chunk_id, shuffled[j].chunk_id))
                    pairs.append(
                        ContextGroup((shuffled[i], shuffled[j]), pair_type="cross_vaccine")
                    )
                    break
    return pairs


def build_context_pools(
    records: list[ChunkRecord],
    rng: random.Random,
) -> dict[str, list[ContextGroup]]:
    """Construct SINGLE_HOP, MULTI_HOP, and TABLE_REASONING context pools."""
    table_chunks = [c for c in records if c.has_table]
    text_chunks = [c for c in records if not c.has_table]

    single_hop_pool = [ContextGroup((c,)) for c in (text_chunks or records)]
    table_pool = [ContextGroup((c,)) for c in (table_chunks or records)]

    by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for c in records:
        by_doc[c.doc_id].append(c)

    same_doc_pairs = _build_same_doc_pairs(by_doc, rng)
    cross_vaccine_pairs = _build_cross_vaccine_pairs(records, rng)
    rng.shuffle(same_doc_pairs)
    rng.shuffle(cross_vaccine_pairs)

    # Interleave same_doc and cross_vaccine pairs for a balanced 50/50 mix.
    multi_hop_pool: list[ContextGroup] = []
    paired_len = min(len(same_doc_pairs), len(cross_vaccine_pairs))
    for idx in range(paired_len):
        multi_hop_pool.append(same_doc_pairs[idx])
        multi_hop_pool.append(cross_vaccine_pairs[idx])
    multi_hop_pool.extend(same_doc_pairs[paired_len:])
    multi_hop_pool.extend(cross_vaccine_pairs[paired_len:])

    if not multi_hop_pool and len(records) >= 2:
        # Minimal fallback for tiny synthetic fixtures without standard categories.
        for doc_chunks in by_doc.values():
            if len(doc_chunks) >= 2:
                multi_hop_pool.append(
                    ContextGroup((doc_chunks[0], doc_chunks[1]), pair_type="same_doc")
                )
                break
        if not multi_hop_pool:
            multi_hop_pool.append(
                ContextGroup((records[0], records[1]), pair_type="same_doc")
            )

    rng.shuffle(single_hop_pool)
    rng.shuffle(table_pool)

    return {
        "SINGLE_HOP": single_hop_pool,
        "MULTI_HOP": multi_hop_pool,
        "TABLE_REASONING": table_pool,
    }


def build_styling_config(
    answerability: str,
    query_style: str,
    complexity: str = "SINGLE_HOP",
    pair_type: str = "single",
) -> StylingConfig:
    """Create DeepEval StylingConfig tailored to answerability, complexity, and query style."""
    if query_style == "PATIENT_STYLE":
        scenario = (
            "Người dân Việt Nam hỏi đáp y tế thực tế qua tin nhắn hoặc diễn đàn sức khỏe "
            "(tự hỏi cho bản thân, hỏi cho con nhỏ/người lớn tuổi, hoặc hỏi nhanh thông tin tiêm chủng)."
        )
        input_format = (
            "Câu hỏi tiếng Việt ngắn gọn, tự nhiên, dùng từ ngữ phổ thông đời thường, đa dạng luân phiên giữa 3 kiểu: "
            "(1) kể ngắn bối cảnh/triệu chứng/bệnh nền rồi hỏi ('em...', 'bé nhà em...', 'vậy có... được không'); "
            "(2) hỏi thẳng vào đối tượng + tình huống ('Trẻ 4 tuổi...', 'Người tiểu đường...'); "
            "(3) tin nhắn ngắn 8-15 từ đi thẳng vào trọng tâm. "
            "Dùng từ ngữ phổ thông và tiểu từ hội thoại ('có sao không ạ', 'được không', 'nhỉ', 'vậy'), "
            "không cần lúc nào cũng viết hoa/chấm hỏi quá chuẩn mực; tránh thuật ngữ y khoa hàn lâm."
        )
    else:
        scenario = (
            "Bác sĩ hoặc nhân viên y tế kiểm tra phác đồ tiêm chủng vaccine "
            "cho ca lâm sàng có nhiều ràng buộc."
        )
        input_format = (
            "Câu hỏi tiếng Việt chuyên môn chứa nhiều điều kiện ràng buộc "
            "(độ tuổi, bệnh nền, tiền sử tiêm, phác đồ trễ mũi, tương tác thuốc)."
        )

    if complexity == "MULTI_HOP" and pair_type == "cross_vaccine":
        input_format += (
            " BẮT BUỘC nêu rõ tên cả 2 vaccine cần so sánh có trong ngữ cảnh; "
            "không dùng đại từ mơ hồ thay cho tên vaccine."
        )
    else:
        input_format += (
            " BẮT BUỘC nêu rõ tên vaccine cụ thể có trong ngữ cảnh; "
            "không dùng đại từ mơ hồ thay cho tên vaccine."
        )

    if complexity == "MULTI_HOP" and pair_type == "cross_vaccine":
        complexity_instruction = (
            " Kết hợp thông tin từ cả 2 đoạn ngữ cảnh của 2 vaccine khác nhau để đặt câu hỏi "
            "so sánh trực tiếp (đối tượng chỉ định, lịch tiêm, chống chỉ định, tác dụng phụ hoặc bảo quản)."
        )
        scenario += " Trọng tâm: so sánh giữa hai loại vaccine."
    elif complexity == "MULTI_HOP":
        complexity_instruction = (
            " Kết hợp thông tin từ cả 2 đoạn ngữ cảnh thuộc các mục khác nhau của cùng một vaccine "
            "(ví dụ: liều dùng với thận trọng, chỉ định với chống chỉ định)."
        )
        scenario += " Trọng tâm: tổng hợp đa mục trong cùng hồ sơ vaccine."
    elif complexity == "TABLE_REASONING":
        complexity_instruction = (
            " Truy xuất và đối chiếu chính xác số liệu, tỷ lệ phần trăm, hiệu giá kháng thể "
            "hoặc mốc độ tuổi từ bảng lâm sàng trong ngữ cảnh."
        )
        scenario += " Trọng tâm: đọc hiểu và đối chiếu bảng số liệu lâm sàng."
    else:
        complexity_instruction = " Tập trung vào một khía cạnh lâm sàng cụ thể trong ngữ cảnh."

    if answerability == "ANSWERABLE":
        task = (
            "Trả lời chính xác, đầy đủ thông tin vaccine dựa hoàn toàn "
            "trên ngữ cảnh được cung cấp." + complexity_instruction
        )
        expected_output_format = (
            "Câu trả lời tiếng Việt súc tích, chính xác, tổng hợp đầy đủ các ý có trong context"
            + (
                " và nêu rõ điểm giống/khác nhau giữa 2 vaccine."
                if complexity == "MULTI_HOP" and pair_type == "cross_vaccine"
                else "."
            )
        )
    elif answerability == "PARTIALLY_ANSWERABLE":
        task = (
            "Tạo câu hỏi gồm 2 phần: một phần có thông tin trong context và một phần "
            "nằm ngoài context để kiểm tra khả năng trả lời có chọn lọc và từ chối phần thiếu."
            + complexity_instruction
        )
        expected_output_format = (
            "Trả lời chính xác phần có dữ liệu trong context, đồng thời nêu rõ "
            "và từ chối phần thông tin không có trong tài liệu."
        )
    else:
        raise ValueError(f"Unsupported answerability: {answerability}")

    return StylingConfig(
        scenario=scenario,
        task=task,
        input_format=input_format,
        expected_output_format=expected_output_format,
    )


def build_evolution_config(complexity: str, pair_type: str = "single") -> EvolutionConfig:
    """Create DeepEval EvolutionConfig tailored to each complexity tier and pair type."""
    if complexity == "MULTI_HOP" and pair_type == "cross_vaccine":
        evolutions = {Evolution.COMPARATIVE: 0.6, Evolution.MULTICONTEXT: 0.4}
    elif complexity == "MULTI_HOP":
        evolutions = {Evolution.MULTICONTEXT: 0.7, Evolution.REASONING: 0.3}
    elif complexity == "TABLE_REASONING":
        evolutions = {Evolution.REASONING: 0.5, Evolution.COMPARATIVE: 0.5}
    else:
        evolutions = {Evolution.CONCRETIZING: 0.5, Evolution.CONSTRAINED: 0.5}
    return EvolutionConfig(num_evolutions=1, evolutions=evolutions)


def _sample_groups(
    pool: list[ContextGroup],
    count: int,
    cursor: int,
) -> tuple[list[ContextGroup], int]:
    """Take ``count`` ContextGroups from pool with wrap-around if needed."""
    if not pool:
        raise ValueError("Context pool is empty")
    selected: list[ContextGroup] = []
    for i in range(count):
        selected.append(pool[(cursor + i) % len(pool)])
    return selected, (cursor + count) % len(pool)


def generate_dataset(
    *,
    chunks_path: Path,
    output_path: Path,
    limit: int,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run stratified golden generation and return the serialized dataset payload."""
    records = load_chunks_from_parquet(chunks_path)
    rng = random.Random(seed)
    pools = build_context_pools(records, rng)
    strata = allocate_strata(limit)

    cursors = {comp: 0 for comp in COMPLEXITY_SHARES}
    plan_items: list[tuple[tuple[str, str, str], int, list[ContextGroup]]] = []
    for key, count in strata.items():
        _, complexity, _ = key
        groups, cursors[complexity] = _sample_groups(
            pools[complexity], count, cursors[complexity]
        )
        plan_items.append((key, count, groups))

    if dry_run:
        return {
            "metadata": {
                "dry_run": True,
                "limit": limit,
                "seed": seed,
                "total_chunks": len(records),
                "strata": {
                    f"{a}|{c}|{s}": cnt for (a, c, s), cnt, _ in plan_items
                },
            },
            "goldens": [],
        }

    config = LightRAGConfig.from_env()
    llm = LightRAGDeepEvalLLM(config)

    goldens_out: list[dict[str, Any]] = []
    query_idx = 1

    for (answerability, complexity, query_style), count, groups in plan_items:
        # Sub-batch by pair_type so cross_vaccine and same_doc MULTI_HOP items
        # receive their tailored StylingConfig and EvolutionConfig.
        batches: dict[str, list[ContextGroup]] = defaultdict(list)
        for group in groups:
            batches[group.pair_type].append(group)

        for pair_type, batch_groups in batches.items():
            synthesizer = Synthesizer(
                model=llm,
                styling_config=build_styling_config(
                    answerability, query_style, complexity, pair_type
                ),
                evolution_config=build_evolution_config(complexity, pair_type),
                async_mode=True,
                cost_tracking=False,
            )
            contexts = [g.texts for g in batch_groups]
            generated = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=True,
                max_goldens_per_context=1,
                _send_data=False,
            )
            for idx, golden in enumerate(generated[: len(batch_groups)]):
                group = batch_groups[idx % len(batch_groups)]
                goldens_out.append(
                    {
                        "input": golden.input,
                        "expected_output": golden.expected_output,
                        "context": golden.context or group.texts,
                        "additional_metadata": {
                            "query_id": f"eval_{query_idx:04d}",
                            "answerability": answerability,
                            "complexity": complexity,
                            "multi_hop_type": (
                                group.pair_type if complexity == "MULTI_HOP" else None
                            ),
                            "query_style": query_style,
                            "source_chunk_ids": group.chunk_ids,
                            "doc_ids": group.doc_ids,
                            "doc_titles": group.doc_titles,
                        },
                    }
                )
                query_idx += 1

    payload = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": config.model,
            "limit": limit,
            "total_generated": len(goldens_out),
            "seed": seed,
            "distribution": _compute_distribution(goldens_out),
        },
        "goldens": goldens_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _compute_distribution(goldens: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute realized counts per stratification axis."""
    dist: dict[str, dict[str, int]] = {
        "answerability": defaultdict(int),
        "complexity": defaultdict(int),
        "query_style": defaultdict(int),
    }
    for item in goldens:
        meta = item.get("additional_metadata", {})
        dist["answerability"][meta.get("answerability", "UNKNOWN")] += 1
        dist["complexity"][meta.get("complexity", "UNKNOWN")] += 1
        dist["query_style"][meta.get("query_style", "UNKNOWN")] += 1
    return {axis: dict(counts) for axis, counts in dist.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate stratified VacxiS evaluation dataset via DeepEval Synthesizer."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of goldens to generate (default: 50).",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks.parquet (default: {DEFAULT_CHUNKS_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible chunk sampling (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stratification plan without calling the LLM API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    args = parse_args(argv)
    result = generate_dataset(
        chunks_path=args.chunks_path,
        output_path=args.output,
        limit=args.limit,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))
    else:
        print(
            f"Generated {result['metadata']['total_generated']} goldens -> {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
