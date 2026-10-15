"""Generate a golden evaluation dataset for the Vaccine RAG system.

Reads the chunked corpus produced by the indexing pipeline
(`chunks_storage/chunks.parquet`) and the vaccine name mapping
(`vaccine_name_mapping.json`), then synthesises a query set that
follows the schema defined in `SCHEMA_REQ.md` (mirrored in
`schema.json`).

Each entry contains:
  * a Vietnamese query (one of the standard QUERY_TYPE categories)
  * a list of RELEVANT_CHUNKS drawn from the corpus, with character
    offsets into the chunk TEXT marking the EVIDENCE_SPAN
  * a RELEVANCE_SCORE per chunk (0..3) so retrieval metrics are
    well-defined
  * metadata fields (entities, constraints, completeness, etc.)

Usage:
    uv run scripts/generate_test_dataset.py
    uv run scripts/generate_test_dataset.py --per-type 8 --out test_data.json
    uv run scripts/generate_test_dataset.py --seed 42 --distractors 3
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "vaccine_name_mapping.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test_dataset.json"

# ---------------------------------------------------------------------------
# Query-type definitions (mirroring SCHEMA_REQ.md)
# ---------------------------------------------------------------------------

# Each query type maps to:
#   - a category in the chunking schema that the question is grounded in
#   - a set of Vietnamese question templates
#   - a list of EVIDENCE_GROUPS the answer is expected to cover
QUERY_TYPE_DEFS: list[dict] = [
    {
        "type": "INDICATION",
        "category": "chỉ định",
        "evidence_groups": ["INDICATIONS"],
        "templates": [
            "Vaccine {vaccine} được chỉ định cho những đối tượng nào?",
            "Vaccine {vaccine} dùng để phòng ngừa những bệnh gì?",
            "{vaccine} có thể dùng cho đối tượng nào theo hướng dẫn sử dụng?",
            "Vắc xin {vaccine} được sử dụng để phòng bệnh gì?",
            "{vaccine} phòng ngừa những bệnh nào?",
            "Vaccine {vaccine} có chỉ định cho những bệnh lý gì?",
            "Vắc xin {vaccine} dành cho những đối tượng nào?",
            "Những ai có thể tiêm vaccine {vaccine}?",
            "Đối tượng nào được tiêm {vaccine}?",
        ],
    },
    {
        "type": "DOSAGE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["NUMBER_OF_DOSES", "DOSING_INTERVAL"],
        "templates": [
            "Vaccine {vaccine} cần tiêm bao nhiêu liều?",
            "Liều lượng khuyến cáo của {vaccine} là bao nhiêu?",
            "{vaccine} nên tiêm với lịch như thế nào?",
            "Vắc xin {vaccine} cần tiêm bao nhiêu liều, mỗi liều cách nhau bao lâu?",
            "Tôi cần tiêm bao nhiêu liều {vaccine} và khoảng cách giữa các liều là bao lâu?",
            "Nếu tôi bỏ lỡ lịch tiêm mũi tiếp theo của vaccine {vaccine}, tôi có cần tiêm lại từ đầu không?",
            "Tôi quên tiêm liều nhắc của {vaccine}, phải làm sao?",
        ],
    },
    {
        "type": "ADMINISTRATION_ROUTE",
        "category": "liều lượng và cách dùng",
        "evidence_groups": ["ADMINISTRATION_ROUTE"],
        "templates": [
            "Vaccine {vaccine} được tiêm bằng đường nào?",
            "Cách dùng của {vaccine} như thế nào?",
            "Vị trí tiêm khuyến cáo cho {vaccine} là ở đâu?",
        ],
    },
    {
        "type": "CONTRAINDICATION",
        "category": "chống chỉ định",
        "evidence_groups": ["CONTRAINDICATIONS"],
        "templates": [
            "Vaccine {vaccine} chống chỉ định trong trường hợp nào?",
            "Đối tượng nào không nên tiêm {vaccine}?",
            "Những ai không được dùng {vaccine}?",
            "Những trường hợp nào chống chỉ định sử dụng vaccine {vaccine}?",
            "Ai không nên tiêm {vaccine}?",
            "Khi nào thì không nên tiêm vaccine {vaccine}?",
        ],
    },
    {
        "type": "ADVERSE_EFFECT",
        "category": "tác dụng không mong muốn",
        "evidence_groups": ["ADVERSE_EFFECTS"],
        "templates": [
            "Vaccine {vaccine} có những tác dụng không mong muốn nào?",
            "Phản ứng phụ thường gặp khi tiêm {vaccine} là gì?",
            "Sau khi tiêm {vaccine} có thể gặp tác dụng phụ nào?",
            "Những phản ứng phụ nào có thể xuất hiện sau khi tiêm {vaccine}?",
            "Sau khi tiêm vaccine {vaccine}, tôi thấy tôi bị đau đầu và mệt, có phải phản ứng phụ của nó không?",
            "Tôi bị sưng ở chỗ tiêm vaccine sau khi tiêm {vaccine}, có bình thường không?",
            "Tôi đã tiêm vaccine {vaccine} được 2 ngày rồi nhưng vẫn còn sốt nhẹ. Trường hợp này có bình thường không?",
            "Những phản ứng phụ nghiêm trọng nào có thể xảy ra sau khi tiêm vaccine {vaccine}?",
            "Tác dụng phụ nguy hiểm nào có thể xảy ra khi tiêm {vaccine}?",
            "Tôi vừa tiêm vaccine {vaccine} và thấy chỗ tiêm đỏ, nóng và đau. Những triệu chứng này kéo dài bao lâu thì được xem là bất thường?",
            "Chỗ tiêm {vaccine} bị sưng đỏ có bình thường không?",
        ],
    },
    {
        "type": "COMPOSITION",
        "category": "thành phần",
        "evidence_groups": ["COMPOSITION"],
        "templates": [
            "Thành phần hoạt chất của vaccine {vaccine} gồm những gì?",
            "{vaccine} chứa những hoạt chất và tá dược nào?",
            "Trong {vaccine} có những thành phần gì?",
        ],
    },
    {
        "type": "STORAGE",
        "category": "bảo quản",
        "evidence_groups": ["STORAGE"],
        "templates": [
            "Vaccine {vaccine} cần được bảo quản như thế nào?",
            "Điều kiện bảo quản của {vaccine} là gì?",
            "{vaccine} có hạn dùng và cách bảo quản ra sao?",
        ],
    },
    {
        "type": "PRECAUTION",
        "category": "cảnh báo và thận trọng",
        "evidence_groups": ["PRECAUTIONS"],
        "templates": [
            "Khi sử dụng {vaccine} cần lưu ý những gì?",
            "Cảnh báo và thận trọng nào khi tiêm {vaccine}?",
            "Có những điều gì cần thận trọng khi dùng {vaccine}?",
            "Nếu trước đây tôi từng bị dị ứng sau khi tiêm một loại vaccine khác thì có tiêm được vaccine {vaccine} không?",
            "Tôi có tiền sử dị ứng vaccine, có tiêm được {vaccine} không?",
            "Tôi bị bệnh nền, vậy tôi có thuộc nhóm cần thận trọng khi tiêm vaccine {vaccine} không?",
            "Người có bệnh lý nền có tiêm được {vaccine} không?",
            "Những bệnh nền nào cần thận trọng khi tiêm {vaccine}?",
            "Sau khi tiêm vaccine {vaccine}, tôi nên theo dõi những triệu chứng gì và trong bao lâu?",
            "Sau khi tiêm {vaccine} tôi cần chú ý điều gì?",
            "Không biết đã tiêm {vaccine} chưa thì có tiêm được không?",
            "Sau khi tiêm vaccine {vaccine}, tôi có thể lái xe hoặc vận hành máy móc không?",
        ],
    },
    {
        "type": "INTERACTION",
        "category": "tương tác thuốc",
        "evidence_groups": ["INTERACTIONS"],
        "templates": [
            "Vaccine {vaccine} có tương tác với thuốc nào không?",
            "Tương tác thuốc của {vaccine} như thế nào?",
            "Vaccine {vaccine} có thể tiêm cùng lúc với các vaccine khác không?",
            "Tôi có thể tiêm {vaccine} đồng thời với vaccine cúm được không?",
            "Những vaccine nào có thể tiêm cùng lúc với {vaccine}?",
        ],
    },
    {
        "type": "SPECIAL_POPULATION",
        "category": "thai kỳ và cho con bú",
        "evidence_groups": ["SPECIAL_POPULATIONS"],
        "templates": [
            "Phụ nữ có thai có thể tiêm {vaccine} không?",
            "{vaccine} có dùng được cho phụ nữ đang cho con bú không?",
        ],
    },
]

DIFFICULTY_DISTRIBUTION: list[tuple[str, float]] = [
    ("EASY", 0.5),
    ("MEDIUM", 0.3),
    ("HARD", 0.2),
]

COMPLEXITY_DISTRIBUTION: list[tuple[str, float]] = [
    ("SINGLE_HOP", 0.7),
    ("MULTI_HOP", 0.3),
]

ANSWERABILITY_DISTRIBUTION: list[tuple[str, float]] = [
    ("ANSWERABLE", 0.8),
    ("PARTIALLY_ANSWERABLE", 0.15),
    ("UNANSWERABLE", 0.05),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """Lightweight view over a row in chunks.parquet."""

    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    text: str
    source_file: str

    @property
    def vaccine_name(self) -> str:
        # doc_title looks like "Hướng dẫn sử dụng Vaccine X"
        prefix = "Hướng dẫn sử dụng Vaccine "
        if self.doc_title.startswith(prefix):
            return self.doc_title[len(prefix):]
        return self.doc_title


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks.parquet into a list of Chunk objects."""
    table = pq.read_table(str(path))
    chunks: list[Chunk] = []
    for i in range(table.num_rows):
        row = table.slice(i, 1).to_pylist()[0]
        metadata = row.get("metadata") or {}
        chunks.append(
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                doc_title=row["doc_title"],
                category=row["category"],
                text=row["text"],
                source_file=metadata.get("source_file", ""),
            )
        )
    return chunks


def load_name_mapping(path: Path) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------


def index_chunks_by_category(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """Group chunks by their normalized category for sampling."""
    index: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        index.setdefault(chunk.category, []).append(chunk)
    return index


def select_evidence_span(text: str, rng: random.Random, span_len: int = 140) -> tuple[int, int]:
    """Pick a `[start, end)` substring inside `text` to use as evidence span.

    Prefers the body of the chunk (skipping the `[CHỦ ĐỀ: ...] > [MỤC: ...]`
    prefix that the indexing pipeline prepends) so the span points to real
    content rather than the synthetic header.
    """
    body_start = text.find("\n")
    if body_start == -1 or body_start > len(text) - 40:
        body_start = 0
    else:
        body_start += 1  # skip the newline

    max_start = max(body_start, len(text) - span_len - 1)
    if max_start <= body_start:
        return 0, min(len(text), span_len)

    start = rng.randint(body_start, max_start)
    end = min(len(text), start + span_len)
    return start, end


def sample_distractors(
    chunk: Chunk,
    chunks: list[Chunk],
    rng: random.Random,
    n: int,
) -> list[Chunk]:
    """Return `n` random chunks from a different doc than `chunk`."""
    pool = [c for c in chunks if c.doc_id != chunk.doc_id]
    rng.shuffle(pool)
    return pool[:n]


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


def weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    values, weights = zip(*options)
    return rng.choices(values, weights=weights, k=1)[0]


def make_query_text(
    template: str,
    vaccine_name: str,
    rng: random.Random,
) -> tuple[str, str]:
    """Fill a Vietnamese query template, optionally with a paraphrase variant.

    Returns (query, source) where `source` is one of the SCHEMA_REQ.md values.
    """
    query = template.format(vaccine=vaccine_name)
    # 25% of queries get a small natural-language tweak to simulate paraphrases
    if rng.random() < 0.25:
        query = _light_paraphrase(query, rng)
        return query, "PARAPHRASE"
    return query, "TEMPLATE"


_PREFIX_RE = re.compile(r"^(vaccine\s+)+", re.IGNORECASE)


def _light_paraphrase(query: str, rng: random.Random) -> str:
    """Apply a tiny stylistic tweak so the query still reads naturally."""
    swap = {
        "Vaccine": ["Vacxin", "Vắc-xin"],
        "vắc-xin": ["vaccine", "Vacxin"],
        "như thế nào": ["ra sao", "thế nào"],
    }
    for src, options in swap.items():
        if src in query and rng.random() < 0.5:
            return query.replace(src, rng.choice(options), 1)
    return query


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_dataset(
    chunks: list[Chunk],
    rng: random.Random,
    per_type: int,
    distractors_per_query: int,
) -> list[dict]:
    by_category = index_chunks_by_category(chunks)
    queries: list[dict] = []
    qid_counter = 1

    for type_def in QUERY_TYPE_DEFS:
        category = type_def["category"]
        pool = by_category.get(category, [])
        if not pool:
            print(f"[WARN] No chunks found for category '{category}'", file=sys.stderr)
            continue

        sampled = rng.sample(pool, min(per_type, len(pool)))
        for chunk in sampled:
            template = rng.choice(type_def["templates"])
            query_text, source = make_query_text(template, chunk.vaccine_name, rng)

            start, end = select_evidence_span(chunk.text, rng)
            evidence_chunk = {
                "CHUNK_ID": chunk.chunk_id,
                "TEXT": chunk.text,
                "RELEVANCE_SCORE": 3,
                "EVIDENCE_SPANS": [{"START": start, "END": end}],
            }

            # Add distractors with mixed relevance scores
            distractors = sample_distractors(chunk, chunks, rng, distractors_per_query)
            for idx, d in enumerate(distractors):
                # Earlier distractors are more relevant (score 2) than later ones (0/1)
                score = max(0, 2 - idx)
                evidence_chunk_list = [
                    {
                        "CHUNK_ID": d.chunk_id,
                        "TEXT": d.text,
                        "RELEVANCE_SCORE": score,
                        "EVIDENCE_SPANS": [],
                    }
                ]
            relevant_chunks = [evidence_chunk] + [
                {
                    "CHUNK_ID": d.chunk_id,
                    "TEXT": d.text,
                    "RELEVANCE_SCORE": max(0, 2 - idx),
                    "EVIDENCE_SPANS": [],
                }
                for idx, d in enumerate(distractors)
            ]

            difficulty = weighted_choice(rng, DIFFICULTY_DISTRIBUTION)
            complexity = weighted_choice(rng, COMPLEXITY_DISTRIBUTION)
            answerability = weighted_choice(rng, ANSWERABILITY_DISTRIBUTION)

            entities: dict[str, str] = {"vaccine": chunk.vaccine_name}
            if complexity == "MULTI_HOP":
                # Multi-hop queries reference two vaccines to force cross-doc reasoning
                other = rng.choice([c for c in chunks if c.doc_id != chunk.doc_id])
                entities["related_vaccine"] = other.vaccine_name
                relevant_docs = sorted({chunk.doc_id, other.doc_id})
            else:
                relevant_docs = [chunk.doc_id]

            # For UNANSWERABLE, empty out the evidence list so a retriever
            # has to learn to abstain.
            if answerability == "UNANSWERABLE":
                relevant_chunks_payload: list[dict] = []
                evidence_completeness = "NOT_ASSESSED"
            elif answerability == "PARTIALLY_ANSWERABLE":
                relevant_chunks_payload = [evidence_chunk]
                evidence_completeness = "PARTIAL"
            else:
                relevant_chunks_payload = relevant_chunks
                evidence_completeness = "COMPLETE"

            queries.append(
                {
                    "QUERY_ID": f"VAX_Q_{qid_counter:04d}",
                    "QUERY": query_text,
                    "QUERY_DIFFICULTY": difficulty,
                    "QUERY_TYPE": type_def["type"],
                    "QUERY_SOURCE": source,
                    "QUERY_COMPLEXITY": complexity,
                    "ENTITIES": entities,
                    "CONSTRAINTS": {},
                    "ANSWERABILITY": answerability,
                    "RELEVANT_DOCS": relevant_docs,
                    "RELEVANT_CHUNKS": relevant_chunks_payload,
                    "REQUIRED_EVIDENCE_GROUPS": type_def["evidence_groups"],
                    "EVIDENCE_COMPLETENESS": evidence_completeness,
                    "REFERENCE_ANSWER": None,
                    "CLAIM_EVIDENCE_MAP": [],
                    "SOURCE_VERSION": "PI_VERSION_ID",
                    "VERIFIED": False,
                    "ANNOTATOR_IDS": ["GEN"],
                    "ADJUDICATED": False,
                    "VERIFICATION_NOTES": None,
                }
            )
            qid_counter += 1

    rng.shuffle(queries)
    # Re-stamp QUERY_ID in shuffled order
    for i, q in enumerate(queries, start=1):
        q["QUERY_ID"] = f"VAX_Q_{i:04d}"
    return queries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a golden evaluation dataset from chunks.parquet",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks.parquet (default: chunks_storage/chunks.parquet)",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to vaccine_name_mapping.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file (default: test_dataset.json at project root)",
    )
    p.add_argument(
        "--per-type",
        type=int,
        default=3,
        help="Number of queries to generate per QUERY_TYPE (default: 3)",
    )
    p.add_argument(
        "--distractors",
        type=int,
        default=2,
        help="Number of distractor chunks to attach per query (default: 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.chunks.exists():
        print(f"[ERROR] chunks parquet not found at {args.chunks}", file=sys.stderr)
        print("        Run the indexing pipeline first: uv run vacxis-capstone-fa26", file=sys.stderr)
        return 1

    print(f"Loading chunks from {args.chunks} ...")
    chunks = load_chunks(args.chunks)
    print(f"  -> {len(chunks)} chunks across {len({c.doc_id for c in chunks})} documents")

    if args.mapping.exists():
        # Mapping is informational here; chunk.vaccine_name is already resolved
        # from doc_title. We only verify the file is well-formed.
        load_name_mapping(args.mapping)
        print(f"  -> vaccine name mapping loaded from {args.mapping}")

    print(
        f"Generating {args.per_type} query/QUERY_TYPE with {args.distractors} "
        "distractors per query ..."
    )
    queries = generate_dataset(chunks, rng, args.per_type, args.distractors)
    print(f"  -> {len(queries)} queries generated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"Wrote dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
