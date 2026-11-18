"""Unit tests for scripts/generate_eval_dataset.py."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_eval_dataset as gen  # noqa: E402


def _write_sample_parquet(path: Path) -> None:
    """Create a minimal chunks.parquet fixture covering text and table rows."""
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "doc_title": "Vaccine A",
            "category": "liều lượng",
            "header": "Liều cơ bản",
            "content_type": "text",
            "text": "Tiêm bắp 0.5 ml cho trẻ từ 2 tháng tuổi.",
            "metadata": {"has_table": False, "source_file": "a.md"},
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "doc_title": "Vaccine A",
            "category": "thận trọng",
            "header": "Lưu ý",
            "content_type": "text",
            "text": "Hoãn tiêm nếu trẻ đang sốt cao cấp tính.",
            "metadata": {"has_table": False, "source_file": "a.md"},
        },
        {
            "chunk_id": "c3",
            "doc_id": "d2",
            "doc_title": "Vaccine B",
            "category": "tác dụng không mong muốn",
            "header": "Bảng tỷ lệ phản ứng",
            "content_type": "table",
            "text": "| Phản ứng | Tỷ lệ |\n| Sốt | 12% |\n| Đau chỗ tiêm | 25% |",
            "metadata": {"has_table": True, "source_file": "b.md"},
        },
        {
            "chunk_id": "c4",
            "doc_id": "d2",
            "doc_title": "Vaccine B",
            "category": "liều lượng",
            "header": "Lịch tiêm",
            "content_type": "text",
            "text": "Tiêm 2 mũi cách nhau ít nhất 4 tuần.",
            "metadata": {"has_table": False, "source_file": "b.md"},
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))


def test_allocate_strata_sums_and_margins() -> None:
    for limit in (10, 50, 100):
        strata = gen.allocate_strata(limit)
        assert sum(strata.values()) == limit

    strata_100 = gen.allocate_strata(100)
    ans_totals = {
        a: sum(v for (ka, _, _), v in strata_100.items() if ka == a)
        for a in gen.ANSWERABILITY_SHARES
    }
    comp_totals = {
        c: sum(v for (_, kc, _), v in strata_100.items() if kc == c)
        for c in gen.COMPLEXITY_SHARES
    }
    style_totals = {
        s: sum(v for (_, _, ks), v in strata_100.items() if ks == s)
        for s in gen.QUERY_STYLE_SHARES
    }
    assert ans_totals == {"ANSWERABLE": 70, "PARTIALLY_ANSWERABLE": 30}
    assert comp_totals == {"SINGLE_HOP": 50, "MULTI_HOP": 30, "TABLE_REASONING": 20}
    assert style_totals == {"PATIENT_STYLE": 60, "CLINICAL_STYLE": 40}


def test_allocate_strata_edge_cases() -> None:
    assert gen.allocate_strata(0) == {}
    with pytest.raises(ValueError):
        gen.allocate_strata(-1)


def test_load_chunks_and_build_pools(tmp_path: Path) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    _write_sample_parquet(parquet_path)

    records = gen.load_chunks_from_parquet(parquet_path)
    assert len(records) == 4
    pools = gen.build_context_pools(records, random.Random(42))

    assert len(pools["SINGLE_HOP"]) == 3
    assert len(pools["TABLE_REASONING"]) == 1
    assert pools["TABLE_REASONING"][0].chunk_ids == ["c3"]
    assert len(pools["MULTI_HOP"]) >= 1
    assert all(len(g.chunks) >= 2 for g in pools["MULTI_HOP"])


def test_lightrag_config_strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    monkeypatch.delenv("LIGHTRAG_BASE_URL", raising=False)
    monkeypatch.delenv("LIGHTRAG_LLM", raising=False)
    monkeypatch.setenv("LIGHTRAG_MODEL", "should-be-ignored")

    with pytest.raises(ValueError, match="LIGHTRAG_API_KEY"):
        gen.LightRAGConfig.from_env()

    monkeypatch.setenv("LIGHTRAG_API_KEY", "sk-test")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct")

    cfg = gen.LightRAGConfig.from_env()
    assert cfg.model == "qwen/qwen-2.5-7b-instruct"


def test_styling_and_evolution_configs() -> None:
    for ans in gen.ANSWERABILITY_SHARES:
        for style in gen.QUERY_STYLE_SHARES:
            sc = gen.build_styling_config(ans, style)
            assert sc.scenario
            assert sc.task
            assert sc.input_format
            assert sc.expected_output_format

    for comp in gen.COMPLEXITY_SHARES:
        ec = gen.build_evolution_config(comp)
        assert ec.num_evolutions == 1
        assert ec.evolutions


def test_generate_dataset_with_mocked_synthesizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "evaluation" / "eval_dataset.json"
    _write_sample_parquet(parquet_path)

    monkeypatch.setenv("LIGHTRAG_API_KEY", "sk-test")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct")

    class FakeSynthesizer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate_goldens_from_contexts(
            self,
            *,
            contexts: list[list[str]],
            include_expected_output: bool = True,
            max_goldens_per_context: int = 1,
            _send_data: bool = False,
        ) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    input=f"Câu hỏi mẫu {idx}",
                    expected_output=f"Trả lời mẫu {idx}",
                    context=ctx,
                )
                for idx, ctx in enumerate(contexts)
            ]

    with patch.object(gen, "Synthesizer", FakeSynthesizer):
        payload = gen.generate_dataset(
            chunks_path=parquet_path,
            output_path=output_path,
            limit=20,
            seed=42,
            dry_run=False,
        )

    assert output_path.is_file()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["metadata"]["total_generated"] == 20
    assert saved["metadata"]["model"] == "qwen/qwen-2.5-7b-instruct"
    assert len(saved["goldens"]) == 20
    assert len(payload["goldens"]) == 20

    first = saved["goldens"][0]
    assert set(first.keys()) == {"input", "expected_output", "context", "additional_metadata"}
    assert set(first["additional_metadata"].keys()) == {
        "query_id",
        "answerability",
        "complexity",
        "multi_hop_type",
        "query_style",
        "source_chunk_ids",
        "doc_ids",
        "doc_titles",
    }


@pytest.mark.asyncio
async def test_lightrag_llm_omits_response_format_and_parses_schema() -> None:
    from pydantic import BaseModel

    class DummySchema(BaseModel):
        data: list[str]

    captured_kwargs: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='```json\n{"data": ["ok"]}\n```'))]
            )

    class FakeAsyncCompletions:
        async def create(self, **kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"data": ["async_ok"]}'))]
            )

    cfg = gen.LightRAGConfig(
        api_key="sk-test",
        base_url="https://example.com/v1",
        model="gpt-oss-120b",
    )
    llm = gen.LightRAGDeepEvalLLM(cfg)
    llm._sync_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    llm._async_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeAsyncCompletions()))

    sync_res = llm.generate("prompt", schema=DummySchema)
    async_res = await llm.a_generate("prompt", schema=DummySchema)

    assert isinstance(sync_res, DummySchema) and sync_res.data == ["ok"]
    assert isinstance(async_res, DummySchema) and async_res.data == ["async_ok"]
    assert len(captured_kwargs) == 2
    assert all("response_format" not in kw for kw in captured_kwargs)


def test_parse_schema_wraps_plain_text_for_response_schema() -> None:
    from deepeval.synthesizer.schema import Response

    result = gen.LightRAGDeepEvalLLM._parse_schema(
        "Trẻ 2 tháng tuổi có tiêm được vaccine này không?",
        Response,
    )
    assert isinstance(result, Response)
    assert result.response == "Trẻ 2 tháng tuổi có tiêm được vaccine này không?"


def test_styling_config_enforces_vaccine_name_and_cross_vaccine() -> None:
    from deepeval.synthesizer.config import Evolution

    for ans in ("ANSWERABLE", "PARTIALLY_ANSWERABLE"):
        for style in gen.QUERY_STYLE_SHARES:
            sc_single = gen.build_styling_config(ans, style, "SINGLE_HOP", "single")
            assert "tên vaccine" in sc_single.input_format.lower()
            assert "thiếu chủ ngữ" not in sc_single.input_format.lower()

            sc_cross = gen.build_styling_config(
                ans, style, "MULTI_HOP", "cross_vaccine"
            )
            assert "cả 2 vaccine" in sc_cross.input_format.lower()
            assert "so sánh" in sc_cross.task.lower()

    ec_cross = gen.build_evolution_config("MULTI_HOP", pair_type="cross_vaccine")
    assert Evolution.COMPARATIVE in ec_cross.evolutions

    sc_patient = gen.build_styling_config("ANSWERABLE", "PATIENT_STYLE", "SINGLE_HOP", "single")
    sc_clinical = gen.build_styling_config("ANSWERABLE", "CLINICAL_STYLE", "SINGLE_HOP", "single")
    assert "từ ngữ phổ thông đời thường" in sc_patient.input_format.lower()
    assert "từ ngữ phổ thông đời thường" not in sc_clinical.input_format.lower()


def test_smart_multi_hop_pairing_and_overdose_filter() -> None:
    records = [
        gen.ChunkRecord("a1", "d1", "Vaccine A", "chỉ định", "Chỉ định", "text", False, "Chỉ định cho trẻ."),
        gen.ChunkRecord("a2", "d1", "Vaccine A", "chống chỉ định", "Chống chỉ định", "text", False, "Không tiêm nếu dị ứng."),
        gen.ChunkRecord("a3", "d1", "Vaccine A", "dược lực học", "5.1 Dược lực học", "text", False, "Nghiên cứu miễn dịch."),
        gen.ChunkRecord("a4", "d1", "Vaccine A", "quá liều", "Quá liều", "text", False, "Chưa có trường hợp dùng quá liều nào được báo cáo."),
        gen.ChunkRecord("a5", "d1", "Vaccine A", "quá liều", "Bảng 1", "table", True, "Dữ liệu bảng không liên quan quá liều " * 15),
        gen.ChunkRecord("b1", "d2", "Vaccine B", "chỉ định", "Chỉ định", "text", False, "Chỉ định cho người lớn."),
        gen.ChunkRecord("b2", "d2", "Vaccine B", "cảnh báo và thận trọng", "Thận trọng", "text", False, "Theo dõi sau tiêm."),
        gen.ChunkRecord(
            "b3",
            "d2",
            "Vaccine B",
            "quá liều",
            "4.9 Quá liều",
            "text",
            False,
            "Đã có báo cáo dùng quá liều ở trẻ em; cần theo dõi chức năng sinh tồn và xử trí triệu chứng theo phác đồ lâm sàng hiện hành. " * 3,
        ),
    ]

    pools = gen.build_context_pools(records, random.Random(42))
    mh = pools["MULTI_HOP"]
    assert mh
    assert any(g.pair_type == "same_doc" for g in mh)
    assert any(g.pair_type == "cross_vaccine" for g in mh)

    for group in mh:
        cats = {c.category for c in group.chunks}
        assert "dược lực học" not in cats
        chunk_ids = set(group.chunk_ids)
        assert "a4" not in chunk_ids  # stub overdose filtered out
        assert "a5" not in chunk_ids  # mislabeled overdose header filtered out
        if group.pair_type == "cross_vaccine":
            assert len(group.doc_ids) == 2
            assert len(cats) == 1
        elif group.pair_type == "same_doc":
            assert len(group.doc_ids) == 1
            assert len(cats) == 2


def test_generate_dataset_sub_batches_multi_hop_by_pair_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "evaluation" / "eval_dataset.json"
    rows = [
        {"chunk_id": "a1", "doc_id": "d1", "doc_title": "Vaccine A", "category": "chỉ định", "header": "Chỉ định", "content_type": "text", "text": "Chỉ định A.", "metadata": {"has_table": False, "source_file": "a.md"}},
        {"chunk_id": "a2", "doc_id": "d1", "doc_title": "Vaccine A", "category": "chống chỉ định", "header": "Chống chỉ định", "content_type": "text", "text": "Chống chỉ định A.", "metadata": {"has_table": False, "source_file": "a.md"}},
        {"chunk_id": "b1", "doc_id": "d2", "doc_title": "Vaccine B", "category": "chỉ định", "header": "Chỉ định", "content_type": "text", "text": "Chỉ định B.", "metadata": {"has_table": False, "source_file": "b.md"}},
        {"chunk_id": "b2", "doc_id": "d2", "doc_title": "Vaccine B", "category": "bảo quản", "header": "Bảng bảo quản", "content_type": "table", "text": "| Nhiệt độ | 2-8C |", "metadata": {"has_table": True, "source_file": "b.md"}},
    ]
    pq.write_table(pa.Table.from_pylist(rows), str(parquet_path))

    monkeypatch.setenv("LIGHTRAG_API_KEY", "sk-test")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct")

    class FakeSynthesizer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate_goldens_from_contexts(
            self,
            *,
            contexts: list[list[str]],
            include_expected_output: bool = True,
            max_goldens_per_context: int = 1,
            _send_data: bool = False,
        ) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(input=f"Q {i}", expected_output=f"A {i}", context=ctx)
                for i, ctx in enumerate(contexts)
            ]

    with patch.object(gen, "Synthesizer", FakeSynthesizer):
        payload = gen.generate_dataset(
            chunks_path=parquet_path,
            output_path=output_path,
            limit=20,
            seed=42,
            dry_run=False,
        )

    mh_items = [
        g for g in payload["goldens"] if g["additional_metadata"]["complexity"] == "MULTI_HOP"
    ]
    non_mh_items = [
        g for g in payload["goldens"] if g["additional_metadata"]["complexity"] != "MULTI_HOP"
    ]
    assert mh_items
    assert {g["additional_metadata"]["multi_hop_type"] for g in mh_items} == {
        "same_doc",
        "cross_vaccine",
    }
    assert all(g["additional_metadata"]["multi_hop_type"] is None for g in non_mh_items)


def test_generate_dataset_matches_out_of_order_and_partial_async_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    output_path = tmp_path / "evaluation" / "eval_dataset.json"
    _write_sample_parquet(parquet_path)

    records = gen.load_chunks_from_parquet(parquet_path)
    expected_by_ctx = {r.format_context(): r for r in records}

    monkeypatch.setenv("LIGHTRAG_API_KEY", "sk-test")
    monkeypatch.setenv("LIGHTRAG_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LIGHTRAG_LLM", "qwen/qwen-2.5-7b-instruct")

    class ReversingPartialSynthesizer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def generate_goldens_from_contexts(
            self,
            *,
            contexts: list[list[str]],
            include_expected_output: bool = True,
            max_goldens_per_context: int = 1,
            _send_data: bool = False,
        ) -> list[SimpleNamespace]:
            # Simulate async completion order (reversed) and a dropped item when batch > 1.
            ordered = list(reversed(contexts))
            if len(ordered) > 1:
                ordered = ordered[:-1]
            return [
                SimpleNamespace(
                    input=f"Q {i}",
                    expected_output=f"A {i}",
                    context=ctx,
                )
                for i, ctx in enumerate(ordered)
            ]

    with patch.object(gen, "Synthesizer", ReversingPartialSynthesizer):
        payload = gen.generate_dataset(
            chunks_path=parquet_path,
            output_path=output_path,
            limit=20,
            seed=42,
            dry_run=False,
        )

    assert payload["goldens"]
    for item in payload["goldens"]:
        matched = [expected_by_ctx[c] for c in item["context"]]
        meta = item["additional_metadata"]
        assert meta["source_chunk_ids"] == [m.chunk_id for m in matched]
        assert meta["doc_ids"] == list(dict.fromkeys(m.doc_id for m in matched))
        assert meta["doc_titles"] == list(dict.fromkeys(m.doc_title for m in matched))


def test_repair_dataset_metadata_restores_aligned_ids(tmp_path: Path) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    broken_path = tmp_path / "broken.json"
    fixed_path = tmp_path / "fixed.json"
    _write_sample_parquet(parquet_path)

    records = gen.load_chunks_from_parquet(parquet_path)
    c1, c2, c3, c4 = records

    broken_payload = {
        "metadata": {"limit": 2, "total_generated": 2},
        "goldens": [
            {
                "input": "Câu hỏi 1",
                "expected_output": "Trả lời 1",
                "context": [c1.format_context()],
                "additional_metadata": {
                    "query_id": "eval_0001",
                    "answerability": "ANSWERABLE",
                    "complexity": "SINGLE_HOP",
                    "multi_hop_type": None,
                    "query_style": "PATIENT_STYLE",
                    "source_chunk_ids": [c3.chunk_id],
                    "doc_ids": [c3.doc_id],
                    "doc_titles": [c3.doc_title],
                },
            },
            {
                "input": "Câu hỏi 2",
                "expected_output": "Trả lời 2",
                "context": [c1.format_context(), c4.format_context()],
                "additional_metadata": {
                    "query_id": "eval_0002",
                    "answerability": "ANSWERABLE",
                    "complexity": "MULTI_HOP",
                    "multi_hop_type": "cross_vaccine",
                    "query_style": "CLINICAL_STYLE",
                    "source_chunk_ids": [c2.chunk_id, c3.chunk_id],
                    "doc_ids": [c2.doc_id, c3.doc_id],
                    "doc_titles": [c2.doc_title, c3.doc_title],
                },
            },
        ],
    }
    broken_path.write_text(json.dumps(broken_payload, ensure_ascii=False), encoding="utf-8")

    stats = gen.repair_dataset_metadata(
        dataset_path=broken_path,
        chunks_path=parquet_path,
        output_path=fixed_path,
    )
    assert stats == {"total": 2, "updated": 2, "unchanged": 0}

    repaired = json.loads(fixed_path.read_text(encoding="utf-8"))
    g1, g2 = repaired["goldens"]
    assert g1["input"] == "Câu hỏi 1"
    assert g1["expected_output"] == "Trả lời 1"
    assert g1["additional_metadata"]["source_chunk_ids"] == [c1.chunk_id]
    assert g1["additional_metadata"]["doc_ids"] == [c1.doc_id]
    assert g1["additional_metadata"]["doc_titles"] == [c1.doc_title]

    assert g2["additional_metadata"]["source_chunk_ids"] == [c1.chunk_id, c4.chunk_id]
    assert g2["additional_metadata"]["doc_ids"] == [c1.doc_id, c4.doc_id]
    assert g2["additional_metadata"]["doc_titles"] == [c1.doc_title, c4.doc_title]


def test_repair_dataset_metadata_fails_on_unmatched_or_ambiguous_context(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "chunks.parquet"
    rows = [
        {"chunk_id": "c1", "doc_id": "d1", "doc_title": "Vaccine A", "category": "liều", "header": "H", "content_type": "text", "text": "Dup text", "metadata": {"has_table": False}},
        {"chunk_id": "c2", "doc_id": "d2", "doc_title": "Vaccine A", "category": "liều", "header": "H", "content_type": "text", "text": "Dup text", "metadata": {"has_table": False}},
    ]
    pq.write_table(pa.Table.from_pylist(rows), str(parquet_path))
    dup_ctx = gen.load_chunks_from_parquet(parquet_path)[0].format_context()

    unmatched_path = tmp_path / "unmatched.json"
    unmatched_path.write_text(
        json.dumps(
            {
                "goldens": [
                    {
                        "input": "Q",
                        "expected_output": "A",
                        "context": ["[Unknown]\nMissing"],
                        "additional_metadata": {"query_id": "eval_0001"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unmatched"):
        gen.repair_dataset_metadata(dataset_path=unmatched_path, chunks_path=parquet_path)

    ambiguous_path = tmp_path / "ambiguous.json"
    ambiguous_path.write_text(
        json.dumps(
            {
                "goldens": [
                    {
                        "input": "Q",
                        "expected_output": "A",
                        "context": [dup_ctx],
                        "additional_metadata": {"query_id": "eval_0001"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        gen.repair_dataset_metadata(dataset_path=ambiguous_path, chunks_path=parquet_path)
