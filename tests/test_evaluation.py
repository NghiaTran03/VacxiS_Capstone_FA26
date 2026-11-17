"""Tests for the DeepEval-based VacxiS evaluation module."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from openai import OpenAI

from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.models import DeepEvalBaseLLM
from deepeval.tracing.api import MetricData

from vacxis_capstone_fa26.evaluation import (
    VacxisRAGEvaluator,
    align_results,
    answer_fields,
    load_chunk_texts,
    load_goldens,
    retrieval_context_from_answer,
    validate_query_ids,
)
from vacxis_capstone_fa26.evaluation.cli import _save_parquet
from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_DATASET = PROJECT_ROOT / "evaluation" / "eval_dataset_300_final.json"
FLAT_DATASET = PROJECT_ROOT / "test_dataset.json"
CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"


class _StubJudge(DeepEvalBaseLLM):
    """Offline judge that satisfies the abstract methods without a live client."""

    def __init__(self) -> None:
        super().__init__(model="stub-judge")

    def load_model(self) -> OpenAI:
        return OpenAI(api_key="x", base_url="http://localhost")

    def get_model_name(self) -> str:
        return "stub-judge"

    def generate(self, prompt: str, schema=None):
        return "ok"

    async def a_generate(self, prompt: str, schema=None):
        return "ok"


def _make_evaluator(enable_geval: bool = True) -> VacxisRAGEvaluator:
    """An evaluator with a stub judge so no network or key is needed."""
    return VacxisRAGEvaluator(
        judge_model=_StubJudge(), threshold=0.5, enable_geval=enable_geval
    )


def _fake_metric(name: str, score: float, success: bool = True) -> MetricData:
    """Build a MetricData row the way DeepEval's evaluate() does."""
    return MetricData(
        name=name,
        threshold=0.5,
        success=success,
        score=score,
        reason="stub reason",
        strict_mode=False,
        flaky=False,
        evaluation_model="stub-judge",
        error=None,
        evaluation_cost=0.0,
        input_tokens=0,
        output_tokens=0,
        verbose_logs=None,
        expectations_data=None,
    )


def _fake_result(num_cases: int, metric_names: list[str]) -> EvaluationResult:
    """Fabricate a DeepEval EvaluationResult with one TestResult per case."""
    test_results = []
    for i in range(num_cases):
        metrics_data = [_fake_metric(name, 0.5 + 0.01 * i) for name in metric_names]
        test_results.append(
            TestResult(
                name=f"case{i}",
                success=True,
                conversational=False,
                index=i,
                input=f"q{i}",
                actual_output=f"a{i}",
                expected_output=f"e{i}",
                retrieval_context=["ctx"],
                metadata={"query_id": f"qid{i}"},
                metrics_data=metrics_data,
            )
        )
    return EvaluationResult(
        test_results=test_results, confident_link=None, test_run_id="stub-run"
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def test_load_goldens_main_dataset():
    records = load_goldens(MAIN_DATASET)
    assert len(records) == 300
    first = records[0]
    assert first.query_id == "eval_0001"
    assert first.answerability == "ANSWERABLE"
    assert first.complexity == "SINGLE_HOP"
    assert first.query_style == "PATIENT_STYLE"
    assert first.query.strip()
    assert first.expected_output.strip()
    assert first.context and first.context[0].strip()


def test_load_goldens_flat_dataset():
    records = load_goldens(FLAT_DATASET)
    assert len(records) == 460
    first = records[0]
    assert first.query_id == "VAX_Q_0001"
    assert first.answerability == "UNANSWERABLE"
    assert first.difficulty == "MEDIUM"
    assert first.query_type == "ADVERSE_EFFECT"
    # The flat RELEVANT_CHUNKS carry a TEXT field, normalized to context strings.
    has_context = any(r.context for r in records)
    assert has_context


def test_load_chunk_texts():
    texts = load_chunk_texts(CHUNKS_PATH)
    assert texts, "chunk text store should not be empty"
    assert all(isinstance(v, str) and v.strip() for v in texts.values())


def test_load_chunk_texts_missing_returns_empty(tmp_path: Path):
    assert load_chunk_texts(tmp_path / "nope.parquet") == {}


def test_retrieval_context_from_answer(tmp_path: Path):
    chunk_texts = {"abc_c_001": "text A", "abc_c_002": "text B"}
    hydrated = retrieval_context_from_answer(
        {"retrieved_chunk_ids": ["abc_c_001", "abc_c_002", "missing"]},
        chunk_texts,
    )
    assert hydrated == ["text A", "text B"]
    # Unknown ids are dropped.
    assert retrieval_context_from_answer({"retrieved_chunk_ids": ["nope"]}, chunk_texts) == []


def test_answer_fields_unwrapped():
    fields = answer_fields(
        {
            "answer": "a",
            "is_refusal": True,
            "retrieved_chunk_ids": ["c1", "c2"],
        }
    )
    assert fields == {
        "actual_output": "a",
        "is_refusal": True,
        "retrieved_chunk_ids": ["c1", "c2"],
    }


def test_answer_fields_wrapped_in_generated_answer():
    fields = answer_fields(
        {"GENERATED_ANSWER": {"answer": "a", "is_refusal": False, "retrieved_chunk_ids": ["c"]}}
    )
    assert fields["actual_output"] == "a"
    assert fields["is_refusal"] is False
    assert fields["retrieved_chunk_ids"] == ["c"]


# ---------------------------------------------------------------------------
# Alignment & IDs
# ---------------------------------------------------------------------------


def test_validate_query_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_query_ids([{"QUERY_ID": "a"}, {"QUERY_ID": "a"}])


def test_validate_query_ids_rejects_missing():
    with pytest.raises(ValueError):
        validate_query_ids([{"QUERY": "q"}])


def test_validate_query_ids_reads_deep_metadata():
    by_id = validate_query_ids([{"additional_metadata": {"query_id": "eval_1"}}])
    assert "eval_1" in by_id


def test_align_results_preserves_result_order():
    dataset = [
        {"QUERY_ID": "B", "QUERY": "qb"},
        {"QUERY_ID": "A", "QUERY": "qa"},
    ]
    results = [
        {"QUERY_ID": "B", "GENERATED_ANSWER": {"answer": "xb"}},
        {"QUERY_ID": "A", "GENERATED_ANSWER": {"answer": "xa"}},
    ]
    aligned = align_results(dataset, results)
    assert [item["QUERY_ID"] for item in aligned] == ["B", "A"]


def test_align_results_mismatched_ids_raises():
    dataset = [{"QUERY_ID": "A"}]
    results = [{"QUERY_ID": "B"}]
    with pytest.raises(ValueError, match="disagree"):
        align_results(dataset, results)


# ---------------------------------------------------------------------------
# Metrics & test-case construction
# ---------------------------------------------------------------------------


def test_build_metrics_default_includes_geval():
    evaluator = _make_evaluator(enable_geval=True)
    names = [m.__name__ for m in evaluator.build_metrics()]
    assert len(names) == 6
    assert "Answer Relevancy" in names
    assert "Faithfulness" in names
    assert any("Contextual" in n for n in names)


def test_build_metrics_without_geval():
    evaluator = _make_evaluator(enable_geval=False)
    assert len(evaluator.build_metrics()) == 5


# ---------------------------------------------------------------------------
# Summary construction, refusal correctness, and exports
# ---------------------------------------------------------------------------


def _small_dataset_and_results(chunk_texts: dict[str, str]):
    """Two flat goldens plus matching precomputed results."""
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "Liều dùng Verorab?",
            "QUERY_TYPE": "DOSAGE",
            "QUERY_DIFFICULTY": "EASY",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "QUERY_STYLE": "PATIENT_STYLE",
            "ANSWERABILITY": "ANSWERABLE",
            "REFERENCE_ANSWER": "Tiêm 0.5ml.",
            "RELEVANT_CHUNKS": [],
        },
        {
            "QUERY_ID": "Q2",
            "QUERY": "Có chữa được sốt rét không?",
            "QUERY_TYPE": "INDICATION",
            "QUERY_DIFFICULTY": "HARD",
            "QUERY_COMPLEXITY": "SINGLE_HOP",
            "QUERY_STYLE": "CLINICAL_STYLE",
            "ANSWERABILITY": "UNANSWERABLE",
            "REFERENCE_ANSWER": "Không có thông tin.",
            "RELEVANT_CHUNKS": [],
        },
    ]
    results = [
        {
            "QUERY_ID": "Q1",
            "GENERATED_ANSWER": {
                "answer": "Tiêm 0.5ml [verorab_c_001].",
                "citations": ["verorab_c_001"],
                "invalid_citations": [],
                "retrieved_chunk_ids": list(chunk_texts.keys())[:1],
                "is_refusal": False,
            },
        },
        {
            "QUERY_ID": "Q2",
            "GENERATED_ANSWER": {
                "answer": "Tài liệu không đề cập đến sốt rét.",
                "citations": [],
                "invalid_citations": [],
                "retrieved_chunk_ids": [],
                "is_refusal": True,
            },
        },
    ]
    return dataset, results


def test_evaluate_results_builds_full_summary(tmp_path: Path, monkeypatch):
    evaluator = _make_evaluator(enable_geval=True)
    chunk_texts = {"verorab_c_001": "text about Verorab dosing"}
    dataset, results = _small_dataset_and_results(chunk_texts)

    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    monkeypatch.setattr(
        evaluator,
        "_run_deepeval",
        lambda cases, **kw: _fake_result(len(cases), metric_names),
    )

    summary = evaluator.evaluate_results(dataset, results, chunk_texts=chunk_texts)

    assert summary.num_queries == 2
    # 5 RAG + GEval + refusal indicator.
    assert len(summary.metric_names) == 7
    assert "refusal_correctness" in summary.metric_names
    # Both answers align with their ANSWERABILITY ground truth.
    assert summary.overall_metrics["refusal_correctness"] == 1.0
    assert "ANSWERABLE" in summary.by_answerability
    assert "UNANSWERABLE" in summary.by_answerability
    assert "EASY" in summary.by_difficulty
    # Per-query records carry the judge reasons and hydrated context.
    assert summary.per_query_results[0]["query_id"] == "Q1"
    assert "Answer Relevancy" in summary.per_query_results[0]["metric_results"]

    # Exports.
    json_path = tmp_path / "summary.json"
    VacxisRAGEvaluator.save_json(summary, json_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["num_queries"] == 2
    assert loaded["overall_metrics"]["refusal_correctness"] == 1.0

    md_path = tmp_path / "report.md"
    VacxisRAGEvaluator.save_markdown(summary, md_path)
    markdown = md_path.read_text(encoding="utf-8")
    assert "# VacxiS DeepEval Report" in markdown
    assert "Answer Relevancy" in markdown


def test_refusal_mismatch_scores_zero():
    evaluator = _make_evaluator(enable_geval=False)
    # An ANSWERABLE question answered with a refusal should score 0.
    dataset = [
        {
            "QUERY_ID": "Q1",
            "QUERY": "q",
            "ANSWERABILITY": "ANSWERABLE",
            "REFERENCE_ANSWER": "a",
        }
    ]
    results = [
        {"QUERY_ID": "Q1", "GENERATED_ANSWER": {"answer": "x", "is_refusal": True}}
    ]
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(1, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    assert summary.overall_metrics["refusal_correctness"] == 0.0


def test_save_parquet_round_trip(tmp_path: Path):
    evaluator = _make_evaluator(enable_geval=True)
    dataset, results = _small_dataset_and_results({"verorab_c_001": "t"})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(2, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})

    parquet_path = tmp_path / "per_query.parquet"
    _save_parquet(summary, parquet_path)
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert len(table) == 2
    assert "query_id" in table.column_names
    assert "metric_refusal_correctness" in table.column_names
    assert "meta_answerability" in table.column_names


def test_summary_to_dict_is_json_safe(tmp_path: Path):
    evaluator = _make_evaluator(enable_geval=True)
    dataset, results = _small_dataset_and_results({})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]
    evaluator._run_deepeval = lambda cases, **kw: _fake_result(2, metric_names)
    summary = evaluator.evaluate_results(dataset, results, chunk_texts={})
    # A JSON round-trip must not raise (no sets, no non-serializable objects).
    serialized = json.dumps(summary.to_dict(), ensure_ascii=False)
    assert "refusal_correctness" in serialized


# ---------------------------------------------------------------------------
# DeepEval cache (resume) wiring
# ---------------------------------------------------------------------------


def test_run_deepeval_defaults_read_the_cache():
    """_run_deepeval must read and write the cache so stopped runs can resume."""
    from vacxis_capstone_fa26.evaluation import evaluator as evaluator_module
    from vacxis_capstone_fa26.evaluation.datasets import to_record

    captured = {}

    def _fake_deepeval_evaluate(**kwargs):
        captured.update(kwargs)
        return _fake_result(1, [])

    evaluator = _make_evaluator()
    record = to_record(
        {"QUERY_ID": "Q1", "QUERY": "q", "REFERENCE_ANSWER": "a"}
    )
    info = evaluator._case_info(
        {"actual_output": "a", "is_refusal": False, "retrieved_chunk_ids": []}, {}
    )
    cases = [evaluator._build_test_case(record, info)]

    original = evaluator_module.deepeval_evaluate
    evaluator_module.deepeval_evaluate = _fake_deepeval_evaluate
    try:
        evaluator._run_deepeval(cases, max_concurrent=1)
    finally:
        evaluator_module.deepeval_evaluate = original

    cache_config = captured["cache_config"]
    assert cache_config.write_cache is True
    assert cache_config.use_cache is True


def test_evaluate_live_forwards_cache_flags():
    """evaluate_live must forward use_cache/write_cache down to _run_deepeval."""
    evaluator = _make_evaluator()
    dataset, results = _small_dataset_and_results({})
    metric_names = [m.__name__ for m in evaluator.build_metrics()]

    seen: dict = {}

    def _capture(cases, **kw):
        seen.update(kw)
        return _fake_result(len(cases), metric_names)

    evaluator._run_deepeval = _capture
    evaluator.evaluate_live(
        dataset,
        lambda query, golden: {"answer": "a", "is_refusal": False, "retrieved_chunk_ids": []},
        chunk_texts={},
        use_cache=False,
        write_cache=False,
    )
    assert seen["use_cache"] is False
    assert seen["write_cache"] is False


# ---------------------------------------------------------------------------
# Judge-timeout CLI flag
# ---------------------------------------------------------------------------


def test_apply_judge_timeout_sets_env_override(monkeypatch):
    from vacxis_capstone_fa26.evaluation import cli as eval_cli

    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    args = argparse.Namespace(judge_timeout=1200.0)
    eval_cli._apply_judge_timeout(args)
    assert os.environ.get("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE") == "1200.0"


def test_apply_judge_timeout_noop_when_unset(monkeypatch):
    from vacxis_capstone_fa26.evaluation import cli as eval_cli

    monkeypatch.delenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", raising=False)
    args = argparse.Namespace(judge_timeout=None)
    eval_cli._apply_judge_timeout(args)
    assert "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE" not in os.environ
