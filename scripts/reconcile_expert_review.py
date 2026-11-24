#!/usr/bin/env python3
"""Prepare expert-review deltas and merge updated per-query evaluation rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "evaluation/eval_dataset_300_final_fixed.json"
DEFAULT_REVIEW = ROOT / "review-web/test_dataset_verified.json"
DEFAULT_OUT_DIR = ROOT / "evaluation/expert_review"
DEFAULT_PIPELINES = {
    "hybrid": ROOT / "evaluation/gen_hybrid.json",
    "table_dual_hybrid": ROOT / "evaluation/gen_table_dual_hybrid.json",
    "table_dual_lightrag": ROOT / "evaluation/gen_table_dual_lightrag.json",
}

GOLDEN_FIELDS = ("input", "expected_output", "context")
LABEL_FIELDS = (
    "query_id",
    "answerability",
    "complexity",
    "multi_hop_type",
    "query_style",
    "source_chunk_ids",
    "doc_ids",
    "doc_titles",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def query_id(row: dict[str, Any]) -> str:
    metadata = row.get("additional_metadata") or {}
    raw = metadata.get("query_id") or row.get("query_id") or row.get("QUERY_ID")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Every row must have a nonblank query_id.")
    return raw


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = query_id(row)
        if qid in indexed:
            raise ValueError(f"Duplicate query_id {qid!r} in {label}.")
        indexed[qid] = row
    return indexed


def semantic_changes(base: dict[str, Any], reviewed: dict[str, Any]) -> list[str]:
    changed = [field for field in GOLDEN_FIELDS if base.get(field) != reviewed.get(field)]
    base_meta = base.get("additional_metadata") or {}
    review_meta = reviewed.get("additional_metadata") or {}
    changed.extend(
        f"additional_metadata.{field}"
        for field in LABEL_FIELDS
        if base_meta.get(field) != review_meta.get(field)
    )
    return changed


def overlay_review(base: dict[str, Any], reviewed: dict[str, Any]) -> dict[str, Any]:
    """Copy benchmark fields while retaining generated outputs and review privacy."""
    merged = dict(base)
    for field in GOLDEN_FIELDS:
        if field in reviewed:
            merged[field] = reviewed[field]

    metadata = dict(base.get("additional_metadata") or {})
    reviewed_metadata = reviewed.get("additional_metadata") or {}
    for field in LABEL_FIELDS:
        if field in reviewed_metadata:
            metadata[field] = reviewed_metadata[field]
    if metadata:
        merged["additional_metadata"] = metadata
    return merged


def delta_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distribution: dict[str, dict[str, int]] = {
        key: defaultdict(int)
        for key in ("answerability", "complexity", "query_style")
    }
    for row in rows:
        metadata = row.get("additional_metadata") or {}
        for key in distribution:
            value = metadata.get(key)
            if value:
                distribution[key][str(value)] += 1
    return {key: dict(values) for key, values in distribution.items()}


def prepare(args: argparse.Namespace) -> None:
    base_payload = read_json(args.base)
    review_payload = read_json(args.review)
    base_rows = base_payload.get("goldens")
    review_rows = review_payload.get("goldens")
    if not isinstance(base_rows, list) or not isinstance(review_rows, list):
        raise ValueError("Both input files must contain a 'goldens' list.")

    base_by_id = index_rows(base_rows, str(args.base))
    review_by_id = index_rows(review_rows, str(args.review))
    if base_by_id.keys() != review_by_id.keys():
        missing_review = sorted(base_by_id.keys() - review_by_id.keys())
        extra_review = sorted(review_by_id.keys() - base_by_id.keys())
        raise ValueError(
            f"Dataset IDs differ; missing from review={missing_review[:10]}, "
            f"extra in review={extra_review[:10]}."
        )

    changed_ids = {
        qid
        for qid, base in base_by_id.items()
        if semantic_changes(base, review_by_id[qid])
    }
    delta_rows = [
        overlay_review(base_by_id[query_id(row)], row)
        for row in review_rows
        if query_id(row) in changed_ids
    ]

    reviewed_full_rows = [
        overlay_review(row, review_by_id[query_id(row)]) for row in base_rows
    ]
    reviewed_full = dict(base_payload)
    reviewed_full["goldens"] = reviewed_full_rows
    reviewed_full["metadata"] = dict(base_payload.get("metadata") or {})
    reviewed_full["metadata"]["expert_review_source"] = str(args.review)
    reviewed_full["metadata"]["expert_review_delta_count"] = len(changed_ids)

    write_json(args.out_dir / "expert_reviewed_dataset.json", reviewed_full)
    delta_metadata = dict(base_payload.get("metadata") or {})
    delta_metadata.update(
        {
            "expert_review_source": str(args.review),
            "limit": len(delta_rows),
            "total_generated": len(delta_rows),
            "delta_count": len(delta_rows),
            "distribution": delta_distribution(delta_rows),
        }
    )
    write_json(
        args.out_dir / "expert_review_delta.json",
        {"metadata": delta_metadata, "goldens": delta_rows},
    )

    print(f"Base and reviewed rows: {len(base_rows)}; semantic delta: {len(changed_ids)}")
    print("Changed fields exclude reviewer status, names, notes, and adjudication bookkeeping.")

    for name, path in args.pipeline:
        if not path.is_file():
            print(f"{name}: skipped; results file not found: {path}")
            continue
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Pipeline output must be a JSON list: {path}")
        result_by_id = index_rows(payload, str(path))
        unknown = result_by_id.keys() - base_by_id.keys()
        if unknown:
            raise ValueError(f"{path} contains unknown query IDs: {sorted(unknown)[:10]}")

        full_overlay = [
            overlay_review(row, review_by_id[query_id(row)]) for row in payload
        ]
        # Preserve the existing generation payload while updating all reviewed goldens.
        full_path = args.out_dir / f"{name}_expert_reviewed_results.json"
        delta_ids_present = changed_ids & result_by_id.keys()
        pipeline_delta_dataset = [
            overlay_review(base_by_id[qid], review_by_id[qid])
            for qid in changed_ids
            if qid in result_by_id
        ]
        pipeline_delta_results = [
            overlay_review(result_by_id[qid], review_by_id[qid])
            for qid in result_by_id
            if qid in changed_ids
        ]
        write_json(full_path, full_overlay)
        write_json(
            args.out_dir / f"{name}_delta_dataset.json",
            {"metadata": {"delta_count": len(pipeline_delta_dataset)}, "goldens": pipeline_delta_dataset},
        )
        write_json(args.out_dir / f"{name}_delta_results.json", pipeline_delta_results)
        missing = changed_ids - result_by_id.keys()
        state = "complete" if not missing else "INCOMPLETE"
        print(
            f"{name}: {state}; results={len(payload)}, "
            f"delta rows available={len(delta_ids_present)}/{len(changed_ids)}, "
            f"merged file={full_path}"
        )
        if missing:
            print(f"  Missing changed IDs: {', '.join(sorted(missing))}")


def merge_results(args: argparse.Namespace) -> None:
    """Replace only regenerated answers while preserving full reviewed rows."""
    base_rows = read_json(args.base_results)
    delta_rows = read_json(args.delta_results)
    if not isinstance(base_rows, list) or not isinstance(delta_rows, list):
        raise ValueError("Base and delta generation results must be JSON lists.")

    base_by_id = index_rows(base_rows, str(args.base_results))
    delta_by_id = index_rows(delta_rows, str(args.delta_results))
    unknown = delta_by_id.keys() - base_by_id.keys()
    if unknown:
        raise ValueError(f"Delta results contain IDs absent from base: {sorted(unknown)[:10]}.")

    dataset_payload = read_json(args.delta_dataset)
    dataset_rows = dataset_payload.get("goldens") if isinstance(dataset_payload, dict) else None
    if not isinstance(dataset_rows, list):
        raise ValueError("--delta-dataset must contain a 'goldens' list.")
    expected_ids = set(index_rows(dataset_rows, str(args.delta_dataset)))
    if delta_by_id.keys() != expected_ids:
        raise ValueError(
            "Delta result IDs differ from regenerated delta dataset: "
            f"missing={sorted(expected_ids - delta_by_id.keys())[:10]}, "
            f"extra={sorted(delta_by_id.keys() - expected_ids)[:10]}."
        )

    merged_rows = []
    for row in base_rows:
        qid = query_id(row)
        replacement = delta_by_id.get(qid)
        merged = dict(row)
        if replacement is not None:
            answer = replacement.get("GENERATED_ANSWER")
            if not isinstance(answer, dict):
                raise ValueError(f"Regenerated result {qid!r} has no GENERATED_ANSWER object.")
            merged["GENERATED_ANSWER"] = answer
        merged_rows.append(merged)

    write_json(args.out_results, merged_rows)
    print(f"Merged {len(delta_by_id)} regenerated answers into {len(merged_rows)} reviewed rows.")


def rounded(value: float) -> float:
    return round(value, 4)


def calculate_report(base: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = base.get("metric_names") or []
    threshold = float((base.get("metadata") or {}).get("threshold", 0.5))
    summary = dict(base)
    summary["num_queries"] = len(rows)
    summary["per_query_results"] = rows

    aggregates: dict[str, dict[str, Any]] = {}
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {
        axis: defaultdict(lambda: defaultdict(list))
        for axis in ("answerability", "complexity", "query_style", "difficulty")
    }
    group_counts: dict[str, dict[str, int]] = {
        axis: defaultdict(int) for axis in grouped
    }

    for row in rows:
        scores = row.get("metric_scores") or {}
        results = row.get("metric_results") or {}
        for axis in grouped:
            axis_value = str((row.get("metadata") or {}).get(axis) or "UNSPECIFIED")
            group_counts[axis][axis_value] += 1
            for name in metric_names:
                if name in scores:
                    grouped[axis][axis_value][name].append(float(scores[name]))

    for name in metric_names:
        values = [
            float((row.get("metric_scores") or {})[name])
            for row in rows
            if name in (row.get("metric_scores") or {})
        ]
        if not values:
            continue
        passed = sum(
            bool(((row.get("metric_results") or {}).get(name) or {}).get("success"))
            for row in rows
        )
        aggregates[name] = {
            "mean": rounded(sum(values) / len(values)),
            "min": rounded(min(values)),
            "max": rounded(max(values)),
            "count": len(values),
            "pass_rate": rounded(passed / len(values)),
        }

    summary["metric_aggregates"] = aggregates
    summary["overall_metrics"] = {name: item["mean"] for name, item in aggregates.items()}
    for axis in grouped:
        table: dict[str, dict[str, float]] = {}
        for key, count in group_counts[axis].items():
            item: dict[str, float] = {"count": float(count)}
            for name, values in grouped[axis][key].items():
                item[name] = rounded(sum(values) / len(values))
            table[key] = item
        summary[f"by_{axis}"] = table

    metadata = dict(base.get("metadata") or {})
    metadata["threshold"] = threshold
    if base.get("task") == "retrieval" or "scored_queries" in metadata:
        scored = sum(bool((row.get("metadata") or {}).get("ir_scored")) for row in rows)
        metadata["scored_queries"] = scored
        metadata["unscored_queries"] = len(rows) - scored
    summary["metadata"] = metadata
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# VacxiS DeepEval Report: {str(report.get('task', 'rag')).upper()}",
        f"Total evaluated queries: {report.get('num_queries', 0)}",
        "",
        "## Overall Metrics",
        "| Metric | Mean | Min | Max | Pass Rate | Count |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in report.get("metric_names", []):
        metric = (report.get("metric_aggregates") or {}).get(name)
        if metric:
            lines.append(
                f"| {name} | {metric['mean']:.4f} | {metric['min']:.4f} | "
                f"{metric['max']:.4f} | {metric['pass_rate']:.2%} | {metric['count']} |"
            )
    for axis, heading in (
        ("answerability", "Answerability"),
        ("complexity", "Complexity"),
        ("query_style", "Query Style"),
        ("difficulty", "Difficulty"),
    ):
        table = report.get(f"by_{axis}") or {}
        if not table:
            continue
        lines += ["", f"## Breakdown by {heading}", ""]
        names = [name for name in report.get("metric_names", []) if any(name in row for row in table.values())]
        lines.append("| Group | Count | " + " | ".join(names) + " |")
        lines.append("| :--- | ---: | " + " | ".join("---:" for _ in names) + " |")
        for group, values in sorted(table.items()):
            vals = [f"{values.get(name, 0.0):.4f}" for name in names]
            lines.append(f"| {group} | {int(values.get('count', 0))} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def merge_report(args: argparse.Namespace) -> None:
    base = read_json(args.base_report)
    delta = read_json(args.delta_report)
    if base.get("task") != delta.get("task"):
        raise ValueError("Base and delta reports have different task types.")
    if base.get("metric_names") != delta.get("metric_names"):
        raise ValueError("Base and delta reports have different metric_names/settings.")
    base_threshold = (base.get("metadata") or {}).get("threshold", 0.5)
    delta_threshold = (delta.get("metadata") or {}).get("threshold", 0.5)
    if base_threshold != delta_threshold:
        raise ValueError("Base and delta reports have different thresholds.")
    base_by_id = index_rows(base.get("per_query_results", []), str(args.base_report))
    delta_by_id = index_rows(delta.get("per_query_results", []), str(args.delta_report))
    duplicate_unknown = delta_by_id.keys() - base_by_id.keys()
    if duplicate_unknown:
        raise ValueError(f"Delta report has IDs absent from base report: {sorted(duplicate_unknown)}")
    dataset_payload = read_json(args.delta_dataset)
    dataset_rows = dataset_payload.get("goldens") if isinstance(dataset_payload, dict) else None
    if not isinstance(dataset_rows, list):
        raise ValueError("--delta-dataset must contain a 'goldens' list.")
    expected_ids = set(index_rows(dataset_rows, str(args.delta_dataset)))
    if delta_by_id.keys() != expected_ids:
        raise ValueError(
            "Delta report IDs differ from prepared delta dataset: "
            f"missing={sorted(expected_ids - delta_by_id.keys())[:10]}, "
            f"extra={sorted(delta_by_id.keys() - expected_ids)[:10]}."
        )
    if args.expected_delta_count is not None and len(delta_by_id) != args.expected_delta_count:
        raise ValueError(
            f"Delta report has {len(delta_by_id)} rows; expected {args.expected_delta_count}."
        )

    # Keep the original full-report order and replace only rows freshly evaluated.
    merged_rows = [delta_by_id.get(qid, row) for qid, row in base_by_id.items()]
    report = calculate_report(base, merged_rows)
    report["metadata"] = dict(report.get("metadata") or {})
    report["metadata"]["incremental_merge"] = {
        "base_report": str(args.base_report),
        "delta_report": str(args.delta_report),
        "updated_query_ids": list(delta_by_id),
    }
    write_json(args.out_json, report)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Merged {len(delta_by_id)} refreshed rows into {len(merged_rows)} total rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Extract delta and pipeline result subsets.")
    prep.add_argument("--base", type=Path, default=DEFAULT_BASE)
    prep.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    prep.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    prep.add_argument(
        "--pipeline",
        action="append",
        nargs=2,
        metavar=("NAME", "RESULTS_JSON"),
        default=[],
        help="Override a default pipeline path or add another pipeline; repeatable.",
    )

    merge = subparsers.add_parser("merge-report", help="Replace delta rows in a full report and reaggregate.")
    merge.add_argument("--base-report", required=True, type=Path)
    merge.add_argument("--delta-report", required=True, type=Path)
    merge.add_argument("--out-json", required=True, type=Path)
    merge.add_argument("--out-md", type=Path)
    merge.add_argument(
        "--delta-dataset",
        required=True,
        type=Path,
        help="Require the delta report IDs to exactly match this prepared dataset.",
    )
    merge.add_argument(
        "--expected-delta-count",
        type=int,
        default=None,
        help="Fail unless the delta report contains this many refreshed rows.",
    )
    results = subparsers.add_parser(
        "merge-results", help="Replace regenerated answers in a full reviewed results file."
    )
    results.add_argument("--base-results", required=True, type=Path)
    results.add_argument("--delta-results", required=True, type=Path)
    results.add_argument("--delta-dataset", required=True, type=Path)
    results.add_argument("--out-results", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        pipelines = dict(DEFAULT_PIPELINES)
        pipelines.update({name: Path(path) for name, path in args.pipeline})
        args.pipeline = list(pipelines.items())
        prepare(args)
    elif args.command == "merge-report":
        merge_report(args)
    else:
        merge_results(args)


if __name__ == "__main__":
    main()
