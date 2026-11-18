"""Generate and inspect the stratified 3-expert review allocation plan for the VacxiS RAG dataset."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REVIEW_WEB_DIR = ROOT / "review-web"
if str(REVIEW_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(REVIEW_WEB_DIR))

from dataset_format import as_golden  # noqa: E402
from sampling import allocate_scientific_batches, summarize_allocation  # noqa: E402


DEFAULT_DATASET = ROOT / "evaluation" / "eval_dataset_300_final_fixed.json"


def load_questions(path):
    """Load DeepEval or legacy JSON records into the shape expected by sampling."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    records = payload.get("goldens") if isinstance(payload, dict) else payload
    questions = []
    for record in records:
        golden = as_golden(record)
        qid = (
            golden["additional_metadata"]["query_id"]
            if golden is not None
            else record["QUERY_ID"]
        )
        questions.append({"id": qid, "original": record})
    return questions


def write_output(path, allocations, fmt):
    """Persist allocation assignments as JSON or CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        fieldnames = ["question_id", "batch_type", "stratum", "primary_slot", "auditor_slot", "requires_audit"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(allocations)
        return
    output = {"summary": summarize_allocation(allocations), "allocations": allocations}
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_report(allocations):
    """Print concise batch and marginal balance metrics."""
    summary = summarize_allocation(allocations)
    print(f"Total questions: {len(allocations)}")
    print(f"Batches: {summary['by_batch']}")
    for slot, counts in summary["by_slot"].items():
        print(
            f"  Expert {slot}: calibration={counts['calibration']}, overlap={counts['overlap']}, "
            f"primary={counts['primary']}, audit={counts['audit']} (total_turns={counts['total_turns']})"
        )

    audit_strata = Counter(row["stratum"] for row in allocations if row["requires_audit"])
    print(f"Cross-audit questions total: {sum(audit_strata.values())}")
    for stratum, count in sorted(audit_strata.items()):
        print(f"  {stratum}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Plan stratified 3-expert review batches for VacxiS.")
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET, help="Path to evaluation JSON file.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    parser.add_argument("--output", type=Path, help="Optional output file path (.json or .csv).")
    parser.add_argument("--format", choices=("json", "csv"), help="Explicit output format.")
    args = parser.parse_args()

    questions = load_questions(args.dataset)
    allocations = allocate_scientific_batches(questions, seed=args.seed)
    print_report(allocations)
    if args.output:
        fmt = args.format or ("csv" if args.output.suffix.lower() == ".csv" else "json")
        write_output(args.output, allocations, fmt)
        print(f"Saved allocation plan to {args.output}")


if __name__ == "__main__":
    main()
