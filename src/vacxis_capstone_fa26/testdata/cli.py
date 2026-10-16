"""CLI entry point for the test data generator."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .chunks import load_chunks
from .generator import (
    derive_default_seed,
    generate_and_validate,
)
from .refiner import LiteLLMRefiner, noop_refiner
from .taxonomy import QUERY_TYPE_DEFS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "75_vaccine_names_mapping.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test_dataset.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help="Path to vaccine name mapping JSON (id -> name)",
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
        help=(
            "Random seed for reproducibility. If omitted, a deterministic "
            "seed is derived from the chunk content so re-runs are stable."
        ),
    )
    p.add_argument(
        "--refine-spans",
        action="store_true",
        help=(
            "Opt-in: call an LLM (LiteLLM, model from $VACXIS_LLM_MODEL, "
            "default claude-haiku-4-5) to upgrade the primary chunk's "
            "EVIDENCE_SPAN. Requires network + provider credentials."
        ),
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-write validation pass.",
    )
    p.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat validation warnings as failures (exit non-zero).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.chunks.exists():
        print(f"[ERROR] chunks parquet not found at {args.chunks}", file=sys.stderr)
        return 1

    print(f"Loading chunks from {args.chunks} ...")
    chunks = load_chunks(args.chunks)
    print(
        f"  -> {len(chunks)} chunks across "
        f"{len({c.doc_id for c in chunks})} documents"
    )

    vaccine_mapping: dict[str, str] = {}
    if args.mapping.exists():
        with open(args.mapping, "r", encoding="utf-8") as f:
            vaccine_mapping = json.load(f)
        print(f"  -> vaccine name mapping loaded from {args.mapping}")

    seed = args.seed if args.seed is not None else derive_default_seed(chunks, args.per_type)
    rng = random.Random(seed)
    print(f"Using seed: {seed} ({'explicit' if args.seed is not None else 'derived from chunk content'})")

    refiner = LiteLLMRefiner() if args.refine_spans else noop_refiner()

    print(
        f"Generating {args.per_type} query/QUERY_TYPE with {args.distractors} "
        f"distractors per query ..."
    )
    queries, rpt = generate_and_validate(
        chunks, rng, args.per_type, args.distractors,
        vaccine_mapping=vaccine_mapping,
        refiner=refiner,
        expected_per_type=args.per_type,
    )
    print(f"  -> {len(queries)} queries generated")

    if not args.no_validate:
        print(rpt.render())
        if not rpt.ok:
            print("[ERROR] validation failed; not writing dataset.", file=sys.stderr)
            return 2
        if args.strict_validation and rpt.warnings:
            print(
                f"[ERROR] --strict-validation: {len(rpt.warnings)} warnings "
                "treated as failures.",
                file=sys.stderr,
            )
            return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"Wrote dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
