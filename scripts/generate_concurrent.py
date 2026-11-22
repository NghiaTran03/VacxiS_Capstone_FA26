"""Concurrent generation adapter that leaves the project CLI unchanged.

Retrieval is kept serial because local retrievers may own GPU/model state. Only
independent provider generation requests are run in the worker pool.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.generation.cli import (
    _get_query_id,
    build_retriever,
    load_resume_results,
    save_results,
)
from vacxis_capstone_fa26.generation.datasets import load_dataset
from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.resilience import (
    add_retry_arguments,
    atomic_json,
    fingerprint,
    retry_from_args,
)

logger = logging.getLogger("vacxis.generation.concurrent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent generation adapter; retrieval remains serial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--dataset-format", choices=("deepeval", "legacy"), default="deepeval")
    parser.add_argument("--out-file", type=Path, default=Path("generation_results.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-ground-truth-chunks", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum simultaneous generation requests")
    parser.add_argument("--model", help="OpenRouter model override for VACXIS_GENERATION_MODEL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--chunks", type=Path, default=Path("chunks_storage/chunks.parquet"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--baseline", action="store_true", help="Use BM25 retrieval")
    modes.add_argument("--advanced-rag", action="store_true", help="Use dense-only retrieval")
    from vacxis_capstone_fa26.cli import add_retrieval_arguments

    add_retrieval_arguments(parser)
    add_retry_arguments(parser)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    retry_from_args(args)
    return args


def _generate(generator: MedicalGenerator, query: str, chunks: list[Any]):
    return generator.generate(query=query, chunks=chunks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if not args.input_file.is_file():
        logger.error("Input file not found: %s", args.input_file)
        raise SystemExit(1)

    items = load_dataset(args.input_file, args.dataset_format, args.limit)
    config = GenerationConfig(
        temperature=args.temperature,
        dry_run=args.dry_run,
        timeout=args.timeout,
        retry_policy=retry_from_args(args),
    )
    if args.model is not None:
        config.model = args.model
    if not args.dry_run:
        from vacxis_capstone_fa26.llm import validate_config

        validate_config(config.model)
    generator = MedicalGenerator(config=config)

    settings = vars(args).copy()
    for key in ("resume", "out_file", "retries", "retry_initial", "retry_max", "timeout", "llm_call_delay", "concurrency"):
        settings.pop(key, None)
    signature = fingerprint({
        "items": [item.original for item in items],
        "settings": settings,
        "generation": {k: v for k, v in asdict(config).items() if k not in ("retry_policy", "timeout")},
    })
    manifest = args.out_file.with_suffix(args.out_file.suffix + ".manifest.json")
    if args.resume and args.out_file.exists():
        try:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot resume: generation manifest is missing or invalid.") from exc
        if previous.get("fingerprint") != signature:
            raise ValueError("Generation checkpoint settings changed; use a new output file.")
        existing, completed_ids = load_resume_results(args.out_file, items)
    else:
        existing, completed_ids = [], set()
        save_results([], args.out_file)
    atomic_json(manifest, {"fingerprint": signature})

    output_by_index: dict[int, dict[str, Any]] = {}
    if existing:
        by_id = {_get_query_id(result): result for result in existing}
        for index, item in enumerate(items):
            query_id = _get_query_id(item.original)
            if query_id in by_id:
                output_by_index[index] = by_id[query_id]

    pending = [
        (index, item)
        for index, item in enumerate(items)
        if not (args.resume and _get_query_id(item.original) in completed_ids)
    ]
    if not args.use_ground_truth_chunks:
        from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline

    def save_checkpoint() -> None:
        save_results([output_by_index[i] for i in sorted(output_by_index)], args.out_file)

    errors: list[tuple[int, BaseException]] = []
    if args.use_ground_truth_chunks:
        contexts = [(index, item, item.chunks) for index, item in pending]
        _run_concurrent(contexts, generator, args.concurrency, output_by_index, save_checkpoint, errors)
    else:
        with build_retriever(args) as retriever:
            pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
            # Keep local retrieval/model use serialized; provider calls are independent.
            contexts = []
            for index, item in pending:
                try:
                    chunks = pipeline.retriever.retrieve(
                        query=item.query, top_k=args.top_k, skip_rerank=args.skip_rerank,
                    )
                    contexts.append((index, item, chunks))
                except Exception as exc:
                    errors.append((index, exc))
                    logger.exception("Retrieval failed for input row %d", index + 1)
            _run_concurrent(contexts, generator, args.concurrency, output_by_index, save_checkpoint, errors)

    save_checkpoint()
    if errors:
        raise RuntimeError(
            f"{len(errors)} row(s) failed; successful rows are checkpointed. "
            "Fix the cause and rerun with --resume."
        ) from errors[0][1]
    logger.info("Saved %d answers to %s", len(output_by_index), args.out_file)


def _run_concurrent(contexts, generator, concurrency, output_by_index, save_checkpoint, errors):
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="vacxis-gen") as pool:
        futures = {
            pool.submit(_generate, generator, item.query, chunks): (index, item)
            for index, item, chunks in contexts
        }
        for future in as_completed(futures):
            index, item = futures[future]
            try:
                answer = future.result()
                result = item.original.copy()
                result["GENERATED_ANSWER"] = answer.to_dict()
                output_by_index[index] = result
                save_checkpoint()
                logger.info("Completed row %d/%d", index + 1, len(contexts))
            except Exception as exc:
                errors.append((index, exc))
                logger.exception("Generation failed for input row %d", index + 1)


if __name__ == "__main__":
    main()
