"""CLI utility for running the VacxiS generation and end-to-end RAG pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import sys
from pathlib import Path
from typing import Any

from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.prompts import get_chunk_field
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.generation.datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.generation.cli")


@dataclass
class _GroundTruthChunk:
    chunk_id: str
    text: str
    doc_title: str = ""
    category: str = ""
    header: str = ""
    score: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse generation CLI command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS Generation CLI: Grounded Medical Answer Synthesis with Citations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Medical question to answer",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenRouter model override for VACXIS_GENERATION_MODEL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LLM inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve for generation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run offline extractive answer synthesis without calling LLM APIs",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        default=False,
        help="Skip cross-encoder reranking during retrieval",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON response",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Input JSON file containing queries (e.g. test_dataset.json)",
    )
    parser.add_argument(
    "--dataset-format",
    choices=["deepeval", "legacy"],
    default="deepeval",
    help="Input dataset format",
)
    parser.add_argument(
        "--out-file",
        type=Path,
        default=None,
        help="Output JSON path to save generation results",
    )
    parser.add_argument(
        "--use-ground-truth-chunks",
        action="store_true",
        default=False,
        help="Use RELEVANT_CHUNKS from input file instead of running retrieval",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries processed from input file",
    )
    from vacxis_capstone_fa26.cli import add_retrieval_arguments
    add_retrieval_arguments(parser)
    return parser.parse_args()

def save_results(results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    temp_path.replace(out_path)

def main() -> None:
    """Run generation CLI entrypoint."""
    from vacxis_capstone_fa26.cli import retrieval_config
    args = parse_args()

    gen_config = GenerationConfig(
        temperature=args.temperature,
        dry_run=args.dry_run,
    )
    if args.model is not None:
        gen_config.model = args.model
    if not args.dry_run:
        from vacxis_capstone_fa26.llm import validate_config
        validate_config(gen_config.model)
    generator = MedicalGenerator(config=gen_config)

    # Mode 1: Batch process from input file (e.g. test_dataset.json)

    if args.input_file:
        if not args.input_file.exists():
            logger.error("Input file not found: %s", args.input_file)
            sys.exit(1)

        try:
            items = load_dataset(
                path=args.input_file,
                dataset_format=args.dataset_format,
                limit=args.limit,
            )
        except (ValueError, KeyError, TypeError) as exc:
            logger.error("Failed to load dataset: %s", exc)
            sys.exit(1)

        logger.info(
            "Processing %d items using %s format...",
            len(items),
            args.dataset_format,
        )

        results = []

        if args.use_ground_truth_chunks:
            logger.info("Using provided ground-truth contexts.")

            for item in items:
                ans = generator.generate(
                    query=item.query,
                    chunks=item.chunks,
                )

                result = item.original.copy()
                result["GENERATED_ANSWER"] = ans.to_dict()
                results.append(result)
                save_results(results, out_path)

        else:
            try:
                from vacxis_capstone_fa26.retrieval.retriever import (
                    VacxisRetriever,
                )
            except ImportError as exc:
                logger.error(
                    "Retrieval module requires PyTorch/CUDA dependencies (%s).",
                    exc,
                )
                sys.exit(1)

            ret_config = retrieval_config(args)

            with VacxisRetriever(config=ret_config) as retriever:
                pipeline = VacxisRAGPipeline(
                    retriever=retriever,
                    generator=generator,
                )

                for item in items:
                    ans = pipeline.answer(
                        query=item.query,
                        top_k=args.top_k,
                        skip_rerank=args.skip_rerank,
                    )

                    result = item.original.copy()
                    result["GENERATED_ANSWER"] = ans.to_dict()
                    results.append(result)
                    save_results(results, out_path)

        out_path = args.out_file or Path("generation_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(
            "Saved %d generation results to %s",
            len(results),
            out_path,
        )
        return

    # Mode 2: Single interactive query
    if not args.query or not args.query.strip():
        logger.error("Please provide either --query or --input-file.")
        sys.exit(1)

    query = args.query
    try:
        from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
    except ImportError as exc:
        logger.error("Retrieval module requires PyTorch/CUDA dependencies (%s).", exc)
        sys.exit(1)

    ret_config = retrieval_config(args)

    with VacxisRetriever(config=ret_config) as retriever:
        pipeline = VacxisRAGPipeline(retriever=retriever, generator=generator)
        answer_obj = pipeline.answer(query=query.strip(), top_k=args.top_k, skip_rerank=args.skip_rerank)

    if args.json:
        print(json.dumps(answer_obj.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"Query: {answer_obj.query}")
    print(f"Model: {answer_obj.model}")
    print(f"Latency: {answer_obj.latency_seconds:.3f}s")
    print(f"Refusal: {answer_obj.is_refusal}")
    print(f"Citations: {answer_obj.citations}")
    if answer_obj.invalid_citations:
        print(f"Invalid Citations: {answer_obj.invalid_citations}")
    print("\nAnswer:")
    print(answer_obj.answer)
    print("\nRetrieved Context:")
    for idx, c in enumerate(answer_obj.retrieved_chunks, start=1):
        print(f"[{idx}] {get_chunk_field(c, 'chunk_id', '')} | {get_chunk_field(c, 'doc_title', '')} | {get_chunk_field(c, 'category', '')}")


if __name__ == "__main__":
    main()
