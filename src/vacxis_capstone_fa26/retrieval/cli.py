"""CLI utility for running the VacxiS retrieval pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever
from vacxis_capstone_fa26.retrieval.schemas import RetrievalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vacxis.retrieval.cli")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VacxiS Retrieval CLI: Hybrid Search (Dense + Sparse) + RRF + Reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query text",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final top-k chunks to return",
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=50,
        help="Number of dense search candidates",
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=50,
        help="Number of sparse search candidates",
    )
    parser.add_argument(
        "--rrf-top-k",
        type=int,
        default=30,
        help="Number of candidates after RRF fusion",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        default=False,
        help="Skip cross-encoder reranking and output RRF top-k directly",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="VacxiS_Capstone_FA26",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default="localhost",
        help="Qdrant server host",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/bge-m3",
        help="Path to BGE-M3 model",
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        default="models/bge-reranker",
        help="Path to BGE-reranker model",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON array of retrieved chunks",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval CLI entrypoint."""
    args = parse_args()

    query = args.query.strip()
    if not query:
        logger.error("Query cannot be empty or whitespace-only.")
        sys.exit(1)

    config = RetrievalConfig(
        collection_name=args.collection_name,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        embed_model_path=args.model_path,
        reranker_model_path=args.reranker_path,
        top_k_dense=args.top_k_dense,
        top_k_sparse=args.top_k_sparse,
        rrf_top_k=args.rrf_top_k,
        top_k=args.top_k,
        skip_rerank=args.skip_rerank,
    )

    with VacxisRetriever(config=config) as retriever:
        results = retriever.retrieve(
            query=query,
            top_k=args.top_k,
            top_k_dense=args.top_k_dense,
            top_k_sparse=args.top_k_sparse,
            rrf_top_k=args.rrf_top_k,
            skip_rerank=args.skip_rerank,
        )

    if args.json:
        data = [chunk.to_dict() for chunk in results]
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return

    print(f"Query: {query}")
    print(f"Total retrieved: {len(results)}")
    for idx, chunk in enumerate(results, start=1):
        score_info = f"Score: {chunk.score:.4f}"
        if chunk.rerank_score is not None:
            score_info += f" (Rerank: {chunk.rerank_score:.4f}, RRF: {chunk.rrf_score:.5f})"
        else:
            score_info += f" (RRF: {chunk.rrf_score:.5f})"

        print(f"[{idx}] {chunk.chunk_id} | {chunk.doc_title} | {chunk.category} | {score_info}")
        print(f"Header: {chunk.header}")
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"Snippet: {snippet}\n")


if __name__ == "__main__":
    main()
