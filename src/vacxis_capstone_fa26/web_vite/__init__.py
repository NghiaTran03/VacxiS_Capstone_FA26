"""CLI entry point for the FastAPI + React/Vite VaxiS web server."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import threading

import uvicorn

from ..generation import VacxisRAGPipeline
from ..llm import validate_config
from ..query import QueryPlanner
from ..retrieval.schemas import RetrievalConfig
from ..web.documents import Documents
from .server import create_api_app
from .storage import ConversationStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
CONVERSATIONS_RELATIVE_DIR = Path("data") / "web_vite_conversations"
FRONTEND_DIST_RELATIVE_DIR = Path("web-vite") / "dist"
PLANNER_PIPELINE_LOGGER = "vacxis_capstone_fa26.generation.pipeline"
_VERBOSE_HANDLER_ATTR = "_vacxis_web_vite_verbose_handler"


def configure_logging(verbose: bool) -> None:
    """Configure verbose terminal logging for the query planner pipeline."""
    pipeline_logger = logging.getLogger(PLANNER_PIPELINE_LOGGER)
    existing_handlers = [
        handler
        for handler in pipeline_logger.handlers
        if getattr(handler, _VERBOSE_HANDLER_ATTR, False)
    ]
    if verbose:
        pipeline_logger.setLevel(logging.DEBUG)
        if not existing_handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            setattr(handler, _VERBOSE_HANDLER_ATTR, True)
            pipeline_logger.addHandler(handler)
    else:
        for handler in existing_handlers:
            pipeline_logger.removeHandler(handler)
        pipeline_logger.setLevel(logging.NOTSET)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse deployment-focused CLI flags for the Vite web server."""
    parser = argparse.ArgumentParser(description="Run the VaxiS FastAPI + Vite web server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("chunks_storage/chunks.parquet"),
        help="Path to the indexed chunks Parquet file",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview UI using sample extractive responses without model services",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose QueryPlanner output in server logs (enabled by default; pass --no-verbose to disable)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    return args


def default_retrieval_config(project_root: Path, chunks_path: Path) -> RetrievalConfig:
    """Build the main Table-aware + LightRAG retrieval configuration."""
    return RetrievalConfig(
        mode="hybrid_lightrag",
        collection_name="VacxiS_Capstone_FA26_table_dual",
        lightrag_fusion="rrf_rerank",
        embed_model_path=project_root / "models" / "bge-m3",
        reranker_model_path=project_root / "models" / "bge-reranker",
        lightrag_working_dir=project_root / "lightrag_storage",
        chunks_parquet_path=chunks_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    project_root = args.project_root.resolve()
    chunks_path = args.chunks if args.chunks.is_absolute() else project_root / args.chunks
    retrieval = default_retrieval_config(project_root, chunks_path)

    with ExitStack() as resources:
        pipeline = None
        pipeline_lock = threading.Lock()
        if not args.preview:
            print(f"* Retriever: {retrieval.mode} | collection: {retrieval.collection_name}", flush=True)
            print(
                f"* LightRAG mode: {retrieval.lightrag_mode} | workspace: {retrieval.lightrag_workspace}",
                flush=True,
            )
            print(f"* LightRAG fusion: {retrieval.lightrag_fusion}", flush=True)
            if args.verbose:
                print("* Verbose planner logging: enabled", flush=True)

        def run_turn(query: str, **options):
            nonlocal pipeline
            if pipeline is None:
                with pipeline_lock:
                    if pipeline is None:
                        validate_config()
                        planner = QueryPlanner(model=validate_config(planner=True))
                        pipeline = resources.enter_context(
                            VacxisRAGPipeline(retrieval_config=retrieval, query_planner=planner)
                        )
            return pipeline.run_turn(query, **options)

        documents = Documents(chunks_path, project_root=project_root)
        if args.preview:
            if not documents.indexed_ids:
                raise SystemExit("Preview requires an indexed --chunks file with vaccine documents.")
            from ..web.preview import preview_responder

            run_turn = preview_responder(documents)

        store = ConversationStore(project_root / CONVERSATIONS_RELATIVE_DIR)
        frontend_dist = project_root / FRONTEND_DIST_RELATIVE_DIR
        app = create_api_app(
            run_turn,
            documents,
            store,
            preview=args.preview,
            frontend_dist=frontend_dist,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    return 0
