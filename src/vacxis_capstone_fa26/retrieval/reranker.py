"""Reranker module using BGE-reranker-v2-m3."""
from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path
from time import perf_counter
from typing import Sequence

import torch

from vacxis_capstone_fa26.retrieval.batching import run_with_oom_backoff
from vacxis_capstone_fa26.retrieval.schemas import (
    DEFAULT_RERANK_BATCH_SIZE,
    RetrievedChunk,
)

logger = logging.getLogger("vacxis.retrieval.reranker")

DEFAULT_RERANKER_PATH = "models/bge-reranker"


class BGEReranker:
    """Cross-encoder reranker wrapping BGE-reranker-v2-m3."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RERANKER_PATH,
        use_fp16: bool | None = None,
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
    ) -> None:
        """Initialize FlagReranker model.

        Args:
            model_path: Path to local BGE-reranker directory or model name.
            use_fp16: Whether to use FP16 half-precision. Defaults to torch.cuda.is_available().
            batch_size: Default batch size for cross-encoder inference.
        """
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        self.model_path = str(model_path)
        self.batch_size = batch_size
        if use_fp16 is None:
            self.use_fp16 = torch.cuda.is_available()
        else:
            self.use_fp16 = use_fp16

        logger.info(
            "Loading BGE-Reranker from '%s' (fp16=%s)...",
            self.model_path,
            self.use_fp16,
        )
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(self.model_path, use_fp16=self.use_fp16)
        logger.info("BGE-Reranker loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int = 5,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> list[RetrievedChunk]:
        """Compute cross-encoder relevance scores for query and candidates.

        Args:
            query: User search query string.
            candidates: Sequence of retrieved candidate chunks from prior stage.
            top_k: Number of highest-ranked chunks to return.
            batch_size: Optional override batch size for cross-encoder inference.
            normalize: Whether to apply sigmoid/normalization to scores (range [0, 1]).

        Returns:
            List of top-k RetrievedChunk sorted by rerank score descending.
        """
        results = self.batch_rerank(
            queries=[query],
            candidates_list=[candidates],
            top_k=top_k,
            batch_size=batch_size,
            normalize=normalize,
        )
        return results[0] if results else []

    def batch_rerank(
        self,
        queries: Sequence[str],
        candidates_list: Sequence[Sequence[RetrievedChunk]],
        top_k: int = 5,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> list[list[RetrievedChunk]]:
        """Compute cross-encoder relevance scores across multiple queries in a batched manner.

        Args:
            queries: Sequence of user search query strings.
            candidates_list: Sequence of candidate chunk sequences per query.
            top_k: Number of highest-ranked chunks to return per query.
            batch_size: Optional override batch size for cross-encoder inference.
            normalize: Whether to apply sigmoid/normalization to scores (range [0, 1]).

        Returns:
            List of top-k RetrievedChunk lists for each query.
        """
        if len(queries) != len(candidates_list):
            raise ValueError("Expected one candidate group per query.")
        effective_batch_size = (
            batch_size
            if batch_size is not None
            else getattr(self, "batch_size", DEFAULT_RERANK_BATCH_SIZE)
        )
        if type(effective_batch_size) is not int or effective_batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        num_queries = len(queries)
        if num_queries == 0:
            return []

        all_pairs: list[list[str]] = []
        query_slices: list[int] = []

        for q_idx, q_text in enumerate(queries):
            cands = candidates_list[q_idx] if q_idx < len(candidates_list) else []
            count = len(cands)
            query_slices.append(count)
            for chunk in cands:
                all_pairs.append([q_text, chunk.text])

        if not all_pairs:
            return [[] for _ in range(num_queries)]

        started = perf_counter()
        scores = run_with_oom_backoff(
            lambda bs: self.model.compute_score(
                all_pairs,
                batch_size=bs,
                normalize=normalize,
            ),
            batch_size=effective_batch_size,
            operation="BGEReranker.batch_rerank",
        )
        logger.info(
            "BGE-Reranker scored %d candidate pair(s) across %d query(ies) in %.2fs (top_k=%d).",
            len(all_pairs),
            num_queries,
            perf_counter() - started,
            top_k,
        )

        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)) or (hasattr(scores, "item") and getattr(scores, "ndim", 0) == 0):
            scores = [float(scores)]

        if len(scores) != len(all_pairs):
            raise ValueError("Expected one reranker score per candidate.")
        results: list[list[RetrievedChunk]] = []
        offset = 0
        for q_idx in range(num_queries):
            count = query_slices[q_idx]
            if count == 0:
                results.append([])
                continue

            q_cands = candidates_list[q_idx]
            q_scores = scores[offset : offset + count]
            offset += count

            reranked: list[RetrievedChunk] = []
            for chunk, score in zip(q_cands, q_scores):
                score_float = float(score)
                reranked.append(
                    replace(
                        chunk,
                        score=score_float,
                        rerank_score=score_float,
                    )
                )

            reranked.sort(key=lambda c: c.score, reverse=True)
            results.append(reranked[:top_k])

        return results
