# Retrieval evaluation

Use deterministic IR metrics to compare ranked chunk IDs with the existing
300-query dataset. These metrics treat only labeled source chunks as relevant;
they do not judge whether an unlabeled alternative chunk also answers the query.

The existing `main_evaluation.json` and `.md` match saved answers in
`eval_dataset_300_generation_results.json`. Each saved answer contains four
retrieved IDs. Their @5 and @10 scores evaluate those four-result rankings with
missing ranks treated as nonrelevant, rather than retrieval at depths 5 and 10.
The original reports are preserved. The CLI defaults to the corrected
`eval_dataset_300_final_fixed.json`: 279 of 300 source label lists differ from
the original `eval_dataset_300_final.json`, whose first Abrysvo question labels
a ProQuad chunk. The corrected dataset matches the goldens embedded in the
saved generation results. Rescore the saved rankings without retrieval:

```bash
uv run vacxis-eval --retrieval-only \
  --dataset evaluation/eval_dataset_300_final_fixed.json \
  --results evaluation/eval_dataset_300_generation_results.json \
  --ir-k-values 1 3 4 --skip-unanswerable \
  --out-json evaluation/main_evaluation_audited.json \
  --out-md evaluation/main_evaluation_audited.md
```

Run actual depth-10 comparisons on the machine with the corpus, retrieval
models, and Qdrant collection. Both runs must use the same dataset and corpus:

```bash
uv run vacxis-eval --retrieval-only --baseline --top-k 10 \
  --dataset evaluation/eval_dataset_300_final_fixed.json \
  --chunks chunks_storage/chunks.parquet \
  --out-json evaluation/bm25_k10.json --out-md evaluation/bm25_k10.md
uv run vacxis-eval --retrieval-only --top-k 10 \
  --dataset evaluation/eval_dataset_300_final_fixed.json \
  --chunks chunks_storage/chunks.parquet \
  --out-json evaluation/hybrid_k10.json --out-md evaluation/hybrid_k10.md
```

New reports retain ranked IDs and relevance labels per query, run settings,
scoring coverage, and counts of rankings shorter than each metric cutoff.
Live CLI evaluation caps default IR cutoffs at `--top-k`; explicit cutoffs
above that depth are rejected. Precomputed rankings may legitimately be short;
reports flag those cases without assuming the intended retrieval depth.

Use Hit@K for finding at least one source, Recall@K for multi-source coverage,
and MRR/NDCG for rank quality. Precision can be low when only one or two chunks
are labeled relevant. The generic 0.5 pass threshold is not an accuracy target:
with at most two gold chunks per query, Precision@5 cannot exceed 0.4.

## Advanced RAG

`--advanced-rag` uses the following single-query embedding pipeline:

`user query -> BGE-M3 dense embedding -> Qdrant search -> top-k chunks -> LLM`

The original question is embedded once and sent unchanged
to the standard generator with the top-k chunks from the existing Qdrant dense
index. Search order, similarity scores, document metadata, and citations are
preserved. Default top-k is 5 (`--top-k`). This replaces the advanced experiment
introduced in commit `b623183`; the other retrieval variants remain available.

There is no query planner, query expansion, BM25, rank fusion, or reranker in
this mode. The former `--planner-model`, `--planner-timeout`,
`--advanced-candidate-k`, and `--advanced-rrf-k` options have been removed.
`--skip-rerank` is redundant because this mode always skips reranking.

Configure `--collection-name`, `--qdrant-host`, `--qdrant-port`, and
`--model-path` to select the store and embedding model. The collection must
already contain indexed chunks with dense vectors and text payloads. Retrieval
and generation use Qdrant payloads. A local `--chunks` Parquet file is optional
for `--retrieval-only`; full evaluation requires the matching corpus to hydrate
retrieval context for the quality metrics. Reports identify the retriever as
`advanced_dense` and retain the
original search query, top-k, and retrieved IDs under `metadata.retrieval_trace`.

Run retrieval evaluation with local models and Qdrant, without LLM credentials:

```bash
uv run vacxis-eval --advanced-rag --retrieval-only --top-k 10 \
  --out-json evaluation/advanced_k10.json --out-md evaluation/advanced_k10.md
```

Full RAG evaluation also requires the standard generation/judge configuration:

```bash
uv run vacxis-eval --advanced-rag --top-k 4 --ir-k-values 1 3 4 \
  --skip-unanswerable \
  --out-json evaluation/advanced_rag_k4.json --out-md evaluation/advanced_rag_k4.md
```

Compare variants with the same dataset, corpus, and final top-k. Offline tests
verify the pipeline contract; retrieval quality requires a live evaluation.
The known question/reference inconsistencies in the corrected gold dataset
still need human review before treating every miss as a retrieval failure.
