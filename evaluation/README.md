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

## Separate BM25 + dense hybrid baseline

`--hybrid-baseline` combines BM25 over `--chunks` with BGE-M3 dense search
against the existing Qdrant collection. Equal-weight RRF merges the rankings
by chunk ID. It does not use BGE learned sparse search or cross-encoder
reranking. This differs from the main hybrid pipeline's dense + learned sparse
retrieval. The generator and evaluation metrics remain shared.

Each branch retrieves 50 candidates by default (`--baseline-candidate-k`),
RRF uses a smoothing constant of 60 (`--baseline-rrf-k`), and final top-k is 5.
Candidate depth expands when final top-k exceeds the requested candidate depth.
Corpus text and document metadata are retained for generation. The corpus and
Qdrant collection must describe the same chunks; mismatched returned IDs fail
rather than silently mixing corpora. `--skip-rerank` is redundant for this
baseline, which always disables reranking. `--baseline` still selects BM25 only;
it cannot be combined with `--hybrid-baseline`.

Run on the machine with models and Qdrant, first without generation:

```bash
uv run vacxis-eval --hybrid-baseline --retrieval-only --top-k 10 \
  --out-json evaluation/bm25_dense_k10.json \
  --out-md evaluation/bm25_dense_k10.md
```

Full RAG evaluation uses the same retriever plus the standard generator and
configured DeepEval judge (requires generation/judge credentials):

```bash
uv run vacxis-eval --hybrid-baseline --top-k 4 --ir-k-values 1 3 4 \
  --skip-unanswerable \
  --out-json evaluation/bm25_dense_rag_k4.json \
  --out-md evaluation/bm25_dense_rag_k4.md
```

Both commands use the corrected default dataset. Set `--chunks`,
`--collection-name`, `--qdrant-host`, and `--model-path` for a different store.
Reports identify this variant as `bm25_dense_rrf`. Live performance and quality
must be verified on the retrieval machine; mocked offline tests do not establish
retrieval quality.

## Separate advanced RAG experiment

`--advanced-rag` adds query planning and cross-encoder reranking to BM25+dense
retrieval. It always searches the original question and adds up to two distinct
planner searches. Each search uses the separate BM25+dense baseline. RRF fuses
the query rankings into a bounded pool, then the existing BGE reranker ranks
that pool against the original question. The standard generator also receives
the original question. This experiment does not change the main pipeline.

Defaults: 50 candidates per search and merged pool (`--advanced-candidate-k`),
query-fusion RRF constant 60 (`--advanced-rrf-k`), and final top-k 5. The inner
BM25/dense branch settings remain `--baseline-candidate-k 50` and
`--baseline-rrf-k 60`. Depths expand when final top-k is larger. Use
`--skip-rerank` for an advanced-pipeline ablation.

This mode calls an LLM planner even with `--retrieval-only`. Configure
`OPENROUTER_API_KEY` and `VACXIS_QUERY_PLANNER_MODEL`, or pass `--planner-model`.
Planner timeout defaults to 20 seconds (`--planner-timeout`, maximum 120).
If the planner requests clarification or its request/response fails, evaluation
uses the original search. Store/model failures propagate instead of becoming
empty rankings. JSON reports preserve per-query searches, planner/fallback
status, pool sizes and final IDs under `metadata.retrieval_trace`; the summary
includes `planner_status_counts`. Run settings identify `advanced_bm25_dense`.
No gold reference answers or source IDs are sent to the planner or retriever.

Run on the machine with models, Qdrant, and planner credentials:

```bash
uv run vacxis-eval --advanced-rag --retrieval-only --top-k 10 \
  --out-json evaluation/advanced_k10.json --out-md evaluation/advanced_k10.md
uv run vacxis-eval --advanced-rag --top-k 4 --ir-k-values 1 3 4 \
  --skip-unanswerable \
  --out-json evaluation/advanced_rag_k4.json --out-md evaluation/advanced_rag_k4.md
```

The second command also needs the standard generation/judge configuration.
Compare the same dataset, corpus, and final top-k across variants. Query
expansion increases model calls and search cost; its quality improvement is
an experimental hypothesis until these live evaluations are run. The known
question/reference inconsistencies in the corrected gold dataset still need
human review before treating every miss as a retrieval failure.
