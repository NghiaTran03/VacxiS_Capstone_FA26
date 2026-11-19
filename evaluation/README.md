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
