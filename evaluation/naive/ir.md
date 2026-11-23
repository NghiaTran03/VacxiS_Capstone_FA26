# VacxiS DeepEval Report: RETRIEVAL
Total evaluated queries: 300

## Overall Metrics
| Metric | Mean | Min | Max | Pass Rate | Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hit@1 | 0.2778 | 0.0000 | 1.0000 | 27.78% | 288 |
| Recall@1 | 0.2378 | 0.0000 | 1.0000 | 27.78% | 288 |
| Precision@1 | 0.2778 | 0.0000 | 1.0000 | 27.78% | 288 |
| MRR@1 | 0.2778 | 0.0000 | 1.0000 | 27.78% | 288 |
| NDCG@1 | 0.2778 | 0.0000 | 1.0000 | 27.78% | 288 |
| MAP@1 | 0.2378 | 0.0000 | 1.0000 | 27.78% | 288 |
| Hit@3 | 0.4618 | 0.0000 | 1.0000 | 46.18% | 288 |
| Recall@3 | 0.3958 | 0.0000 | 1.0000 | 46.18% | 288 |
| Precision@3 | 0.1597 | 0.0000 | 0.6667 | 1.74% | 288 |
| MRR@3 | 0.3600 | 0.0000 | 1.0000 | 40.28% | 288 |
| NDCG@3 | 0.3447 | 0.0000 | 1.0000 | 39.93% | 288 |
| MAP@3 | 0.3102 | 0.0000 | 1.0000 | 36.11% | 288 |
| Hit@5 | 0.5486 | 0.0000 | 1.0000 | 54.86% | 288 |
| Recall@5 | 0.4826 | 0.0000 | 1.0000 | 54.86% | 288 |
| Precision@5 | 0.1181 | 0.0000 | 0.4000 | 0.00% | 288 |
| MRR@5 | 0.3799 | 0.0000 | 1.0000 | 40.28% | 288 |
| NDCG@5 | 0.3826 | 0.0000 | 1.0000 | 40.62% | 288 |
| MAP@5 | 0.3325 | 0.0000 | 1.0000 | 36.11% | 288 |

IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.
Pass rate is the fraction of scored queries meeting the configured threshold (0.5); it is not a retrieval accuracy measure.
Scored queries: 288; excluded without relevance labels: 12.
Warning: 12 queries have no relevance labels and are excluded from IR averages.

## Run Settings
- mode: live
- dataset: review-web/test_dataset_verified.json
- results: None
- retriever: bm25
- retriever_mode: naive
- top_k: 5
- chunks: chunks_storage/chunks.parquet

## Breakdown by Answerability
| Answerability | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| ANSWERABLE | 197 | 0.3147 | 0.2741 | 0.3147 | 0.3147 | 0.3147 | 0.2741 | 0.5228 | 0.4543 | 0.1810 | 0.4095 | 0.3960 | 0.3583 | 0.6041 | 0.5431 | 0.1299 | 0.4280 | 0.4338 | 0.3808 |
| PARTIALLY_ANSWERABLE | 91 | 0.1978 | 0.1593 | 0.1978 | 0.1978 | 0.1978 | 0.1593 | 0.3297 | 0.2692 | 0.1136 | 0.2527 | 0.2337 | 0.2060 | 0.4286 | 0.3516 | 0.0923 | 0.2758 | 0.2717 | 0.2280 |
| UNANSWERABLE | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Breakdown by Complexity
| Complexity | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| MULTI_HOP | 90 | 0.2706 | 0.1353 | 0.2706 | 0.2706 | 0.2706 | 0.1353 | 0.5059 | 0.2824 | 0.1882 | 0.3765 | 0.2693 | 0.2078 | 0.5882 | 0.3647 | 0.1459 | 0.3947 | 0.3096 | 0.2340 |
| SINGLE_HOP | 150 | 0.3472 | 0.3472 | 0.3472 | 0.3472 | 0.3472 | 0.3472 | 0.5347 | 0.5347 | 0.1782 | 0.4317 | 0.4582 | 0.4317 | 0.6250 | 0.6250 | 0.1250 | 0.4529 | 0.4959 | 0.4529 |
| TABLE_REASONING | 60 | 0.1186 | 0.1186 | 0.1186 | 0.1186 | 0.1186 | 0.1186 | 0.2203 | 0.2203 | 0.0734 | 0.1610 | 0.1761 | 0.1610 | 0.3051 | 0.3051 | 0.0610 | 0.1805 | 0.2112 | 0.1805 |

## Breakdown by Query Style
| Query Style | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| CLINICAL_STYLE | 120 | 0.2018 | 0.1667 | 0.2018 | 0.2018 | 0.2018 | 0.1667 | 0.3509 | 0.2939 | 0.1199 | 0.2690 | 0.2541 | 0.2259 | 0.4211 | 0.3684 | 0.0930 | 0.2852 | 0.2877 | 0.2467 |
| PATIENT_STYLE | 180 | 0.3276 | 0.2845 | 0.3276 | 0.3276 | 0.3276 | 0.2845 | 0.5345 | 0.4626 | 0.1858 | 0.4195 | 0.4040 | 0.3654 | 0.6322 | 0.5575 | 0.1345 | 0.4420 | 0.4448 | 0.3887 |

## Breakdown by Difficulty
| Difficulty | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| UNSPECIFIED | 300 | 0.2778 | 0.2378 | 0.2778 | 0.2778 | 0.2778 | 0.2378 | 0.4618 | 0.3958 | 0.1597 | 0.3600 | 0.3447 | 0.3102 | 0.5486 | 0.4826 | 0.1181 | 0.3799 | 0.3826 | 0.3325 |