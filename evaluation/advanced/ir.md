# VacxiS DeepEval Report: RETRIEVAL
Total evaluated queries: 300

## Overall Metrics
| Metric | Mean | Min | Max | Pass Rate | Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hit@1 | 0.3090 | 0.0000 | 1.0000 | 30.90% | 288 |
| Recall@1 | 0.2569 | 0.0000 | 1.0000 | 30.90% | 288 |
| Precision@1 | 0.3090 | 0.0000 | 1.0000 | 30.90% | 288 |
| MRR@1 | 0.3090 | 0.0000 | 1.0000 | 30.90% | 288 |
| NDCG@1 | 0.3090 | 0.0000 | 1.0000 | 30.90% | 288 |
| MAP@1 | 0.2569 | 0.0000 | 1.0000 | 30.90% | 288 |
| Hit@3 | 0.4896 | 0.0000 | 1.0000 | 48.96% | 288 |
| Recall@3 | 0.4236 | 0.0000 | 1.0000 | 48.96% | 288 |
| Precision@3 | 0.1736 | 0.0000 | 0.6667 | 3.12% | 288 |
| MRR@3 | 0.3877 | 0.0000 | 1.0000 | 42.01% | 288 |
| NDCG@3 | 0.3711 | 0.0000 | 1.0000 | 44.10% | 288 |
| MAP@3 | 0.3356 | 0.0000 | 1.0000 | 39.58% | 288 |
| Hit@5 | 0.6042 | 0.0000 | 1.0000 | 60.42% | 288 |
| Recall@5 | 0.5278 | 0.0000 | 1.0000 | 60.42% | 288 |
| Precision@5 | 0.1313 | 0.0000 | 0.4000 | 0.00% | 288 |
| MRR@5 | 0.4138 | 0.0000 | 1.0000 | 42.01% | 288 |
| NDCG@5 | 0.4167 | 0.0000 | 1.0000 | 45.49% | 288 |
| MAP@5 | 0.3615 | 0.0000 | 1.0000 | 39.93% | 288 |

IR metrics use binary relevance against labeled chunk IDs; unlabeled alternatives count as nonrelevant.
Pass rate is the fraction of scored queries meeting the configured threshold (0.5); it is not a retrieval accuracy measure.
Scored queries: 288; excluded without relevance labels: 12.
Warning: 12 queries have no relevance labels and are excluded from IR averages.

## Run Settings
- mode: live
- dataset: review-web/test_dataset_verified.json
- results: None
- retriever: advanced_dense
- retriever_mode: advanced
- top_k: 5
- chunks: chunks_storage/chunks.parquet
- dense_only: True
- skip_rerank: True
- collection_name: VacxiS_Capstone_FA26
- model_path: models/bge-m3

## Breakdown by Answerability
| Answerability | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| ANSWERABLE | 197 | 0.3350 | 0.2766 | 0.3350 | 0.3350 | 0.3350 | 0.2766 | 0.5431 | 0.4721 | 0.1912 | 0.4247 | 0.4081 | 0.3672 | 0.6497 | 0.5711 | 0.1391 | 0.4488 | 0.4507 | 0.3912 |
| PARTIALLY_ANSWERABLE | 91 | 0.2527 | 0.2143 | 0.2527 | 0.2527 | 0.2527 | 0.2143 | 0.3736 | 0.3187 | 0.1355 | 0.3077 | 0.2909 | 0.2674 | 0.5055 | 0.4341 | 0.1143 | 0.3379 | 0.3430 | 0.2973 |
| UNANSWERABLE | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Breakdown by Complexity
| Complexity | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| MULTI_HOP | 90 | 0.3529 | 0.1765 | 0.3529 | 0.3529 | 0.3529 | 0.1765 | 0.5529 | 0.3294 | 0.2196 | 0.4392 | 0.3234 | 0.2627 | 0.6941 | 0.4353 | 0.1741 | 0.4710 | 0.3762 | 0.2939 |
| SINGLE_HOP | 150 | 0.3542 | 0.3542 | 0.3542 | 0.3542 | 0.3542 | 0.3542 | 0.5417 | 0.5417 | 0.1806 | 0.4375 | 0.4643 | 0.4375 | 0.6319 | 0.6319 | 0.1264 | 0.4583 | 0.5016 | 0.4583 |
| TABLE_REASONING | 60 | 0.1356 | 0.1356 | 0.1356 | 0.1356 | 0.1356 | 0.1356 | 0.2712 | 0.2712 | 0.0904 | 0.1921 | 0.2123 | 0.1921 | 0.4068 | 0.4068 | 0.0814 | 0.2226 | 0.2677 | 0.2226 |

## Breakdown by Query Style
| Query Style | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| CLINICAL_STYLE | 120 | 0.2807 | 0.2368 | 0.2807 | 0.2807 | 0.2807 | 0.2368 | 0.4649 | 0.3904 | 0.1608 | 0.3582 | 0.3405 | 0.3056 | 0.5614 | 0.4868 | 0.1211 | 0.3801 | 0.3824 | 0.3301 |
| PATIENT_STYLE | 180 | 0.3276 | 0.2701 | 0.3276 | 0.3276 | 0.3276 | 0.2701 | 0.5057 | 0.4454 | 0.1820 | 0.4071 | 0.3911 | 0.3554 | 0.6322 | 0.5546 | 0.1379 | 0.4358 | 0.4391 | 0.3821 |

## Breakdown by Difficulty
| Difficulty | Count | Hit@1 | Recall@1 | Precision@1 | MRR@1 | NDCG@1 | MAP@1 | Hit@3 | Recall@3 | Precision@3 | MRR@3 | NDCG@3 | MAP@3 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | NDCG@5 | MAP@5 |
|  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |  :---  |
| UNSPECIFIED | 300 | 0.3090 | 0.2569 | 0.3090 | 0.3090 | 0.3090 | 0.2569 | 0.4896 | 0.4236 | 0.1736 | 0.3877 | 0.3711 | 0.3356 | 0.6042 | 0.5278 | 0.1313 | 0.4138 | 0.4167 | 0.3615 |