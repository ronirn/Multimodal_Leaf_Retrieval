# Notebook 06 - Weighted Score Fusion: Experiment Report

Run ID: `20261127T090108Z_c12d4354b8`
Pipeline version: `06_wsf_final_v1.0`
Test status: **PROVISIONAL_AUDIT_UNRESOLVED** (independent estimate: False)

## Data and features
- Samples: 104 (5 classes) - gallery 62, validation 21, test 21 (group-aware, seed 42).
- Visual: Notebook 03 vectors, 143 dims (HSV 64 + GLCM 72 + Hu 7), NB03 normalizer fitted on gallery only.
- Text: cleaned_description, TF-IDF fitted on gallery only (vocabulary: Unigram = 66, Unigram + Bigram = 233).
- Visual similarity: S_visual = 1 / (1 + d_euclidean / median_nonzero_gallery_distance) (scale = 2.262902).
- Fusion: S_fusion = alpha * S_visual + (1 - alpha) * S_text.

## Validation-only selection
- Selected: **Unigram + Bigram, alpha = 0.3** (Weighted Late Fusion); validation mAP@5 = 0.7052.
- The legacy alpha = 0.3 was not reused; alpha was re-selected from the corrected features.

### Validation metrics at K=5
| method | Precision@K | Recall@K | MRR@K | mAP@K | nDCG@K |
|---|---|---|---|---|---|
| Visual Only | 0.6000 | 0.2374 | 0.9048 | 0.5522 | 0.6525 |
| Text Only - Unigram | 0.6952 | 0.2748 | 0.7738 | 0.6246 | 0.6988 |
| Text Only - Unigram + Bigram | 0.7619 | 0.3037 | 0.8492 | 0.6911 | 0.7623 |
| Weighted Fusion - Unigram + Bigram (alpha=0.5, default) | 0.7429 | 0.2942 | 0.8770 | 0.6933 | 0.7565 |
| Weighted Fusion - Unigram + Bigram (alpha=0.3) [selected] | 0.7714 | 0.3061 | 0.8532 | 0.7052 | 0.7715 |

## Test (PROVISIONAL_AUDIT_UNRESOLVED)
| method | Precision@K | Recall@K | MRR@K | mAP@K | nDCG@K |
|---|---|---|---|---|---|
| Weighted Fusion - Unigram + Bigram (alpha=0.3) | 0.8286 | 0.3294 | 1.0000 | 0.7941 | 0.8573 |
| Visual Only | 0.6286 | 0.2538 | 0.9444 | 0.5878 | 0.6940 |
| Text Only - Unigram + Bigram | 0.8381 | 0.3333 | 0.9524 | 0.7908 | 0.8509 |

- Selected configuration: test mAP@5 = 0.7941 vs validation 0.7052.
- Top-1 errors: validation 5/21, test 0/21.
- Top-1 neighbour has an identical description: validation 1, test 3.
- Top-1 neighbour is a flagged near-duplicate image: validation 0, test 2.

## Audit status
- Image hash source: processed_images_recomputed.
- Cross-split near-duplicate candidates: 44 (open pair reviews: 44).
- Cross-split repeated-description groups: 4.
- Specimen identity column: None.
- Query samples with visual features far outside the gallery range (|z| > 10): 7 (leaf_1a80f5749cc68900, leaf_3c3fafa9c7913736, leaf_45414f432e49469a, leaf_620d73b8cfb217b6, leaf_8809ccde35d4b945, leaf_a838808299383884, leaf_ef0854556175e2b0).
- Test inspected in earlier runs: True.
- Human sign-off valid: False.

Open issues:
| issue_id | finding_count | resolution_state |
|---|---|---|
| IMG_NEAR_SPLIT | 44 | PENDING_PAIR_REVIEWS |
| TEXT_REPEAT_SPLIT | 4 | PENDING_REVIEW |
| SPECIMEN_PROVENANCE | 1 | PENDING_REVIEW |
| TEST_PREVIOUSLY_INSPECTED | 1 | PENDING_REVIEW |

## Interpretation constraints
- Technical execution success does not establish methodological validity.
- Test numbers must be reported with the status label above. They are not an independent
  generalisation estimate while the status is not INDEPENDENT_TEST_APPROVED.
- Candidate near-duplicates and repeated descriptions are not proof of leakage, but they may
  inflate retrieval scores until they are reviewed.
