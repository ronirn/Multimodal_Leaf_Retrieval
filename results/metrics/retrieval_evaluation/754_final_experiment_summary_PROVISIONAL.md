# Notebook 06 - Final Experiment Summary

**Technical status:** TECHNICAL_COMPLETE_PROVISIONAL
**Scientific interpretation:** PROVISIONAL_AUDIT_UNRESOLVED

## Dataset and Frozen Configuration
- Total samples: 104
- Gallery / validation / test: 62 / 21 / 21
- Selected text: Unigram + Bigram
- Visual weight alpha: 0.3
- Selection metric: validation mAP@5
- Test scores were not used to select or alter the configuration.

## Retrieval Metrics at K=5
| Metric | Validation | Test (provisional) | Test - Validation |
|---|---:|---:|---:|
| Precision@5 | 0.7905 | 0.8381 | +0.0476 |
| Recall@5 | 0.3148 | 0.3341 | +0.0193 |
| MRR@5 | 0.8532 | 1.0000 | +0.1468 |
| mAP@5 | 0.7200 | 0.8087 | +0.0887 |
| nDCG@5 | 0.7854 | 0.8721 | +0.0867 |

## Audit and Interpretation Limits
- Flagged cross-split near-image pairs: 44; these are not automatically exact duplicates.
- Repeated-description groups crossing splits: 4.
- Source specimen/capture grouping and Notebook 03 extractor equivalence remain unverified.
- Test results are exploratory/provisional, not approved independent generalization estimates.
- Do not tune alpha or select a model using test scores.

**Technical checks:** 15/15

## Final Conclusion
The retrieval pipeline generated validation and provisional test analyses under the frozen validation-selected configuration. Methodological review remains required before independent-performance claims.
