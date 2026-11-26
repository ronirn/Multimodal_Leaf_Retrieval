# Final Summary — Multimodal Medicinal Leaf Retrieval

## 1. Dataset

The full gallery contains 104 paired leaf samples from 5 classes.

| class_name    |   sample_count |
|:--------------|---------------:|
| Mangkok       |             20 |
| Salam         |             19 |
| Sambang Darah |             20 |
| Sirih         |             20 |
| Tapak Kuda    |             25 |

## 2. Feature Representations

- Visual: HSV Histogram (64), GLCM (72), Hu Moments (7).

- Text: TF-IDF Unigram (68) and Unigram + Bigram (261).

## 3. Similarity and Fusion

- Visual: Euclidean Distance converted through inverse-distance similarity.

- Text: Cosine Similarity.

- Fusion: alpha * Visual Similarity + (1 - alpha) * Text Similarity.

- Provisional alpha = 0.5; no optimal alpha selected.

## 4. Full-Gallery Descriptive Results

| TF-IDF | Method | Mean Gap | Gallery Top-1 (%) | Gallery Top-5 Share |
|---|---|---:|---:|---:|
| Unigram | Visual Only | 0.04559 | 84.62 | 0.68846 |
| Unigram | Text Only | 0.23516 | 87.50 | 0.82308 |
| Unigram | Weighted Fusion | 0.14037 | 87.50 | 0.85000 |
| Unigram + Bigram | Visual Only | 0.04559 | 84.62 | 0.68846 |
| Unigram + Bigram | Text Only | 0.23506 | 91.35 | 0.83654 |
| Unigram + Bigram | Weighted Fusion | 0.14032 | 93.27 | 0.85769 |

The score gap and class-share statistics above are descriptive diagnostics, not held-out retrieval accuracy.

## 5. Cross-Cell and Artifact Validation

- Final checks passed: 59/59.

- Artifact integrity entries checked: 13.

- Artifact files verified using SHA-256 hashes from the Cell 19 manifest.

## 6. Limitations

- Gallery samples are used as queries against the same reference gallery, with self-matches excluded.

- Class agreement is not independent held-out test performance.

- Similarity scores are not calibrated class-membership probabilities.

- The alpha grid was analyzed for sensitivity, not used to select a final best configuration.

- External raw-image and raw-text queries require the original fitted preprocessing and vectorization pipeline.

## 7. Next Methodological Steps

Use a dedicated validation split for fusion-weight and text-configuration selection, followed by independent test evaluation.

Report appropriate retrieval metrics, such as Precision@K, Recall@K and MRR, with clearly defined relevance and non-leaking gallery/query partitions.
