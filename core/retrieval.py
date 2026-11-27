
"""
LeafMatch AI - Multimodal Retrieval

Pipeline:
1. Terima fitur visual query yang sudah dinormalisasi
2. Transformasi deskripsi query menggunakan TF-IDF
3. Hitung visual similarity (Euclidean)
4. Hitung text similarity (Cosine)
5. Gabungkan skor menggunakan weighted late fusion
6. Urutkan Top-K hasil retrieval

Konfigurasi:
Visual = 0.3
Text = 0.7
Top-K = 5

Catatan:
Fitur gallery harus berasal dari artefak eksperimen
yang sama dengan model TF-IDF dan normalizer visual.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.text_features import transform_text_query
from core.similarity import (
    validate_visual_features,
    calculate_visual_similarity,
    calculate_text_similarity,
)
from core.fusion import weighted_score_fusion


class RetrievalError(ValueError):
    """Kesalahan pada proses multimodal retrieval."""
    pass


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5


# ============================================================
# 2. MULTIMODAL RETRIEVAL
# ============================================================

def retrieve_top_k(
    query_visual_features,
    query_description: str,
    gallery_visual_features,
    gallery_text_features,
    gallery_metadata: Sequence[Mapping[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    distance_scale: float | None = None,
) -> list[dict]:
    """
    Melakukan pencarian daun menggunakan dua modalitas.

    Parameters
    ----------
    query_visual_features:
        Fitur visual query berukuran (1, 143),
        sudah dinormalisasi dengan normalizer gallery.

    query_description:
        Deskripsi karakteristik daun dalam bahasa Indonesia.

    gallery_visual_features:
        Fitur visual gallery berukuran (N, 143),
        sudah dinormalisasi.

    gallery_text_features:
        Matriks TF-IDF gallery berukuran (N, vocabulary_size).

    gallery_metadata:
        Metadata gallery sesuai urutan baris matriks fitur.

    top_k:
        Jumlah hasil pencarian yang ditampilkan.

    distance_scale:
        Skala jarak visual dari eksperimen.
        Jika None, dihitung dari fitur gallery.

    Returns
    -------
    list[dict]:
        Hasil retrieval yang diurutkan berdasarkan
        skor fusion tertinggi.
    """

    # ========================================================
    # 3. VALIDASI INPUT
    # ========================================================

    query_visual = validate_visual_features(
        query_visual_features
    )

    gallery_visual = validate_visual_features(
        gallery_visual_features
    )

    if query_visual.shape[0] != 1:
        raise RetrievalError(
            "Retrieval hanya menerima satu query per pencarian."
        )

    gallery_size = gallery_visual.shape[0]

    if len(gallery_metadata) != gallery_size:
        raise RetrievalError(
            "Jumlah metadata tidak sesuai dengan gallery visual."
        )

    if gallery_text_features.shape[0] != gallery_size:
        raise RetrievalError(
            "Jumlah fitur teks tidak sesuai dengan gallery."
        )

    if isinstance(top_k, bool) or not isinstance(
        top_k, int
    ) or top_k < 1:
        raise RetrievalError(
            "top_k harus berupa bilangan bulat positif."
        )

    # ========================================================
    # 4. TEXT FEATURE EXTRACTION
    # ========================================================

    query_text = transform_text_query(
        query_description
    )

    if query_text.shape[1] != gallery_text_features.shape[1]:
        raise RetrievalError(
            "Vocabulary TF-IDF query dan gallery tidak sesuai."
        )

    # ========================================================
    # 5. VISUAL SIMILARITY
    # ========================================================

    visual_scores = calculate_visual_similarity(
        query_visual,
        gallery_visual,
        distance_scale=distance_scale,
    )

    # ========================================================
    # 6. TEXT SIMILARITY
    # ========================================================

    text_scores = calculate_text_similarity(
        query_text,
        gallery_text_features,
    )

    # ========================================================
    # 7. WEIGHTED SCORE-LEVEL LATE FUSION
    # ========================================================

    final_scores = weighted_score_fusion(
        visual_scores,
        text_scores,
        alpha=0.3,
    )

    # ========================================================
    # 8. TOP-K RANKING
    # ========================================================

    scores = final_scores[0]

    # Urutkan dari skor tertinggi.
    # Jika skor sama, pertahankan urutan indeks gallery.
    ranking = np.lexsort(
        (np.arange(gallery_size), -scores)
    )

    selected_indices = ranking[
        :min(top_k, gallery_size)
    ]

    # ========================================================
    # 9. BUILD RETRIEVAL RESULTS
    # ========================================================

    results = []

    for rank, index in enumerate(
        selected_indices,
        start=1,
    ):
        index = int(index)

        item = dict(gallery_metadata[index])

        item.update(
            {
                "rank": rank,
                "gallery_index": index,
                "visual_similarity": float(
                    visual_scores[0, index]
                ),
                "text_similarity": float(
                    text_scores[0, index]
                ),
                "fusion_score": float(
                    final_scores[0, index]
                ),
            }
        )

        results.append(item)

    return results
