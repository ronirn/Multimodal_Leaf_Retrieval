
"""
LeafMatch AI - Similarity Calculation

Mengikuti Notebook 06:
1. Visual: Euclidean Distance
2. Text: Cosine Similarity

Visual similarity:
    S = 1 / (1 + distance / scale)

Scale dihitung dari median jarak Euclidean
nonzero antarsampel gallery.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityError(ValueError):
    """Kesalahan pada perhitungan similarity."""
    pass


# ============================================================
# 1. VALIDASI FITUR VISUAL
# ============================================================

def validate_visual_features(features):
    """Pastikan matriks fitur visual valid."""

    matrix = np.asarray(features, dtype=np.float64)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SimilarityError(
            "Fitur visual harus berupa matriks 2D."
        )

    if matrix.shape[1] != 143:
        raise SimilarityError(
            "Fitur visual harus memiliki 143 dimensi."
        )

    if not np.isfinite(matrix).all():
        raise SimilarityError(
            "Fitur visual mengandung NaN atau infinity."
        )

    return matrix


# ============================================================
# 2. VISUAL DISTANCE SCALE
# ============================================================

def calculate_visual_distance_scale(gallery_features):
    """
    Hitung skala menggunakan gallery saja.

    Sesuai Notebook 06:
    median nonzero gallery-gallery Euclidean distance.
    """

    gallery = validate_visual_features(gallery_features)

    if gallery.shape[0] < 2:
        raise SimilarityError(
            "Gallery minimal berisi dua sampel."
        )

    distances = cdist(
        gallery,
        gallery,
        metric="euclidean"
    )

    upper = distances[
        np.triu_indices(len(gallery), k=1)
    ]

    nonzero = upper[upper > 0]

    if nonzero.size == 0:
        raise SimilarityError(
            "Tidak dapat menghitung distance scale."
        )

    scale = float(np.median(nonzero))

    if not np.isfinite(scale) or scale <= 0:
        raise SimilarityError(
            "Visual distance scale tidak valid."
        )

    return scale


# ============================================================
# 3. VISUAL SIMILARITY
# ============================================================

def calculate_visual_similarity(
    query_features,
    gallery_features,
    distance_scale=None
):
    """
    Menghitung similarity gambar menggunakan
    Euclidean Distance.

    Input harus berupa fitur yang sudah dinormalisasi
    dengan normalizer gallery dari Notebook 06.

    Output:
        Matriks similarity (n_query, n_gallery)
    """

    query = validate_visual_features(query_features)
    gallery = validate_visual_features(gallery_features)

    if distance_scale is None:
        distance_scale = calculate_visual_distance_scale(
            gallery
        )

    scale = float(distance_scale)

    if not np.isfinite(scale) or scale <= 0:
        raise SimilarityError(
            "Distance scale harus positif."
        )

    distances = cdist(
        query,
        gallery,
        metric="euclidean"
    )

    similarities = 1.0 / (
        1.0 + distances / scale
    )

    return similarities


# ============================================================
# 4. TEXT SIMILARITY
# ============================================================

def calculate_text_similarity(
    query_tfidf,
    gallery_tfidf
):
    """
    Menghitung Cosine Similarity.

    Input:
        Query TF-IDF
        Gallery TF-IDF

    Output:
        Matriks similarity (n_query, n_gallery)
    """

    if query_tfidf.shape[1] != gallery_tfidf.shape[1]:
        raise SimilarityError(
            "Dimensi TF-IDF query dan gallery berbeda."
        )

    if (
        query_tfidf.shape[0] == 0
        or gallery_tfidf.shape[0] == 0
    ):
        raise SimilarityError(
            "Matriks TF-IDF tidak boleh kosong."
        )

    similarities = cosine_similarity(
        query_tfidf,
        gallery_tfidf,
        dense_output=True
    )

    if not np.isfinite(similarities).all():
        raise SimilarityError(
            "Cosine Similarity mengandung nilai tidak valid."
        )

    return np.clip(
        similarities,
        0.0,
        1.0
    )
