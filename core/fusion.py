
"""
LeafMatch AI - Weighted Score-Level Late Fusion

Konfigurasi terpilih dari Notebook 06:
- Visual weight = 0.3
- Text weight = 0.7

Formula:
S_final = alpha * S_visual + (1-alpha) * S_text
"""

import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

VISUAL_WEIGHT = 0.3
TEXT_WEIGHT = 0.7


class FusionError(ValueError):
    """Kesalahan pada proses weighted score fusion."""
    pass


# ============================================================
# 2. VALIDATE SIMILARITY SCORES
# ============================================================

def validate_similarity_scores(scores, name):
    """Memastikan skor similarity valid."""

    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    if scores.ndim != 2 or scores.size == 0:
        raise FusionError(
            f"{name} harus berupa matriks 2D yang tidak kosong."
        )

    if not np.isfinite(scores).all():
        raise FusionError(
            f"{name} mengandung NaN atau infinity."
        )

    if np.any(scores < -1e-10) or np.any(scores > 1 + 1e-10):
        raise FusionError(
            f"{name} harus berada dalam rentang 0 sampai 1."
        )

    return np.clip(scores, 0.0, 1.0)


# ============================================================
# 3. WEIGHTED SCORE FUSION
# ============================================================

def weighted_score_fusion(
    visual_scores,
    text_scores,
    alpha=VISUAL_WEIGHT
):
    """
    Menggabungkan skor visual dan teks.

    Parameters:
        visual_scores : array (n_query, n_gallery)
        text_scores   : array (n_query, n_gallery)
        alpha         : bobot visual (default 0.3)

    Returns:
        final_scores : array (n_query, n_gallery)
    """

    visual = validate_similarity_scores(
        visual_scores, "Visual similarity"
    )

    text = validate_similarity_scores(
        text_scores, "Text similarity"
    )

    if visual.shape != text.shape:
        raise FusionError(
            "Dimensi skor visual dan teks harus sama."
        )

    alpha = float(alpha)

    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise FusionError(
            "Bobot alpha harus berada di antara 0 dan 1."
        )

    final_scores = (
        alpha * visual
        + (1.0 - alpha) * text
    )

    return np.clip(final_scores, 0.0, 1.0)
