
"""
LeafMatch AI - Gallery Loader

Memuat artefak gallery dari Notebook 06 dan 07.

Menggunakan:
- 62 sampel gallery
- Visual features: 143 dimensi
- TF-IDF features: 233 dimensi
- Gallery-fitted visual normalizer
- Weighted late fusion: 0.3 visual, 0.7 text

Tidak melakukan fitting atau training ulang.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from core.image_features import (
    load_gallery_normalizer,
    transform_visual_features,
)

from core.similarity import calculate_visual_distance_scale
from core.retrieval import retrieve_top_k
from core.text_features import load_gallery_vectorizer


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "20261127T090108Z_c12d4354b8"

ARTIFACT_DIR = (
    ROOT
    / "results"
    / "artifacts"
    / "retrieval_evaluation"
    / "runs"
    / RUN_ID
)

SPLIT_PATH = (
    ROOT
    / "results"
    / "metrics"
    / "retrieval_evaluation"
    / "runs"
    / RUN_ID
    / "751_final_dataset_split_inventory.csv"
)

METADATA_PATH = (
    ROOT
    / "dataset"
    / "processed"
    / "metadata_processed.csv"
)


# ============================================================
# CSV LOADER
# ============================================================

def read_csv(path):
    with open(
        path,
        newline="",
        encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


# ============================================================
# LOAD GALLERY ARTIFACTS
# ============================================================

@lru_cache(maxsize=1)
def load_gallery_data():
    """
    Memuat seluruh artefak gallery.

    Hanya dijalankan sekali selama proses Flask aktif.
    """

    # --------------------------------------------------------
    # 1. Metadata and split inventory
    # --------------------------------------------------------

    metadata = read_csv(METADATA_PATH)
    inventory = read_csv(SPLIT_PATH)

    gallery_rows = sorted(
        [
            row
            for row in inventory
            if row["split"] == "gallery"
        ],
        key=lambda row: int(row["matrix_row_index"])
    )

    gallery_indices = [
        int(row["matrix_row_index"])
        for row in gallery_rows
    ]

    # --------------------------------------------------------
    # 2. Align gallery metadata
    # --------------------------------------------------------

    gallery_metadata = []

    for row in gallery_rows:
        index = int(row["matrix_row_index"])

        if not 0 <= index < len(metadata):
            raise ValueError(
                f"Invalid gallery index: {index}"
            )

        sample = metadata[index]

        if sample["sample_id"] != row["sample_id"]:
            raise ValueError(
                "Gallery sample ID mismatch."
            )

        if sample["kelas"] != row["class_label"]:
            raise ValueError(
                "Gallery class label mismatch."
            )

        gallery_metadata.append(dict(sample))

    # --------------------------------------------------------
    # 3. Load normalized visual gallery
    # --------------------------------------------------------

    with np.load(
        ARTIFACT_DIR / "visual_gallery_fitted_splits.npz",
        allow_pickle=False
    ) as data:

        gallery_visual = np.asarray(
            data["gallery"],
            dtype=np.float64
        ).copy()

    # --------------------------------------------------------
    # 4. Load fitted visual normalizer
    # --------------------------------------------------------

    normalizer = load_gallery_normalizer()

    # --------------------------------------------------------
    # 5. Load gallery TF-IDF
    # --------------------------------------------------------

    gallery_text = load_npz(
        ARTIFACT_DIR
        / "tfidf_unigram_bigram_gallery.npz"
    ).tocsr()

    vectorizer = load_gallery_vectorizer()

    # --------------------------------------------------------
    # 6. Validate gallery dimensions
    # --------------------------------------------------------

    n_gallery = len(gallery_metadata)

    if n_gallery != 62:
        raise ValueError(
            f"Expected 62 gallery samples, got {n_gallery}"
        )

    if gallery_visual.shape != (n_gallery, 143):
        raise ValueError(
            "Invalid visual gallery dimensions."
        )

    if gallery_text.shape != (
        n_gallery,
        len(vectorizer.vocabulary_)
    ):
        raise ValueError(
            "Invalid text gallery dimensions."
        )

    # --------------------------------------------------------
    # 7. Automatic visual consistency verification
    # --------------------------------------------------------

    raw_visual = np.load(
        ARTIFACT_DIR / "visual_raw_143.npy",
        allow_pickle=False
    )

    reconstructed_visual = transform_visual_features(
        raw_visual[gallery_indices],
        normalizer
    )

    if not np.allclose(
        reconstructed_visual,
        gallery_visual,
        rtol=1e-10,
        atol=1e-10
    ):
        raise ValueError(
            "Visual gallery does not match "
            "the fitted normalizer."
        )

    # --------------------------------------------------------
    # 8. Visual distance scale
    # --------------------------------------------------------

    distance_scale = calculate_visual_distance_scale(
        gallery_visual
    )

    return {
        "visual_features": gallery_visual,
        "text_features": gallery_text,
        "metadata": gallery_metadata,
        "normalizer": normalizer,
        "distance_scale": distance_scale,
        "gallery_size": n_gallery,
        "run_id": RUN_ID,
    }


# ============================================================
# SEARCH USING REAL GALLERY
# ============================================================

def search_gallery(
    query_visual_raw,
    query_description,
    top_k=5
):
    """
    Menjalankan multimodal retrieval terhadap
    gallery asli.

    Parameters
    ----------
    query_visual_raw:
        Fitur gambar mentah (143 dimensi)
        hasil extract_features_from_preprocessed().

    query_description:
        Deskripsi daun berbahasa Indonesia.

    top_k:
        Jumlah hasil pencarian.

    Returns
    -------
    list[dict]:
        Top-K retrieval results beserta metadata.
    """

    gallery = load_gallery_data()

    # Normalize uploaded image features
    query_visual_normalized = transform_visual_features(
        query_visual_raw,
        gallery["normalizer"]
    )

    # Run multimodal retrieval
    results = retrieve_top_k(
        query_visual_features=query_visual_normalized,
        query_description=query_description,
        gallery_visual_features=gallery["visual_features"],
        gallery_text_features=gallery["text_features"],
        gallery_metadata=gallery["metadata"],
        top_k=top_k,
        distance_scale=gallery["distance_scale"],
    )

    return results
