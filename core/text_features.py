
"""
LeafMatch AI - Text Feature Extraction

Preprocessing teks mengikuti Notebook 02.
TF-IDF menggunakan model gallery-fitted dari Notebook 06.

Konfigurasi:
- TF-IDF Unigram + Bigram
- Vocabulary dari 62 sampel gallery
- Tidak melakukan fitting ulang
"""

from pathlib import Path
from functools import lru_cache
import re
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RUN_ID = "20261127T090108Z_c12d4354b8"

MODEL_PATH = (
    ROOT
    / "results"
    / "artifacts"
    / "retrieval_evaluation"
    / "runs"
    / RUN_ID
    / "tfidf_unigram_bigram_gallery_fitted.joblib"
)


# ============================================================
# 2. CUSTOM ERROR
# ============================================================

class TextFeatureError(Exception):
    """Error pada pemrosesan atau ekstraksi fitur teks."""
    pass


# ============================================================
# 3. CLASS NAME ALIASES
# ============================================================

CLASS_ALIASES = [
    "daun mangkok",
    "mangkok",
    "mangkuk",
    "daun salam",
    "salam",
    "daun sambang darah",
    "sambang darah",
    "daun sirih",
    "sirih",
    "daun tapak kuda",
    "tapak kuda",
]


# ============================================================
# 4. BASIC TEXT CLEANING
# ============================================================

def clean_text_basic(text):
    """
    Mengikuti preprocessing teks Notebook 02:
    - Unicode NFKC
    - Case folding
    - Menghapus karakter format
    - Mengganti tanda baca dan simbol dengan spasi
    - Normalisasi whitespace
    """

    if not isinstance(text, str):
        raise TextFeatureError(
            "Deskripsi daun harus berupa teks."
        )

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Cf"
    )

    text = "".join(
        " "
        if unicodedata.category(char).startswith(("P", "S"))
        else char
        for char in text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 5. REMOVE CLASS NAMES
# ============================================================

NORMALIZED_ALIASES = sorted(
    {clean_text_basic(alias) for alias in CLASS_ALIASES},
    key=lambda alias: (-len(alias), alias)
)

CLASS_NAME_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(alias) for alias in NORMALIZED_ALIASES)
    + r")(?!\w)",
    flags=re.IGNORECASE
)


def preprocess_text_query(text):
    """
    Membersihkan deskripsi query agar sesuai
    dengan cleaned_description pada Notebook 02.
    """

    cleaned = clean_text_basic(text)

    cleaned = CLASS_NAME_PATTERN.sub(
        " ",
        cleaned
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip()

    if not cleaned:
        raise TextFeatureError(
            "Masukkan deskripsi karakteristik daun "
            "yang valid, bukan hanya nama tanaman."
        )

    return cleaned


# ============================================================
# 6. LOAD FITTED TF-IDF MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_gallery_vectorizer():
    """
    Memuat TF-IDF fitted dari Notebook 06.

    Model hanya dimuat dari artefak lokal terpercaya.
    Tidak dilakukan fit atau fit_transform.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model TF-IDF tidak ditemukan: {MODEL_PATH}"
        )

    vectorizer = joblib.load(MODEL_PATH)

    if not isinstance(vectorizer, TfidfVectorizer):
        raise TextFeatureError(
            "Artefak bukan TfidfVectorizer."
        )

    if vectorizer.ngram_range != (1, 2):
        raise TextFeatureError(
            "Konfigurasi TF-IDF bukan Unigram + Bigram."
        )

    if not hasattr(vectorizer, "vocabulary_"):
        raise TextFeatureError(
            "TF-IDF belum memiliki vocabulary fitted."
        )

    return vectorizer


# ============================================================
# 7. TRANSFORM QUERY TEXT
# ============================================================

def transform_text_query(text):
    """
    Mengubah deskripsi menjadi vektor TF-IDF.

    Return:
        scipy.sparse.csr_matrix
        shape: (1, vocabulary_size)
    """

    cleaned_text = preprocess_text_query(text)

    vectorizer = load_gallery_vectorizer()

    # Hanya transform; tidak melakukan fitting ulang
    vector = vectorizer.transform(
        [cleaned_text]
    ).tocsr()

    if vector.nnz == 0:
        raise TextFeatureError(
            "Deskripsi tidak memiliki kata yang dikenali "
            "oleh vocabulary TF-IDF. Gunakan karakteristik "
            "daun yang lebih spesifik."
        )

    if not np.isfinite(vector.data).all():
        raise TextFeatureError(
            "Vektor TF-IDF mengandung nilai tidak valid."
        )

    return vector


# ============================================================
# 8. MODEL INFORMATION
# ============================================================

def get_text_model_info():
    """Informasi model TF-IDF yang sedang digunakan."""

    vectorizer = load_gallery_vectorizer()

    return {
        "model": "TF-IDF",
        "configuration": "Unigram + Bigram",
        "ngram_range": vectorizer.ngram_range,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "fitted_on": "Gallery (62 samples)",
        "run_id": RUN_ID,
    }
