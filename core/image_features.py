"""Visual feature extraction for the Multimodal Leaf Retrieval application.

Re-implements, without changing behaviour, the visual features of
``notebooks/03_image_feature_extraction.ipynb``:

    HSV histogram  64 dims  (Cell 6)   H 32 bins [0,180), S 16 [0,256), V 16 [0,256),
                                       foreground mask, L1 normalised per channel
    GLCM           72 dims  (Cell 8)   gray -> 32 levels, distances 1/2/3,
                                       angles 0/45/90/135 (discrete offsets),
                                       both pixels in foreground, symmetric, normed,
                                       6 properties, order Property -> Distance -> Angle
    Hu Moments      7 dims  (Cell 10)  cv2.moments(mask, binaryImage=True),
                                       -sign(hu) * log10(max(|hu|, 1e-30)), zero kept

Concatenation order (Cell 15): HSV | GLCM | Hu = 143 raw dimensions; this is
``dataset/features/visual/visual_features_raw.npy`` (and Notebook 06's
``visual_raw_143.npy``).

Raw extraction and normalisation are separate:

* :func:`extract_visual_features` -> raw 143-d vector (no fitting).
* :func:`transform_visual_features` -> applies an *already fitted* normaliser.
  The normaliser used for retrieval is the gallery-only one saved by
  Notebook 06 (``visual_gallery_fitted_normalizer.joblib``); it is loaded with
  :func:`load_gallery_normalizer` and never refitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from skimage.feature import graycoprops

from core.preprocessing import PreprocessedImage, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DTYPE = np.float64

# ---------------------------------------------------------------------------
# Configuration (Notebook 03, Cell 5)
# ---------------------------------------------------------------------------

TARGET_HEIGHT = 256
TARGET_WIDTH = 256
EXPECTED_IMAGE_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 3)
EXPECTED_MASK_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH)

# Cell 4: MIN_FOREGROUND_PIXELS = preprocessing_config["roi_min_mask_pixels"]
MIN_FOREGROUND_PIXELS = int(load_config().raw.get("roi_min_mask_pixels", 100))

# HSV
HSV_H_BINS, HSV_S_BINS, HSV_V_BINS = 32, 16, 16
HSV_BINS = (HSV_H_BINS, HSV_S_BINS, HSV_V_BINS)
HSV_RANGES = ((0, 180), (0, 256), (0, 256))
HSV_FEATURE_DIM = sum(HSV_BINS)  # 64
HSV_FEATURE_NAMES = (
    [f"hsv_h_{i:02d}" for i in range(HSV_H_BINS)]
    + [f"hsv_s_{i:02d}" for i in range(HSV_S_BINS)]
    + [f"hsv_v_{i:02d}" for i in range(HSV_V_BINS)]
)

# GLCM
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 2, 3)
GLCM_ANGLES_DEGREES = (0, 45, 90, 135)
GLCM_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_SYMMETRIC = True
GLCM_NORMED = True
# (row direction, column direction); rows grow downward, columns to the right.
GLCM_DIRECTION_OFFSETS = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}
GLCM_OFFSET_LOOKUP = {
    (d, a): (d * GLCM_DIRECTION_OFFSETS[a][0], d * GLCM_DIRECTION_OFFSETS[a][1])
    for d in GLCM_DISTANCES
    for a in GLCM_ANGLES_DEGREES
}
GLCM_FEATURE_DIM = len(GLCM_PROPERTIES) * len(GLCM_DISTANCES) * len(GLCM_ANGLES_DEGREES)  # 72
GLCM_FEATURE_NAMES = [
    f"glcm_{prop.lower()}_d{d}_a{a}"
    for prop in GLCM_PROPERTIES
    for d in GLCM_DISTANCES
    for a in GLCM_ANGLES_DEGREES
]

# Hu Moments
HU_FEATURE_DIM = 7
HU_LOG_EPSILON = 1e-30
HU_FEATURE_NAMES = [f"hu_moment_{i}" for i in range(1, HU_FEATURE_DIM + 1)]

# Combined
FEATURE_GROUP_ORDER = ("HSV", "GLCM", "Hu Moments")
FEATURE_GROUP_SLICES = {
    "HSV": (0, HSV_FEATURE_DIM),
    "GLCM": (HSV_FEATURE_DIM, HSV_FEATURE_DIM + GLCM_FEATURE_DIM),
    "Hu Moments": (HSV_FEATURE_DIM + GLCM_FEATURE_DIM, HSV_FEATURE_DIM + GLCM_FEATURE_DIM + HU_FEATURE_DIM),
}
TOTAL_VISUAL_FEATURE_DIM = HSV_FEATURE_DIM + GLCM_FEATURE_DIM + HU_FEATURE_DIM  # 143
VISUAL_FEATURE_NAMES = HSV_FEATURE_NAMES + GLCM_FEATURE_NAMES + HU_FEATURE_NAMES

# Gallery-only normaliser fitted by Notebook 06 (locked run).
DEFAULT_RUN_ID = "20261127T090108Z_c12d4354b8"
DEFAULT_NORMALIZER_PATH = (
    PROJECT_ROOT
    / "results"
    / "artifacts"
    / "retrieval_evaluation"
    / "runs"
    / DEFAULT_RUN_ID
    / "visual_gallery_fitted_normalizer.joblib"
)

assert len(VISUAL_FEATURE_NAMES) == TOTAL_VISUAL_FEATURE_DIM == 143


class FeatureExtractionError(ValueError):
    """Raised when features cannot be extracted from an image/mask pair."""


# ---------------------------------------------------------------------------
# Input validation (same checks as the notebook extractors)
# ---------------------------------------------------------------------------


def validate_inputs(rgb_image: np.ndarray, binary_mask: np.ndarray) -> None:
    if rgb_image.shape != EXPECTED_IMAGE_SHAPE:
        raise FeatureExtractionError(f"Invalid RGB image shape: {rgb_image.shape}")
    if rgb_image.dtype != np.uint8:
        raise FeatureExtractionError("RGB image must have uint8 data type.")
    validate_mask(binary_mask)


def validate_mask(binary_mask: np.ndarray) -> None:
    if binary_mask.shape != EXPECTED_MASK_SHAPE:
        raise FeatureExtractionError(f"Invalid binary mask shape: {binary_mask.shape}")
    if binary_mask.dtype != np.uint8:
        raise FeatureExtractionError("Binary mask must have uint8 data type.")
    if set(np.unique(binary_mask).tolist()) != {0, 255}:
        raise FeatureExtractionError("Binary mask must contain 0 and 255.")
    if int(np.count_nonzero(binary_mask == 255)) < MIN_FOREGROUND_PIXELS:
        raise FeatureExtractionError("Insufficient foreground pixels.")


def _finite(vector: np.ndarray, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=FEATURE_DTYPE)
    if vector.shape != (dim,):
        raise FeatureExtractionError(f"{name}: invalid dimension {vector.shape}")
    if not np.isfinite(vector).all():
        raise FeatureExtractionError(f"{name}: NaN or infinity detected.")
    return vector


# ---------------------------------------------------------------------------
# HSV histogram (Cell 6)
# ---------------------------------------------------------------------------


def extract_hsv_features(rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    validate_inputs(rgb_image, binary_mask)
    foreground_pixels = int((binary_mask == 255).sum())
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    parts = []
    for channel_index, (bins, channel_range) in enumerate(zip(HSV_BINS, HSV_RANGES)):
        histogram = cv2.calcHist(
            [hsv_image], [channel_index], binary_mask, [bins], list(channel_range)
        )
        histogram = histogram.flatten().astype(FEATURE_DTYPE)
        histogram_sum = float(histogram.sum())
        if histogram_sum <= 0:
            raise FeatureExtractionError(f"Empty histogram in channel {channel_index}")
        if not np.isclose(histogram_sum, foreground_pixels, rtol=0, atol=1e-6):
            raise FeatureExtractionError(f"Histogram count mismatch in channel {channel_index}")
        parts.append(histogram / histogram_sum)

    return _finite(np.concatenate(parts).astype(FEATURE_DTYPE), HSV_FEATURE_DIM, "HSV")


# ---------------------------------------------------------------------------
# GLCM (Cell 8)
# ---------------------------------------------------------------------------


def quantize_grayscale(gray_image: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    if gray_image.ndim != 2 or gray_image.dtype != np.uint8:
        raise FeatureExtractionError("Grayscale image must be 2-D uint8.")
    if not 2 <= levels <= 256:
        raise FeatureExtractionError("Gray levels must be between 2 and 256.")
    return (gray_image.astype(np.uint16) * levels // 256).astype(np.uint8)


def masked_glcm_counts(
    quantized_image: np.ndarray,
    foreground_mask: np.ndarray,
    row_offset: int,
    col_offset: int,
    levels: int = GLCM_LEVELS,
) -> tuple[np.ndarray, int]:
    """Count gray-level pairs where both pixels are foreground."""
    if quantized_image.shape != foreground_mask.shape:
        raise FeatureExtractionError("Image and mask dimensions do not match.")
    if row_offset == 0 and col_offset == 0:
        raise FeatureExtractionError("Zero spatial offset is not allowed.")
    foreground_mask = foreground_mask.astype(bool)
    height, width = quantized_image.shape

    row_start, row_end = max(0, -row_offset), min(height, height - row_offset)
    col_start, col_end = max(0, -col_offset), min(width, width - col_offset)
    if row_start >= row_end or col_start >= col_end:
        raise FeatureExtractionError("Spatial offset exceeds image dimensions.")

    ref = (slice(row_start, row_end), slice(col_start, col_end))
    nbr = (
        slice(row_start + row_offset, row_end + row_offset),
        slice(col_start + col_offset, col_end + col_offset),
    )
    valid_pairs = foreground_mask[ref] & foreground_mask[nbr]
    valid_pair_count = int(valid_pairs.sum())

    reference_values = quantized_image[ref][valid_pairs].astype(np.int64)
    neighbor_values = quantized_image[nbr][valid_pairs].astype(np.int64)
    counts = np.bincount(
        reference_values * levels + neighbor_values, minlength=levels * levels
    ).reshape(levels, levels)
    return counts, valid_pair_count


def build_masked_glcm(quantized_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """GLCM tensor of shape (levels, levels, distances, angles)."""
    foreground_mask = binary_mask == 255
    tensor = np.zeros(
        (GLCM_LEVELS, GLCM_LEVELS, len(GLCM_DISTANCES), len(GLCM_ANGLES_DEGREES)),
        dtype=FEATURE_DTYPE,
    )
    for d_idx, distance in enumerate(GLCM_DISTANCES):
        for a_idx, angle in enumerate(GLCM_ANGLES_DEGREES):
            row_offset, col_offset = GLCM_OFFSET_LOOKUP[(distance, angle)]
            counts, n_pairs = masked_glcm_counts(
                quantized_image, foreground_mask, row_offset, col_offset, GLCM_LEVELS
            )
            if n_pairs == 0:
                raise FeatureExtractionError(
                    f"No valid foreground pairs for distance={distance}, angle={angle}."
                )
            if GLCM_SYMMETRIC:
                counts = counts + counts.T
            matrix = counts.astype(FEATURE_DTYPE)
            if GLCM_NORMED:
                total = matrix.sum()
                if total <= 0:
                    raise FeatureExtractionError("Empty GLCM matrix.")
                matrix = matrix / total
            tensor[:, :, d_idx, a_idx] = matrix
    return tensor


def extract_glcm_features(rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    validate_inputs(rgb_image, binary_mask)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    quantized = quantize_grayscale(gray_image, GLCM_LEVELS)
    tensor = build_masked_glcm(quantized, binary_mask)
    parts = [
        graycoprops(tensor, prop).astype(FEATURE_DTYPE).ravel(order="C")  # distance -> angle
        for prop in GLCM_PROPERTIES
    ]
    return _finite(np.concatenate(parts).astype(FEATURE_DTYPE), GLCM_FEATURE_DIM, "GLCM")


# ---------------------------------------------------------------------------
# Hu Moments (Cell 10)
# ---------------------------------------------------------------------------


def transform_hu_moments(raw_hu: np.ndarray) -> np.ndarray:
    """-sign(hu) * log10(max(|hu|, 1e-30)); exact zeros are preserved."""
    raw_hu = _finite(np.asarray(raw_hu, dtype=FEATURE_DTYPE).reshape(-1), HU_FEATURE_DIM, "Raw Hu")
    out = np.zeros_like(raw_hu, dtype=FEATURE_DTYPE)
    nz = raw_hu != 0
    out[nz] = -np.sign(raw_hu[nz]) * np.log10(np.maximum(np.abs(raw_hu[nz]), HU_LOG_EPSILON))
    return _finite(out, HU_FEATURE_DIM, "Hu Moments")


def extract_raw_hu_moments(binary_mask: np.ndarray) -> np.ndarray:
    """The seven untransformed Hu Moments of the binary mask."""
    validate_mask(binary_mask)
    moments = cv2.moments(binary_mask, binaryImage=True)
    if float(moments["m00"]) <= 0:
        raise FeatureExtractionError("Zero foreground area detected.")
    raw = cv2.HuMoments(moments).flatten().astype(FEATURE_DTYPE)
    return _finite(raw, HU_FEATURE_DIM, "Raw Hu")


def extract_hu_features(binary_mask: np.ndarray) -> np.ndarray:
    return transform_hu_moments(extract_raw_hu_moments(binary_mask))


# ---------------------------------------------------------------------------
# Combined raw features (Cells 12 and 15)
# ---------------------------------------------------------------------------


def extract_visual_feature_groups(rgb_image: np.ndarray, binary_mask: np.ndarray) -> dict:
    """Return {'HSV': (64,), 'GLCM': (72,), 'Hu Moments': (7,)} raw vectors."""
    return {
        "HSV": extract_hsv_features(rgb_image, binary_mask),
        "GLCM": extract_glcm_features(rgb_image, binary_mask),
        "Hu Moments": extract_hu_features(binary_mask),
    }


def concatenate_visual_features(groups: dict) -> np.ndarray:
    vector = np.concatenate([np.asarray(groups[g], dtype=FEATURE_DTYPE) for g in FEATURE_GROUP_ORDER])
    return _finite(vector, TOTAL_VISUAL_FEATURE_DIM, "Combined visual")


def extract_visual_features(rgb_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Raw 143-d visual vector (HSV | GLCM | Hu), before any normalisation."""
    return concatenate_visual_features(extract_visual_feature_groups(rgb_image, binary_mask))


def extract_features_from_preprocessed(result: PreprocessedImage) -> np.ndarray:
    """Raw 143-d vector for the output of :func:`core.preprocessing.preprocess_image`.

    Uses ``result.rgb`` (unmasked 256x256 RGB) and ``result.mask`` (0/255),
    i.e. exactly the processed image/mask pair Notebook 03 read from disk.
    """
    if not result.technical_valid:
        raise FeatureExtractionError(
            "Preprocessing output is not technically valid "
            f"({result.info.get('image_quality_issues')}); features are not extracted."
        )
    return extract_visual_features(result.rgb, result.mask)


# ---------------------------------------------------------------------------
# Normalisation with an already fitted normaliser (no fitting here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisualNormalizer:
    """Parameters of a fitted group-wise normaliser (Notebook 06 format)."""

    params: dict[str, Any]
    source: str

    @property
    def n_reference_samples(self) -> int:
        return int(self.params["n_reference_samples"])


def load_gallery_normalizer(path: str | Path | None = None) -> VisualNormalizer:
    """Load the gallery-only normaliser saved by Notebook 06 and check its layout."""
    path = Path(path) if path is not None else DEFAULT_NORMALIZER_PATH
    params = joblib.load(path)
    if not isinstance(params, dict) or "groups" not in params:
        raise ValueError(f"Unexpected normaliser format in {path}")
    if tuple(params["groups"]) != FEATURE_GROUP_ORDER:
        raise ValueError(f"Unexpected normaliser groups: {tuple(params['groups'])}")
    for group, (start, stop) in FEATURE_GROUP_SLICES.items():
        g = params["groups"][group]
        if (int(g["start"]), int(g["stop"])) != (start, stop):
            raise ValueError(f"Normaliser slice mismatch for {group}")
        for key in ("mean", "scale", "constant_mask"):
            if np.asarray(g[key]).shape != (stop - start,):
                raise ValueError(f"Normaliser {group}.{key} has wrong shape")
        if not np.all(np.asarray(g["scale"]) > 0):
            raise ValueError(f"Normaliser {group}.scale must be positive")
    return VisualNormalizer(params=params, source=str(path))


def transform_visual_features(features: np.ndarray, normalizer: VisualNormalizer) -> np.ndarray:
    """Apply a fitted normaliser (Notebook 06 ``eval06_transform_visual``).

    For each group: z = (x - mean) / scale; constant features -> 0; z * weight.
    Accepts a (143,) vector or an (n, 143) matrix and returns the same shape.
    """
    x = np.asarray(features, dtype=FEATURE_DTYPE)
    single = x.ndim == 1
    matrix = x.reshape(1, -1) if single else x
    if matrix.ndim != 2 or matrix.shape[1] != TOTAL_VISUAL_FEATURE_DIM:
        raise FeatureExtractionError(f"Expected (*, 143) features, got {x.shape}")
    out = np.empty_like(matrix)
    for p in normalizer.params["groups"].values():
        start, stop = int(p["start"]), int(p["stop"])
        z = (matrix[:, start:stop] - p["mean"]) / p["scale"]
        z[:, np.asarray(p["constant_mask"], dtype=bool)] = 0.0
        out[:, start:stop] = z * p["weight"]
    if not np.isfinite(out).all():
        raise FeatureExtractionError("Non-finite normalised visual features.")
    return out[0] if single else out


__all__ = [
    "FeatureExtractionError",
    "VisualNormalizer",
    "VISUAL_FEATURE_NAMES",
    "FEATURE_GROUP_ORDER",
    "FEATURE_GROUP_SLICES",
    "TOTAL_VISUAL_FEATURE_DIM",
    "validate_inputs",
    "extract_hsv_features",
    "quantize_grayscale",
    "masked_glcm_counts",
    "build_masked_glcm",
    "extract_glcm_features",
    "transform_hu_moments",
    "extract_raw_hu_moments",
    "extract_hu_features",
    "extract_visual_feature_groups",
    "concatenate_visual_features",
    "extract_visual_features",
    "extract_features_from_preprocessed",
    "load_gallery_normalizer",
    "transform_visual_features",
]
