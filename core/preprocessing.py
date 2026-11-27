"""Image preprocessing for the Multimodal Leaf Retrieval application.

This module re-implements, without changing behaviour, the final image
preprocessing pipeline of ``notebooks/02_preprocessing.ipynb`` that produced
``dataset/processed/images`` and ``dataset/processed/masks``:

    load (EXIF transpose, RGB)
      -> resize with aspect ratio + white padding (256 x 256)
      -> refined adaptive segmentation (HSV / Otsu dark / Otsu light / GrabCut)
      -> ROI cropping on the original resolution + re-segmentation
      -> controlled internal hole filling
      -> final mask quality + image quality inspection

Source cells in Notebook 02:
    Cell 6   load_image, resize_with_padding
    Cell 7   create_valid_region, segment_hsv, segment_otsu, segment_grabcut,
             clean_binary_mask, calculate_mask_quality, segment_leaf
    Cell 8   calculate_image_quality
    Cell 10  calculate_refined_mask_quality, segment_leaf_refined
    Cell 11  get_foreground_bbox, map_bbox_to_original, project_mask_to_roi,
             calculate_mask_iou, apply_leaf_mask, preprocess_leaf_image
    Cell 12  refine_internal_mask_holes, preprocess_leaf_image_final

Configurable parameters are read from
``dataset/processed/preprocessing_config_final.json``. Values that were
hard-coded inside the notebook functions (and therefore are not part of that
JSON) are kept as module constants below with the same values.

Only two implementation details differ from the notebook text, and both
produce identical outputs (verified by ``tests/test_preprocessing.py``):

1. ``skimage.morphology.remove_small_holes(area_threshold=32)`` is deprecated
   in scikit-image 0.26. In that version the deprecated value is passed on
   unchanged as ``max_size`` (holes with area <= 32 are filled), so
   ``max_size=32`` is used here. ``max_size=31`` would NOT be equivalent.
2. ``segment_leaf_refined`` in the notebook called ``segment_leaf`` again,
   recomputing the same four candidate masks (GrabCut is re-seeded before
   every call, so the result is deterministic). Here the candidates are
   computed once and the original ``segment_leaf`` selection rule is applied
   to them, which halves the run time without changing the result.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import morphology

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "dataset" / "processed" / "preprocessing_config_final.json"
)

# ---------------------------------------------------------------------------
# Constants hard-coded in Notebook 02 (not stored in the JSON configuration)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42  # Cell 1, also cv2.setRNGSeed before every GrabCut call

# Cell 7 - segment_hsv: OpenCV HSV ranges (H in [0, 179])
HSV_GREEN_RANGE = ((20, 30, 25), (100, 255, 255))
HSV_RED_LOW_RANGE = ((0, 35, 25), (20, 255, 255))
HSV_RED_HIGH_RANGE = ((125, 30, 20), (179, 255, 255))

# Cell 7 - segment_grabcut
GRABCUT_INNER_MARGIN_RATIO = 0.03
GRABCUT_ITERATIONS = 5

# Cell 7 - clean_binary_mask: remove_small_holes(area_threshold=32).
# In scikit-image 0.26 (the version Notebook 02 ran with) the deprecated
# `area_threshold` value is forwarded unchanged as `max_size`, so holes with
# area <= 32 px were filled. max_size=32 reproduces that exactly.
SMALL_HOLE_MAX_SIZE = 32

# Cell 7 - calculate_mask_quality
BORDER_ERODE_KERNEL_SIZE = 5
INITIAL_MAX_VALID_REGION_RATIO = 0.95
INITIAL_MAX_BORDER_RATIO = 0.35
INITIAL_SIZE_PENALTY = 0.8
INITIAL_FULL_IMAGE_PENALTY = 0.8
INITIAL_BORDER_PENALTY_WEIGHT = 0.7
INITIAL_AREA_PENALTY_WEIGHT = 0.15
INITIAL_PREFERRED_VALID_RATIO = 0.45

# Cell 10 - calculate_refined_mask_quality
REFINED_INSUFFICIENT_PIXELS_PENALTY = 2.0
REFINED_REGION_PENALTY = 1.0

# Cell 8 - calculate_image_quality
BLUR_THRESHOLD = 100.0
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220
MIN_CONTRAST = 15.0
QUALITY_MIN_MASK_PIXELS = 100

# Candidate order matters for tie-breaking in max(); it follows the notebook.
CANDIDATE_METHODS = (
    "HSV Thresholding",
    "Otsu Dark Foreground",
    "Otsu Light Foreground",
    "GrabCut",
)

PASSED = "Passed Initial Check"
NEEDS_REVIEW = "Needs Review"
FAILED = "Failed"

_INTERPOLATIONS = {
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_NEAREST": cv2.INTER_NEAREST,
}
_MORPH_SHAPES = {
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "RECT": cv2.MORPH_RECT,
    "CROSS": cv2.MORPH_CROSS,
}

SUPPORTED_SEGMENTATION_METHOD = "Refined Adaptive Selection"


class PreprocessingError(ValueError):
    """Raised when an input image cannot be preprocessed."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessingConfig:
    """Typed view of ``preprocessing_config_final.json``."""

    target_width: int
    target_height: int
    padding_color: tuple[int, int, int]
    downscale_interpolation: int
    upscale_interpolation: int
    apply_denoising: bool
    median_kernel_size: int
    morph_kernel: np.ndarray = field(repr=False, compare=False)
    opening_iterations: int
    closing_iterations: int
    keep_largest_component: bool
    min_foreground_ratio: float
    max_foreground_ratio: float
    refined_min_mask_pixels: int
    refined_max_border_ratio: float
    refined_max_valid_region_ratio: float
    small_foreground_review_ratio: float
    border_penalty_weight: float
    area_penalty_weight: float
    preferred_foreground_ratio: float
    preserve_original_valid_selection: bool
    apply_roi_cropping: bool
    roi_margin_ratio: float
    min_roi_width: int
    min_roi_height: int
    roi_min_mask_pixels: int
    roi_min_mask_overlap: float
    roi_mask_interpolation: int
    apply_controlled_hole_filling: bool
    max_internal_hole_pixels: int
    max_internal_hole_ratio: float
    max_total_fill_ratio: float
    mask_foreground_value: int
    mask_background_value: int
    raw: dict = field(repr=False, compare=False)

    @property
    def target_size(self) -> tuple[int, int]:
        """(width, height), as used by the notebook's TARGET_SIZE."""
        return (self.target_width, self.target_height)

    @classmethod
    def from_dict(cls, cfg: dict) -> "PreprocessingConfig":
        def interp(name: str) -> int:
            if name not in _INTERPOLATIONS:
                raise ValueError(f"Unsupported interpolation: {name}")
            return _INTERPOLATIONS[name]

        # Reject settings whose behaviour was never implemented in Notebook 02.
        if cfg.get("segmentation_method") != SUPPORTED_SEGMENTATION_METHOD:
            raise ValueError(
                "Only 'Refined Adaptive Selection' segmentation is supported, "
                f"got {cfg.get('segmentation_method')!r}."
            )
        if not cfg.get("apply_segmentation", True):
            raise ValueError("apply_segmentation=False is not supported.")
        if not cfg.get("preserve_aspect_ratio", True):
            raise ValueError("preserve_aspect_ratio=False is not supported.")
        if cfg.get("roi_source", "Original Resolution") != "Original Resolution":
            raise ValueError("Only roi_source='Original Resolution' is supported.")
        if not cfg.get("roi_resegmentation", True):
            raise ValueError("roi_resegmentation=False is not supported.")
        if cfg.get("mask_foreground_value", 255) != 255 or cfg.get(
            "mask_background_value", 0
        ) != 0:
            raise ValueError("Mask values must be 255 (foreground) / 0 (background).")

        shape_name = cfg["morph_kernel_shape"]
        if shape_name not in _MORPH_SHAPES:
            raise ValueError(f"Unsupported morph_kernel_shape: {shape_name}")
        ksize = int(cfg["morph_kernel_size"])
        kernel = cv2.getStructuringElement(_MORPH_SHAPES[shape_name], (ksize, ksize))

        return cls(
            target_width=int(cfg["target_width"]),
            target_height=int(cfg["target_height"]),
            padding_color=tuple(int(v) for v in cfg["padding_color"]),
            downscale_interpolation=interp(cfg["downscale_interpolation"]),
            upscale_interpolation=interp(cfg["upscale_interpolation"]),
            apply_denoising=bool(cfg["apply_denoising"]),
            median_kernel_size=int(cfg["median_kernel_size"]),
            morph_kernel=kernel,
            opening_iterations=int(cfg["opening_iterations"]),
            closing_iterations=int(cfg["closing_iterations"]),
            keep_largest_component=bool(cfg["keep_largest_component"]),
            min_foreground_ratio=float(cfg["min_foreground_ratio"]),
            max_foreground_ratio=float(cfg["max_foreground_ratio"]),
            refined_min_mask_pixels=int(cfg["refined_min_mask_pixels"]),
            refined_max_border_ratio=float(cfg["refined_max_border_ratio"]),
            refined_max_valid_region_ratio=float(cfg["refined_max_valid_region_ratio"]),
            small_foreground_review_ratio=float(cfg["small_foreground_review_ratio"]),
            border_penalty_weight=float(cfg["border_penalty_weight"]),
            area_penalty_weight=float(cfg["area_penalty_weight"]),
            preferred_foreground_ratio=float(cfg["preferred_foreground_ratio"]),
            preserve_original_valid_selection=bool(
                cfg.get("preserve_original_valid_selection", True)
            ),
            apply_roi_cropping=bool(cfg["apply_roi_cropping"]),
            roi_margin_ratio=float(cfg["roi_margin_ratio"]),
            min_roi_width=int(cfg["min_roi_width"]),
            min_roi_height=int(cfg["min_roi_height"]),
            roi_min_mask_pixels=int(cfg["roi_min_mask_pixels"]),
            roi_min_mask_overlap=float(cfg["roi_min_mask_overlap"]),
            roi_mask_interpolation=interp(cfg.get("roi_mask_interpolation", "INTER_NEAREST")),
            apply_controlled_hole_filling=bool(cfg["apply_controlled_hole_filling"]),
            max_internal_hole_pixels=int(cfg["max_internal_hole_pixels"]),
            max_internal_hole_ratio=float(cfg["max_internal_hole_ratio"]),
            max_total_fill_ratio=float(cfg["max_total_fill_ratio"]),
            mask_foreground_value=int(cfg.get("mask_foreground_value", 255)),
            mask_background_value=int(cfg.get("mask_background_value", 0)),
            raw=dict(cfg),
        )


_CONFIG_CACHE: dict[Path, PreprocessingConfig] = {}


def load_config(path: str | Path | None = None) -> PreprocessingConfig:
    """Load (and cache) the final preprocessing configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if config_path not in _CONFIG_CACHE:
        with open(config_path, "r", encoding="utf-8") as fh:
            _CONFIG_CACHE[config_path] = PreprocessingConfig.from_dict(json.load(fh))
    return _CONFIG_CACHE[config_path]


def _cfg(config: PreprocessingConfig | None) -> PreprocessingConfig:
    return config if config is not None else load_config()


# ---------------------------------------------------------------------------
# Loading and resizing (Cell 6)
# ---------------------------------------------------------------------------


def load_image(source: str | Path | bytes | BinaryIO) -> np.ndarray:
    """Load an image as an RGB uint8 array, applying EXIF orientation.

    ``source`` may be a path, raw bytes, or a binary file object (for example
    the ``stream`` of a Flask ``FileStorage``).
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image = np.array(img)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PreprocessingError(f"Cannot read image: {exc}") from exc
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise PreprocessingError("Image must be a non-empty RGB image.")
    return image


def resize_with_padding(
    image: np.ndarray,
    target_size: tuple[int, int] | None = None,
    padding_color: tuple[int, int, int] | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Resize preserving aspect ratio, then pad to ``target_size`` (w, h)."""
    cfg = _cfg(config)
    target_width, target_height = target_size or cfg.target_size
    padding_color = padding_color if padding_color is not None else cfg.padding_color

    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))
    interpolation = (
        cfg.downscale_interpolation if scale < 1 else cfg.upscale_interpolation
    )
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )
    resize_info = {
        "original_width": original_width,
        "original_height": original_height,
        "scale": scale,
        "resized_width": new_width,
        "resized_height": new_height,
        "output_width": padded_image.shape[1],
        "output_height": padded_image.shape[0],
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }
    return padded_image, resize_info


# ---------------------------------------------------------------------------
# Segmentation candidates (Cell 7)
# ---------------------------------------------------------------------------


def create_valid_region(image: np.ndarray, resize_info: dict | None = None) -> np.ndarray:
    """255 inside the real image area, 0 on artificial padding."""
    height, width = image.shape[:2]
    valid_region = np.full((height, width), 255, dtype=np.uint8)
    if resize_info is not None:
        valid_region[:] = 0
        x1 = resize_info["pad_left"]
        y1 = resize_info["pad_top"]
        x2 = x1 + resize_info["resized_width"]
        y2 = y1 + resize_info["resized_height"]
        valid_region[y1:y2, x1:x2] = 255
    return valid_region


def segment_hsv(image: np.ndarray, valid_region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array(HSV_GREEN_RANGE[0]), np.array(HSV_GREEN_RANGE[1]))
    red_mask_1 = cv2.inRange(hsv, np.array(HSV_RED_LOW_RANGE[0]), np.array(HSV_RED_LOW_RANGE[1]))
    red_mask_2 = cv2.inRange(hsv, np.array(HSV_RED_HIGH_RANGE[0]), np.array(HSV_RED_HIGH_RANGE[1]))
    mask = cv2.bitwise_or(green_mask, red_mask_1)
    mask = cv2.bitwise_or(mask, red_mask_2)
    return cv2.bitwise_and(mask, valid_region)


def segment_otsu(image: np.ndarray, valid_region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dark_foreground_mask, light_foreground_mask).

    The Otsu threshold is computed from non-padding pixels only.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    valid_pixels = gray[valid_region > 0]
    if valid_pixels.size == 0:
        # The notebook returned a single array here, which would have broken
        # tuple unpacking; this branch is unreachable for real images (the
        # valid region is never empty). Two empty masks keep the contract.
        empty = np.zeros_like(gray)
        return empty, empty.copy()
    threshold, _ = cv2.threshold(
        valid_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    dark_mask = cv2.bitwise_and(dark_mask, valid_region)
    light_mask = cv2.bitwise_and(light_mask, valid_region)
    return dark_mask, light_mask


def segment_grabcut(image: np.ndarray, valid_region: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)

    ys, xs = np.where(valid_region > 0)
    if len(xs) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    margin_x = max(1, int((x2 - x1) * GRABCUT_INNER_MARGIN_RATIO))
    margin_y = max(1, int((y2 - y1) * GRABCUT_INNER_MARGIN_RATIO))

    grabcut_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    grabcut_mask[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.setRNGSeed(RANDOM_STATE)
        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            None,
            bg_model,
            fg_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_MASK,
        )
        mask = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return cv2.bitwise_and(mask, valid_region)
    except cv2.error:
        return np.zeros((height, width), dtype=np.uint8)


def clean_binary_mask(
    mask: np.ndarray, valid_region: np.ndarray, config: PreprocessingConfig | None = None
) -> np.ndarray:
    """Opening, closing, largest component, small-hole filling, padding removal."""
    cfg = _cfg(config)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cfg.morph_kernel, iterations=cfg.opening_iterations
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cfg.morph_kernel, iterations=cfg.closing_iterations
    )
    mask = cv2.bitwise_and(mask, valid_region)

    if cfg.keep_largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    mask = (
        morphology.remove_small_holes(mask.astype(bool), max_size=SMALL_HOLE_MAX_SIZE).astype(
            np.uint8
        )
        * 255
    )
    return cv2.bitwise_and(mask, valid_region)


def calculate_mask_quality(
    mask: np.ndarray, valid_region: np.ndarray, config: PreprocessingConfig | None = None
) -> dict:
    """Initial heuristic mask quality (Cell 7)."""
    cfg = _cfg(config)
    foreground_pixels = np.count_nonzero(mask)
    valid_pixels = np.count_nonzero(valid_region)
    if foreground_pixels == 0 or valid_pixels == 0:
        return {
            "foreground_ratio": 0.0,
            "valid_region_ratio": 0.0,
            "border_ratio": 0.0,
            "quality_score": -1.0,
            "quality_status": FAILED,
        }

    foreground_ratio = foreground_pixels / mask.size
    valid_region_ratio = foreground_pixels / valid_pixels

    eroded_region = cv2.erode(
        valid_region,
        np.ones((BORDER_ERODE_KERNEL_SIZE, BORDER_ERODE_KERNEL_SIZE), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    border_region = cv2.subtract(valid_region, eroded_region)
    border_foreground = np.count_nonzero(cv2.bitwise_and(mask, border_region))
    border_pixels = np.count_nonzero(border_region)
    border_ratio = border_foreground / max(border_pixels, 1)

    quality_score = 1.0
    if foreground_ratio < cfg.min_foreground_ratio:
        quality_score -= INITIAL_SIZE_PENALTY
    if foreground_ratio > cfg.max_foreground_ratio:
        quality_score -= INITIAL_SIZE_PENALTY
    if valid_region_ratio > INITIAL_MAX_VALID_REGION_RATIO:
        quality_score -= INITIAL_FULL_IMAGE_PENALTY
    quality_score -= INITIAL_BORDER_PENALTY_WEIGHT * border_ratio
    quality_score -= INITIAL_AREA_PENALTY_WEIGHT * abs(
        valid_region_ratio - INITIAL_PREFERRED_VALID_RATIO
    )

    if (
        cfg.min_foreground_ratio <= foreground_ratio <= cfg.max_foreground_ratio
        and valid_region_ratio <= INITIAL_MAX_VALID_REGION_RATIO
        and border_ratio <= INITIAL_MAX_BORDER_RATIO
    ):
        quality_status = PASSED
    else:
        quality_status = NEEDS_REVIEW

    return {
        "foreground_ratio": float(foreground_ratio),
        "valid_region_ratio": float(valid_region_ratio),
        "border_ratio": float(border_ratio),
        "quality_score": float(quality_score),
        "quality_status": quality_status,
    }


def _generate_candidate_masks(image: np.ndarray, valid_region: np.ndarray) -> dict:
    otsu_dark, otsu_light = segment_otsu(image, valid_region)
    candidates = {
        "HSV Thresholding": segment_hsv(image, valid_region),
        "Otsu Dark Foreground": otsu_dark,
        "Otsu Light Foreground": otsu_light,
        "GrabCut": segment_grabcut(image, valid_region),
    }
    assert tuple(candidates) == CANDIDATE_METHODS
    return candidates


def _select_initial_method(initial_results: dict) -> str:
    """Selection rule of the original ``segment_leaf`` (Cell 7)."""
    passed = [m for m, r in initial_results.items() if r["quality_status"] == PASSED]
    available = passed if passed else list(initial_results.keys())
    return max(available, key=lambda m: initial_results[m]["quality_score"])


def segment_leaf(
    image: np.ndarray, resize_info: dict | None = None, config: PreprocessingConfig | None = None
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Initial adaptive segmentation (Cell 7). Kept for completeness."""
    cfg = _cfg(config)
    valid_region = create_valid_region(image, resize_info)
    results = {}
    for method, mask in _generate_candidate_masks(image, valid_region).items():
        cleaned = clean_binary_mask(mask, valid_region, cfg)
        results[method] = {"mask": cleaned, **calculate_mask_quality(cleaned, valid_region, cfg)}
    best_method = _select_initial_method(results)
    best = results[best_method]
    info = {
        "selected_method": best_method,
        "foreground_ratio": best["foreground_ratio"],
        "valid_region_ratio": best["valid_region_ratio"],
        "border_ratio": best["border_ratio"],
        "quality_score": best["quality_score"],
        "quality_status": best["quality_status"],
    }
    return apply_leaf_mask(image, best["mask"]), best["mask"], info, results


# ---------------------------------------------------------------------------
# Refined segmentation (Cell 10)
# ---------------------------------------------------------------------------


def calculate_refined_mask_quality(
    mask: np.ndarray, valid_region: np.ndarray, config: PreprocessingConfig | None = None
) -> dict:
    cfg = _cfg(config)
    original_quality = calculate_mask_quality(mask, valid_region, cfg)
    foreground_pixels = int(np.count_nonzero(mask))
    foreground_ratio = original_quality["foreground_ratio"]
    valid_region_ratio = original_quality["valid_region_ratio"]
    border_ratio = original_quality["border_ratio"]

    sufficient_pixels = foreground_pixels >= cfg.refined_min_mask_pixels
    acceptable_border = border_ratio <= cfg.refined_max_border_ratio
    acceptable_region = valid_region_ratio <= cfg.refined_max_valid_region_ratio
    eligible_mask = sufficient_pixels and acceptable_border and acceptable_region

    refined_score = 1.0
    refined_score -= cfg.border_penalty_weight * border_ratio
    refined_score -= cfg.area_penalty_weight * abs(
        valid_region_ratio - cfg.preferred_foreground_ratio
    )
    if not sufficient_pixels:
        refined_score -= REFINED_INSUFFICIENT_PIXELS_PENALTY
    if not acceptable_region:
        refined_score -= REFINED_REGION_PENALTY

    if not sufficient_pixels:
        quality_status = FAILED
    elif not eligible_mask:
        quality_status = NEEDS_REVIEW
    elif foreground_ratio < cfg.small_foreground_review_ratio:
        quality_status = NEEDS_REVIEW
    else:
        quality_status = PASSED

    return {
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": float(foreground_ratio),
        "valid_region_ratio": float(valid_region_ratio),
        "border_ratio": float(border_ratio),
        "quality_score": float(refined_score),
        "eligible_mask": bool(eligible_mask),
        "quality_status": quality_status,
    }


def segment_leaf_refined(
    image: np.ndarray, resize_info: dict | None = None, config: PreprocessingConfig | None = None
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Refined adaptive segmentation (Cell 10).

    Returns ``(segmented_image, mask, segmentation_info, refined_results)``.
    """
    cfg = _cfg(config)
    valid_region = create_valid_region(image, resize_info)
    candidates = _generate_candidate_masks(image, valid_region)

    refined_results: dict[str, dict] = {}
    initial_results: dict[str, dict] = {}
    for method, raw_mask in candidates.items():
        cleaned = clean_binary_mask(raw_mask, valid_region, cfg)
        refined_results[method] = {
            "mask": cleaned,
            **calculate_refined_mask_quality(cleaned, valid_region, cfg),
        }
        # Same cleaned mask the notebook's segment_leaf() would have produced.
        initial_results[method] = calculate_mask_quality(cleaned, valid_region, cfg)

    eligible_methods = [m for m, r in refined_results.items() if r["eligible_mask"]]

    original_method = _select_initial_method(initial_results)
    original_status = initial_results[original_method]["quality_status"]
    preserve_original = (
        cfg.preserve_original_valid_selection
        and original_status == PASSED
        and refined_results[original_method]["eligible_mask"]
    )

    if preserve_original:
        best_method = original_method
        selection_reason = "Preserved original valid selection"
    elif eligible_methods:
        best_method = max(eligible_methods, key=lambda m: refined_results[m]["quality_score"])
        selection_reason = "Selected eligible refined mask"
    else:
        best_method = max(refined_results, key=lambda m: refined_results[m]["quality_score"])
        selection_reason = "Fallback - no eligible mask"

    best = refined_results[best_method]
    best_mask = best["mask"]
    info = {
        "selected_method": best_method,
        "foreground_pixels": best["foreground_pixels"],
        "foreground_ratio": best["foreground_ratio"],
        "valid_region_ratio": best["valid_region_ratio"],
        "border_ratio": best["border_ratio"],
        "quality_score": best["quality_score"],
        "eligible_mask": best["eligible_mask"],
        "quality_status": best["quality_status"],
        "selection_reason": selection_reason,
        "original_method": original_method,
        "method_changed": best_method != original_method,
    }
    if not eligible_methods:
        info["quality_status"] = (
            NEEDS_REVIEW if best["foreground_pixels"] >= cfg.refined_min_mask_pixels else FAILED
        )
    return apply_leaf_mask(image, best_mask), best_mask, info, refined_results


# ---------------------------------------------------------------------------
# ROI cropping (Cell 11)
# ---------------------------------------------------------------------------


def get_foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return int(x), int(y), int(width), int(height)


def map_bbox_to_original(
    bbox: tuple[int, int, int, int],
    resize_info: dict,
    original_image_shape: tuple[int, ...],
    margin_ratio: float | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[int, int, int, int]:
    """Map a bbox on the padded 256x256 image to original-resolution (x1, y1, x2, y2)."""
    cfg = _cfg(config)
    margin_ratio = cfg.roi_margin_ratio if margin_ratio is None else margin_ratio
    x, y, width, height = bbox
    original_height, original_width = original_image_shape[:2]
    pad_left = resize_info["pad_left"]
    pad_top = resize_info["pad_top"]
    scale_x = resize_info["resized_width"] / original_width
    scale_y = resize_info["resized_height"] / original_height

    x1 = int(np.floor((x - pad_left) / scale_x))
    y1 = int(np.floor((y - pad_top) / scale_y))
    x2 = int(np.ceil((x + width - pad_left) / scale_x))
    y2 = int(np.ceil((y + height - pad_top) / scale_y))

    margin_x = int(np.ceil((x2 - x1) * margin_ratio))
    margin_y = int(np.ceil((y2 - y1) * margin_ratio))

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(original_width, x2 + margin_x)
    y2 = min(original_height, y2 + margin_y)
    return x1, y1, x2, y2


def project_mask_to_roi(
    initial_mask: np.ndarray,
    resize_info: dict,
    original_shape: tuple[int, ...],
    roi_bbox: tuple[int, int, int, int],
    roi_resize_info: dict,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Project the initial mask into the padded ROI frame (for overlap checks)."""
    cfg = _cfg(config)
    original_height, original_width = original_shape[:2]
    left, top = resize_info["pad_left"], resize_info["pad_top"]
    right = left + resize_info["resized_width"]
    bottom = top + resize_info["resized_height"]

    mask_without_padding = initial_mask[top:bottom, left:right]
    original_resolution_mask = cv2.resize(
        mask_without_padding,
        (original_width, original_height),
        interpolation=cfg.roi_mask_interpolation,
    )
    x1, y1, x2, y2 = roi_bbox
    cropped_mask = original_resolution_mask[y1:y2, x1:x2]
    resized_mask = cv2.resize(
        cropped_mask,
        (roi_resize_info["resized_width"], roi_resize_info["resized_height"]),
        interpolation=cfg.roi_mask_interpolation,
    )
    projected = cv2.copyMakeBorder(
        resized_mask,
        roi_resize_info["pad_top"],
        roi_resize_info["pad_bottom"],
        roi_resize_info["pad_left"],
        roi_resize_info["pad_right"],
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return np.where(projected > 0, 255, 0).astype(np.uint8)


def calculate_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a, b = mask_a > 0, mask_b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def apply_leaf_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """White background outside the mask (preview / segmented image)."""
    segmented = np.full_like(image, 255)
    segmented[mask > 0] = image[mask > 0]
    return segmented


# ---------------------------------------------------------------------------
# Image quality (Cell 8)
# ---------------------------------------------------------------------------


def calculate_image_quality(image: np.ndarray, mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    foreground_pixels = gray[mask > 0]
    foreground_count = len(foreground_pixels)
    if foreground_count < QUALITY_MIN_MASK_PIXELS:
        return {
            "blur_score": math.nan,
            "mean_brightness": math.nan,
            "contrast": math.nan,
            "foreground_pixels": foreground_count,
            "is_blurry": False,
            "is_dark": False,
            "is_bright": False,
            "is_low_contrast": False,
            "quality_measurable": False,
        }

    inner_mask = cv2.erode(
        mask, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0
    )
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_pixels = laplacian[inner_mask > 0]
    blur_score = (
        float(laplacian_pixels.var())
        if laplacian_pixels.size >= QUALITY_MIN_MASK_PIXELS
        else math.nan
    )
    mean_brightness = float(foreground_pixels.mean())
    contrast = float(foreground_pixels.std())
    measurable = not math.isnan(blur_score)
    return {
        "blur_score": blur_score,
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "foreground_pixels": foreground_count,
        "is_blurry": bool(measurable and blur_score < BLUR_THRESHOLD),
        "is_dark": bool(mean_brightness < MIN_BRIGHTNESS),
        "is_bright": bool(mean_brightness > MAX_BRIGHTNESS),
        "is_low_contrast": bool(contrast < MIN_CONTRAST),
        "quality_measurable": measurable,
    }


# ---------------------------------------------------------------------------
# Full pipeline (Cells 11 and 12)
# ---------------------------------------------------------------------------


def preprocess_leaf_image(
    original_image: np.ndarray,
    apply_roi: bool | None = None,
    config: PreprocessingConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Resize + refined segmentation + ROI cropping (Cell 11).

    Returns ``(final_rgb, final_mask, final_segmented, info, intermediates)``.
    """
    cfg = _cfg(config)
    apply_roi = cfg.apply_roi_cropping if apply_roi is None else apply_roi

    working_image = original_image.copy()
    if cfg.apply_denoising:
        working_image = cv2.medianBlur(working_image, cfg.median_kernel_size)

    initial_resized, resize_info = resize_with_padding(working_image, config=cfg)
    initial_segmented, initial_mask, initial_info, _ = segment_leaf_refined(
        initial_resized, resize_info, cfg
    )

    final_rgb = initial_resized.copy()
    final_mask = initial_mask.copy()
    final_segmented = initial_segmented.copy()
    final_info = initial_info.copy()
    final_resize_info = resize_info

    roi_image = None
    roi_bbox = None
    roi_resize_info = None
    roi_applied = False
    roi_mask_overlap = math.nan
    roi_status = "ROI Disabled"

    if apply_roi:
        roi_status = "ROI Not Applied"
        bbox = get_foreground_bbox(initial_mask)
        if not initial_info["eligible_mask"]:
            roi_status = "Initial Mask Ineligible"
        elif bbox is None:
            roi_status = "Empty Initial Mask"
        else:
            roi_bbox = map_bbox_to_original(bbox, resize_info, working_image.shape, config=cfg)
            x1, y1, x2, y2 = roi_bbox
            if (x2 - x1) < cfg.min_roi_width or (y2 - y1) < cfg.min_roi_height:
                roi_status = "ROI Too Small"
            else:
                roi_image = working_image[y1:y2, x1:x2].copy()
                roi_resized, roi_resize_info = resize_with_padding(roi_image, config=cfg)
                projected_mask = project_mask_to_roi(
                    initial_mask, resize_info, working_image.shape, roi_bbox, roi_resize_info, cfg
                )
                roi_segmented, roi_mask, roi_seg_info, _ = segment_leaf_refined(
                    roi_resized, roi_resize_info, cfg
                )
                roi_mask_overlap = calculate_mask_iou(projected_mask, roi_mask)
                valid_roi = (
                    roi_seg_info["eligible_mask"]
                    and np.count_nonzero(roi_mask) >= cfg.roi_min_mask_pixels
                    and roi_mask_overlap >= cfg.roi_min_mask_overlap
                )
                if valid_roi:
                    final_rgb = roi_resized
                    final_mask = roi_mask
                    final_segmented = roi_segmented
                    final_info = roi_seg_info
                    final_resize_info = roi_resize_info
                    roi_applied = True
                    roi_status = "ROI Applied - Re-segmented"
                else:
                    roi_status = "ROI Rejected - Initial Mask Retained"

    valid_output_dimensions = final_rgb.shape[:2] == (cfg.target_height, cfg.target_width)
    valid_mask_dimensions = final_mask.shape == (cfg.target_height, cfg.target_width)
    valid_binary_mask = set(np.unique(final_mask).tolist()).issubset({0, 255})
    final_foreground_pixels = int(np.count_nonzero(final_mask))
    technical_valid = bool(
        valid_output_dimensions
        and valid_mask_dimensions
        and valid_binary_mask
        and final_foreground_pixels >= cfg.roi_min_mask_pixels
        and final_info["eligible_mask"]
    )

    info = {
        "original_width": original_image.shape[1],
        "original_height": original_image.shape[0],
        "output_width": final_rgb.shape[1],
        "output_height": final_rgb.shape[0],
        "initial_method": initial_info["selected_method"],
        "final_method": final_info["selected_method"],
        "initial_foreground_ratio": initial_info["foreground_ratio"],
        "final_foreground_ratio": float(final_foreground_pixels / final_mask.size),
        "final_foreground_pixels": final_foreground_pixels,
        "final_border_ratio": final_info["border_ratio"],
        "roi_applied": roi_applied,
        "roi_status": roi_status,
        "roi_mask_overlap": roi_mask_overlap,
        "roi_width": roi_image.shape[1] if roi_image is not None else math.nan,
        "roi_height": roi_image.shape[0] if roi_image is not None else math.nan,
        "valid_output_dimensions": valid_output_dimensions,
        "valid_mask_dimensions": valid_mask_dimensions,
        "valid_binary_mask": valid_binary_mask,
        "technical_valid": technical_valid,
    }
    intermediates = {
        "original_image": original_image,
        "initial_resized": initial_resized,
        "initial_mask": initial_mask,
        "initial_segmented": initial_segmented,
        "initial_info": initial_info,
        "resize_info": resize_info,
        "roi_bbox": roi_bbox,
        "roi_image": roi_image,
        "roi_resize_info": roi_resize_info,
        "final_resize_info": final_resize_info,
        "final_rgb": final_rgb,
        "final_mask": final_mask,
        "final_segmented": final_segmented,
        "final_info": final_info,
    }
    return final_rgb, final_mask, final_segmented, info, intermediates


def refine_internal_mask_holes(
    mask: np.ndarray, config: PreprocessingConfig | None = None
) -> tuple[np.ndarray, dict]:
    """Controlled filling of small internal holes (Cell 12)."""
    cfg = _cfg(config)
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    refined_mask = binary_mask.copy()
    foreground_pixels = int(np.count_nonzero(binary_mask))

    inverse_mask = (binary_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse_mask, connectivity=8)
    boundary_labels = set(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]).tolist()
    )
    internal_holes = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, num_labels)
        if label not in boundary_labels
    ]
    internal_holes.sort(key=lambda item: item[1])  # stable: ties keep label order

    max_single_hole = min(
        cfg.max_internal_hole_pixels, int(foreground_pixels * cfg.max_internal_hole_ratio)
    )
    max_total_fill = int(foreground_pixels * cfg.max_total_fill_ratio)

    filled_holes = 0
    filled_pixels = 0
    for label, hole_area in internal_holes:
        if (
            cfg.apply_controlled_hole_filling
            and hole_area <= max_single_hole
            and filled_pixels + hole_area <= max_total_fill
        ):
            refined_mask[labels == label] = 255
            filled_holes += 1
            filled_pixels += hole_area

    hole_info = {
        "internal_holes_before": len(internal_holes),
        "internal_holes_filled": filled_holes,
        "filled_hole_pixels": filled_pixels,
        "remaining_internal_holes": len(internal_holes) - filled_holes,
        "hole_filling_applied": filled_holes > 0,
    }
    return refined_mask, hole_info


def preprocess_leaf_image_final(
    original_image: np.ndarray, config: PreprocessingConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Final pipeline that produced ``dataset/processed`` (Cell 12).

    Returns ``(processed_rgb, binary_mask, segmented_preview, final_info)``.
    ``processed_rgb`` is the unmasked 256x256 RGB image (what was saved to
    ``dataset/processed/images``) and ``binary_mask`` is the 0/255 mask saved
    to ``dataset/processed/masks``.
    """
    cfg = _cfg(config)
    final_rgb, final_mask, _, preprocessing_info, intermediates = preprocess_leaf_image(
        original_image, apply_roi=cfg.apply_roi_cropping, config=cfg
    )
    refined_mask, hole_info = refine_internal_mask_holes(final_mask, cfg)

    # Notebook: roi_resize_info if ROI applied, else resize_with_padding(original)
    # which equals the initial resize_info (denoising is disabled; median blur
    # does not change the shape either way).
    valid_region = create_valid_region(final_rgb, intermediates["final_resize_info"])
    refined_mask = cv2.bitwise_and(refined_mask, valid_region)

    final_mask_quality = calculate_refined_mask_quality(refined_mask, valid_region, cfg)
    final_segmented = apply_leaf_mask(final_rgb, refined_mask)
    final_image_quality = calculate_image_quality(final_rgb, refined_mask)

    valid_image_dimensions = final_rgb.shape == (cfg.target_height, cfg.target_width, 3)
    valid_mask_dimensions = refined_mask.shape == (cfg.target_height, cfg.target_width)
    valid_binary_mask = set(np.unique(refined_mask).tolist()).issubset({0, 255})
    image_preprocessing_valid = bool(
        preprocessing_info["technical_valid"]
        and valid_image_dimensions
        and valid_mask_dimensions
        and valid_binary_mask
        and final_mask_quality["eligible_mask"]
    )

    quality_issues = []
    if not image_preprocessing_valid:
        quality_issues.append("Invalid preprocessing output")
    if final_mask_quality["foreground_ratio"] < cfg.small_foreground_review_ratio:
        quality_issues.append("Small foreground")
    if cfg.apply_roi_cropping and not preprocessing_info["roi_applied"]:
        quality_issues.append("ROI not applied or rejected")
    if hole_info["remaining_internal_holes"] > 0:
        quality_issues.append("Remaining internal mask holes")
    if hole_info["hole_filling_applied"]:
        quality_issues.append("Mask hole filling requires review")
    if final_image_quality["is_blurry"]:
        quality_issues.append("Potential blur")
    if final_image_quality["is_low_contrast"]:
        quality_issues.append("Low contrast")
    if final_image_quality["is_dark"]:
        quality_issues.append("Low brightness")
    if final_image_quality["is_bright"]:
        quality_issues.append("High brightness")
    if not final_image_quality["quality_measurable"]:
        quality_issues.append("Insufficient quality measurement pixels")

    if not image_preprocessing_valid:
        batch_status = FAILED
    elif quality_issues:
        batch_status = NEEDS_REVIEW
    else:
        batch_status = PASSED

    final_info = {
        **preprocessing_info,
        **hole_info,
        "final_foreground_ratio": final_mask_quality["foreground_ratio"],
        "final_foreground_pixels": final_mask_quality["foreground_pixels"],
        "final_border_ratio": final_mask_quality["border_ratio"],
        "final_valid_region_ratio": final_mask_quality["valid_region_ratio"],
        "final_mask_eligible": final_mask_quality["eligible_mask"],
        "final_blur_score": final_image_quality["blur_score"],
        "final_brightness": final_image_quality["mean_brightness"],
        "final_contrast": final_image_quality["contrast"],
        "valid_output_dimensions": valid_image_dimensions,
        "valid_mask_dimensions": valid_mask_dimensions,
        "valid_binary_mask": valid_binary_mask,
        "technical_valid": image_preprocessing_valid,
        "batch_status": batch_status,
        "image_quality_issues": "; ".join(quality_issues) if quality_issues else "None",
    }
    return final_rgb, refined_mask, final_segmented, final_info


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedImage:
    """Result of preprocessing one uploaded image."""

    rgb: np.ndarray  # 256x256x3 uint8, unmasked (same as dataset/processed/images)
    mask: np.ndarray  # 256x256 uint8, 0/255 (same as dataset/processed/masks)
    segmented: np.ndarray  # 256x256x3 uint8, white background outside the mask
    info: dict[str, Any]

    @property
    def technical_valid(self) -> bool:
        return bool(self.info["technical_valid"])

    @property
    def status(self) -> str:
        return str(self.info["batch_status"])


def preprocess_image(
    source: str | Path | bytes | BinaryIO | np.ndarray,
    config: PreprocessingConfig | None = None,
) -> PreprocessedImage:
    """Load (if needed) and preprocess one image with the final pipeline.

    ``source`` may be a path, bytes, a binary stream (Flask upload), or an
    RGB uint8 array that was already loaded with :func:`load_image`.
    The caller decides what to do when ``technical_valid`` is False; the
    notebook did not save such outputs to the processed dataset.
    """
    if isinstance(source, np.ndarray):
        image = source
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise PreprocessingError("Array input must be an RGB uint8 image (H, W, 3).")
    else:
        image = load_image(source)
    rgb, mask, segmented, info = preprocess_leaf_image_final(image, config)
    return PreprocessedImage(rgb=rgb, mask=mask, segmented=segmented, info=info)


def save_processed_outputs(
    result: PreprocessedImage, image_path: str | Path, mask_path: str | Path
) -> None:
    """Save RGB and mask as PNG exactly as Notebook 02 did (PIL, format PNG)."""
    image_path, mask_path = Path(image_path), Path(mask_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.rgb.astype(np.uint8)).save(image_path, format="PNG")
    Image.fromarray(result.mask.astype(np.uint8)).save(mask_path, format="PNG")


__all__ = [
    "PreprocessingConfig",
    "PreprocessingError",
    "PreprocessedImage",
    "load_config",
    "load_image",
    "resize_with_padding",
    "create_valid_region",
    "segment_hsv",
    "segment_otsu",
    "segment_grabcut",
    "clean_binary_mask",
    "calculate_mask_quality",
    "segment_leaf",
    "calculate_refined_mask_quality",
    "segment_leaf_refined",
    "get_foreground_bbox",
    "map_bbox_to_original",
    "project_mask_to_roi",
    "calculate_mask_iou",
    "apply_leaf_mask",
    "calculate_image_quality",
    "preprocess_leaf_image",
    "refine_internal_mask_holes",
    "preprocess_leaf_image_final",
    "preprocess_image",
    "save_processed_outputs",
]
