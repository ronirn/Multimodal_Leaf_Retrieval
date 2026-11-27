"""Tests for core/preprocessing.py.

The parity test re-runs the final pipeline on every raw image listed in
``dataset/processed/metadata_processed.csv`` and compares the result with the
files that Notebook 02 saved in ``dataset/processed/images`` and
``dataset/processed/masks`` (pixel by pixel) and with the stored metadata.

Nothing in the dataset is modified. Run from the project root:

    .venv\\Scripts\\python.exe -m unittest tests.test_preprocessing -v

Set PREPROCESSING_PARITY_REPORT=<path.csv> to also write a per-image report.
"""

from __future__ import annotations

import csv
import io
import math
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import morphology

from core import preprocessing as pp

ROOT = pp.PROJECT_ROOT
METADATA = ROOT / "dataset" / "processed" / "metadata_processed.csv"


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"))


class ConfigTests(unittest.TestCase):
    def test_final_config_values(self):
        cfg = pp.load_config()
        self.assertEqual(cfg.target_size, (256, 256))
        self.assertEqual(cfg.padding_color, (255, 255, 255))
        self.assertTrue(cfg.apply_roi_cropping)
        self.assertTrue(cfg.apply_controlled_hole_filling)
        self.assertFalse(cfg.apply_denoising)
        self.assertEqual(cfg.morph_kernel.shape, (3, 3))

    def test_unsupported_segmentation_rejected(self):
        raw = dict(pp.load_config().raw)
        raw["segmentation_method"] = "Adaptive Selection"
        with self.assertRaises(ValueError):
            pp.PreprocessingConfig.from_dict(raw)


class UnitTests(unittest.TestCase):
    def test_small_holes_max_size_equivalent_to_area_threshold(self):
        """SMALL_HOLE_MAX_SIZE must reproduce the notebook's area_threshold=32.

        scikit-image 0.26 forwards area_threshold to max_size, so a hole of
        exactly 32 px is filled; this test fails if that behaviour changes
        (e.g. after a scikit-image upgrade).
        """
        rng = np.random.default_rng(0)
        for _ in range(50):
            mask = rng.random((96, 96)) > 0.35
            mask = ndimage.binary_opening(mask)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                old = morphology.remove_small_holes(mask, area_threshold=32)
            new = morphology.remove_small_holes(mask, max_size=pp.SMALL_HOLE_MAX_SIZE)
            np.testing.assert_array_equal(old, new)

    def test_small_hole_boundary_32_filled_33_kept(self):
        for hole_size, expect_filled in ((32, True), (33, False)):
            mask = np.ones((40, 60), dtype=bool)
            mask[10, 10 : 10 + hole_size] = False  # 1-px tall hole of hole_size px
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                old = morphology.remove_small_holes(mask, area_threshold=32)
            new = morphology.remove_small_holes(mask, max_size=pp.SMALL_HOLE_MAX_SIZE)
            np.testing.assert_array_equal(old, new)
            self.assertEqual(bool(new.all()), expect_filled)

    def test_resize_with_padding_shapes(self):
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        out, info = pp.resize_with_padding(image)
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual((info["resized_width"], info["resized_height"]), (256, 128))
        self.assertEqual((info["pad_top"], info["pad_bottom"]), (64, 64))
        self.assertTrue((out[:64] == 255).all())

    def test_load_image_from_bytes_and_invalid(self):
        buf = io.BytesIO()
        Image.new("RGB", (40, 20), (10, 200, 30)).save(buf, format="PNG")
        image = pp.load_image(buf.getvalue())
        self.assertEqual(image.shape, (20, 40, 3))
        with self.assertRaises(pp.PreprocessingError):
            pp.load_image(b"not an image")

    def test_uniform_image_is_not_technically_valid(self):
        image = np.full((200, 200, 3), 240, dtype=np.uint8)
        result = pp.preprocess_image(image)
        self.assertEqual(result.rgb.shape, (256, 256, 3))
        self.assertEqual(result.mask.shape, (256, 256))
        self.assertFalse(result.technical_valid)


class DatasetParityTests(unittest.TestCase):
    """Compare against the saved Notebook 02 outputs for all processed images."""

    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(METADATA)
        cls.records = []
        for row in cls.metadata.itertuples(index=False):
            result = pp.preprocess_image(ROOT / row.image_path)
            saved_rgb = _read_rgb(ROOT / row.processed_image_path)
            saved_mask = _read_mask(ROOT / row.mask_path)
            info = result.info
            stored_issues = (
                "None" if pd.isna(row.image_quality_issues) else str(row.image_quality_issues)
            )
            cls.records.append(
                {
                    "sample_id": row.sample_id,
                    "image_path": row.image_path,
                    "rgb_identical": bool(np.array_equal(result.rgb, saved_rgb)),
                    "rgb_diff_pixels": int(np.any(result.rgb != saved_rgb, axis=2).sum())
                    if result.rgb.shape == saved_rgb.shape
                    else -1,
                    "mask_identical": bool(np.array_equal(result.mask, saved_mask)),
                    "mask_diff_pixels": int((result.mask != saved_mask).sum())
                    if result.mask.shape == saved_mask.shape
                    else -1,
                    "technical_valid": info["technical_valid"],
                    "roi_applied": info["roi_applied"],
                    "roi_applied_stored": bool(row.roi_applied),
                    "hole_filling_applied": info["hole_filling_applied"],
                    "hole_filling_stored": bool(row.hole_filling_applied),
                    "remaining_holes": info["remaining_internal_holes"],
                    "remaining_holes_stored": int(row.remaining_internal_holes),
                    "fg_pixels": info["final_foreground_pixels"],
                    "fg_pixels_stored": int(row.final_foreground_pixels),
                    "fg_ratio": info["final_foreground_ratio"],
                    "fg_ratio_stored": float(row.final_foreground_ratio),
                    "border_ratio": info["final_border_ratio"],
                    "border_ratio_stored": float(row.final_border_ratio),
                    "batch_status": info["batch_status"],
                    "batch_status_stored": row.batch_status,
                    "issues": info["image_quality_issues"],
                    "issues_stored": stored_issues,
                }
            )
        report = os.environ.get("PREPROCESSING_PARITY_REPORT")
        if report:
            with open(report, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(cls.records[0]))
                writer.writeheader()
                writer.writerows(cls.records)

    def _mismatches(self, predicate):
        return [r["image_path"] for r in self.records if not predicate(r)]

    def test_all_samples_processed(self):
        self.assertEqual(len(self.records), 104)

    def test_processed_rgb_pixel_identical(self):
        self.assertEqual(self._mismatches(lambda r: r["rgb_identical"]), [])

    def test_binary_mask_pixel_identical(self):
        self.assertEqual(self._mismatches(lambda r: r["mask_identical"]), [])

    def test_all_technically_valid(self):
        self.assertEqual(self._mismatches(lambda r: r["technical_valid"]), [])

    def test_roi_and_hole_filling_flags_match(self):
        self.assertEqual(
            self._mismatches(
                lambda r: r["roi_applied"] == r["roi_applied_stored"]
                and r["hole_filling_applied"] == r["hole_filling_stored"]
                and r["remaining_holes"] == r["remaining_holes_stored"]
            ),
            [],
        )

    def test_mask_statistics_match(self):
        self.assertEqual(
            self._mismatches(
                lambda r: r["fg_pixels"] == r["fg_pixels_stored"]
                and math.isclose(r["fg_ratio"], r["fg_ratio_stored"], abs_tol=1e-9)
                and math.isclose(r["border_ratio"], r["border_ratio_stored"], abs_tol=1e-9)
            ),
            [],
        )

    def test_batch_status_and_issues_match(self):
        self.assertEqual(
            self._mismatches(
                lambda r: r["batch_status"] == r["batch_status_stored"]
                and r["issues"] == r["issues_stored"]
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
