"""Tests for core/image_features.py.

Compares the new extractor with the Notebook 03 artifacts for all 104
processed images, in the sample order of ``visual_sample_ids.npy``:

* from the saved processed PNGs (exactly the input Notebook 03 read), and
* end-to-end from the raw images through ``core.preprocessing.preprocess_image``.

The normaliser test checks that applying the saved Notebook 06 gallery
normaliser (no refitting) reproduces Notebook 06's normalised split matrices.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest tests.test_image_features -v

Set IMAGE_FEATURE_PARITY_REPORT=<path.csv> to write a per-image report.
"""

from __future__ import annotations

import csv
import json
import os
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from core import image_features as imf
from core import preprocessing as pp

ROOT = imf.PROJECT_ROOT
VISUAL_DIR = ROOT / "dataset" / "features" / "visual"
METADATA = ROOT / "dataset" / "processed" / "metadata_processed.csv"
RUN_DIR = imf.DEFAULT_NORMALIZER_PATH.parent
SPLIT_INVENTORY = (
    ROOT
    / "results"
    / "metrics"
    / "retrieval_evaluation"
    / "runs"
    / imf.DEFAULT_RUN_ID
    / "751_final_dataset_split_inventory.csv"
)

# Tolerance used only if a comparison is not bit-identical (reported, not hidden).
ATOL = 1e-12

GROUP_FILES = {"HSV": "hsv_features.npy", "GLCM": "glcm_features.npy", "Hu Moments": "hu_features.npy"}


def _load_processed_pair(image_path, mask_path):
    """Same loading as Notebook 03 load_processed_pair (no mode conversion)."""
    with Image.open(image_path) as img:
        rgb = np.array(img)
    with Image.open(mask_path) as msk:
        mask = np.array(msk)
    return rgb, mask


class FeatureLayoutTests(unittest.TestCase):
    def test_dimensions_and_slices(self):
        self.assertEqual(imf.HSV_FEATURE_DIM, 64)
        self.assertEqual(imf.GLCM_FEATURE_DIM, 72)
        self.assertEqual(imf.HU_FEATURE_DIM, 7)
        self.assertEqual(imf.TOTAL_VISUAL_FEATURE_DIM, 143)
        self.assertEqual(
            imf.FEATURE_GROUP_SLICES, {"HSV": (0, 64), "GLCM": (64, 136), "Hu Moments": (136, 143)}
        )

    def test_feature_names_match_notebook03(self):
        with open(VISUAL_DIR / "visual_feature_names.json", encoding="utf-8") as fh:
            saved_names = json.load(fh)
        self.assertEqual(imf.VISUAL_FEATURE_NAMES, saved_names)

    def test_glcm_offsets(self):
        self.assertEqual(len(set(imf.GLCM_OFFSET_LOOKUP.values())), 12)
        self.assertEqual(imf.GLCM_OFFSET_LOOKUP[(2, 135)], (2, -2))

    def test_invalid_inputs_rejected(self):
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        with self.assertRaises(imf.FeatureExtractionError):
            imf.extract_visual_features(rgb, np.zeros((256, 256), dtype=np.uint8))
        small = np.zeros((256, 256), dtype=np.uint8)
        small[0:5, 0:5] = 255  # 25 px < 100
        with self.assertRaises(imf.FeatureExtractionError):
            imf.extract_visual_features(rgb, small)
        with self.assertRaises(imf.FeatureExtractionError):
            imf.extract_visual_features(np.zeros((128, 128, 3), dtype=np.uint8), small)


class Notebook03ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_ref = np.load(VISUAL_DIR / "visual_features_raw.npy")
        cls.sample_ids = np.load(VISUAL_DIR / "visual_sample_ids.npy")
        cls.group_ref = {g: np.load(VISUAL_DIR / f) for g, f in GROUP_FILES.items()}
        cls.hu_raw_ref = np.load(VISUAL_DIR / "hu_raw_moments.npy")

        meta = pd.read_csv(METADATA).set_index("sample_id")
        cls.meta = meta.loc[cls.sample_ids]  # artifact order

        from_files, from_pipeline, hu_raw, groups = [], [], [], {g: [] for g in GROUP_FILES}
        for sid, row in cls.meta.iterrows():
            rgb, mask = _load_processed_pair(ROOT / row.processed_image_path, ROOT / row.mask_path)
            grp = imf.extract_visual_feature_groups(rgb, mask)
            for g in groups:
                groups[g].append(grp[g])
            from_files.append(imf.concatenate_visual_features(grp))
            hu_raw.append(imf.extract_raw_hu_moments(mask))

            result = pp.preprocess_image(ROOT / row.image_path)
            from_pipeline.append(imf.extract_features_from_preprocessed(result))

        cls.from_files = np.vstack(from_files)
        cls.from_pipeline = np.vstack(from_pipeline)
        cls.groups = {g: np.vstack(v) for g, v in groups.items()}
        cls.hu_raw = np.vstack(hu_raw)

        report = os.environ.get("IMAGE_FEATURE_PARITY_REPORT")
        if report:
            with open(report, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(
                    ["sample_id", "identical_from_files", "identical_from_pipeline",
                     "max_abs_diff_files", "max_abs_diff_pipeline"]
                )
                for i, sid in enumerate(cls.sample_ids):
                    w.writerow([
                        sid,
                        bool(np.array_equal(cls.from_files[i], cls.raw_ref[i])),
                        bool(np.array_equal(cls.from_pipeline[i], cls.raw_ref[i])),
                        float(np.abs(cls.from_files[i] - cls.raw_ref[i]).max()),
                        float(np.abs(cls.from_pipeline[i] - cls.raw_ref[i]).max()),
                    ])

    def _assert_match(self, actual, expected, label):
        max_diff = float(np.abs(actual - expected).max())
        identical = bool(np.array_equal(actual, expected))
        print(f"\n[{label}] identical={identical} max_abs_diff={max_diff:.3e}")
        if not identical:
            np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL, err_msg=label)

    def test_reference_artifacts_consistent(self):
        self.assertEqual(self.raw_ref.shape, (104, 143))
        self.assertEqual(len(set(self.sample_ids)), 104)
        stacked = np.hstack([self.group_ref[g] for g in imf.FEATURE_GROUP_ORDER])
        np.testing.assert_array_equal(stacked, self.raw_ref)
        # Notebook 06 used the same raw matrix.
        np.testing.assert_array_equal(np.load(RUN_DIR / "visual_raw_143.npy"), self.raw_ref)

    def test_shape_and_finite(self):
        for m in (self.from_files, self.from_pipeline):
            self.assertEqual(m.shape, (104, 143))
            self.assertTrue(np.isfinite(m).all())

    def test_sample_order(self):
        self.assertEqual(list(self.meta.index), list(self.sample_ids))

    def test_group_dimensions_and_values(self):
        for g, (start, stop) in imf.FEATURE_GROUP_SLICES.items():
            self.assertEqual(self.groups[g].shape[1], stop - start)
            np.testing.assert_array_equal(self.from_files[:, start:stop], self.groups[g])
            self._assert_match(self.groups[g], self.group_ref[g], f"group {g}")

    def test_hsv_channels_l1_normalised(self):
        hsv = self.groups["HSV"]
        for a, b in ((0, 32), (32, 48), (48, 64)):
            np.testing.assert_allclose(hsv[:, a:b].sum(axis=1), 1.0, atol=1e-12)

    def test_raw_hu_moments_match(self):
        self._assert_match(self.hu_raw, self.hu_raw_ref, "raw Hu moments")

    def test_from_processed_files_match_visual_features_raw(self):
        self._assert_match(self.from_files, self.raw_ref, "processed PNG -> 143")

    def test_end_to_end_from_raw_images_match(self):
        self._assert_match(self.from_pipeline, self.raw_ref, "raw image -> preprocess -> 143")


class GalleryNormalizerTests(unittest.TestCase):
    def test_transform_reproduces_notebook06_split_matrices(self):
        normalizer = imf.load_gallery_normalizer()
        self.assertEqual(normalizer.n_reference_samples, 62)
        raw = np.load(VISUAL_DIR / "visual_features_raw.npy")
        ids = np.load(VISUAL_DIR / "visual_sample_ids.npy")
        inventory = pd.read_csv(SPLIT_INVENTORY)
        self.assertTrue((inventory.sample_id.values == ids[inventory.matrix_row_index.values]).all())
        splits = np.load(RUN_DIR / "visual_gallery_fitted_splits.npz")
        for split in ("gallery", "validation", "test"):
            rows = inventory[inventory.split == split].sort_values("matrix_row_index")
            out = imf.transform_visual_features(raw[rows.matrix_row_index.values], normalizer)
            np.testing.assert_array_equal(out, splits[split], err_msg=split)

    def test_single_vector_transform_matches_matrix(self):
        normalizer = imf.load_gallery_normalizer()
        raw = np.load(VISUAL_DIR / "visual_features_raw.npy")
        single = imf.transform_visual_features(raw[5], normalizer)
        self.assertEqual(single.shape, (143,))
        np.testing.assert_array_equal(single, imf.transform_visual_features(raw, normalizer)[5])

    def test_normalizer_file_unchanged_by_transform(self):
        normalizer = imf.load_gallery_normalizer()
        before = {g: p["mean"].copy() for g, p in normalizer.params["groups"].items()}
        imf.transform_visual_features(np.zeros((3, 143)), normalizer)
        for g, p in normalizer.params["groups"].items():
            np.testing.assert_array_equal(p["mean"], before[g])


if __name__ == "__main__":
    unittest.main()
