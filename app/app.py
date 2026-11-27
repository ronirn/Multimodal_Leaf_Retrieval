
"""LeafMatch AI - Multimodal Medicinal Plant Leaf Retrieval.

Flask application:
1. Dashboard
2. Upload & Retrieval
3. Model Evaluation

Run from project root:
    .venv\\Scripts\\python.exe app\\app.py
"""

from __future__ import annotations

import csv
import json
import sys

from datetime import date
from functools import lru_cache
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preprocessing import preprocess_image
from core.image_features import extract_features_from_preprocessed
from core.gallery_loader import (
    load_gallery_data,
    search_gallery,
)


METADATA_PATH = (
    ROOT / "dataset" / "processed"
    / "metadata_processed.csv"
)

ASSET_MANIFEST_PATH = (
    ROOT / "static" / "images"
    / "dataset" / "manifest.json"
)

EVAL_REGISTRY_PATH = (
    ROOT / "results" / "evaluation_07"
    / "evaluation_registry.csv"
)

EVAL_RUNS_DIR = (
    ROOT / "results" / "evaluation_07" / "runs"
)

SPLIT_RUNS_DIR = (
    ROOT / "results" / "metrics"
    / "retrieval_evaluation" / "runs"
)

CLASS_ORDER = [
    "Mangkok",
    "Salam",
    "Sambang Darah",
    "Sirih",
    "Tapak Kuda",
]

MAX_IMAGE_SIZE = 10 * 1024 * 1024


# ============================================================
# 2. FLASK INITIALIZATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
    static_url_path="/static",
)

app.config["MAX_CONTENT_LENGTH"] = 11 * 1024 * 1024


# ============================================================
# 3. CSV HELPER
# ============================================================

def _read_csv(path: Path) -> list[dict]:
    """Read CSV file using UTF-8 encoding."""

    with open(
        path,
        newline="",
        encoding="utf-8-sig",
    ) as fh:
        return list(csv.DictReader(fh))


# ============================================================
# 4. DASHBOARD DATA
# ============================================================

@lru_cache(maxsize=1)
def load_plant_classes() -> list[dict]:
    """
    Load plant class information from
    processed dataset metadata.
    """

    rows = _read_csv(METADATA_PATH)

    with open(
        ASSET_MANIFEST_PATH,
        encoding="utf-8",
    ) as fh:
        assets = json.load(fh)["classes"]

    classes = []

    for name in CLASS_ORDER:

        class_rows = [
            row
            for row in rows
            if row["kelas"] == name
        ]

        asset = assets[name]

        source = next(
            row
            for row in class_rows
            if row["image_path"] == asset["source"]
        )

        classes.append({
            "name": name,
            "slug": name.lower().replace(" ", "-"),
            "samples": len(class_rows),

            "front": sum(
                row["tampilan_daun"] == "depan"
                for row in class_rows
            ),

            "back": sum(
                row["tampilan_daun"] == "belakang"
                for row in class_rows
            ),

            "description": source["original_description"],
            "source": asset["source"],
            "card": asset["card"],
            "texture": asset["texture"],
            "thumb": asset["thumb"],
        })

    return classes


# ============================================================
# 5. EXPERIMENT SUMMARY
# ============================================================

@lru_cache(maxsize=1)
def load_experiment_summary() -> dict:
    """
    Load locked experiment configuration and
    dataset split information from Notebook 07.
    """

    registry = _read_csv(EVAL_REGISTRY_PATH)

    if not registry:
        raise ValueError(
            "Evaluation registry is empty."
        )

    latest = registry[-1]
    run_id = latest["source_run_id"]

    provenance_path = (
        EVAL_RUNS_DIR
        / run_id
        / "07_provenance.json"
    )

    with open(
        provenance_path,
        encoding="utf-8",
    ) as fh:
        locked = json.load(fh)["locked_configuration"]

    split_path = (
        SPLIT_RUNS_DIR
        / run_id
        / "751_final_dataset_split_inventory.csv"
    )

    splits = _read_csv(split_path)

    split_counts = {
        name: sum(
            row["split"] == name
            for row in splits
        )
        for name in (
            "gallery",
            "validation",
            "test",
        )
    }

    return {
        "run_id": run_id,
        "text_configuration": locked["text_configuration"],
        "alpha": float(locked["alpha"]),
        "validation_map5": float(
            latest["validation_map_at_5"]
        ),
        "test_status": latest["test_status"],
        "split_counts": split_counts,
        "total_samples": len(splits),
    }


# ============================================================
# 6. EVALUATION REPORT LOADER
# ============================================================

def find_evaluation_report(run_id: str) -> Path | None:
    """
    Find the Notebook 07 evaluation report
    for the selected experiment run.

    Only reports containing the matching
    experiment run ID are accepted.
    """

    results_root = ROOT / "results"

    candidates = [
        EVAL_RUNS_DIR
        / run_id
        / "07_evaluation_report.md",

        EVAL_RUNS_DIR
        / run_id
        / "reports"
        / "07_evaluation_report.md",

        results_root
        / "evaluation_07"
        / "07_evaluation_report.md",
    ]

    if results_root.exists():
        candidates.extend(
            sorted(
                results_root.rglob(
                    "07_evaluation_report.md"
                )
            )
        )

    checked = set()

    for path in candidates:

        if path in checked:
            continue

        checked.add(path)

        if not path.is_file():
            continue

        try:
            content = path.read_text(
                encoding="utf-8-sig"
            )

            if run_id in content:
                return path

        except (OSError, UnicodeError):
            continue

    return None


def load_validation_comparison(summary: dict) -> list[dict]:
    """
    Read validation mAP@5 from the Notebook 07 report.

    Methods:
    - Visual Only
    - Text Only (Unigram + Bigram)
    - Weighted Fusion (selected alpha)

    No model retraining or evaluation is performed.
    """

    run_id = summary["run_id"]
    alpha = summary["alpha"]

    report_path = find_evaluation_report(run_id)

    if report_path is None:
        app.logger.warning(
            "Evaluation report not found for run %s",
            run_id,
        )
        return []

    report = report_path.read_text(
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Extract validation section
    # --------------------------------------------------------

    if "### Validation" not in report:
        app.logger.warning(
            "Validation section not found in %s",
            report_path,
        )
        return []

    validation_section = report.split(
        "### Validation",
        1,
    )[1]

    validation_section = validation_section.split(
        "### Test",
        1,
    )[0]

    # --------------------------------------------------------
    # Parse validation metrics table
    # --------------------------------------------------------

    metrics = {}

    for line in validation_section.splitlines():

        line = line.strip()

        if not line.startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        # Expected columns:
        # Method | Precision@5 | Recall@5 |
        # MRR@5 | mAP@5 | nDCG@5

        if len(cells) != 6:
            continue

        method_name = " ".join(cells[0].split())

        try:
            map_at_5 = float(
                cells[4].replace(",", ".")
            )
        except ValueError:
            continue

        if not 0 <= map_at_5 <= 1:
            continue

        metrics[method_name] = map_at_5

    # --------------------------------------------------------
    # Selected experiment configurations
    # --------------------------------------------------------

    visual_key = "Visual Only"

    text_key = (
        "Text Only - Unigram + Bigram"
    )

    fusion_key = (
        "Weighted Fusion - Unigram + Bigram "
        f"(alpha={alpha:g})"
    )

    methods = [
        {
            "source_name": visual_key,
            "label": "Visual Only",
            "is_selected": False,
        },
        {
            "source_name": text_key,
            "label": "Text Only",
            "is_selected": False,
        },
        {
            "source_name": fusion_key,
            "label": "Multimodal Fusion",
            "is_selected": True,
        },
    ]

    missing = [
        method["source_name"]
        for method in methods
        if method["source_name"] not in metrics
    ]

    if missing:
        app.logger.warning(
            "Missing validation methods in report: %s",
            missing,
        )
        return []

    # --------------------------------------------------------
    # Build comparison
    # --------------------------------------------------------

    comparison = []

    for method in methods:

        comparison.append({
            "label": method["label"],
            "map_at_5": metrics[
                method["source_name"]
            ],
            "is_selected": method["is_selected"],
        })

    # --------------------------------------------------------
    # Verify selected fusion against registry
    # --------------------------------------------------------

    fusion_score = comparison[-1]["map_at_5"]

    registry_score = summary["validation_map5"]

    if abs(fusion_score - registry_score) > 1e-3:
        raise ValueError(
            "Validation mAP@5 in evaluation report "
            "does not match the experiment registry."
        )

    return comparison


# ============================================================
# 7. GLOBAL TEMPLATE DATA
# ============================================================

@app.context_processor
def inject_globals():
    return {
        "current_year": date.today().year,
    }


# ============================================================
# 8. DASHBOARD ROUTE
# ============================================================

@app.route("/")
def dashboard():
    """
    Render Dashboard page.
    """

    classes = load_plant_classes()

    summary = load_experiment_summary()

    leaf_of_day = classes[
        date.today().toordinal() % len(classes)
    ]

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        classes=classes,
        summary=summary,
        leaf_of_day=leaf_of_day,
    )


# ============================================================
# 9. RETRIEVAL PAGE ROUTE
# ============================================================

@app.route("/retrieval")
def retrieval():
    """
    Render Upload & Retrieval page.
    """

    return render_template(
        "retrieval.html",
        active_page="retrieval",
    )


# ============================================================
# 10. EVALUATION PAGE ROUTE
# ============================================================

@app.route("/evaluation")
def evaluation():
    """
    Render Model Evaluation page using
    the locked Notebook 06/07 results.
    """

    # Load experiment information
    summary = load_experiment_summary()

    # Load real validation comparison
    comparison = load_validation_comparison(
        summary
    )

    # Prepare template data
    evaluation_data = {
        **summary,
        "comparison": comparison,
    }

    return render_template(
        "evaluation.html",
        active_page="evaluation",
        evaluation=evaluation_data,
    )


# ============================================================
# 11. GALLERY IMAGE ENDPOINT
# ============================================================

@app.route("/api/gallery-image/<sample_id>")
def gallery_image(sample_id):
    """
    Serve gallery images based on sample IDs.

    Only images registered in the gallery
    and located inside the dataset directory
    can be accessed.
    """

    gallery = load_gallery_data()

    sample = next(
        (
            row
            for row in gallery["metadata"]
            if row["sample_id"] == sample_id
        ),
        None,
    )

    if sample is None:
        abort(404)

    relative_path = Path(
        sample.get("processed_image_path", "")
    )

    if relative_path.is_absolute():
        abort(404)

    image_path = (
        ROOT / relative_path
    ).resolve()

    allowed_root = (
        ROOT / "dataset"
    ).resolve()

    if (
        not image_path.is_relative_to(allowed_root)
        or not image_path.is_file()
        or image_path.suffix.lower()
        not in {".jpg", ".jpeg", ".png"}
    ):
        abort(404)

    return send_file(image_path)


# ============================================================
# 12. MULTIMODAL RETRIEVAL API
# ============================================================

@app.route("/api/retrieval", methods=["POST"])
def api_retrieval():
    """
    Multimodal retrieval endpoint.

    Input:
        Image: JPG / PNG
        Description: Indonesian text

    Processing:
        1. Image preprocessing
        2. Visual feature extraction
        3. Gallery-based visual normalization
        4. TF-IDF feature transformation
        5. Euclidean similarity
        6. Cosine similarity
        7. Weighted score-level late fusion
        8. Top-5 ranking

    Output:
        JSON containing retrieval results.
    """

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    image_file = request.files.get("image")

    description = request.form.get(
        "description",
        "",
    ).strip()

    if image_file is None or not image_file.filename:
        return jsonify({
            "error": "Please upload a leaf image."
        }), 400

    if not description:
        return jsonify({
            "error": "Please enter a leaf description."
        }), 400

    if len(description) > 2000:
        return jsonify({
            "error": (
                "Description is too long. "
                "Maximum 2000 characters."
            )
        }), 400

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    extension = Path(
        image_file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify({
            "error": (
                "Only JPG, JPEG and PNG "
                "images are supported."
            )
        }), 400

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = image_file.read(
        MAX_IMAGE_SIZE + 1
    )

    if not image_bytes:
        return jsonify({
            "error": "Uploaded image is empty."
        }), 400

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify({
            "error": "Maximum image size is 10 MB."
        }), 400

    # --------------------------------------------------------
    # Preprocessing and retrieval
    # --------------------------------------------------------

    try:

        # Original image preprocessing
        processed = preprocess_image(
            image_bytes
        )

        if not processed.technical_valid:
            return jsonify({
                "error": (
                    "Image preprocessing failed. "
                    "Please upload a clear leaf photograph."
                )
            }), 400

        # HSV + GLCM + Hu Moments
        query_visual_raw = (
            extract_features_from_preprocessed(
                processed
            )
        )

        # Multimodal retrieval using real gallery
        results = search_gallery(
            query_visual_raw=query_visual_raw,
            query_description=description,
            top_k=5,
        )

        # ----------------------------------------------------
        # Build JSON response
        # ----------------------------------------------------

        response_results = []

        for item in results:

            response_results.append({
                "rank": item["rank"],
                "sample_id": item["sample_id"],
                "kelas": item["kelas"],
                "nama_file": item.get(
                    "nama_file",
                    "",
                ),
                "visual_similarity": item[
                    "visual_similarity"
                ],
                "text_similarity": item[
                    "text_similarity"
                ],
                "fusion_score": item[
                    "fusion_score"
                ],
                "image_url": url_for(
                    "gallery_image",
                    sample_id=item["sample_id"],
                ),
            })

        return jsonify({
            "success": True,
            "gallery_size": 62,
            "top_k": len(response_results),
            "results": response_results,
        })

    except ValueError as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:

        app.logger.exception(
            "Multimodal retrieval failed."
        )

        return jsonify({
            "error": (
                "Retrieval processing failed. "
                "Please check the Flask terminal."
            )
        }), 500


# ============================================================
# 13. FILE SIZE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error": (
            "Upload is too large. "
            "Maximum image size is 10 MB."
        )
    }), 413


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
