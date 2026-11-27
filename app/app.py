
"""LeafMatch AI - Multimodal Medicinal Plant Leaf Retrieval.

Run from the project root:
    .venv\\Scripts\\python.exe app\\app.py

Pages:
1. Upload & Retrieval
2. Model Evaluation

The root URL redirects to Upload & Retrieval.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys

from datetime import date
from functools import lru_cache
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
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
from core.gallery_loader import load_gallery_data, search_gallery


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

    with path.open(
        newline="",
        encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


# ============================================================
# 4. EXPERIMENT SUMMARY
# ============================================================

@lru_cache(maxsize=1)
def load_experiment_summary() -> dict:

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

    with provenance_path.open(
        encoding="utf-8"
    ) as file:
        locked = json.load(file)[
            "locked_configuration"
        ]

    split_path = (
        SPLIT_RUNS_DIR
        / run_id
        / "751_final_dataset_split_inventory.csv"
    )

    inventory = _read_csv(split_path)

    split_counts = {
        split: sum(
            row["split"] == split
            for row in inventory
        )
        for split in (
            "gallery",
            "validation",
            "test"
        )
    }

    return {
        "run_id": run_id,
        "text_configuration": locked[
            "text_configuration"
        ],
        "alpha": float(locked["alpha"]),
        "validation_map5": float(
            latest["validation_map_at_5"]
        ),
        "test_status": latest["test_status"],
        "split_counts": split_counts,
        "total_samples": len(inventory),
    }


# ============================================================
# 5. FIND EVALUATION REPORT
# ============================================================

def find_evaluation_report(
    run_id: str
) -> Path | None:

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

        ROOT / "07_evaluation_report.md",
    ]

    if results_root.exists():
        candidates.extend(
            sorted(
                results_root.rglob(
                    "07_evaluation_report.md"
                )
            )
        )

    seen = set()

    for path in candidates:

        if path in seen or not path.is_file():
            continue

        seen.add(path)

        try:
            if run_id in path.read_text(
                encoding="utf-8-sig"
            ):
                return path

        except (OSError, UnicodeError):
            continue

    return None


# ============================================================
# 6. MARKDOWN REPORT HELPERS
# ============================================================

def _extract_report_section(
    report: str,
    start_pattern: str
) -> str:

    match = re.search(
        start_pattern,
        report,
        flags=re.MULTILINE | re.IGNORECASE
    )

    if not match:
        return ""

    level = len(match.group(0)) - len(
        match.group(0).lstrip("#")
    )

    remaining = report[match.end():]

    next_heading = re.search(
        rf"(?m)^#{{1,{level}}}\s+",
        remaining
    )

    if next_heading:
        return remaining[:next_heading.start()]

    return remaining


def _read_markdown_rows(
    section: str,
    columns: int
) -> list[list[str]]:

    rows = []

    for line in section.splitlines():

        if not line.strip().startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        if len(cells) == columns:
            rows.append(cells)

    return rows


def _score(text: str) -> float:

    value = float(
        text.replace(",", ".")
    )

    if (
        not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"Invalid evaluation metric: {text}"
        )

    return value


# ============================================================
# 7. LOAD EVALUATION CHARTS
# ============================================================

def load_evaluation_charts(summary: dict) -> dict:

    output = {
        "comparison": [],
        "metric_comparison": [],
        "topk_performance": [],
    }

    report_path = find_evaluation_report(
        summary["run_id"]
    )

    if report_path is None:

        app.logger.warning(
            "Notebook 07 report missing for %s",
            summary["run_id"]
        )

        return output

    report = report_path.read_text(
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 7.1 VALIDATION METRICS
    # --------------------------------------------------------

    validation_section = _extract_report_section(
        report,
        r"^###\s+Validation\b[^\n]*"
    )

    validation_metrics = {}

    for cells in _read_markdown_rows(
        validation_section,
        columns=6
    ):

        try:
            values = [
                _score(cell)
                for cell in cells[1:]
            ]

        except ValueError:
            continue

        method = " ".join(
            cells[0].split()
        )

        validation_metrics[method] = dict(
            zip(
                (
                    "precision",
                    "recall",
                    "mrr",
                    "map",
                    "ndcg"
                ),
                values
            )
        )

    # --------------------------------------------------------
    # 7.2 MATCH MODEL CONFIGURATIONS
    # --------------------------------------------------------

    visual_key = "Visual Only"

    text_key = (
        "Text Only - Unigram + Bigram"
    )

    fusion_key = (
        "Weighted Fusion - Unigram + Bigram "
        f"(alpha={summary['alpha']:g})"
    )

    required = (
        visual_key,
        text_key,
        fusion_key
    )

    if all(
        key in validation_metrics
        for key in required
    ):

        visual = validation_metrics[visual_key]
        textual = validation_metrics[text_key]
        fusion = validation_metrics[fusion_key]

        if abs(
            fusion["map"]
            - summary["validation_map5"]
        ) > 0.001:
            raise ValueError(
                "Validation mAP@5 differs from registry."
            )

        # Comparison chart
        output["comparison"] = [
            {
                "label": "Visual Only",
                "map_at_5": visual["map"],
                "is_selected": False,
            },
            {
                "label": "Text Only",
                "map_at_5": textual["map"],
                "is_selected": False,
            },
            {
                "label": "Multimodal Fusion",
                "map_at_5": fusion["map"],
                "is_selected": True,
            },
        ]

        # Radar and grouped bar charts
        metric_names = [
            ("mAP@5", "map"),
            ("Precision@5", "precision"),
            ("Recall@5", "recall"),
            ("MRR@5", "mrr"),
            ("nDCG@5", "ndcg"),
        ]

        output["metric_comparison"] = [
            {
                "label": label,
                "visual": visual[key],
                "text": textual[key],
                "fusion": fusion[key],
            }
            for label, key in metric_names
        ]

    else:

        app.logger.warning(
            "Incomplete validation data: %s",
            report_path
        )

    # --------------------------------------------------------
    # 7.3 TOP-K PERFORMANCE
    # --------------------------------------------------------

    topk_section = _extract_report_section(
        report,
        r"^##\s+3\.\s+Performa konfigurasi[^\n]*"
    )

    topk_maps = {}

    for cells in _read_markdown_rows(
        topk_section,
        columns=11
    ):

        try:
            k = int(cells[0])
            value = _score(cells[1])

        except ValueError:
            continue

        if k in (1, 3, 5, 10):
            topk_maps[k] = value

    required_k = (1, 3, 5, 10)

    if all(k in topk_maps for k in required_k):

        if abs(
            topk_maps[5]
            - summary["validation_map5"]
        ) > 0.001:
            raise ValueError(
                "Top-K mAP@5 differs from registry."
            )

        output["topk_performance"] = [
            {
                "k": k,
                "fusion": topk_maps[k],
            }
            for k in required_k
        ]

    else:

        app.logger.warning(
            "Incomplete Top-K results: %s",
            report_path
        )

    return output


# ============================================================
# 8. GLOBAL TEMPLATE DATA
# ============================================================

@app.context_processor
def inject_globals():
    return {
        "current_year": date.today().year
    }


# ============================================================
# 9. HOME -> REDIRECT TO RETRIEVAL
# ============================================================

# Endpoint "dashboard" dipertahankan sebagai alias
# sementara agar tautan lama pada base.html
# tidak menyebabkan BuildError.
#
# Tidak ada halaman Dashboard yang dirender.

@app.route("/", endpoint="dashboard")
def home():
    return redirect(
        url_for("retrieval")
    )


# ============================================================
# 10. UPLOAD & RETRIEVAL PAGE
# ============================================================

@app.route("/retrieval")
def retrieval():

    return render_template(
        "retrieval.html",
        active_page="retrieval"
    )


# ============================================================
# 11. EVALUATION PAGE
# ============================================================

@app.route("/evaluation")
def evaluation():

    summary = load_experiment_summary()

    charts = load_evaluation_charts(
        summary
    )

    evaluation_data = {
        **summary,
        **charts
    }

    return render_template(
        "evaluation.html",
        active_page="evaluation",
        evaluation=evaluation_data
    )


# ============================================================
# 12. GALLERY IMAGE ENDPOINT
# ============================================================

@app.route("/api/gallery-image/<sample_id>")
def gallery_image(sample_id):

    gallery = load_gallery_data()

    sample = next(
        (
            item
            for item in gallery["metadata"]
            if item["sample_id"] == sample_id
        ),
        None
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
# 13. MULTIMODAL RETRIEVAL API
# ============================================================

@app.route("/api/retrieval", methods=["POST"])
def api_retrieval():

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    image_file = request.files.get("image")

    description = request.form.get(
        "description",
        ""
    ).strip()

    if image_file is None or not image_file.filename:
        return jsonify(
            error="Please upload a leaf image."
        ), 400

    if not description:
        return jsonify(
            error="Please enter a leaf description."
        ), 400

    if len(description) > 2000:
        return jsonify(
            error="Description exceeds 2000 characters."
        ), 400

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    extension = Path(
        image_file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify(
            error="Only JPG, JPEG and PNG images are supported."
        ), 400

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image_bytes = image_file.read(
        MAX_IMAGE_SIZE + 1
    )

    if not image_bytes:
        return jsonify(
            error="Uploaded image is empty."
        ), 400

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return jsonify(
            error="Maximum image size is 10 MB."
        ), 400

    # --------------------------------------------------------
    # PREPROCESSING + RETRIEVAL
    # --------------------------------------------------------

    try:

        processed = preprocess_image(
            image_bytes
        )

        if not processed.technical_valid:
            return jsonify(
                error=(
                    "Image preprocessing failed. "
                    "Upload a clear leaf photo."
                )
            ), 400

        raw_visual_features = (
            extract_features_from_preprocessed(
                processed
            )
        )

        results = search_gallery(
            query_visual_raw=raw_visual_features,
            query_description=description,
            top_k=5
        )

        # ----------------------------------------------------
        # BUILD RESULTS
        # ----------------------------------------------------

        response_results = [
            {
                "rank": item["rank"],
                "sample_id": item["sample_id"],
                "kelas": item["kelas"],

                "nama_file": item.get(
                    "nama_file",
                    ""
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
                    sample_id=item["sample_id"]
                ),
            }
            for item in results
        ]

        return jsonify(
            success=True,
            gallery_size=load_gallery_data()["gallery_size"],
            top_k=len(response_results),
            results=response_results
        )

    except ValueError as error:

        return jsonify(
            error=str(error)
        ), 400

    except Exception:

        app.logger.exception(
            "Multimodal retrieval failed"
        )

        return jsonify(
            error=(
                "Retrieval failed. "
                "See Flask terminal for details."
            )
        ), 500


# ============================================================
# 14. FILE SIZE ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify(
        error=(
            "Upload too large. "
            "Maximum image size is 10 MB."
        )
    ), 413


# ============================================================
# 15. RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
