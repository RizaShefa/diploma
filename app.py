"""Flask application -- thin routing layer only.

All business logic lives in `src/`:

    src/inference/validation.py   upload validation
    src/preprocessing.py          image preprocessing (shared with training)
    src/inference/predictor.py    model registry + inference
    src/explainability/gradcam.py Grad-CAM
    src/reporting/report.py       structured report
    src/inference/history.py      prediction history
    src/evaluation/              metrics and figures

This module only parses requests, calls those services, and renders responses.

BACKWARD COMPATIBILITY
----------------------
Every original route is preserved. `POST /predict` still returns a plain-text
string so the existing jQuery handler keeps working, but it now runs through the
validated, preprocessing-correct pipeline. Its wording changed from
"Yes Breast Cancer" to a probability statement, because asserting a diagnosis is
not something this system is entitled to do.

`POST /api/predict` is the new JSON endpoint carrying confidence, Grad-CAM and
the full structured report.

CONFIGURATION (environment variables, all optional)
---------------------------------------------------
    BCD_MODEL_NAME        model to serve (default: newest non-legacy, else legacy)
    BCD_STORE_IMAGES      "1" to keep thumbnails in history (default: off)
    BCD_HISTORY_DB        history database path (default: results/history.sqlite3)
    BCD_MAX_UPLOAD_MB     upload size cap in MB (default: 10)
    BCD_SECRET_KEY        Flask secret key (default: a dev-only placeholder)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from src.config import PROJECT_ROOT
from src.inference.history import PredictionHistory
from src.inference.predictor import ModelNotAvailableError, ModelRegistry, predict
from src.inference.validation import ValidationError, validate_upload
from src.reporting.report import build_report, to_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("breast_cancer_detection")

MAX_UPLOAD_MB = float(os.environ.get("BCD_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SECRET_KEY"] = os.environ.get("BCD_SECRET_KEY", "dev-only-not-for-deployment")

registry = ModelRegistry()
history = PredictionHistory(
    db_path=os.environ.get("BCD_HISTORY_DB", PROJECT_ROOT / "results" / "history.sqlite3"),
    store_images=os.environ.get("BCD_STORE_IMAGES", "0") == "1",
)
DEFAULT_MODEL = os.environ.get("BCD_MODEL_NAME") or None
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------------------------------
# Static pages (unchanged from the original application)
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/home", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/service")
def service():
    return render_template("service.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/appointment")
def appointment():
    return render_template("appointment.html")


# --------------------------------------------------------------------------
# New research-facing pages
# --------------------------------------------------------------------------
@app.route("/research")
def research():
    """Evaluation dashboard: metrics, figures, model comparison, error analysis."""
    from src.reporting.results_index import load_results_index

    return render_template("research.html", index=load_results_index(RESULTS_DIR))


@app.route("/dataset")
def dataset_page():
    """Dataset analysis: distribution, integrity audit, split composition."""
    from src.reporting.results_index import load_dataset_report

    return render_template("dataset.html", report=load_dataset_report(RESULTS_DIR))


@app.route("/history")
def history_page():
    return render_template(
        "history.html", rows=history.list(limit=100), summary=history.summary()
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _read_upload():
    """Extract and validate the uploaded file. Raises ValidationError."""
    if "file" not in request.files:
        raise ValidationError("No file was included in the request.")
    uploaded = request.files["file"]
    payload = uploaded.read()
    return validate_upload(payload, uploaded.filename or "", max_bytes=MAX_UPLOAD_BYTES)


@app.route("/predict", methods=["POST"])
def predict_legacy():
    """Original endpoint, preserved. Returns plain text for the existing UI.

    Now runs through validation and the shared preprocessing pipeline, so the
    result matches what the model was trained to receive.
    """
    try:
        validated = _read_upload()
    except ValidationError as exc:
        return str(exc), 400

    try:
        result = predict(
            registry, validated.image_bgr,
            model_name=DEFAULT_MODEL, with_explanation=False,
        )
    except ModelNotAvailableError as exc:
        logger.error("Model unavailable: %s", exc)
        return "No prediction model is available on the server.", 503
    except Exception:
        logger.exception("Prediction failed")
        return "Prediction failed due to an internal error.", 500

    history.record(
        result, original_filename=validated.original_filename,
        image_width=validated.width, image_height=validated.height,
        image_bgr=validated.image_bgr,
    )
    prediction = result["prediction"]
    shown = prediction["probability_positive"]
    if prediction["label"] != prediction["positive_class"]:
        shown = 1.0 - shown
    return f"Model classification: {prediction['label']} (predicted probability {shown * 100:.1f}%)"


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON prediction with confidence, Grad-CAM overlay and structured report."""
    try:
        validated = _read_upload()
    except ValidationError as exc:
        return jsonify({"error": str(exc), "error_type": "validation"}), 400

    model_name = request.form.get("model") or DEFAULT_MODEL
    want_explanation = request.form.get("explain", "1") != "0"

    try:
        result = predict(
            registry, validated.image_bgr,
            model_name=model_name, with_explanation=want_explanation,
        )
    except ModelNotAvailableError as exc:
        logger.error("Model unavailable: %s", exc)
        return jsonify({"error": str(exc), "error_type": "model_unavailable"}), 503
    except Exception:
        logger.exception("Prediction failed")
        return jsonify(
            {"error": "Prediction failed due to an internal error.",
             "error_type": "inference"}
        ), 500

    report = build_report(
        result,
        source_filename=validated.original_filename,
        image_width=validated.width,
        image_height=validated.height,
    )
    row_id = history.record(
        result, original_filename=validated.original_filename,
        image_width=validated.width, image_height=validated.height,
        image_bgr=validated.image_bgr,
    )
    report["history_id"] = row_id
    if result.get("explanation_error"):
        logger.warning("Grad-CAM unavailable: %s", result["explanation_error"])
    return jsonify(report)


@app.route("/api/report/<int:row_id>.txt")
def api_report_text(row_id: int):
    """Plain-text report for a stored prediction (metadata only)."""
    rows = [r for r in history.list(limit=500) if r["id"] == row_id]
    if not rows:
        return "Report not found.", 404
    row = rows[0]
    body = "\n".join([
        "BREAST ULTRASOUND CLASSIFICATION -- STORED PREDICTION",
        "=" * 55,
        f"Recorded   : {row['created_utc']}",
        f"Model      : {row['model_name']}" + (" (LEGACY)" if row["model_legacy"] else ""),
        f"Prediction : {row['predicted_label']}",
        f"P(positive): {row['probability_positive']:.4f}",
        f"Confidence : {row['confidence']:.4f} ({row['confidence_band']})",
        f"Threshold  : {row['threshold']:.2f}",
        f"Image      : {row['original_filename']} ({row['image_width']}x{row['image_height']})",
        "",
        "This is an academic research prototype. Not a medical device.",
        "",
    ])
    return app.response_class(body, mimetype="text/plain")


# --------------------------------------------------------------------------
# JSON APIs
# --------------------------------------------------------------------------
@app.route("/api/models")
def api_models():
    registry.refresh()
    return jsonify({"models": registry.describe_all(), "default": _safe_default()})


def _safe_default():
    try:
        return registry.default_name()
    except ModelNotAvailableError:
        return None


@app.route("/api/history")
def api_history():
    limit = min(int(request.args.get("limit", 50)), 200)
    return jsonify({"rows": history.list(limit=limit), "summary": history.summary()})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    removed = history.clear()
    logger.info("History cleared (%d rows removed)", removed)
    return jsonify({"cleared": removed})


@app.route("/api/history/<int:row_id>/thumbnail")
def api_history_thumbnail(row_id: int):
    from io import BytesIO

    blob = history.thumbnail(row_id)
    if not blob:
        return jsonify({"error": "No thumbnail stored for this prediction."}), 404
    return send_file(BytesIO(blob), mimetype="image/jpeg")


@app.route("/api/results")
def api_results():
    from src.reporting.results_index import load_results_index

    return jsonify(load_results_index(RESULTS_DIR))


@app.route("/results/figures/<path:filename>")
def results_figure(filename: str):
    """Serve generated figures, constrained to the results directory."""
    figures_root = (RESULTS_DIR / "figures").resolve()
    target = (figures_root / filename).resolve()
    if not str(target).startswith(str(figures_root)) or not target.exists():
        return jsonify({"error": "Figure not found."}), 404
    return send_file(target)


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "models_available": registry.names(),
        "default_model": _safe_default(),
        "results_available": (RESULTS_DIR / "index.json").exists(),
    })


# --------------------------------------------------------------------------
# Error handlers -- never leak internals to the client
# --------------------------------------------------------------------------
@app.errorhandler(404)
def handle_404(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("home.html"), 404


@app.errorhandler(413)
def handle_413(_):
    message = f"File is too large. Maximum upload size is {MAX_UPLOAD_MB:.0f} MB."
    if request.path.startswith("/api/"):
        return jsonify({"error": message, "error_type": "validation"}), 413
    return message, 413


@app.errorhandler(500)
def handle_500(error):
    logger.exception("Unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error."}), 500
    return "Internal server error.", 500


if __name__ == "__main__":
    # debug=False by default: the original ran with debug=True, which exposes the
    # Werkzeug console and allows arbitrary code execution if the port is reachable.
    debug = os.environ.get("BCD_DEBUG", "0") == "1"
    app.run(host=os.environ.get("BCD_HOST", "127.0.0.1"),
            port=int(os.environ.get("BCD_PORT", "5000")),
            debug=debug)
