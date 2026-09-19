"""Structured prediction report.

LANGUAGE POLICY
---------------
The original UI rendered the raw string "Yes Breast Cancer" / "No Breast Cancer"
and then offered treatment advice keyed off that string. That phrasing asserts a
diagnosis the system is not entitled to make.

Every statement produced here is framed as a model output with an explicit
probability, e.g.

    "The model classified this image as malignant with a predicted
     probability of 87.3%."

The report always carries the research-prototype disclaimer, always names the
model and its validated test performance (or states that none exists), and
always lists known limitations. Nothing in this module produces clinical advice.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np

DISCLAIMER = (
    "This is an academic research prototype. It is not a medical device, has not "
    "been clinically validated, and must not be used for diagnosis or treatment "
    "decisions. Any clinical question should be directed to a qualified healthcare "
    "professional."
)

BASE_LIMITATIONS: List[str] = [
    "Trained on a single public dataset; performance on images from other "
    "scanners, settings or populations is unknown.",
    "Binary benign/malignant classification only -- it cannot recognise normal "
    "tissue, other pathologies, or image types it was not trained on.",
    "No patient history, clinical context or prior imaging is considered.",
    "Probabilities are raw softmax outputs and are not calibrated unless the "
    "model report states otherwise.",
]


def encode_png(image_rgb: np.ndarray) -> str:
    """Encode an RGB uint8 array as a base64 data URI for inline display."""
    from PIL import Image

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _statement(label: str, probability: float, positive_class: str) -> str:
    """One neutral sentence describing the model's output."""
    shown = probability if label == positive_class else 1.0 - probability
    return (
        f"The model classified this image as {label} with a predicted "
        f"probability of {shown * 100:.1f}%."
    )


def _confidence_note(band: str, calibrated: bool) -> str:
    base = {
        "high": "The model's output is far from its decision threshold.",
        "moderate": "The model's output is moderately far from its decision threshold.",
        "low": (
            "The model's output is close to its decision threshold. Predictions in "
            "this range are the least reliable."
        ),
    }.get(band, "")
    if not calibrated:
        base += (
            " Note: this is a raw softmax score, not a calibrated probability, so it "
            "should not be read as a literal likelihood."
        )
    return base.strip()


def build_report(
    result: Dict[str, Any],
    *,
    source_filename: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    include_images: bool = True,
) -> Dict[str, Any]:
    """Assemble the full structured report from a `predictor.predict` result."""
    prediction = result["prediction"]
    model = result["model"]

    limitations = list(BASE_LIMITATIONS)
    if model.get("warning"):
        limitations.insert(0, model["warning"])

    test_metrics = model.get("test_metrics")
    if test_metrics:
        performance: Dict[str, Any] = {
            "available": True,
            "source": "held-out test split",
            **test_metrics,
        }
    else:
        performance = {
            "available": False,
            "message": (
                "No validated test metrics are recorded for this model. "
                "Run `python scripts/evaluate.py` to generate them."
            ),
        }

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "filename": source_filename,
            "width": image_width,
            "height": image_height,
        },
        "classification": {
            "label": prediction["label"],
            "statement": _statement(
                prediction["label"],
                prediction["probability_positive"],
                prediction["positive_class"],
            ),
            "probabilities": prediction["probabilities"],
            "positive_class": prediction["positive_class"],
            "probability_positive": prediction["probability_positive"],
            "decision_threshold": prediction["threshold"],
            "threshold_note": (
                f"A case is reported as {prediction['positive_class']} when its "
                f"predicted probability is at least {prediction['threshold']:.2f}."
            ),
        },
        "confidence": {
            "value": prediction["confidence"],
            "band": prediction["confidence_band"],
            "calibrated": prediction.get("calibrated", False),
            "note": _confidence_note(
                prediction["confidence_band"], prediction.get("calibrated", False)
            ),
        },
        "model": {
            "name": model.get("name"),
            "architecture": model.get("architecture"),
            "parameters": model.get("n_parameters"),
            "preprocessing_profile": model.get("preprocessing_profile"),
            "legacy": model.get("legacy", False),
            "validated_performance": performance,
        },
        "explanation": None,
        "timing": result.get("timing", {}),
        "limitations": limitations,
        "disclaimer": DISCLAIMER,
    }

    explanation = result.get("explanation")
    if explanation:
        report["explanation"] = {
            "method": explanation["method"],
            "reference": explanation["reference"],
            "layer": explanation["layer"],
            "feature_map_size": explanation["feature_map_size"],
            "resolution_note": _resolution_note(explanation["feature_map_size"]),
            "is_degenerate": explanation["is_degenerate"],
            "disclaimer": explanation["disclaimer"],
        }
        if include_images and explanation.get("overlay_rgb") is not None:
            report["explanation"]["overlay_png"] = encode_png(explanation["overlay_rgb"])
    elif result.get("explanation_error"):
        report["explanation"] = {
            "available": False,
            "message": "A visual explanation could not be generated for this image.",
        }
    return report


def _resolution_note(feature_map_size: List[int]) -> str:
    """State the spatial precision of the heatmap honestly."""
    if not feature_map_size or len(feature_map_size) < 2:
        return ""
    cells = feature_map_size[0] * feature_map_size[1]
    note = (
        f"The heatmap is derived from a {feature_map_size[0]}x{feature_map_size[1]} "
        f"feature map ({cells} cells) upsampled to the display size."
    )
    if min(feature_map_size) < 8:
        note += (
            " At this resolution it indicates only a coarse region of interest, "
            "not a lesion boundary."
        )
    return note


def to_text(report: Dict[str, Any]) -> str:
    """Plain-text rendering, for logs, downloads and thesis appendices."""
    lines = [
        "BREAST ULTRASOUND CLASSIFICATION -- RESEARCH PROTOTYPE REPORT",
        "=" * 62,
        f"Generated : {report['generated_utc']}",
        f"Image     : {report['input'].get('filename') or 'n/a'}",
        "",
        "CLASSIFICATION",
        f"  {report['classification']['statement']}",
        f"  Threshold: {report['classification']['decision_threshold']:.2f}",
        "",
        "PROBABILITIES",
    ]
    for name, value in report["classification"]["probabilities"].items():
        lines.append(f"  {name:<12} {value:.4f}")
    lines += [
        "",
        "CONFIDENCE",
        f"  Band: {report['confidence']['band']}  "
        f"(calibrated: {report['confidence']['calibrated']})",
        f"  {report['confidence']['note']}",
        "",
        "MODEL",
        f"  Name        : {report['model']['name']}",
        f"  Architecture: {report['model']['architecture']}",
        f"  Parameters  : {report['model']['parameters']}",
    ]
    performance = report["model"]["validated_performance"]
    if performance.get("available"):
        for key in ("sensitivity", "specificity", "roc_auc"):
            if key in performance:
                lines.append(f"  {key:<12}: {performance[key]}")
    else:
        lines.append(f"  Performance : {performance.get('message')}")

    explanation = report.get("explanation")
    if explanation and explanation.get("method"):
        lines += [
            "",
            "EXPLANATION",
            f"  Method: {explanation['method']} ({explanation['layer']})",
            f"  {explanation.get('resolution_note', '')}",
            f"  {explanation['disclaimer']}",
        ]

    lines += ["", "LIMITATIONS"]
    lines += [f"  - {item}" for item in report["limitations"]]
    lines += ["", "DISCLAIMER", f"  {report['disclaimer']}", ""]
    return "\n".join(lines)
