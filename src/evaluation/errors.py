"""Error analysis: where does the model fail, and on what kind of image?

PURPOSE
-------
A gallery of misclassified images is a UI page. This module aims at a research
finding instead, by pairing each error with measurable image properties and
testing whether errors concentrate in particular regions of that property space.

The question it is built to answer is the one the thesis should ask:

    "Are false negatives systematically smaller, darker or lower-contrast
     lesions than the cases the model gets right?"

That is falsifiable from the available data, unlike "the model struggles with
difficult cases".

Properties are deliberately simple and computed from the image itself -- BUSI
carries no clinical metadata (age, scanner, BI-RADS), so anything richer would
have to be invented, which this project does not do.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..data.loader import Sample

CATEGORIES = ("true_positive", "true_negative", "false_positive", "false_negative")


def categorize(y_true: int, y_pred: int, positive_label: int = 1) -> str:
    """Map one (truth, prediction) pair to a confusion category."""
    is_positive = y_true == positive_label
    predicted_positive = y_pred == positive_label
    if is_positive and predicted_positive:
        return "true_positive"
    if not is_positive and not predicted_positive:
        return "true_negative"
    if not is_positive and predicted_positive:
        return "false_positive"
    return "false_negative"


def image_properties(path: str) -> Dict[str, Optional[float]]:
    """Cheap, interpretable descriptors of one ultrasound frame.

    mean_intensity      overall brightness
    std_intensity       global contrast
    rms_contrast        normalised contrast
    laplacian_variance  focus/sharpness proxy (low = blurry)
    dark_fraction       proportion of near-black pixels (hypoechoic + background)
    edge_density        Canny edge density, a texture/speckle proxy
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {k: None for k in (
            "width", "height", "mean_intensity", "std_intensity", "rms_contrast",
            "laplacian_variance", "dark_fraction", "edge_density")}
    array = image.astype(np.float32)
    mean = float(array.mean())
    std = float(array.std())
    return {
        "width": float(image.shape[1]),
        "height": float(image.shape[0]),
        "mean_intensity": mean,
        "std_intensity": std,
        "rms_contrast": float(std / mean) if mean > 1e-6 else 0.0,
        "laplacian_variance": float(cv2.Laplacian(image, cv2.CV_64F).var()),
        "dark_fraction": float((image < 40).mean()),
        "edge_density": float((cv2.Canny(image, 50, 150) > 0).mean()),
    }


def build_error_table(
    samples: Sequence[Sample],
    indices: Sequence[int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    positive_label: int = 1,
    *,
    with_properties: bool = True,
) -> List[Dict[str, Any]]:
    """One record per test image: prediction, confidence, category, properties."""
    y_pred = (y_prob >= threshold).astype(int)
    rows: List[Dict[str, Any]] = []
    for position, dataset_index in enumerate(indices):
        sample = samples[dataset_index]
        probability = float(y_prob[position])
        category = categorize(int(y_true[position]), int(y_pred[position]), positive_label)
        row: Dict[str, Any] = {
            "index": int(dataset_index),
            "path": sample.path,
            "filename": sample.stem,
            "class_name": sample.class_name,
            "true_label": int(y_true[position]),
            "predicted_label": int(y_pred[position]),
            "probability_positive": probability,
            # Confidence in the DECISION the model made, not in the positive class.
            "confidence": probability if y_pred[position] == 1 else 1.0 - probability,
            "correct": bool(y_true[position] == y_pred[position]),
            "category": category,
        }
        if with_properties:
            row["properties"] = image_properties(sample.path)
        rows.append(row)
    return rows


def category_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {name: 0 for name in CATEGORIES}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    return counts


def property_comparison(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare image properties of correct vs incorrect predictions.

    Reports mean/std per group and a standardised mean difference (Cohen's d).
    This is the quantitative core of the error analysis: a |d| of ~0.5 or more
    on, say, `rms_contrast` would support the claim that errors concentrate on
    low-contrast images.

    No p-values are reported. With a test set of ~100 images and several
    properties examined, per-property significance testing would invite
    multiple-comparison errors; effect sizes are the honest summary.
    """
    correct = [r for r in rows if r["correct"] and r.get("properties")]
    wrong = [r for r in rows if not r["correct"] and r.get("properties")]
    if not correct or not wrong:
        return {
            "available": False,
            "reason": "Need at least one correct and one incorrect prediction.",
            "n_correct": len(correct), "n_incorrect": len(wrong),
        }

    keys = [k for k, v in correct[0]["properties"].items() if v is not None]
    comparison: Dict[str, Any] = {}
    for key in keys:
        a = np.array([r["properties"][key] for r in correct
                      if r["properties"].get(key) is not None], dtype=float)
        b = np.array([r["properties"][key] for r in wrong
                      if r["properties"].get(key) is not None], dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                            / (len(a) + len(b) - 2))
        cohens_d = float((b.mean() - a.mean()) / pooled_sd) if pooled_sd > 1e-12 else 0.0
        comparison[key] = {
            "correct_mean": float(a.mean()), "correct_std": float(a.std(ddof=1)),
            "incorrect_mean": float(b.mean()), "incorrect_std": float(b.std(ddof=1)),
            "cohens_d": cohens_d,
            "interpretation": _interpret_d(cohens_d),
        }
    return {
        "available": True,
        "n_correct": len(correct),
        "n_incorrect": len(wrong),
        "properties": comparison,
    }


def _interpret_d(d: float) -> str:
    """Conventional Cohen's d bands, stated so readers can judge for themselves."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def confidence_separation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Do correct and incorrect predictions differ in confidence?

    If they overlap heavily the confidence score cannot be used for triage or
    abstention -- a finding worth reporting, and the motivation for calibration.
    """
    correct = np.array([r["confidence"] for r in rows if r["correct"]], dtype=float)
    wrong = np.array([r["confidence"] for r in rows if not r["correct"]], dtype=float)
    if len(correct) == 0 or len(wrong) == 0:
        return {"available": False, "n_correct": len(correct), "n_incorrect": len(wrong)}
    return {
        "available": True,
        "correct_mean_confidence": float(correct.mean()),
        "incorrect_mean_confidence": float(wrong.mean()),
        "correct_median": float(np.median(correct)),
        "incorrect_median": float(np.median(wrong)),
        "gap": float(correct.mean() - wrong.mean()),
        "n_correct": int(len(correct)),
        "n_incorrect": int(len(wrong)),
    }


def abstention_curve(rows: Sequence[Dict[str, Any]], steps: int = 21) -> List[Dict[str, float]]:
    """Accuracy vs coverage when low-confidence cases are referred on.

    Models a realistic clinical workflow: the system decides only when confident
    and refers the rest to a specialist. Reports what accuracy is achieved on
    the cases it does decide.
    """
    confidences = np.array([r["confidence"] for r in rows], dtype=float)
    correctness = np.array([r["correct"] for r in rows], dtype=bool)
    curve: List[Dict[str, float]] = []
    for cutoff in np.linspace(0.5, 1.0, steps):
        keep = confidences >= cutoff
        coverage = float(keep.mean())
        accuracy = float(correctness[keep].mean()) if keep.any() else float("nan")
        curve.append({
            "confidence_cutoff": float(cutoff),
            "coverage": coverage,
            "accuracy_on_covered": accuracy,
            "n_covered": int(keep.sum()),
        })
    return curve


def hardest_cases(rows: Sequence[Dict[str, Any]], n: int = 10) -> Dict[str, List[Dict[str, Any]]]:
    """Most confidently wrong, and least confident overall."""
    wrong = [r for r in rows if not r["correct"]]
    return {
        "most_confident_errors": sorted(wrong, key=lambda r: -r["confidence"])[:n],
        "least_confident": sorted(rows, key=lambda r: r["confidence"])[:n],
        "false_negatives": [r for r in wrong if r["category"] == "false_negative"][:n],
        "false_positives": [r for r in wrong if r["category"] == "false_positive"][:n],
    }


def analyze(
    samples: Sequence[Sample],
    indices: Sequence[int],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run the full error analysis and return a JSON-serialisable bundle."""
    rows = build_error_table(samples, indices, y_true, y_prob, threshold)
    return {
        "threshold": float(threshold),
        "counts": category_counts(rows),
        "property_comparison": property_comparison(rows),
        "confidence_separation": confidence_separation(rows),
        "abstention_curve": abstention_curve(rows),
        "hardest_cases": hardest_cases(rows),
        "rows": rows,
    }
