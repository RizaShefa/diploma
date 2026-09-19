"""Classification metrics with bootstrap confidence intervals.

WHY NOT JUST ACCURACY
---------------------
The original evaluation reported accuracy (99.6%) as the headline figure. For
cancer detection that is the least informative metric available:

  * Sensitivity (recall on malignant) is primary -- a false negative is a missed
    cancer.
  * Specificity is its counterweight -- a false positive is an unnecessary
    follow-up, harmful but recoverable.
  * PR-AUC is more honest than ROC-AUC when the positive class is the minority,
    which it is in any realistic screening population.
  * A point estimate from ~100 test images is not a result. Every metric here
    carries a bootstrap 95% confidence interval so the thesis can report
    "0.88 [0.79-0.95]" instead of a bare number.

All definitions are written out explicitly rather than delegated, so the thesis
methodology section can state exactly how each figure was computed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def confusion_counts(
    y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1
) -> Dict[str, int]:
    """TP/TN/FP/FN for the binary case, relative to `positive_label`."""
    matrix = confusion_matrix(y_true, y_pred, labels=[1 - positive_label, positive_label])
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def point_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    positive_label: int = 1,
) -> Dict[str, float]:
    """All threshold-dependent and threshold-free metrics at one operating point.

    `y_prob` is the predicted probability of the POSITIVE class.
    """
    y_pred = (y_prob >= threshold).astype(int)
    if positive_label == 0:  # normalise so positive is always 1 internally
        y_true = 1 - y_true
        y_pred = 1 - y_pred

    counts = confusion_counts(y_true, y_pred, positive_label=1)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]

    sensitivity = _safe_divide(tp, tp + fn)          # recall / true positive rate
    specificity = _safe_divide(tn, tn + fp)          # true negative rate
    precision = _safe_divide(tp, tp + fp)            # positive predictive value
    npv = _safe_divide(tn, tn + fn)                  # negative predictive value
    f1 = _safe_divide(2 * precision * sensitivity, precision + sensitivity)

    metrics: Dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": _safe_divide(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        # Counts stay integers: they are tallies, and floats read as noise in
        # the dashboard and in thesis tables.
        **{k: int(v) for k, v in counts.items()},
    }

    # Threshold-free metrics require both classes to be present.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CIs by resampling test cases with replacement.

    Resampling the TEST SET (not retraining) estimates the sampling variability
    of the metric given this model. With ~100 test images the intervals are
    wide, which is the honest picture and worth stating in the thesis.
    """
    if metric_names is None:
        metric_names = [
            "accuracy", "balanced_accuracy", "sensitivity", "specificity",
            "precision", "npv", "f1", "roc_auc", "pr_auc",
        ]
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # degenerate resample; skip rather than record a fake value
        sample_metrics = point_metrics(y_true[idx], y_prob[idx], threshold)
        for name in metric_names:
            value = sample_metrics.get(name)
            if value is not None and not np.isnan(value):
                collected[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for name, values in collected.items():
        if not values:
            out[name] = {"lower": float("nan"), "upper": float("nan"), "n": 0}
            continue
        array = np.asarray(values)
        out[name] = {
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
            "n": int(len(array)),
        }
    return out


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 101
) -> List[Dict[str, float]]:
    """Metrics across the full threshold range -- the operating-point table."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    return [point_metrics(y_true, y_prob, float(t)) for t in thresholds]


def select_threshold_for_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.95
) -> Dict[str, float]:
    """Highest threshold that still achieves at least `target_sensitivity`.

    Rationale: in breast cancer screening a missed malignancy is costlier than
    an unnecessary follow-up, so the operating point is chosen for sensitivity
    rather than left at the default 0.5. Choosing the HIGHEST such threshold
    gives the best specificity available at that sensitivity.
    """
    sweep = threshold_sweep(y_true, y_prob)
    qualifying = [row for row in sweep if row["sensitivity"] >= target_sensitivity]
    if not qualifying:
        best = max(sweep, key=lambda r: r["sensitivity"])
        return {**best, "target_met": False, "target_sensitivity": target_sensitivity}
    chosen = max(qualifying, key=lambda r: r["threshold"])
    return {**chosen, "target_met": True, "target_sensitivity": target_sensitivity}


def roc_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": [float(t) for t in thresholds],
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def pr_points(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": [float(t) for t in thresholds],
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """ECE plus the per-bin data needed for a reliability diagram.

    Softmax outputs are typically overconfident. If the UI displays a
    probability, the thesis is obliged to show whether "90%" means 90%.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / total) * abs(accuracy - confidence)
        rows.append({"bin_lower": float(lo), "bin_upper": float(hi), "count": count,
                     "confidence": confidence, "accuracy": accuracy})
    return {"ece": float(ece), "n_bins": n_bins, "bins": rows}


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    bootstrap_iterations: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    target_sensitivity: Optional[float] = None,
) -> Dict[str, Any]:
    """Complete evaluation bundle for one model on one split."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    result: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "class_balance": {
            "positive": int(y_true.sum()),
            "negative": int(len(y_true) - y_true.sum()),
        },
        "metrics": point_metrics(y_true, y_prob, threshold),
        "confidence_intervals": bootstrap_confidence_intervals(
            y_true, y_prob, threshold, bootstrap_iterations, confidence_level, seed
        ),
        "calibration": expected_calibration_error(y_true, y_prob),
    }
    if len(np.unique(y_true)) > 1:
        result["roc"] = roc_points(y_true, y_prob)
        result["pr"] = pr_points(y_true, y_prob)
    if target_sensitivity is not None:
        result["operating_point"] = select_threshold_for_sensitivity(
            y_true, y_prob, target_sensitivity
        )
    return result


def format_metric(value: float, ci: Optional[Dict[str, float]] = None) -> str:
    """Render '0.881 [0.795-0.947]' for thesis tables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if ci and not np.isnan(ci.get("lower", float("nan"))):
        return f"{value:.3f} [{ci['lower']:.3f}-{ci['upper']:.3f}]"
    return f"{value:.3f}"
