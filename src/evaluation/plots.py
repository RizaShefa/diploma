"""Figure generation for the evaluation dashboard and the thesis.

Every figure is produced from real, saved evaluation output. Nothing here
invents data: functions take arrays/metric dicts and render them. If an
experiment has not been run, the caller has nothing to pass and the dashboard
shows "not yet executed" rather than a placeholder chart.

Figures are written at 150 dpi, which is adequate for on-screen use and for
print at typical thesis figure widths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: these run in scripts and inside Flask
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150
FIGSIZE = (6.0, 5.0)


def _save(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def confusion_matrix_figure(
    counts: Dict[str, int], class_names: Sequence[str], path: str | Path, title: str = "Confusion matrix"
) -> str:
    """2x2 confusion matrix annotated with counts and row-normalised rates."""
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    row_totals = matrix.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            rate = matrix[i, j] / row_totals[i, 0] if row_totals[i, 0] else 0.0
            ax.text(
                j, i, f"{matrix[i, j]}\n({rate:.1%})",
                ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                fontsize=12,
            )
    ax.set_xticks([0, 1], labels=[f"Pred {class_names[0]}", f"Pred {class_names[1]}"])
    ax.set_yticks([0, 1], labels=[f"True {class_names[0]}", f"True {class_names[1]}"])
    ax.set_title(title)
    return _save(fig, path)


def roc_figure(roc: Dict[str, Any], path: str | Path, title: str = "ROC curve",
               operating_point: Optional[Dict[str, float]] = None) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(roc["fpr"], roc["tpr"], lw=2, label=f"AUC = {roc['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey", label="Chance")
    if operating_point:
        ax.plot(
            1 - operating_point["specificity"], operating_point["sensitivity"],
            "o", ms=9, color="crimson",
            label=f"Operating point (t={operating_point['threshold']:.2f})",
        )
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    return _save(fig, path)


def pr_figure(pr: Dict[str, Any], path: str | Path, positive_rate: Optional[float] = None,
              title: str = "Precision-recall curve") -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(pr["recall"], pr["precision"], lw=2,
            label=f"AP = {pr['average_precision']:.3f}")
    if positive_rate is not None:
        ax.axhline(positive_rate, ls="--", lw=1, color="grey",
                   label=f"Chance ({positive_rate:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    return _save(fig, path)


def calibration_figure(calibration: Dict[str, Any], path: str | Path,
                       title: str = "Reliability diagram") -> str:
    """Reliability diagram: predicted confidence vs observed accuracy per bin."""
    bins = [b for b in calibration["bins"] if b["count"] > 0]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    if bins:
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                "o-", lw=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of positives")
    ax.set_title(f"{title}  (ECE = {calibration['ece']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    return _save(fig, path)


def threshold_figure(sweep: List[Dict[str, float]], path: str | Path,
                     target_sensitivity: Optional[float] = None,
                     title: str = "Sensitivity / specificity vs threshold") -> str:
    """The clinical trade-off curve -- how the operating point was chosen."""
    thresholds = [r["threshold"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(thresholds, [r["sensitivity"] for r in sweep], lw=2, label="Sensitivity")
    ax.plot(thresholds, [r["specificity"] for r in sweep], lw=2, label="Specificity")
    ax.plot(thresholds, [r["f1"] for r in sweep], lw=1.5, ls=":", label="F1")
    if target_sensitivity is not None:
        ax.axhline(target_sensitivity, ls="--", color="crimson", lw=1,
                   label=f"Target sensitivity {target_sensitivity:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def training_history_figure(history: Dict[str, List[float]], path: str | Path,
                            title: str = "Training history") -> str:
    """Loss and accuracy curves for train vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    epochs = range(1, len(history.get("loss", [])) + 1)
    axes[0].plot(epochs, history.get("loss", []), marker="o", ms=3, label="Train")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], marker="o", ms=3, label="Validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("accuracy", []), marker="o", ms=3, label="Train")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], marker="o", ms=3, label="Validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    return _save(fig, path)


def model_comparison_figure(rows: List[Dict[str, Any]], path: str | Path,
                            metric: str = "sensitivity",
                            title: Optional[str] = None) -> str:
    """Bar chart of one metric across models, with bootstrap CI error bars."""
    names = [r["model_name"] for r in rows]
    values = [r["metrics"].get(metric, float("nan")) for r in rows]
    lowers, uppers = [], []
    for r in rows:
        ci = r.get("confidence_intervals", {}).get(metric, {})
        lo, hi = ci.get("lower"), ci.get("upper")
        value = r["metrics"].get(metric, float("nan"))
        lowers.append(0.0 if lo is None or np.isnan(lo) else max(0.0, value - lo))
        uppers.append(0.0 if hi is None or np.isnan(hi) else max(0.0, hi - value))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(names)), 5.0))
    ax.bar(names, values, yerr=[lowers, uppers], capsize=6, color="#4C72B0")
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.1)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by model (95% CI)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, path)


def cv_stability_figure(fold_metrics: Dict[str, List[float]], path: str | Path,
                        title: str = "Cross-validation stability") -> str:
    """Box plot of metrics across CV folds -- how stable is the model?"""
    labels = list(fold_metrics.keys())
    data = [fold_metrics[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(labels)), 5.0))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return _save(fig, path)
