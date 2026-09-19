"""Dataset analysis and figures.

Every figure here exists to support a methodology or limitations claim in the
thesis, not to decorate the dashboard:

  class_distribution      justifies class weighting
  dimension_scatter       exposes size-vs-class shortcuts (see below)
  intensity_histogram     exposes brightness-vs-class shortcuts
  split_composition       documents the realised train/val/test split
  augmentation_examples   SHOWS the augmentation instead of asserting it
  sample_grid             qualitative look at the data

SHORTCUT CHECK
--------------
`class_correlated_artifacts` is the important one. If image dimensions or mean
brightness differ systematically between benign and malignant, a CNN can reach
high accuracy by reading an acquisition artifact rather than pathology. The
dataset previously shipped with this project contained two distinct image sizes
(224x224 and 227x227) in a single class folder, which is exactly the kind of
signal that warrants checking before any performance claim is made.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .loader import Sample

DPI = 150


def _save(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def summarize(samples: Sequence[Sample]) -> Dict[str, Any]:
    """Counts, dimensions and intensity statistics per class."""
    per_class: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "widths": [], "heights": [], "means": [], "stds": [], "with_mask": 0}
    )
    for sample in samples:
        entry = per_class[sample.class_name]
        entry["count"] += 1
        if sample.mask_paths:
            entry["with_mask"] += 1
        image = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        entry["heights"].append(image.shape[0])
        entry["widths"].append(image.shape[1])
        entry["means"].append(float(image.mean()))
        entry["stds"].append(float(image.std()))

    summary: Dict[str, Any] = {"total": len(samples), "classes": {}}
    for class_name, entry in per_class.items():
        widths = np.array(entry["widths"]) if entry["widths"] else np.array([0])
        heights = np.array(entry["heights"]) if entry["heights"] else np.array([0])
        means = np.array(entry["means"]) if entry["means"] else np.array([0.0])
        summary["classes"][class_name] = {
            "count": entry["count"],
            "with_mask": entry["with_mask"],
            "width": {"min": int(widths.min()), "max": int(widths.max()), "mean": float(widths.mean())},
            "height": {"min": int(heights.min()), "max": int(heights.max()), "mean": float(heights.mean())},
            "intensity": {"mean": float(means.mean()), "std": float(means.std())},
            "distinct_sizes": len({(w, h) for w, h in zip(entry["widths"], entry["heights"])}),
        }
    counts = {k: v["count"] for k, v in summary["classes"].items()}
    if counts:
        summary["imbalance_ratio"] = round(max(counts.values()) / max(1, min(counts.values())), 3)
    return summary


def class_correlated_artifacts(samples: Sequence[Sample]) -> Dict[str, Any]:
    """Test whether trivial image properties separate the classes.

    A large standardised difference here is a warning that reported accuracy may
    reflect an acquisition shortcut rather than lesion appearance.
    """
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"area": [], "mean_intensity": [], "aspect_ratio": []}
    )
    for sample in samples:
        image = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        height, width = image.shape[:2]
        entry = groups[sample.class_name]
        entry["area"].append(float(height * width))
        entry["mean_intensity"].append(float(image.mean()))
        entry["aspect_ratio"].append(float(width / height) if height else 0.0)

    names = sorted(groups)
    if len(names) != 2:
        return {"available": False, "reason": f"Expected 2 classes, found {len(names)}."}

    a_name, b_name = names
    findings: Dict[str, Any] = {}
    for key in ("area", "mean_intensity", "aspect_ratio"):
        a = np.array(groups[a_name][key], dtype=float)
        b = np.array(groups[b_name][key], dtype=float)
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
        d = float((b.mean() - a.mean()) / pooled) if pooled > 1e-12 else 0.0
        findings[key] = {
            f"{a_name}_mean": float(a.mean()),
            f"{b_name}_mean": float(b.mean()),
            "cohens_d": d,
            "concern": abs(d) >= 0.5,
        }
    return {
        "available": True,
        "classes": names,
        "findings": findings,
        "any_concern": any(f.get("concern") for f in findings.values()),
        "note": (
            "A |d| >= 0.5 means a trivial image property separates the classes. "
            "The model could exploit it instead of lesion appearance."
        ),
    }


def class_distribution_figure(samples: Sequence[Sample], path: str | Path) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.class_name] += 1
    names = sorted(counts)
    values = [counts[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, values, color=["#4C72B0", "#DD8452", "#55A868"][: len(names)])
    total = sum(values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.01,
                f"{value}\n({value / total:.1%})", ha="center", fontsize=10)
    ax.set_ylabel("Number of images")
    ax.set_title(f"Class distribution (n = {total})")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def dimension_scatter_figure(samples: Sequence[Sample], path: str | Path) -> str:
    """Image dimensions coloured by class -- the size-shortcut check."""
    per_class: Dict[str, List[tuple]] = defaultdict(list)
    for sample in samples:
        image = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            per_class[sample.class_name].append((image.shape[1], image.shape[0]))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (class_name, points), colour in zip(sorted(per_class.items()),
                                            ["#4C72B0", "#DD8452", "#55A868"]):
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=14, alpha=0.5, label=f"{class_name} (n={len(points)})", color=colour)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions by class")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def intensity_histogram_figure(samples: Sequence[Sample], path: str | Path, bins: int = 40) -> str:
    per_class: Dict[str, List[float]] = defaultdict(list)
    for sample in samples:
        image = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            per_class[sample.class_name].append(float(image.mean()))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (class_name, values), colour in zip(sorted(per_class.items()),
                                            ["#4C72B0", "#DD8452", "#55A868"]):
        ax.hist(values, bins=bins, alpha=0.55, label=f"{class_name} (n={len(values)})", color=colour)
    ax.set_xlabel("Mean pixel intensity"); ax.set_ylabel("Number of images")
    ax.set_title("Mean image intensity by class")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def split_composition_figure(split_report: Dict[str, Any], path: str | Path) -> str:
    """Stacked bars of per-class counts in each split."""
    split_names = [s for s in ("train", "val", "test") if s in split_report]
    class_names = sorted({c for s in split_names for c in split_report[s]["per_class"]})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bottom = np.zeros(len(split_names))
    for class_name, colour in zip(class_names, ["#4C72B0", "#DD8452", "#55A868"]):
        values = np.array([split_report[s]["per_class"].get(class_name, 0) for s in split_names],
                          dtype=float)
        ax.bar(split_names, values, bottom=bottom, label=class_name, color=colour)
        bottom += values
    for i, split_name in enumerate(split_names):
        ax.text(i, bottom[i] + max(bottom) * 0.02,
                f"n={split_report[split_name]['n']}", ha="center", fontsize=10)
    ax.set_ylabel("Number of images")
    ax.set_title("Train / validation / test composition")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def sample_grid_figure(samples: Sequence[Sample], path: str | Path,
                       per_class: int = 4, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_name].append(sample)
    class_names = sorted(by_class)
    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(2.4 * per_class, 2.6 * len(class_names)))
    axes = np.atleast_2d(axes)
    for row, class_name in enumerate(class_names):
        pool = by_class[class_name]
        picks = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(picks):
                image = cv2.imread(pool[int(picks[col])].path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    ax.imshow(image, cmap="gray")
            if col == 0:
                ax.set_title(class_name, loc="left", fontsize=11)
    fig.suptitle("Representative samples per class")
    return _save(fig, path)


def augmentation_examples_figure(sample: Sample, config: Dict[str, Any],
                                 path: str | Path, n: int = 5) -> Optional[str]:
    """Show the original next to n augmented variants. None if disabled."""
    from ..augmentation import augment_examples
    from ..preprocessing import preprocess_path

    profile = config["preprocessing"]["profile"]
    original = preprocess_path(sample.path, profile, as_batch=False)
    variants = augment_examples(original, config, n=n, seed=config.get("seed", 42))
    if variants is None:
        return None

    def _viewable(array: np.ndarray) -> np.ndarray:
        low, high = float(array.min()), float(array.max())
        return (array - low) / (high - low) if high - low > 1e-8 else np.zeros_like(array)

    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
    axes[0].imshow(_viewable(original)); axes[0].set_title("Original", fontsize=10); axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(_viewable(variants[i]))
        axes[i + 1].set_title(f"Aug {i + 1}", fontsize=10); axes[i + 1].axis("off")
    fig.suptitle("Training augmentation examples (training split only)")
    return _save(fig, path)
