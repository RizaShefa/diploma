"""Dataset integrity: duplicate and near-duplicate detection.

WHY THIS MODULE EXISTS
----------------------
The dataset originally shipped with this project (2,500 "benign" PNGs) was a
pre-augmented derivative, not an original collection. Measured with the routines
below:

    exact byte-duplicate groups            51 (103 files)
    images with a near-twin (corr > 0.95)  2,053 / 2,500  (82.1%)
    distinct source clusters               ~891, not 2,500

Because augmentation had been applied BEFORE splitting, a naive
`train_test_split` placed near-identical copies of the same lesion on both sides:
77.3% of test images had a near-duplicate in train. That is the mechanism behind
the originally reported 99.6% accuracy.

These functions make that measurable rather than assumed, and they produce the
`group_id` values that `splitting.py` uses to keep related images together. The
same audit runs on BUSI, so the thesis can state the dataset's integrity instead
of assuming it.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .loader import Sample

DEFAULT_THUMBNAIL = 32
DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.90, 0.95, 0.97, 0.99)


def file_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming MD5 of a file's bytes (used for exact-duplicate detection)."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_duplicate_groups(samples: Sequence[Sample]) -> List[List[int]]:
    """Indices of samples that are byte-identical, grouped."""
    by_hash: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_hash[file_md5(sample.path)].append(idx)
    return [group for group in by_hash.values() if len(group) > 1]


def thumbnail_matrix(samples: Sequence[Sample], size: int = DEFAULT_THUMBNAIL) -> np.ndarray:
    """Z-scored flattened grayscale thumbnails, one row per sample.

    Z-scoring each row makes the dot product in `correlation_matrix` a Pearson
    correlation, so similarity is invariant to brightness and contrast shifts --
    precisely the changes a contrast/brightness augmentation introduces.
    """
    rows = []
    for sample in samples:
        image = cv2.imread(sample.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            rows.append(np.zeros(size * size, dtype=np.float32))
            continue
        thumb = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        vector = thumb.astype(np.float32).ravel()
        vector -= vector.mean()
        std = float(vector.std())
        rows.append(vector / std if std > 1e-8 else vector)
    return np.stack(rows)


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise correlation of thumbnail rows, diagonal suppressed to -1."""
    corr = (matrix @ matrix.T) / matrix.shape[1]
    np.fill_diagonal(corr, -1.0)
    return corr


def cluster_near_duplicates(corr: np.ndarray, threshold: float) -> List[int]:
    """Union-find clustering of samples whose correlation exceeds `threshold`.

    Returns one cluster id per sample. Samples sharing a cluster id must never be
    split across train/val/test.
    """
    n = corr.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows, cols = np.where(corr > threshold)
    for a, b in zip(rows.tolist(), cols.tolist()):
        union(a, b)

    canonical: Dict[int, int] = {}
    clusters: List[int] = []
    for i in range(n):
        root = find(i)
        if root not in canonical:
            canonical[root] = len(canonical)
        clusters.append(canonical[root])
    return clusters


def audit(
    samples: Sequence[Sample],
    threshold: float = 0.95,
    thumbnail_size: int = DEFAULT_THUMBNAIL,
    *,
    assign_groups: bool = True,
) -> Dict[str, Any]:
    """Run the full integrity audit and return a JSON-serialisable report.

    When `assign_groups` is true (the default) each sample's `group_id` is set to
    its near-duplicate cluster, making downstream splitting leakage-safe by
    construction rather than by convention.
    """
    matrix = thumbnail_matrix(samples, thumbnail_size)
    corr = correlation_matrix(matrix)
    clusters = cluster_near_duplicates(corr, threshold)

    if assign_groups:
        for sample, cluster_id in zip(samples, clusters):
            sample.group_id = f"cluster_{cluster_id}"

    best = corr.max(axis=1)
    exact = exact_duplicate_groups(samples)
    sizes: Dict[int, int] = defaultdict(int)
    for cluster_id in clusters:
        sizes[cluster_id] += 1
    multi = sum(1 for count in sizes.values() if count > 1)

    return {
        "n_samples": len(samples),
        "threshold": threshold,
        "thumbnail_size": thumbnail_size,
        "exact_duplicate_groups": len(exact),
        "exact_duplicate_files": sum(len(group) for group in exact),
        "n_clusters": len(sizes),
        "clusters_with_multiple_members": multi,
        "largest_cluster_size": max(sizes.values()) if sizes else 0,
        "near_duplicate_rate": {
            f"corr>{t}": float((best > t).mean()) for t in DEFAULT_THRESHOLDS
        },
        "reduction_factor": (len(samples) / len(sizes)) if sizes else 1.0,
    }


def measure_split_leakage(
    samples: Sequence[Sample],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    thresholds: Tuple[float, ...] = DEFAULT_THRESHOLDS,
    thumbnail_size: int = DEFAULT_THUMBNAIL,
) -> Dict[str, float]:
    """Fraction of test images having a near-duplicate in train, per threshold.

    This is the metric that exposed the original 77.3% leakage. Running it after
    every split turns "we avoided leakage" from a claim into a measurement, and
    it is asserted in `tests/test_splitting.py`.
    """
    matrix = thumbnail_matrix(samples, thumbnail_size)
    train = matrix[list(train_idx)]
    test = matrix[list(test_idx)]
    if len(train) == 0 or len(test) == 0:
        return {f"corr>{t}": 0.0 for t in thresholds}
    cross = (test @ train.T) / matrix.shape[1]
    best = cross.max(axis=1)
    return {f"corr>{t}": float((best > t).mean()) for t in thresholds}
