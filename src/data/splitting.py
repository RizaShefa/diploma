"""Group-aware, stratified train/validation/test splitting.

WHAT CHANGED AND WHY
--------------------
Original implementation (MainTrain.py):

    train_test_split(dataset, label, test_size=0.2, random_state=0)
    ...
    model.fit(..., validation_data=(x_test, y_test))

Three defects:
  1. No `stratify=`, so class proportions drifted between splits.
  2. No grouping, so near-duplicate copies of one lesion landed on both sides.
     Measured leakage: 77.3% of test images had a near-duplicate in train.
  3. The test set was passed as `validation_data`, so early-stopping/model
     selection decisions were made against the very set later reported as the
     unbiased test result.

This module fixes all three. Splitting is performed over GROUPS (near-duplicate
clusters from `integrity.audit`), never over individual images, and produces
three disjoint sets so validation and test serve distinct purposes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .loader import Sample


class SplitError(ValueError):
    """Raised when a requested split cannot be produced."""


def _folds_for_fraction(fraction: float) -> int:
    """Number of folds whose single held-out fold approximates `fraction`."""
    if not 0.0 < fraction < 1.0:
        raise SplitError(f"Split fraction must be in (0,1), got {fraction}.")
    return max(2, int(round(1.0 / fraction)))


def _holdout(
    labels: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve `fraction` of `indices` into a held-out set, respecting groups."""
    n_splits = _folds_for_fraction(fraction)
    unique_groups = len(np.unique(groups[indices]))
    if unique_groups < n_splits:
        raise SplitError(
            f"Cannot create a {fraction:.0%} split: only {unique_groups} distinct "
            f"groups available but {n_splits} folds required. Either lower the "
            "near_duplicate_threshold or supply more data."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_local, hold_local = next(
        splitter.split(np.zeros(len(indices)), labels[indices], groups[indices])
    )
    return indices[keep_local], indices[hold_local]


def train_val_test_split(
    samples: Sequence[Sample],
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Split sample indices into train/val/test with no group spanning splits.

    Returns a dict of index lists. The fractions are approximate because whole
    groups are moved atomically -- exact proportions are reported by
    `split_report` so the thesis can state the realised sizes rather than the
    requested ones.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise SplitError(f"Split fractions must sum to 1.0, got {total}.")

    labels = np.array([s.label for s in samples])
    groups = np.array([s.group_id for s in samples])
    all_idx = np.arange(len(samples))

    remaining, test_idx = _holdout(labels, groups, all_idx, test, seed)
    # val fraction is re-expressed relative to what remains after the test carve
    val_relative = val / (train + val)
    train_idx, val_idx = _holdout(labels, groups, remaining, val_relative, seed + 1)

    result = {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }
    _assert_disjoint_groups(samples, result)
    return result


def _assert_disjoint_groups(samples: Sequence[Sample], split: Dict[str, List[int]]) -> None:
    """Hard guarantee that no group id appears in more than one split."""
    seen: Dict[str, str] = {}
    for name, indices in split.items():
        for idx in indices:
            group = samples[idx].group_id
            previous = seen.get(group)
            if previous is not None and previous != name:
                raise SplitError(
                    f"Group {group!r} appears in both {previous!r} and {name!r}. "
                    "This is the leakage condition the splitter exists to prevent."
                )
            seen[group] = name


def stratified_group_kfold(
    samples: Sequence[Sample], n_splits: int = 5, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Group-aware stratified K-fold, for cross-validation stability analysis."""
    labels = np.array([s.label for s in samples])
    groups = np.array([s.group_id for s in samples])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_local, test_local in splitter.split(np.zeros(len(samples)), labels, groups):
        folds.append((sorted(train_local.tolist()), sorted(test_local.tolist())))
    return folds


def split_report(samples: Sequence[Sample], split: Dict[str, List[int]]) -> Dict[str, object]:
    """Realised sizes and per-class counts for each split (a thesis table)."""
    report: Dict[str, object] = {}
    total = sum(len(v) for v in split.values())
    for name, indices in split.items():
        counts: Dict[str, int] = {}
        for idx in indices:
            cls = samples[idx].class_name
            counts[cls] = counts.get(cls, 0) + 1
        report[name] = {
            "n": len(indices),
            "fraction": round(len(indices) / total, 4) if total else 0.0,
            "per_class": counts,
            "n_groups": len({samples[i].group_id for i in indices}),
        }
    return report
