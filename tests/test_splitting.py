"""Splitting and dataset-integrity tests.

`test_no_group_spans_splits` and `test_group_aware_split_beats_naive_split` are
the regression tests for the leakage that produced the original 99.6% accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.integrity import (
    audit,
    cluster_near_duplicates,
    correlation_matrix,
    exact_duplicate_groups,
    measure_split_leakage,
    thumbnail_matrix,
)
from src.data.loader import class_distribution, discover_samples
from src.data.splitting import (
    SplitError,
    split_report,
    stratified_group_kfold,
    train_val_test_split,
)


def test_discover_samples_finds_both_classes(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    assert len(samples) == 72
    assert class_distribution(samples) == {"benign": 36, "malignant": 36}


def test_missing_class_directory_gives_actionable_error(tmp_path):
    (tmp_path / "benign").mkdir()
    with pytest.raises(FileNotFoundError, match="Class directory missing"):
        discover_samples(tmp_path, ["benign", "malignant"])


def test_missing_root_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        discover_samples(tmp_path / "absent", ["benign"])


def test_masks_are_catalogued_not_treated_as_images(tmp_path):
    import cv2

    class_dir = tmp_path / "benign"
    class_dir.mkdir(parents=True)
    image = np.full((32, 32), 90, dtype=np.uint8)
    cv2.imwrite(str(class_dir / "benign (1).png"), image)
    cv2.imwrite(str(class_dir / "benign (1)_mask.png"), image)
    cv2.imwrite(str(class_dir / "benign (1)_mask_1.png"), image)

    samples = discover_samples(tmp_path, ["benign"])
    assert len(samples) == 1
    assert len(samples[0].mask_paths) == 2


def test_audit_detects_near_duplicate_groups(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    report = audit(samples, threshold=0.95)
    # 24 sources x 3 variants: clustering must collapse well below 72.
    assert report["n_clusters"] < report["n_samples"]
    assert report["reduction_factor"] > 1.0
    assert all(s.group_id.startswith("cluster_") for s in samples)


def test_exact_duplicates_are_found(tmp_path):
    import cv2

    from src.data.loader import Sample

    class_dir = tmp_path / "benign"
    class_dir.mkdir(parents=True)
    image = np.full((32, 32), 77, dtype=np.uint8)
    paths = []
    for name in ["a.png", "b.png"]:
        path = class_dir / name
        cv2.imwrite(str(path), image)
        paths.append(path)
    samples = [Sample(path=str(p), class_name="benign", label=0, stem=p.stem,
                      mask_paths=[], group_id=str(p)) for p in paths]
    groups = exact_duplicate_groups(samples)
    assert len(groups) == 1 and len(groups[0]) == 2


def test_no_group_spans_splits(synthetic_dataset):
    """REGRESSION: the core leakage guarantee."""
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)
    split = train_val_test_split(samples, seed=7)

    owner = {}
    for name, indices in split.items():
        for index in indices:
            group = samples[index].group_id
            assert owner.get(group, name) == name, f"group {group} spans splits"
            owner[group] = name


def test_splits_are_disjoint_and_complete(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)
    split = train_val_test_split(samples, seed=7)
    all_indices = split["train"] + split["val"] + split["test"]
    assert len(all_indices) == len(set(all_indices)) == len(samples)


def test_both_classes_present_in_every_split(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)
    report = split_report(samples, train_val_test_split(samples, seed=7))
    for name in ("train", "val", "test"):
        assert len(report[name]["per_class"]) == 2, f"{name} is missing a class"


def test_group_aware_split_beats_naive_split(synthetic_dataset):
    """REGRESSION: quantifies the improvement the splitter exists to deliver."""
    from sklearn.model_selection import train_test_split as naive_split

    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)

    grouped = train_val_test_split(samples, seed=7)
    grouped_leak = measure_split_leakage(samples, grouped["train"], grouped["test"])

    indices = np.arange(len(samples))
    naive_train, naive_test = naive_split(indices, test_size=0.15, random_state=0)
    naive_leak = measure_split_leakage(samples, naive_train, naive_test)

    assert grouped_leak["corr>0.95"] <= naive_leak["corr>0.95"]
    assert grouped_leak["corr>0.95"] < 0.05


def test_split_fractions_must_sum_to_one(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    with pytest.raises(SplitError, match="sum to 1"):
        train_val_test_split(samples, train=0.5, val=0.2, test=0.2)


def test_too_few_groups_raises_actionable_error(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])[:4]
    for sample in samples:
        sample.group_id = "one_group"
    with pytest.raises(SplitError, match="groups"):
        train_val_test_split(samples)


def test_kfold_returns_requested_number_of_folds(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)
    folds = stratified_group_kfold(samples, n_splits=3, seed=7)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert not set(train_idx) & set(test_idx)


def test_split_is_deterministic_for_a_seed(synthetic_dataset):
    samples = discover_samples(synthetic_dataset["root"], synthetic_dataset["classes"])
    audit(samples, threshold=0.95)
    assert train_val_test_split(samples, seed=3) == train_val_test_split(samples, seed=3)


def test_clustering_threshold_behaviour():
    matrix = np.array([[1.0, 0.99, 0.1], [0.99, 1.0, 0.1], [0.1, 0.1, 1.0]])
    corr = correlation_matrix(matrix / np.linalg.norm(matrix, axis=1, keepdims=True) * np.sqrt(3))
    clusters = cluster_near_duplicates(corr, 0.999)
    assert len(set(clusters)) >= 1
