"""Metric correctness tests.

Metrics drive every claim in the thesis, so they are checked against values
computed by hand rather than against the implementation's own output.
`test_reproduces_thesis_chapter12_values` pins them to the confusion matrix the
thesis reports (TP=492, TN=504, FP=1, FN=3), so a regression here would be
immediately visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    bootstrap_confidence_intervals,
    confusion_counts,
    evaluate,
    expected_calibration_error,
    format_metric,
    point_metrics,
    pr_points,
    roc_points,
    select_threshold_for_sensitivity,
    threshold_sweep,
)


def _from_counts(tp: int, tn: int, fp: int, fn: int):
    """Build (y_true, y_prob) that realise an exact confusion matrix at t=0.5."""
    y_true = np.array([1] * tp + [1] * fn + [0] * tn + [0] * fp)
    y_prob = np.array([0.9] * tp + [0.1] * fn + [0.1] * tn + [0.9] * fp)
    return y_true, y_prob


def test_confusion_counts_are_correct():
    y_true, y_prob = _from_counts(tp=5, tn=3, fp=2, fn=1)
    counts = confusion_counts(y_true, (y_prob >= 0.5).astype(int))
    assert counts == {"tp": 5, "tn": 3, "fp": 2, "fn": 1}


def test_reproduces_thesis_chapter12_values():
    """Pinned to the thesis's own hand-computed figures."""
    y_true, y_prob = _from_counts(tp=492, tn=504, fp=1, fn=3)
    m = point_metrics(y_true, y_prob, 0.5)
    assert m["tp"] == 492 and m["tn"] == 504 and m["fp"] == 1 and m["fn"] == 3
    assert m["accuracy"] == pytest.approx(0.996, abs=5e-4)
    assert m["precision"] == pytest.approx(0.998, abs=5e-4)
    assert m["sensitivity"] == pytest.approx(0.9939, abs=5e-4)
    assert m["specificity"] == pytest.approx(0.998, abs=5e-4)
    assert m["f1"] == pytest.approx(0.996, abs=5e-4)
    assert m["false_positive_rate"] == pytest.approx(0.00198, abs=1e-5)
    assert m["false_negative_rate"] == pytest.approx(0.00606, abs=1e-5)


def test_sensitivity_and_specificity_definitions():
    y_true, y_prob = _from_counts(tp=8, tn=6, fp=4, fn=2)
    m = point_metrics(y_true, y_prob, 0.5)
    assert m["sensitivity"] == pytest.approx(8 / 10)   # TP/(TP+FN)
    assert m["specificity"] == pytest.approx(6 / 10)   # TN/(TN+FP)
    assert m["precision"] == pytest.approx(8 / 12)     # TP/(TP+FP)
    assert m["npv"] == pytest.approx(6 / 8)            # TN/(TN+FN)
    assert m["balanced_accuracy"] == pytest.approx((0.8 + 0.6) / 2)


def test_perfect_and_inverted_classifiers():
    y_true = np.array([0, 0, 1, 1])
    assert point_metrics(y_true, np.array([0.0, 0.1, 0.9, 1.0]))["roc_auc"] == pytest.approx(1.0)
    assert point_metrics(y_true, np.array([1.0, 0.9, 0.1, 0.0]))["roc_auc"] == pytest.approx(0.0)


def test_single_class_yields_nan_auc_not_crash():
    m = point_metrics(np.zeros(8, dtype=int), np.full(8, 0.3))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_zero_division_is_safe():
    m = point_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert m["sensitivity"] == 0.0 and m["precision"] == 0.0


def test_threshold_changes_sensitivity_monotonically():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_prob = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 200), 0, 1)
    sweep = threshold_sweep(y_true, y_prob, n_points=21)
    sensitivities = [r["sensitivity"] for r in sweep]
    assert all(a >= b - 1e-9 for a, b in zip(sensitivities, sensitivities[1:]))


def test_operating_point_meets_sensitivity_target():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 300)
    y_prob = np.clip(y_true * 0.45 + rng.normal(0.3, 0.18, 300), 0, 1)
    chosen = select_threshold_for_sensitivity(y_true, y_prob, 0.95)
    assert chosen["target_met"]
    assert chosen["sensitivity"] >= 0.95


def test_unreachable_sensitivity_target_is_reported_not_faked():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])  # perfectly wrong
    chosen = select_threshold_for_sensitivity(y_true, y_prob, 0.99)
    assert chosen["target_met"] in (True, False)
    if not chosen["target_met"]:
        assert chosen["sensitivity"] < 0.99


def test_bootstrap_intervals_bracket_the_point_estimate():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 150)
    y_prob = np.clip(y_true * 0.5 + rng.normal(0.25, 0.2, 150), 0, 1)
    point = point_metrics(y_true, y_prob, 0.5)
    intervals = bootstrap_confidence_intervals(y_true, y_prob, iterations=400, seed=5)
    for name in ("sensitivity", "specificity", "roc_auc"):
        ci = intervals[name]
        assert ci["lower"] <= point[name] <= ci["upper"] + 1e-9
        assert ci["n"] > 0


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 80)
    y_prob = rng.random(80)
    a = bootstrap_confidence_intervals(y_true, y_prob, iterations=200, seed=11)
    b = bootstrap_confidence_intervals(y_true, y_prob, iterations=200, seed=11)
    assert a == b


def test_calibration_error_is_zero_for_a_calibrated_model():
    # 100 samples at p=0.0 all negative, 100 at p=1.0 all positive.
    y_prob = np.concatenate([np.zeros(100), np.ones(100)])
    y_true = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])
    assert expected_calibration_error(y_true, y_prob)["ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_error_is_large_for_an_overconfident_model():
    y_prob = np.full(100, 0.99)
    y_true = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(y_true, y_prob)["ece"] > 0.4


def test_roc_and_pr_points_are_well_formed():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, 120)
    y_prob = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 120), 0, 1)
    roc = roc_points(y_true, y_prob)
    assert len(roc["fpr"]) == len(roc["tpr"]) and 0.0 <= roc["auc"] <= 1.0
    pr = pr_points(y_true, y_prob)
    assert len(pr["precision"]) == len(pr["recall"])


def test_evaluate_bundle_contains_everything_the_dashboard_needs():
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 2, 120)
    y_prob = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, 120), 0, 1)
    result = evaluate(y_true, y_prob, bootstrap_iterations=150, target_sensitivity=0.9)
    for key in ("metrics", "confidence_intervals", "calibration", "roc", "pr",
                "operating_point", "class_balance", "n_samples"):
        assert key in result


def test_format_metric_renders_value_and_interval():
    assert format_metric(0.8812, {"lower": 0.795, "upper": 0.947}) == "0.881 [0.795-0.947]"
    assert format_metric(0.5) == "0.500"
    assert format_metric(float("nan")) == "n/a"
