"""Fast numerical sanity checks. No MIMIC-IV data is required."""

import numpy as np

from hematrace.metrics import (
    calibration_table,
    classification_metrics,
    regression_metrics,
    sigmoid_np,
    tune_threshold,
)


def test_sigmoid_is_monotonic_and_bounded():
    result = sigmoid_np(np.array([-5.0, 0.0, 5.0]))
    assert np.all(np.diff(result) > 0)
    assert np.all((result >= 0) & (result <= 1))


def test_regression_metric_perfect_predictions():
    y = np.array([7.0, 8.0, 9.0, 10.0])
    m = regression_metrics(y, y.copy())
    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"] == 1.0
    assert m["N"] == len(y)


def test_classification_and_thresholding():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p)
    assert m["Accuracy"] == 1.0
    assert m["AUROC"] == 1.0
    threshold, _ = tune_threshold(y, p)
    assert 0.05 <= threshold <= 0.95
    table = calibration_table(y, p)
    assert table["n"].sum() == len(y)
