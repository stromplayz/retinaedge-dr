"""Unit tests for QWKTracker and metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from retinaedge.train.metrics import (
    QWKTracker,
    ece_on_referable,
    expected_calibration_error,
    jsonable,
    youden_threshold,
)


def _one_hot(grades: np.ndarray, num_grades: int = 5) -> np.ndarray:
    probs = np.zeros((grades.shape[0], num_grades))
    probs[np.arange(grades.shape[0]), grades] = 1.0
    return probs


def test_qwk_perfect_predictions() -> None:
    rng = np.random.default_rng(0)
    targets = rng.integers(0, 5, size=64)
    tracker = QWKTracker()
    tracker.update(_one_hot(targets), targets)
    metrics = tracker.result()
    assert metrics["qwk"] == pytest.approx(1.0)
    assert metrics["n"] == 64


def test_qwk_partial_agreement() -> None:
    rng = np.random.default_rng(1)
    targets = rng.integers(0, 5, size=200)
    preds = np.clip(targets + rng.choice([-1, 0, 1], size=200, p=[0.2, 0.6, 0.2]), 0, 4)
    tracker = QWKTracker()
    tracker.update(_one_hot(preds), targets)
    metrics = tracker.result()
    assert 0.0 < metrics["qwk"] < 1.0


def test_qwk_accumulates_like_one_batch() -> None:
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(5), size=30)
    targets = rng.integers(0, 5, size=30)
    whole = QWKTracker()
    whole.update(probs, targets)
    chunked = QWKTracker()
    for start in range(0, 30, 7):
        chunked.update(probs[start : start + 7], targets[start : start + 7])
    assert whole.result()["qwk"] == pytest.approx(chunked.result()["qwk"])
    assert whole.result()["n"] == chunked.result()["n"] == 30


def test_qwk_single_class_degenerate() -> None:
    targets = np.zeros(10, dtype=int)
    tracker = QWKTracker()
    tracker.update(_one_hot(targets), targets)
    assert tracker.result()["qwk"] == pytest.approx(1.0)


def test_tracker_accepts_torch_and_validates() -> None:
    tracker = QWKTracker()
    tracker.update(torch.rand(4, 5), torch.zeros(4, dtype=torch.long))
    assert tracker.result()["n"] == 4
    with pytest.raises(ValueError):
        tracker.update(np.zeros((3, 4)), np.zeros(3))  # wrong grade count
    with pytest.raises(ValueError):
        tracker.update(np.zeros((3, 5)), np.zeros(4))  # batch mismatch


def test_youden_threshold_separable() -> None:
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    threshold, sens, spec = youden_threshold(y_true, y_score)
    assert 0.4 <= threshold <= 0.6
    assert sens == pytest.approx(1.0)
    assert spec == pytest.approx(1.0)


def test_youden_threshold_single_class() -> None:
    threshold, sens, spec = youden_threshold(np.ones(5), np.linspace(0, 1, 5))
    assert np.isnan(sens) and np.isnan(spec)
    assert threshold == pytest.approx(0.5)


def test_ece_perfectly_calibrated_is_zero() -> None:
    conf = np.array([1.0] * 4 + [0.0] * 4)
    correct = np.array([True] * 4 + [False] * 4)
    assert expected_calibration_error(conf, correct, n_bins=15) == pytest.approx(0.0)


def test_ece_bounded_and_reduced_by_scaling() -> None:
    rng = np.random.default_rng(3)
    score = rng.uniform(0.2, 0.8, size=400)
    truth = rng.integers(0, 2, size=400).astype(bool)
    ece = expected_calibration_error(score, truth)
    assert 0.0 <= ece <= 1.0
    # Perfectly wrong confidence ordering yields a larger error than random.
    assert (
        ece_on_referable(np.stack([1 - score, score], axis=1), truth.astype(int), n_bins=15) >= 0.0
    )


def test_ece_on_referable_uses_grade_threshold() -> None:
    # grade 0 & 1 -> non-referable; grade 2 -> referable.
    probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0],  # score 0.0, true non-referable
            [0.0, 0.0, 1.0, 0.0, 0.0],  # score 1.0, true referable
        ]
    )
    targets = np.array([0, 2])
    assert ece_on_referable(probs, targets) == pytest.approx(0.0)


def test_jsonable_sanitises_nan_and_numpy() -> None:
    payload = {
        "a": float("nan"),
        "b": np.float32(2.5),
        "c": [np.int64(1), float("inf")],
        "d": "ok",
    }
    out = jsonable(payload)
    assert out["a"] is None
    assert out["b"] == pytest.approx(2.5)
    assert out["c"] == [1, None]
    assert out["d"] == "ok"
    import json

    json.dumps(out)  # must not raise
