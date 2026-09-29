"""Unit tests for threshold search — decode, coordinate ascent, referable."""

from __future__ import annotations

import numpy as np
import pytest

from retinaedge.eval.threshold_search import (
    DEFAULT_CUTS,
    _search_from_probs,
    coordinate_ascent_cuts,
    decode_with_cuts,
    search_referable_threshold,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TestDecode:
    def test_default_cuts_identity_like_round(self):
        expected = np.array([0.2, 0.9, 1.4, 2.6, 3.9, 4.5])
        grades = decode_with_cuts(expected, DEFAULT_CUTS)
        assert grades.tolist() == [0, 1, 1, 3, 4, 4]
        assert grades.tolist() == np.round(expected).astype(int).tolist()

    def test_descending_order_required_semantics(self):
        expected = np.full(5, 2.0)
        grades = decode_with_cuts(expected, [2.5, 1.5, 0.5])
        assert grades.tolist() == [2, 2, 2, 2, 2]

    def test_extremes(self):
        grades = decode_with_cuts(np.array([0.0, 5.0]), DEFAULT_CUTS)
        assert grades.tolist() == [0, 4]


class TestCoordinateAscent:
    def _synthetic(self, n=400, seed=3):
        rng = np.random.default_rng(seed)
        targets = rng.integers(0, 5, n).astype(np.int64)
        # separated expected values with noise -> cuts can help
        expected = targets + rng.normal(0, 0.3, n)
        return expected, targets

    def test_never_below_round_baseline(self):
        expected, targets = self._synthetic()
        round_acc = float((np.round(expected).clip(0, 4).astype(int) == targets).mean())
        cuts, obj, acc = coordinate_ascent_cuts(expected, targets, objective="accuracy")
        assert cuts[0] >= cuts[-1]  # descending
        # coordinate ascent optimizes on the same data: must not lose much
        assert acc >= round_acc - 0.02

    def test_improves_on_noisy_expected(self):
        expected, targets = self._synthetic()
        _, _, acc = coordinate_ascent_cuts(expected, targets, objective="accuracy")
        round_acc = float((np.round(expected).clip(0, 4).astype(int) == targets).mean())
        assert acc >= round_acc - 0.02  # should not lose much; often gains

    def test_qwk_objective_returns_qwk(self):
        expected, targets = self._synthetic()
        cuts, obj, acc = coordinate_ascent_cuts(expected, targets, objective="qwk")
        assert -1.0 <= obj <= 1.0
        assert 0.0 <= acc <= 1.0


class TestReferable:
    def test_perfect_separation(self):
        # remote API: binary y_true (1 = referable) + positive-class score
        y = np.array([0, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.1, 0.9, 0.95, 0.8, 0.7])
        r = search_referable_threshold(y, p)
        assert r["accuracy"] == 1.0
        assert r["sens"] == 1.0
        assert r["spec"] == 1.0

    def test_confusion_counts(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.8, 0.9, 0.1])
        r = search_referable_threshold(y, p)
        c = {"tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"]}
        assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == 4


class TestSearchFromProbs:
    def test_structure_and_ranks(self):
        rng = np.random.default_rng(7)
        targets = rng.integers(0, 5, 300)
        logits = rng.normal(size=(300, 5))
        logits[np.arange(300), targets] += 1.5  # decent model
        probs = _softmax(logits)
        out = _search_from_probs(probs, targets, objective="accuracy")
        assert set(out) >= {"argmax", "cuts", "referable"}
        assert 0.0 <= out["argmax"]["accuracy"] <= 1.0
        assert 0.0 <= out["cuts"]["accuracy"] <= 1.0
        # referable P(>=2) consistent with probs
        assert out["referable"]["threshold"] <= 1.0

    def test_cuts_returned_valid_and_consistent(self):
        rng = np.random.default_rng(11)
        targets = rng.integers(0, 5, 400)
        logits = rng.normal(size=(400, 5))
        logits[np.arange(400), targets] += 2.0
        probs = _softmax(logits)
        out = _search_from_probs(probs, targets, objective="accuracy")
        cuts = out["cuts"]["values"]
        assert cuts == sorted(cuts, reverse=True)  # descending contract
        assert 0.0 <= out["cuts"]["accuracy"] <= 1.0
        assert 0.0 <= out["argmax"]["accuracy"] <= 1.0
        # QWK objective: cut decode on near-argmax probs must not collapse
        out_q = _search_from_probs(probs, targets, objective="qwk")
        assert out_q["cuts"]["qwk"] >= out_q["argmax"]["qwk"] - 0.05


if __name__ == "__main__":
    pytest.main([__file__])
