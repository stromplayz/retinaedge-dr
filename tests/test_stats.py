"""Unit tests for retinaedge.eval.stats — statistical rigor for the 97% goal."""

from __future__ import annotations

import math

import numpy as np
import pytest

from retinaedge.eval.stats import (
    accuracy_ci_report,
    bootstrap_accuracy_ci,
    mcnemar_test,
    wilson_ci,
)


class TestWilsonCI:
    def test_known_value(self):
        # 970/1000 correct -> Wilson 95% (0.9575, 0.9789), point 0.97
        lo, hi = wilson_ci(970, 1000)
        assert lo < 0.97 < hi
        assert 0.95 < lo < 0.96
        assert hi > 0.975

    def test_lower_bound_needs_more_n_for_target(self):
        # small n cannot honestly claim 97%
        lo_small, _ = wilson_ci(97, 100)
        lo_big, _ = wilson_ci(9700, 10000)
        assert lo_big > lo_small

    def test_edge_cases(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0 and hi < 0.3
        lo, hi = wilson_ci(10, 10)
        assert lo > 0.7 and hi == pytest.approx(1.0)

    def test_monotone_in_k(self):
        prev = -1.0
        for k in range(0, 101, 10):
            lo, _ = wilson_ci(k, 100)
            assert lo >= prev
            prev = lo


class TestMcNemar:
    def test_no_difference(self):
        r = mcnemar_test(0, 0)
        assert r["p_value"] == 1.0

    def test_symmetric_difference_not_significant(self):
        r = mcnemar_test(10, 10)
        assert r["p_value"] > 0.05

    def test_clear_difference_significant(self):
        r = mcnemar_test(40, 5)
        assert r["p_value"] < 0.001

    def test_exact_path_small_n(self):
        r = mcnemar_test(4, 0)
        assert r["method"] == "exact"
        assert abs(r["p_value"] - 0.125) < 1e-9  # 2 * 0.5^4 * C(4,0)

    def test_chi2_path_large_n(self):
        r = mcnemar_test(100, 20)
        assert r["method"] == "chi2"
        # manual check: stat = (80-1)^2/120
        assert abs(r["statistic"] - (79.0**2) / 120.0) < 1e-9
        assert r["p_value"] < 1e-9


class TestBootstrap:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 3]
        lo, hi = bootstrap_accuracy_ci(y, y, n_boot=200)
        assert lo == 1.0 and hi == 1.0

    def test_known_accuracy(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 5, 500)
        y_pred = y_true.copy()
        flip = rng.random(500) < 0.05
        y_pred[flip] = (y_pred[flip] + 1) % 5
        lo, hi = bootstrap_accuracy_ci(y_true.tolist(), y_pred.tolist(), n_boot=500, seed=0)
        assert 0.85 < lo < 0.99 < 1.0
        assert lo <= hi


class TestAccuracyReport:
    def test_target_reached_requires_lower_bound(self):
        # 97% point estimate with n=1000 does NOT reach honest 97%
        rep = accuracy_ci_report(970, 1000, target=0.97)
        assert rep["accuracy"] == 0.97
        assert rep["target_reached"] is False
        assert rep["point_only_reached"] is True
        assert rep["ci95_lo"] < 0.97

    def test_perfect_large_n_reaches(self):
        rep = accuracy_ci_report(1000, 1000, target=0.97)
        assert rep["target_reached"] is True

    def test_zero_n_raises(self):
        with pytest.raises(ValueError):
            accuracy_ci_report(0, 0)

    def test_math_consistency(self):
        rep = accuracy_ci_report(985, 1000)
        lo, hi = wilson_ci(985, 1000)
        assert math.isclose(rep["ci95_lo"], lo)
        assert math.isclose(rep["ci95_hi"], hi)


if __name__ == "__main__":
    pytest.main([__file__])
