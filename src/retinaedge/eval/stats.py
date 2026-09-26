"""Statistical rigor helpers (contract: ``retinaedge.eval.stats``).

Accuracy claims need uncertainty. This module provides:

* :func:`wilson_ci` — score confidence interval for a binomial proportion
  (never produces the degenerate [0,1] or negative bounds Wald gives);
* :func:`mcnemar_test` — paired comparison of two models on the same val set;
* :func:`bootstrap_accuracy_ci` — percentile bootstrap for accuracy/QWK-style
  per-sample statistics.

The 97% goal protocol: every headline number is reported as
``acc (95% CI lo-hi, n)``. A model only "reaches" the target when the Wilson
**lower bound** at n >= 1000 clears the target — point estimates on small
validation sets systematically flatter the model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["wilson_ci", "mcnemar_test", "bootstrap_accuracy_ci", "accuracy_ci_report"]


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        k: Number of successes.
        n: Number of trials (``k <= n``).
        z: Two-sided normal quantile (default 1.96 -> 95%).

    Returns:
        ``(lo, hi)`` bounds clipped to ``[0, 1]``. ``n == 0`` gives ``(0, 1)``.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k} n={n}")
    z2 = z * z
    denom = n + z2
    center = (k + 0.5 * z2) / denom
    half = z * math.sqrt((k * (n - k)) / n + 0.25 * z2) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_test(b: int, c: int, exact_threshold: int = 25) -> dict[str, float | str]:
    """McNemar's test on paired discordant predictions.

    Given counts ``b`` (model A right / model B wrong) and ``c`` (A wrong /
    B right), tests whether both models make the same error rate.

    Args:
        b: Discordant count A-correct/B-wrong.
        c: Discordant count A-wrong/B-correct.
        exact_threshold: Use the exact binomial test when ``b + c`` is below
            this (chi-square continuity correction is unreliable for tiny
            discordant counts).

    Returns:
        ``{"method": "chi2"|"exact", "statistic", "p_value"}``.
    """
    if min(b, c) < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return {"method": "chi2", "statistic": 0.0, "p_value": 1.0}
    if n < exact_threshold:
        # Two-sided exact binomial: p = 2 * P(X < min(b, c)) with X ~ Bin(n, 0.5)
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / float(2**n)
        return {"method": "exact", "statistic": float(min(b, c)), "p_value": min(1.0, 2.0 * tail)}
    stat = (abs(b - c) - 1.0) ** 2 / float(n)
    # chi-square survival with 1 dof: p = erfc(sqrt(stat / 2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return {"method": "chi2", "statistic": float(stat), "p_value": float(p)}


def bootstrap_accuracy_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for multi-class accuracy.

    Args:
        y_true: Ground-truth grades.
        y_pred: Predicted grades (same length as ``y_true``).
        n_boot: Resample count (deterministic given ``seed``).
        seed: RNG seed for reproducibility.
        alpha: Significance level (0.05 -> 95% CI).

    Returns:
        ``(lo, hi)`` accuracy bounds.
    """
    t = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(y_pred, dtype=np.int64).ravel()
    if t.shape != p.shape or t.size == 0:
        raise ValueError("y_true/y_pred must be non-empty with equal shapes")
    rng = np.random.default_rng(seed)
    n = t.size
    accs = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        accs[i] = float((t[idx] == p[idx]).mean())
    lo_q, hi_q = 100.0 * (alpha / 2), 100.0 * (1 - alpha / 2)
    return float(np.percentile(accs, lo_q)), float(np.percentile(accs, hi_q))


def accuracy_ci_report(correct: int, n: int, target: float = 0.97) -> dict[str, float | bool]:
    """One-stop report used by the improvement loop.

    Args:
        correct: Correct predictions count.
        n: Validation size.
        target: The accuracy goal (default 0.97 for referable DR).

    Returns:
        Dict with point accuracy, Wilson 95% CI, margin of error and whether
        the **lower bound** clears the target (the honest "goal reached" test).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    lo, hi = wilson_ci(correct, n)
    acc = correct / float(n)
    return {
        "accuracy": acc,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "margin": hi - acc,
        "n": float(n),
        "target": target,
        "target_reached": bool(lo >= target),
        "point_only_reached": bool(acc >= target),
    }
