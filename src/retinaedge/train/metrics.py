"""Evaluation metrics (contract: ``retinaedge.train.metrics``).

Provides :class:`QWKTracker`, an accumulator computing quadratic-weighted
Cohen's kappa on argmax grades plus referable-DR operating points
(AUC / sensitivity / specificity at the Youden-optimal threshold) and a 15-bin
expected calibration error on the referable probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve

__all__ = [
    "QWKTracker",
    "expected_calibration_error",
    "ece_on_referable",
    "youden_threshold",
    "jsonable",
]


def _as_float_array(x: Any) -> np.ndarray:
    """Coerce input (np/torch/list) to a 1-D or 2-D float numpy array."""
    if hasattr(x, "detach"):  # torch tensor without importing torch here
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    return arr


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    """Youden-optimal threshold on accumulated scores.

    Args:
        y_true: Binary ground truth (0/1), shape ``(N,)``.
        y_score: Positive-class scores, shape ``(N,)``.

    Returns:
        ``(threshold, sensitivity, specificity)``. When only one class is
        present the metrics are undefined and ``nan`` is returned with a
        neutral threshold of 0.5.
    """
    y_true = y_true.astype(bool)
    if y_true.size == 0 or y_true.all() or not y_true.any():
        return 0.5, float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5, float("nan"), float("nan")
    j_index = tpr[finite] - fpr[finite]
    best = int(np.argmax(j_index))
    thr = float(thresholds[finite][best])
    # Guard against a threshold above every observed score.
    thr = min(thr, float(y_score.max()) + 1e-12)
    tpr_best = float(tpr[finite][best])
    fpr_best = float(fpr[finite][best])
    return thr, tpr_best, 1.0 - fpr_best


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> float:
    """Expected calibration error over equal-width confidence bins.

    Args:
        confidence: Predicted confidence in ``[0, 1]``, shape ``(N,)``.
        correct: Boolean correctness flags, shape ``(N,)``.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        Weighted mean of ``|accuracy - mean confidence|`` over non-empty bins.
    """
    confidence = _as_float_array(confidence).clip(0.0, 1.0)
    correct = _as_float_array(correct).astype(bool)
    if confidence.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = indices == b
        if not mask.any():
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += (mask.sum() / confidence.size) * abs(acc - conf)
    return float(ece)


def ece_on_referable(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE of the referable probability (``P(grade >= 2)`` vs ``grade >= 2``).

    Confidence is ``max(score, 1 - score)`` and correctness is the prediction
    at the default 0.5 threshold — the standard binary ECE definition.
    """
    probs = _as_float_array(probs)
    targets = _as_float_array(targets).astype(np.int64).ravel()
    refer_score = probs[:, 2:].sum(axis=1)
    refer_true = targets >= 2
    conf = np.maximum(refer_score, 1.0 - refer_score)
    correct = (refer_score >= 0.5) == refer_true
    return expected_calibration_error(conf, correct, n_bins=n_bins)


def _quadratic_weighted_kappa(preds: np.ndarray, targets: np.ndarray) -> float:
    """QWK with guards for degenerate (single-class) label sets."""
    labels = np.unique(np.concatenate([preds, targets]))
    if labels.size <= 1:
        return 1.0 if np.array_equal(preds, targets) else 0.0
    try:
        score = cohen_kappa_score(preds, targets, labels=labels, weights="quadratic")
    except ValueError:  # pragma: no cover - defensive
        return float("nan")
    return float(score)


class QWKTracker:
    """Accumulates ``(probs, targets)`` batches and reports DR-grade metrics.

    Usage:
        tracker = QWKTracker()
        for probs, targets in batches:
            tracker.update(probs, targets)
        metrics = tracker.result()  # {qwk, auc_refer, sens, spec, threshold, ece, n}
    """

    def __init__(self, num_grades: int = 5, n_bins: int = 15) -> None:
        self.num_grades = int(num_grades)
        self.n_bins = int(n_bins)
        self._probs: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []

    def reset(self) -> None:
        """Drop all accumulated predictions."""
        self._probs.clear()
        self._targets.clear()

    def update(self, probs: np.ndarray, targets: np.ndarray) -> None:
        """Accumulate one batch.

        Args:
            probs: Grade probabilities ``(B, num_grades)`` (rows need not be
                normalized; argmax and the referable sum are used).
            targets: Ground-truth grades ``(B,)``.
        """
        p = _as_float_array(probs)
        t = _as_float_array(targets).astype(np.int64).ravel()
        if p.ndim != 2 or p.shape[0] != t.shape[0]:
            raise ValueError(f"probs {p.shape} incompatible with targets {t.shape}")
        if p.shape[1] != self.num_grades:
            raise ValueError(f"expected {self.num_grades} grades, got {p.shape[1]}")
        self._probs.append(p.copy())
        self._targets.append(t.copy())

    def result(self) -> dict[str, float | int]:
        """Compute metrics over everything accumulated so far.

        Returns:
            Dict with keys ``qwk`` (quadratic-weighted kappa of argmax grades),
            ``auc_refer``, ``sens``, ``spec`` (referable DR at the
            Youden-optimal ``threshold``), ``ece`` (15-bin ECE on the referable
            probability) and ``n`` (sample count).
        """
        if not self._probs:
            return {
                "qwk": float("nan"),
                "auc_refer": float("nan"),
                "sens": float("nan"),
                "spec": float("nan"),
                "threshold": 0.5,
                "ece": float("nan"),
                "n": 0,
            }
        probs = np.concatenate(self._probs, axis=0)
        targets = np.concatenate(self._targets, axis=0)
        preds = probs.argmax(axis=1)
        qwk = _quadratic_weighted_kappa(preds, targets)

        refer_score = probs[:, 2:].sum(axis=1)
        refer_true = targets >= 2
        if refer_true.all() or not refer_true.any():
            auc = float("nan")
            threshold, sens, spec = 0.5, float("nan"), float("nan")
        else:
            auc = float(roc_auc_score(refer_true.astype(int), refer_score))
            threshold, sens, spec = youden_threshold(refer_true, refer_score)

        ece = ece_on_referable(probs, targets, n_bins=self.n_bins)

        return {
            "qwk": qwk,
            "auc_refer": auc,
            "sens": sens,
            "spec": spec,
            "threshold": threshold,
            "ece": ece,
            "n": int(targets.shape[0]),
        }


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch values and NaN/inf to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if hasattr(obj, "item"):  # numpy / torch scalar wrappers
        try:
            obj = obj.item()
        except (TypeError, ValueError, RuntimeError):  # pragma: no cover
            return str(obj)
    if isinstance(obj, (int, float)):
        return obj if (isinstance(obj, int) or np.isfinite(obj)) else None
    return str(obj)
