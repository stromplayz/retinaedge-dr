"""Decision-threshold optimization (contract: ``retinaedge.eval.threshold_search``).

Two "loopholes" that convert a fixed model into a better *clinical decision
rule* at zero training cost:

1. **Ordinal cut-point search.** The probability decoder defaults to
   ``argmax``; clinically we decode the expected grade ``E[Y] = sum_j j*p_j``
   through ``K-1`` ordered cut points ``c_0 > c_1 > ... > c_{K-2}`` and count
   how many the score exceeds. Coordinate ascent on the validation set finds
   cuts that maximize QWK / accuracy — typically worth +1-3 QWK points on
   imbalanced DR data because grade boundaries absorb class-prior skew.
2. **Referable-DR threshold search.** The binary "refer if P(grade>=2) > t"
   operating point is chosen to maximize *accuracy* (with sensitivity /
   specificity reported at that point, plus the Youden optimum for
   comparison). This is the number the 97% goal is defined against.

Thresholds are fitted on validation data only and stored as JSON; the test
set is never touched. Unbiased evaluation then applies the frozen thresholds.

CLI:
    python -m retinaedge.eval.threshold_search --config <cfg> --ckpt best.pt \
        [--split val] [--objective qwk] [--out thresholds.json] [overrides...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from retinaedge.train.metrics import _quadratic_weighted_kappa
from retinaedge.train.trainer import build_split_loader, collect_predictions, load_checkpoint
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = [
    "decode_with_cuts",
    "coordinate_ascent_cuts",
    "search_referable_threshold",
    "run_threshold_search",
]

_LOGGER = get_logger("eval.threshold_search")

#: Default cut points: midpoints between adjacent grades on the expected-grade scale.
DEFAULT_CUTS = (0.5, 1.5, 2.5, 3.5)


def decode_with_cuts(expected: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    """Decode expected grades into integer grades via ordered cut points.

    Args:
        expected: Continuous grade scores, shape ``(N,)`` (e.g. ``E[Y]``).
        cuts: ``K-1`` cut points; a sample's grade is the number of cuts it
            exceeds. Order is normalized internally (descending).

    Returns:
        Integer grades in ``[0, len(cuts)]``.
    """
    ordered = sorted((float(c) for c in cuts), reverse=True)
    e = np.asarray(expected, dtype=np.float64).ravel()
    grades = np.zeros(e.shape[0], dtype=np.int64)
    for cut in ordered:
        grades += (e > cut).astype(np.int64)
    return grades


def _accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean()) if targets.size else float("nan")


def coordinate_ascent_cuts(
    expected: np.ndarray,
    targets: np.ndarray,
    init: Sequence[float] = DEFAULT_CUTS,
    objective: str = "qwk",
    grid_radius: float = 0.75,
    grid_step: float = 0.05,
    passes: int = 3,
) -> tuple[list[float], float, float]:
    """Optimize the ``K-1`` cut points by coordinate ascent.

    Args:
        expected: Continuous scores, shape ``(N,)``.
        targets: Ground-truth integer grades, shape ``(N,)``.
        init: Starting cut points.
        objective: ``"qwk"`` (quadratic-weighted kappa) or ``"accuracy"``.
        grid_radius: Search each cut within ``+-grid_radius`` of its current value.
        grid_step: Grid resolution.
        passes: Number of sweeps over all cuts.

    Returns:
        ``(cuts_desc, best_qwk, best_accuracy)`` with cuts sorted descending.
    """
    e = np.asarray(expected, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.int64).ravel()
    cuts = sorted((float(c) for c in init), reverse=True)

    def score(cs: Sequence[float]) -> tuple[float, float]:
        preds = decode_with_cuts(e, cs)
        if objective == "accuracy":
            return _accuracy(preds, t), _quadratic_weighted_kappa(preds, t)
        return _quadratic_weighted_kappa(preds, t), _accuracy(preds, t)

    best_primary, best_secondary = score(cuts)
    grid = np.arange(-grid_radius, grid_radius + 1e-9, grid_step)
    for _ in range(max(1, int(passes))):
        improved = False
        for i, cut in enumerate(cuts):
            for delta in grid:
                cand = list(cuts)
                cand[i] = float(cut + delta)
                primary, secondary = score(cand)
                if primary > best_primary + 1e-12 or (
                    abs(primary - best_primary) <= 1e-12 and secondary > best_secondary + 1e-12
                ):
                    cuts, best_primary, best_secondary = cand, primary, secondary
                    improved = True
        if not improved:
            break
    cuts = sorted(cuts, reverse=True)
    preds = decode_with_cuts(e, cuts)
    return cuts, _quadratic_weighted_kappa(preds, t), _accuracy(preds, t)


def search_referable_threshold(
    y_true: np.ndarray,
    p_refer: np.ndarray,
) -> dict[str, float]:
    """Search the referable-DR threshold maximizing accuracy.

    Also reports sensitivity/specificity at the accuracy-optimal point and at
    the Youden optimum, so the clinical trade-off stays visible.

    Args:
        y_true: Binary ground truth (1 = referable), shape ``(N,)``.
        p_refer: Positive-class probability, shape ``(N,)``.

    Returns:
        Dict with ``threshold`` (max-accuracy), ``accuracy``, ``sens``,
        ``spec``, ``youden_threshold``, ``youden_j``, and the confusion counts
        ``tp/fp/tn/fn`` at the returned threshold.
    """
    y = np.asarray(y_true).astype(bool).ravel()
    p = np.asarray(p_refer, dtype=np.float64).ravel()
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true/p_refer must be non-empty with equal shapes")
    if y.all() or not y.any():
        raise ValueError("search needs both classes present in the validation split")

    order = np.argsort(p)
    ps = p[order]
    # Candidate thresholds: midpoints between consecutive unique scores.
    uniq = np.unique(ps)
    if uniq.size > 1:
        cands = (uniq[:-1] + uniq[1:]) / 2.0
        cands = np.concatenate([[uniq[0] - 1e-6], cands, [uniq[-1] + 1e-6]])
    else:  # pragma: no cover - degenerate single-score val set
        cands = np.array([0.5])

    def stats(thr: float) -> tuple[float, float, float, int, int, int, int]:
        pred = p > thr
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        tn = int((~pred & ~y).sum())
        fn = int((~pred & y).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        return (tp + tn) / y.size, sens, spec, tp, fp, tn, fn

    best = max(cands, key=lambda thr: stats(thr)[0])
    acc, sens, spec, tp, fp, tn, fn = stats(best)
    # Youden optimum for comparison (J = sens + spec - 1)
    youden_thr, j_best = 0.5, -1.0
    for thr in cands:
        _, s, sp, *_ = stats(thr)
        if s + sp - 1.0 > j_best:
            j_best, youden_thr = s + sp - 1.0, float(thr)
    return {
        "threshold": float(best),
        "accuracy": float(acc),
        "sens": float(sens),
        "spec": float(spec),
        "youden_threshold": float(youden_thr),
        "youden_j": float(j_best),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "n": float(y.size),
    }


def run_threshold_search(cfg: dict, ckpt: str, split: str = "val", objective: str = "qwk") -> dict:
    """Full threshold search over a split; returns the JSON-able result dict."""
    from retinaedge.models.build import build_model

    seed_everything(int(cfg.get("train", {}).get("seed", 0)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = load_checkpoint(ckpt, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    temperature = float(payload.get("temperature", 1.0)) if isinstance(payload, dict) else 1.0
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.set_temperature(temperature)

    loader = build_split_loader(cfg, split, shuffle=False)
    probs, targets = collect_predictions(model, loader, device)
    return _search_from_probs(probs, targets, objective=objective)


def _search_from_probs(probs: np.ndarray, targets: np.ndarray, objective: str = "qwk") -> dict:
    """Pure-numpy threshold search shared by the CLI and tests."""
    k = probs.shape[1]
    grades_idx = np.arange(k, dtype=np.float64)
    expected = probs @ grades_idx
    argmax_preds = probs.argmax(axis=1)

    cuts, qwk_cuts, acc_cuts = coordinate_ascent_cuts(expected, targets, objective=objective)
    baseline_qwk = _quadratic_weighted_kappa(argmax_preds, targets)
    baseline_acc = _accuracy(argmax_preds, targets)

    refer_score = probs[:, 2:].sum(axis=1)
    refer_true = (targets >= 2).astype(int)
    refer = search_referable_threshold(refer_true, refer_score)

    return {
        "objective": objective,
        "n": int(targets.size),
        "argmax": {"qwk": float(baseline_qwk), "accuracy": float(baseline_acc)},
        "cuts": {
            "values": [float(c) for c in cuts],
            "qwk": float(qwk_cuts),
            "accuracy": float(acc_cuts),
        },
        "referable": refer,
    }


def _print_table(result: dict) -> None:
    """Human-readable summary on stdout."""
    print("\n=== Threshold search ===")
    print(f"  n={result['n']}  objective={result['objective']}")
    print(
        f"  argmax decode     : QWK {result['argmax']['qwk']:.4f}  acc {result['argmax']['accuracy']:.4f}"
    )
    print(
        f"  cut-point decode  : QWK {result['cuts']['qwk']:.4f}  acc {result['cuts']['accuracy']:.4f}"
    )
    print(f"  cuts              : {[round(c, 3) for c in result['cuts']['values']]}")
    r = result["referable"]
    print(
        f"  referable @acc-opt: acc {r['accuracy']:.4f}  sens {r['sens']:.4f}  spec {r['spec']:.4f}  thr {r['threshold']:.4f}"
    )
    print(f"  referable @Youden : J {r['youden_j']:.4f}  thr {r['youden_threshold']:.4f}")
    print()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.eval.threshold_search",
        description="Fit ordinal cut points and the referable-DR threshold on a split.",
    )
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--objective", default="qwk", choices=["qwk", "accuracy"])
    parser.add_argument(
        "--out", default=None, help="output JSON (default: thresholds.json next to ckpt)"
    )
    parser.add_argument(
        "overrides", nargs="*", help="dotted config overrides, e.g. data.img_size=160"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    from retinaedge.utils.config import load_config

    cfg = load_config(args.config, args.overrides)
    try:
        result = run_threshold_search(cfg, args.ckpt, split=args.split, objective=args.objective)
    except Exception:
        _LOGGER.exception("threshold search failed")
        return 1
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _LOGGER.info("wrote %s", out_path)
    _print_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
