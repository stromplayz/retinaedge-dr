"""Checkpoint evaluation CLI (contract: ``retinaedge.eval.evaluate``).

CLI:
    python -m retinaedge.eval.evaluate --config <train cfg> --ckpt best.pt [--split val]

Writes ``eval.json`` next to the checkpoint and prints a metrics table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from retinaedge.models.build import build_model
from retinaedge.train.metrics import QWKTracker, jsonable
from retinaedge.train.trainer import build_split_loader, collect_predictions, load_checkpoint
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = ["evaluate", "main"]

_LOGGER = get_logger("eval.evaluate")


def _print_table(metrics: dict, confusion: np.ndarray, threshold_used: float) -> None:
    """Human-readable metrics table on stdout."""
    print("\n=== RetinaEdge-DR evaluation ===")
    print(f"  n={metrics['n']}  temperature={threshold_used:.4f}")
    print(f"  QWK                : {metrics['qwk']:.4f}")
    print(f"  AUC (referable)    : {metrics['auc_refer']:.4f}")
    print(f"  Sensitivity @Youden: {metrics['sens']:.4f} (thr={metrics['threshold']:.4f})")
    print(f"  Specificity @Youden: {metrics['spec']:.4f}")
    print(f"  ECE (referable)    : {metrics['ece']:.4f}")
    print("\nConfusion matrix (rows=true grade 0-4, cols=predicted):")
    for row in confusion:
        print("  " + " ".join(f"{int(v):5d}" for v in row))
    print()


def evaluate(cfg: dict, ckpt: str | Path, split: str = "val") -> dict:
    """Evaluate a checkpoint on one split.

    Args:
        cfg: Training config (embedded ``data:`` section is reused).
        ckpt: Path to ``best.pt``/``last.pt`` (or a raw state_dict file).
        split: One of ``{"train", "val", "test"}``.

    Returns:
        The eval dict that is written to ``eval.json`` next to the checkpoint.
    """
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
    tracker = QWKTracker(num_grades=int(cfg.get("model", {}).get("num_grades", 5)))
    tracker.update(probs, targets)
    metrics = tracker.result()

    preds = probs.argmax(axis=1)
    num_grades = int(cfg.get("model", {}).get("num_grades", 5))
    confusion = np.zeros((num_grades, num_grades), dtype=np.int64)
    for t, p in zip(targets, preds, strict=True):
        confusion[int(t), int(p)] += 1
    per_grade = {}
    for grade in range(num_grades):
        support = int(confusion[grade].sum())
        correct = int(confusion[grade, grade])
        per_grade[str(grade)] = {
            "support": support,
            "recall": (correct / support) if support else None,
        }

    ckpt_path = Path(ckpt)
    result = {
        "ckpt": str(ckpt_path),
        "split": split,
        "temperature": temperature,
        "metrics": jsonable(metrics),
        "confusion_matrix": confusion.tolist(),
        "per_grade": per_grade,
    }
    out_path = ckpt_path.parent / "eval.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _LOGGER.info("wrote %s", out_path)
    _print_table(metrics, confusion, temperature)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.eval.evaluate",
        description="Evaluate a trained checkpoint on a dataset split.",
    )
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "overrides",
        nargs="*",
        help="dotted config overrides, e.g. data.img_size=64 train.epochs=1",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    from retinaedge.utils.config import load_config

    cfg = load_config(args.config, args.overrides)
    try:
        evaluate(cfg, args.ckpt, split=args.split)
    except Exception:
        _LOGGER.exception("evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
