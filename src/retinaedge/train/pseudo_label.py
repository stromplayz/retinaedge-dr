"""Pseudo-labeling for self-training (contract: ``retinaedge.train.pseudo_label``).

Semi-supervised "loophole": label unlabeled fundus images with the current
best model, keep only the *confident and self-consistent* ones, and fold them
back into training as extra supervision. Confidence alone is not enough — we
require agreement between the two decoders of the same model:

* ``max_j p_j >= tau`` (the model is sure), and
* ``hard_grade(p) == round(E[Y])`` (argmax and the ordinal mean agree), and
* the auxiliary referable head agrees with the ordinal decode
  (``referable_prob(p) >= 0.5`` iff ``grade >= 2``).

Triple agreement filters most of the noise that normally caps self-training
gains. The emitted ``labels_pseudo.csv`` uses the standard ``image,grade``
layout, so the ``folder``/``aptos`` dataset readers ingest it unchanged (a
distinct ``root`` keeps pseudo data separate from labeled data).

CLI:
    python -m retinaedge.train.pseudo_label --config <cfg> --ckpt best.pt \
        --images-dir data/unlabeled/images --out data/pseudo/labels_pseudo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from retinaedge.models.build import build_model
from retinaedge.train.trainer import load_checkpoint
from retinaedge.utils.logging_utils import get_logger

__all__ = ["PseudoImageDataset", "generate_pseudo_labels", "main"]

_LOGGER = get_logger("train.pseudo_label")

_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(root: Path, max_images: int | None = None) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTENSIONS)
    if max_images is not None:
        files = files[: int(max_images)]
    return files


class PseudoImageDataset(Dataset):
    """Unlabeled image folder returning ``(tensor, index)`` pairs."""

    def __init__(
        self,
        files: Sequence[Path],
        transform,
        use_ben_graham: bool = False,
        ben_graham_radius: int = 300,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self.files = list(files)
        self.transform = transform
        self.use_ben_graham = bool(use_ben_graham)
        self.radius = int(ben_graham_radius)

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, path: Path) -> np.ndarray:
        bgr = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"unreadable image: {path}")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        if self.use_ben_graham:
            from retinaedge.data.ben_graham import preprocess_ben_graham

            rgb = preprocess_ben_graham(rgb, radius=self.radius)
        return rgb

    def __getitem__(self, idx: int):
        tensor = self.transform(image=self._load(self.files[idx]))["image"]
        return tensor, idx


def generate_pseudo_labels(
    model,
    files: Sequence[Path],
    transform,
    tau: float = 0.92,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    use_ben_graham: bool = False,
    ben_graham_radius: int = 300,
) -> tuple[list[dict[str, int | float | str]], dict]:
    """Label unlabeled images, keeping confident + self-consistent predictions.

    Args:
        model: Module exposing ``predict_probs(imgs) -> (B, K)``.
        files: Image paths (relative names are preserved in the CSV).
        transform: Albumentations val transform (Resize/CenterCrop/Normalize).
        tau: Confidence threshold on ``max_j p_j``.
        device: Torch device string.
        batch_size: Inference batch size.
        num_workers: DataLoader workers.
        use_ben_graham: Apply Ben-Graham preprocess at load time.
        ben_graham_radius: Crop radius for Ben-Graham preprocess.

    Returns:
        ``(rows, stats)`` where rows are ``{"image", "grade", "confidence"}``
        dicts sorted by descending confidence, and stats reports filter yield.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0.0, 1.0], got {tau}")
    if not files:
        return [], {"n_total": 0, "n_kept": 0, "yield": 0.0, "grade_counts": {}}

    was_training = model.training
    model.eval()
    ds = PseudoImageDataset(files, transform, use_ben_graham, ben_graham_radius)
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=num_workers)

    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, _idx in loader:
            probs = model.predict_probs(imgs.to(device, non_blocking=True))
            probs_all.append(probs.float().cpu().numpy())
    probs_np = np.concatenate(probs_all, axis=0)

    k = probs_np.shape[1]
    expected = probs_np @ np.arange(k, dtype=np.float64)
    hard = probs_np.argmax(axis=1)
    refer = probs_np[:, 2:].sum(axis=1)
    max_conf = probs_np.max(axis=1)

    keep_mask = (
        (max_conf >= float(tau))
        & (hard == np.round(expected).astype(np.int64))
        & ((refer >= 0.5) == (hard >= 2))
    )

    rows: list[dict[str, int | float | str]] = []
    for idx in np.flatnonzero(keep_mask):
        rows.append(
            {
                "image": str(files[int(idx)].name),
                "grade": int(hard[idx]),
                "confidence": float(max_conf[idx]),
            }
        )
    rows.sort(key=lambda r: -float(r["confidence"]))

    grade_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["grade"])
        grade_counts[key] = grade_counts.get(key, 0) + 1
    stats = {
        "n_total": int(len(files)),
        "n_kept": int(len(rows)),
        "yield": float(len(rows) / max(len(files), 1)),
        "tau": float(tau),
        "grade_counts": grade_counts,
        "mean_confidence": float(np.mean([r["confidence"] for r in rows])) if rows else 0.0,
    }
    if was_training:
        model.train()
    return rows, stats


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.train.pseudo_label",
        description="Pseudo-label unlabeled fundus images with a trained checkpoint.",
    )
    parser.add_argument("--config", required=True, help="training YAML config (transforms/model)")
    parser.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    parser.add_argument("--images-dir", required=True, help="folder with unlabeled images")
    parser.add_argument("--out", required=True, help="output CSV path (image,grade)")
    parser.add_argument("--stats-out", default=None, help="optional stats JSON path")
    parser.add_argument("--tau", type=float, default=0.92, help="confidence threshold")
    parser.add_argument("--max-images", type=int, default=None)
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
        from retinaedge.data.dataset import build_train_val_transforms

        device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = load_checkpoint(args.ckpt, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        temperature = float(payload.get("temperature", 1.0)) if isinstance(payload, dict) else 1.0
        model = build_model(cfg).to(device)
        model.load_state_dict(state)
        model.set_temperature(temperature)

        _, val_tf = build_train_val_transforms(cfg)
        files = _list_images(Path(args.images_dir), max_images=args.max_images)
        rows, stats = generate_pseudo_labels(
            model,
            files,
            val_tf,
            tau=args.tau,
            device=device,
            batch_size=int(cfg.get("eval", {}).get("batch_size", 32)),
            use_ben_graham=bool(cfg.get("data", {}).get("ben_graham", False)),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "grade"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"image": row["image"], "grade": row["grade"]})
        if args.stats_out:
            Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        _LOGGER.info(
            "pseudo-labels: %d/%d kept (yield %.1f%%) -> %s",
            stats["n_kept"],
            stats["n_total"],
            100.0 * stats["yield"],
            out_path,
        )
    except Exception:
        _LOGGER.exception("pseudo-labeling failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
