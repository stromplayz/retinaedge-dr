"""Training loop and CLI (contract: ``retinaedge.train.trainer``).

CLI:
    python -m retinaedge.train.trainer --config configs/train/smoke.yaml [dotted.overrides...]

Artifacts written to ``cfg["train"]["save_dir"]``:
    ``best.pt``  — payload ``{"state_dict", "cfg", "val_qwk", "temperature", "epoch"}``
    ``last.pt``  — same payload plus optimizer/scheduler state for manual resume
    ``metrics.json`` / ``history.csv`` — run summary and per-epoch log

Features: AdamW + cosine schedule with linear warmup, grad clipping, opt-in
AMP, opt-in EMA, opt-in class-balanced sampler, early stopping on val QWK.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from retinaedge.models.build import DrNet, build_model
from retinaedge.models.loss import DrLoss, build_loss
from retinaedge.models.ordinal_ops import ordinal_probs
from retinaedge.train.metrics import QWKTracker, jsonable
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = [
    "train",
    "main",
    "build_loaders",
    "build_split_loader",
    "evaluate_loader",
    "collect_predictions",
    "load_checkpoint",
]

_LOGGER = get_logger("train.trainer")

_HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_loss_ordinal",
    "train_loss_refer",
    "val_loss",
    "val_qwk",
    "val_auc_refer",
    "val_sens",
    "val_spec",
    "val_threshold",
    "val_ece",
    "is_best",
    "elapsed_s",
]


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


class _BootstrapSynthetic(torch.utils.data.Dataset):
    """TEMPORARY procedural dataset used only when ``retinaedge.data`` (agent 2-a)
    is not installed yet. Mirrors the interface-contract synthetic spec:
    class distribution ~[0.45, 0.20, 0.15, 0.10, 0.10], per-split rng seeded
    with ``seed + split_offset``, ``__getitem__`` -> (normalized CHW tensor, int).
    Remove once the real data module lands.
    """

    _SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}
    _DISTRIBUTION = (0.45, 0.20, 0.15, 0.10, 0.10)

    def __init__(self, n_samples: int, size: int, seed: int, split: str, train: bool) -> None:
        self.size = int(size)
        self.train = bool(train)
        self.split_seed = int(seed) + self._SPLIT_OFFSET[split]
        rng = np.random.default_rng(self.split_seed)
        self.labels: np.ndarray = rng.choice(5, size=int(n_samples), p=self._DISTRIBUTION)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        rng = np.random.default_rng([self.split_seed, int(idx)])
        grade = int(self.labels[idx])
        # Grade-dependent luminance so the smoke run is actually learnable.
        base = 30.0 + 45.0 * grade
        img = rng.normal(base, 28.0, (self.size, self.size, 3)) + rng.uniform(-10, 10, 3)
        img = img.clip(0.0, 255.0).astype(np.uint8)
        if self.train:
            if torch.rand(()) < 0.5:
                img = img[:, ::-1, :]
            if torch.rand(()) < 0.5:
                img = img[::-1, :, :]
        tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std, grade


def _import_dataset_module() -> Any | None:
    """Import the (agent 2-a) dataset module, returning None when unavailable."""
    try:
        from retinaedge.data import dataset as ds_module

        return ds_module
    except ImportError:
        return None


def _build_split_dataset(cfg: dict, split: str):
    """Resolve one split's dataset via the real data module or the bootstrap shim."""
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("dataset", "synthetic"))
    ds_module = _import_dataset_module()
    if ds_module is not None:
        train_tf, val_tf = ds_module.build_train_val_transforms(cfg)
        transform = train_tf if split == "train" else val_tf
        return ds_module.build_dataset(cfg, split, transform=transform), name
    if name == "synthetic":
        _LOGGER.warning(
            "retinaedge.data.dataset not installed — using the internal bootstrap "
            "synthetic shim (plain resize/flip/normalize, no CLAHE/augmentation)."
        )
        syn = data_cfg.get("synthetic", {})
        n_key = {"train": "n_train", "val": "n_val", "test": "n_test"}.get(split, "n_val")
        ds = _BootstrapSynthetic(
            n_samples=int(syn.get(n_key, 80)),
            size=int(syn.get("size", data_cfg.get("img_size", 64))),
            seed=int(syn.get("seed", 0)),
            split=split,
            train=(split == "train"),
        )
        return ds, name
    raise RuntimeError(
        "retinaedge.data.dataset is not available and there is no bootstrap "
        f"implementation for dataset={name!r}. Install the data module "
        "(agent 2-a) or use dataset: synthetic."
    )


def build_split_loader(cfg: dict, split: str, shuffle: bool = False) -> DataLoader:
    """Build a loader for a single split (``train``/``val``/``test``).

    Used by the trainer (train/val) and by ``retinaedge.eval`` for any split.
    """
    data_cfg = cfg.get("data", {})
    ds, name = _build_split_dataset(cfg, split)
    num_workers = int(data_cfg.get("num_workers", 0))
    batch_size = int(data_cfg.get("batch_size", 32))
    if split != "train":
        batch_size = int(cfg.get("eval", {}).get("batch_size", batch_size))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    _LOGGER.info("loader ready: split=%s n=%d dataset=%s", split, len(ds), name)
    return loader


def build_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train/val loaders from the embedded ``data:`` config section.

    Prefers ``retinaedge.data.dataset`` (contract owners). Falls back to an
    internal procedural synthetic dataset only for ``dataset: synthetic`` when
    that module is not installed yet (bootstrap shim, logged as a warning).

    Returns:
        ``(train_loader, val_loader)``.
    """
    train_loader = build_split_loader(cfg, "train", shuffle=True)
    if bool(cfg.get("train", {}).get("sampler", False)):
        sampler = _make_weighted_sampler(
            train_loader.dataset, cfg.get("model", {}).get("num_grades", 5)
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            persistent_workers=train_loader.persistent_workers,
        )
        _LOGGER.info("using class-balanced WeightedRandomSampler")
    val_loader = build_split_loader(cfg, "val", shuffle=False)
    return train_loader, val_loader


def _dataset_labels(dataset: Any) -> np.ndarray:
    """Extract integer labels from a dataset (attribute first, iteration fallback)."""
    for attr in ("labels", "targets"):
        if hasattr(dataset, attr):
            arr = np.asarray(getattr(dataset, attr)).astype(np.int64).ravel()
            if arr.shape[0] == len(dataset):
                return arr
    _LOGGER.info("inferring sampler labels by iterating the train dataset once")
    return np.asarray([int(y) for _, y in dataset], dtype=np.int64)


def _make_weighted_sampler(dataset: Any, num_grades: int) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-frequency class-balanced sampler over the dataset labels."""
    labels = _dataset_labels(dataset)
    counts = np.bincount(labels, minlength=int(num_grades)).astype(np.float64)
    class_weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = torch.from_numpy(class_weights[labels]).double()
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class _Ema:
    """Exponential moving average of model weights (including float buffers)."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, Tensor] = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow


# ---------------------------------------------------------------------------
# Evaluation helpers (shared with retinaedge.eval)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_loader(
    model: DrNet,
    loader: DataLoader,
    device: str,
    loss_fn: DrLoss | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference over a loader (single forward per batch).

    Args:
        model: Model already on ``device``; probabilities are computed from the
            ordinal logits scaled by the model's stored temperature.
        loader: Loader yielding ``(imgs, targets)``.
        device: Torch device string.
        loss_fn: When given, also accumulates the (untemperatured) loss.

    Returns:
        ``(probs (N, K) float64 numpy, targets (N,) int64 numpy, mean_loss)``
        with ``mean_loss = nan`` when ``loss_fn`` is None.
    """
    was_training = model.training
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    loss_sum, loss_n = 0.0, 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(imgs)
        probs = ordinal_probs(outputs["ordinal_logits"] / model.temperature)
        all_probs.append(probs.double().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        if loss_fn is not None:
            loss, _ = loss_fn(outputs, targets)
            loss_sum += float(loss) * imgs.shape[0]
            loss_n += imgs.shape[0]
    if was_training:
        model.train()
    probs_np = np.concatenate(all_probs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0).astype(np.int64)
    mean_loss = loss_sum / max(loss_n, 1) if loss_fn is not None else float("nan")
    return probs_np, targets_np, mean_loss


def collect_predictions(
    model: DrNet, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning only ``(probs, targets)`` for a split."""
    probs, targets, _ = evaluate_loader(model, loader, device, loss_fn=None)
    return probs, targets


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def _make_lr_lambda(warmup_epochs: float, total_steps: int, steps_per_epoch: int):
    """Linear warmup followed by cosine decay to zero (per optimizer step)."""
    warmup_steps = max(1, int(round(warmup_epochs * steps_per_epoch)))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a trainer checkpoint, preferring the safe ``weights_only`` path."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:  # noqa: BLE001 - legacy/extra payloads may need full pickle
        return torch.load(path, map_location=map_location, weights_only=False)


def _save_checkpoint(
    path: Path,
    model: DrNet,
    cfg: dict,
    val_qwk: float,
    epoch: int,
    state_dict: dict[str, Tensor] | None = None,
    extra: dict | None = None,
) -> None:
    payload: dict[str, Any] = {
        "state_dict": state_dict or {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "cfg": cfg,
        "val_qwk": float(val_qwk),
        "temperature": float(model.temperature),
        "epoch": int(epoch),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: dict, device: str = "auto") -> dict:
    """Run a full training pass; artifacts land in ``cfg["train"]["save_dir"]``.

    Args:
        cfg: Full training config (YAML dict, ``data:`` section embedded).
        device: ``"auto"`` (cuda when available), or ``"cpu"``/``"cuda"``.

    Returns:
        Summary dict with best epoch/QWK, epochs run and artifact paths.
    """
    train_cfg = cfg.get("train", {})
    seed_everything(int(train_cfg.get("seed", 0)))
    device = _resolve_device(device)
    use_amp = bool(train_cfg.get("amp", False)) and device == "cuda"
    if bool(train_cfg.get("amp", False)) and device != "cuda":
        _LOGGER.warning("amp requested but device is %s — running in fp32", device)
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16

    save_dir = Path(str(train_cfg.get("save_dir", "artifacts/run")))
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(train_cfg.get("epochs", 10))
    patience = int(train_cfg.get("patience", 10))

    train_loader, val_loader = build_loaders(cfg)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _LOGGER.info(
        "model=%s embed_dim=%d params=%.2fM device=%s",
        model.backbone_name,
        model.embed_dim,
        n_params / 1e6,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _make_lr_lambda(float(train_cfg.get("warmup_epochs", 0.0)), total_steps, steps_per_epoch),
    )
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    ema = None
    if bool(train_cfg.get("ema", False)):
        ema = _Ema(model, float(train_cfg.get("ema_decay", 0.999)))
        _LOGGER.info("EMA enabled (decay=%.4f)", ema.decay)

    tracker = QWKTracker(num_grades=int(cfg.get("model", {}).get("num_grades", 5)))
    history: list[dict[str, Any]] = []
    best_qwk = -math.inf
    best_epoch = -1
    stale_epochs = 0
    epoch = 0
    val_qwk = float("nan")
    val_metrics: dict[str, float | int] = tracker.result()
    run_t0 = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        sums = OrderedDict(loss=0.0, loss_ordinal=0.0, loss_refer=0.0)
        batches = 0
        for imgs, targets in train_loader:
            imgs, targets = (
                imgs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                outputs = model(imgs)
                loss, parts = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            for key in sums:
                sums[key] += parts[key]
            batches += 1

        train_parts = {key: value / max(batches, 1) for key, value in sums.items()}
        lr_now = float(optimizer.param_groups[0]["lr"])

        if ema is not None:
            backup = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict())
        tracker.reset()
        probs, targets, val_loss = evaluate_loader(model, val_loader, device, loss_fn)
        tracker.update(probs, targets)
        val_metrics = tracker.result()
        if ema is not None:
            model.load_state_dict(backup)

        val_qwk = float(val_metrics["qwk"])
        is_best = val_qwk > best_qwk
        if is_best:
            best_qwk, best_epoch, stale_epochs = val_qwk, epoch, 0
            state_source = ema.state_dict() if ema is not None else model.state_dict()
            _save_checkpoint(
                save_dir / "best.pt",
                model,
                cfg,
                val_qwk,
                epoch,
                state_dict={k: v.detach().cpu().clone() for k, v in state_source.items()},
            )
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_parts["loss"],
            "train_loss_ordinal": train_parts["loss_ordinal"],
            "train_loss_refer": train_parts["loss_refer"],
            "val_loss": val_loss,
            "val_qwk": val_qwk,
            "val_auc_refer": val_metrics["auc_refer"],
            "val_sens": val_metrics["sens"],
            "val_spec": val_metrics["spec"],
            "val_threshold": val_metrics["threshold"],
            "val_ece": val_metrics["ece"],
            "is_best": is_best,
            "elapsed_s": time.time() - epoch_t0,
        }
        history.append(row)
        _LOGGER.info(
            "epoch %03d | lr %.2e | train %.4f (ord %.4f ref %.4f) | val %.4f | "
            "qwk %.4f auc %.4f ece %.4f%s",
            epoch,
            lr_now,
            train_parts["loss"],
            train_parts["loss_ordinal"],
            train_parts["loss_refer"],
            val_loss,
            val_qwk,
            float(val_metrics["auc_refer"]),
            float(val_metrics["ece"]),
            " *best*" if is_best else "",
        )
        _write_history(save_dir / "history.csv", history)

        if stale_epochs >= patience:
            _LOGGER.info(
                "early stopping at epoch %d (no val QWK gain in %d epochs)", epoch, patience
            )
            break

    # last.pt keeps the raw (non-EMA) weights + optimizer for manual resume.
    _save_checkpoint(
        save_dir / "last.pt",
        model,
        cfg,
        val_qwk,
        epoch,
        extra={
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_qwk": best_qwk if math.isfinite(best_qwk) else None,
        "epochs_run": epoch,
        "n_parameters": n_params,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "device": device,
        "ema": ema is not None,
        "amp": use_amp,
        "elapsed_s": time.time() - run_t0,
        "save_dir": str(save_dir),
        "final_val": jsonable(val_metrics),
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(jsonable(summary), indent=2), encoding="utf-8"
    )
    _LOGGER.info("done: best_val_qwk=%.4f @ epoch %d -> %s", best_qwk, best_epoch, save_dir)
    return summary


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HISTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.train.trainer",
        description="Train DrNet on an embedded-data training config.",
    )
    parser.add_argument("--config", required=True, help="path to a train YAML config")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device selection (default: cuda when available)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="dotted.key=value config overrides, e.g. train.lr=3e-4 data.img_size=96",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: parse args, load config, run :func:`train`."""
    args = _parse_args(argv)
    from retinaedge.utils.config import load_config

    cfg = load_config(args.config, args.overrides)
    try:
        train(cfg, device=args.device)
    except Exception:  # pragma: no cover - surface a clean CLI error
        _LOGGER.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
