"""Model soup — uniform weight averaging (contract: ``retinaedge.models.soup``).

Averaging the weights of checkpoints that share one architecture and sit in
the same loss basin (e.g. best/last checkpoints of one run, EMA weights, or
fine-tunes from a shared init) yields a "soup" that often beats every
ingredient. Unlike output-level ensembling it costs nothing at inference:
the soup is a single standard ``DrNet`` checkpoint and exports to ONNX/TFLite
exactly like any other.

Reference: Wortsman et al., "Model soups: averaging weights of multiple
fine-tuned models improves accuracy without increasing inference time" (2022).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from retinaedge.train.trainer import load_checkpoint
from retinaedge.utils.logging_utils import get_logger

__all__ = ["average_state_dicts", "make_soup"]

_LOGGER = get_logger("models.soup")


def average_state_dicts(
    paths: Sequence[str | Path], weights: Sequence[float] | None = None
) -> dict[str, Tensor]:
    """Uniformly (or weightedly) average state dicts of same-architecture models.

    Args:
        paths: Checkpoint paths (trainer payloads or raw state dicts).
        weights: Optional mixing weights; must match ``len(paths)`` and sum to
            a positive value. Default: uniform.

    Returns:
        The averaged state dict (new tensors, CPU).

    Raises:
        ValueError: On empty input, weight mismatch, or incompatible state
            dicts (different keys/shapes — averaging across architectures is
            not meaningful and is refused).
    """
    if not paths:
        raise ValueError("need at least one checkpoint to average")
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise ValueError(f"weights/paths mismatch: {len(weights)} vs {len(paths)}")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")

    acc: dict[str, Tensor] | None = None
    for path, weight in zip(paths, weights, strict=True):
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if acc is None:
            acc = {k: v.detach().cpu().clone().float() * (weight / total) for k, v in state.items()}
        else:
            if set(state.keys()) != set(acc.keys()):
                raise ValueError(f"state dict keys differ between {paths[0]} and {path}")
            for key, value in state.items():
                if value.shape != acc[key].shape:
                    raise ValueError(
                        f"shape mismatch for {key!r}: {acc[key].shape} vs {value.shape} ({path})"
                    )
                acc[key] += value.detach().cpu().float() * (weight / total)
    assert acc is not None  # for type checkers; loop ran at least once
    return acc


def make_soup(
    paths: Sequence[str | Path],
    out_path: str | Path,
    weights: Sequence[float] | None = None,
    extra: dict | None = None,
) -> Path:
    """Average checkpoints and write a trainer-compatible soup checkpoint.

    Metadata (``cfg``, ``temperature``) is inherited from the first
    checkpoint; ``val_qwk``/``epoch`` become the mean/max of the ingredients
    so downstream tooling keeps working unchanged.

    Args:
        paths: Ingredient checkpoints (same architecture).
        out_path: Destination ``.pt`` path (parent dirs are created).
        weights: Optional per-ingredient weights.
        extra: Optional extra payload entries merged into the checkpoint.

    Returns:
        The resolved output path.
    """
    if not paths:
        raise ValueError("need at least one checkpoint to make a soup")
    state = average_state_dicts(paths, weights=weights)
    first = load_checkpoint(paths[0], map_location="cpu")
    payload: dict = {}
    if isinstance(first, dict) and "state_dict" in first:
        cfg = first.get("cfg")
        temperature = first.get("temperature", 1.0)
        val_qwk = first.get("val_qwk")
        epoch = first.get("epoch", 0)
        if len(paths) > 1:
            q = [load_checkpoint(p, map_location="cpu").get("val_qwk") for p in paths[1:]]
            vals = [v for v in [val_qwk, *q] if isinstance(v, (int, float))]
            ep = [load_checkpoint(p, map_location="cpu").get("epoch", 0) for p in paths]
            ep_vals = [v for v in ep if isinstance(v, (int, float))]
            val_qwk = sum(vals) / len(vals) if vals else None
            epoch = max(ep_vals) if ep_vals else epoch
        payload = {
            "state_dict": state,
            "cfg": cfg,
            "temperature": float(temperature),
            "val_qwk": val_qwk,
            "epoch": int(epoch or 0),
            "soup_of": [str(p) for p in paths],
        }
    else:
        payload = {"state_dict": state, "soup_of": [str(p) for p in paths]}
    if extra:
        payload.update(extra)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    _LOGGER.info("soup written: %s (%d ingredients)", out, len(paths))
    return out
