"""Test-time augmentation (contract: ``retinaedge.models.tta``).

Averages temperature-scaled grade probabilities over geometric variants of
the input batch (horizontal flip and optional multi-scale zooms). Because the
probability simplex is closed under averaging, the output stays a valid
``(B, K)`` probability row-summing to 1 — the Android/export contract is
unaffected (TTA is an evaluation-time wrapper only).

Reference: the "loophole" is free accuracy — ensembling views costs no
training time and typically adds +0.2-1.0 pt AUC/accuracy on fundus data.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from retinaedge.models.build import DrNet

__all__ = ["tta_variants", "tta_probs"]


def tta_variants(
    imgs: Tensor, scales: Sequence[float] = (1.0,), hflip: bool = True
) -> list[Tensor]:
    """Build the list of TTA input variants.

    Args:
        imgs: Float tensor ``(B, 3, H, W)``.
        scales: Zoom factors relative to the input (1.0 = identity). Values
            ``!= 1.0`` resize with bilinear interpolation.
        hflip: Include the horizontally flipped view of every scale.

    Returns:
        List of ``(B, 3, H, W)`` tensors; identity is always first.
    """
    variants: list[Tensor] = []
    for scale in scales:
        s = float(scale)
        if s <= 0.0:
            raise ValueError(f"scales must be positive, got {s}")
        x = (
            imgs
            if abs(s - 1.0) < 1e-6
            else F.interpolate(imgs, scale_factor=s, mode="bilinear", align_corners=False)
        )
        variants.append(x)
        if hflip:
            variants.append(torch.flip(x, dims=[-1]))
    return variants


def tta_probs(
    model: DrNet,
    imgs: Tensor,
    scales: Sequence[float] = (1.0,),
    hflip: bool = True,
) -> Tensor:
    """Predict grade probabilities averaged over TTA variants.

    Args:
        model: A :class:`~retinaedge.models.build.DrNet` (or any module
            exposing ``predict_probs(imgs) -> (B, K)``).
        imgs: Float tensor ``(B, 3, H, W)`` ImageNet-normalized.
        scales: Zoom factors (see :func:`tta_variants`).
        hflip: Whether to include the horizontal flip view.

    Returns:
        Float tensor ``(B, K)`` of per-grade probabilities (rows sum to 1).
    """
    if not scales:
        raise ValueError("scales must contain at least one value")
    acc: Tensor | None = None
    for variant in tta_variants(imgs, scales=scales, hflip=hflip):
        probs = model.predict_probs(variant)
        acc = probs if acc is None else acc + probs
    return acc / float(len(scales) * (2 if hflip else 1))
