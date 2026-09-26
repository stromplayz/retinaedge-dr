"""Ordinal + referable training loss (contract: ``retinaedge.models.loss``).

Ordinal part: BCE-with-logits over the ``K-1`` cumulative binary tasks. For the
logit at index ``j`` (0-based, task ``grade > j``) the binary target is
``targets >= j + 1`` (equivalently ``targets > j``), matching the CORAL
formulation used by :func:`retinaedge.models.ordinal_ops.ordinal_probs`.

Referable part: BCE-with-logits on the auxiliary head with target
``targets >= 2`` (referable DR). ``focal_gamma > 0`` switches the ordinal BCE
to focal weighting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DrLoss", "build_loss"]


class DrLoss(nn.Module):
    """Weighted ordinal + referable loss (nn.Module so ``.to(device)`` works).

    Args:
        ordinal_weight: Weight of the cumulative-task BCE term.
        refer_weight: Weight of the auxiliary referable-DR BCE term.
        focal_gamma: Focal exponent for the ordinal BCE (0 disables focal).
    """

    def __init__(
        self,
        ordinal_weight: float = 1.0,
        refer_weight: float = 0.3,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if ordinal_weight < 0.0 or refer_weight < 0.0 or focal_gamma < 0.0:
            raise ValueError("loss weights and focal_gamma must be non-negative")
        self.ordinal_weight = float(ordinal_weight)
        self.refer_weight = float(refer_weight)
        self.focal_gamma = float(focal_gamma)

    def _ordinal_bce(self, ordinal_logits: Tensor, targets: Tensor) -> Tensor:
        """BCE (optionally focal) over all cumulative tasks, mean over batch+tasks."""
        num_tasks = ordinal_logits.shape[-1]
        # Binary label for task j ("grade > j") is targets >= j + 1.
        task_ids = torch.arange(num_tasks, device=ordinal_logits.device)
        labels = (targets.unsqueeze(-1) > task_ids).to(ordinal_logits.dtype)
        bce = F.binary_cross_entropy_with_logits(ordinal_logits, labels, reduction="none")
        if self.focal_gamma > 0.0:
            probs = torch.sigmoid(ordinal_logits)
            p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
            bce = bce * torch.pow(1.0 - p_t, self.focal_gamma)
        return bce.mean()

    def _refer_bce(self, refer_logits: Tensor, targets: Tensor) -> Tensor:
        """BCE on the auxiliary referable head (``grade >= 2``)."""
        labels = (targets >= 2).to(refer_logits.dtype)
        return F.binary_cross_entropy_with_logits(
            refer_logits.squeeze(-1), labels, reduction="mean"
        )

    def forward(self, outputs: dict, targets: Tensor) -> tuple[Tensor, dict[str, float]]:
        """Compute the combined loss.

        Args:
            outputs: Model output dict with ``"ordinal_logits"`` ``(B, K-1)``
                and ``"refer_logits"`` ``(B, 1)``.
            targets: Long tensor ``(B,)`` of ground-truth grades.

        Returns:
            Tuple ``(loss, parts)`` where ``loss`` is the weighted scalar loss
            tensor and ``parts`` maps ``loss``/``loss_ordinal``/``loss_refer``
            to detached floats for logging.
        """
        if targets.ndim != 1:
            raise ValueError(f"targets must be (B,) grades, got shape {tuple(targets.shape)}")
        targets = targets.long()
        ordinal_logits = outputs["ordinal_logits"]
        refer_logits = outputs["refer_logits"]
        if ordinal_logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"batch mismatch: logits {ordinal_logits.shape[0]} vs targets {targets.shape[0]}"
            )

        loss_ordinal = self._ordinal_bce(ordinal_logits, targets)
        loss_refer = self._refer_bce(refer_logits, targets)
        loss = self.ordinal_weight * loss_ordinal + self.refer_weight * loss_refer
        parts = {
            "loss": float(loss.detach()),
            "loss_ordinal": float(loss_ordinal.detach()),
            "loss_refer": float(loss_refer.detach()),
        }
        return loss, parts


def build_loss(cfg: dict) -> DrLoss:
    """Build :class:`DrLoss` from ``cfg["train"]["loss"]``.

    Supported keys with defaults: ``ordinal_weight=1.0``, ``refer_weight=0.3``,
    ``focal_gamma=0.0``.
    """
    loss_cfg: dict = cfg.get("train", {}).get("loss", {})
    return DrLoss(
        ordinal_weight=float(loss_cfg.get("ordinal_weight", 1.0)),
        refer_weight=float(loss_cfg.get("refer_weight", 0.3)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 0.0)),
    )
