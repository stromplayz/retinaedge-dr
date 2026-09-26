"""DrNet model construction (contract: ``retinaedge.models.build``).

The model is a timm backbone with two linear heads:

* ordinal head — ``K-1`` cumulative logits (CORAL-style binary tasks
  ``grade > k``), decoded by :mod:`retinaedge.models.ordinal_ops`;
* auxiliary referable head — a single logit for ``grade >= 2``.

The export graph contract consumes ``DrNet`` through temperature-scaled
:class:`predict_probs`, so calibration (temperature) is stored on the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from retinaedge.models.ordinal_ops import ordinal_probs
from retinaedge.utils.logging_utils import get_logger

__all__ = ["DrNet", "build_model"]

_LOGGER = get_logger("models.build")

_DEFAULT_BACKBONE = "mobilenetv3_small_100"


def _create_backbone(name: str, pretrained: bool) -> nn.Module:
    """Create a timm backbone, degrading to ``pretrained=False`` when offline.

    Args:
        name: timm model name.
        pretrained: Whether to attempt downloading pretrained weights.

    Returns:
        Feature extractor with ``num_classes=0`` and average pooling, i.e. a
        module mapping ``(B, 3, H, W)`` images to ``(B, embed_dim)`` features.
    """
    import timm

    if not pretrained:
        return timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")
    try:
        return timm.create_model(name, pretrained=True, num_classes=0, global_pool="avg")
    except Exception as exc:  # noqa: BLE001 - any hub/network failure must degrade
        _LOGGER.warning(
            "pretrained weights unavailable for %s (%s: %s); falling back to random init",
            name,
            type(exc).__name__,
            exc,
        )
        return timm.create_model(name, pretrained=False, num_classes=0, global_pool="avg")


def _infer_embed_dim(feature_extractor: nn.Module) -> int:
    """Infer the true feature dimension of a ``num_classes=0`` timm model.

    A dummy forward is the only reliable source: some backbones (e.g. the
    MobileNetV3 family) run an extra 1x1 head conv after pooling, so their
    ``num_features`` attribute disagrees with the actual output width.
    Falls back to ``num_features`` when the probe input is unsupported
    (e.g. transformers with fixed input sizes).
    """
    was_training = feature_extractor.training
    feature_extractor.eval()  # batchnorm cannot run on a 1-sample training probe
    try:
        with torch.no_grad():
            probe = feature_extractor(torch.zeros(1, 3, 32, 32))
        return int(probe.flatten(1).shape[-1])
    except Exception:  # noqa: BLE001 - probe input may not suit every backbone
        return int(feature_extractor.num_features)
    finally:
        if was_training:
            feature_extractor.train()


class DrNet(nn.Module):
    """Ordinal DR grading network: timm backbone + ordinal/referable heads.

    Attributes:
        embed_dim: Feature dimension of the timm backbone.
        num_grades: Number of DR grades (default 5, ICDRSS scheme).
        temperature: Temperature applied to ordinal logits inside
            :meth:`predict_probs` (calibration; default 1.0 = no scaling).
    """

    def __init__(
        self,
        backbone: str = _DEFAULT_BACKBONE,
        pretrained: bool = False,
        dropout: float = 0.1,
        num_grades: int = 5,
    ) -> None:
        """Build the network.

        Args:
            backbone: timm model name (e.g. ``mobilenetv3_small_100``).
            pretrained: Attempt to load pretrained backbone weights (degrades
                gracefully to random init when the hub is unreachable).
            dropout: Dropout probability applied to backbone features.
            num_grades: Number of ordinal grades; the ordinal head emits
                ``num_grades - 1`` cumulative logits.
        """
        super().__init__()
        if num_grades < 2:
            raise ValueError(f"num_grades must be >= 2, got {num_grades}")
        self.backbone_name = backbone
        self.num_grades = num_grades
        self.feature_extractor = _create_backbone(backbone, pretrained)
        self.embed_dim: int = _infer_embed_dim(self.feature_extractor)
        self.dropout = nn.Dropout(float(dropout))
        self.ordinal_head = nn.Linear(self.embed_dim, num_grades - 1)
        self.refer_head = nn.Linear(self.embed_dim, 1)
        self.temperature: float = 1.0

    def forward(self, imgs: Tensor) -> dict[str, Tensor]:
        """Run the network.

        Args:
            imgs: Float tensor ``(B, 3, H, W)`` ImageNet-normalized.

        Returns:
            Dict with ``"ordinal_logits"`` ``(B, K-1)`` and ``"refer_logits"``
            ``(B, 1)`` float tensors.
        """
        feats = self.feature_extractor(imgs)
        feats = self.dropout(feats)
        return {
            "ordinal_logits": self.ordinal_head(feats),
            "refer_logits": self.refer_head(feats),
        }

    def set_temperature(self, t: float) -> None:
        """Store the calibration temperature used by :meth:`predict_probs`."""
        if not t > 0.0:
            raise ValueError(f"temperature must be > 0, got {t}")
        self.temperature = float(t)

    @torch.no_grad()
    def predict_probs(self, imgs: Tensor) -> Tensor:
        """Temperature-scaled grade probabilities.

        Temporarily switches the module to eval mode (no dropout/batchnorm
        updates) so the result is deterministic even on a freshly built model.

        Args:
            imgs: Float tensor ``(B, 3, H, W)`` ImageNet-normalized.

        Returns:
            Float tensor ``(B, K)`` of per-grade probabilities, rows sum to 1.
        """
        was_training = self.training
        self.eval()
        try:
            outputs = self.forward(imgs)
            scaled = outputs["ordinal_logits"] / self.temperature
            return ordinal_probs(scaled)
        finally:
            if was_training:
                self.train()


def build_model(cfg: dict) -> DrNet:
    """Build :class:`DrNet` from a training config.

    Args:
        cfg: Full config dict; the ``cfg["model"]`` section supports:
            ``backbone`` (timm name), ``pretrained`` (bool),
            ``dropout`` (float), ``num_grades`` (int, default 5).

    Returns:
        A :class:`DrNet` instance (not moved to any device).
    """
    model_cfg = cfg.get("model", {})
    return DrNet(
        backbone=str(model_cfg.get("backbone", _DEFAULT_BACKBONE)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        num_grades=int(model_cfg.get("num_grades", 5)),
    )
