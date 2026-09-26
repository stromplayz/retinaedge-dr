"""Probability-level ensembling (contract: ``retinaedge.models.ensemble``).

Averages the temperature-scaled grade probabilities of several checkpoints
(possibly different backbones trained on the same splits). Output-level
ensembling is the strongest single "loophole" in the accuracy ladder: for
correlated-but-not-identical models it reliably adds +0.5-1.5 pt AUC over the
best single model. Unlike the soup it multiplies inference cost by the number
of members, so it is an evaluation/reporting-time wrapper — the exported
Android artifact stays a single model unless explicitly chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from retinaedge.models.build import DrNet, build_model
from retinaedge.train.trainer import load_checkpoint
from retinaedge.utils.logging_utils import get_logger

__all__ = ["EnsemblePredictor", "load_ensemble"]

_LOGGER = get_logger("models.ensemble")


class EnsemblePredictor(nn.Module):
    """Weighted probability-average over member models.

    Members must expose ``predict_probs(imgs) -> (B, K)`` (DrNet does).
    The wrapper is itself an ``nn.Module`` holding the members as submodules
    so ``.to(device)`` and ``.eval()`` propagate naturally.
    """

    def __init__(self, models: Sequence[DrNet], weights: Sequence[float] | None = None) -> None:
        super().__init__()
        if not models:
            raise ValueError("need at least one member model")
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models):
            raise ValueError(f"weights/models mismatch: {len(weights)} vs {len(models)}")
        total = float(sum(weights))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        self.weights = [float(w) / total for w in weights]
        self.members = nn.ModuleList(models)

    @torch.no_grad()
    def predict_probs(self, imgs: Tensor) -> Tensor:
        """Weighted-average grade probabilities, ``(B, K)`` rows summing to 1."""
        acc: Tensor | None = None
        for weight, member in zip(self.weights, self.members, strict=True):
            probs = member.predict_probs(imgs) * weight
            acc = probs if acc is None else acc + probs
        assert acc is not None
        return acc


def load_ensemble(
    cfg: dict,
    ckpt_paths: Sequence[str | Path],
    weights: Sequence[float] | None = None,
    device: str = "cpu",
) -> EnsemblePredictor:
    """Build an ensemble from checkpoints sharing one architecture.

    Each member inherits its own stored calibration temperature; member
    hyperparameters (backbone/dropout) come from each checkpoint's embedded
    ``cfg`` when present, else from the passed ``cfg``.

    Args:
        cfg: Fallback training config (used when a checkpoint payload lacks
            an embedded ``cfg``).
        ckpt_paths: Trainer checkpoints or raw state dicts.
        weights: Optional member weights.
        device: Device string for the returned ensemble.

    Returns:
        An :class:`EnsemblePredictor` in eval mode on ``device``.
    """
    members: list[DrNet] = []
    for path in ckpt_paths:
        payload = load_checkpoint(path, map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        member_cfg = payload.get("cfg", cfg) if isinstance(payload, dict) else cfg
        model = build_model(member_cfg)
        model.load_state_dict(state)
        temperature = payload.get("temperature", 1.0) if isinstance(payload, dict) else 1.0
        model.set_temperature(float(temperature))
        model.to(device)
        model.eval()
        members.append(model)
    ensemble = EnsemblePredictor(members, weights=weights).to(device)
    ensemble.eval()
    _LOGGER.info("ensemble ready: %d members from %s", len(members), [str(p) for p in ckpt_paths])
    return ensemble
