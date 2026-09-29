"""Unit tests for the DrLoss ordinal + referable loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from retinaedge.models import DrLoss, build_loss


def _outputs(targets: torch.Tensor, offset: float = 12.0) -> dict:
    """Ordinal logits perfectly aligned with `targets` (plus a large margin)."""
    batch = targets.shape[0]
    task_ids = torch.arange(4).expand(batch, 4)
    ordinal = torch.where(
        targets.unsqueeze(-1) > task_ids,
        torch.full((batch, 4), offset),
        torch.full((batch, 4), -offset),
    )
    refer = torch.where(targets >= 2, offset, -offset).unsqueeze(-1)
    return {"ordinal_logits": ordinal, "refer_logits": refer}


def test_loss_zero_for_perfect_logits() -> None:
    targets = torch.tensor([0, 1, 3, 4])
    loss, parts = DrLoss()(_outputs(targets), targets)
    assert math.isfinite(float(loss))
    assert float(loss) == pytest.approx(0.0, abs=1e-4)
    assert parts["loss_ordinal"] == pytest.approx(0.0, abs=1e-4)
    assert parts["loss_refer"] == pytest.approx(0.0, abs=1e-4)


def test_ordinal_targets_are_cumulative() -> None:
    # Hand-check task labels: y=2 -> cumulative labels (y>j) = [1, 1, 0, 0].
    loss_fn = DrLoss(ordinal_weight=1.0, refer_weight=0.0)
    outputs = {"ordinal_logits": torch.zeros(1, 4), "refer_logits": torch.zeros(1, 1)}
    targets = torch.tensor([2])
    loss, parts = loss_fn(outputs, targets)
    expected = F.binary_cross_entropy_with_logits(
        torch.zeros(1, 4), torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    )
    assert parts["loss_ordinal"] == pytest.approx(float(expected))
    assert float(loss) == pytest.approx(float(expected))


def test_focal_matches_bce_when_gamma_zero() -> None:
    torch.manual_seed(0)
    outputs = {"ordinal_logits": torch.randn(6, 4), "refer_logits": torch.randn(6, 1)}
    targets = torch.randint(0, 5, (6,))
    plain, _ = DrLoss(focal_gamma=0.0)(outputs, targets)
    focal0, _ = DrLoss(focal_gamma=1e-9)(outputs, targets)
    assert float(plain) == pytest.approx(float(focal0), rel=1e-4)


def test_focal_behaves_like_bce_only_downweights_easy() -> None:
    # Focal weighting multiplies each element by (1 - p_t)^gamma <= 1, so the
    # focal loss can only match (hard examples) or reduce (easy examples) the
    # plain BCE — never exceed it.
    hard = {
        "ordinal_logits": torch.tensor([[-4.0, -4.0, -4.0, -4.0]]),
        "refer_logits": torch.tensor([[0.0]]),
    }
    easy = {
        "ordinal_logits": torch.tensor([[4.0, 4.0, 4.0, 4.0]]),
        "refer_logits": torch.tensor([[0.0]]),
    }
    targets = torch.tensor([4])
    bce_hard, _ = DrLoss(focal_gamma=0.0, refer_weight=0.0)(hard, targets)
    focal_hard, _ = DrLoss(focal_gamma=2.0, refer_weight=0.0)(hard, targets)
    bce_easy, _ = DrLoss(focal_gamma=0.0, refer_weight=0.0)(easy, targets)
    focal_easy, _ = DrLoss(focal_gamma=2.0, refer_weight=0.0)(easy, targets)
    assert float(focal_hard) == pytest.approx(float(bce_hard), rel=0.2)
    assert float(focal_easy) < 0.05 * float(bce_easy)


def test_weights_are_respected() -> None:
    torch.manual_seed(1)
    outputs = {"ordinal_logits": torch.randn(4, 4), "refer_logits": torch.randn(4, 1)}
    targets = torch.randint(0, 5, (4,))
    _, parts = DrLoss()(outputs, targets)
    only_ordinal, parts_ord = DrLoss(refer_weight=0.0)(outputs, targets)
    only_refer, parts_ref = DrLoss(ordinal_weight=0.0)(outputs, targets)
    assert float(only_ordinal) == pytest.approx(parts_ord["loss_ordinal"])
    assert float(only_refer) == pytest.approx(0.3 * parts_ref["loss_refer"])
    combined = 1.0 * parts["loss_ordinal"] + 0.3 * parts["loss_refer"]
    assert parts["loss"] == pytest.approx(combined)


def test_batch_and_shape_validation() -> None:
    outputs = {"ordinal_logits": torch.randn(3, 4), "refer_logits": torch.randn(3, 1)}
    with pytest.raises(ValueError):
        DrLoss()(outputs, torch.randint(0, 5, (2,)))
    with pytest.raises(ValueError):
        DrLoss()(outputs, torch.randint(0, 5, (2, 1)))


def test_build_loss_from_config() -> None:
    cfg = {"train": {"loss": {"ordinal_weight": 2.0, "refer_weight": 0.0, "focal_gamma": 1.5}}}
    loss_fn = build_loss(cfg)
    assert loss_fn.ordinal_weight == 2.0
    assert loss_fn.refer_weight == 0.0
    assert loss_fn.focal_gamma == 1.5
    defaults = build_loss({})
    assert defaults.ordinal_weight == 1.0 and defaults.refer_weight == 0.3


def test_label_smoothing_changes_bce() -> None:
    torch.manual_seed(0)
    targets = torch.randint(0, 5, (8,))
    out = _outputs(targets, offset=1.0)  # imperfect logits so BCE > 0
    clean, _ = DrLoss(label_smoothing=0.0)(out, targets)
    smooth, _ = DrLoss(label_smoothing=0.05)(out, targets)
    assert torch.isfinite(smooth)
    assert not torch.isclose(clean, smooth)


def test_label_smoothing_validation() -> None:
    with pytest.raises(ValueError):
        DrLoss(label_smoothing=1.0)
    with pytest.raises(ValueError):
        DrLoss(label_smoothing=-0.1)


def test_build_loss_label_smoothing_from_config() -> None:
    cfg = {"train": {"loss": {"label_smoothing": 0.05}}}
    assert build_loss(cfg).label_smoothing == pytest.approx(0.05)
    assert build_loss({}).label_smoothing == 0.0
