"""Unit tests for the shared ordinal ops (network-free, torch-only)."""

from __future__ import annotations

import torch

from retinaedge.models.ordinal_ops import (
    expected_grade,
    hard_grade,
    ordinal_probs,
    referable_prob,
)


def test_ordinal_probs_shape_and_normalisation() -> None:
    torch.manual_seed(0)
    logits = torch.randn(8, 4) * 2.0
    probs = ordinal_probs(logits)
    assert probs.shape == (8, 5)
    assert torch.all(probs >= 0.0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_ordinal_probs_extreme_logits() -> None:
    # Very negative cumulative logits -> all mass on grade 0.
    logits = torch.full((2, 4), -30.0)
    probs = ordinal_probs(logits)
    assert torch.allclose(probs[:, 0], torch.ones(2), atol=1e-4)
    # Very positive -> all mass on top grade 4.
    logits = torch.full((2, 4), 30.0)
    probs = ordinal_probs(logits)
    assert torch.allclose(probs[:, 4], torch.ones(2), atol=1e-4)


def test_ordinal_probs_matches_hand_computation() -> None:
    # Monotone cumulative logits: g = [2, 1, -1, -2] -> sigma = [.8808, .7311, .2689, .1192]
    # P(y=k) = sigma(g_{k-1}) - sigma(g_k) with sigma(g_{-1})=1, sigma(g_3)=0
    logits = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
    probs = ordinal_probs(logits)
    expected = torch.tensor([[0.1192, 0.1497, 0.4622, 0.1497, 0.1192]])
    assert torch.allclose(probs, expected, atol=1e-3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_expected_and_hard_grade() -> None:
    probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
    exp = expected_grade(probs)
    hard = hard_grade(probs)
    assert torch.allclose(exp, torch.tensor([0.0, 4.0]))
    assert torch.equal(hard, torch.tensor([0, 4]))


def test_referable_prob() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.05, 0.03, 0.02], [0.0, 0.0, 0.4, 0.3, 0.3]])
    ref = referable_prob(probs)
    assert torch.allclose(ref, torch.tensor([0.10, 1.00]), atol=1e-5)
