"""Ordinal regression probability utilities (cumulative-link / CORAL-style).

The grading head emits ``K-1`` logits where logit ``g_k`` solves the binary task
``"grade > k"`` (for k = 0..K-2). This is the CORAL formulation of ordinal
regression (Cao, Mirjalili & Raschka, 2020; Shi, Cao & Raschka, 2021).

Grade scheme (ICDRSS):
    0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
Referable DR = grade >= 2.
"""

from __future__ import annotations

import torch

__all__ = ["ordinal_probs", "expected_grade", "hard_grade", "referable_prob"]


def ordinal_probs(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """Convert K-1 cumulative logits into K class probabilities.

    Args:
        ordinal_logits: float tensor of shape ``(B, K-1)`` — raw logits of the
            binary tasks ``P(grade > k)``.

    Returns:
        Float tensor ``(B, K)`` of non-negative per-grade probabilities whose
        rows sum to 1. Computed as first differences of the sigmoid cumulative
        estimates, clamped and re-normalised for numerical safety (the binary
        heads are not guaranteed monotone during early training).
    """
    if ordinal_logits.dim() != 2:
        raise ValueError(f"expected (B, K-1) logits, got shape {tuple(ordinal_logits.shape)}")
    batch = ordinal_logits.shape[0]
    ones = torch.ones((batch, 1), dtype=ordinal_logits.dtype, device=ordinal_logits.device)
    zeros = torch.zeros((batch, 1), dtype=ordinal_logits.dtype, device=ordinal_logits.device)
    cum = torch.cat([ones, torch.sigmoid(ordinal_logits), zeros], dim=-1)
    probs = (cum[:, :-1] - cum[:, 1:]).clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def expected_grade(probs: torch.Tensor) -> torch.Tensor:
    """Soft expectation ``E[Y]`` from class probabilities — shape ``(B,)`` float."""
    num_grades = probs.shape[-1]
    weights = torch.arange(num_grades, dtype=probs.dtype, device=probs.device)
    return (probs * weights).sum(dim=-1)


def hard_grade(probs: torch.Tensor) -> torch.Tensor:
    """Argmax grade prediction — shape ``(B,)`` long."""
    return torch.argmax(probs, dim=-1)


def referable_prob(probs: torch.Tensor) -> torch.Tensor:
    """P(referable DR) = P(grade >= 2) = sum of probs[:, 2:] — shape ``(B,)``."""
    return probs[:, 2:].sum(dim=-1)
