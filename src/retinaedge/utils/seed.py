"""Deterministic seeding across random / numpy / torch."""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed python ``random``, numpy and torch (CPU + all CUDA devices).

    Also sets cudnn to deterministic mode so runs are reproducible at a small
    speed cost. Safe to call multiple times.
    """
    if seed is None:
        seed = 0
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
