#!/usr/bin/env python3
"""Rotate the Kaggle training dataset across scheduled improve invocations.

Scheduled runs should not hammer a single mirror, and rotating data sources is
part of the "improve data" mandate: every invocation picks the next handle
from the validated catalog (``configs/data/kaggle_dr_catalog.yaml``), indexed
by how far the campaign has advanced (history length). The pick is
deterministic — the same campaign progress always yields the same dataset, so
re-runs and investigations reproduce.

Only the ``--top`` highest-priority (largest 5-grade union) handles rotate by
default; the workflow falls back to its default handle whenever a rotated
download fails, so a bad mirror never blocks the campaign.

Usage:
    python scripts/rotate_dataset.py \
        --catalog configs/data/kaggle_dr_catalog.yaml \
        --state runs/improve_state.json [--top 2]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HANDLE_RE = re.compile(r"^\s*-\s+handle:\s*(\S+)\s*$", re.MULTILINE)


def load_handles(catalog: str | Path, top: int) -> list[str]:
    """Ordered dataset handles from the catalog (priority order), first ``top``."""
    handles = _HANDLE_RE.findall(Path(catalog).read_text(encoding="utf-8"))
    return handles[: max(1, top)]


def pick_index(state_path: str | Path, n: int) -> int:
    """Rotation index from campaign progress (0 when state is unreadable)."""
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return len(state.get("history", [])) % n
    except (OSError, ValueError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rotate_dataset", description=__doc__)
    parser.add_argument("--catalog", default="configs/data/kaggle_dr_catalog.yaml")
    parser.add_argument("--state", default="runs/improve_state.json")
    parser.add_argument("--top", type=int, default=2, help="rotate among the first N handles")
    args = parser.parse_args(argv)

    try:
        handles = load_handles(args.catalog, args.top)
    except OSError:
        handles = []
    if not handles:
        # Empty output -> the workflow keeps its default handle.
        return 0
    print(handles[pick_index(args.state, len(handles))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
