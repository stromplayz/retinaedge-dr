#!/usr/bin/env python3
"""Rotate the Kaggle training dataset across scheduled improve invocations.

Scheduled runs should not hammer a single mirror, and rotating data sources is
part of the "data maximalism" mandate: every invocation picks the next handle
from the validated catalog (``configs/data/kaggle_dr_catalog.yaml``), indexed
by how far the campaign has advanced (history length). The pick is
deterministic — the same campaign progress always yields the same dataset, so
re-runs and investigations reproduce.

Disk guard: handles whose catalog ``bytes`` exceed ``--max-bytes`` are skipped
(a 22 GB archive cannot fit next to its extraction on a ~45 GB runner), so
rotation only ever proposes datasets that actually fit. If every catalog entry
is too large the output is empty and the workflow keeps its default handle.
Any rotated download that still fails falls back to the default handle in the
workflow.

Usage:
    python scripts/rotate_dataset.py \
        --catalog configs/data/kaggle_dr_catalog.yaml \
        --state runs/improve_state.json \
        --max-bytes 9000000000 --top 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HANDLE_RE = re.compile(r"^\s*-\s+handle:\s*(\S+)")
_BYTES_RE = re.compile(r"^\s*bytes:\s*(\d+)")


def load_entries(catalog: str | Path) -> list[tuple[str, int]]:
    """Ordered ``(handle, archive_bytes)`` pairs from the catalog."""
    entries: list[tuple[str, int]] = []
    handle: str | None = None
    for line in Path(catalog).read_text(encoding="utf-8").splitlines():
        m = _HANDLE_RE.match(line)
        if m:
            handle = m.group(1)
            continue
        m = _BYTES_RE.match(line)
        if m and handle:
            entries.append((handle, int(m.group(1))))
            handle = None
    return entries


def load_handles(catalog: str | Path, top: int, max_bytes: int) -> list[str]:
    """First ``top`` size-valid handles (priority order)."""
    fits = [h for h, b in load_entries(catalog) if b <= max_bytes]
    return fits[: max(1, top)]


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
    parser.add_argument("--top", type=int, default=4, help="rotate among the first N valid handles")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=9_000_000_000,
        help="skip catalog entries whose archive exceeds this size (runner disk guard)",
    )
    args = parser.parse_args(argv)

    try:
        handles = load_handles(args.catalog, args.top, args.max_bytes)
    except OSError:
        handles = []
    if not handles:
        # Empty output -> the workflow keeps its default handle.
        return 0
    print(handles[pick_index(args.state, len(handles))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
