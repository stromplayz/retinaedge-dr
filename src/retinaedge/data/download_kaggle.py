"""Download a Kaggle competition/dataset with kagglehub and stage it locally.

CLI (docs/INTERFACES.md):
    python -m retinaedge.data.download_kaggle --handle aptos2019-blindness-detection --dest data/aptos

Credentials are read by kagglehub from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` env
vars or ``~/.kaggle/kaggle.json`` (free Kaggle account required). Files land in
``<dest>/raw`` — run ``python -m retinaedge.data.prepare --config <cfg>`` next
to normalise them into ``images/`` + ``labels.csv``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from retinaedge.utils.logging_utils import get_logger

__all__ = ["main"]

logger = get_logger("data.download_kaggle")


def _stage_into(src: Path, dst: Path) -> int:
    """Copy (hardlink-first) everything under ``src`` into ``dst``; returns file count."""
    staged = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == path.stat().st_size:
            continue  # already staged (idempotent re-runs)
        try:
            os.link(path, target)
        except OSError:  # cross-device -> plain copy
            shutil.copyfile(path, target)
        staged += 1
    return staged


def _download(handle: str, kind: str) -> Path:
    """Download via kagglehub and return the local (extracted) path."""
    import kagglehub  # imported lazily: no network at import time

    if kind == "competition":
        return Path(kagglehub.competition_download(handle))
    if kind == "dataset":
        return Path(kagglehub.dataset_download(handle))
    # auto: APTOS/EyePACS are competitions; public datasets use the dataset route.
    try:
        return Path(kagglehub.competition_download(handle))
    except Exception as exc:  # noqa: BLE001 - report both attempts below
        logger.info("competition download failed (%s) — trying dataset route", exc)
        return Path(kagglehub.dataset_download(handle))


def main(argv: Sequence[str] | None = None) -> int:
    """Download ``--handle`` into ``--dest/raw``. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.data.download_kaggle",
        description="Fetch a Kaggle competition/dataset via kagglehub and stage it into <dest>/raw.",
    )
    parser.add_argument("--handle", required=True, help="e.g. aptos2019-blindness-detection")
    parser.add_argument("--dest", default=None, help="target dir (default: data/<handle-tail>)")
    parser.add_argument(
        "--type",
        choices=("auto", "competition", "dataset"),
        default="auto",
        help="kagglehub route (default: auto)",
    )
    args = parser.parse_args(argv)

    dest = Path(args.dest or Path("data") / args.handle.split("/")[-1])
    raw = dest / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s (%s) ...", args.handle, args.type)
    src = _download(args.handle, args.type)
    n = _stage_into(src, raw)
    logger.info("staged %d new file(s) into %s (source cache: %s)", n, raw, src)
    logger.info(
        "next: python -m retinaedge.data.prepare --config configs/data/<name>.yaml (or --root %s)",
        dest,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
