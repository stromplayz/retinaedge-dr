"""Download a Hugging Face dataset snapshot and stage it locally.

CLI (docs/INTERFACES.md):
    python -m retinaedge.data.download_hf --repo <hf dataset id> --dest data/hf_<name>

Uses ``huggingface_hub.snapshot_download`` (part of the ``hf`` extra:
``pip install -e ".[hf]"``). Public datasets work anonymously; gated datasets
need ``HF_TOKEN``. Files land in ``<dest>/raw`` — normalise afterwards with
``python -m retinaedge.data.prepare --config <cfg>``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from retinaedge.utils.logging_utils import get_logger

__all__ = ["main"]

logger = get_logger("data.download_hf")


def main(argv: Sequence[str] | None = None) -> int:
    """Download ``--repo`` into ``--dest/raw``. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.data.download_hf",
        description="Fetch a Hugging Face dataset snapshot into <dest>/raw.",
    )
    parser.add_argument("--repo", required=True, help="HF dataset id, e.g. user/dr-fundus")
    parser.add_argument("--dest", default=None, help="target dir (default: data/hf_<repo-tail>)")
    parser.add_argument("--revision", default=None, help="git revision/branch/tag (default: main)")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - install hint only
        logger.error("huggingface_hub is not installed — run: pip install -e '.[hf]' (%s)", exc)
        return 1

    tail = args.repo.split("/")[-1]
    dest = Path(args.dest or Path("data") / f"hf_{tail}")
    raw = dest / "raw"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s @ %s ...", args.repo, args.revision or "main")
    snapshot_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        local_dir=raw,
    )
    src = Path(snapshot_path)
    if src.resolve() != raw.resolve():  # hub cached outside local_dir -> stage it in
        for path in src.rglob("*"):
            if path.is_file():
                target = raw / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path, target)
    logger.info("staged dataset files into %s", raw)
    logger.info(
        "next: python -m retinaedge.data.prepare --config configs/data/hf.yaml (or --root %s)", dest
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
