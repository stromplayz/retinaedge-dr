"""Import a manifest-style dataset into the standard retinaedge layout.

Many curated Kaggle collections ship a CSV manifest (``image_path``, ``grade``,
optional ``split``) plus an image tree, rather than the plain
``images/ + labels.csv`` layout this repo trains from. This importer
recursively discovers such manifests inside a raw download directory, resolves
image paths, and writes::

    <out>/images/<name>.<ext>     (hardlinked when possible, copied otherwise)
    <out>/labels.csv              columns: image,grade[,split]
    <out>/provenance.json         source manifest, counts, per-source/grade stats

Discovery rules: any ``*.csv`` whose header contains a path-ish column
(``image_path``/``image``/``filename``/``path``) AND a grade-ish column
(``grade``/``label``/``diagnosis``). The first matching manifest wins per
output directory; ``--all`` imports every discovered manifest into
``<out>__<stem>`` directories.

CLI:
    python -m retinaedge.data.manifest_import --raw data/kaggle/raw --out data/kaggle_blend
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from retinaedge.utils.logging_utils import get_logger

__all__ = ["discover_manifests", "import_manifest", "main"]

logger = get_logger("data.manifest_import")

_PATH_COLS = ("image_path", "image", "filename", "file", "path")
_GRADE_COLS = ("grade", "label", "diagnosis")
_SPLIT_COLS = ("split", "subset", "stage")


def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    for cand in candidates:  # substring fallback (e.g. "original_image_path")
        for low, orig in lowered.items():
            if cand in low and "id" not in low:
                return orig
    return None


def discover_manifests(raw: Path) -> list[Path]:
    """Return CSVs under ``raw`` that look like image+grade manifests."""
    manifests: list[Path] = []
    for csv_path in sorted(raw.rglob("*.csv")):
        try:
            header = pd.read_csv(csv_path, nrows=0)
        except Exception:  # noqa: BLE001 - unreadable CSVs are simply skipped
            continue
        path_col = _find_column(header.columns, _PATH_COLS)
        grade_col = _find_column(header.columns, _GRADE_COLS)
        if path_col and grade_col:
            manifests.append(csv_path)
    return manifests


def _stage_image(src: Path, dst: Path) -> bool:
    """Hardlink (fallback copy) ``src`` to ``dst``; returns False when unreadable."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        try:
            shutil.copyfile(src, dst)
        except OSError:
            return False
    return True


def import_manifest(manifest: Path, raw: Path, out: Path, force: bool = False) -> dict:
    """Import one manifest into ``out`` (standard images/+labels.csv layout).

    Returns a stats dict (n_rows, n_staged, n_missing, grade_counts, splits).
    """
    df = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    path_col = _find_column(df.columns, _PATH_COLS)
    grade_col = _find_column(df.columns, _GRADE_COLS)
    split_col = _find_column(df.columns, _SPLIT_COLS)
    assert path_col and grade_col

    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    n_missing = 0
    used_names: set[str] = set()

    for _, row in df.iterrows():
        rel = str(row[path_col]).strip()
        grade = str(row[grade_col]).strip()
        if not rel or grade == "" or grade.lower() in {"nan", "none"}:
            continue
        try:
            grade_i = int(float(grade))
        except ValueError:
            continue
        if not 0 <= grade_i <= 4:
            continue
        src = None
        for base in (raw, manifest.parent, manifest.parent.parent):
            cand = base / rel
            if cand.exists():
                src = cand
                break
        if src is None:
            n_missing += 1
            continue
        name = Path(rel).name
        stem, ext = Path(name).stem, Path(name).suffix.lower() or ".jpg"
        unique = f"{stem}{ext}"
        bump = 1
        while unique in used_names:
            unique = f"{stem}_{bump}{ext}"
            bump += 1
        used_names.add(unique)
        if force or not (images_dir / unique).exists():
            if not _stage_image(src, images_dir / unique):
                n_missing += 1
                continue
        entry = {"image": unique, "grade": str(grade_i)}
        if split_col:
            tag = str(row[split_col]).strip().lower().replace("validation", "val")
            if tag in {"train", "val", "test"}:
                entry["split"] = tag
        rows.append(entry)

    if not rows:
        raise RuntimeError(f"no importable rows found in {manifest}")

    import csv as _csv

    fieldnames = ["image", "grade"] + (
        ["split"] if split_col and any("split" in r for r in rows) else []
    )
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    grade_counts: dict[str, int] = {}
    for r in rows:
        grade_counts[r["grade"]] = grade_counts.get(r["grade"], 0) + 1
    split_counts: dict[str, int] = {}
    for r in rows:
        if "split" in r:
            split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    stats = {
        "manifest": str(manifest.relative_to(raw))
        if manifest.is_relative_to(raw)
        else str(manifest),
        "n_rows": len(df),
        "n_staged": len(rows),
        "n_missing": n_missing,
        "grade_counts": dict(sorted(grade_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
    }
    (out / "provenance.json").write_text(
        json.dumps({"source": str(manifest), "raw_root": str(raw), **stats}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "imported %d/%d images (%d missing) from %s -> %s",
        stats["n_staged"],
        stats["n_rows"],
        n_missing,
        manifest,
        out,
    )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.data.manifest_import",
        description="Import manifest-style Kaggle datasets into images/+labels.csv layout.",
    )
    parser.add_argument("--raw", required=True, help="raw download directory to scan")
    parser.add_argument("--out", required=True, help="output dataset dir (data/<name>)")
    parser.add_argument(
        "--manifest", default=None, help="explicit manifest CSV (default: auto-discover)"
    )
    parser.add_argument("--all", action="store_true", help="import every discovered manifest")
    parser.add_argument("--force", action="store_true", help="re-stage images even if present")
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    if not raw.exists():
        logger.error("raw dir does not exist: %s", raw)
        return 1
    if args.manifest:
        targets = [Path(args.manifest)]
    else:
        targets = discover_manifests(raw)
        if not targets:
            logger.error("no manifest CSVs discovered under %s", raw)
            return 1
        logger.info("discovered %d manifest(s): %s", len(targets), [str(t) for t in targets])
        if not args.all:
            # Prefer the largest manifest (usually the master list).
            def _size(p: Path) -> int:
                try:
                    return len(pd.read_csv(p))
                except Exception:  # noqa: BLE001
                    return -1

            targets = [max(targets, key=_size)]
    for manifest in targets:
        out = (
            Path(args.out)
            if len(targets) == 1
            else Path(f"{args.out.rstrip('/')}__{manifest.stem}")
        )
        try:
            import_manifest(manifest, raw, out, force=args.force)
        except Exception:  # noqa: BLE001 - keep importing the remaining manifests
            logger.exception("failed to import %s", manifest)
            if len(targets) == 1:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
