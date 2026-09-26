"""Normalise every supported data source into ``data/<name>/{images/, labels.csv}``.

CLI (docs/INTERFACES.md):
    python -m retinaedge.data.prepare --config configs/data/aptos.yaml

Input layout: whatever the ``retinaedge.data.download_kaggle`` /
``retinaedge.data.download_hf`` CLIs staged under ``<root>/raw`` (archives are
extracted automatically), or a local ImageFolder-like tree (``folder``).

Output layout (consumed by ``retinaedge.data.dataset``):
    data/<name>/images/<file>      # fundus images
    data/<name>/labels.csv         # columns: image,grade (paths relative to images/)

Supported sources: aptos, eyepacs, ddr, folder, hf. The normalisation is
idempotent: files already staged with matching size are skipped, so re-runs are
cheap. ``--force`` re-copies everything.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from retinaedge.utils.config import load_config
from retinaedge.utils.logging_utils import get_logger

__all__ = ["main"]

logger = get_logger("data.prepare")

IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
GRADE_ALIASES = ("grade", "label", "level", "diagnosis")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _find_dir(root: Path, names: Sequence[str]) -> Path | None:
    """First directory under ``root`` whose name is in ``names`` (depth-first)."""
    for path in root.rglob("*"):
        if path.is_dir() and path.name in names:
            return path
    return None


def _resolve_image(images_dir: Path, name: str) -> Path | None:
    """Resolve ``name`` (with or without extension) to an existing image file."""
    cand = images_dir / name
    if cand.is_file():
        return cand
    stem = glob.escape(Path(name).stem)
    for ext in sorted(IMG_EXTS):
        cand = images_dir / f"{Path(name).stem}{ext}"
        if cand.is_file():
            return cand
    matches = sorted(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _ensure_extracted(raw: Path) -> None:
    """Concatenate multi-part zips and extract archives under ``raw/_extracted``."""
    extracted = raw / "_extracted"

    # Kaggle ships some competitions (e.g. EyePACS) as split zips: x.zip.001, ...
    part_groups: dict[str, list[Path]] = {}
    for part in raw.glob("*.zip.*"):
        stem = part.name.rsplit(".zip.", 1)[0]
        part_groups.setdefault(stem, []).append(part)
    for stem, parts in part_groups.items():
        out = extracted / f"{stem}.zip"
        if out.exists() and out.stat().st_size > 0:
            continue
        extracted.mkdir(parents=True, exist_ok=True)
        logger.info("concatenating %d part(s) of %s.zip ...", len(parts), stem)
        with open(out, "wb") as sink:
            for part in sorted(parts):
                with open(part, "rb") as chunk:
                    shutil.copyfileobj(chunk, sink, length=1 << 24)

    archives = list(raw.glob("*.zip")) + list(raw.glob("*.tar")) + list(raw.glob("*.tar.gz"))
    archives += list(raw.glob("*.tgz")) + list(extracted.glob("*.zip"))
    for archive in sorted(set(archives)):
        out_dir = extracted / archive.name.replace(".", "_")
        if out_dir.exists() and any(out_dir.iterdir()):
            continue
        tmp = out_dir.with_suffix(".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        logger.info("extracting %s ...", archive.name)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                tf.extractall(tmp, filter="data")
        else:
            logger.warning("skipping unknown archive format: %s", archive.name)
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.rename(tmp, out_dir)


def _collect_grade_tree(root: Path) -> list[tuple[Path, int]]:
    """Collect ``(image, grade)`` from an ImageFolder-like tree ``<...>/<grade>/<img>``.

    Only immediate parent directories named ``0..4`` count as grade folders —
    this deliberately ignores lesion-mask/annotation directories (DDR's
    ``12/13/14``).
    """
    pairs: list[tuple[Path, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        try:
            grade = int(path.parent.name)
        except ValueError:
            continue
        if 0 <= grade <= 4:
            pairs.append((path, grade))
    return pairs


def _pick_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    """Locate (image-column, grade-column) in a labels dataframe."""
    image_col = next(
        (
            c
            for c in df.columns
            if str(c).lower() in {"image", "id_code", "image_name", "file_name", "img"}
        ),
        None,
    )
    if image_col is None:  # fall back: first column whose values look like image names
        for col in df.columns:
            if (
                df[col]
                .astype(str)
                .str.contains(r"\.(jpg|jpeg|png|tif)", case=False, regex=True)
                .any()
            ):
                image_col = col
                break
    grade_col = next(
        (c for c in df.columns if str(c).lower() in GRADE_ALIASES),
        None,
    )
    if image_col is None or grade_col is None:
        return None
    return str(image_col), str(grade_col)


def _rows_from_labels_csv(csv_path: Path, images_dir: Path | None) -> list[tuple[Path, int]]:
    """Build ``(image, grade)`` pairs from a labels CSV, resolving image files.

    The CSV is read with ``dtype=str`` so leading-zero ids (``00007``) survive
    pandas' numeric coercion.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    picked = _pick_columns(df)
    if picked is None:
        logger.warning(
            "%s: no (image, grade) columns found — columns are %s", csv_path, list(df.columns)
        )
        return []
    image_col, grade_col = picked
    pairs: list[tuple[Path, int]] = []
    for name, grade in zip(
        df[image_col].astype(str), pd.to_numeric(df[grade_col], errors="coerce"), strict=True
    ):
        if pd.isna(grade) or not 0 <= int(grade) <= 4:
            continue
        path = _resolve_image(images_dir, name) if images_dir else Path(name)
        if path is not None and path.is_file():
            pairs.append((path, int(grade)))
    return pairs


def _collect_aptos(raw: Path) -> list[tuple[Path, int]]:
    """APTOS 2019: ``train.csv`` (id_code, diagnosis) + ``train_images/``."""
    images_dir = _find_dir(raw, ("train_images", "images", "train"))
    if images_dir is None:
        raise FileNotFoundError(f"no train_images/ directory under {raw}")
    csvs = sorted(raw.rglob("*.csv"))
    preferred = [p for p in csvs if p.name.lower() == "train.csv"] or csvs
    for csv_path in preferred:
        pairs = _rows_from_labels_csv(csv_path, images_dir)
        if pairs:
            logger.info("aptos: %d rows from %s", len(pairs), csv_path)
            return pairs
    raise FileNotFoundError(f"no usable labels CSV (id_code/diagnosis) under {raw}")


def _collect_eyepacs(raw: Path) -> list[tuple[Path, int]]:
    """EyePACS: ``trainLabels.csv`` (or ``retinopathy_solution.csv``) + flat ``train/``."""
    images_dir = _find_dir(raw, ("train", "images", "train_images"))
    if images_dir is None:
        raise FileNotFoundError(f"no train/ image directory under {raw}")
    csvs = sorted(raw.rglob("*.csv"))
    order = ["trainlabels.csv", "retinopathy_solution.csv"]
    csvs.sort(key=lambda p: order.index(p.name.lower()) if p.name.lower() in order else len(order))
    for csv_path in csvs:
        pairs = _rows_from_labels_csv(csv_path, images_dir)
        if pairs:
            logger.info("eyepacs: %d rows from %s", len(pairs), csv_path)
            return pairs
    raise FileNotFoundError(f"no usable labels CSV (image/level) under {raw}")


def _collect_hf(raw: Path, hf_cfg: dict) -> list[tuple[Path, int]]:
    """Hugging Face snapshots: grade trees, CSV/parquet label files, embedded images."""
    pairs = _collect_grade_tree(raw)
    if pairs:
        return pairs
    # CSV / parquet label files with paths (or, best-effort, embedded image bytes).
    label_files = sorted(
        p
        for p in raw.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet"} and "_hf_images" not in p.parts
    )
    for label_file in label_files:
        try:
            df = (
                pd.read_parquet(label_file)
                if label_file.suffix == ".parquet"
                else pd.read_csv(label_file)
            )
        except Exception as exc:  # noqa: BLE001 - try the next candidate file
            logger.warning("skipping %s (%s)", label_file, exc)
            continue
        image_col = hf_cfg.get("image_column") or next(
            (
                c
                for c in df.columns
                if str(c).lower() in {"image", "image_path", "file_name", "img"}
            ),
            None,
        )
        grade_col = hf_cfg.get("label_column") or next(
            (c for c in df.columns if str(c).lower() in GRADE_ALIASES), None
        )
        if image_col is None or grade_col is None:
            continue
        pairs.extend(_rows_from_labels_csv(label_file, raw))
        if pairs:
            return pairs
        # Embedded image bytes (datasets-style parquet) -> decode via PIL.
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow required to decode embedded HF images") from exc
        if not isinstance(df[image_col].iloc[0], dict):
            continue
        staged = raw / "_hf_images"
        staged.mkdir(parents=True, exist_ok=True)
        for i, cell in enumerate(df[image_col]):
            payload = cell.get("bytes") if isinstance(cell, dict) else None
            if not payload:
                continue
            out = staged / f"{label_file.stem}_{i}.jpg"
            if not out.exists():
                Image.open(io.BytesIO(payload)).convert("RGB").save(out, format="JPEG")
            pairs.append((out, int(df[grade_col].iloc[i])))
        if pairs:
            logger.info("hf: decoded %d embedded image(s) from %s", len(pairs), label_file)
            return pairs
    raise FileNotFoundError(
        f"could not find grade folders or (image, grade) columns under {raw} — "
        "see docs/DATASETS.md for supported HF layouts"
    )


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _dedupe_name(used: set[str], src: Path) -> str:
    """Unique file name within the images dir (prefix with parent dir on clash)."""
    name = src.name
    while name in used:
        name = f"{src.parent.name}_{name}"
    used.add(name)
    return name


def _stage_file(src: Path, dst: Path) -> bool:
    """Hardlink-first copy; returns True when bytes were written."""
    if src.resolve() == dst.resolve():
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    return True


def _normalize(root: Path, pairs: list[tuple[Path, int]], force: bool = False) -> dict:
    """Stage images into ``root/images`` and write ``root/labels.csv``."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    rows: list[tuple[str, int]] = []
    staged = 0
    for src, grade in pairs:
        name = _dedupe_name(used, src)
        dst = images_dir / name
        if force and src.resolve() != dst.resolve() and dst.exists():
            dst.unlink()
        if _stage_file(src, dst):
            staged += 1
        rows.append((name, grade))
    rows.sort(key=lambda row: row[0])
    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "grade"])
        writer.writerows(rows)
    return {"total": len(rows), "staged": staged}


_GRADE_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


def _ensure_readme(root: Path, name: str, cfg: dict, stats: dict) -> None:
    """Create the repo-level ``data/README.md`` (if missing) and a per-dataset one."""
    repo_data_dir = Path("data")
    if not (repo_data_dir / "README.md").exists():
        repo_data_dir.mkdir(parents=True, exist_ok=True)
        (repo_data_dir / "README.md").write_text(
            "# Data directory\n\n"
            "Downloaded at runtime — never committed. See docs/DATASETS.md for "
            "sources and licences.\n\nLayout per dataset: `images/` + `labels.csv` "
            "(columns: image,grade).\n",
            encoding="utf-8",
        )
    per_grade = {g: 0 for g in _GRADE_NAMES}
    with open(root / "labels.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            per_grade[int(row["grade"])] += 1
    lines = [
        f"# Dataset: {name}",
        "",
        f"- source: `{cfg.get('data', {}).get('dataset', name)}` (see docs/DATASETS.md for licence/access)",
        f"- images: {stats['total']} ({stats['staged']} newly staged)",
        "- grade distribution:",
    ]
    lines += [f"  - {grade} ({_GRADE_NAMES[grade]}): {count}" for grade, count in per_grade.items()]
    lines += [
        "",
        "Layout: `images/` + `labels.csv` (columns: image,grade, paths relative to images/).",
        "Splitting into train/val/test is deterministic (hash of the image name) and",
        "happens in `retinaedge.data.dataset` — no split files are stored.",
        "",
        "> Research use only. Respect the upstream dataset licence; never redistribute images.",
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Normalise the source described by ``--config`` into ``data/<name>``."""
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.data.prepare",
        description="Normalise a downloaded dataset into <root>/images + <root>/labels.csv.",
    )
    parser.add_argument("--config", required=True, help="configs/data/<name>.yaml")
    parser.add_argument("--root", default=None, help="override cfg data.root")
    parser.add_argument(
        "--raw", default=None, help="override raw download dir (default: <root>/raw)"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="folder dataset: ImageFolder-like source tree (default: <root>/raw)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-stage images even if already present"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data = cfg.get("data") or {}
    name = data.get("dataset")
    if not name:
        logger.error("cfg['data']['dataset'] is missing in %s", args.config)
        return 1
    root = Path(args.root or data.get("root") or Path("data") / str(name))
    raw = Path(args.raw or root / "raw")

    try:
        if name == "aptos":
            _ensure_extracted(raw)
            pairs = _collect_aptos(raw)
        elif name == "eyepacs":
            _ensure_extracted(raw)
            pairs = _collect_eyepacs(raw)
        elif name == "ddr":
            _ensure_extracted(raw)
            pairs = _collect_grade_tree(raw)
        elif name == "folder":
            source = Path(args.source or (data.get("folder") or {}).get("source") or raw)
            if (source / "labels.csv").exists() and (source / "images").exists():
                pairs = _rows_from_labels_csv(source / "labels.csv", source / "images")
            else:
                pairs = _collect_grade_tree(source)
        elif name == "hf":
            _ensure_extracted(raw)
            pairs = _collect_hf(raw, data.get("hf") or {})
        else:
            logger.error("unsupported dataset %r", name)
            return 1
    except FileNotFoundError as exc:
        logger.error("%s — run the download CLI first (see docs/DATASETS.md)", exc)
        return 1

    if not pairs:
        logger.error("no labelled images found for dataset %r under %s", name, raw)
        return 1

    stats = _normalize(root, pairs, force=args.force)
    _ensure_readme(root, name, cfg, stats)

    per_grade: dict[int, int] = {}
    for _, grade in pairs:
        per_grade[grade] = per_grade.get(grade, 0) + 1
    dist = ", ".join(f"{g}={per_grade.get(g, 0)}" for g in sorted(per_grade))
    logger.info(
        "%s: %d images (%d newly staged) -> %s | grades: %s",
        name,
        stats["total"],
        stats["staged"],
        root,
        dist,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
