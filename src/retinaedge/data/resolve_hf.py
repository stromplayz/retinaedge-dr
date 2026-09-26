"""Runtime dataset resolver: find a real DR/fundus dataset on the Hugging Face Hub.

Used by CI (``train-online.yml``) so that training runs *inside GitHub* on real
public data without any committed images or manual downloads:

    python -m retinaedge.data.resolve_hf --out data/online_dr \
        [--repo <hf_dataset_id> ...] [--max-images N] [--bake-ben-graham] \
        [--max-bytes 4000000000] [--json artifacts/online/provenance.json]

Strategy
--------
1. Candidates = explicit ``--repo`` ids, then Hub searches (downloads-sorted):
   aptos / diabetic retinopathy / retinopathy / messidor / fundus / eyepacs / idrid.
2. Skip candidates larger than ``--max-bytes`` (guards against EyePACS-scale pulls).
3. Streaming-inspect the first rows: must contain an image column AND a grade
   column that maps to the ICDRSS 0-4 scheme.
4. Full download, then normalize into the repository-wide layout::

       <out>/images/000001.jpg ...
       <out>/labels.csv   (columns: image,grade)
       <out>/provenance.json (source repo, revision, license, mapping, counts)

Binary referable labels (0/1) are conservatively expanded to (0/2) so they stay
usable with the ordinal head; the scheme used is recorded in provenance.json.

Dependencies (``pip install -e ".[hf]"``): datasets, huggingface_hub, Pillow.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from retinaedge.utils.logging_utils import get_logger

__all__ = ["UnmappableGrades", "map_labels_to_icdrss", "resolve_dataset"]

logger = get_logger("data.resolve_hf")

#: Search terms, tried in order (small, well-labelled DR sets first).
DEFAULT_SEARCHES = (
    "aptos",
    "diabetic retinopathy",
    "retinopathy grading",
    "messidor",
    "idrid",
    "eyepacs",
    "retinal fundus",
)

#: Column names that plausibly hold the DR grade, by priority.
GRADE_COLUMN_CANDIDATES = (
    "label",
    "grade",
    "dr_grade",
    "dr-grade",
    "drlevel",
    "dr_level",
    "diagnosis",
    "level",
    "target",
    "retinopathy_grade",
    "class",
    "category",
)

NAME_TO_GRADE: dict[str, int] = {
    "no dr": 0,
    "nodr": 0,
    "no_dr": 0,
    "no retinopathy": 0,
    "normal": 0,
    "healthy": 0,
    "grade 0": 0,
    "r0": 0,
    "mild": 1,
    "mild npdr": 1,
    "mild retinopathy": 1,
    "grade 1": 1,
    "r1": 1,
    "moderate": 2,
    "moderate npdr": 2,
    "moderate retinopathy": 2,
    "grade 2": 2,
    "r2": 2,
    "severe": 3,
    "severe npdr": 3,
    "severe retinopathy": 3,
    "grade 3": 3,
    "r3": 3,
    "proliferative": 4,
    "proliferative dr": 4,
    "proliferative retinopathy": 4,
    "pdr": 4,
    "grade 4": 4,
    "r4": 4,
}

MAX_ROWS_DEFAULT = 60_000

#: Canonical APTOS 2019 train distribution per ICDRSS grade 0..4 (public knowledge).
#: Some HF mirrors store grades as ALPHABETICAL class indices (Mild=0, Moderate=1,
#: NoDR=2, PDR=3, Severe=4); the multiset of counts matches this reference exactly,
#: which lets us detect and repair the permutation deterministically.
APTOS_REFERENCE_COUNTS = (1805, 370, 999, 194, 294)

#: alphabetical index -> ICDRSS grade (Mild, Moderate, NoDR, PDR, Severe)
ALPHA_INDEX_TO_GRADE = {0: 1, 1: 2, 2: 0, 3: 4, 4: 3}


def repair_aptos_permutation(
    counts: Sequence[int],
    reference: Sequence[int] = APTOS_REFERENCE_COUNTS,
    min_match: float = 0.97,
) -> list[int] | None:
    """If ``counts`` is a permutation of the canonical APTOS distribution, recover it.

    Returns the permutation ``p`` (alpha/stored index -> true ICDRSS grade) when the
    multiset matches the reference with >= ``min_match`` total agreement, else None.
    """
    from itertools import permutations

    if len(counts) != 5 or len(reference) != 5 or sum(counts) != sum(reference):
        return None
    best_perm, best_score = None, 0.0
    for perm in permutations(range(5)):
        score = sum(min(counts[i], reference[perm[i]]) for i in range(5)) / sum(reference)
        if score > best_score:
            best_perm, best_score = list(perm), score
    return best_perm if best_score >= min_match else None


class UnmappableGrades(ValueError):
    """Raised when a candidate's label column cannot be mapped to ICDRSS 0-4."""


def map_labels_to_icdrss(raw: Sequence[Any]) -> tuple[list[int], str]:
    """Map arbitrary grade labels to ICDRSS 0-4.

    Returns ``(grades, scheme)`` where scheme describes the transformation, or
    raises :class:`UnmappableGrades`. Supported schemes:

    * ``icdrss5``        — integers already in 0..4
    * ``binary_expanded``— {0,1} referable flags, 1 conservatively -> 2
    * ``names``          — strings like "No DR"/"Mild"/"Moderate"/"Severe"/"Proliferative"
    """
    if not raw:
        raise UnmappableGrades("empty label column")

    def _as_int(value: Any) -> int | None:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return int(round(f)) if f == int(f) else None

    ints = [_as_int(v) for v in raw]
    if all(i is not None for i in ints):
        unique = {i for i in ints if i is not None}
        if unique == {0, 1}:
            # {0,1}-valued fundus columns are almost always referable flags —
            # expand 1 -> 2 (conservative, keeps the ordinal head meaningful).
            # The chosen scheme is always recorded in provenance.json.
            return [2 if i == 1 else 0 for i in ints if i is not None], "binary_expanded"
        if unique and unique <= set(range(5)):
            return [int(i) for i in ints if i is not None], "icdrss5"
        raise UnmappableGrades(f"integer labels outside 0-4: {sorted(unique)[:8]}")

    lowered: list[int] = []
    for value in raw:
        key = str(value).strip().lower().replace("-", " ").replace("_", " ")
        key = " ".join(key.split())
        if key not in NAME_TO_GRADE:
            raise UnmappableGrades(f"unrecognised grade label: {value!r}")
        lowered.append(NAME_TO_GRADE[key])
    return lowered, "names"


# --------------------------------------------------------------------------- #
# Hub access (lazy imports: module stays importable without the hf extra)
# --------------------------------------------------------------------------- #
def _hf_imports():
    try:
        from datasets import ClassLabel, load_dataset
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            'Missing HF dependencies — install with: pip install -e ".[hf]" '
            "(packages: datasets, huggingface_hub)"
        ) from exc
    return ClassLabel, HfApi, HfHubHTTPError, load_dataset


def _candidate_ids(api: Any, searches: Sequence[str], limit: int) -> Iterator[str]:
    seen: set[str] = set()
    for term in searches:
        try:
            results = api.list_datasets(search=term, sort="downloads", limit=limit)
            ids = [d.id for d in results]
        except Exception as exc:  # noqa: BLE001 - one failed search must not abort
            logger.warning("search %r failed: %s", term, exc)
            continue
        logger.info("search %-24r -> %d candidates", term, len(ids))
        for ds_id in ids:
            if ds_id not in seen:
                seen.add(ds_id)
                yield ds_id


def _dataset_bytes(api: Any, ds_id: str) -> int | None:
    """Best-effort total download size in bytes (None when unknowable)."""
    try:
        info = api.dataset_info(ds_id, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  info failed for %s: %s", ds_id, exc)
        return None
    total = getattr(info, "used_storage", None)
    if not total:
        total = sum(s.size or 0 for s in (info.siblings or []) if s.size)
    return int(total) if total else None


def _split_label_column(features: Any, row: dict) -> str | None:
    """Pick the grade column from a row + its dataset features."""
    for name in GRADE_COLUMN_CANDIDATES:
        if name in row:
            return name
    fallbacks = []
    for key, value in row.items():
        if isinstance(value, (int, float, str)) or hasattr(value, "item"):
            fallbacks.append(key)
    return fallbacks[0] if fallbacks else None


def _row_image(row: dict, image_col: str):
    """Decode the image from a datasets row (PIL Image | dict | path) or None."""
    from PIL import Image

    value = row.get(image_col)
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw))
        path = value.get("path")
        if path and Path(path).exists():
            return Image.open(path)
    if isinstance(value, str) and Path(value).exists():
        return Image.open(value)
    return None


def _pick_columns(features: Any, row: dict) -> tuple[str | None, str | None]:
    """Pick the image column and grade column from a row + dataset features."""
    image_col = None
    keys = list(row.keys())
    if hasattr(features, "keys"):
        keys = list(dict.fromkeys([*features.keys(), *keys]))
    for key in keys:
        feat = features.get(key) if hasattr(features, "get") else None
        value = row.get(key)
        is_image = (
            type(feat).__name__ == "Image"
            or isinstance(value, dict)
            and ("bytes" in value or "path" in value)
            or isinstance(value, str)
            and Path(value).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if is_image:
            image_col = key
            break
    label_col = _split_label_column(features, row)
    return image_col, label_col


def _probe_candidate(ds_id: str, load_dataset) -> dict | None:
    """Streaming-inspect one candidate. Returns probe info or None if unusable."""
    try:
        streamed = load_dataset(ds_id, streaming=True, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  load(stream) failed for %s: %s", ds_id, str(exc)[:160])
        return None
    for split_name, split_ds in streamed.items():
        try:
            row = next(iter(split_ds))
        except StopIteration:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("  stream read failed for %s[%s]: %s", ds_id, split_name, str(exc)[:120])
            continue
        features = split_ds.features
        image_col, label_col = _pick_columns(features, row)
        if image_col is None or label_col is None:
            logger.info(
                "  %s[%s]: missing image/label column (image=%s label=%s)",
                ds_id,
                split_name,
                image_col,
                label_col,
            )
            continue
        if _row_image(row, image_col) is None:
            logger.info("  %s[%s]: image column %r not decodable", ds_id, split_name, image_col)
            continue
        return {
            "split": split_name,
            "image_col": image_col,
            "label_col": label_col,
            "features": features,
        }
    return None


def _labels_from_features(features: Any, label_col: str) -> tuple[list[int], str] | None:
    """Remap a ClassLabel feature through its names (e.g. ['No DR','Mild',...])."""
    feat = features.get(label_col) if hasattr(features, "get") else None
    if feat is None or type(feat).__name__ != "ClassLabel":
        return None
    try:
        return map_labels_to_icdrss(list(feat.names))
    except UnmappableGrades:
        return None


# --------------------------------------------------------------------------- #
# Normalization / output
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Resolved:
    """Provenance record for the dataset that was resolved and normalized."""

    repo_id: str
    revision: str
    license: str | None
    split: str
    image_col: str
    label_col: str
    scheme: str
    n_images: int
    grade_counts: dict[str, int]
    baked_ben_graham: bool
    out_dir: str


def _write_layout(
    rows: Iterable[dict],
    image_col: str,
    label_col: str,
    remap: dict | None,
    out: Path,
    max_images: int,
    bake_ben_graham: bool,
) -> tuple[int, Counter]:
    """Write ``images/`` + ``labels.csv`` from a HF split. Returns (n, grade counter)."""

    import cv2
    import numpy as np
    from PIL import Image

    if bake_ben_graham:
        from retinaedge.data.ben_graham import preprocess_ben_graham

    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    counter: Counter = Counter()
    written = 0
    labels_path = out / "labels.csv"
    with open(labels_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("image,grade\n")
        for row in rows:
            if written >= max_images:
                break
            grade = row.get("_grade")
            if grade is None:
                continue
            pil = _row_image(row, image_col)
            if pil is None:
                continue
            rgb = pil.convert("RGB")
            arr = np.asarray(rgb) if bake_ben_graham else None
            name = f"{written + 1:06d}.jpg"
            if bake_ben_graham:
                try:
                    arr = preprocess_ben_graham(arr)
                except Exception:  # noqa: BLE001 - keep degenerate images raw
                    arr = np.asarray(rgb)
                rgb = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            rgb.save(images_dir / name, "JPEG", quality=92)
            fh.write(f"{name},{int(grade)}\n")
            counter[int(grade)] += 1
            written += 1
    _ = cv2  # keep import for the bake path readability
    return written, counter


def resolve_dataset(
    out: str | Path,
    repos: Sequence[str] = (),
    searches: Sequence[str] = DEFAULT_SEARCHES,
    max_images: int = MAX_ROWS_DEFAULT,
    max_bytes: int = 4_000_000_000,
    bake_ben_graham: bool = False,
    probe_rows: int = 40,
    label_map: str = "auto",
) -> Resolved:
    """Search → validate → download → normalize one DR dataset. See module docstring."""
    ClassLabel, HfApi, _HfHubHTTPError, load_dataset = _hf_imports()
    _ = ClassLabel
    api = HfApi()
    out = Path(out)

    candidates = [*repos, *_candidate_ids(api, searches, limit=25)]
    logger.info("resolving dataset -> %s (%d candidates, explicit first)", out, len(candidates))

    for ds_id in candidates:
        if not ds_id:
            continue
        logger.info("candidate: %s", ds_id)
        size = _dataset_bytes(api, ds_id)
        if size and size > max_bytes:
            logger.info("  skip: too large (%.2f GB > %.2f GB)", size / 1e9, max_bytes / 1e9)
            continue
        probe = _probe_candidate(ds_id, load_dataset)
        if probe is None:
            continue

        features = probe["features"]
        label_col = probe["label_col"]
        feature_map = _labels_from_features(features, label_col)
        remap: dict | None = None
        if feature_map is not None:
            scheme = feature_map[1]
        else:
            try:
                probe_labels = [
                    row.get(label_col)
                    for row in _safe_take(
                        load_dataset(ds_id, streaming=True, trust_remote_code=False)[
                            probe["split"]
                        ],
                        probe_rows,
                    )
                ]
                _, scheme = map_labels_to_icdrss([v for v in probe_labels if v is not None])
            except (UnmappableGrades, Exception) as exc:  # noqa: BLE001
                logger.info("  reject: labels unmappable (%s)", str(exc)[:120])
                continue

        info = api.dataset_info(ds_id)
        try:
            full = load_dataset(ds_id, revision=info.sha, trust_remote_code=False)
        except Exception:
            try:
                full = load_dataset(ds_id, trust_remote_code=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  full load failed for %s: %s", ds_id, str(exc)[:160])
                continue
        split_name = (
            probe["split"]
            if probe["split"] in full
            else max(full.keys(), key=lambda k: len(full[k]))
        )
        ds = full[split_name]
        if len(ds) > max_images:
            logger.info("  truncating %d -> %d rows", len(ds), max_images)
            ds = ds.select(range(max_images))

        if feature_map is not None:
            remap = {
                old: new
                for old, new in zip(range(len(feature_map[0])), feature_map[0], strict=True)
            }
        ds = ds.with_format(None)
        # attach mapped grades as a plain column so the writer stays simple

        def _grade_of(value: Any, _remap=remap) -> int | None:
            if _remap is not None and isinstance(value, int):
                return _remap.get(value)
            try:
                grades_list, _ = map_labels_to_icdrss([value])
                return grades_list[0]
            except UnmappableGrades:
                return None

        grades_col = [_grade_of(v) for v in ds[label_col]]
        scheme_used = scheme
        present = [g for g in grades_col if g is not None]
        if label_map == "alpha":
            # Force the alphabetical-index -> ICDRSS mapping (Mild=0, Moderate=1,
            # NoDR=2, PDR=3, Severe=4 as stored by some mirrors).
            grades_col = [
                ALPHA_INDEX_TO_GRADE.get(g) if g is not None else None for g in grades_col
            ]
            scheme_used = "alpha_label_map"
        elif label_map == "auto" and "aptos" in ds_id.lower() and len(set(present)) == 5:
            from collections import Counter as _Counter

            counts = _Counter(present)
            perm = repair_aptos_permutation([counts.get(i, 0) for i in range(5)])
            if perm is not None:
                grades_col = [perm[g] if g is not None else None for g in grades_col]
                scheme_used = "icdrss5_aptos_permutation_repaired"
                logger.info("  applied APTOS permutation repair %s", perm)
        ds = ds.add_column("_grade", grades_col)
        ds = ds.filter(lambda row: row["_grade"] is not None)

        out.mkdir(parents=True, exist_ok=True)
        n, counter = _write_layout(
            iter(ds), probe["image_col"], "_grade", None, out, max_images, bake_ben_graham
        )
        if n < min(200, max_images):
            logger.info("  reject: only %d usable images", n)
            continue
        if len(counter) < 2:
            logger.info("  reject: single-class dataset (%s)", dict(counter))
            continue

        resolved = Resolved(
            repo_id=ds_id,
            revision=str(info.sha or "unknown"),
            license=(info.cardData.get("license") if info.cardData else None),
            split=split_name,
            image_col=probe["image_col"],
            label_col=label_col,
            scheme=scheme_used,
            n_images=n,
            grade_counts={str(k): int(v) for k, v in sorted(counter.items())},
            baked_ben_graham=bake_ben_graham,
            out_dir=str(out),
        )
        with open(out / "provenance.json", "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(resolved), fh, indent=2)
        logger.info(
            "RESOLVED %s -> %d images %s (scheme=%s, license=%s)",
            ds_id,
            n,
            dict(counter),
            resolved.scheme,
            resolved.license,
        )
        return resolved

    raise SystemExit(
        "No usable DR dataset found on the Hugging Face Hub. "
        "Pass an explicit dataset id: --repo <org/dataset> (verify it has image + grade columns)."
    )


def _safe_take(iterable_ds: Any, k: int) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(iterable_ds):
        if i >= k:
            break
        out.append(row)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.data.resolve_hf",
        description="Find and normalize a real DR/fundus dataset from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--out", required=True, help="output dir (images/ + labels.csv + provenance.json)"
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="explicit HF dataset id (repeatable, tried first)",
    )
    parser.add_argument("--max-images", type=int, default=MAX_ROWS_DEFAULT)
    parser.add_argument("--max-bytes", type=int, default=4_000_000_000)
    parser.add_argument(
        "--bake-ben-graham",
        action="store_true",
        help="apply Ben-Graham preprocess once at resolve time (saves CPU per epoch)",
    )
    parser.add_argument(
        "--label-map",
        choices=("auto", "none", "alpha"),
        default="auto",
        help="auto: repair known APTOS permutations; alpha: force alphabetical-index map; none: trust ints",
    )
    parser.add_argument("--json", default=None, help="also copy provenance.json to this path")
    args = parser.parse_args(argv)

    resolved = resolve_dataset(
        out=args.out,
        repos=args.repo,
        max_images=args.max_images,
        max_bytes=args.max_bytes,
        bake_ben_graham=args.bake_ben_graham,
        label_map=args.label_map,
    )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(dataclasses.asdict(resolved), indent=2), encoding="utf-8"
        )
    print(json.dumps(dataclasses.asdict(resolved), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
