"""Datasets and transform pipelines (owner: agent 2-a, contract v1.0).

Public API (docs/INTERFACES.md):
    build_dataset(cfg: dict, split: str, transform=None) -> torch.utils.data.Dataset
    build_train_val_transforms(cfg: dict) -> tuple[train_t, val_t]

Contract:
    * ``split`` in {"train", "val", "test"}.
    * ``cfg["data"]["dataset"]`` in {"synthetic", "aptos", "eyepacs", "ddr", "folder", "hf"}.
    * ``__getitem__`` returns ``(float Tensor CHW normalized, int grade)``.
    * Internally: load RGB uint8 HWC -> optional Ben-Graham preprocess ->
      albumentations transform -> ToTensorV2.

Determinism: every randomness source is derived from ``(cfg seed, split, index)``
via ``numpy.random.default_rng``; no global RNG state is consumed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from retinaedge.data.ben_graham import preprocess_ben_graham
from retinaedge.utils.logging_utils import get_logger

__all__ = [
    "FileListDataset",
    "SyntheticDRDataset",
    "build_dataset",
    "build_train_val_transforms",
    "split_tags",
]

logger = get_logger("data.dataset")

#: ImageNet normalisation — must match the export/Android contract exactly.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Per-grade sampling probabilities for the synthetic dataset (contract).
GRADE_DISTRIBUTION = (0.45, 0.20, 0.15, 0.10, 0.10)

#: Split offsets for ``numpy.random.default_rng(seed + split_offset)`` (contract).
SPLIT_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

_FILE_DATASETS = frozenset({"aptos", "eyepacs", "ddr", "folder", "hf"})
_VALID_SPLITS = frozenset({"train", "val", "test"})


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_train_val_transforms(cfg: dict) -> tuple[A.Compose, A.Compose]:
    """Build the albumentations train/val pipelines from ``cfg["data"]``.

    train = CLAHE(prob=clahe_prob) + Resize(longest 256) + RandomResizedCrop
    (img_size, scale=(0.8,1.0)) + h/v flip + RandomRotate90 + ColorJitter +
    Normalize(ImageNet) + ToTensorV2;
    val  = CLAHE(prob=clahe_prob) + Resize(longest 256) + CenterCrop(img_size)
    + Normalize + ToTensorV2.

    Config keys (all optional): ``img_size`` (224), ``clahe_prob`` (0.0),
    ``resize_longest`` (256).
    """
    data = cfg.get("data", {}) or {}
    img_size = int(data.get("img_size", 224))
    clahe_prob = float(data.get("clahe_prob", 0.0))
    longest = int(data.get("resize_longest", 256))

    clahe = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=clahe_prob)
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    to_tensor = ToTensorV2()

    train_t = A.Compose(
        [
            clahe,
            A.LongestMaxSize(max_size=longest),
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            normalize,
            to_tensor,
        ]
    )
    val_t = A.Compose(
        [
            clahe,
            A.LongestMaxSize(max_size=longest),
            # pad_if_needed: fundus photos are rarely square; longest side is
            # capped at `longest`, so the short side may be < img_size.
            A.CenterCrop(height=img_size, width=img_size, pad_if_needed=True),
            normalize,
            to_tensor,
        ]
    )
    return train_t, val_t


def _fallback_transform() -> A.Compose:
    """Normalize-only transform used when a Dataset is built with ``transform=None``."""
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


# --------------------------------------------------------------------------- #
# Split assignment (deterministic, stateless)
# --------------------------------------------------------------------------- #
def split_tags(
    names: Sequence[str], val_fraction: float = 0.15, test_fraction: float = 0.0, salt: str = ""
) -> dict[str, str]:
    """Assign each image name to a split via an md5 hash of its name.

    Deterministic across runs/processes (no state file, stable order), and
    roughly class-balanced for large datasets. ``salt`` (``data.split_salt``)
    reshuffles the assignment without touching the data.
    """
    tags: dict[str, str] = {}
    for name in names:
        digest = hashlib.md5(f"{salt}/{name}".encode()).hexdigest()
        u = int(digest[:8], 16) / float(16**8)
        if u < test_fraction:
            tags[name] = "test"
        elif u < test_fraction + val_fraction:
            tags[name] = "val"
        else:
            tags[name] = "train"
    return tags


# --------------------------------------------------------------------------- #
# Synthetic dataset (procedural, no files on disk)
# --------------------------------------------------------------------------- #
def _render_synthetic_fundus(size: int, grade: int, seed: int) -> np.ndarray:
    """Render a fake fundus image whose appearance shifts with DR grade.

    Higher grades are slightly paler (less red), carry more dark lesion blobs
    and bright exudate-like spots, and receive stronger vignetting. All draws
    come from a per-image ``default_rng(seed)`` — index-deterministic under any
    DataLoader shuffling/worker layout.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    cx = rng.uniform(0.42, 0.58) * size
    cy = rng.uniform(0.42, 0.58) * size
    r = rng.uniform(0.36, 0.44) * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    inside = dist <= r

    img = rng.normal(5.0, 3.0, (size, size, 3)).astype(np.float32)  # dark background
    # Base retinal colour: severity drains redness / adds a yellow tint.
    base = np.array([150.0 - 8.0 * grade, 70.0 + 5.0 * grade, 52.0 + 4.0 * grade], dtype=np.float32)
    vignette = 1.0 - 0.35 * (dist / max(r, 1.0)) ** 2
    tint = rng.uniform(0.92, 1.08, (3,)).astype(np.float32)
    pixels = base[None, None, :] * tint[None, None, :] * vignette[..., None]
    img[inside] = pixels[inside]
    img[inside] += rng.normal(0.0, 10.0, (int(inside.sum()), 3)).astype(np.float32)

    # Lesion-like blobs: dark dots (microaneurysm/haemorrhage proxies) and
    # bright spots (exudate proxies), both increasing with grade.
    half = max(size // 32, 1)
    for _ in range(int(rng.poisson(2.5 * grade + 1))):
        bx, by = rng.uniform(cx - r, cx + r), rng.uniform(cy - r, cy + r)
        rad = rng.uniform(1.0, half)
        spot = (xx - bx) ** 2 + (yy - by) ** 2 <= rad**2
        img[spot] -= rng.uniform(25.0, 60.0)
    for _ in range(int(rng.poisson(1.5 * grade))):
        bx, by = rng.uniform(cx - r, cx + r), rng.uniform(cy - r, cy + r)
        rad = rng.uniform(1.0, half)
        spot = (xx - bx) ** 2 + (yy - by) ** 2 <= rad**2
        img[spot] += rng.uniform(25.0, 55.0)

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


class SyntheticDRDataset(Dataset):
    """Procedural DR dataset — no files on disk (contract: ``dataset: synthetic``).

    Grades are drawn once at construction from
    ``numpy.random.default_rng(seed + SPLIT_OFFSETS[split])`` with the class
    distribution ``GRADE_DISTRIBUTION``; image pixels are derived from a
    per-index seeded rng, so the dataset is fully deterministic.
    """

    def __init__(
        self,
        n: int,
        size: int = 64,
        seed: int = 0,
        split: str = "train",
        transform: A.Compose | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        self.split = split
        self.size = int(size)
        self.transform = transform if transform is not None else _fallback_transform()
        rng = np.random.default_rng(int(seed) + SPLIT_OFFSETS[split])
        self.grades = rng.choice(len(GRADE_DISTRIBUTION), size=n, p=GRADE_DISTRIBUTION).astype(
            np.int64
        )
        self._seed = int(seed)

    def __len__(self) -> int:
        return int(self.grades.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return ``(float CHW tensor normalized, int grade)`` for sample ``idx``."""
        grade = int(self.grades[idx])
        seed = self._seed + SPLIT_OFFSETS[self.split] + 7919 * (int(idx) + 1)
        img = _render_synthetic_fundus(self.size, grade, seed)
        tensor = self.transform(image=img)["image"]
        return tensor, grade


# --------------------------------------------------------------------------- #
# File-backed dataset (aptos / eyepacs / ddr / folder / hf after `prepare`)
# --------------------------------------------------------------------------- #
class FileListDataset(Dataset):
    """Reads a prepared ``images/`` + ``labels.csv`` layout.

    ``labels.csv`` columns: ``image,grade`` (paths relative to ``images/``).
    Rows whose image file is missing are dropped with a warning (documented
    behaviour — `prepare` guarantees this does not happen for its own output).
    """

    def __init__(
        self,
        root: str | Path,
        image_names: Sequence[str],
        grades: Sequence[int],
        transform: A.Compose | None = None,
        use_ben_graham: bool = False,
        ben_graham_radius: int = 300,
    ) -> None:
        if len(image_names) != len(grades):
            raise ValueError(f"names/grades length mismatch: {len(image_names)} vs {len(grades)}")
        self.root = Path(root)
        self.transform = transform if transform is not None else _fallback_transform()
        self.use_ben_graham = bool(use_ben_graham)
        self.ben_graham_radius = int(ben_graham_radius)

        self.samples: list[tuple[Path, int]] = []
        missing = 0
        for name, grade in zip(image_names, grades, strict=True):
            path = self.root / "images" / str(name)
            if path.exists():
                self.samples.append((path, int(grade)))
            else:
                missing += 1
        if missing:
            logger.warning("%s: dropping %d entries with missing image files", self.root, missing)
        if not self.samples:
            raise FileNotFoundError(f"no readable images under {self.root / 'images'}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.use_ben_graham:
            rgb = preprocess_ben_graham(rgb, radius=self.ben_graham_radius)
        return rgb

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return ``(float CHW tensor normalized, int grade)`` for sample ``idx``."""
        path, grade = self.samples[idx]
        tensor = self.transform(image=self._load(path))["image"]
        return tensor, grade


def _build_file_dataset(cfg: dict, split: str, transform: A.Compose | None) -> FileListDataset:
    import pandas as pd  # local import keeps module import light for synthetic-only runs

    data = cfg.get("data", {}) or {}
    name = str(data["dataset"])
    root = Path(data.get("root") or Path("data") / name)
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} not found — run `python -m retinaedge.data.download_kaggle ...` "
            f"then `python -m retinaedge.data.prepare --config <cfg>` first"
        )
    # dtype=str keeps leading-zero ids intact; grades are parsed explicitly below.
    df = pd.read_csv(labels_csv, dtype=str, keep_default_na=False)
    for col in ("image", "grade"):
        if col not in df.columns:
            raise ValueError(f"{labels_csv}: missing required column {col!r}")

    grades = pd.to_numeric(df["grade"], errors="coerce")
    labelled = [
        (str(img), int(g))
        for img, g in zip(df["image"], grades, strict=True)
        if not pd.isna(g) and 0 <= int(g) <= 4
    ]
    readable = [(str(img), int(g)) for img, g in labelled if (root / "images" / img).exists()]
    if len(readable) < len(labelled):
        logger.warning(
            "%s: dropping %d entries with missing image files", root, len(labelled) - len(readable)
        )
    if not readable:
        raise FileNotFoundError(f"no readable images under {root / 'images'}")

    # Curated datasets may ship their own (e.g. patient-aware) splits via an
    # optional ``split`` column — honoring them prevents same-patient leakage
    # that hash-based name splits cannot see. Synonyms are normalized; rows
    # with unknown tags are excluded from every split (counted in a warning).
    override_tags: dict[str, str] | None = None
    if "split" in df.columns:
        norm = (
            df["split"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"validation": "val", "valid": "val", "": "unknown"})
        )
        allowed = {"train", "val", "test"}
        override_tags, unknown = {}, 0
        for img, tag in zip(df["image"].astype(str), norm, strict=True):
            if tag in allowed:
                override_tags.setdefault(img, tag)
            else:
                unknown += 1
        if unknown:
            logger.warning("%s: %d rows have unknown/empty split tags", labels_csv, unknown)

    if override_tags is not None:
        keep = [(img, grade) for img, grade in readable if override_tags.get(img) == split]
    else:
        tags = split_tags(
            [img for img, _ in readable],
            val_fraction=float(data.get("val_fraction", 0.15)),
            test_fraction=float(data.get("test_fraction", 0.0)),
            salt=str(data.get("split_salt", "")),
        )
        keep = [(img, grade) for img, grade in readable if tags[img] == split]
    if not keep:
        raise ValueError(
            f"split {split!r} is empty for {root} (val_fraction={data.get('val_fraction')}, "
            f"test_fraction={data.get('test_fraction')}) — adjust the config"
        )
    if data.get("limit"):
        keep = keep[: int(data["limit"])]
    return FileListDataset(
        root=root,
        image_names=[img for img, _ in keep],
        grades=[grade for _, grade in keep],
        transform=transform,
        use_ben_graham=bool(data.get("ben_graham", False))
        and name in {"aptos", "eyepacs", "ddr", "folder"},
        ben_graham_radius=int(data.get("ben_graham_radius", 300)),
    )


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #
def build_dataset(cfg: dict, split: str, transform: A.Compose | None = None) -> Dataset:
    """Build the dataset described by ``cfg`` for ``split``.

    Args:
        cfg: Full experiment config; uses ``cfg["data"]``.
        split: One of ``{"train", "val", "test"}``.
        transform: Pre-built albumentations pipeline; when ``None`` the
            pipeline from :func:`build_train_val_transforms` is selected by
            split (train pipeline for "train", val pipeline otherwise).

    Returns:
        A ``Dataset`` whose ``__getitem__`` yields ``(float Tensor CHW
        normalized, int grade)`` per the interface contract.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
    data = cfg.get("data", {}) or {}
    name = data.get("dataset")
    if not name:
        raise ValueError("cfg['data']['dataset'] is required")

    if transform is None:
        train_t, val_t = build_train_val_transforms(cfg)
        transform = train_t if split == "train" else val_t

    if name == "synthetic":
        syn = data.get("synthetic", {}) or {}
        key = f"n_{split}"
        if key not in syn:
            raise ValueError(f"cfg['data']['synthetic']['{key}'] is required for split {split!r}")
        size = int(syn.get("size", data.get("img_size", 64)))
        return SyntheticDRDataset(
            n=int(syn[key]),
            size=size,
            seed=int(syn.get("seed", 0)),
            split=split,
            transform=transform,
        )
    if name in _FILE_DATASETS:
        return _build_file_dataset(cfg, split, transform)
    raise ValueError(
        f"unknown dataset {name!r} — expected one of {sorted({'synthetic'} | _FILE_DATASETS)}"
    )
