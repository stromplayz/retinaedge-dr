"""Tests for retinaedge.data — network-free, CPU-only, all data synthetic/tmp."""

from __future__ import annotations

import csv
import io
import zipfile

import cv2
import numpy as np
import pytest
import torch
import yaml

from retinaedge.data.ben_graham import preprocess_ben_graham
from retinaedge.data.dataset import (
    FileListDataset,
    SyntheticDRDataset,
    build_dataset,
    build_train_val_transforms,
    split_tags,
)
from retinaedge.data.prepare import main as prepare_main


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fundus(h: int, w: int, cy: int, cx: int, radius: int, disc: int = 180) -> np.ndarray:
    """Synthetic fundus: bright disc on a black background (uint8 RGB)."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    img[mask] = (disc, disc // 2, disc // 3)
    return img


def _write_image(path, grade_dir: str, name: str, size: int = 24) -> str:
    folder = path / grade_dir
    folder.mkdir(parents=True, exist_ok=True)
    img = _fundus(size, size, size // 2, size // 2, size // 3)
    cv2.imwrite(str(folder / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return name


# --------------------------------------------------------------------------- #
# ben_graham
# --------------------------------------------------------------------------- #
def test_ben_graham_scales_fundus_radius() -> None:
    # image exactly spans the disc bounding box (300x300, disc radius 150)
    img = _fundus(300, 300, 150, 150, radius=150)
    out = preprocess_ben_graham(img, radius=300)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[2] == 3
    # fundus radius 150 -> scaled to 300 => square output of 2*300
    assert out.shape[0] == 600 and out.shape[1] == 600


def test_ben_graham_clahe_changes_pixels_and_handles_degenerate() -> None:
    img = _fundus(200, 300, 100, 150, radius=80)
    out = preprocess_ben_graham(img, radius=100)
    assert out.size > 0 and out.shape[0] == out.shape[1]
    degenerate = np.full((64, 64, 3), 7, dtype=np.uint8)  # flat image -> no crash
    flat = preprocess_ben_graham(degenerate, radius=50)
    assert flat.shape[0] == flat.shape[1] > 0
    float_img = img.astype(np.float32)  # non-uint8 input is coerced
    assert preprocess_ben_graham(float_img, radius=50).dtype == np.uint8


# --------------------------------------------------------------------------- #
# split_tags
# --------------------------------------------------------------------------- #
def test_split_tags_deterministic_and_disjoint() -> None:
    names = [f"img_{i:04d}.png" for i in range(500)]
    tags1 = split_tags(names, val_fraction=0.2, test_fraction=0.1)
    tags2 = split_tags(names, val_fraction=0.2, test_fraction=0.1)
    assert tags1 == tags2  # deterministic, stateless
    assert set(tags1) == set(names)
    train = {n for n in names if tags1[n] == "train"}
    val = {n for n in names if tags1[n] == "val"}
    test = {n for n in names if tags1[n] == "test"}
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) > len(val) > 0 and len(test) > 0  # rough proportions
    assert 0.6 < len(train) / 500 < 0.75
    salted = split_tags(names, val_fraction=0.2, test_fraction=0.1, salt="s1")
    assert any(salted[n] != tags1[n] for n in names)  # salt reshuffles


# --------------------------------------------------------------------------- #
# synthetic dataset
# --------------------------------------------------------------------------- #
def _synth_cfg(n_train=6, n_val=4, size=32, seed=0) -> dict:
    return {
        "data": {
            "dataset": "synthetic",
            "img_size": 32,
            "clahe_prob": 0.0,
            "synthetic": {"n_train": n_train, "n_val": n_val, "size": size, "seed": seed},
        }
    }


def test_synthetic_dataset_contract() -> None:
    ds = build_dataset(_synth_cfg(), "train")
    assert isinstance(ds, SyntheticDRDataset) and len(ds) == 6
    img, grade = ds[0]
    assert isinstance(img, torch.Tensor) and img.dtype == torch.float32
    assert img.shape == (3, 32, 32)
    assert isinstance(grade, int) and 0 <= grade <= 4


def test_synthetic_grades_distribution_and_determinism() -> None:
    ds = SyntheticDRDataset(n=2000, size=8, seed=0, split="train")
    counts = np.bincount(ds.grades, minlength=5) / len(ds)
    expected = np.array([0.45, 0.20, 0.15, 0.10, 0.10])
    assert np.allclose(counts, expected, atol=0.06)
    a = SyntheticDRDataset(n=4, size=16, seed=3, split="val")
    b = SyntheticDRDataset(n=4, size=16, seed=3, split="val")
    assert (a.grades == b.grades).all()
    for i in range(4):  # per-index rendering is deterministic
        assert torch.equal(a[i][0], b[i][0])
    c = SyntheticDRDataset(n=4, size=16, seed=4, split="val")
    assert a.grades.sum() != c.grades.sum() or not torch.equal(a[0][0], c[0][0])


def test_synthetic_split_offsets_differ() -> None:
    a = SyntheticDRDataset(n=4, size=16, seed=0, split="train")
    b = SyntheticDRDataset(n=4, size=16, seed=0, split="val")
    assert not np.array_equal(a.grades, b.grades) or not torch.equal(a[0][0], b[0][0])


def test_synthetic_invalid_split_and_missing_n() -> None:
    with pytest.raises(ValueError):
        SyntheticDRDataset(n=2, split="dev")
    cfg = _synth_cfg()
    with pytest.raises(ValueError):
        build_dataset(cfg, "test")  # no n_test in config


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_transforms_shapes_and_types() -> None:
    train_t, val_t = build_train_val_transforms(
        {"data": {"img_size": 32, "clahe_prob": 0.5, "resize_longest": 64}}
    )
    img = _fundus(300, 200, 150, 100, 90)
    out_v = val_t(image=img)["image"]
    assert out_v.shape == (3, 32, 32) and out_v.dtype == torch.float32
    assert -2.2 <= out_v.min() <= out_v.max() <= 2.8  # ImageNet-normalised range
    out_t = train_t(image=img)["image"]
    assert out_t.shape == (3, 32, 32)


def test_val_transform_pads_small_images() -> None:
    _, val_t = build_train_val_transforms(
        {"data": {"img_size": 64, "clahe_prob": 0.0, "resize_longest": 80}}
    )
    img = _fundus(100, 60, 50, 30, 25)  # short side 60 < 64 after longest-80
    assert val_t(image=img)["image"].shape == (3, 64, 64)


# --------------------------------------------------------------------------- #
# file-backed datasets + build_dataset dispatch
# --------------------------------------------------------------------------- #
def _make_prepared_root(tmp_path, n: int = 40, val_fraction: float = 0.25) -> dict:
    root = tmp_path / "ds"
    names = []
    for i in range(n):
        names.append(_write_image(root, "images", f"img_{i:03d}.png"))
    with open(root / "labels.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "grade"])
        for i, name in enumerate(names):
            writer.writerow([name, i % 5])
    cfg = {
        "data": {
            "dataset": "folder",
            "root": str(root),
            "img_size": 24,
            "clahe_prob": 0.0,
            "resize_longest": 32,
            "val_fraction": val_fraction,
        }
    }
    return cfg


def test_build_dataset_file_splits_and_items(tmp_path) -> None:
    cfg = _make_prepared_root(tmp_path)
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "val")
    assert isinstance(train_ds, FileListDataset)
    assert len(train_ds) > 0 and len(val_ds) > 0
    assert len(train_ds) + len(val_ds) == 40
    train_names = {p.name for p, _ in train_ds.samples}
    val_names = {p.name for p, _ in val_ds.samples}
    assert not train_names & val_names
    img, grade = train_ds[0]
    assert img.shape == (3, 24, 24) and img.dtype == torch.float32
    assert isinstance(grade, int) and 0 <= grade <= 4
    # same cfg -> identical membership (deterministic hash split)
    assert build_dataset(cfg, "val").samples == val_ds.samples


def test_build_dataset_ben_graham_flag_changes_pixels(tmp_path) -> None:
    cfg = _make_prepared_root(tmp_path)
    cfg["data"]["ben_graham"] = True
    ds_plain = build_dataset({**cfg, "data": {**cfg["data"], "ben_graham": False}}, "train")
    ds_bg = build_dataset(cfg, "train")
    x_plain, _ = ds_plain[0]
    x_bg, _ = ds_bg[0]
    assert not torch.equal(x_plain, x_bg)


def test_build_dataset_errors(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_dataset({"data": {"dataset": "nope"}}, "train")
    with pytest.raises(ValueError):
        build_dataset(_synth_cfg(), "invalid")
    cfg = {"data": {"dataset": "aptos", "root": str(tmp_path / "missing"), "img_size": 24}}
    with pytest.raises(FileNotFoundError):
        build_dataset(cfg, "train")
    # labels.csv points to missing files -> all dropped -> FileNotFoundError
    root = tmp_path / "empty"
    root.mkdir()
    (root / "labels.csv").write_text("image,grade\nnope.png,0\n")
    cfg = {"data": {"dataset": "folder", "root": str(root), "img_size": 24}}
    with pytest.raises(FileNotFoundError):
        build_dataset(cfg, "train")


# --------------------------------------------------------------------------- #
# prepare CLI (aptos / eyepacs / folder / multipart-zip)
# --------------------------------------------------------------------------- #
def _run_prepare(tmp_path, dataset: str, build_raw, extra_cfg=None) -> dict:
    root = tmp_path / f"data_{dataset}"
    raw = root / "raw"
    build_raw(raw)
    cfg = {"data": {"dataset": dataset, "root": str(root), "img_size": 224}}
    if extra_cfg:
        cfg["data"].update(extra_cfg)
    cfg_path = tmp_path / f"{dataset}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rc = prepare_main(["--config", str(cfg_path)])
    assert rc == 0
    return cfg


def test_prepare_aptos_layout(tmp_path) -> None:
    def build(raw):
        img_dir = raw / "train_images"
        img_dir.mkdir(parents=True)
        for i in range(6):
            cv2.imwrite(str(img_dir / f"{i:05d}.png"), np.zeros((16, 16, 3), np.uint8))
        (raw / "train.csv").write_text(
            "id_code,diagnosis\n" + "\n".join(f"{i:05d},{i % 5}" for i in range(6))
        )

    cfg = _run_prepare(tmp_path, "aptos", build)
    root = tmp_path / "data_aptos"
    labels = (root / "labels.csv").read_text().strip().splitlines()
    assert labels[0] == "image,grade" and len(labels) == 7
    assert labels[1] == "00000.png,0"
    assert (root / "images" / "00003.png").exists()
    assert (root / "README.md").exists()
    assert cfg["data"]["root"] == str(root)  # config untouched


def test_prepare_eyepacs_layout(tmp_path) -> None:
    def build(raw):
        img_dir = raw / "train"
        img_dir.mkdir(parents=True)
        for i in range(5):
            cv2.imwrite(str(img_dir / f"{i}_left.jpeg"), np.zeros((16, 16, 3), np.uint8))
        (raw / "trainLabels.csv").write_text(
            "image,level\n" + "\n".join(f"{i}_left,{i % 5}" for i in range(5))
        )

    _run_prepare(tmp_path, "eyepacs", build)
    root = tmp_path / "data_eyepacs"
    labels = (root / "labels.csv").read_text().strip().splitlines()
    assert len(labels) == 6
    assert labels[1] == "0_left.jpeg,0"
    assert (root / "images" / "4_left.jpeg").exists()


def test_prepare_folder_grade_tree_and_idempotent(tmp_path) -> None:
    def build(raw):
        for grade in range(5):
            for i in range(3):
                _write_image(raw, str(grade), f"g{grade}_{i}.jpg", size=16)

    _run_prepare(tmp_path, "folder", build)
    root = tmp_path / "data_folder"
    labels = (root / "labels.csv").read_text().strip().splitlines()
    assert len(labels) == 16  # header + 15 images
    counts = {g: 0 for g in range(5)}
    for line in labels[1:]:
        counts[int(line.split(",")[1])] += 1
    assert all(v == 3 for v in counts.values())
    # idempotent re-run: same labels.csv content, exit code 0
    before = (root / "labels.csv").read_text()
    assert prepare_main(["--config", str(tmp_path / "folder.yaml")]) == 0
    assert (root / "labels.csv").read_text() == before


def test_prepare_multipart_zip(tmp_path) -> None:
    def build(raw):
        raw.mkdir(parents=True)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for i in range(4):
                img = np.zeros((16, 16, 3), np.uint8)
                ok, enc = cv2.imencode(".png", img)
                assert ok
                zf.writestr(f"train_images/{i:05d}.png", enc.tobytes())
            zf.writestr(
                "train.csv", "id_code,diagnosis\n" + "\n".join(f"{i:05d},{i % 5}" for i in range(4))
            )
        data = buf.getvalue()
        mid = len(data) // 2
        (raw / "train.zip.001").write_bytes(data[:mid])
        (raw / "train.zip.002").write_bytes(data[mid:])

    _run_prepare(tmp_path, "aptos", build)
    root = tmp_path / "data_aptos"
    labels = (root / "labels.csv").read_text().strip().splitlines()
    assert len(labels) == 5
    assert (root / "images" / "00000.png").exists()


def test_prepare_missing_raw_fails_gracefully(tmp_path) -> None:
    cfg_path = tmp_path / "aptos.yaml"
    cfg_path.write_text(yaml.safe_dump({"data": {"dataset": "aptos", "root": str(tmp_path / "x")}}))
    assert prepare_main(["--config", str(cfg_path)]) == 1
