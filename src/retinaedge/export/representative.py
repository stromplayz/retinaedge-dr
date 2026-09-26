"""Calibration data for full-integer post-training quantization.

Pure numpy/cv2 — deliberately free of TensorFlow imports so the representative
dataset can be built (and unit-tested) on machines without TF. Each yielded
sample is a float32 ``(1, 3, H, W)`` ImageNet-normalized tensor, i.e. exactly
what the exported graph expects at inference time; the TF converter observes
these to compute activation ranges and inserts the uint8 quantize/dequantize
boundaries itself (full-integer graph, uint8 in/out per the export contract).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from retinaedge.export.wrappers import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["RepresentativeDataset", "preprocess_calibration_image"]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess_calibration_image(rgb: np.ndarray, img_size: int) -> np.ndarray:
    """Turn an RGB uint8 image into a normalized float32 ``(1, 3, H, W)`` sample.

    No CLAHE / fundus crop here on purpose: calibration just needs a
    representative spread of pixel statistics, matching the on-device
    preprocessing (resize + ImageNet normalize).
    """
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ((chw[None] - _MEAN) / _STD).astype(np.float32)


class RepresentativeDataset:
    """Iterable of quantization calibration batches.

    Yields ``(float32 (1, 3, img_size, img_size),)`` tuples — the shape
    ``tf.lite.TFLiteConverter.representative_dataset`` expects.

    Args:
        img_size: Square edge length of the model input.
        image_dir: Optional folder of images (``jpg/png/...``). Sorted by filename
            for determinism; cycled if fewer images than ``num_samples``.
        num_samples: Number of calibration batches to yield.
        seed: Seed for the synthetic fallback stream (also the batch order seed).
    """

    def __init__(
        self,
        img_size: int,
        image_dir: str | Path | None = None,
        num_samples: int = 200,
        seed: int = 0,
    ) -> None:
        self.img_size = int(img_size)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}")

    def _image_paths(self) -> list[Path]:
        if self.image_dir is None:
            return []
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"representative-dir is not a directory: {self.image_dir}")
        paths = sorted(
            p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        return paths

    def __iter__(self) -> Iterator[tuple[np.ndarray]]:
        """Yield ``num_samples`` calibration batches (real images if available)."""
        paths = self._image_paths()
        if paths:
            count = 0
            while count < self.num_samples:
                for path in paths:
                    if count >= self.num_samples:
                        break
                    bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
                    if bgr is None:  # unreadable/corrupt file — skip deterministically
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield (preprocess_calibration_image(rgb, self.img_size),)
                    count += 1
            return
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            rgb = rng.integers(0, 256, size=(self.img_size, self.img_size, 3), dtype=np.uint8)
            yield (preprocess_calibration_image(rgb, self.img_size),)
