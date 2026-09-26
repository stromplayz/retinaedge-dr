"""Ben-Graham fundus preprocessing.

Standard preprocessing for colour fundus photographs (Ben-Graham et al., 2016,
"Automated Diabetic Retinopathy Detection ..."): crop to the fundus disc, scale
so the estimated fundus radius equals ``radius`` pixels, then apply CLAHE to
the L-channel of the LAB representation. Pure cv2/numpy — no extra deps.

Contract (docs/INTERFACES.md): operates on uint8 RGB HWC arrays and returns
uint8 RGB HWC arrays. Deterministic.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["preprocess_ben_graham", "estimate_fundus_radius"]

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def estimate_fundus_radius(gray: np.ndarray) -> tuple[int, int, int]:
    """Estimate the fundus disc from a grayscale image.

    Returns:
        ``(cy, cx, radius)`` — disc centre and estimated fundus radius in
        pixels. Falls back to the image centre / half of the short side when no
        meaningful mask is found (e.g. very dark or flat images).
    """
    h, w = gray.shape
    # Vectorised variant of the original row/column-mean thresholding: pixels
    # brighter than 10% of the mean are considered inside the fundus disc.
    threshold = float(gray.mean()) / 10.0
    mask = gray > threshold
    if int(mask.sum()) < 32:  # degenerate image -> no-op geometry
        return h // 2, w // 2, min(h, w) // 2
    ys, xs = np.nonzero(mask)
    # Radius = half of the larger bounding-box extent; centre = bbox centre.
    radius = max((ys.max() - ys.min()) // 2, (xs.max() - xs.min()) // 2, 1)
    cy, cx = (ys.max() + ys.min()) // 2, (xs.max() + xs.min()) // 2
    return int(cy), int(cx), int(radius)


def preprocess_ben_graham(rgb: np.ndarray, radius: int = 300) -> np.ndarray:
    """Crop the fundus, rescale to ``radius`` px and CLAHE the LAB L-channel.

    Args:
        rgb: uint8 RGB array of shape ``(H, W, 3)``.
        radius: Target fundus radius in pixels after scaling.

    Returns:
        uint8 RGB square array around the estimated disc centre, rescaled so
        the fundus radius is ``radius`` pixels.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB (H, W, 3), got shape {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cy, cx, r = estimate_fundus_radius(gray)

    # Largest square crop around the disc centre that fits inside the image.
    half = max(min(cy, h - cy, cx, w - cx), 1)
    crop = rgb[cy - half : cy + half, cx - half : cx + half]

    # Rescale so the estimated fundus radius equals `radius`.
    scale = float(radius) / float(r)
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, (0, 0), fx=scale, fy=scale, interpolation=interp)

    # CLAHE on the LAB L-channel (lightness only — chroma is preserved).
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
