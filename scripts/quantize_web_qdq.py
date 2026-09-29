#!/usr/bin/env python3
"""QDQ-format int8 variants for the WEB playground (onnxruntime-web wasm EP).

Why: the released int8 artifacts use onnxruntime's *dynamic* quantization,
which lowers Conv to ConvInteger — a kernel the WebAssembly execution
provider does not ship (session creation fails with "Could not find an
implementation for ConvInteger(10)"). QDQ (QuantizeLinear/DequantizeLinear)
static quantization keeps the graph inside the op set the wasm EP implements,
so the same int8 size story can be demonstrated in the browser.

Calibration here is synthetic (smoothed random fields, seeded) because the
web build must not depend on the training data; the drift report quantifies
the cost honestly. The Android/native int8 artifacts remain the
dynamic-quantized ones from the release pipeline.

Usage:
    python3 scripts/quantize_web_qdq.py --onnx artifacts/release/dr_s5fusion_soup.onnx \
        --out site/models/dr_s5fusion_soup.qdq.onnx [--img-size 224]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _smooth_field(rng: np.random.Generator, shape: list[int], gaussian_filter) -> np.ndarray:
    """Fundus-ish smooth random field: blurred noise, mild vascular structure."""
    x = rng.standard_normal(shape).astype(np.float32)
    x = gaussian_filter(x, sigma=(0, 0, 6, 6))
    x /= max(1e-6, float(np.abs(x).max()))
    return (x * 1.5).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--n-calib", type=int, default=48)
    ap.add_argument("--n-probe", type=int, default=4)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args(argv)

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError:
        print("ERROR: pip install onnx onnxruntime", file=sys.stderr)
        return 1

    src, dst = Path(args.onnx), Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else args.img_size for d in inp.shape]
    shape[0] = 1

    rng = np.random.default_rng(0)

    try:
        from scipy.ndimage import gaussian_filter

        make_field = lambda s: _smooth_field(rng, s, gaussian_filter)  # noqa: E731
    except ImportError:
        make_field = None

    class Reader(CalibrationDataReader):
        def __init__(self, n: int) -> None:
            self.n, self.i = n, 0

        def get_next(self):
            if self.i >= self.n:
                return None
            self.i += 1
            if make_field is not None:
                x = make_field(shape)
            else:
                x = rng.standard_normal(shape).astype(np.float32)
            return {inp.name: x}

    t0 = time.time()
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=Reader(args.n_calib),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    dt = time.time() - t0

    # ---- drift + latency vs fp32 -------------------------------------------
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s32 = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])

    def run(m, x):
        return m.run(None, {inp.name: x})[0]

    max_dev = mean_dev = 0.0
    t32, t8 = [], []
    for _ in range(args.n_probe):
        x = make_field(shape) if make_field else rng.standard_normal(shape).astype(np.float32)
        a = time.perf_counter()
        p32 = run(s32, x)
        t32.append(time.perf_counter() - a)
        a = time.perf_counter()
        p8 = run(s8, x)
        t8.append(time.perf_counter() - a)
        dev = np.abs(p32 - p8)
        max_dev = max(max_dev, float(dev.max()))
        mean_dev += float(dev.mean()) / args.n_probe

    report = {
        "src": str(src),
        "dst": str(dst),
        "fp32_bytes": src.stat().st_size,
        "qdq_bytes": dst.stat().st_size,
        "compression_ratio": round(src.stat().st_size / max(1, dst.stat().st_size), 3),
        "drift_max_abs": round(max_dev, 6),
        "drift_mean_abs": round(mean_dev, 6),
        "latency_ms_fp32_p50": round(float(np.percentile(t32, 50)) * 1000, 3),
        "latency_ms_qdq_p50": round(float(np.percentile(t8, 50)) * 1000, 3),
        "quantize_seconds": round(dt, 1),
        "calibration": f"synthetic smooth fields (seeded), n={args.n_calib:d}",
    }
    print(json.dumps(report, indent=2))
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
