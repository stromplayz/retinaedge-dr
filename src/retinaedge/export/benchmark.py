"""Benchmark an exported ONNX or TFLite model on CPU.

CLI::

    python -m retinaedge.export.benchmark --model artifacts/smoke/model.onnx \
        --backend onnxruntime --img-size 64 --runs 50
    python -m retinaedge.export.benchmark --model artifacts/smoke/dr_model.tflite \
        --backend tflite --img-size 224 --runs 50

``--backend auto`` (default) picks the backend from the file extension. Timing
covers one full forward pass (ORT ``run`` / TFLite ``invoke``) per sample and
reports mean / p50 / p95 / min / max latency plus throughput and model size.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from retinaedge.utils.logging_utils import get_logger

__all__ = ["benchmark_model", "make_onnxrunner", "make_tflite_runner", "main"]

logger = get_logger(__name__)

_WARMUP_DEFAULT = 10


def make_onnxrunner(model_path: str | Path, img_size: int = 224) -> tuple[dict, Callable[[], None]]:
    """Build an onnxruntime runner. Returns ``(info, run_fn)``; ``run_fn`` is timed.

    ``img_size`` fills in dynamic H/W dims of fully-dynamic graphs; static shapes win.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime not installed — pip install -e '.[dev]'") from exc
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    info = {
        "backend": "onnxruntime",
        "input_name": inp.name,
        "input_shape": list(inp.shape),
        "input_dtype": inp.type,
    }
    if "float" not in str(inp.type):
        raise RuntimeError(f"only float32 ONNX inputs are supported, got {inp.type}")

    def run_fn() -> None:
        session.run(None, {inp.name: _input_buffer})

    _input_buffer = _random_input(list(inp.shape), np.float32, img_size=img_size)
    return info, run_fn


def make_tflite_runner(
    model_path: str | Path, img_size: int = 224
) -> tuple[dict, Callable[[], None]]:
    """Build a TFLite interpreter runner (TF or tflite_runtime). Returns ``(info, run_fn)``."""
    interpreter = _tflite_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    info = {
        "backend": "tflite",
        "input_name": inp.get("name", ""),
        "input_shape": list(inp["shape"]),
        "input_dtype": str(np.dtype(inp["dtype"])),
        "quantization": {
            "scale": np.asarray(inp.get("quantization", (0.0, 0.0))[0]).ravel().tolist(),
            "zero_point": np.asarray(inp.get("quantization", (0.0, 0.0))[1]).ravel().tolist(),
        },
    }
    buffer = _random_input(
        list(inp["shape"]), inp["dtype"], quantization=info["quantization"], img_size=img_size
    )

    def run_fn() -> None:
        interpreter.set_tensor(inp["index"], buffer)
        interpreter.invoke()

    return info, run_fn


def _tflite_interpreter(model_path: str | Path):
    """Return a TFLite Interpreter from tensorflow or tflite_runtime."""
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=str(model_path))
    except ImportError as exc:
        raise RuntimeError(
            "neither tensorflow nor tflite_runtime is installed — pip install tensorflow-cpu"
        ) from exc


def _random_input(
    shape: list,
    dtype,
    quantization: dict | None = None,
    seed: int = 0,
    img_size: int = 224,
) -> np.ndarray:
    """Deterministic random input in the model's expected dtype/layout.

    Float inputs are ImageNet-normalized-like samples (mean 0, unit-ish std).
    uint8 inputs are produced by requantizing a float sample with the tensor's
    scale/zero-point so calibration ranges are exercised realistically. Unknown
    dims resolve to: batch=1, H/W=``img_size``, anything else=1.
    """
    resolved = []
    for axis, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif axis == 2:  # H of CHW
            resolved.append(img_size)
        else:
            resolved.append(1)
    rng = np.random.default_rng(seed)
    if np.dtype(dtype) == np.uint8:
        scale = (
            float(quantization["scale"][0])
            if quantization and quantization["scale"]
            else 1.0 / 255.0
        )
        zero = (
            int(quantization["zero_point"][0])
            if quantization and quantization["zero_point"]
            else 128
        )
        floats = rng.standard_normal(resolved).astype(np.float32)
        return np.clip(np.round(floats / scale + zero), 0, 255).astype(np.uint8)
    return rng.standard_normal(resolved).astype(np.float32)


def benchmark_model(
    model_path: str | Path,
    backend: str = "auto",
    runs: int = 50,
    warmup: int = _WARMUP_DEFAULT,
    img_size: int = 224,
) -> dict:
    """Benchmark a model file; returns a stats dict (also used for ``--json``)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if backend == "auto":
        backend = {".onnx": "onnxruntime", ".tflite": "tflite"}.get(model_path.suffix.lower(), "")
        if not backend:
            raise ValueError(f"cannot infer backend from {model_path.suffix!r}; pass --backend")
    if backend == "onnxruntime":
        info, run_fn = make_onnxrunner(model_path, img_size=img_size)
    elif backend == "tflite":
        info, run_fn = make_tflite_runner(model_path, img_size=img_size)
    else:
        raise ValueError(f"unknown backend {backend!r} (expected onnxruntime|tflite|auto)")

    for _ in range(max(0, warmup)):
        run_fn()
    samples_ms = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        run_fn()
        samples_ms.append((time.perf_counter() - start) * 1e3)

    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(arr.mean())
    stats = {
        **info,
        "model": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "runs": int(runs),
        "warmup": int(warmup),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
        "fps": 1e3 / mean if mean > 0 else 0.0,
    }
    return stats


def _print_report(stats: dict) -> None:
    print("\n=== Benchmark report ===")
    print(f"model        : {stats['model']}  ({stats['size_bytes'] / 1e6:.2f} MB)")
    print(f"backend      : {stats['backend']} (CPU)")
    print(
        f"input        : {stats['input_dtype']} {stats['input_shape']}  name={stats['input_name']!r}"
    )
    if stats.get("quantization") and any(stats["quantization"]["scale"]):
        print(
            f"quantization : scale={stats['quantization']['scale']} zp={stats['quantization']['zero_point']}"
        )
    print(f"runs         : {stats['runs']} (warmup {stats['warmup']})")
    print(
        f"latency      : mean={stats['mean_ms']:.2f} ms  p50={stats['p50_ms']:.2f} ms  "
        f"p95={stats['p95_ms']:.2f} ms  min={stats['min_ms']:.2f} ms  max={stats['max_ms']:.2f} ms"
    )
    print(f"throughput   : {stats['fps']:.1f} fps")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.export.benchmark",
        description="Benchmark an exported ONNX / TFLite model (CPU latency + throughput).",
    )
    parser.add_argument("--model", required=True, help="path to .onnx or .tflite")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "onnxruntime", "tflite"],
        help="execution backend (default: infer from extension)",
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="input edge size (info/shaping only)"
    )
    parser.add_argument("--runs", type=int, default=50, help="timed runs")
    parser.add_argument("--warmup", type=int, default=_WARMUP_DEFAULT, help="warmup runs")
    parser.add_argument("--json", default=None, help="optional path to dump the stats as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    args = _parse_args(argv)
    try:
        stats = benchmark_model(
            args.model,
            backend=args.backend,
            runs=args.runs,
            warmup=args.warmup,
            img_size=args.img_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    _print_report(stats)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"stats written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
