"""Export a trained DrNet checkpoint to ONNX.

Deployment graph contract (docs/INTERFACES.md — the Android side depends on it):

    input : ``image``  float32 (1, 3, H, W), ImageNet-normalized
    output: ``probs``  float32 (1, 5)  per-grade probabilities (rows sum to 1)

CLI::

    python -m retinaedge.export.export_onnx --config configs/train/smoke.yaml \
        --ckpt artifacts/smoke/best.pt --out artifacts/smoke/model.onnx \
        [--img-size 64] [--opset 17] [--dynamic-batch] [dotted.overrides...]

Behaviour:
    - verifies parity vs PyTorch with onnxruntime (atol 1e-3) when available;
    - prints a file-size + CPU latency summary after writing the model.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from retinaedge.export.defaults import resolve_export_settings
from retinaedge.export.wrappers import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    InferenceWrapper,
    build_inference_model,
)
from retinaedge.utils.config import load_config
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = ["export_to_onnx", "measure_onnx_latency", "verify_onnx_parity", "main"]

logger = get_logger(__name__)


def _import_onnxruntime():
    """Import onnxruntime lazily so the module stays usable without it."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - dev extra normally installed
        raise RuntimeError(
            "onnxruntime is required for ONNX parity/latency checks — pip install -e '.[dev]'"
        ) from exc
    return ort


def export_to_onnx(
    wrapper: InferenceWrapper,
    img_size: int,
    out_path: str | Path,
    opset: int = 17,
    dynamic_batch: bool = False,
) -> Path:
    """Trace ``wrapper`` to ONNX and write it to ``out_path``.

    Uses the legacy TorchScript exporter (dynamo=False) for predictable opset
    coverage on torch 2.x; falls back to the dynamo exporter if the legacy path
    is unavailable. Batch dim is dynamic only when ``dynamic_batch`` is set —
    H/W stay fixed so the TFLite/Android graph keeps a static input shape.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs: dict[str, Any] = {
        "input_names": [ONNX_INPUT_NAME],
        "output_names": [ONNX_OUTPUT_NAME],
        "opset_version": int(opset),
        "dynamo": False,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        kwargs["dynamic_axes"] = {ONNX_INPUT_NAME: {0: "batch"}, ONNX_OUTPUT_NAME: {0: "batch"}}
    try:
        torch.onnx.export(wrapper, (dummy,), str(out_path), **kwargs)
    except (RuntimeError, ImportError) as exc:
        logger.warning("legacy ONNX export failed (%s); retrying with the dynamo exporter", exc)
        kwargs["dynamo"] = True
        kwargs.pop("do_constant_folding", None)
        torch.onnx.export(wrapper, (dummy,), str(out_path), **kwargs)
    logger.info("wrote %s (opset=%d, dynamic_batch=%s)", out_path, opset, dynamic_batch)
    return out_path


def verify_onnx_parity(
    onnx_path: str | Path,
    wrapper: InferenceWrapper,
    img_size: int,
    atol: float = 1e-3,
    n_inputs: int = 4,
    batch: int = 1,
    seed: int = 1234,
) -> float:
    """Check ONNX vs PyTorch outputs on seeded random inputs.

    Returns:
        The worst-case max absolute difference across inputs — the caller
        decides pass/fail against ``atol`` (kept out so tests can introspect).

    Raises:
        AssertionError: on a shape mismatch between the ONNX and torch outputs.
        RuntimeError: if onnxruntime is not installed.
    """
    ort = _import_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(n_inputs):
        x = torch.randn(batch, 3, img_size, img_size, generator=generator)
        with torch.no_grad():
            reference = wrapper(x).numpy().astype(np.float32)
        produced = session.run(None, {ONNX_INPUT_NAME: x.numpy()})[0].astype(np.float32)
        if produced.shape != reference.shape:
            raise AssertionError(
                f"shape mismatch: onnx {produced.shape} vs torch {reference.shape}"
            )
        worst = max(worst, float(np.abs(produced - reference).max()))
    return worst


def measure_onnx_latency(
    onnx_path: str | Path,
    img_size: int,
    runs: int = 30,
    warmup: int = 5,
    batch: int = 1,
) -> dict[str, float]:
    """Measure single-thread-friendly CPU latency of the ONNX graph via ORT.

    Returns:
        ``{"mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "fps", "runs", "warmup"}``.
    """
    ort = _import_onnxruntime()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feed = {ONNX_INPUT_NAME: rng.standard_normal((batch, 3, img_size, img_size)).astype(np.float32)}
    for _ in range(max(0, warmup)):
        session.run(None, feed)
    samples_ms = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        session.run(None, feed)
        samples_ms.append((time.perf_counter() - start) * 1e3)
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": 1e3 / mean if mean > 0 else 0.0,
        "runs": float(runs),
        "warmup": float(warmup),
    }


def measure_torch_latency(
    wrapper: nn.Module,
    img_size: int,
    runs: int = 30,
    warmup: int = 5,
    batch: int = 1,
) -> dict[str, float]:
    """Fallback latency summary measured on the eager PyTorch model (no ORT)."""
    x = torch.randn(batch, 3, img_size, img_size)
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            wrapper(x)
        samples_ms = []
        for _ in range(max(1, runs)):
            start = time.perf_counter()
            wrapper(x)
            samples_ms.append((time.perf_counter() - start) * 1e3)
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": 1e3 / mean if mean > 0 else 0.0,
        "runs": float(runs),
        "warmup": float(warmup),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.export.export_onnx",
        description="Export DrNet -> ONNX with parity verification and a latency summary.",
    )
    parser.add_argument("--config", required=True, help="train YAML config (model + data.img_size)")
    parser.add_argument(
        "--ckpt", default=None, help="trainer checkpoint (best.pt); omit = random init"
    )
    parser.add_argument("--out", required=True, help="output .onnx path")
    parser.add_argument("--img-size", type=int, default=None, help="override input edge H=W")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset (default 17)")
    parser.add_argument(
        "--dynamic-batch", action="store_true", default=None, help="make the batch dim dynamic"
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        default=None,
        help="skip ORT parity check",
    )
    parser.add_argument("--atol", type=float, default=None, help="parity tolerance (default 1e-3)")
    parser.add_argument("--runs", type=int, default=None, help="latency timing runs (default 30)")
    parser.add_argument("--warmup", type=int, default=None, help="latency warmup runs (default 5)")
    parser.add_argument(
        "--export-config", default=None, help="export profile YAML (default profile)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="dotted train-config overrides, e.g. data.img_size=96 model.backbone=... ",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    args = _parse_args(argv)
    cfg = load_config(args.config, tuple(args.overrides))
    settings = resolve_export_settings(
        cfg,
        args.export_config,
        {
            "img_size": args.img_size,
            "opset": args.opset,
            "dynamic_batch": args.dynamic_batch,
            "verify": args.verify,
            "verify_atol": args.atol,
            "latency_runs": args.runs,
            "warmup": args.warmup,
        },
    )
    seed_everything(int(cfg.get("train", {}).get("seed", 0)))

    img_size = int(settings["img_size"] or cfg.get("data", {}).get("img_size", 224))
    if args.ckpt is None:
        logger.warning(
            "no --ckpt given: exporting RANDOMLY INITIALISED weights (graph-only export)"
        )
    wrapper = build_inference_model(cfg, args.ckpt)

    onnx_path = export_to_onnx(
        wrapper,
        img_size,
        args.out,
        opset=int(settings["opset"]),
        dynamic_batch=bool(settings["dynamic_batch"]),
    )

    size_mb = onnx_path.stat().st_size / 1e6
    parity_diff: float | None = None
    latency: dict[str, float]
    try:
        if settings["verify"]:
            parity_diff = verify_onnx_parity(
                onnx_path, wrapper, img_size, atol=float(settings["verify_atol"])
            )
    except RuntimeError as exc:
        logger.warning("%s — falling back to eager-torch latency only", exc)
        parity_diff = None
    if parity_diff is not None and parity_diff > float(settings["verify_atol"]):
        logger.error("ONNX parity check FAILED (max|diff|=%.3e) — model NOT trusted", parity_diff)
        return 1
    try:
        latency = measure_onnx_latency(
            onnx_path, img_size, runs=int(settings["latency_runs"]), warmup=int(settings["warmup"])
        )
        backend = "onnxruntime-cpu"
    except RuntimeError:
        latency = measure_torch_latency(
            wrapper, img_size, runs=int(settings["latency_runs"]), warmup=int(settings["warmup"])
        )
        backend = "torch-eager-cpu"

    print("\n=== ONNX export summary ===")
    print(f"model        : {onnx_path}")
    print(f"size         : {size_mb:.2f} MB")
    print(f"input        : float32 (1, 3, {img_size}, {img_size}) ImageNet-normalized")
    print("output       : float32 (1, 5) grade probs")
    print(f"opset        : {settings['opset']}  dynamic_batch={settings['dynamic_batch']}")
    parity_str = f"PASS (max|diff|={parity_diff:.2e})" if parity_diff is not None else "SKIPPED"
    print(f"parity       : {parity_str}")
    print(
        f"latency[{backend}] : mean={latency['mean_ms']:.2f} ms  p50={latency['p50_ms']:.2f} ms  "
        f"p95={latency['p95_ms']:.2f} ms  ({latency['fps']:.1f} fps, {int(latency['runs'])} runs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
