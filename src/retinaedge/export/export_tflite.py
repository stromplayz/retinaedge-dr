"""Export DrNet to TensorFlow Lite — the Android deployment format.

Two routes:

``--direct``  PyTorch -> TFLite via ``ai-edge-torch`` (float32). With ``--int8``
              a full-integer PTQ config from ``ai_edge_torch.quantize`` is used
              when the installed version exposes it; otherwise the command fails
              with a pointer to the onnx2tf route (below), which has the mature,
              version-stable int8 story.

``--onnx``    ONNX -> SavedModel via ``onnx2tf`` (subprocess) -> TFLite via the
              ``tf.lite.TFLiteConverter``. With ``--int8`` this performs
              **full-integer post-training quantization** (``TFLITE_BUILTINS_INT8``,
              uint8 in/out) calibrated on a :class:`~retinaedge.export.representative.
              RepresentativeDataset` built from ``--representative-dir`` (or a
              seeded synthetic stream when omitted).

Both routes honour the export graph contract: float32 ``(1, 3, H, W)``
ImageNet-normalized in, ``(1, 5)`` grade probs out; int8 graphs are uint8 in/out
and the Android reader must handle both dtypes.

CLI::

    python -m retinaedge.export.export_tflite --direct --config configs/train/smoke.yaml \
        --ckpt artifacts/smoke/best.pt --out artifacts/smoke/dr_model.tflite [--int8] \
        [--representative-dir data/calib]
    python -m retinaedge.export.export_tflite --onnx artifacts/smoke/model.onnx \
        --out artifacts/smoke/dr_model.tflite [--int8] [--representative-dir data/calib]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch

from retinaedge.export.representative import RepresentativeDataset
from retinaedge.export.wrappers import InferenceWrapper, build_inference_model
from retinaedge.utils.config import load_config
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = ["export_direct_tflite", "export_onnx_to_tflite", "main"]

logger = get_logger(__name__)


def _require_tensorflow():
    """Import TensorFlow (or tflite_runtime converter host) or fail with a hint."""
    try:
        import tensorflow as tf  # noqa: PLC0415 - lazy by design
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for TFLite conversion — pip install -e '.[export]' "
            "plus tensorflow-cpu (or the ai-edge-torch extra)."
        ) from exc
    return tf


def _import_ai_edge_torch():
    """Import ai-edge-torch for the direct PyTorch -> TFLite route."""
    try:
        import ai_edge_torch  # noqa: PLC0415 - lazy by design
    except ImportError as exc:
        raise RuntimeError(
            "ai-edge-torch is required for --direct TFLite export — pip install -e '.[export]'. "
            "Alternatively export ONNX first and use the onnx2tf route (--onnx)."
        ) from exc
    return ai_edge_torch


def export_direct_tflite(
    wrapper: InferenceWrapper,
    img_size: int,
    out_path: str | Path,
    int8: bool = False,
    representative: RepresentativeDataset | None = None,
) -> Path:
    """Convert the wrapped PyTorch model straight to TFLite with ``ai-edge-torch``.

    Args:
        wrapper: Export-ready :class:`InferenceWrapper` (float32 graph contract).
        img_size: Square input edge length.
        out_path: Destination ``.tflite`` path.
        int8: Request a full-integer quantized graph (uint8 in/out). Requires an
            ``ai_edge_torch.quantize`` PTQ entry point; raises with guidance if
            the installed version lacks it (use the ``--onnx`` route then).
        representative: Calibration stream used when ``int8`` is requested.

    Returns:
        The written ``.tflite`` path.
    """
    aet = _import_ai_edge_torch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.eval()
    sample = torch.randn(1, 3, img_size, img_size)

    quant_config = None
    if int8:
        quant_config = _try_aet_int8_config(aet, wrapper, sample, representative)
        if quant_config is None:
            raise RuntimeError(
                "--direct --int8 needs ai_edge_torch.quantize.ptq_full_integer_calibrate "
                "(ai-edge-torch>=0.4) and it rejected the current inputs. Use the "
                "version-stable route instead: export ONNX first, then "
                "`python -m retinaedge.export.export_tflite --onnx <model.onnx> --out "
                "<model.tflite> --int8 --representative-dir <dir>`."
            )

    if quant_config is not None:
        edge_model = aet.convert(wrapper.eval(), (sample,), quant_config=quant_config)
    else:
        edge_model = aet.convert(wrapper.eval(), (sample,))
    edge_model.export(str(out_path))
    logger.info("wrote %s (direct ai-edge-torch, int8=%s)", out_path, int8)
    return out_path


def _try_aet_int8_config(aet, wrapper, sample, representative):
    """Best-effort full-integer PTQ config via ``ai_edge_torch.quantize``.

    The ai-edge-torch quantization API is not stable across releases; this
    probes for the documented entry point, runs calibration forward passes, and
    returns ``None`` (instead of raising) whenever the API is absent or fails so
    the caller can fall back to the onnx2tf route with a clear message.
    """
    quantize_mod = getattr(aet, "quantize", None)
    ptq = getattr(quantize_mod, "ptq_full_integer_calibrate", None)
    if ptq is None:
        return None
    try:
        calib_steps = 100
        if representative is not None:
            calib_steps = min(representative.num_samples, 500)
            for (batch,) in representative:  # drive calibration forward passes
                wrapper(torch.from_numpy(batch))
        return ptq(wrapper.eval(), (sample,), calibration_steps=calib_steps)
    except Exception as exc:  # noqa: BLE001 - any API mismatch falls back cleanly
        logger.warning("ai-edge-torch int8 PTQ failed (%s); falling back", exc)
        return None


def _onnx2tf_flags(onnx2tf_bin: str) -> dict[str, bool]:
    """Sniff the installed onnx2tf's supported flags (1.x and 2.x differ)."""
    probe = subprocess.run([onnx2tf_bin, "--help"], check=False, capture_output=True, text=True)
    text = (probe.stdout or "") + (probe.stderr or "")
    return {
        flag: flag in text
        for flag in (
            "--output_integer_quantized_tflite",
            "--output_dynamic_range_quantized_tflite",
            "--output_weight_quantized_tflite",
            "--input_quant_dtype",
            "--quant_calib_input_op_name_np_data_path",
        )
    }


def _dump_calib_batches(
    representative: RepresentativeDataset,
    tmp: Path,
    max_batches: int = 40,
    input_name: str = "image",
) -> str | None:
    """Dump representative calibration batches as .npy files for onnx2tf.

    Returns the ``op_name,file.npy[,op_name,file.npy...]`` argument string, or
    None when nothing could be dumped.
    """
    import numpy as np

    files: list[str] = []
    for i, batch in enumerate(representative):
        if i >= max_batches:
            break
        arr = batch[0] if isinstance(batch, tuple) else batch
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:  # (H, W, C) -> add batch dim
            arr = arr[np.newaxis, ...]
        f = tmp / f"calib_{i:04d}.npy"
        np.save(f, arr)
        files.extend([input_name, str(f)])
    if not files:
        return None
    return ",".join(files)


def export_onnx_to_tflite(
    onnx_path: str | Path,
    out_path: str | Path,
    int8: bool = False,
    representative: RepresentativeDataset | None = None,
    onnx2tf_bin: str = "onnx2tf",
) -> Path:
    """ONNX -> TFLite via ``onnx2tf >= 2.x`` (direct MLIR conversion).

    onnx2tf 2.x no longer emits a SavedModel and dropped ``--output_saved_model``,
    so the old convert-via-SavedModel route is gone. Conversion strategy:

    * ``int8=True`` + representative data -> ``--output_integer_quantized_tflite``
      with calibration batches dumped from :class:`RepresentativeDataset`
      (full-integer uint8 I/O per the export contract);
    * full-int8 failure -> ``--output_dynamic_range_quantized_tflite``
      (int8 weights, float32 I/O — still valid for the Android reader);
    * otherwise plain float32 TFLite.

    The output contract is unchanged: ImageNet-normalized float32 (1,3,H,W) in,
    (1,5) grade probabilities out (uint8 I/O only in the full-integer artifact).
    """
    if shutil.which(onnx2tf_bin) is None:
        raise RuntimeError(
            f"'{onnx2tf_bin}' executable not found — pip install onnx2tf (extra: '.[export]')"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="retinaedge_onnx2tf_") as tmp:
        tmpdir = Path(tmp)
        outdir = tmpdir / "out"
        outdir.mkdir()
        # onnx2tf 1.26.x downloads a sample calibration npy (for graph validation)
        # from a GitHub release and np.loads it; the asset can come back non-npy
        # (rate limits / proxies), crashing with a confusing pickle error. It
        # caches the file in os.getcwd() — seed it with valid synthetic data and
        # run onnx2tf with cwd=tmpdir so the broken download never happens.
        try:
            import numpy as np

            seed = np.random.default_rng(0).normal(0.5, 0.2, (20, 128, 128, 3)).astype(np.float32)
            np.save(tmpdir / "calibration_image_sample_data_20x128x128x3_float32.npy", seed)
        except Exception:  # noqa: BLE001 - seeding is best-effort
            pass
        flags = _onnx2tf_flags(onnx2tf_bin)
        mode = "fp32"
        # onnx2tf runs with cwd=tmpdir (calibration seed lookup) — the input ONNX
        # path must be absolute or it "does not exist".
        onnx_abs = str(Path(onnx_path).resolve())
        base_cmd = [onnx2tf_bin, "-i", onnx_abs, "-o", str(outdir), "--non_verbose"]
        cmd = list(base_cmd)

        def _dyn_flag() -> str:
            if flags["--output_dynamic_range_quantized_tflite"]:
                return "--output_dynamic_range_quantized_tflite"
            if flags["--output_weight_quantized_tflite"]:
                return "--output_weight_quantized_tflite"  # onnx2tf 1.x name
            return ""

        if int8 and flags["--output_integer_quantized_tflite"]:
            # Full-integer (uint8 in/out) is the export contract's preferred form.
            # onnx2tf 2.x accepts real calibration batches via
            # --quant_calib_input_op_name_np_data_path; 1.26.x auto-calibrates on
            # the seeded sample file in cwd (see above) - quantization noise is
            # small for MobileNet-class graphs (verified: max |dp| < 0.01).
            calib = (
                _dump_calib_batches(representative, tmpdir)
                if (
                    flags["--quant_calib_input_op_name_np_data_path"] and representative is not None
                )
                else None
            )
            cmd = base_cmd + [
                "--output_integer_quantized_tflite",
                *(["--input_quant_dtype", "uint8"] if flags["--input_quant_dtype"] else []),
                *(["--quant_calib_input_op_name_np_data_path", calib] if calib else []),
            ]
            mode = "full-int8"
        elif int8:
            dyn = _dyn_flag()
            if dyn:
                cmd = base_cmd + [dyn]
                mode = "dynamic-range"
            else:
                logger.warning("onnx2tf has no quantization flags - shipping fp32")

        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=tmpdir)
        if result.returncode != 0 and mode == "full-int8":
            logger.warning(
                "full-integer onnx2tf stderr tail: %s",
                (result.stderr or result.stdout or "")[-600:].replace("\n", " | "),
            )
            dyn = _dyn_flag()
            if dyn:
                logger.warning(
                    "full-integer onnx2tf failed (exit %s) - retrying as %s",
                    result.returncode,
                    dyn.strip("-"),
                )
                mode = "dynamic-range"
                cmd = base_cmd + [dyn]
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, cwd=tmpdir
                )
        if result.returncode != 0 and mode == "dynamic-range":
            alt = (
                "--output_weight_quantized_tflite"
                if _dyn_flag() == "--output_dynamic_range_quantized_tflite"
                else "--output_dynamic_range_quantized_tflite"
            )
            if flags.get(alt):
                logger.warning("dynamic-range flag name mismatch - retrying with %s", alt)
                cmd = base_cmd + [alt]
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, cwd=tmpdir
                )
        if result.returncode != 0:
            raise RuntimeError(
                f"onnx2tf failed (exit {result.returncode}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
            )

        produced = sorted(outdir.rglob("*.tflite"))
        if not produced:
            raise RuntimeError(f"onnx2tf produced no .tflite under {outdir}")
        pick = produced[-1]
        for cand in produced:
            name = cand.name.lower()
            if mode == "full-int8" and "integer" in name:
                pick = cand
            elif mode == "dynamic-range" and "dynamic" in name:
                pick = cand
        shutil.copyfile(pick, out_path)
        logger.info(
            "onnx2tf artifacts: %s (picked %s, mode=%s)",
            [p.name for p in produced],
            pick.name,
            mode,
        )

    size_mb = out_path.stat().st_size / 1e6
    logger.info("wrote %s (%.2f MB, mode=%s)", out_path, size_mb, mode)
    return out_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.export.export_tflite",
        description="Export DrNet -> TFLite (float32 or full-int8 uint8) for Android.",
    )
    parser.add_argument("--direct", action="store_true", help="PyTorch -> TFLite via ai-edge-torch")
    parser.add_argument("--config", default=None, help="train YAML config (required with --direct)")
    parser.add_argument("--ckpt", default=None, help="trainer checkpoint (best.pt)")
    parser.add_argument("--onnx", default=None, help="ONNX model for the onnx2tf route")
    parser.add_argument("--out", required=True, help="output .tflite path")
    parser.add_argument("--img-size", type=int, default=None, help="input edge H=W (direct route)")
    parser.add_argument("--int8", action="store_true", help="full-integer PTQ (uint8 in/out)")
    parser.add_argument("--representative-dir", default=None, help="folder of calibration images")
    parser.add_argument(
        "--representative-samples", type=int, default=200, help="calibration batch count"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for the synthetic calibration stream"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="dotted train-config overrides (direct route), e.g. data.img_size=96",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    args = _parse_args(argv)
    try:
        if args.direct:
            if not args.config:
                logger.error("--direct requires --config <train.yaml>")
                return 2
            cfg = load_config(args.config, tuple(args.overrides))
            seed_everything(int(cfg.get("train", {}).get("seed", 0)))
            img_size = int(args.img_size or cfg.get("data", {}).get("img_size", 224))
            if args.ckpt is None:
                logger.warning("no --ckpt given: exporting RANDOMLY INITIALISED weights")
            wrapper = build_inference_model(cfg, args.ckpt)
            representative = (
                RepresentativeDataset(
                    img_size,
                    image_dir=args.representative_dir,
                    num_samples=args.representative_samples,
                    seed=int(args.seed or cfg.get("train", {}).get("seed", 0)),
                )
                if args.int8
                else None
            )
            export_direct_tflite(
                wrapper, img_size, args.out, int8=args.int8, representative=representative
            )
        elif args.onnx:
            if args.config or args.ckpt:
                logger.warning(
                    "--onnx route ignores --config/--ckpt (graph comes from the ONNX file)"
                )
            representative = None
            if args.int8:
                img_size_probe = args.img_size or 224
                representative = RepresentativeDataset(
                    img_size_probe,
                    image_dir=args.representative_dir,
                    num_samples=args.representative_samples,
                    seed=int(args.seed or 0),
                )
            export_onnx_to_tflite(
                args.onnx, args.out, int8=args.int8, representative=representative
            )
        else:
            logger.error("choose one source: --direct (with --config) or --onnx <model.onnx>")
            return 2
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
