"""Write deployment metadata (``labels.txt`` + ``model_info.json``) for an exported model.

The Android app loads ``labels.txt`` alongside the TFLite asset and reads
``model_info.json`` for input preprocessing rules, I/O dtypes and the referable
threshold — this CLI is the single source of truth for both.

CLI::

    python -m retinaedge.export.metadata --model artifacts/smoke/model.tflite \
        --out-dir artifacts/smoke --temperature 1.07
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from retinaedge.export.wrappers import GRADE_LABELS, IMAGENET_MEAN, IMAGENET_STD
from retinaedge.utils.logging_utils import get_logger

__all__ = ["probe_onnx", "probe_tflite", "write_metadata", "main"]

logger = get_logger(__name__)

_MODEL_NAME = "retinaedge-dr"


def probe_onnx(model_path: str | Path) -> dict:
    """Inspect an ONNX file: input/output names, shapes, dtypes, opset, size."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is not installed — pip install -e '.[dev]'") from exc

    model = onnx.load(str(model_path))
    graph = model.graph

    def _shape(tensor_type) -> list:
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return dims

    opset = next((o.version for o in model.opset_import if not o.domain), None)
    return {
        "format": "onnx",
        "input": {
            "name": graph.input[0].name,
            "shape": _shape(graph.input[0].type.tensor_type),
            "dtype": onnx.TensorProto.DataType.Name(
                graph.input[0].type.tensor_type.elem_type
            ).lower(),
        },
        "outputs": [
            {
                "name": out.name,
                "shape": _shape(out.type.tensor_type),
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type).lower(),
            }
            for out in graph.output
        ],
        "opset": int(opset) if opset else None,
        "file_size_bytes": Path(model_path).stat().st_size,
    }


def probe_tflite(model_path: str | Path) -> dict:
    """Inspect a TFLite flatbuffer via a (tflite_runtime or TF) interpreter."""
    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=str(model_path))
        except ImportError as exc:
            raise RuntimeError(
                "probing a .tflite model needs tensorflow or tflite_runtime installed — "
                "metadata will be written without I/O details."
            ) from exc

    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    scale, zero_point = inp.get("quantization", (np.array([0.0]), np.array([0])))
    quantized = bool(np.any(np.asarray(scale) > 0))
    return {
        "format": "tflite",
        "input": {
            "name": inp.get("name", ""),
            "shape": [int(d) for d in inp["shape"]],
            "dtype": str(np.dtype(inp["dtype"])),
            "quantization": (
                {
                    "scale": np.asarray(scale).ravel().tolist(),
                    "zero_point": np.asarray(zero_point).ravel().tolist(),
                    "scheme": "full-integer (uint8) PTQ" if quantized else "none",
                }
                if quantized
                else None
            ),
        },
        "outputs": [
            {
                "name": out.get("name", ""),
                "shape": [int(d) for d in out["shape"]],
                "dtype": str(np.dtype(out["dtype"])),
            }
            for out in outputs
        ],
        "file_size_bytes": Path(model_path).stat().st_size,
    }


def write_metadata(
    model_path: str | Path,
    out_dir: str | Path,
    temperature: float | None = None,
    referable_threshold: float = 0.5,
    model_name: str = _MODEL_NAME,
) -> dict:
    """Write ``labels.txt`` and ``model_info.json`` into ``out_dir``.

    Returns:
        The info dict that was serialized to ``model_info.json``.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # labels.txt — order MUST match the grade scheme (0=No DR ... 4=Proliferative DR).
    labels_path = out_dir / "labels.txt"
    labels_path.write_text("\n".join(GRADE_LABELS) + "\n", encoding="utf-8")

    suffix = model_path.suffix.lower()
    try:
        probed = probe_onnx(model_path) if suffix == ".onnx" else probe_tflite(model_path)
    except RuntimeError as exc:
        logger.warning("%s", exc)
        probed = {"format": suffix.lstrip("."), "note": str(exc)}

    info: dict = {
        "model_name": model_name,
        "format": probed.get("format"),
        "file": model_path.name,
        "file_size_bytes": model_path.stat().st_size,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "input": {
            **probed.get("input", {}),
            "layout": "CHW",
            "normalization": {
                "scheme": "ImageNet",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "usage": "x = (rgb_float_0_1 - mean) / std ; quantize to uint8 if the input tensor is uint8",
            },
        },
        "outputs": [
            {
                **out_spec,
                "interpretation": "P(grade=k) for k=0..4; rows sum to 1",
            }
            for out_spec in probed.get("outputs", [])
        ],
        "grades": {
            "num_grades": len(GRADE_LABELS),
            "labels": list(GRADE_LABELS),
            "scheme": "ICDRSS 0-4",
            "referable_definition": "grade >= 2",
        },
        "postprocess": {
            "hard_grade": "argmax(probs)",
            "referable_prob": "probs[2] + probs[3] + probs[4]",
            "referable_threshold": float(referable_threshold),
        },
        "temperature": float(temperature) if temperature is not None else None,
        "opset": probed.get("opset"),
        "quantization": probed.get("input", {}).get("quantization"),
        "research_use_only": True,
        "labels_file": labels_path.name,
    }
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s and %s", labels_path, out_dir / "model_info.json")
    return info


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.export.metadata",
        description="Write labels.txt + model_info.json next to an exported model.",
    )
    parser.add_argument("--model", required=True, help="path to the .onnx or .tflite model")
    parser.add_argument(
        "--out-dir", required=True, help="directory for labels.txt / model_info.json"
    )
    parser.add_argument("--temperature", type=float, default=None, help="calibration temperature")
    parser.add_argument(
        "--referable-threshold", type=float, default=0.5, help="referable DR threshold"
    )
    parser.add_argument("--model-name", default=_MODEL_NAME, help="human-readable model name")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 = success)."""
    args = _parse_args(argv)
    try:
        write_metadata(
            args.model,
            args.out_dir,
            temperature=args.temperature,
            referable_threshold=args.referable_threshold,
            model_name=args.model_name,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
