"""RetinaEdge-DR — Gradio demo.

Upload a fundus image and get the 5-grade ICDRSS probability distribution, the expected grade and
the referable-DR risk (``P(grade >= 2)``) against a movable threshold.

Weight resolution order (first hit wins):
    1. ``--onnx <model.onnx>``           -> onnxruntime
    2. ``--ckpt <best.pt>``              -> PyTorch
    3. ``artifacts/smoke/best.pt``       -> PyTorch
    4. ``artifacts/smoke/model.onnx``    -> onnxruntime
    5. untrained fallback weights        -> clearly flagged in the UI

Run: ``python demo/gradio_app.py`` (or ``make demo``). Extras: ``pip install -e ".[demo]"``.

⚠️ Research/educational demo — NOT a medical device. See docs/MODEL_CARD.md.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "gradio is required for the demo. Install it with: pip install -e '.[demo]'"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    # Allows running `python demo/gradio_app.py` from a repo checkout without editable install.
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from retinaedge.models.ordinal_ops import ordinal_probs  # noqa: E402

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CKPT = REPO_ROOT / "artifacts/smoke/best.pt"
DEFAULT_ONNX = REPO_ROOT / "artifacts/smoke/model.onnx"
DEFAULT_THRESHOLD = 0.5

DISCLAIMER = (
    "### ⚠️ Research demo — not a medical device\n"
    "RetinaEdge-DR is an open-source research prototype. It has **no regulatory clearance** and "
    "must **not** be used for diagnosis, triage or any clinical decision. Outputs may be wrong, "
    "uncalibrated or biased. See `docs/MODEL_CARD.md`."
)

ABOUT = (
    "**Contract** (docs/INTERFACES.md v1.0): input `float32 (1,3,H,W)` ImageNet-normalised → "
    "output `float32 (1,5)` grade probs. Referable DR = grade ≥ 2. Grade order: "
    "`No DR, Mild, Moderate, Severe, Proliferative DR`."
)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce any gradio image array (H,W), (H,W,3) or (H,W,4) into uint8 RGB (H,W,3)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    """Square-resize RGB uint8 to (size, size) with anti-aliasing, cv2 if available."""
    import cv2

    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _maybe_ben_graham(rgb: np.ndarray) -> np.ndarray:
    """Apply the repo Ben-Graham preprocess when the data module is available."""
    try:
        from retinaedge.data.ben_graham import preprocess_ben_graham
    except ImportError:
        return rgb  # module not implemented/installed yet -> model was trained robust anyway
    return preprocess_ben_graham(rgb)


def preprocess(rgb: np.ndarray, img_size: int, ben_graham: bool) -> np.ndarray:
    """RGB uint8 HWC -> float32 CHW (1,3,H,W)-ready, ImageNet-normalised (contract input)."""
    x = _to_rgb(rgb)
    if ben_graham:
        x = _maybe_ben_graham(x)
    x = _resize_rgb(x, img_size).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class TorchPredictor:
    """PyTorch checkpoint predictor (DrNet-style: (B,4) ordinal + (B,1) refer logits)."""

    def __init__(self, model: Any, img_size: int, temperature: float, source: str) -> None:
        self.model, self.img_size = model, img_size
        self.temperature, self.source = max(float(temperature), 1e-3), source

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(torch.from_numpy(chw[None]).float())
        logits = out["ordinal_logits"] if isinstance(out, dict) else out
        return ordinal_probs(logits / self.temperature)[0].numpy().astype(np.float32)


class OnnxPredictor:
    """onnxruntime predictor for contract-conformant exports (input CHW -> (1,5) probs)."""

    def __init__(self, session: Any, img_size: int, source: str) -> None:
        self.session, self.img_size, self.source = session, img_size, source

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        name = self.session.get_inputs()[0].name
        out = self.session.run(None, {name: chw[None].astype(np.float32)})
        probs = np.asarray(out[0], dtype=np.float32)
        probs = probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-12)
        return probs[0]


def _backbone_dim(backbone: nn.Module, img_size: int) -> int:
    """True pooled-feature dim (timm's ``num_features`` lies for some families, e.g. MBV3)."""
    try:
        with torch.no_grad():
            out = backbone(torch.zeros(1, 3, img_size, img_size))
        return int(out.shape[-1])
    except Exception:
        return int(getattr(backbone, "num_features", 0))


def _build_torch_from_ckpt(path: Path, img_size_override: int | None) -> TorchPredictor:
    """Load a training checkpoint payload {'state_dict','cfg','temperature',...} into DrNet."""
    import timm

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    model_cfg = (cfg.get("model") or {}) if isinstance(cfg, dict) else {}
    backbone = model_cfg.get("backbone", "mobilenetv3_small_100")
    img_size = img_size_override or int((cfg.get("data") or {}).get("img_size", 224))
    temperature = float(payload.get("temperature", 1.0) or 1.0)

    backbone_m = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
    dim = _backbone_dim(backbone_m, img_size)
    model = nn.Sequential(  # demo-side stand-in for DrNet; same head semantics
        backbone_m,
        nn.Dropout(float(model_cfg.get("dropout", 0.0))),
        _DualHead(dim),
    )
    state = payload.get("state_dict", {}) if isinstance(payload, dict) else {}
    own = model.state_dict()
    matched = {
        k: v
        for k, v in state.items()
        if k in own and own[k].shape == v.shape  # tolerate head-name differences across builds
    }
    missing = len(own) - len(matched)
    if matched:
        model.load_state_dict({**own, **matched}, strict=False)
    print(f"[demo] {path.name}: matched {len(matched)}/{len(own)} tensors; missing={missing}")
    model.eval()
    return TorchPredictor(model, img_size, temperature, f"PyTorch ckpt: {path.name}")


class _DualHead(nn.Module):
    """Ordinal (K-1=4) + referable (1) heads — contract-shaped forward output."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ordinal_head = nn.Linear(dim, 4)
        self.refer_head = nn.Linear(dim, 1)

    def forward(self, feats: Any) -> dict[str, Any]:
        return {"ordinal_logits": self.ordinal_head(feats), "refer_logits": self.refer_head(feats)}


def _build_untrained(img_size: int | None) -> TorchPredictor:
    """Fallback: random-init DrNet-shaped model so the demo always runs (flagged in the UI)."""
    import timm

    from retinaedge.utils.seed import seed_everything

    seed_everything(0)
    backbone = timm.create_model(
        "mobilenetv3_small_100", pretrained=False, num_classes=0, global_pool="avg"
    )
    model = nn.Sequential(backbone, _DualHead(_backbone_dim(backbone, img_size or 224)))
    model.eval()
    return TorchPredictor(model, img_size or 224, 1.0, "⚠️ UNTRAINED fallback weights")


def build_predictor(ckpt: Path | None, onnx: Path | None, img_size_override: int | None) -> Any:
    """Resolve the best available predictor per the documented priority order."""
    candidates: list[Path] = []
    if onnx:
        candidates.append(onnx)
    if ckpt:
        candidates.append(ckpt)
    candidates += [DEFAULT_CKPT, DEFAULT_ONNX]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".onnx":
            import onnxruntime as ort

            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            size = img_size_override or sess.get_inputs()[0].shape[-1]
            print(f"[demo] loaded ONNX {path}")
            return OnnxPredictor(
                sess, int(size) if isinstance(size, int) else 224, f"ONNX: {path.name}"
            )
        if path.suffix == ".pt":
            print(f"[demo] loaded checkpoint {path}")
            return _build_torch_from_ckpt(path, img_size_override)
    print("[demo] no checkpoint/ONNX found — falling back to UNTRAINED weights")
    return _build_untrained(img_size_override)


def make_predict_fn(predictor: Any, default_ben_graham: bool) -> Any:
    """Bind the resolved predictor into a gradio fn (img, threshold, ben_graham) -> outputs."""

    def predict(img: np.ndarray, threshold: float, ben_graham: bool) -> tuple:
        if img is None:
            return {}, "Upload a fundus image to see a prediction."
        chw = preprocess(img, predictor.img_size, ben_graham or default_ben_graham)
        probs = predictor(chw)
        grade = int(np.argmax(probs))
        expected = float((probs * np.arange(5)).sum())
        refer = float(probs[2:].sum())
        flags = " 🚩 REFERABLE" if refer >= threshold else " below threshold"
        label = {GRADE_NAMES[i]: float(probs[i]) for i in range(5)}
        summary = (
            f"**Hard grade:** {grade} — {GRADE_NAMES[grade]}  \n"
            f"**Expected grade (soft):** {expected:.2f}  \n"
            f"**Referable P(grade ≥ 2):** {refer:.1%} vs threshold {threshold:.0%} → {flags}  \n"
            f"**Weights:** {predictor.source} · **input:** {predictor.img_size}×"
            f"{predictor.img_size}·Ben-Graham={bool(ben_graham or default_ben_graham)}  \n\n"
            f"{ABOUT}"
        )
        return label, summary

    return predict


def build_ui(predictor: Any, default_ben_graham: bool = False) -> gr.Blocks:
    """Assemble the Gradio Blocks UI."""
    with gr.Blocks(title="RetinaEdge-DR demo") as demo:
        gr.Markdown("# RetinaEdge-DR — Diabetic Retinopathy grading demo")
        gr.Markdown(ABOUT)
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="numpy", label="Fundus image")
                ben_g = gr.Checkbox(
                    value=default_ben_graham,
                    label="Ben-Graham preprocess (crop+scale+CLAHE)",
                )
                thr = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_THRESHOLD,
                    step=0.01,
                    label="Referable threshold on P(grade ≥ 2)",
                )
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label_out = gr.Label(num_top_classes=5, label="Grade probabilities (ICDRSS 0–4)")
                text_out = gr.Markdown(label="Report")
        btn.click(
            make_predict_fn(predictor, default_ben_graham),
            inputs=[img_in, thr, ben_g],
            outputs=[label_out, text_out],
        )
        gr.Markdown("---")
        gr.Markdown(DISCLAIMER)
    return demo


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: resolve weights, build the UI, launch the server."""
    parser = argparse.ArgumentParser(description="RetinaEdge-DR Gradio demo")
    parser.add_argument("--ckpt", type=Path, default=None, help="Path to best.pt")
    parser.add_argument("--onnx", type=Path, default=None, help="Path to model.onnx")
    parser.add_argument("--img-size", type=int, default=None, help="Override input size")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public gradio link")
    args = parser.parse_args(argv)

    predictor = build_predictor(args.ckpt, args.onnx, args.img_size)
    print(f"[demo] predictor ready: {predictor.source} @ {predictor.img_size}px")
    demo = build_ui(predictor)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
