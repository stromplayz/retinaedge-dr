"""Export-safe inference wrapper and checkpoint loading.

Deployment graph contract (docs/INTERFACES.md, consumed by the Android app):

    input : float32 ``(1, 3, H, W)`` ImageNet-normalized
            (mean ``(0.485, 0.456, 0.406)``, std ``(0.229, 0.224, 0.225)``)
    output: float32 ``(1, 5)`` per-grade probabilities, rows sum to 1

``InferenceWrapper`` fuses DrNet + temperature scaling + the cumulative-link
conversion into a single ``forward(x) -> probs`` module so the exported graph is
plain tensor-in / tensor-out (no dict outputs, no post-processing on device).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from retinaedge.models.ordinal_ops import ordinal_probs
from retinaedge.utils.logging_utils import get_logger

__all__ = [
    "GRADE_LABELS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NUM_GRADES",
    "ONNX_INPUT_NAME",
    "ONNX_OUTPUT_NAME",
    "InferenceWrapper",
    "build_export_model",
    "build_inference_model",
    "load_checkpoint",
]

logger = get_logger(__name__)

#: Fixed export constants — the Android reader and metadata tooling rely on these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_GRADES = 5
GRADE_LABELS = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")
ONNX_INPUT_NAME = "image"
ONNX_OUTPUT_NAME = "probs"

_CKPT_PREFIXES = ("module.", "n_ema.", "ema_model.")


class InferenceWrapper(nn.Module):
    """Wraps DrNet (+ temperature) -> forward(x) -> probs (B,5) float32.

    Export-safe (no dict output). The wrapped model's raw ``forward`` output
    (dict with ``ordinal_logits``/``refer_logits``) is consumed unchanged;
    temperature scaling and the cumulative-link -> grade-probability conversion
    happen here, mirroring ``DrNet.predict_probs`` (per the interface contract,
    temperature applies at probability time, never inside the training loss path).

    Args:
        model: A contract-conformant DrNet (``forward`` -> dict with
            ``"ordinal_logits"`` ``(B, K-1)``) or any module returning raw
            ``(B, K-1)`` ordinal logits directly.
        temperature: Calibration temperature applied to the ordinal logits.
            Defaults to the model's stored ``temperature`` attribute (1.0 unset).
        num_grades: Number of ordinal grades (ICDRSS: 5). Informational.
    """

    def __init__(
        self,
        model: nn.Module,
        temperature: float | None = None,
        num_grades: int = NUM_GRADES,
    ) -> None:
        super().__init__()
        if temperature is None:
            temperature = float(getattr(model, "temperature", 1.0))
        temperature = float(temperature)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.model = model.eval()
        self.temperature = temperature
        self.num_grades = int(num_grades)

    def forward(self, imgs: Tensor) -> Tensor:
        """Map normalized images ``(B, 3, H, W)`` to grade probs ``(B, K)`` float32."""
        out = self.model(imgs)
        logits = self._extract_ordinal_logits(out)
        probs = ordinal_probs(logits / self.temperature)
        return probs.to(torch.float32)

    @staticmethod
    def _extract_ordinal_logits(out: Any) -> Tensor:
        """Accept the contract dict output, or a bare ``(B, K-1)`` logits tensor."""
        if isinstance(out, dict):
            if "ordinal_logits" not in out:
                raise KeyError(
                    f"model dict output must contain 'ordinal_logits', got keys {sorted(out)}"
                )
            return out["ordinal_logits"]
        if isinstance(out, Tensor) and out.dim() == 2:
            return out
        raise TypeError(
            "expected model output to be a dict with 'ordinal_logits' or a (B, K-1) tensor, "
            f"got {type(out)!r}"
        )

    @torch.no_grad()
    def predict_probs(self, imgs: Tensor) -> Tensor:
        """Alias of :meth:`forward` for parity with the DrNet API."""
        return self.forward(imgs)


def load_checkpoint(path: str | Path) -> dict:
    """Load a trainer checkpoint payload (``{"state_dict", "cfg", "temperature", ...}``).

    ``weights_only=True`` is tried first (safe); payloads saved by trusted local
    tooling that carry exotic objects fall back to a full unpickle with a warning.

    Returns:
        The raw payload dict (empty ``state_dict`` keys are the caller's problem).
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - payload contents are repo-controlled
        logger.warning("weights_only load failed (%s); retrying full unpickle of local file", exc)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict, got {type(payload)!r}")
    return payload


def _strip_ckpt_prefixes(state_dict: dict) -> dict:
    """Drop common wrapper prefixes (DataParallel / EMA) from checkpoint keys."""
    stripped: dict = {}
    for key, value in state_dict.items():
        for prefix in _CKPT_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        stripped[key] = value
    return stripped


def _import_build_model():
    """Lazily import ``retinaedge.models.build.build_model`` (owner: agent 2-b)."""
    try:
        from retinaedge.models.build import build_model
    except ImportError as exc:  # pragma: no cover - exercised when 2-b lands
        raise RuntimeError(
            "retinaedge.models.build is unavailable — install the package "
            "(pip install -e .) or pass a prebuilt model to build_export_model()."
        ) from exc
    return build_model


def build_export_model(
    cfg: dict,
    ckpt_path: str | Path | None = None,
    model: nn.Module | None = None,
) -> tuple[nn.Module, dict]:
    """Build DrNet from ``cfg`` and load training weights + temperature, eval mode.

    Args:
        cfg: Train config (``cfg["model"]`` drives ``build_model``).
        ckpt_path: Trainer checkpoint (``{"state_dict", "temperature", ...}``).
            ``None`` keeps the freshly built (random) weights — useful for
            graph-only export before a training run exists.
        model: Pre-built model to skip ``build_model`` (used by tests/tools).

    Returns:
        ``(model, payload)`` — the eval-mode, frozen model and the raw payload.
    """
    payload: dict = {}
    if model is None:
        model = _import_build_model()(cfg)
    if ckpt_path is not None:
        payload = load_checkpoint(ckpt_path)
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise KeyError(f"checkpoint {ckpt_path} carries no 'state_dict' — not a trainer ckpt?")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            incompatible = model.load_state_dict(_strip_ckpt_prefixes(state_dict), strict=False)
            logger.warning(
                "strict load failed (%s); recovered after prefix-stripping: "
                "missing=%s unexpected=%s",
                exc,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        temperature = payload.get("temperature")
        if temperature is not None and hasattr(model, "set_temperature"):
            model.set_temperature(float(temperature))
            logger.info("applied checkpoint temperature=%.4f", float(temperature))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, payload


def build_inference_model(
    cfg: dict,
    ckpt_path: str | Path | None = None,
    model: nn.Module | None = None,
) -> InferenceWrapper:
    """Convenience: :func:`build_export_model` wrapped in :class:`InferenceWrapper`."""
    export_model, _payload = build_export_model(cfg, ckpt_path, model)
    return InferenceWrapper(export_model)
