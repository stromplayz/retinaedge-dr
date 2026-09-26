"""Temperature-scaling calibration CLI (contract: ``retinaedge.eval.calibration``).

CLI:
    python -m retinaedge.eval.calibration --config <train cfg> --ckpt best.pt --out temperature.json

Fits a single temperature on the validation split by minimizing the ordinal
negative log-likelihood (LBFGS on ``log T``), then writes
``{"temperature", "ece_before", "ece_after", ...}`` for the export/metadata
steps to consume.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from retinaedge.models.build import build_model
from retinaedge.models.ordinal_ops import ordinal_probs
from retinaedge.train.metrics import ece_on_referable, jsonable
from retinaedge.train.trainer import build_split_loader, load_checkpoint
from retinaedge.utils.logging_utils import get_logger
from retinaedge.utils.seed import seed_everything

__all__ = ["fit_temperature", "calibrate", "main"]

_LOGGER = get_logger("eval.calibration")


@torch.no_grad()
def _collect_logits(model, loader, device) -> tuple[Tensor, Tensor]:
    """Gather raw ordinal logits and targets for a split (T temporarily 1.0)."""
    model.eval()  # deterministic logits: no dropout / batchnorm updates
    logits_list: list[Tensor] = []
    targets_list: list[Tensor] = []
    for imgs, targets in loader:
        outputs = model(imgs.to(device, non_blocking=True))
        logits_list.append(outputs["ordinal_logits"].float().cpu())
        targets_list.append(targets.cpu().long())
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)


def _ordinal_nll(logits: Tensor, targets: Tensor, temperature: Tensor) -> Tensor:
    """Mean negative log-likelihood of the ordinal class probabilities."""
    probs = ordinal_probs(logits / temperature)
    log_probs = probs.clamp_min(1e-12).log()
    return F.nll_loss(log_probs, targets)


def fit_temperature(
    logits: Tensor, targets: Tensor, max_iter: int = 100
) -> tuple[float, float, float]:
    """Fit the temperature by LBFGS on the ordinal NLL.

    Args:
        logits: ``(N, K-1)`` raw ordinal logits (temperature 1.0 model).
        targets: ``(N,)`` ground-truth grades.
        max_iter: LBFGS iterations.

    Returns:
        ``(temperature, nll_before, nll_after)``.
    """
    if logits.shape[0] == 0:
        raise ValueError("no logits collected — empty split?")
    nll_before = float(_ordinal_nll(logits, targets, torch.ones(1)))
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t], lr=0.5, max_iter=int(max_iter), line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = _ordinal_nll(logits, targets, log_t.exp())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = float(log_t.exp().clamp(0.05, 20.0).item())
        nll_after = float(_ordinal_nll(logits, targets, torch.tensor([temperature])))
    return temperature, nll_before, nll_after


def calibrate(
    cfg: dict, ckpt: str | Path, split: str = "val", out: str | Path | None = None
) -> dict:
    """Fit temperature on a split and write the result JSON.

    Args:
        cfg: Training config.
        ckpt: Checkpoint whose weights are calibrated.
        split: Split used for fitting (default ``val``).
        out: Output JSON path; defaults to ``temperature.json`` next to ckpt.

    Returns:
        The written payload dict.
    """
    seed_everything(int(cfg.get("train", {}).get("seed", 0)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = load_checkpoint(ckpt, map_location="cpu")
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model = build_model(cfg).to(device)
    model.load_state_dict(state)
    model.set_temperature(1.0)  # fit on raw logits

    loader = build_split_loader(cfg, split, shuffle=False)
    logits, targets = _collect_logits(model, loader, device)
    temperature, nll_before, nll_after = fit_temperature(logits, targets)

    probs_before = ordinal_probs(logits).numpy()
    probs_after = ordinal_probs(logits / temperature).numpy()
    targets_np = targets.numpy()
    ece_before = ece_on_referable(probs_before, targets_np)
    ece_after = ece_on_referable(probs_after, targets_np)

    result = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "n": int(targets.shape[0]),
        "split": split,
        "ckpt": str(ckpt),
    }
    out_path = Path(out) if out is not None else Path(ckpt).parent / "temperature.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    _LOGGER.info(
        "temperature=%.4f (nll %.4f -> %.4f, ece %.4f -> %.4f) -> %s",
        temperature,
        nll_before,
        nll_after,
        ece_before,
        ece_after,
        out_path,
    )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m retinaedge.eval.calibration",
        description="Fit temperature scaling on a split and write temperature.json.",
    )
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--ckpt", required=True, help="path to best.pt / last.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None, help="output JSON (default: next to ckpt)")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="dotted config overrides, e.g. data.img_size=64 data.synthetic.n_train=64",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    from retinaedge.utils.config import load_config

    cfg = load_config(args.config, args.overrides)
    try:
        calibrate(cfg, args.ckpt, split=args.split, out=args.out)
    except Exception:
        _LOGGER.exception("calibration failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
