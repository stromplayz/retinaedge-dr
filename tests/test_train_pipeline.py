"""End-to-end pipeline test: trainer -> evaluate -> calibration on synthetic data.

Marked ``slow`` (deselect with ``-m "not slow"``); still fully network-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from retinaedge.eval.calibration import main as calibration_main
from retinaedge.eval.evaluate import main as evaluate_main
from retinaedge.train.trainer import load_checkpoint
from retinaedge.train.trainer import main as trainer_main

pytestmark = pytest.mark.slow

_OVERRIDES = [
    "data.synthetic.n_train=64",
    "data.synthetic.n_val=24",
    "data.synthetic.size=32",
    "data.img_size=32",
    "data.batch_size=16",
    "train.epochs=1",
    "train.patience=5",
]


def test_full_train_eval_calibrate_pipeline(tmp_path: Path) -> None:
    save_dir = tmp_path / "run"
    args = [
        "--config",
        "configs/train/smoke.yaml",
        f"train.save_dir={save_dir}",
        *_OVERRIDES,
    ]

    # --- trainer -----------------------------------------------------------
    assert trainer_main(args) == 0
    for artifact in ("best.pt", "last.pt", "history.csv", "metrics.json"):
        assert (save_dir / artifact).exists(), f"missing artifact {artifact}"

    payload = load_checkpoint(save_dir / "best.pt")
    assert set(payload) >= {"state_dict", "cfg", "val_qwk", "temperature", "epoch"}
    assert payload["temperature"] == pytest.approx(1.0)
    assert isinstance(payload["val_qwk"], float)

    history = (save_dir / "history.csv").read_text().strip().splitlines()
    assert len(history) == 2  # header + one epoch
    assert "val_qwk" in history[0]

    # --- evaluate -----------------------------------------------------------
    assert (
        evaluate_main(
            ["--config", "configs/train/smoke.yaml", "--ckpt", str(save_dir / "best.pt")]
            + _OVERRIDES
        )
        == 0
    )
    eval_json = save_dir / "eval.json"
    assert eval_json.exists()
    assert "confusion_matrix" in eval_json.read_text()

    # --- calibration ----------------------------------------------------------
    assert (
        calibration_main(
            [
                "--config",
                "configs/train/smoke.yaml",
                "--ckpt",
                str(save_dir / "best.pt"),
                "--out",
                str(save_dir / "temperature.json"),
            ]
            + _OVERRIDES
        )
        == 0
    )
    import json

    temp = json.loads((save_dir / "temperature.json").read_text())
    assert 0.05 <= temp["temperature"] <= 20.0
    assert temp["n"] == 24
    assert temp["nll_after"] <= temp["nll_before"] + 1e-6  # LBFGS must not worsen NLL


def test_trainer_rejects_unknown_dataset(tmp_path: Path) -> None:
    args = [
        "--config",
        "configs/train/smoke.yaml",
        f"train.save_dir={tmp_path / 'x'}",
        "data.dataset=does_not_exist",
    ]
    assert trainer_main(args) == 1


def test_checkpoint_roundtrip_into_fresh_model(tmp_path: Path) -> None:
    """best.pt weights must load into a freshly built model (export path)."""
    from retinaedge.models import build_model
    from retinaedge.utils.config import load_config

    save_dir = tmp_path / "run"
    assert (
        trainer_main(
            ["--config", "configs/train/smoke.yaml", f"train.save_dir={save_dir}", *_OVERRIDES]
        )
        == 0
    )
    cfg = load_config("configs/train/smoke.yaml", _OVERRIDES)
    model = build_model(cfg)
    payload = load_checkpoint(save_dir / "best.pt")
    model.load_state_dict(payload["state_dict"])
    model.set_temperature(float(payload["temperature"]))
    imgs = torch.randn(1, 3, 32, 32)
    probs = model.predict_probs(imgs)
    assert probs.shape == (1, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_mixup_epoch_runs(tmp_path: Path) -> None:
    """v0.3.0: one epoch with mixup + label smoothing must train and save."""
    save_dir = tmp_path / "run_mixup"
    args = [
        "--config",
        "configs/train/smoke.yaml",
        f"train.save_dir={save_dir}",
        *_OVERRIDES,
        "train.mixup_alpha=0.3",
        "train.loss.label_smoothing=0.05",
    ]
    assert trainer_main(args) == 0
    payload = load_checkpoint(save_dir / "best.pt")
    assert isinstance(payload["val_qwk"], float)
