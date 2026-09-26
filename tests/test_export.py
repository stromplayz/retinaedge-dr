"""Tests for retinaedge.export (agent 2-c): wrapper, ONNX export, benchmark, metadata.

Everything runs offline with a contract-conformant stub model — no pretrained
weights, no network, no TensorFlow. TFLite conversion paths are exercised only
for argument handling / graceful degradation here (TF is not a test dependency).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from retinaedge.export import benchmark as benchmark_mod
from retinaedge.export import export_onnx, metadata
from retinaedge.export import wrappers as wrappers_mod
from retinaedge.export.defaults import DEFAULTS, resolve_export_settings
from retinaedge.export.representative import RepresentativeDataset, preprocess_calibration_image
from retinaedge.export.wrappers import GRADE_LABELS, IMAGENET_MEAN, IMAGENET_STD, InferenceWrapper
from retinaedge.models.ordinal_ops import ordinal_probs

requires_onnx = pytest.mark.requires_onnx
onnx_available = (
    importlib.util.find_spec("onnx") is not None
    and importlib.util.find_spec("onnxruntime") is not None
)
aet_available = importlib.util.find_spec("ai_edge_torch") is not None


class StubDrNet(nn.Module):
    """Minimal DrNet stand-in obeying the interface contract (tests only)."""

    embed_dim = 8

    def __init__(self, num_grades: int = 5) -> None:
        super().__init__()
        self.num_grades = num_grades
        self.temperature = 1.0
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ordinal_head = nn.Linear(8, num_grades - 1)
        self.refer_head = nn.Linear(8, 1)

    def forward(self, imgs: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.features(imgs).flatten(1)
        return {"ordinal_logits": self.ordinal_head(z), "refer_logits": self.refer_head(z)}

    def set_temperature(self, t: float) -> None:
        self.temperature = float(t)

    @torch.no_grad()
    def predict_probs(self, imgs: torch.Tensor) -> torch.Tensor:
        return ordinal_probs(self.forward(imgs)["ordinal_logits"] / self.temperature)


@pytest.fixture()
def stub_cfg(tmp_path: Path) -> Path:
    cfg = {
        "data": {"img_size": 32, "dataset": "synthetic"},
        "model": {"backbone": "stub", "pretrained": False, "dropout": 0.0, "num_grades": 5},
        "train": {"seed": 0, "save_dir": str(tmp_path / "out")},
        "export": {"latency_runs": 3, "warmup": 1},
    }
    path = tmp_path / "train_stub.yaml"
    lines = []
    for section, values in cfg.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            lines.append(f"  {key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def stub_ckpt(tmp_path: Path) -> Path:
    model = StubDrNet()
    with torch.no_grad():  # non-trivial, deterministic-ish weights
        for param in model.parameters():
            param.add_(0.1)
    payload = {
        "state_dict": model.state_dict(),
        "cfg": {"model": {"backbone": "stub"}},
        "val_qwk": 0.42,
        "temperature": 1.7,
        "epoch": 3,
    }
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


@pytest.fixture()
def stub_builder(monkeypatch: pytest.MonkeyPatch):
    """Redirect build_export_model's lazy import to the stub (CLI-level tests)."""
    monkeypatch.setattr(wrappers_mod, "_import_build_model", lambda: lambda cfg: StubDrNet())


# --------------------------------------------------------------------------- #
# InferenceWrapper
# --------------------------------------------------------------------------- #
def test_wrapper_output_contract() -> None:
    model = StubDrNet()
    wrapper = InferenceWrapper(model)
    x = torch.randn(4, 3, 32, 32)
    probs = wrapper(x)
    assert probs.shape == (4, 5)
    assert probs.dtype == torch.float32
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_wrapper_temperature_matches_predict_probs() -> None:
    """Wrapper semantics must equal DrNet.predict_probs with the same temperature."""
    model = StubDrNet()
    x = torch.randn(2, 3, 32, 32)
    wrapper = InferenceWrapper(model, temperature=1.7)
    model.set_temperature(1.7)
    assert torch.allclose(wrapper(x), model.predict_probs(x), atol=1e-6)
    # and it must differ from the unscaled variant
    assert not torch.allclose(wrapper(x), InferenceWrapper(model, temperature=1.0)(x))


def test_wrapper_reads_model_temperature_and_validates() -> None:
    model = StubDrNet()
    model.set_temperature(2.5)
    assert InferenceWrapper(model).temperature == 2.5
    with pytest.raises(ValueError, match="temperature"):
        InferenceWrapper(model, temperature=0.0)
    with pytest.raises(TypeError, match="ordinal_logits"):
        InferenceWrapper(nn.Identity())(torch.randn(1, 3, 8, 8))


def test_build_export_model_loads_ckpt_and_temperature(stub_ckpt: Path) -> None:
    cfg = {"model": {"backbone": "stub"}}
    model, payload = wrappers_mod.build_export_model(cfg, stub_ckpt, model=StubDrNet())
    assert payload["temperature"] == pytest.approx(1.7)
    assert model.temperature == pytest.approx(1.7)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    # state_dict actually applied: rebuilt-from-scratch model must match outputs
    rebuilt = wrappers_mod.build_export_model(cfg, stub_ckpt, model=StubDrNet())[0]
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(model(x)["ordinal_logits"], rebuilt(x)["ordinal_logits"])


def test_build_export_model_strips_ckpt_prefixes(tmp_path: Path) -> None:
    model = StubDrNet()
    payload = {"state_dict": {f"module.{k}": v for k, v in model.state_dict().items()}}
    ckpt = tmp_path / "wrapped.pt"
    torch.save(payload, ckpt)
    loaded, _ = wrappers_mod.build_export_model({}, ckpt, model=model)
    x = torch.randn(1, 3, 32, 32)
    ref = StubDrNet()
    ref.load_state_dict(model.state_dict())
    assert torch.allclose(loaded(x)["ordinal_logits"], ref(x)["ordinal_logits"])


def test_build_export_model_requires_builder_module(tmp_path: Path) -> None:
    if importlib.util.find_spec("retinaedge.models.build") is not None:
        pytest.skip("retinaedge.models.build landed (agent 2-b) — real builder path is exercised")
    with pytest.raises(RuntimeError, match="models.build"):
        wrappers_mod.build_export_model({"model": {}})  # no model, no lazy import patch


def test_load_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        wrappers_mod.load_checkpoint(tmp_path / "nope.pt")


# --------------------------------------------------------------------------- #
# ONNX export
# --------------------------------------------------------------------------- #
@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_onnx_export_static_and_parity(tmp_path: Path) -> None:
    wrapper = InferenceWrapper(StubDrNet(), temperature=1.3)
    out = export_onnx.export_to_onnx(wrapper, 32, tmp_path / "model.onnx", opset=17)
    assert out.exists()
    worst = export_onnx.verify_onnx_parity(out, wrapper, 32, atol=1e-3)
    assert worst <= 1e-3


@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_onnx_export_dynamic_batch(tmp_path: Path) -> None:
    import onnxruntime as ort

    wrapper = InferenceWrapper(StubDrNet())
    out = export_onnx.export_to_onnx(wrapper, 32, tmp_path / "dyn.onnx", dynamic_batch=True)
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": torch.randn(3, 3, 32, 32).numpy()})[0]
    assert got.shape == (3, 5)


@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_export_onnx_cli_end_to_end(
    tmp_path: Path, stub_cfg: Path, stub_ckpt: Path, stub_builder, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "model.onnx"
    rc = export_onnx.main(
        [
            "--config",
            str(stub_cfg),
            "--ckpt",
            str(stub_ckpt),
            "--out",
            str(out),
            "--img-size",
            "32",
            "--runs",
            "3",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "parity" in printed.lower() and "PASS" in printed
    assert "latency" in printed.lower()

    # temperature from the ckpt (1.7) must flow into the exported graph:
    import onnxruntime as ort

    rebuilt, _ = wrappers_mod.build_export_model({}, stub_ckpt, model=StubDrNet())
    wrapper = InferenceWrapper(rebuilt)  # temperature 1.7 comes from the ckpt payload
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    ref = wrapper(x).numpy()
    got = session.run(None, {"image": x.numpy()})[0]
    assert np.abs(ref - got).max() <= 1e-3


@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_export_onnx_cli_without_ckpt(tmp_path: Path, stub_cfg: Path, stub_builder) -> None:
    out = tmp_path / "rand.onnx"
    rc = export_onnx.main(
        ["--config", str(stub_cfg), "--out", str(out), "--img-size", "32", "--no-verify"]
    )
    assert rc == 0
    assert out.exists()


@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_export_onnx_parity_failure_returns_error(
    tmp_path: Path, stub_builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the parity check reports a violation, the CLI must exit non-zero."""
    monkeypatch.setattr(
        export_onnx, "verify_onnx_parity", lambda *a, **k: 0.5
    )  # simulated violation
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "data:\n  img_size: 32\nmodel:\n  backbone: stub\ntrain:\n  seed: 0\n", encoding="utf-8"
    )
    rc = export_onnx.main(
        ["--config", str(cfg), "--out", str(tmp_path / "m.onnx"), "--atol", "1e-12"]
    )
    assert rc == 1


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_benchmark_onnx(tmp_path: Path) -> None:
    out = export_onnx.export_to_onnx(InferenceWrapper(StubDrNet()), 32, tmp_path / "m.onnx")
    stats = benchmark_mod.benchmark_model(out, runs=5, warmup=1)
    assert stats["backend"] == "onnxruntime"
    assert stats["runs"] == 5
    assert stats["mean_ms"] > 0 and stats["fps"] > 0
    assert stats["p95_ms"] >= stats["p50_ms"] >= stats["min_ms"] - 1e-9


@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_benchmark_cli_and_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = export_onnx.export_to_onnx(InferenceWrapper(StubDrNet()), 32, tmp_path / "m.onnx")
    json_path = tmp_path / "bench.json"
    rc = benchmark_mod.main(
        [
            "--model",
            str(out),
            "--img-size",
            "32",
            "--runs",
            "3",
            "--warmup",
            "1",
            "--json",
            str(json_path),
        ]
    )
    assert rc == 0
    assert json_path.exists()
    stats = json.loads(json_path.read_text(encoding="utf-8"))
    assert stats["input_shape"][0] == 1
    printed = capsys.readouterr().out
    assert "p95" in printed and "throughput" in printed.lower()


def test_benchmark_errors(tmp_path: Path) -> None:
    assert benchmark_mod.main(["--model", str(tmp_path / "missing.onnx")]) == 1
    junk = tmp_path / "junk.txt"
    junk.write_text("x", encoding="utf-8")
    assert benchmark_mod.main(["--model", str(junk)]) == 1


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_metadata_onnx(tmp_path: Path) -> None:
    out = export_onnx.export_to_onnx(InferenceWrapper(StubDrNet()), 32, tmp_path / "m.onnx")
    info = metadata.write_metadata(out, tmp_path, temperature=1.7)
    labels = (tmp_path / "labels.txt").read_text(encoding="utf-8").splitlines()
    assert labels == list(GRADE_LABELS)
    on_disk = json.loads((tmp_path / "model_info.json").read_text(encoding="utf-8"))
    assert on_disk == info  # write_metadata returns exactly what it serializes
    assert on_disk["input"]["shape"][:2] == [1, 3]
    assert on_disk["input"]["normalization"]["mean"] == list(IMAGENET_MEAN)
    assert on_disk["input"]["normalization"]["std"] == list(IMAGENET_STD)
    assert on_disk["grades"]["labels"] == list(GRADE_LABELS)
    assert on_disk["postprocess"]["referable_threshold"] == pytest.approx(0.5)
    assert on_disk["temperature"] == pytest.approx(1.7)
    assert on_disk["format"] == "onnx"


def test_metadata_cli_missing_model(tmp_path: Path) -> None:
    assert metadata.main(["--model", str(tmp_path / "no.onnx"), "--out-dir", str(tmp_path)]) == 1


# --------------------------------------------------------------------------- #
# Representative dataset (int8 PTQ calibration)
# --------------------------------------------------------------------------- #
def test_preprocess_calibration_image_normalizes() -> None:
    rgb = np.full((64, 64, 3), 127, dtype=np.uint8)
    sample = preprocess_calibration_image(rgb, 32)
    assert sample.shape == (1, 3, 32, 32)
    assert sample.dtype == np.float32
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        expected = (127.0 / 255.0 - mean) / std
        assert sample[0, channel].mean() == pytest.approx(expected, abs=1e-4)


def test_representative_dataset_synthetic_deterministic() -> None:
    ds = RepresentativeDataset(32, num_samples=4, seed=0)
    batches = [batch for (batch,) in ds]
    assert len(batches) == 4
    assert all(b.shape == (1, 3, 32, 32) and b.dtype == np.float32 for b in batches)
    again = [batch for (batch,) in RepresentativeDataset(32, num_samples=4, seed=0)]
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))


def test_representative_dataset_from_image_dir(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    for i in range(5):
        img = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    (tmp_path / "ignored.txt").write_text("not an image", encoding="utf-8")
    ds = RepresentativeDataset(32, image_dir=tmp_path, num_samples=3)
    batches = [batch for (batch,) in ds]
    assert len(batches) == 3
    assert all(b.shape == (1, 3, 32, 32) for b in batches)


def test_representative_dataset_cycles_when_fewer_images(tmp_path: Path) -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    ds = RepresentativeDataset(32, image_dir=tmp_path, num_samples=4)
    batches = [batch for (batch,) in ds]
    assert len(batches) == 4
    assert np.array_equal(batches[0], batches[3])


def test_representative_dataset_bad_dir(tmp_path: Path) -> None:
    ds = RepresentativeDataset(32, image_dir=tmp_path / "nope", num_samples=2)
    with pytest.raises(FileNotFoundError):
        list(ds)


# --------------------------------------------------------------------------- #
# Export settings resolution
# --------------------------------------------------------------------------- #
def test_resolve_settings_precedence(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("export:\n  opset: 15\n  verify: false\n", encoding="utf-8")
    cfg = {"export": {"opset": 13, "latency_runs": 7}}
    settings = resolve_export_settings(cfg, profile, {"opset": 11})
    assert settings["opset"] == 11  # cli wins
    assert settings["verify"] is False  # profile beats cfg block
    assert settings["latency_runs"] == 7  # cfg block beats defaults
    assert settings["referable_threshold"] == DEFAULTS["referable_threshold"]  # default survives
    # None-valued CLI entries must not clobber lower-precedence sources
    settings2 = resolve_export_settings(cfg, profile, {"opset": None})
    assert settings2["opset"] == 13
    # resolve must not mutate DEFAULTS
    assert DEFAULTS["opset"] == 17


def test_resolve_settings_missing_profile(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_export_settings(None, tmp_path / "nope.yaml", None)


# --------------------------------------------------------------------------- #
# TFLite CLI (graceful degradation — TF / ai-edge-torch are not test deps)
# --------------------------------------------------------------------------- #
def test_tflite_cli_requires_source() -> None:
    from retinaedge.export import export_tflite

    # neither --direct nor --onnx -> handled by main(), not argparse
    assert export_tflite.main(["--out", "/tmp/x.tflite"]) == 2


def test_tflite_cli_missing_deps_reports_cleanly(tmp_path: Path, stub_cfg: Path) -> None:
    from retinaedge.export import export_tflite

    if aet_available:
        pytest.skip("ai-edge-torch installed — real conversion path not exercised in unit tests")
    rc = export_tflite.main(
        ["--direct", "--config", str(stub_cfg), "--out", str(tmp_path / "m.tflite")]
    )
    assert rc == 1  # clean error, no traceback


def test_tflite_cli_direct_requires_config(tmp_path: Path) -> None:
    from retinaedge.export import export_tflite

    assert export_tflite.main(["--direct", "--out", str(tmp_path / "m.tflite")]) == 2


# --------------------------------------------------------------------------- #
# Integration with the REAL DrNet (agent 2-b module, offline backbone)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@requires_onnx
@pytest.mark.skipif(not onnx_available, reason="onnx/onnxruntime not installed")
def test_full_export_pipeline_with_real_dranet(tmp_path: Path) -> None:
    """Real timm backbone -> ckpt -> ONNX -> benchmark + metadata, all offline."""
    from retinaedge.utils.seed import seed_everything

    seed_everything(0)
    cfg = {
        "model": {
            "backbone": "mobilenetv3_small_100",
            "pretrained": False,
            "dropout": 0.0,
            "num_grades": 5,
        },
        "data": {"img_size": 64},
        "train": {"seed": 0},
    }
    model = wrappers_mod.build_export_model(cfg)[0]  # exercises the lazy 2-b import
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "val_qwk": 0.1,
            "temperature": 1.35,
            "epoch": 1,
        },
        ckpt,
    )
    cfg_path = tmp_path / "train_real.yaml"
    cfg_path.write_text(
        "data:\n  img_size: 64\nmodel:\n  backbone: mobilenetv3_small_100\n"
        "  pretrained: false\n  dropout: 0.0\n  num_grades: 5\ntrain:\n  seed: 0\n",
        encoding="utf-8",
    )
    out = tmp_path / "model.onnx"
    rc = export_onnx.main(
        [
            "--config",
            str(cfg_path),
            "--ckpt",
            str(ckpt),
            "--out",
            str(out),
            "--img-size",
            "64",
            "--runs",
            "3",
            "--warmup",
            "1",
        ]
    )
    assert rc == 0

    assert (
        benchmark_mod.main(
            ["--model", str(out), "--img-size", "64", "--runs", "3", "--warmup", "1"]
        )
        == 0
    )
    info = metadata.write_metadata(out, tmp_path, temperature=1.35)
    assert info["input"]["shape"] == [1, 3, 64, 64]
    assert (tmp_path / "labels.txt").read_text(encoding="utf-8").splitlines() == list(GRADE_LABELS)
