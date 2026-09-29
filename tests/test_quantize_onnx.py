"""Tests for ONNX int8 dynamic quantization + mobile fit report."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from retinaedge.export.export_onnx import export_to_onnx
from retinaedge.export.quantize_onnx import mobile_fit_report, quantize_onnx_dynamic
from retinaedge.export.wrappers import InferenceWrapper
from retinaedge.models.build import build_model

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

CFG = {
    "model": {
        "backbone": "mobilenetv3_small_050",
        "pretrained": False,
        "dropout": 0.0,
        "num_grades": 5,
    }
}


@pytest.fixture(scope="module")
def fp32_onnx(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("quant")
    torch.manual_seed(0)
    model = build_model(CFG)
    wrapper = InferenceWrapper(model, temperature=1.0)
    return export_to_onnx(wrapper, 64, d / "model.onnx")


class TestQuantize:
    def test_shrinks_and_low_drift(self, fp32_onnx, tmp_path):
        report = quantize_onnx_dynamic(fp32_onnx, tmp_path / "int8.onnx", img_size=64)
        assert report["compression_ratio"] > 1.2
        assert report["drift_max_abs"] < 0.05
        assert report["latency_ms_int8_p50"] > 0

    def test_fit_report(self):
        fit = mobile_fit_report(2 * 1024 * 1024, 30.0)
        assert fit["fits_mobile"] is True
        fit_big = mobile_fit_report(50 * 1024 * 1024, 300.0)
        assert fit_big["fits_mobile"] is False

    def test_int8_session_runs(self, fp32_onnx, tmp_path):
        out = tmp_path / "int8.onnx"
        quantize_onnx_dynamic(fp32_onnx, out, img_size=64)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = torch.randn(1, 3, 64, 64).numpy().astype("float32") * 0 + 0.5
        y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        assert y.shape == (1, 5)
        assert abs(float(y.sum()) - 1.0) < 0.05
