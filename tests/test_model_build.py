"""Unit tests for DrNet construction and inference (network-free)."""

from __future__ import annotations

import pytest
import torch

from retinaedge.models import DrNet, build_model


@pytest.fixture(scope="module")
def tiny_model() -> DrNet:
    """Small CPU model shared by the tests in this module."""
    torch.manual_seed(0)
    return build_model(
        {
            "model": {
                "backbone": "mobilenetv3_small_100",
                "pretrained": False,
                "dropout": 0.0,
                "num_grades": 5,
            }
        }
    )


def test_build_model_defaults(tiny_model: DrNet) -> None:
    assert isinstance(tiny_model, DrNet)
    assert tiny_model.num_grades == 5
    # timm's mobilenetv3_small_100 emits 1024-d features with num_classes=0
    # (a 1x1 head conv runs after pooling; num_features=576 is NOT the output).
    assert tiny_model.embed_dim == 1024
    assert tiny_model.temperature == 1.0


def test_forward_output_shapes(tiny_model: DrNet) -> None:
    model = tiny_model.eval()
    imgs = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        outputs = model(imgs)
    assert set(outputs) == {"ordinal_logits", "refer_logits"}
    assert outputs["ordinal_logits"].shape == (2, 4)
    assert outputs["refer_logits"].shape == (2, 1)
    assert outputs["ordinal_logits"].dtype == torch.float32


def test_predict_probs_normalised_and_temperable(tiny_model: DrNet) -> None:
    from retinaedge.models import ordinal_probs

    model = tiny_model.eval()
    imgs = torch.randn(3, 3, 64, 64)
    base = model.predict_probs(imgs)
    assert base.shape == (3, 5)
    assert torch.all(base >= 0)
    assert torch.allclose(base.sum(dim=-1), torch.ones(3), atol=1e-5)

    # predict_probs must equal ordinal_probs(logits / T) for any stored T.
    for temperature in (3.0, 0.5):
        model.set_temperature(temperature)
        with torch.no_grad():
            expected = ordinal_probs(model(imgs)["ordinal_logits"] / temperature)
        assert torch.allclose(model.predict_probs(imgs), expected, atol=1e-6)

    model.set_temperature(1.0)
    assert torch.allclose(model.predict_probs(imgs), base, atol=1e-6)


def test_set_temperature_validation(tiny_model: DrNet) -> None:
    with pytest.raises(ValueError):
        tiny_model.set_temperature(0.0)
    with pytest.raises(ValueError):
        tiny_model.set_temperature(-1.0)


def test_invalid_num_grades_rejected() -> None:
    with pytest.raises(ValueError):
        DrNet(backbone="mobilenetv3_small_100", num_grades=1)


def test_build_model_reads_config_section() -> None:
    cfg = {"model": {"backbone": "mobilenetv3_small_100", "dropout": 0.2}}
    model = build_model(cfg)
    assert model.num_grades == 5
    assert isinstance(model.dropout.p, float) and model.dropout.p == pytest.approx(0.2)
