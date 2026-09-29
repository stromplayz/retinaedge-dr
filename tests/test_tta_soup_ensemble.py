"""Tests for TTA / soup / ensemble — inference-time accuracy loopholes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from retinaedge.models.build import build_model
from retinaedge.models.ensemble import EnsemblePredictor, load_ensemble
from retinaedge.models.soup import average_state_dicts, make_soup
from retinaedge.models.tta import tta_probs, tta_variants

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def small_cfg() -> dict:
    return {
        "data": {"img_size": 64},
        "model": {
            "backbone": "mobilenetv3_small_050",
            "pretrained": False,
            "dropout": 0.0,
            "num_grades": 5,
        },
    }


@pytest.fixture(scope="module")
def ckpt_pair(tmp_path_factory, small_cfg) -> tuple[Path, Path]:
    """Two checkpoints with identical architecture, different random weights."""
    d = tmp_path_factory.mktemp("soup")
    paths = []
    for i, seed in enumerate((0, 1)):
        torch.manual_seed(seed)
        model = build_model(small_cfg)
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": small_cfg,
            "val_qwk": 0.4 + 0.05 * i,
            "temperature": 0.9 + 0.1 * i,
            "epoch": 3 + i,
        }
        p = d / f"m{i}.pt"
        torch.save(payload, p)
        paths.append(p)
    return paths[0], paths[1]


class TestTTA:
    def test_variants_include_identity_and_flip(self):
        imgs = torch.randn(2, 3, 32, 32)
        variants = tta_variants(imgs, scales=(1.0,), hflip=True)
        assert len(variants) == 2
        assert torch.equal(variants[0], imgs)
        assert torch.equal(variants[1], torch.flip(imgs, dims=[-1]))

    def test_multiscale_shapes(self):
        imgs = torch.randn(1, 3, 32, 32)
        for v in tta_variants(imgs, scales=(1.0, 1.25), hflip=True):
            assert v.shape[-2] >= 32 or v.shape[-2] == 40

    def test_tta_probs_rows_sum_to_one(self, small_cfg):
        model = build_model(small_cfg)
        model.eval()
        imgs = torch.randn(2, 3, 64, 64)
        probs = tta_probs(model, imgs)
        assert probs.shape == (2, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
        assert (probs >= 0).all()


class TestSoup:
    def test_average_state_dicts_shapes(self, ckpt_pair):
        sd = average_state_dicts(ckpt_pair)
        ref = build_model({"model": {"backbone": "mobilenetv3_small_050", "pretrained": False}})
        assert set(sd) == set(ref.state_dict())

    def test_mismatched_keys_rejected(self, ckpt_pair, tmp_path):
        torch.save(
            {
                "state_dict": {"bogus": torch.zeros(1)},
                "cfg": {},
                "temperature": 1.0,
                "epoch": 0,
                "val_qwk": 0.0,
            },
            tmp_path / "bad.pt",
        )
        with pytest.raises(ValueError):
            average_state_dicts([ckpt_pair[0], tmp_path / "bad.pt"])

    def test_make_soup_payload(self, ckpt_pair, tmp_path):
        out = tmp_path / "soup" / "soup.pt"
        path = make_soup(ckpt_pair, out)
        assert path.exists()
        payload = torch.load(path, weights_only=False)
        assert payload["val_qwk"] == pytest.approx(0.425)
        assert payload["epoch"] == 4
        assert len(payload["soup_of"]) == 2


class TestEnsemble:
    def test_weighted_average_matches_manual(self, ckpt_pair, small_cfg):
        ens = load_ensemble(small_cfg, ckpt_pair, weights=[3.0, 1.0])
        imgs = torch.randn(2, 3, 64, 64)
        got = ens.predict_probs(imgs)
        # manual recompute via members (3:1 weights normalized to 0.75/0.25)
        m0, m1 = ens.members
        expected = 0.75 * m0.predict_probs(imgs) + 0.25 * m1.predict_probs(imgs)
        assert torch.allclose(got, expected, atol=1e-6)

    def test_deterministic(self, ckpt_pair, small_cfg):
        ens = load_ensemble(small_cfg, ckpt_pair)
        imgs = torch.randn(1, 3, 64, 64)
        assert torch.allclose(ens.predict_probs(imgs), ens.predict_probs(imgs))

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            EnsemblePredictor([])


if __name__ == "__main__":
    pytest.main([__file__])
