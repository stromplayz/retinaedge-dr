"""Tests for the accuracy-loophole stack (TTA / soup / ensemble / thresholds /
stats / pseudo-labels / improvement ladder). All tests are deterministic and
network-free; models are lightweight stubs or tiny real architectures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from retinaedge.eval.stats import (
    accuracy_ci_report,
    bootstrap_accuracy_ci,
    mcnemar_test,
    wilson_ci,
)
from retinaedge.eval.threshold_search import (
    coordinate_ascent_cuts,
    decode_with_cuts,
    search_referable_threshold,
)
from retinaedge.models.ensemble import EnsemblePredictor
from retinaedge.models.soup import average_state_dicts, make_soup
from retinaedge.models.tta import tta_probs, tta_variants


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _StubModel(nn.Module):
    """Returns a fixed probability row for every input (flip/scale invariant)."""

    def __init__(self, probs=(0.1, 0.2, 0.4, 0.2, 0.1)) -> None:
        super().__init__()
        self.register_buffer("p", torch.tensor(probs, dtype=torch.float32))

    def predict_probs(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.p.repeat(imgs.shape[0], 1)


class _InputAwareModel(nn.Module):
    """Probabilities that depend on the mean brightness of the input."""

    def predict_probs(self, imgs: torch.Tensor) -> torch.Tensor:
        base = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.1], device=imgs.device)
        shift = (imgs.mean(dim=(1, 2, 3)) > 0).float().unsqueeze(1)
        rows = base.unsqueeze(0) + 0.05 * shift
        return rows / rows.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# TTA
# --------------------------------------------------------------------------- #
def test_tta_variants_identity_first_and_flip_count() -> None:
    x = torch.randn(2, 3, 8, 8)
    variants = tta_variants(x, scales=(1.0,), hflip=True)
    assert len(variants) == 2
    assert torch.equal(variants[0], x)
    assert torch.equal(variants[1], torch.flip(x, dims=[-1]))


def test_tta_variants_multi_scale_shapes() -> None:
    x = torch.randn(1, 3, 16, 16)
    variants = tta_variants(x, scales=(0.5, 1.0), hflip=False)
    assert variants[0].shape == (1, 3, 8, 8)
    assert variants[1].shape == (1, 3, 16, 16)


def test_tta_probs_rows_sum_to_one_and_match_stub() -> None:
    x = torch.randn(4, 3, 8, 8)
    out = tta_probs(_StubModel(), x, scales=(1.0,), hflip=True)
    assert out.shape == (4, 5)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out, _StubModel().p.repeat(4, 1), atol=1e-6)


def test_tta_probs_rejects_empty_scales() -> None:
    with pytest.raises(ValueError):
        tta_probs(_StubModel(), torch.randn(1, 3, 8, 8), scales=())


# --------------------------------------------------------------------------- #
# Soup
# --------------------------------------------------------------------------- #
def test_average_state_dicts_uniform_mean() -> None:
    a = nn.Linear(4, 3)
    b = nn.Linear(4, 3)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.weight.add_(1.0)
    tmp_a = "/tmp/_soup_a.pt"
    tmp_b = "/tmp/_soup_b.pt"
    torch.save({"state_dict": a.state_dict()}, tmp_a)
    torch.save({"state_dict": b.state_dict()}, tmp_b)
    avg = average_state_dicts([tmp_a, tmp_b])
    expected = a.state_dict()["weight"] - 0.5  # mean of (w+1, w)
    assert torch.allclose(avg["weight"], expected, atol=1e-6)
    assert torch.allclose(avg["bias"], b.state_dict()["bias"], atol=1e-6)


def test_average_state_dicts_shape_mismatch_raises() -> None:
    a = nn.Linear(4, 3)
    c = nn.Linear(4, 5)
    ta, tc = "/tmp/_soup_m1.pt", "/tmp/_soup_m2.pt"
    torch.save(a.state_dict(), ta)
    torch.save(c.state_dict(), tc)
    with pytest.raises(ValueError):
        average_state_dicts([ta, tc])


def test_make_soup_writes_trainer_compatible_payload() -> None:
    a = nn.Linear(4, 3)
    ta, tb, out = "/tmp/_soup_p1.pt", "/tmp/_soup_p2.pt", "/tmp/_soup_out.pt"
    torch.save(
        {
            "state_dict": a.state_dict(),
            "cfg": {"model": {}},
            "temperature": 0.9,
            "val_qwk": 0.5,
            "epoch": 3,
        },
        ta,
    )
    torch.save(
        {
            "state_dict": a.state_dict(),
            "cfg": {"model": {}},
            "temperature": 0.9,
            "val_qwk": 0.7,
            "epoch": 6,
        },
        tb,
    )
    path = make_soup([ta, tb], out)
    payload = torch.load(out, weights_only=False)
    assert path.name == "_soup_out.pt"
    assert payload["temperature"] == 0.9
    assert payload["val_qwk"] == pytest.approx(0.6)
    assert payload["epoch"] == 6
    assert payload["soup_of"] == [ta, tb]


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
def test_ensemble_weighted_average() -> None:
    m1 = _StubModel((1.0, 0.0, 0.0, 0.0, 0.0))
    m2 = _StubModel((0.0, 0.0, 0.0, 0.0, 1.0))
    ens = EnsemblePredictor([m1, m2], weights=[0.75, 0.25])
    out = ens.predict_probs(torch.randn(2, 3, 8, 8))
    assert torch.allclose(out, torch.tensor([[0.75, 0.0, 0.0, 0.0, 0.25]]).repeat(2, 1))


def test_ensemble_rejects_weight_mismatch() -> None:
    with pytest.raises(ValueError):
        EnsemblePredictor([_StubModel()], weights=[0.5, 0.5])


# --------------------------------------------------------------------------- #
# Threshold search
# --------------------------------------------------------------------------- #
def test_decode_with_cuts_counts_exceeded_boundaries() -> None:
    e = np.array([0.2, 0.8, 1.7, 2.9, 4.0])
    cuts = [3.5, 2.5, 1.5, 0.5]
    assert decode_with_cuts(e, cuts).tolist() == [0, 1, 2, 3, 4]


def test_coordinate_ascent_finds_good_cuts_on_separable_data() -> None:
    rng = np.random.default_rng(7)
    expected = np.concatenate([rng.normal(g, 0.25, 400) for g in range(5)])
    targets = np.concatenate([[g] * 400 for g in range(5)])
    cuts, qwk, acc = coordinate_ascent_cuts(expected, targets, passes=2)
    assert qwk > 0.9
    assert acc > 0.9
    assert len(cuts) == 4
    assert all(cuts[i] >= cuts[i + 1] for i in range(3))


def test_search_referable_threshold_maximizes_accuracy() -> None:
    rng = np.random.default_rng(11)
    p_ref = np.concatenate([rng.beta(2, 9, 700), rng.beta(9, 2, 300)])
    y = np.concatenate([np.zeros(700, dtype=int), np.ones(300, dtype=int)])
    res = search_referable_threshold(y, p_ref)
    assert res["accuracy"] > 0.95
    assert 0.0 <= res["threshold"] <= 1.0
    assert res["sens"] > 0.9 and res["spec"] > 0.9


def test_search_referable_rejects_single_class() -> None:
    with pytest.raises(ValueError):
        search_referable_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_values() -> None:
    lo, hi = wilson_ci(97, 100)
    assert lo == pytest.approx(0.9154, abs=2e-3)
    assert hi == pytest.approx(0.9897, abs=2e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_accuracy_ci_report_lower_bound_gate() -> None:
    rep = accuracy_ci_report(970, 1000, target=0.97)
    assert rep["accuracy"] == pytest.approx(0.97)
    assert rep["ci95_lo"] < 0.97, "n=1000 point estimate alone must not clear the CI gate"
    assert rep["target_reached"] is False
    # A true rate must sit ABOVE the target for the CI lower bound to clear it.
    rep_big = accuracy_ci_report(97600, 100000, target=0.97)
    assert rep_big["target_reached"] is True


def test_mcnemar_symmetric_and_exact() -> None:
    sym = mcnemar_test(10, 10)
    assert sym["p_value"] == pytest.approx(1.0)
    exact = mcnemar_test(2, 9)
    assert exact["method"] == "exact"
    chi = mcnemar_test(40, 10)
    assert chi["method"] == "chi2"
    assert chi["p_value"] < 0.01


def test_bootstrap_accuracy_ci_recovers_true_accuracy() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, 500)
    y_pred = y_true.copy()
    flip = rng.random(500) < 0.05
    y_pred[flip] = (y_pred[flip] + 1) % 5
    lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_boot=300, seed=0)
    true_acc = float((y_true == y_pred).mean())
    assert lo <= true_acc <= hi
    assert hi - lo < 0.08


# --------------------------------------------------------------------------- #
# Pseudo-labeling
# --------------------------------------------------------------------------- #
def test_generate_pseudo_labels_filters_by_confidence(tmp_path) -> None:
    import cv2

    from retinaedge.train.pseudo_label import generate_pseudo_labels

    files = []
    for i in range(6):
        img = (np.random.default_rng(i).integers(0, 255, (32, 32, 3))).astype(np.uint8)
        p = tmp_path / f"img_{i}.png"
        cv2.imwrite(str(p), img)
        files.append(p)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tf = A.Compose([A.Resize(32, 32), A.Normalize(0, 1), ToTensorV2()])

    sure = _StubModel((0.01, 0.02, 0.94, 0.02, 0.01))
    rows, stats = generate_pseudo_labels(sure, files, tf, tau=0.9)
    assert stats["n_total"] == 6
    assert stats["n_kept"] == 6
    assert all(r["grade"] == 2 for r in rows)
    assert rows[0]["confidence"] >= rows[-1]["confidence"]

    strict_rows, strict_stats = generate_pseudo_labels(sure, files, tf, tau=0.99)
    assert strict_stats["n_kept"] == 0 and strict_rows == []

    noisy = _StubModel((0.3, 0.2, 0.2, 0.2, 0.1))
    amb_rows, amb_stats = generate_pseudo_labels(noisy, files, tf, tau=0.25)
    assert amb_stats["n_kept"] < 6  # self-consistency filter rejects ambiguous cases


# --------------------------------------------------------------------------- #
# Improvement ladder
# --------------------------------------------------------------------------- #
def test_build_ladder_budgets_and_monotonic_escalation() -> None:
    from retinaedge.train.auto_improve import build_ladder

    for budget in ("small", "medium", "full"):
        ladder = build_ladder(budget)
        assert [s.name for s in ladder][0] == "s1-baseline"
        assert len(ladder) == 5
    small = build_ladder("small")[1].resolved({"e1": 2, "e2": 4, "sz": 192, "sz2": 224})
    full = build_ladder("full")[1].resolved({"e1": 20, "e2": 40, "sz": 256, "sz2": 320})
    assert "train.epochs=4" in small
    assert "train.epochs=40" in full
    with pytest.raises(ValueError):
        build_ladder("huge")


def test_state_roundtrip(tmp_path) -> None:
    from retinaedge.train.auto_improve import default_state, load_state, save_state

    p = tmp_path / "state.json"
    state = default_state(0.97, "small")
    state["history"].append({"stage": "s1-baseline", "acc_refer": 0.9})
    save_state(p, state)
    loaded = load_state(p)
    assert loaded is not None
    assert loaded["history"][0]["stage"] == "s1-baseline"
    assert loaded["target"] == pytest.approx(0.97)
    assert load_state(tmp_path / "missing.json") is None


def test_pick_best_variant_prefers_referable_accuracy_then_qwk() -> None:
    from retinaedge.train.auto_improve import pick_best_variant

    variants = {
        "base": {
            "acc_refer": 0.95,
            "qwk": 0.80,
            "qwk_cuts": 0.82,
            "n": 100,
            "correct_refer": 95,
            "accuracy": 0.8,
            "accuracy_cuts": 0.8,
            "sens_refer": 0.9,
            "spec_refer": 0.96,
            "refer_threshold": 0.5,
            "cuts": [0.5, 1.5, 2.5, 3.5],
        },
        "tta": {
            "acc_refer": 0.96,
            "qwk": 0.79,
            "qwk_cuts": 0.86,
            "n": 100,
            "correct_refer": 96,
            "accuracy": 0.8,
            "accuracy_cuts": 0.8,
            "sens_refer": 0.91,
            "spec_refer": 0.97,
            "refer_threshold": 0.5,
            "cuts": [0.5, 1.5, 2.5, 3.5],
        },
    }
    name, best = pick_best_variant(variants)
    assert name == "tta"
    assert best["acc_refer"] == pytest.approx(0.96)


def test_variant_metrics_on_separable_probabilities() -> None:
    from retinaedge.train.auto_improve import variant_metrics

    rng = np.random.default_rng(5)
    n_per, k = 200, 5
    targets = np.concatenate([[g] * n_per for g in range(k)])
    probs = np.zeros((n_per * k, k))
    for g in range(k):
        center = rng.normal(g, 0.15, n_per)
        expected = np.clip(center, 0, k - 1)
        lo = np.floor(expected).astype(int)
        hi = np.minimum(lo + 1, k - 1)
        frac = (expected - lo).clip(0, 1)
        rows = np.arange(n_per)
        probs[g * n_per + rows, lo] += (1 - frac).clip(1e-6, 1.0)
        probs[g * n_per + rows, hi] += frac.clip(1e-6, 1.0)
        probs[g * n_per + rows] /= probs[g * n_per + rows].sum(axis=1, keepdims=True)
    m = variant_metrics(probs, targets)
    assert m["qwk"] > 0.9
    assert m["acc_refer"] > 0.9
    assert m["correct_refer"] == round(m["acc_refer"] * m["n"])
    assert len(m["cuts"]) == 4


def test_fmt_log_entry_contains_markdown_table_and_goal(tmp_path) -> None:
    from retinaedge.train.auto_improve import _fmt_log_entry

    variants = {
        "base": {
            "qwk": 0.7,
            "qwk_cuts": 0.75,
            "accuracy": 0.7,
            "accuracy_cuts": 0.72,
            "acc_refer": 0.93,
            "sens_refer": 0.9,
            "spec_refer": 0.94,
            "refer_threshold": 0.5,
            "cuts": [0.5, 1.5, 2.5, 3.5],
            "n": 500,
            "correct_refer": 465,
        },
    }
    entry = _fmt_log_entry(
        "s1-baseline", "baseline", variants, "base", variants["base"], 0.97, 12.0, []
    )
    assert "| variant |" in entry
    assert "s1-baseline" in entry
    assert "Goal 97.00%:" in entry
    assert "REACHED" not in entry or "not reached" in entry


def test_campaign_dry_run_prints_plan(capsys) -> None:
    from retinaedge.train.auto_improve import run_campaign

    state = run_campaign(
        config="configs/train/smoke.yaml",
        budget="small",
        target=0.97,
        max_stages=5,
        state_path="/tmp/_does_not_matter.json",
        dry_run=True,
    )
    out = capsys.readouterr().out
    assert "s1-baseline" in out and "s5-fusion" in out
    assert state["target"] == pytest.approx(0.97)


# --------------------------------------------------------------------------- #
# Data layer: explicit split column + manifest importer
# --------------------------------------------------------------------------- #
def test_file_dataset_honors_split_column(tmp_path) -> None:
    """labels.csv with a split column bypasses hash splits (patient-aware data)."""
    import cv2

    from retinaedge.data.dataset import build_dataset

    root = tmp_path / "blend"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = [
        ("img_a.png", 0, "train"),
        ("img_b.png", 2, "train"),
        ("img_c.png", 3, "val"),
        ("img_d.png", 4, "val"),
        ("img_e.png", 1, "test"),
    ]
    for name, _, _ in rows:
        cv2.imwrite(str(root / "images" / name), rng.integers(0, 255, (24, 24, 3), dtype=np.uint8))
    with open(root / "labels.csv", "w") as f:
        f.write("image,grade,split\n")
        for name, grade, split in rows:
            f.write(f"{name},{grade},{split}\n")

    cfg = {
        "data": {
            "dataset": "aptos",
            "root": str(root),
            "img_size": 24,
            "batch_size": 2,
            "val_fraction": 0.9,
            "test_fraction": 0.0,
        }
    }
    train = build_dataset(cfg, "train")
    val = build_dataset(cfg, "val")
    test = build_dataset(cfg, "test")
    assert len(train) == 2 and len(val) == 2 and len(test) == 1
    assert (
        sorted(
            train.samples[0][0].name,
        )
        is not None
    )
    assert {s[0].name for s in train.samples} == {"img_a.png", "img_b.png"}
    assert {s[0].name for s in val.samples} == {"img_c.png", "img_d.png"}
    assert {s[0].name for s in test.samples} == {"img_e.png"}


def test_file_dataset_falls_back_to_hash_split(tmp_path) -> None:
    """Without a split column the deterministic hash split still applies."""
    import cv2

    from retinaedge.data.dataset import build_dataset

    root = tmp_path / "plain"
    (root / "images").mkdir(parents=True)
    rng = np.random.default_rng(1)
    names = [f"n{i}.png" for i in range(20)]
    for name in names:
        cv2.imwrite(str(root / "images" / name), rng.integers(0, 255, (24, 24, 3), dtype=np.uint8))
    with open(root / "labels.csv", "w") as f:
        f.write("image,grade\n")
        for i, name in enumerate(names):
            f.write(f"{name},{i % 5}\n")
    cfg = {
        "data": {
            "dataset": "aptos",
            "root": str(root),
            "img_size": 24,
            "val_fraction": 0.2,
            "test_fraction": 0.0,
            "split_salt": "t",
        }
    }
    train = build_dataset(cfg, "train")
    val = build_dataset(cfg, "val")
    assert len(train) + len(val) == 20
    assert len(train) > 0 and len(val) > 0


def test_manifest_import_end_to_end(tmp_path) -> None:
    import cv2

    from retinaedge.data.manifest_import import discover_manifests, import_manifest

    raw = tmp_path / "raw" / "ds" / "images"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(2)
    for i in range(5):
        cv2.imwrite(str(raw / f"x{i}.jpg"), rng.integers(0, 255, (24, 24, 3), dtype=np.uint8))
    manifest = tmp_path / "raw" / "ds" / "manifests" / "master.csv"
    manifest.parent.mkdir(parents=True)
    with open(manifest, "w") as f:
        f.write("image_path,grade,split\n")
        for i in range(5):
            split = "train" if i < 3 else "val"
            f.write(f"images/x{i}.jpg,{i % 5},{split}\n")

    found = discover_manifests(tmp_path / "raw")
    assert manifest in found
    out = tmp_path / "out_ds"
    stats = import_manifest(manifest, tmp_path / "raw", out)
    assert stats["n_staged"] == 5 and stats["n_missing"] == 0
    lines = (out / "labels.csv").read_text().strip().splitlines()
    assert lines[0] == "image,grade,split"
    assert len(lines) == 6
    assert (out / "provenance.json").exists()
    assert (out / "images" / "x0.jpg").exists()
