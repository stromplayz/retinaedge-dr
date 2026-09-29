"""Tests for the remote campaign engine's pure functions (auto_improve)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from retinaedge.train.auto_improve import (
    Stage,
    build_ladder,
    load_state,
    pick_best_variant,
    save_state,
    variant_metrics,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TestLadder:
    def test_budgets_build(self):
        for budget in ("small", "medium", "full"):
            ladder = build_ladder(budget)
            assert ladder, budget
            assert ladder[0].name == "s1-baseline"

    def test_unknown_budget_rejected(self):
        with pytest.raises(ValueError):
            build_ladder("galactic")

    def test_stage_resolves_placeholders(self):
        stage = Stage("t", "d", ("train.epochs={e1}", "data.img_size={sz}"))
        resolved = stage.resolved({"e1": 3, "sz": 192})
        assert resolved == ("train.epochs=3", "data.img_size=192")


class TestState:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "state" / "improve_state.json"
        assert load_state(p) is None  # absent -> fresh start
        save_state(p, {"history": [1], "goal_reached": False})
        state = load_state(p)
        assert state is not None
        assert state["history"] == [1]
        assert "updated_at" in state

    def test_corrupt_state_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json{{{")
        assert load_state(p) is None

    def test_missing_history_returns_none(self, tmp_path):
        p = tmp_path / "nohist.json"
        p.write_text(json.dumps({"target": 0.97}))
        assert load_state(p) is None


class TestVariantMetrics:
    def test_structure(self):
        rng = np.random.default_rng(5)
        targets = rng.integers(0, 5, 200)
        logits = rng.normal(size=(200, 5))
        logits[np.arange(200), targets] += 1.8
        probs = _softmax(logits)
        m = variant_metrics(probs, targets)
        assert set(m) >= {
            "qwk",
            "qwk_cuts",
            "accuracy",
            "accuracy_cuts",
            "acc_refer",
            "sens_refer",
            "spec_refer",
            "n",
        }
        assert 0.0 <= m["acc_refer"] <= 1.0
        assert m["n"] == 200

    def test_pick_best_prefers_referable_accuracy(self):
        a = {"acc_refer": 0.80, "qwk": 0.90, "qwk_cuts": 0.90}
        b = {"acc_refer": 0.85, "qwk": 0.60, "qwk_cuts": 0.60}
        name, best = pick_best_variant({"a": a, "b": b})
        assert name == "b"
        assert best is b


if __name__ == "__main__":
    pytest.main([__file__])


class TestLadderV3:
    """v0.3.0: s6-mixup + s7-distill rungs."""

    def test_ladder_ends_with_v3_stages(self):
        ladder = build_ladder("small")
        names = [s.name for s in ladder]
        assert names[-2:] == ["s6-mixup", "s7-distill"]

    def test_distill_runner_flag(self):
        ladder = build_ladder("medium")
        assert all(s.runner == "trainer" for s in ladder[:-1])
        assert ladder[-1].runner == "distill"

    def test_v3_stages_resolve_placeholders(self):
        ladder = build_ladder("full")
        s6, s7 = ladder[-2], ladder[-1]
        r6 = s6.resolved({"e2": 40, "sz2": 320})
        r7 = s7.resolved({"e2": 40, "sz2": 320})
        assert "train.mixup_alpha=0.2" in r6
        assert "train.loss.label_smoothing=0.05" in r6
        assert "data.img_size=320" in r7
        assert "model.backbone=efficientnet_lite0" in r7

    def test_every_stage_keeps_prior_tricks(self):
        """Each rung must keep the escalation tricks (EMA, backbone, resolution)."""
        ladder = build_ladder("medium")
        for stage in ladder[1:]:
            resolved = stage.resolved({"e1": 6, "e2": 14, "sz": 224, "sz2": 288})
            assert "train.ema=true" in resolved, stage.name
