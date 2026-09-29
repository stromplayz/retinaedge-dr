"""Tests for scripts/bump_release.py (cadence release decision, pure stdlib)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bump_release.py"
_spec = importlib.util.spec_from_file_location("bump_release", SCRIPT)
bump_release = importlib.util.module_from_spec(_spec)
sys.modules["bump_release"] = bump_release
_spec.loader.exec_module(bump_release)

decide_version = bump_release.decide_version
parse_semver = bump_release.parse_semver


class TestParseSemver:
    def test_plain(self):
        assert parse_semver("0.3.0") == (0, 3, 0)

    def test_v_prefix_and_suffix(self):
        assert parse_semver("v1.0.0") == (1, 0, 0)
        assert parse_semver("v0.2.0-mobile") == (0, 2, 0)

    def test_garbage(self):
        assert parse_semver("banana") is None


class TestDecideVersion:
    def test_goal_reaches_1(self):
        assert decide_version("0.3.0", 0.975, 0.89, True) == "1.0.0"

    def test_goal_ignored_if_already_1(self):
        assert decide_version("1.0.0", 0.975, 0.974, True) is None

    def test_minor_on_improvement(self):
        assert decide_version("0.3.0", 0.90, 0.8904, False) == "0.4.0"

    def test_no_bump_below_delta(self):
        assert decide_version("0.3.0", 0.8910, 0.8904, False) is None

    def test_no_regression_release(self):
        assert decide_version("0.4.0", 0.80, 0.90, False) is None

    def test_patch_after_goal(self):
        assert decide_version("1.0.0", 0.98, 0.975, False) == "1.0.1"

    def test_exact_delta_bumps(self):
        assert decide_version("0.3.0", 0.8954, 0.8904, False) == "0.4.0"


class TestEndToEnd:
    def test_main_releases_and_updates_ledger(self, tmp_path):
        campaign = {
            "target": 0.97,
            "budget": "small",
            "round": 2,
            "goal_reached": False,
            "best": {
                "stage": "s5-fusion",
                "variant": "soup+tta",
                "acc_refer": 0.9100,
                "qwk": 0.86,
                "ci95_lo": 0.895,
                "ci95_hi": 0.924,
                "sens_refer": 0.87,
                "spec_refer": 0.93,
                "n": 1843,
                "ckpt": "runs/s5-fusion/best.pt",
            },
            "history": [{"stage": "s1"}],
        }
        state_p = tmp_path / "improve_state.json"
        state_p.write_text(json.dumps(campaign))
        ledger_p = tmp_path / "version_state.json"
        ledger_p.write_text(json.dumps({"last_version": "0.3.0", "last_acc": 0.8904}))
        notes_p = tmp_path / "release_notes.md"

        rc = bump_release.main(
            [
                "--state",
                str(state_p),
                "--version-state",
                str(ledger_p),
                "--out",
                str(notes_p),
            ]
        )
        assert rc == 0
        ledger = json.loads(ledger_p.read_text())
        assert ledger["last_version"] == "0.4.0"
        assert ledger["last_acc"] == 0.9100
        notes = notes_p.read_text()
        assert "0.4.0" in notes and "soup+tta" in notes

    def test_main_no_release_keeps_ledger(self, tmp_path):
        campaign = {
            "goal_reached": False,
            "best": {"acc_refer": 0.8910, "stage": "s1", "variant": "base", "n": 100},
        }
        state_p = tmp_path / "improve_state.json"
        state_p.write_text(json.dumps(campaign))
        ledger_p = tmp_path / "version_state.json"
        ledger_p.write_text(json.dumps({"last_version": "0.3.0", "last_acc": 0.8904}))

        rc = bump_release.main(
            [
                "--state",
                str(state_p),
                "--version-state",
                str(ledger_p),
                "--out",
                str(tmp_path / "notes.md"),
            ]
        )
        assert rc == 0
        ledger = json.loads(ledger_p.read_text())
        assert ledger["last_version"] == "0.3.0"

    def test_main_without_champion_is_noop(self, tmp_path):
        state_p = tmp_path / "improve_state.json"
        state_p.write_text(json.dumps({"goal_reached": False, "best": None}))
        rc = bump_release.main(
            [
                "--state",
                str(state_p),
                "--version-state",
                str(tmp_path / "v.json"),
                "--out",
                str(tmp_path / "notes.md"),
            ]
        )
        assert rc == 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
