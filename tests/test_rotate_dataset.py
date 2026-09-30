"""Tests for scripts/rotate_dataset.py (size-guarded catalog rotation)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rotate_dataset.py"
_spec = importlib.util.spec_from_file_location("rotate_dataset", SCRIPT)
rotate_dataset = importlib.util.module_from_spec(_spec)
sys.modules["rotate_dataset"] = rotate_dataset
_spec.loader.exec_module(rotate_dataset)

CATALOG = """# catalog fixture
datasets:
  - handle: a/huge
    bytes: 22000000000
  - handle: b/medium
    bytes: 8600000000
  - handle: c/small
    bytes: 447582906
  - handle: d/tiny
    bytes: 399000000
  - handle: e/nolabel
"""


class TestLoadEntries:
    def test_pairs_parsed_in_order(self, tmp_path):
        p = tmp_path / "catalog.yaml"
        p.write_text(CATALOG)
        entries = rotate_dataset.load_entries(p)
        assert entries == [
            ("a/huge", 22_000_000_000),
            ("b/medium", 8_600_000_000),
            ("c/small", 447_582_906),
            ("d/tiny", 399_000_000),
        ]


class TestLoadHandles:
    def test_size_guard_skips_giants(self, tmp_path):
        p = tmp_path / "catalog.yaml"
        p.write_text(CATALOG)
        handles = rotate_dataset.load_handles(p, top=4, max_bytes=9_000_000_000)
        assert handles == ["b/medium", "c/small", "d/tiny"]

    def test_all_too_large_returns_empty(self, tmp_path):
        p = tmp_path / "catalog.yaml"
        p.write_text(CATALOG)
        assert rotate_dataset.load_handles(p, top=4, max_bytes=1) == []


class TestPickIndex:
    def test_history_length_rotates(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text(json.dumps({"history": [1, 2, 3, 4, 5]}))
        assert rotate_dataset.pick_index(p, 3) == 5 % 3

    def test_unreadable_state_is_zero(self, tmp_path):
        p = tmp_path / "missing.json"
        assert rotate_dataset.pick_index(p, 3) == 0


def test_main_prints_handle(tmp_path, capsys):
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(CATALOG)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"history": []}))
    rc = rotate_dataset.main(["--catalog", str(catalog), "--state", str(state), "--top", "4"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "b/medium"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
