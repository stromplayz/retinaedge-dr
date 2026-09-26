"""Network-free tests for the HF dataset resolver's pure logic."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from retinaedge.data.resolve_hf import (
    UnmappableGrades,
    _pick_columns,
    _row_image,
    _write_layout,
    map_labels_to_icdrss,
)


class TestMapLabels:
    def test_icdrss5_passthrough(self) -> None:
        grades, scheme = map_labels_to_icdrss([0, 1, 2, 3, 4, 0])
        assert grades == [0, 1, 2, 3, 4, 0]
        assert scheme == "icdrss5"

    def test_floats_round_to_int(self) -> None:
        grades, scheme = map_labels_to_icdrss([0.0, 2.0, 4.0])
        assert grades == [0, 2, 4]
        assert scheme == "icdrss5"

    def test_binary_referable_expanded(self) -> None:
        grades, scheme = map_labels_to_icdrss([0, 1, 1, 0])
        assert grades == [0, 2, 2, 0]
        assert scheme == "binary_expanded"

    def test_names_mapping(self) -> None:
        raw = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        grades, scheme = map_labels_to_icdrss(raw)
        assert grades == [0, 1, 2, 3, 4]
        assert scheme == "names"

    def test_names_case_whitespace(self) -> None:
        grades, scheme = map_labels_to_icdrss(["  NO DR ", "Proliferative"])
        assert grades == [0, 4]
        assert scheme == "names"

    def test_unmappable_raises(self) -> None:
        with pytest.raises(UnmappableGrades):
            map_labels_to_icdrss([0, 1, 7])
        with pytest.raises(UnmappableGrades):
            map_labels_to_icdrss(["glaucoma", "cataract"])
        with pytest.raises(UnmappableGrades):
            map_labels_to_icdrss([])


class TestColumnPicking:
    def test_prefers_known_grade_names(self) -> None:
        row = {"image": {"bytes": b"x"}, "diagnosis": 2, "zzz": 1}
        image_col, label_col = _pick_columns({}, row)
        assert image_col == "image"
        assert label_col == "diagnosis"

    def test_fallback_numeric_column(self) -> None:
        row = {"img": {"bytes": b"x"}, "some_col": 3}
        _, label_col = _pick_columns({}, row)
        assert label_col in ("img", "some_col")


class TestRowImage:
    def test_decodes_bytes_dict(self) -> None:
        buf = Image.new("RGB", (8, 8), (200, 30, 30))
        import io

        bio = io.BytesIO()
        buf.save(bio, "JPEG")
        pil = _row_image({"image": {"bytes": bio.getvalue()}}, "image")
        assert pil is not None and pil.size == (8, 8)

    def test_missing_returns_none(self) -> None:
        assert _row_image({"image": None}, "image") is None


class TestWriteLayout:
    def test_writes_images_csv_and_counts(self, tmp_path) -> None:
        rows = []
        for i in range(12):
            img = Image.fromarray(np.full((16, 16, 3), (i * 20 % 255, 40, 60), dtype=np.uint8))
            import io

            bio = io.BytesIO()
            img.save(bio, "JPEG")
            rows.append({"image": {"bytes": bio.getvalue()}, "_grade": i % 5})

        n, counter = _write_layout(
            iter(rows), "image", "_grade", None, tmp_path, max_images=10, bake_ben_graham=False
        )
        assert n == 10
        assert sum(counter.values()) == 10
        assert (tmp_path / "labels.csv").exists()
        assert len(list((tmp_path / "images").glob("*.jpg"))) == 10
        first = (tmp_path / "labels.csv").read_text().splitlines()[1]
        assert first == "000001.jpg,0"

    def test_respects_max_images(self, tmp_path) -> None:
        import io

        rows = []
        for _ in range(5):
            bio = io.BytesIO()
            Image.new("RGB", (8, 8), (10, 10, 10)).save(bio, "JPEG")
            rows.append({"image": {"bytes": bio.getvalue()}, "_grade": 0})
        n, _ = _write_layout(iter(rows), "image", "_grade", None, tmp_path, 3, False)
        assert n == 3
