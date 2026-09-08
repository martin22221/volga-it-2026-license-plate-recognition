"""Tests for submission CSV generation."""

from __future__ import annotations

from pathlib import Path

from src.classifier import PlateType
from src.csv_writer import PlateRecord, write_csv


def test_writes_header_and_rows(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    rows = write_csv(
        [
            PlateRecord("a.jpg", "A123BC77", PlateType.TYPE1, 0.9123),
            PlateRecord("b.png", "M111MM102", PlateType.TYPE1B, 0.5),
        ],
        output,
    )

    assert rows == 2
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "image;plate_num;plate_type;confidence",
        "a.jpg;A123BC77;type1;0.912",
        "b.png;M111MM102;type1b;0.500",
    ]


def test_empty_result_writes_header_only(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    assert write_csv([], output) == 0
    assert output.read_text(encoding="utf-8") == "image;plate_num;plate_type;confidence\n"


def test_header_can_be_omitted(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    write_csv([PlateRecord("a.jpg", "A123BC77", PlateType.TYPE1A, 1.0)], output, write_header=False)
    assert output.read_text(encoding="utf-8") == "a.jpg;A123BC77;type1a;1.000\n"


def test_confidence_is_clamped_to_unit_range() -> None:
    assert PlateRecord("a.jpg", "A123BC77", PlateType.OTHER, 1.7).as_row()[3] == "1.000"
    assert PlateRecord("a.jpg", "A123BC77", PlateType.OTHER, -0.4).as_row()[3] == "0.000"


def test_parent_directory_is_created(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dir" / "out.csv"
    write_csv([], output)
    assert output.exists()


def test_file_is_utf8_encoded(tmp_path: Path) -> None:
    output = tmp_path / "out.csv"
    write_csv([PlateRecord("\u043c\u0430\u0448\u0438\u043d\u0430.jpg", "A123BC77", PlateType.TYPE1, 0.5)], output)
    assert "\u043c\u0430\u0448\u0438\u043d\u0430.jpg" in output.read_text(encoding="utf-8")
