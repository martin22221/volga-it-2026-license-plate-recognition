"""Tests for the run.py command line interface."""

from __future__ import annotations

from pathlib import Path

import run


def test_cli_creates_csv_for_empty_result(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"")
    output = tmp_path / "out.csv"

    assert run.main(["--input", str(images), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == "image;plate_num;plate_type;confidence\n"


def test_cli_reports_missing_input_directory(tmp_path: Path) -> None:
    assert run.main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o.csv")]) == 2
