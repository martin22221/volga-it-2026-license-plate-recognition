"""Tests for the participant-side dataset validator CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import validate_dataset_local as cli
from src.dataset_meta import BBOX_COLUMNS, QUAD_COLUMNS, validate_meta
from tests.test_dataset_meta import make_dataset, row


def background_row(**overrides: str) -> dict[str, str]:
    zeros = {column: "0" for column in BBOX_COLUMNS + QUAD_COLUMNS}
    return row(plate_type="other", plate_num="", is_vehicle="false", **zeros, **overrides)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A small but complete dataset: real, synthetic and background rows."""
    rows = [
        row(),
        row(
            image="images/real/000002.jpg",
            plate_num="M111MM102",
            plate_type="type1b",
            conditions="night|glare",
        ),
        row(
            image="images/synthetic/000003.png",
            plate_num="X001XX01",
            is_synthetic="true",
            source="",
            license="",
            conditions="day",
        ),
        background_row(image="images/real/000004.jpg", conditions="rain"),
    ]
    make_dataset(tmp_path, rows)
    return tmp_path


def test_valid_dataset_exits_zero(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--dataset", str(dataset)]) == cli.EXIT_OK

    output = capsys.readouterr().out
    assert "VALID - 0 error(s), 0 warning(s)" in output


def test_report_contains_every_required_section(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--dataset", str(dataset)])
    output = capsys.readouterr().out

    for section in (
        "Totals",
        "Annotations by plate_type",
        "Annotations by condition",
        "Source / license coverage",
        "Missing image files",
        "Errors",
        "Warnings",
        "Verdict",
    ):
        assert section in output, section


def test_report_counts_are_correct(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["--dataset", str(dataset)])
    output = capsys.readouterr().out

    assert "total images      : 4" in output
    assert "total annotations : 4" in output
    assert "real images       : 3" in output
    assert "synthetic images  : 1" in output
    assert "background rows   : 1" in output


def test_report_lists_plate_types_and_conditions(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--dataset", str(dataset)])
    output = capsys.readouterr().out

    assert "type1a" in output
    assert "type1b" in output
    assert "night" in output
    assert "never used: snow, dirt, motion_blur" in output


def test_report_shows_source_and_license_coverage(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["--dataset", str(dataset)])
    output = capsys.readouterr().out

    assert "real annotations with a source  : 3/3" in output
    assert "real annotations with a license : 3/3" in output
    assert "CC BY 4.0" in output


def test_invalid_dataset_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_dataset(tmp_path, [row(plate_type="type9")])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_INVALID
    assert "INVALID - 1 error(s)" in capsys.readouterr().out


def test_missing_files_are_listed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_dataset(tmp_path, [row()], create_images=False)

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_INVALID
    output = capsys.readouterr().out
    assert "Missing image files (1)" in output
    assert "images/real/000001.jpg" in output


def test_blank_provenance_warns_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_dataset(tmp_path, [row(source="", license="")])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_OK
    output = capsys.readouterr().out
    assert "source is blank for a real image" in output
    assert "license is blank for a real image" in output


def test_real_image_flagged_synthetic_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_dataset(tmp_path, [row(is_synthetic="true")])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_OK
    assert "is_synthetic=true" in capsys.readouterr().out


def test_synthetic_image_flagged_real_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_dataset(tmp_path, [row(image="images/synthetic/a.png", is_synthetic="false")])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_OK
    assert "is_synthetic=false" in capsys.readouterr().out


def test_duplicate_annotation_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_dataset(tmp_path, [row(), row()])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_INVALID
    assert "exact duplicate" in capsys.readouterr().out


def test_strict_mode_fails_on_warnings(tmp_path: Path) -> None:
    make_dataset(tmp_path, [row(source="")])

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_OK
    assert cli.main(["--dataset", str(tmp_path), "--strict"]) == cli.EXIT_INVALID


def test_skip_file_check(tmp_path: Path) -> None:
    make_dataset(tmp_path, [row()], create_images=False)

    assert cli.main(["--dataset", str(tmp_path)]) == cli.EXIT_INVALID
    assert cli.main(["--dataset", str(tmp_path), "--skip-file-check"]) == cli.EXIT_OK


def test_quiet_mode_prints_only_the_verdict(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--dataset", str(dataset), "--quiet"]) == cli.EXIT_OK

    output = capsys.readouterr().out.strip()
    assert output == "VALID - 0 error(s), 0 warning(s)"


def test_report_can_be_written_to_a_file(dataset: Path, tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "dataset.txt"

    assert cli.main(["--dataset", str(dataset), "--report", str(destination)]) == cli.EXIT_OK
    assert "Verdict" in destination.read_text(encoding="utf-8")


def test_explicit_meta_path_is_honoured(dataset: Path, tmp_path: Path) -> None:
    moved = tmp_path / "elsewhere.csv"
    moved.write_text((dataset / "meta.csv").read_text(encoding="utf-8"), encoding="utf-8")

    assert cli.main(["--dataset", str(dataset), "--meta", str(moved)]) == cli.EXIT_OK


def test_absent_meta_csv_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--dataset", str(tmp_path / "nope")]) == cli.EXIT_INVALID
    assert "does not exist" in capsys.readouterr().err


def test_committed_dataset_meta_validates() -> None:
    """The dataset checked into the repository must pass as-is.

    Only ``meta.csv`` and the folder structure are committed; the images are
    rebuilt (see ``dataset/README.md``), so the schema is validated without the
    file-existence check, and the files are checked only when they are present.
    """
    root = Path(__file__).resolve().parents[1] / "dataset"
    report = validate_meta(root / "meta.csv", root, check_files=False)

    assert report.is_valid
    assert "VALID" in cli.build_report_text(report)
    stats = report.stats
    # The promoted synthetic set (generator 2.3.0, seed 2026091401); no real images yet.
    assert stats.total_rows == stats.synthetic_rows == 12_000
    assert stats.real_rows == 0
    assert stats.by_plate_type == {"type1": 2_400, "type1a": 4_200, "type1b": 5_400}
    assert stats.by_source == {"volga_synthetic_generator": 12_000}
    assert stats.by_license == {"CC BY 4.0": 12_000}

    if any((root / "images" / "synthetic").glob("*.jpg")):
        present = validate_meta(root / "meta.csv", root)
        assert present.is_valid and not present.stats.missing_files
