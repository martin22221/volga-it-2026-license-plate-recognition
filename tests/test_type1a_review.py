"""Tests for the full type1a candidate review."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import build_type1a_review as cli
from src.rare_review import (
    TYPE1A_CANDIDATE,
    TYPE1A_COLUMNS,
    TYPE1A_CRITERIA,
    RareCandidate,
    build_page,
    write_type1a_csv,
)
from src.review_sample import build_review_items, discover_pairs
from tests.test_external_audit import write_image, write_label


def make_dataset(root: Path, count: int = 4, *, box: str = "0 0.5 0.5 0.10 0.06") -> Path:
    """A Roboflow-shaped dataset: train/images, train/labels, data.yaml, READMEs."""
    for index in range(count):
        # Distinct sizes so the images are not byte-identical -- a real dataset
        # of duplicates would otherwise be baked into every expectation here.
        write_image(
            root / "train" / "images" / f"img{index}.jpg", 1000 + index * 7, 800 + index * 5
        )
        write_label(root / "train" / "labels" / f"img{index}.txt", [box])
    (root / "data.yaml").write_text(
        "train: ../train/images\nnc: 1\nnames: ['license-plate']\n", encoding="utf-8"
    )
    (root / "README.dataset.txt").write_text(
        "# demo\nProvided by a Roboflow user\nLicense: CC BY 4.0\n", encoding="utf-8"
    )
    (root / "README.roboflow.txt").write_text(
        "demo - v1\nThe dataset includes 4 images.\n", encoding="utf-8"
    )
    return root


def candidates_for(root: Path) -> list[RareCandidate]:
    pairs = discover_pairs(root)
    items = build_review_items(root, sorted(pairs), pairs)
    return [
        RareCandidate(item=item, reasons=list(item.flags), class_label="license-plate")
        for item in items
    ]


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def test_csv_has_the_agreed_columns(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "out" / "review.csv"

    assert write_type1a_csv(candidates_for(root), destination) == 4

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))
    assert tuple(rows[0]) == TYPE1A_COLUMNS
    assert len(rows) == 5


def test_csv_marks_candidates_and_leaves_every_verdict_empty(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "review.csv"
    write_type1a_csv(candidates_for(root), destination)

    rows = list(csv.DictReader(destination.open(encoding="utf-8", newline=""), delimiter=";"))

    for row in rows:
        assert row["candidate_plate_type"] == TYPE1A_CANDIDATE
        assert row["human_plate_type"] == ""
        assert row["russian_plate"] == ""
        assert row["two_line_physical_plate"] == ""
        assert row["visible_face"] == ""
        assert row["review_status"] == "pending"


def test_csv_never_writes_a_confirmed_type(tmp_path: Path) -> None:
    """A dataset named "two-line" must not become a type1a label by itself."""
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "review.csv"
    write_type1a_csv(candidates_for(root), destination)

    text = destination.read_text(encoding="utf-8")
    assert "type1a_candidate" in text
    assert ";type1a;" not in text


def test_csv_records_the_datasets_own_class_name(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "review.csv"
    write_type1a_csv(candidates_for(root), destination)

    rows = list(csv.DictReader(destination.open(encoding="utf-8", newline=""), delimiter=";"))
    assert {row["annotation_class"] for row in rows} == {"license-plate"}


def test_csv_notes_carry_geometry_only(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "review.csv"
    write_type1a_csv(candidates_for(root), destination)

    rows = list(csv.DictReader(destination.open(encoding="utf-8", newline=""), delimiter=";"))
    notes = rows[0]["quality_notes"]

    assert "1000x800" in notes and "aspect" in notes
    assert "type1a" not in notes


def test_csv_delimiter_never_appears_inside_a_field(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext")
    destination = tmp_path / "review.csv"
    write_type1a_csv(candidates_for(root), destination)

    for line in destination.read_text(encoding="utf-8").splitlines():
        assert line.count(";") == len(TYPE1A_COLUMNS) - 1


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------


def test_sheet_shows_filename_class_box_crop_and_verdict(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 3)
    candidates = candidates_for(root)

    page = build_page(
        candidates, root, title="t", intro_html="<p>i</p>", verdict=True
    )

    assert page.count('class="card"') == 3
    assert page.count('class="crop"') == 3
    assert page.count("<i style=") == 3          # one annotation box each
    assert page.count('class="verdict"') == 3
    assert page.count("YOLO class:") == 3
    for candidate in candidates:
        assert Path(candidate.image_path).name in page


def test_sheet_lists_every_type1a_criterion(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 1)
    page = build_page(candidates_for(root), root, title="t", intro_html="", verdict=True)

    for criterion in TYPE1A_CRITERIA:
        assert criterion in page


def test_sheet_has_no_verdict_block_when_not_requested(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 1)
    page = build_page(candidates_for(root), root, title="t", intro_html="")

    assert 'class="verdict"' not in page


def test_sheet_references_images_in_place(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 2)
    page = build_page(candidates_for(root), root, title="t", intro_html="")

    assert (root / "train" / "images" / "img0.jpg").as_uri() in page
    assert "http://" not in page and "https://" not in page


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_builds_all_three_artefacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_dataset(tmp_path / "ext", 5)
    out = tmp_path / "review"

    assert cli.main([str(root), "--source-id", "demo", "--out", str(out)]) == cli.EXIT_OK

    assert (out / "contact_sheet.html").is_file()
    assert (out / "review.csv").is_file()
    assert (out / "statistics.txt").is_file()

    output = capsys.readouterr().out
    assert "type1a candidates : 5" in output
    assert "HUMAN-CONFIRMED type1a: 0" in output


def test_cli_covers_every_image_without_sampling(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 9)
    out = tmp_path / "review"
    cli.main([str(root), "--source-id", "demo", "--out", str(out)])

    rows = list(
        csv.DictReader((out / "review.csv").open(encoding="utf-8", newline=""), delimiter=";")
    )
    assert len(rows) == 9


def test_statistics_report_states_nothing_is_confirmed(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 3)
    out = tmp_path / "review"
    cli.main([str(root), "--source-id", "demo", "--out", str(out)])

    text = (out / "statistics.txt").read_text(encoding="utf-8")

    assert "HUMAN-CONFIRMED type1a: 0" in text
    assert "duplicate images (SHA-256): 0 group(s)" in text
    assert "license-plate" in text
    assert "consistency is not confirmation" in text


def test_cli_does_not_modify_the_dataset(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 4)
    before = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    cli.main([str(root), "--source-id", "demo", "--out", str(tmp_path / "review")])

    after = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    assert before == after


def test_cli_refuses_to_write_into_our_dataset_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_dataset(tmp_path / "ext", 2)
    target = cli.REPO_ROOT / "dataset" / "review_out"

    assert cli.main([str(root), "--source-id", "d", "--out", str(target)]) == cli.EXIT_FAILED
    assert "competition dataset" in capsys.readouterr().err
    assert not target.exists()


def test_cli_refuses_to_write_into_the_audited_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_dataset(tmp_path / "ext", 2)

    assert cli.main(
        [str(root), "--source-id", "d", "--out", str(root / "out")]
    ) == cli.EXIT_FAILED
    assert "refusing to write" in capsys.readouterr().err


def test_cli_rejects_a_directory_without_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    assert cli.main(
        [str(root), "--source-id", "d", "--out", str(tmp_path / "o")]
    ) == cli.EXIT_FAILED
    assert "no images found" in capsys.readouterr().err
