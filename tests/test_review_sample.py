"""Tests for the reproducible review-sample builder."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import src.review_sample as rs
from scripts import sample_review_set as cli
from src.external_audit import YoloBox
from src.review_sample import (
    FLAG_EXTENSION_MISMATCH,
    FLAG_MULTIPLE_PLATES,
    FLAG_NO_ANNOTATION,
    FLAG_ONE_LINE_SHAPE,
    FLAG_SMALL_PLATE,
    FLAG_SQUARE_CANDIDATE,
    FLAG_TINY_PLATE,
    FLAG_TOUCHES_EDGE,
    NEEDS_HUMAN_REVIEW,
    REVIEW_COLUMNS,
    build_contact_sheet,
    build_review_item,
    build_review_items,
    discover_pairs,
    flag_counts,
    select_sample,
    suggest_plate_type,
    write_contact_sheet,
    write_review_csv,
)
from tests.test_external_audit import jpeg_bytes, write_image, write_label


def make_dataset(root: Path, count: int = 10, *, box: str = "0 0.5 0.5 0.2 0.04") -> Path:
    """A small YOLO dataset with images/ and labels/ side by side."""
    for index in range(count):
        write_image(root / "images" / f"{index}.jpg", 1000, 800)
        write_label(root / "labels" / f"{index}.txt", [box])
    return root


# --------------------------------------------------------------------------
# Deterministic sampling
# --------------------------------------------------------------------------


def test_sample_is_reproducible() -> None:
    population = [f"img{i}" for i in range(500)]
    assert select_sample(population, 20) == select_sample(population, 20)


def test_sample_does_not_depend_on_input_order() -> None:
    population = [f"img{i}" for i in range(500)]
    assert select_sample(population, 20) == select_sample(list(reversed(population)), 20)


def test_sample_honours_size_and_seed() -> None:
    population = [f"img{i}" for i in range(500)]

    assert len(select_sample(population, 20)) == 20
    assert select_sample(population, 20, seed=1) != select_sample(population, 20, seed=2)


def test_sample_smaller_than_requested_returns_everything() -> None:
    population = ["a", "b", "c"]
    assert select_sample(population, 10) == ["a", "b", "c"]


def test_sample_is_sorted_and_unique() -> None:
    chosen = select_sample([f"img{i}" for i in range(500)], 30)
    assert chosen == sorted(chosen)
    assert len(set(chosen)) == 30


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("norm_w", "norm_h", "expected"),
    [
        (0.20, 0.04, FLAG_ONE_LINE_SHAPE),      # aspect 5.0
        (0.30, 0.05, FLAG_ONE_LINE_SHAPE),      # aspect 4.8
        (0.10, 0.06, FLAG_SQUARE_CANDIDATE),    # aspect 2.08
        (0.08, 0.06, FLAG_SQUARE_CANDIDATE),    # aspect 1.67
    ],
)
def test_shape_flags_follow_the_aspect_ratio(
    norm_w: float, norm_h: float, expected: str
) -> None:
    geometry = rs._geometry(YoloBox(0, [0.5, 0.5, norm_w, norm_h]), 1000, 800)
    assert geometry.shape_flag() == expected


def test_intermediate_aspect_is_ambiguous_not_guessed() -> None:
    geometry = rs._geometry(YoloBox(0, [0.5, 0.5, 0.14, 0.06]), 1000, 800)
    assert geometry.shape_flag() == rs.FLAG_AMBIGUOUS_SHAPE


def test_geometry_converts_to_pixels() -> None:
    geometry = rs._geometry(YoloBox(0, [0.5, 0.5, 0.2, 0.05]), 1000, 800)

    assert geometry.width_px == 200
    assert geometry.height_px == 40
    assert geometry.aspect == pytest.approx(5.0)
    assert geometry.area_fraction == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("centre_x", "centre_y", "touches"),
    [(0.5, 0.5, False), (0.05, 0.5, True), (0.5, 0.02, True), (0.97, 0.5, True)],
)
def test_edge_detection(centre_x: float, centre_y: float, touches: bool) -> None:
    geometry = rs._geometry(YoloBox(0, [centre_x, centre_y, 0.1, 0.05]), 1000, 800)
    assert geometry.touches_edge() is touches


# --------------------------------------------------------------------------
# Review items and flags
# --------------------------------------------------------------------------


def test_item_carries_size_shape_and_no_type_claim(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 1)
    pairs = discover_pairs(root)
    item = build_review_item(*pairs["0"], root)

    assert item.image_path == "images/0.jpg"
    assert (item.width, item.height) == (1000, 800)
    assert len(item.plates) == 1
    assert FLAG_ONE_LINE_SHAPE in item.flags
    assert suggest_plate_type(item)[0] == NEEDS_HUMAN_REVIEW


def test_small_and_tiny_plates_are_flagged(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "small.jpg", 1000, 800)
    write_label(root / "labels" / "small.txt", ["0 0.5 0.5 0.05 0.01"])  # 50px wide
    write_image(root / "images" / "tiny.jpg", 1000, 800)
    write_label(root / "labels" / "tiny.txt", ["0 0.5 0.5 0.02 0.004"])  # 20px wide

    pairs = discover_pairs(root)
    small = build_review_item(*pairs["small"], root)
    tiny = build_review_item(*pairs["tiny"], root)

    assert FLAG_SMALL_PLATE in small.flags and FLAG_TINY_PLATE not in small.flags
    assert FLAG_TINY_PLATE in tiny.flags and FLAG_SMALL_PLATE not in tiny.flags


def test_multiple_plates_are_flagged(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "two.jpg", 1000, 800)
    write_label(
        root / "labels" / "two.txt",
        ["0 0.3 0.5 0.2 0.04", "0 0.7 0.5 0.1 0.06"],
    )

    item = build_review_item(*discover_pairs(root)["two"], root)

    assert FLAG_MULTIPLE_PLATES in item.flags
    assert len(item.plates) == 2
    # The shape flag describes the largest plate.
    assert FLAG_ONE_LINE_SHAPE in item.flags


def test_edge_plate_is_flagged(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "edge.jpg", 1000, 800)
    write_label(root / "labels" / "edge.txt", ["0 0.06 0.5 0.12 0.04"])

    item = build_review_item(*discover_pairs(root)["edge"], root)

    assert FLAG_TOUCHES_EDGE in item.flags


def test_missing_label_is_flagged(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "lonely.jpg", 1000, 800)

    item = build_review_item(*discover_pairs(root)["lonely"], root)

    assert FLAG_NO_ANNOTATION in item.flags
    assert item.plates == []
    assert suggest_plate_type(item) == (NEEDS_HUMAN_REVIEW, "no annotated plate to judge")


def test_extension_mismatch_is_carried_into_the_review(tmp_path: Path) -> None:
    """The AUTO.RIA case: a .bmp holding JPEG data is still measurable."""
    root = tmp_path / "d"
    (root / "images").mkdir(parents=True)
    (root / "images" / "x.bmp").write_bytes(jpeg_bytes(1000, 800))
    write_label(root / "labels" / "x.txt", ["0 0.5 0.5 0.2 0.04"])

    item = build_review_item(*discover_pairs(root)["x"], root)

    assert FLAG_EXTENSION_MISMATCH in item.flags
    assert item.image_format == "jpeg"
    assert (item.width, item.height) == (1000, 800)
    assert len(item.plates) == 1


def test_unreadable_image_abstains(tmp_path: Path) -> None:
    root = tmp_path / "d"
    (root / "images").mkdir(parents=True)
    (root / "images" / "broken.jpg").write_bytes(b"not an image")
    write_label(root / "labels" / "broken.txt", ["0 0.5 0.5 0.2 0.04"])

    item = build_review_item(*discover_pairs(root)["broken"], root)

    assert rs.FLAG_UNREADABLE in item.flags
    assert suggest_plate_type(item) == (NEEDS_HUMAN_REVIEW, "image header could not be read")


def test_flag_counts(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 5)
    items = build_review_items(root, select_sample(discover_pairs(root), 5), discover_pairs(root))

    assert flag_counts(items)[FLAG_ONE_LINE_SHAPE] == 5


# --------------------------------------------------------------------------
# The tool never claims a plate type
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "box",
    [
        "0 0.5 0.5 0.2 0.04",   # clean one-line shape
        "0 0.5 0.5 0.08 0.06",  # clean square shape
        "0 0.5 0.5 0.14 0.06",  # ambiguous
    ],
)
def test_plate_type_is_never_asserted_from_geometry(tmp_path: Path, box: str) -> None:
    """Colour separates the types, and we cannot read colour. So we abstain."""
    root = make_dataset(tmp_path, 1, box=box)
    item = build_review_item(*discover_pairs(root)["0"], root)

    suggestion, reason = suggest_plate_type(item)

    assert suggestion == NEEDS_HUMAN_REVIEW
    assert "colour" in reason


def test_square_candidate_reason_does_not_claim_type1a(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 1, box="0 0.5 0.5 0.08 0.06")
    item = build_review_item(*discover_pairs(root)["0"], root)

    _, reason = suggest_plate_type(item)

    assert FLAG_SQUARE_CANDIDATE in reason
    assert "type1a" not in reason


# --------------------------------------------------------------------------
# Review CSV
# --------------------------------------------------------------------------


def test_csv_has_the_agreed_columns(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 3)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 3), pairs)
    destination = tmp_path / "out" / "review.csv"

    assert write_review_csv(items, destination) == 3

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter=";"))

    assert tuple(rows[0]) == REVIEW_COLUMNS
    assert len(rows) == 4


def test_csv_leaves_the_human_columns_empty(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 3)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 3), pairs)
    destination = tmp_path / "review.csv"
    write_review_csv(items, destination)

    with destination.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))

    for row in rows:
        assert row["human_plate_type"] == ""
        assert row["has_visible_face"] == ""
        assert row["suggested_plate_type"] == NEEDS_HUMAN_REVIEW
        assert row["review_status"] == "pending"


def test_csv_quality_notes_carry_geometry_not_a_verdict(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 1)
    pairs = discover_pairs(root)
    items = build_review_items(root, ["0"], pairs)
    destination = tmp_path / "review.csv"
    write_review_csv(items, destination)

    notes = list(csv.DictReader(destination.open(encoding="utf-8", newline=""), delimiter=";"))[0][
        "quality_notes"
    ]

    assert "one_line_shape" in notes
    assert "aspect" in notes
    assert "1000x800" in notes
    for plate_type in ("type1a", "type1b"):
        assert plate_type not in notes


def test_csv_delimiter_does_not_appear_inside_quality_notes(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 4)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 4), pairs)
    destination = tmp_path / "review.csv"
    write_review_csv(items, destination)

    for line in destination.read_text(encoding="utf-8").splitlines():
        assert line.count(";") == len(REVIEW_COLUMNS) - 1


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------


def test_contact_sheet_labels_every_image_with_its_filename(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 5)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 5), pairs)

    page = build_contact_sheet(items, root, source_id="demo")

    for item in items:
        assert Path(item.image_path).name in page
    assert page.count('class="card"') == 5


def test_contact_sheet_references_images_in_place(tmp_path: Path) -> None:
    """Nothing is copied: the sheet points at the originals by file:// URL."""
    root = make_dataset(tmp_path, 2)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 2), pairs)

    page = build_contact_sheet(items, root, source_id="demo")

    assert (root / "images" / "0.jpg").as_uri() in page
    assert "file:///" in page


def test_contact_sheet_states_that_types_are_not_filled_in(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 2)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 2), pairs)

    page = build_contact_sheet(items, root, source_id="demo")

    assert "needs_human_review" in page
    assert "not filled in for you" in page
    assert "colour" in page


def test_contact_sheet_draws_a_box_per_plate(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "two.jpg", 1000, 800)
    write_label(
        root / "labels" / "two.txt", ["0 0.3 0.5 0.2 0.04", "0 0.7 0.5 0.1 0.06"]
    )
    pairs = discover_pairs(root)
    items = build_review_items(root, ["two"], pairs)

    page = build_contact_sheet(items, root, source_id="demo")

    assert page.count("<i style=") == 2


def test_contact_sheet_is_written_and_is_standalone(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 3)
    pairs = discover_pairs(root)
    items = build_review_items(root, select_sample(pairs, 3), pairs)
    destination = tmp_path / "out" / "sheet.html"

    write_contact_sheet(items, root, destination, source_id="demo")
    page = destination.read_text(encoding="utf-8")

    assert page.startswith("<!doctype html>")
    assert "http://" not in page and "https://" not in page  # no external resources


def test_contact_sheet_survives_an_item_without_plates(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "bare.jpg", 800, 600)
    pairs = discover_pairs(root)
    items = build_review_items(root, ["bare"], pairs)

    page = build_contact_sheet(items, root, source_id="demo")

    assert "no annotated plate" in page


# --------------------------------------------------------------------------
# Discovery and CLI
# --------------------------------------------------------------------------


def test_discover_pairs_matches_by_stem(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 3)
    pairs = discover_pairs(root)

    assert sorted(pairs) == ["0", "1", "2"]
    image, label = pairs["1"]
    assert image.name == "1.jpg"
    assert label is not None and label.name == "1.txt"


def test_discover_pairs_skips_ambiguous_stems(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "x.jpg", fmt="jpg")
    write_image(root / "images" / "x.png", fmt="png")
    write_image(root / "images" / "y.jpg", fmt="jpg")

    pairs = discover_pairs(root)

    assert sorted(pairs) == ["y"]


def test_discover_pairs_ignores_paperwork(tmp_path: Path) -> None:
    root = make_dataset(tmp_path, 2)
    (root / "README.txt").write_text("notes", encoding="utf-8")
    (root / "classes.txt").write_text("plate", encoding="utf-8")

    pairs = discover_pairs(root)

    assert sorted(pairs) == ["0", "1"]


def test_cli_writes_both_artefacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = make_dataset(tmp_path / "ext", 12)
    csv_path = tmp_path / "out" / "review.csv"
    sheet_path = tmp_path / "out" / "sheet.html"

    assert cli.main(
        [
            str(root),
            "--source-id", "demo",
            "--csv", str(csv_path),
            "--contact-sheet", str(sheet_path),
            "--size", "5",
        ]
    ) == cli.EXIT_OK

    assert csv_path.is_file() and sheet_path.is_file()
    output = capsys.readouterr().out
    assert "sample size  : 5" in output
    assert "needs_human_review" in output
    assert "Still requires a person" in output


def test_cli_is_reproducible(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 40)
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"

    cli.main([str(root), "--source-id", "d", "--csv", str(first), "--size", "10"])
    cli.main([str(root), "--source-id", "d", "--csv", str(second), "--size", "10"])

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_cli_does_not_modify_the_dataset(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ext", 6)
    before = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    cli.main([str(root), "--source-id", "d", "--csv", str(tmp_path / "r.csv")])

    after = {p: p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}
    assert before == after


def test_cli_refuses_to_write_into_the_dataset_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_dataset(tmp_path / "ext", 3)
    target = cli.REPO_ROOT / "dataset" / "review.csv"

    assert cli.main([str(root), "--source-id", "d", "--csv", str(target)]) == cli.EXIT_FAILED
    assert "competition dataset" in capsys.readouterr().err
    assert not target.exists()


def test_cli_refuses_to_write_into_the_audited_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_dataset(tmp_path / "ext", 3)

    assert cli.main(
        [str(root), "--source-id", "d", "--csv", str(root / "review.csv")]
    ) == cli.EXIT_FAILED
    assert "refusing to write" in capsys.readouterr().err


def test_cli_rejects_a_missing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(
        [str(tmp_path / "nope"), "--source-id", "d", "--csv", str(tmp_path / "r.csv")]
    ) == cli.EXIT_FAILED
    assert "not a directory" in capsys.readouterr().err


def test_cli_rejects_a_directory_without_images(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    assert cli.main(
        [str(root), "--source-id", "d", "--csv", str(tmp_path / "r.csv")]
    ) == cli.EXIT_FAILED
    assert "no images found" in capsys.readouterr().err
