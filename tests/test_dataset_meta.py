"""Tests for the dataset metadata schema and validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset_meta import (
    ALLOWED_CONDITIONS,
    ALLOWED_PLATE_TYPES,
    BBOX_COLUMNS,
    CSV_DELIMITER,
    DATASET_LICENSE,
    QUAD_COLUMNS,
    REJECTED_SOURCES,
    REQUIRED_COLUMNS,
    MetaFormatError,
    Severity,
    ValidationReport,
    is_redistributable_license,
    license_incompatibility,
    normalize_license,
    parse_bool,
    parse_number,
    polygon_signed_area,
    read_meta,
    split_conditions,
    validate_meta,
)

#: A well-formed real annotation used as the base for most test rows.
GOOD_ROW: dict[str, str] = {
    "image": "images/real/000001.jpg",
    "plate_num": "A123BC77",
    "plate_type": "type1a",
    "bbox_x": "100",
    "bbox_y": "200",
    "bbox_w": "80",
    "bbox_h": "40",
    "quad_x1": "100",
    "quad_y1": "200",
    "quad_x2": "180",
    "quad_y2": "200",
    "quad_x3": "180",
    "quad_y3": "240",
    "quad_x4": "100",
    "quad_y4": "240",
    "is_vehicle": "true",
    "is_synthetic": "false",
    "source": "example-collection",
    "license": "CC BY 4.0",
    "conditions": "day|angle",
}


def make_dataset(
    root: Path,
    rows: list[dict[str, str]],
    *,
    header: list[str] | None = None,
    create_images: bool = True,
) -> Path:
    """Write a dataset tree under ``root`` and return the ``meta.csv`` path."""
    columns = header if header is not None else list(REQUIRED_COLUMNS)
    (root / "images" / "real").mkdir(parents=True, exist_ok=True)
    (root / "images" / "synthetic").mkdir(parents=True, exist_ok=True)

    lines = [CSV_DELIMITER.join(columns)]
    for row in rows:
        lines.append(CSV_DELIMITER.join(row.get(column, "") for column in columns))
        image = row.get("image", "").replace("\\", "/")
        if create_images and image:
            path = root / image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    meta_path = root / "meta.csv"
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return meta_path


def row(**overrides: str) -> dict[str, str]:
    """A copy of :data:`GOOD_ROW` with ``overrides`` applied."""
    return {**GOOD_ROW, **overrides}


def messages(report: ValidationReport, severity: Severity) -> str:
    """All messages of ``severity``, joined for substring assertions."""
    return " | ".join(
        issue.message for issue in report.issues if issue.severity is severity
    )


def errors(report: ValidationReport) -> str:
    return messages(report, Severity.ERROR)


def warnings(report: ValidationReport) -> str:
    return messages(report, Severity.WARNING)


# --------------------------------------------------------------------------
# Schema constants
# --------------------------------------------------------------------------


def test_required_columns_match_the_agreed_schema() -> None:
    assert REQUIRED_COLUMNS == (
        "image",
        "plate_num",
        "plate_type",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "quad_x1",
        "quad_y1",
        "quad_x2",
        "quad_y2",
        "quad_x3",
        "quad_y3",
        "quad_x4",
        "quad_y4",
        "is_vehicle",
        "is_synthetic",
        "source",
        "license",
        "conditions",
    )


def test_allowed_vocabularies() -> None:
    assert ALLOWED_PLATE_TYPES == {"type1", "type1a", "type1b", "other"}
    assert ALLOWED_CONDITIONS == (
        "day",
        "night",
        "rain",
        "snow",
        "dirt",
        "glare",
        "motion_blur",
        "angle",
    )


def test_geometry_column_groups() -> None:
    assert BBOX_COLUMNS == ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    assert QUAD_COLUMNS == (
        "quad_x1",
        "quad_y1",
        "quad_x2",
        "quad_y2",
        "quad_x3",
        "quad_y3",
        "quad_x4",
        "quad_y4",
    )


def test_committed_meta_csv_has_the_canonical_header() -> None:
    meta_path = Path(__file__).resolve().parents[1] / "dataset" / "meta.csv"
    header = meta_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == CSV_DELIMITER.join(REQUIRED_COLUMNS)


# --------------------------------------------------------------------------
# Cell parsing helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", " Y "])
def test_parse_bool_true(value: str) -> None:
    assert parse_bool(value) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "0", "no", " N "])
def test_parse_bool_false(value: str) -> None:
    assert parse_bool(value) is False


@pytest.mark.parametrize("value", ["", "maybe", "2", "-"])
def test_parse_bool_unrecognised(value: str) -> None:
    assert parse_bool(value) is None


@pytest.mark.parametrize(
    ("value", "expected"), [("12", 12.0), ("12.5", 12.5), ("12,5", 12.5), (" -3 ", -3.0)]
)
def test_parse_number(value: str, expected: float) -> None:
    assert parse_number(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "nan", "inf", "1.2.3"])
def test_parse_number_rejects_non_numbers(value: str) -> None:
    assert parse_number(value) is None


def test_split_conditions_accepts_both_separators_and_normalises_case() -> None:
    assert split_conditions("Day| Glare ,angle") == ["day", "glare", "angle"]


def test_split_conditions_on_blank_cell() -> None:
    assert split_conditions("   ") == []


def test_clockwise_quad_has_positive_area_in_image_coordinates() -> None:
    clockwise = [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)]
    assert polygon_signed_area(clockwise) == pytest.approx(50.0)
    assert polygon_signed_area(list(reversed(clockwise))) == pytest.approx(-50.0)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def test_read_meta_returns_header_and_rows(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row()])
    header, rows = read_meta(meta_path)

    assert header == list(REQUIRED_COLUMNS)
    assert len(rows) == 1
    assert rows[0].line == 2
    assert rows[0].plate_num == "A123BC77"
    assert rows[0].conditions == ["day", "angle"]
    assert rows[0].is_vehicle is True
    assert rows[0].is_synthetic is False


def test_read_meta_skips_blank_lines(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row()])
    meta_path.write_text(
        meta_path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8"
    )
    _, rows = read_meta(meta_path)
    assert len(rows) == 1


def test_read_meta_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MetaFormatError):
        read_meta(tmp_path / "nope.csv")


def test_read_meta_rejects_empty_file(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.csv"
    meta_path.write_text("", encoding="utf-8")
    with pytest.raises(MetaFormatError):
        read_meta(meta_path)


def test_header_only_file_is_valid(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, []))
    assert report.is_valid
    assert report.stats.total_rows == 0


# --------------------------------------------------------------------------
# Header validation
# --------------------------------------------------------------------------


def test_missing_column_is_an_error(tmp_path: Path) -> None:
    header = [c for c in REQUIRED_COLUMNS if c != "license"]
    report = validate_meta(make_dataset(tmp_path, [row()], header=header))

    assert not report.is_valid
    assert "missing required column(s): license" in errors(report)


def test_unknown_column_is_only_a_warning(tmp_path: Path) -> None:
    header = [*REQUIRED_COLUMNS, "annotator"]
    report = validate_meta(make_dataset(tmp_path, [row(annotator="mp")], header=header))

    assert report.is_valid
    assert "unrecognised column(s) ignored: annotator" in warnings(report)


def test_reordered_columns_warn_but_stay_valid(tmp_path: Path) -> None:
    header = [REQUIRED_COLUMNS[1], REQUIRED_COLUMNS[0], *REQUIRED_COLUMNS[2:]]
    report = validate_meta(make_dataset(tmp_path, [row()], header=header))

    assert report.is_valid
    assert "canonical order" in warnings(report)


def test_repeated_column_is_an_error(tmp_path: Path) -> None:
    header = [*REQUIRED_COLUMNS, "source"]
    report = validate_meta(make_dataset(tmp_path, [row()], header=header))

    assert not report.is_valid
    assert "declared more than once: source" in errors(report)


# --------------------------------------------------------------------------
# Row validation
# --------------------------------------------------------------------------


def test_valid_row_passes_without_findings(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row()]))
    assert report.is_valid, errors(report)
    assert report.warnings == []


@pytest.mark.parametrize("plate_type", sorted(ALLOWED_PLATE_TYPES))
def test_every_allowed_plate_type_is_accepted(tmp_path: Path, plate_type: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_type=plate_type)]))
    assert report.is_valid, errors(report)


@pytest.mark.parametrize("plate_type", ["type2", "TYPE1A", "yellow", ""])
def test_invalid_plate_type_is_rejected(tmp_path: Path, plate_type: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_type=plate_type)]))
    assert not report.is_valid
    assert "plate_type" in errors(report)


@pytest.mark.parametrize("column", ["is_vehicle", "is_synthetic"])
@pytest.mark.parametrize("value", ["", "maybe", "2"])
def test_invalid_boolean_is_rejected(tmp_path: Path, column: str, value: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(**{column: value})]))
    assert not report.is_valid
    assert column in errors(report)


@pytest.mark.parametrize(
    "overrides",
    [
        {"bbox_w": "0"},
        {"bbox_h": "-5"},
        {"bbox_x": "-1"},
        {"bbox_w": "wide"},
    ],
    ids=["zero-width", "negative-height", "negative-origin", "not-a-number"],
)
def test_invalid_bbox_is_rejected(tmp_path: Path, overrides: dict[str, str]) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(**overrides)]))
    assert not report.is_valid
    assert "bbox" in errors(report)


def test_counter_clockwise_quad_is_rejected(tmp_path: Path) -> None:
    counter_clockwise = row(
        quad_x1="100", quad_y1="200",
        quad_x2="100", quad_y2="240",
        quad_x3="180", quad_y3="240",
        quad_x4="180", quad_y4="200",
    )
    report = validate_meta(make_dataset(tmp_path, [counter_clockwise]))

    assert not report.is_valid
    assert "counter-clockwise" in errors(report)


def test_degenerate_quad_is_rejected(tmp_path: Path) -> None:
    collinear = row(
        quad_x1="100", quad_y1="200",
        quad_x2="140", quad_y2="200",
        quad_x3="180", quad_y3="200",
        quad_x4="160", quad_y4="200",
    )
    report = validate_meta(make_dataset(tmp_path, [collinear]))

    assert not report.is_valid
    assert "degenerate" in errors(report)


def test_repeated_quad_corner_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(quad_x3="100", quad_y3="200")]))
    assert not report.is_valid
    assert "repeated corners" in errors(report)


def test_negative_quad_coordinate_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(quad_x1="-1")]))
    assert not report.is_valid
    assert "non-negative" in errors(report)


def test_non_numeric_quad_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(quad_y2="left")]))
    assert not report.is_valid
    assert "quad columns must all be numeric" in errors(report)


def test_quad_disagreeing_with_bbox_warns(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(bbox_w="400")]))
    assert "quad extent disagrees with bbox" in warnings(report)


def test_rotated_plate_quad_is_accepted(tmp_path: Path) -> None:
    rotated = row(
        bbox_x="100", bbox_y="200", bbox_w="80", bbox_h="48",
        quad_x1="100", quad_y1="208",
        quad_x2="180", quad_y2="200",
        quad_x3="180", quad_y3="240",
        quad_x4="100", quad_y4="248",
    )
    report = validate_meta(make_dataset(tmp_path, [rotated]))
    assert report.is_valid, errors(report)
    assert report.warnings == []


def test_missing_plate_num_for_target_type_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_num="")]))
    assert not report.is_valid
    assert "plate_num is empty" in errors(report)


def test_partially_unreadable_plate_num_is_accepted(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_num="A12#BC7#")]))
    assert report.is_valid, errors(report)
    assert report.warnings == []


def test_fully_unreadable_plate_num_warns(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_num="########")]))
    assert report.is_valid
    assert "unreadable" in warnings(report)


def test_other_type_may_omit_plate_num(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_type="other", plate_num="")]))
    assert report.is_valid, errors(report)


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ALLOWED_CONDITIONS)
def test_every_allowed_condition_is_accepted(tmp_path: Path, condition: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(conditions=condition)]))
    assert report.is_valid, errors(report)


def test_unknown_condition_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(conditions="day|fog")]))
    assert not report.is_valid
    assert "condition 'fog' is not in the controlled list" in errors(report)


def test_day_and_night_together_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(conditions="day|night")]))
    assert not report.is_valid
    assert "both 'day' and 'night'" in errors(report)


def test_repeated_condition_warns(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(conditions="day|day")]))
    assert report.is_valid
    assert "condition(s) repeated: day" in warnings(report)


def test_blank_conditions_are_allowed(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(conditions="")]))
    assert report.is_valid, errors(report)
    assert report.stats.rows_without_conditions == 1


# --------------------------------------------------------------------------
# Image paths and files
# --------------------------------------------------------------------------


def test_missing_image_file_is_reported(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row()], create_images=False)
    report = validate_meta(meta_path)

    assert not report.is_valid
    assert "not found" in errors(report)
    assert report.stats.missing_files == ["images/real/000001.jpg"]


def test_file_check_can_be_skipped(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row()], create_images=False)
    report = validate_meta(meta_path, check_files=False)

    assert report.is_valid, errors(report)
    assert report.stats.missing_files == []


@pytest.mark.parametrize(
    "image",
    ["000001.jpg", "labels/000001.jpg", "images/000001.jpg"],
)
def test_image_outside_the_expected_folders_is_rejected(tmp_path: Path, image: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(image=image)]))
    assert not report.is_valid
    assert "must start with" in errors(report)


@pytest.mark.parametrize("image", ["../secret.jpg", "images/real/../../x.jpg"])
def test_image_escaping_the_dataset_root_is_rejected(tmp_path: Path, image: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(image=image)], create_images=False))
    assert not report.is_valid
    assert "stay inside it" in errors(report)


def test_empty_image_path_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(image="")]))
    assert not report.is_valid
    assert "image path is empty" in errors(report)


def test_backslash_paths_are_normalised(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row(image=r"images\real\000001.jpg")])
    report = validate_meta(meta_path)
    assert report.is_valid, errors(report)


def test_real_image_marked_synthetic_warns(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(is_synthetic="true")]))
    assert report.is_valid
    assert "stored as a real image but is_synthetic=true" in warnings(report)


def test_synthetic_image_marked_real_warns(tmp_path: Path) -> None:
    synthetic_path = row(image="images/synthetic/000001.png", is_synthetic="false")
    report = validate_meta(make_dataset(tmp_path, [synthetic_path]))

    assert report.is_valid
    assert "stored as a synthetic image but is_synthetic=false" in warnings(report)


def test_synthetic_row_needs_no_source_or_license(tmp_path: Path) -> None:
    synthetic = row(
        image="images/synthetic/000001.png",
        is_synthetic="true",
        source="",
        license="",
    )
    report = validate_meta(make_dataset(tmp_path, [synthetic]))

    assert report.is_valid, errors(report)
    assert report.warnings == []


@pytest.mark.parametrize("column", ["source", "license"])
def test_blank_provenance_on_a_real_image_warns(tmp_path: Path, column: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(**{column: ""})]))

    assert report.is_valid
    assert f"{column} is blank for a real image" in warnings(report)


# --------------------------------------------------------------------------
# License compatibility with the CC BY 4.0 dataset license
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("CC BY 4.0", "cc by 4.0"),
        ("cc-by-4.0", "cc by 4.0"),
        ("Creative Commons Attribution 4.0 International", "cc by 4.0"),
        ("  CC0  1.0  ", "cc0 1.0"),
        ("Own Work", "own work"),
        ("", ""),
    ],
)
def test_normalize_license(value: str, expected: str) -> None:
    assert normalize_license(value) == expected


@pytest.mark.parametrize(
    "value",
    ["own work", "CC0 1.0", "public domain", "CC BY 4.0", "CC BY 3.0", "cc-by-2.0"],
)
def test_redistributable_licenses_are_recognised(value: str) -> None:
    assert is_redistributable_license(value)
    assert license_incompatibility(value) is None


@pytest.mark.parametrize(
    ("value", "clause"),
    [
        ("CC BY-NC 4.0", "non-commercial"),
        ("CC BY-NC-SA 4.0", "non-commercial"),
        ("CC BY-ND 4.0", "no-derivatives"),
        ("CC BY-SA 4.0", "share-alike"),
        ("Creative Commons Attribution-NonCommercial 4.0", "non-commercial"),
        ("CC BY-SA", "share-alike"),
    ],
)
def test_incompatible_license_clauses_are_named(value: str, clause: str) -> None:
    assert license_incompatibility(value) == clause
    assert not is_redistributable_license(value)


def test_share_alike_is_rejected_despite_allowing_redistribution() -> None:
    """ShareAlike permits redistribution but forbids relicensing under CC BY."""
    assert license_incompatibility("CC BY-SA 4.0") == "share-alike"


@pytest.mark.parametrize(
    "license_name",
    ["CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-SA 4.0", "CC BY-NC-ND 2.0"],
)
def test_non_redistributable_license_is_an_error(tmp_path: Path, license_name: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(license=license_name)]))

    assert not report.is_valid
    assert "cannot be redistributed" in errors(report)


def test_unrecognised_license_warns_for_human_review(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(license="Unsplash License")]))

    assert report.is_valid, errors(report)
    assert "not on the confirmed-redistributable list" in warnings(report)


@pytest.mark.parametrize("license_name", ["CC BY 4.0", "CC0 1.0", "own work"])
def test_confirmed_license_passes_without_findings(tmp_path: Path, license_name: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(license=license_name)]))

    assert report.is_valid, errors(report)
    assert report.warnings == []


def test_blank_license_warns_once_and_is_not_double_reported(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(license="")]))
    license_warnings = [i for i in report.warnings if i.column == "license"]

    assert len(license_warnings) == 1
    assert "blank" in license_warnings[0].message


def test_synthetic_rows_are_exempt_from_the_license_check(tmp_path: Path) -> None:
    synthetic = row(
        image="images/synthetic/000001.png",
        is_synthetic="true",
        source="",
        license="CC BY-NC 4.0",
    )
    report = validate_meta(make_dataset(tmp_path, [synthetic]))

    assert report.is_valid, errors(report)


def test_dataset_license_constant_is_cc_by_4() -> None:
    assert DATASET_LICENSE == "CC BY 4.0"


# --------------------------------------------------------------------------
# Rejected sources
# --------------------------------------------------------------------------


def test_rejected_source_is_an_error(tmp_path: Path) -> None:
    rejected = sorted(REJECTED_SOURCES)[0]
    report = validate_meta(make_dataset(tmp_path, [row(source=rejected)]))

    assert not report.is_valid
    assert f"source {rejected!r} is rejected" in errors(report)


def test_rejected_source_is_blocked_even_with_an_acceptable_license(tmp_path: Path) -> None:
    """The objection is to provenance; a good licence string cannot cure it."""
    rejected = sorted(REJECTED_SOURCES)[0]
    report = validate_meta(
        make_dataset(tmp_path, [row(source=rejected, license="CC BY 4.0")])
    )

    assert not report.is_valid
    assert "rejected" in errors(report)


def test_roboflow_two_line_source_is_rejected_for_provenance() -> None:
    reason = REJECTED_SOURCES["roboflow_two_line_russian_license_plates"]

    assert "REJECTED_FOR_SUBMISSION_PROVENANCE" in reason
    assert "provenance" in reason


def test_accepted_source_is_unaffected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(source="own-photos-ulyanovsk")]))

    assert report.is_valid, errors(report)


def test_synthetic_rows_are_not_checked_against_rejected_sources(tmp_path: Path) -> None:
    rejected = sorted(REJECTED_SOURCES)[0]
    synthetic = row(
        image="images/synthetic/000001.png", is_synthetic="true", source=rejected, license=""
    )
    report = validate_meta(make_dataset(tmp_path, [synthetic]))

    assert report.is_valid, errors(report)


def test_short_row_is_reported(tmp_path: Path) -> None:
    meta_path = make_dataset(tmp_path, [row()])
    lines = meta_path.read_text(encoding="utf-8").splitlines()
    lines[1] = CSV_DELIMITER.join(lines[1].split(CSV_DELIMITER)[:5])
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validate_meta(meta_path)
    assert not report.is_valid
    assert "header declares" in errors(report)


# --------------------------------------------------------------------------
# Background (negative) rows
# --------------------------------------------------------------------------


def background_row(**overrides: str) -> dict[str, str]:
    zeros = {column: "0" for column in BBOX_COLUMNS + QUAD_COLUMNS}
    return row(
        plate_type="other",
        plate_num="",
        is_vehicle="false",
        **zeros,
        **overrides,
    )


def test_background_row_skips_geometry_checks(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [background_row()]))

    assert report.is_valid, errors(report)
    assert report.warnings == []
    assert report.stats.background_rows == 1


def test_zero_geometry_outside_other_is_still_rejected(tmp_path: Path) -> None:
    zeros = {column: "0" for column in BBOX_COLUMNS + QUAD_COLUMNS}
    report = validate_meta(make_dataset(tmp_path, [row(**zeros)]))

    assert not report.is_valid
    assert "bbox size" in errors(report)


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------


def test_exact_duplicate_row_is_rejected(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(), row()]))

    assert not report.is_valid
    assert "exact duplicate of line 2" in errors(report)


def test_same_plate_twice_on_one_image_is_rejected(tmp_path: Path) -> None:
    duplicate = row(conditions="night")
    report = validate_meta(make_dataset(tmp_path, [row(), duplicate]))

    assert not report.is_valid
    assert "annotated twice" in errors(report)


def test_two_plates_on_one_image_are_allowed(tmp_path: Path) -> None:
    second = row(
        plate_num="M111MM102",
        bbox_x="300", bbox_y="200",
        quad_x1="300", quad_y1="200",
        quad_x2="380", quad_y2="200",
        quad_x3="380", quad_y3="240",
        quad_x4="300", quad_y4="240",
    )
    report = validate_meta(make_dataset(tmp_path, [row(), second]))

    assert report.is_valid, errors(report)
    assert report.stats.total_images == 1
    assert report.stats.total_rows == 2


def test_same_file_name_under_two_paths_warns(tmp_path: Path) -> None:
    clash = row(image="images/synthetic/000001.jpg", is_synthetic="true")
    report = validate_meta(make_dataset(tmp_path, [row(), clash]))

    assert report.is_valid, errors(report)
    assert "is used by two different paths" in warnings(report)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_statistics_are_collected(tmp_path: Path) -> None:
    rows = [
        row(),
        row(
            image="images/real/000002.jpg",
            plate_num="M111MM102",
            plate_type="type1b",
            conditions="night|glare",
            source="example-collection",
            license="CC BY 4.0",
        ),
        row(
            image="images/synthetic/000003.png",
            plate_num="X001XX01",
            plate_type="type1a",
            is_synthetic="true",
            source="",
            license="",
            conditions="day",
        ),
        background_row(image="images/real/000004.jpg", conditions="night"),
    ]
    report = validate_meta(make_dataset(tmp_path, rows))
    stats = report.stats

    assert report.is_valid, errors(report)
    assert stats.total_rows == 4
    assert stats.total_images == 4
    assert stats.real_images == 3
    assert stats.synthetic_images == 1
    assert stats.background_rows == 1
    assert stats.by_plate_type == {"type1a": 2, "type1b": 1, "other": 1}
    assert stats.by_condition == {"day": 2, "angle": 1, "night": 2, "glare": 1}
    assert stats.unique_plates_by_type == {"type1a": 2, "type1b": 1}
    assert stats.by_source == {"example-collection": 3}
    assert stats.by_license == {"CC BY 4.0": 3}
    assert stats.rows_missing_source == 0
    assert stats.rows_missing_license == 0


def test_missing_provenance_is_counted_for_real_rows_only(tmp_path: Path) -> None:
    rows = [
        row(source="", license=""),
        row(image="images/synthetic/000002.png", is_synthetic="true", source="", license=""),
    ]
    report = validate_meta(make_dataset(tmp_path, rows))

    assert report.stats.rows_missing_source == 1
    assert report.stats.rows_missing_license == 1


def test_issue_string_names_the_line_and_column(tmp_path: Path) -> None:
    report = validate_meta(make_dataset(tmp_path, [row(plate_type="type9")]))
    rendered = str(report.errors[0])

    assert "line 2" in rendered
    assert "plate_type" in rendered
    assert rendered.startswith("[error]")
