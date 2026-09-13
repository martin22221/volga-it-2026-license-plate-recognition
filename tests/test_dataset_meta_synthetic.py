"""Dataset-level guards on synthetic rows: per-type grammar and V1 artefacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dataset_meta import validate_meta
from tests.test_dataset_meta import errors, make_dataset, row


def synthetic(**overrides: str) -> dict[str, str]:
    base = {
        "image": "images/synthetic/syn_v2_1_00000.jpg",
        "is_synthetic": "true",
        "source": "volga_synthetic_generator",
        "license": "CC BY 4.0",
    }
    return row(**{**base, **overrides})


@pytest.mark.parametrize(
    "plate_type, plate_num",
    [("type1b", "AB12377"), ("type1", "A123BC77"), ("type1a", "A123BC777"), ("type1b", "AB123##"), ("type1", "A#23BC77")],
)
def test_synthetic_rows_following_their_structure_pass(tmp_path: Path, plate_type: str, plate_num: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [synthetic(plate_type=plate_type, plate_num=plate_num)]))
    assert report.is_valid, errors(report)


@pytest.mark.parametrize(
    "plate_type, plate_num",
    [
        ("type1b", "A123BC77"),   # the V1 mistake
        ("type1b", "A123BC777"),
        ("type1b", "AB123777"),   # 3-digit region on type 1B
        ("type1", "AB12377"),
        ("type1a", "AB12377"),
        ("type1b", "A#23BC77"),   # masked, but still the wrong structure
    ],
)
def test_synthetic_rows_violating_their_structure_fail(tmp_path: Path, plate_type: str, plate_num: str) -> None:
    report = validate_meta(make_dataset(tmp_path, [synthetic(plate_type=plate_type, plate_num=plate_num)]))
    assert not report.is_valid
    assert "violates the" in errors(report)


def test_real_rows_are_not_second_guessed(tmp_path: Path) -> None:
    # Pre-existing behaviour: a real annotation records what a person saw.
    real = row(image="images/real/000009.jpg", plate_type="type1b", plate_num="M111MM102")
    report = validate_meta(make_dataset(tmp_path, [real]))
    assert report.is_valid, errors(report)


def test_generator_v1_output_is_rejected(tmp_path: Path) -> None:
    v1 = synthetic(image="images/synthetic/syn_20260913_00000.jpg", plate_type="type1", plate_num="A123BC77")
    report = validate_meta(make_dataset(tmp_path, [v1]))
    assert not report.is_valid
    assert "generator V1" in errors(report)


def test_generator_v2_output_names_are_accepted(tmp_path: Path) -> None:
    v2 = synthetic(image="images/synthetic/syn_v2_20260913_00000.jpg", plate_type="type1b", plate_num="TX58216")
    assert validate_meta(make_dataset(tmp_path, [v2])).is_valid
