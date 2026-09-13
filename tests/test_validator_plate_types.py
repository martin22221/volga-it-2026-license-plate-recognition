"""Type-aware plate validation: type 1 / 1A vs the dedicated type 1B structure.

The untyped behaviour pinned by ``tests/test_validator.py`` is unchanged;
these tests cover the GOST R 50577-2018 type 1B structure ``MM 000 55``.
"""

from __future__ import annotations

import pytest

from src.validator import (
    FORMAT_BY_PLATE_TYPE,
    TYPE1B_REGION_LENGTHS,
    is_valid_plate,
    validate_plate,
)


@pytest.mark.parametrize("plate", ["AB12377", "XX00101", "TK99999", "EM50016"])
def test_type1b_structure_is_valid_for_type1b(plate: str) -> None:
    result = validate_plate(plate, "type1b")
    assert result.is_valid, result.reason
    assert result.format == "type1b"


@pytest.mark.parametrize("plate", ["A123BC77", "A123BC777", "M111MM102"])
def test_type1_structure_is_rejected_for_type1b(plate: str) -> None:
    result = validate_plate(plate, "type1b")
    assert not result.is_valid
    assert "2 LETTERS" in result.reason


@pytest.mark.parametrize("plate_type", ["type1", "type1a"])
def test_type1b_structure_is_rejected_for_type1_and_type1a(plate_type: str) -> None:
    assert not is_valid_plate("AB12377", plate_type)
    assert is_valid_plate("A123BC77", plate_type)


def test_type1b_region_is_two_digits() -> None:
    assert TYPE1B_REGION_LENGTHS == (2,)
    result = validate_plate("AB123777", "type1b")
    assert not result.is_valid
    assert "2 digits" in result.reason
    assert not is_valid_plate("AB12300", "type1b")


@pytest.mark.parametrize("plate", ["AB12D77", "ZB12377", "AB1237", "AB1234", "12AB377", "A12377"])
def test_malformed_type1b(plate: str) -> None:
    assert not is_valid_plate(plate, "type1b")


def test_untyped_validation_accepts_either_structure() -> None:
    assert validate_plate("A123BC77").format == "type1"
    assert validate_plate("AB12377").format == "type1b"
    assert validate_plate("ab 123 77").is_valid  # normalisation applies to 1B too
    assert validate_plate("АВ 123 77").normalized == "AB12377"


def test_other_and_unknown_types_fall_back_to_either_structure() -> None:
    assert is_valid_plate("AB12377", "other")
    assert is_valid_plate("A123BC77", "other")
    assert is_valid_plate("AB12377", None)


def test_format_table_covers_the_text_bearing_types() -> None:
    assert FORMAT_BY_PLATE_TYPE == {"type1": "type1", "type1a": "type1", "type1b": "type1b"}
