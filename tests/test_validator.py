"""Tests for the Russian plate format validator."""

from __future__ import annotations

import pytest

from src.validator import is_valid_plate, normalize_plate, validate_plate


@pytest.mark.parametrize(
    "plate",
    [
        "A123BC77",     # two-digit region
        "X001XX01",     # lowest two-digit region
        "O999OO99",      # highest two-digit region
        "M111MM102",    # three-digit region starting with 1
        "K555KK299",    # three-digit region starting with 2
        "T777TT799",    # three-digit region starting with 7
        "E100KX716",
    ],
)
def test_valid_plates(plate: str) -> None:
    assert is_valid_plate(plate), validate_plate(plate).reason


def test_two_digit_region_is_accepted() -> None:
    result = validate_plate("A123BC73")
    assert result.is_valid
    assert result.normalized == "A123BC73"


@pytest.mark.parametrize("region", ["102", "116", "197", "277", "716", "750", "797"])
def test_three_digit_regions_with_allowed_prefix(region: str) -> None:
    assert is_valid_plate(f"A123BC{region}")


@pytest.mark.parametrize("region", ["302", "402", "502", "602", "802", "902", "002"])
def test_three_digit_regions_with_forbidden_prefix(region: str) -> None:
    result = validate_plate(f"A123BC{region}")
    assert not result.is_valid
    assert "1, 2 or 7" in result.reason


@pytest.mark.parametrize("plate", ["A123BC00", "A123BC000"])
def test_unassigned_zero_region(plate: str) -> None:
    assert not is_valid_plate(plate)


@pytest.mark.parametrize(
    "plate",
    [
        "D123BC77",     # D is not a plate letter
        "A123BZ77",     # Z is not a plate letter
        "A123QQ77",
        "G123GG16",
    ],
)
def test_invalid_letters(plate: str) -> None:
    assert not is_valid_plate(plate)


@pytest.mark.parametrize(
    "plate",
    [
        "",             # empty
        "   ",          # whitespace only
        "A12BC77",      # too few digits
        "A1234BC77",    # too many digits
        "AA123BC77",    # letter in the wrong position
        "A123B77",      # only one trailing letter
        "A123BCC77",    # three trailing letters
        "A123BC7",      # one-digit region
        "A123BC7777",   # four-digit region
        "123ABC77",     # digits first
        "A123BC7A",     # letter inside the region
        "A123-BC-77!",  # stray punctuation
    ],
)
def test_malformed_numbers(plate: str) -> None:
    assert not is_valid_plate(plate)


def test_normalization_uppercases_and_strips_spaces() -> None:
    assert normalize_plate(" a 123 bc 77 ") == "A123BC77"
    assert is_valid_plate(" a 123 bc 77 ")


def test_normalization_folds_cyrillic_lookalikes() -> None:
    # Cyrillic А, В, С — visually identical to the Latin letters on the plate.
    assert normalize_plate("\u0410123\u0412\u042177") == "A123BC77"
    assert is_valid_plate("\u0430123\u0432\u044177")


def test_hyphen_separators_are_removed() -> None:
    assert normalize_plate("A123-BC-77") == "A123BC77"


def test_validation_result_is_falsy_when_invalid() -> None:
    assert not validate_plate("nonsense")
    assert validate_plate("A123BC77")


def test_reason_is_empty_for_valid_plate() -> None:
    assert validate_plate("A123BC77").reason == ""
