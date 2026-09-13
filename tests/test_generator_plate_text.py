"""Plate identities: allowed characters, region rules, determinism, uniqueness."""

from __future__ import annotations

import re

import numpy as np
import pytest

from dataset.generator.plate_text import (
    LETTERS,
    PlateIdentitySampler,
    PlateText,
    is_standard_plate,
    is_type1b_plate,
    sample_plate_text,
    sample_region,
)
from dataset.generator.rng import derive_seed, make_rng
from src.validator import ALLOWED_LETTERS, is_valid_plate


def _draw(seed: int, n: int, plate_type: str = "type1", p3: float = 0.35) -> list[str]:
    rng = make_rng(seed, "identity")
    return [sample_plate_text(rng, plate_type, p3).full for _ in range(n)]


def test_letter_set_matches_competition_alphabet() -> None:
    assert set(LETTERS) == set(ALLOWED_LETTERS) == set("ABEKMHOPCTYX")
    assert len(LETTERS) == 12


def test_same_seed_same_identities() -> None:
    assert _draw(20260913, 200) == _draw(20260913, 200)


def test_different_seed_different_identities() -> None:
    assert _draw(1, 50) != _draw(2, 50)


def test_derived_seeds_are_stable_across_processes_and_machines() -> None:
    # SHA-256 based, so these golden values never change (hash() is salted).
    assert derive_seed(1, "a") == 0x22E4A339DDFD6241
    assert derive_seed(20260913, "sample", 0) == 0xBE2CE4478C73B894
    assert derive_seed(20260913, "sample", 0) != derive_seed(20260913, "sample", 1)


@pytest.mark.parametrize("seed", [0, 7, 20260913])
def test_generated_plates_use_only_allowed_characters(seed: int) -> None:
    allowed = set(LETTERS) | set("0123456789")
    for plate in _draw(seed, 1000):
        assert set(plate) <= allowed, plate


def test_generated_plates_pass_the_repository_validator() -> None:
    plates = _draw(20260913, 5000)
    invalid = [plate for plate in plates if not is_valid_plate(plate)]
    assert invalid == []
    assert all(is_standard_plate(plate) for plate in plates)


def test_competition_mask_structure() -> None:
    pattern = re.compile(rf"^[{LETTERS}]\d{{3}}[{LETTERS}]{{2}}\d{{2,3}}$")
    for plate in _draw(3, 2000):
        assert pattern.fullmatch(plate), plate


def test_serial_number_never_000() -> None:
    assert all(plate[1:4] != "000" for plate in _draw(5, 5000))


def test_region_rules() -> None:
    rng = np.random.default_rng(0)
    regions = [sample_region(rng, 0.5) for _ in range(5000)]
    two = [r for r in regions if len(r) == 2]
    three = [r for r in regions if len(r) == 3]
    assert two and three
    assert all(r != "00" for r in two)
    assert all(r[0] in "127" for r in three)
    assert all(set(r) != {"0"} for r in regions)
    assert {len(r) for r in regions} == {2, 3}


def test_three_digit_probability_is_respected() -> None:
    rng = np.random.default_rng(1)
    assert all(len(sample_region(rng, 0.0)) == 2 for _ in range(500))
    assert all(len(sample_region(rng, 1.0)) == 3 for _ in range(500))


def test_standard_structure_agrees_with_src_validator_on_edge_cases() -> None:
    cases = ["A123BC77", "A123BC777", "A123BC877", "A123BC00", "A123BC000", "A123BC7", "Z123BC77", "A12BC77", "AB12377"]
    for case in cases:
        assert is_standard_plate(case) == is_valid_plate(case, "type1"), case


def test_type1b_structure_agrees_with_src_validator_on_edge_cases() -> None:
    cases = ["AB12377", "AB123777", "AB12300", "A123BC77", "AB1237", "ZB12377", "AB1234577", "XX00101"]
    for case in cases:
        assert is_type1b_plate(case) == is_valid_plate(case, "type1b"), case


def test_sampler_avoids_duplicate_identities() -> None:
    sampler = PlateIdentitySampler(make_rng(9, "identity"), three_digit_probability=0.35, max_images_per_plate=1)
    plates = [sampler.sample(plate_type).full for plate_type in ("type1", "type1a", "type1b") * 1000]
    assert len(set(plates)) == len(plates)


def test_type1b_draws_use_the_gost_1b_structure() -> None:
    rng = make_rng(4, "identity")
    plates = [sample_plate_text(rng, "type1b", 0.35) for _ in range(300)]
    for plate in plates:
        assert re.fullmatch(rf"[{LETTERS}]{{2}}\d{{3}}\d{{2}}", plate.full), plate.full
        assert plate.series2 == "" and plate.text_format == "type1b"
        assert is_valid_plate(plate.full, "type1b")
        assert not is_valid_plate(plate.full, "type1")


def test_unknown_plate_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        sample_plate_text(np.random.default_rng(0), "fancy", 0.3)


def test_characters_positions_and_roles() -> None:
    text = PlateText("A", "123", "BC", "777", "type1")
    chars = list(text.characters())
    assert [c for _, c, _ in chars] == list("A123BC777")
    assert [p for p, _, _ in chars] == list(range(9))
    assert [r for _, _, r in chars] == ["letter"] + ["digit"] * 3 + ["letter"] * 2 + ["region"] * 3
