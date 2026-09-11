"""Tests for JPEG DC decoding and the conservative plate-colour heuristic."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.jpeg_dc import (
    Component,
    DcImage,
    JpegUnsupported,
    _build_huffman,
    _extend,
    _strip_entropy,
    decode_dc,
    ycbcr_to_rgb,
)
from src.plate_color import (
    MIN_BLOCKS_FOR_COLOUR,
    ColourReading,
    PlateColour,
    read_plate_colour,
    sample_plate_colour,
)


def make_image(
    luma: int, blue: int, red: int, *, blocks: int = 16, colour: bool = True
) -> DcImage:
    """A uniform DC image of the given YCbCr value."""
    def plane(value: int) -> Component:
        component = Component(0, 1, 1, 0)
        component.blocks_wide = blocks
        component.blocks_high = blocks
        component.values = [value] * (blocks * blocks)
        return component

    components = [plane(luma)]
    if colour:
        components += [plane(blue), plane(red)]
    return DcImage(width=blocks * 8, height=blocks * 8, components=components)


def patched_image(
    background: tuple[int, int, int], patch: tuple[int, int, int], blocks: int = 16
) -> DcImage:
    """An image with a distinct 4x4-block patch in the middle."""
    image = make_image(*background, blocks=blocks)
    for index, value in enumerate(patch):
        plane = image.components[index]
        assert plane.values is not None
        for row in range(6, 10):
            for column in range(6, 10):
                plane.values[row * blocks + column] = value
    return image


# --------------------------------------------------------------------------
# JPEG primitives
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "length", "expected"),
    [(0, 0, 0), (1, 1, 1), (0, 1, -1), (3, 2, 3), (0, 2, -3), (5, 3, 5), (2, 3, -5)],
)
def test_extend_recovers_signed_values(value: int, length: int, expected: int) -> None:
    assert _extend(value, length) == expected


def test_huffman_table_maps_canonical_codes() -> None:
    # One 1-bit code (0) and one 2-bit code (10).
    counts = [1, 1] + [0] * 14
    table = _build_huffman(counts, [0xAA, 0xBB])

    assert table[0x0000] >> 5 == 0xAA  # 0...
    assert table[0x0000] & 31 == 1
    assert table[0x8000] >> 5 == 0xBB  # 10...
    assert table[0x8000] & 31 == 2


def test_strip_entropy_unstuffs_and_drops_restarts() -> None:
    data = b"\x00\x01\xff\x00\x02\xff\xd0\x03\xff\xd9tail"
    assert _strip_entropy(data, 0) == b"\x00\x01\xff\x02\x03"


def test_strip_entropy_stops_at_a_real_marker() -> None:
    assert _strip_entropy(b"\x01\x02\xff\xda\x99", 0) == b"\x01\x02"


def test_decode_rejects_non_jpeg() -> None:
    with pytest.raises(JpegUnsupported, match="not a JPEG"):
        decode_dc(b"not a jpeg at all")


def test_decode_rejects_progressive_jpeg() -> None:
    # SOI then an SOF2 (progressive) segment.
    data = b"\xff\xd8\xff\xc2\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00"
    with pytest.raises(JpegUnsupported, match="progressive"):
        decode_dc(data)


@pytest.mark.parametrize(
    ("ycbcr", "expected"),
    [
        ((255, 128, 128), (255, 255, 255)),  # white
        ((0, 128, 128), (0, 0, 0)),          # black
        ((128, 128, 128), (128, 128, 128)),  # grey
    ],
)
def test_ycbcr_to_rgb_neutrals(ycbcr: tuple[int, int, int], expected: tuple[int, int, int]) -> None:
    assert ycbcr_to_rgb(*ycbcr) == expected


def test_ycbcr_to_rgb_yellow_is_yellow() -> None:
    """Low Cb with raised Cr must come out yellow, not something else."""
    red, green, blue = ycbcr_to_rgb(226, 1, 149)

    assert red > 200 and green > 200
    assert blue < 60


# --------------------------------------------------------------------------
# Region sampling
# --------------------------------------------------------------------------


def test_sample_returns_the_region_mean() -> None:
    image = patched_image((100, 128, 128), (200, 90, 150))
    middle = image.sample_ycbcr(0.40, 0.40, 0.60, 0.60)

    assert middle is not None
    assert middle[0] == pytest.approx(200, abs=1)
    assert middle[1] == pytest.approx(90, abs=1)


def test_sample_of_a_different_region_sees_the_background() -> None:
    image = patched_image((100, 128, 128), (200, 90, 150))
    corner = image.sample_ycbcr(0.0, 0.0, 0.2, 0.2)

    assert corner is not None
    assert corner[0] == pytest.approx(100, abs=1)


def test_greyscale_reports_neutral_chroma() -> None:
    image = make_image(120, 0, 0, colour=False)
    sampled = image.sample_ycbcr(0, 0, 1, 1)

    assert sampled == (120, 128.0, 128.0)


def test_tiny_region_still_samples_a_block() -> None:
    image = make_image(140, 128, 128)
    assert image.sample_ycbcr(0.5, 0.5, 0.5001, 0.5001) is not None


# --------------------------------------------------------------------------
# Yellowness
# --------------------------------------------------------------------------


def test_white_plate_has_no_yellowness() -> None:
    assert ColourReading(200, 128, 128).yellowness == pytest.approx(0.0)


def test_yellow_scores_high_blue_scores_negative() -> None:
    yellow = ColourReading(180, 95, 145).yellowness
    blue = ColourReading(120, 170, 110).yellowness

    assert yellow > 40
    assert blue < -40


def test_red_alone_is_not_classified_as_yellow() -> None:
    """Red raises Cr too, so the classifier also requires Cb to fall."""
    image = patched_image((110, 128, 128), (120, 130, 170))  # red patch
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.reliable
    assert not colour.is_yellow_candidate()
    assert colour.plate is not None and colour.plate.blue_diff > 115


# --------------------------------------------------------------------------
# The detector must actually fire on yellow
# --------------------------------------------------------------------------


def test_yellow_plate_on_a_neutral_car_is_a_candidate() -> None:
    """The decisive test: a genuinely yellow plate must be flagged."""
    image = patched_image((110, 128, 128), (170, 92, 146))
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.reliable, colour.note
    assert colour.is_yellow_candidate()
    assert colour.strength() in {"weak", "strong"}
    assert colour.plate is not None and colour.plate.yellowness > 30


def test_strongly_yellow_plate_is_marked_strong() -> None:
    image = patched_image((110, 128, 128), (190, 60, 160))
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.strength() == "strong"


def test_white_plate_is_not_a_candidate() -> None:
    image = patched_image((110, 128, 128), (210, 127, 129))
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.reliable
    assert not colour.is_yellow_candidate()
    assert colour.strength() == "none"


def test_yellow_car_with_a_white_plate_is_rejected() -> None:
    """The false positive the guard exists for."""
    image = patched_image((160, 85, 150), (210, 127, 129))  # yellow body, white plate
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.reliable
    assert not colour.is_yellow_candidate()
    assert colour.margin < 0  # plate is less yellow than its surroundings


def test_uniformly_yellow_scene_is_rejected() -> None:
    """A yellow plate on an equally yellow car fails the margin test."""
    image = make_image(170, 90, 148)
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert colour.reliable
    assert colour.margin == pytest.approx(0.0, abs=0.5)
    assert not colour.is_yellow_candidate()


def test_dark_region_is_not_called_yellow() -> None:
    image = patched_image((30, 128, 128), (20, 100, 140))
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert not colour.is_yellow_candidate()


def test_small_plate_abstains_rather_than_guessing() -> None:
    image = patched_image((110, 128, 128), (170, 92, 146), blocks=16)
    colour = sample_plate_colour(image, 0.5, 0.5, 0.05, 0.05)  # under 3 blocks wide

    assert not colour.reliable
    assert "too small" in colour.note
    assert not colour.is_yellow_candidate()
    assert colour.blocks_wide < MIN_BLOCKS_FOR_COLOUR


def test_greyscale_image_abstains() -> None:
    image = make_image(180, 0, 0, colour=False)
    colour = sample_plate_colour(image, 0.5, 0.5, 0.25, 0.25)

    assert not colour.reliable
    assert "greyscale" in colour.note
    assert not colour.is_yellow_candidate()


# --------------------------------------------------------------------------
# File-level entry point
# --------------------------------------------------------------------------


def test_read_plate_colour_abstains_on_a_missing_file(tmp_path: Path) -> None:
    colour = read_plate_colour(tmp_path / "nope.jpg", 0.5, 0.5, 0.2, 0.1)

    assert not colour.reliable
    assert not colour.is_yellow_candidate()


def test_read_plate_colour_abstains_on_a_non_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    colour = read_plate_colour(path, 0.5, 0.5, 0.2, 0.1)

    assert not colour.reliable
    assert "baseline JPEG only" in colour.note


def test_read_plate_colour_abstains_on_corrupt_data(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"\xff\xd8" + b"\x00" * 64)

    colour = read_plate_colour(path, 0.5, 0.5, 0.2, 0.1)

    assert not colour.reliable
    assert not colour.is_yellow_candidate()


def test_unreliable_colour_never_claims_a_candidate() -> None:
    colour = PlateColour(ColourReading(200, 60, 160), None, 0, False, "unreliable")
    assert not colour.is_yellow_candidate()
    assert colour.strength() == "none"
