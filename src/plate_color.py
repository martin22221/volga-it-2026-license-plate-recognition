"""Conservative colour sampling of annotated plate regions.

Answers one narrow question: *is the plate itself yellow?*  It is used to
surface `type1b` candidates for a human to confirm -- never to assign a plate
type.

Colour comes from :mod:`src.jpeg_dc`, which recovers a JPEG's DC coefficients
(a 1/8-scale thumbnail) without a decoding dependency.  Two deliberate guards
keep the answer conservative:

**The surroundings must not be yellow too.**  The single most likely false
positive is a yellow car.  Every reading therefore samples a ring around the
plate as well, and a plate only counts as a candidate when it is materially
more yellow than what surrounds it.  A yellow car with a white plate has a
yellow ring and a neutral plate, and is rejected.

**Small plates abstain.**  A DC block is 8x8 pixels, so a plate only a few
blocks wide is heavily contaminated by the bodywork around it.  Below
:data:`MIN_BLOCKS_FOR_COLOUR` blocks the reading is reported as unreliable
rather than used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from src.jpeg_dc import DcImage, JpegCorrupt, JpegUnsupported, decode_dc, ycbcr_to_rgb

logger = logging.getLogger(__name__)

#: Chroma value meaning "no colour".
NEUTRAL: Final[float] = 128.0

#: A plate must be at least this many DC blocks wide for colour to be trusted.
MIN_BLOCKS_FOR_COLOUR: Final[int] = 3

#: Fraction of the box sampled, centred -- trims edge blocks that mix in the
#: surrounding bodywork.
INNER_FRACTION: Final[float] = 0.8

#: How far the box is grown to sample the surrounding ring.
RING_SCALE: Final[float] = 2.2

#: Yellow has a markedly low Cb and a raised Cr. Both must hold.
YELLOW_MAX_CB: Final[float] = 115.0
YELLOW_MIN_CR: Final[float] = 133.0

#: Minimum brightness -- a near-black region has no reliable hue.
YELLOW_MIN_LUMA: Final[float] = 55.0

#: How much more yellow the plate must be than its surroundings.
MIN_YELLOW_MARGIN: Final[float] = 12.0

#: Yellowness at or above this, with the margin met, is called "strong".
STRONG_YELLOWNESS: Final[float] = 45.0


@dataclass(frozen=True)
class ColourReading:
    """Mean colour of one region, in both YCbCr and RGB."""

    luma: float
    blue_diff: float
    red_diff: float

    @property
    def rgb(self) -> tuple[int, int, int]:
        return ycbcr_to_rgb(self.luma, self.blue_diff, self.red_diff)

    @property
    def hex_colour(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)

    @property
    def yellowness(self) -> float:
        """How far towards yellow this colour sits.

        Yellow is the one hue that pushes Cb down and Cr up at the same time,
        so summing both deviations separates it from blue (Cb up), red (Cr up
        only) and neutral greys (neither).
        """
        return (NEUTRAL - self.blue_diff) + (self.red_diff - NEUTRAL)


@dataclass(frozen=True)
class PlateColour:
    """The verdict on one annotated plate's colour."""

    plate: ColourReading | None
    surround: ColourReading | None
    blocks_wide: int
    reliable: bool
    note: str

    @property
    def margin(self) -> float:
        if self.plate is None or self.surround is None:
            return 0.0
        return self.plate.yellowness - self.surround.yellowness

    def is_yellow_candidate(self) -> bool:
        """Whether this plate is worth a human look as a possible `type1b`."""
        if not self.reliable or self.plate is None:
            return False
        return (
            self.plate.luma >= YELLOW_MIN_LUMA
            and self.plate.blue_diff <= YELLOW_MAX_CB
            and self.plate.red_diff >= YELLOW_MIN_CR
            and self.margin >= MIN_YELLOW_MARGIN
        )

    def strength(self) -> str:
        if not self.is_yellow_candidate():
            return "none"
        assert self.plate is not None
        if self.plate.yellowness >= STRONG_YELLOWNESS:
            return "strong"
        return "weak"


def _clamp_box(
    centre_x: float, centre_y: float, width: float, height: float, scale: float
) -> tuple[float, float, float, float]:
    half_w = width * scale / 2
    half_h = height * scale / 2
    return (
        max(0.0, centre_x - half_w),
        max(0.0, centre_y - half_h),
        min(1.0, centre_x + half_w),
        min(1.0, centre_y + half_h),
    )


def _reading(image: DcImage, box: tuple[float, float, float, float]) -> ColourReading | None:
    sampled = image.sample_ycbcr(*box)
    if sampled is None:
        return None
    return ColourReading(*sampled)


def sample_plate_colour(
    image: DcImage,
    centre_x: float,
    centre_y: float,
    norm_width: float,
    norm_height: float,
) -> PlateColour:
    """Measure a plate's colour and that of the ring around it."""
    blocks_wide = int(norm_width * image.components[0].blocks_wide)

    inner = _clamp_box(centre_x, centre_y, norm_width, norm_height, INNER_FRACTION)
    outer = _clamp_box(centre_x, centre_y, norm_width, norm_height, RING_SCALE)

    plate = _reading(image, inner)
    outer_reading = _reading(image, outer)

    if plate is None:
        return PlateColour(None, None, blocks_wide, False, "plate region not decoded")

    if not image.is_colour:
        return PlateColour(
            plate, None, blocks_wide, False, "greyscale image carries no colour"
        )

    surround = _ring_colour(plate, outer_reading, norm_width, norm_height)

    if blocks_wide < MIN_BLOCKS_FOR_COLOUR:
        return PlateColour(
            plate,
            surround,
            blocks_wide,
            False,
            f"plate spans only {blocks_wide} colour block(s); too small to judge",
        )

    return PlateColour(plate, surround, blocks_wide, True, "")


def _ring_colour(
    plate: ColourReading,
    outer: ColourReading | None,
    norm_width: float,
    norm_height: float,
) -> ColourReading | None:
    """Back out the ring's colour from the plate and the enclosing box.

    The outer sample includes the plate, so the plate's contribution is removed
    by area weighting; otherwise a yellow plate would make its own surroundings
    look yellow and defeat the guard.
    """
    if outer is None:
        return None

    plate_area = norm_width * norm_height * INNER_FRACTION**2
    outer_area = norm_width * norm_height * RING_SCALE**2
    ring_area = outer_area - plate_area
    if ring_area <= 0:
        return None

    def unmix(outer_value: float, plate_value: float) -> float:
        return (outer_value * outer_area - plate_value * plate_area) / ring_area

    return ColourReading(
        unmix(outer.luma, plate.luma),
        unmix(outer.blue_diff, plate.blue_diff),
        unmix(outer.red_diff, plate.red_diff),
    )


def read_plate_colour(
    image_path: Path,
    centre_x: float,
    centre_y: float,
    norm_width: float,
    norm_height: float,
) -> PlateColour:
    """Decode ``image_path`` far enough to measure its plate colour.

    Returns an unreliable :class:`PlateColour` -- never an exception and never
    a guess -- when the file cannot be decoded.
    """
    try:
        data = Path(image_path).read_bytes()
    except OSError as error:
        return PlateColour(None, None, 0, False, f"unreadable: {error.strerror or error}")

    if not data.startswith(b"\xff\xd8"):
        return PlateColour(
            None, None, 0, False, "colour sampling supports baseline JPEG only"
        )

    bottom = min(1.0, centre_y + norm_height * RING_SCALE / 2)
    try:
        image = decode_dc(data, region_bottom=bottom)
    except (JpegUnsupported, JpegCorrupt) as error:
        return PlateColour(None, None, 0, False, f"not decoded: {error}")

    return sample_plate_colour(image, centre_x, centre_y, norm_width, norm_height)
