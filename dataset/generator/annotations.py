"""Annotations in the ``dataset/meta.csv`` schema.

The column list and the condition vocabulary are duplicated from
``src/dataset_meta.py`` so the generator runs without the rest of the
repository; ``tests/test_generator_output.py`` fails if they drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from . import OUTPUT_LICENSE, SOURCE_ID

META_COLUMNS: Final[tuple[str, ...]] = (
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

CONDITION_TAGS: Final[tuple[str, ...]] = (
    "day",
    "night",
    "rain",
    "snow",
    "dirt",
    "glare",
    "motion_blur",
    "angle",
)

#: ``angle`` is tagged beyond this rotation, per the annotation guide (~15 deg).
ANGLE_TAG_DEGREES: Final[float] = 15.0
#: ``motion_blur`` is tagged from this blur length, in output pixels.
MOTION_BLUR_TAG_PX: Final[float] = 3.0
#: ``dirt`` is tagged from this dirt strength.
DIRT_TAG_STRENGTH: Final[float] = 0.25

COORD_DECIMALS: Final[int] = 2
UNREADABLE_CHAR: Final[str] = "#"


@dataclass(frozen=True)
class Annotation:
    """One plate on one generated image."""

    image: str
    plate_num: str
    plate_type: str
    quad: tuple[tuple[float, float], ...]
    conditions: tuple[str, ...]
    is_vehicle: bool = True
    is_synthetic: bool = True
    source: str = SOURCE_ID
    license: str = OUTPUT_LICENSE

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return bbox_from_quad(self.quad)

    def meta_row(self) -> dict[str, str]:
        x, y, w, h = self.bbox
        row = {
            "image": self.image,
            "plate_num": self.plate_num,
            "plate_type": self.plate_type,
            "bbox_x": _fmt(x),
            "bbox_y": _fmt(y),
            "bbox_w": _fmt(w),
            "bbox_h": _fmt(h),
            "is_vehicle": "true" if self.is_vehicle else "false",
            "is_synthetic": "true" if self.is_synthetic else "false",
            "source": self.source,
            "license": self.license,
            "conditions": "|".join(self.conditions),
        }
        for index, (qx, qy) in enumerate(self.quad, start=1):
            row[f"quad_x{index}"] = _fmt(qx)
            row[f"quad_y{index}"] = _fmt(qy)
        return {column: row[column] for column in META_COLUMNS}


def _fmt(value: float) -> str:
    return f"{value:.{COORD_DECIMALS}f}"


def round_quad(quad: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x), COORD_DECIMALS), round(float(y), COORD_DECIMALS)) for x, y in quad)


def bbox_from_quad(quad: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Tightest axis-aligned box around the quad: ``(x, y, w, h)``."""
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0, y0 = min(xs), min(ys)
    return (
        round(x0, COORD_DECIMALS),
        round(y0, COORD_DECIMALS),
        round(max(xs) - x0, COORD_DECIMALS),
        round(max(ys) - y0, COORD_DECIMALS),
    )


def masked_plate_number(full: str, hidden_positions: set[int] | frozenset[int]) -> str:
    """``full`` with every hidden character replaced by ``#``."""
    return "".join(UNREADABLE_CHAR if i in hidden_positions else ch for i, ch in enumerate(full))


def derive_conditions(
    *,
    night: bool,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    effects: dict[str, dict],
) -> tuple[str, ...]:
    """Condition tags for what is actually visible, in vocabulary order."""
    tags = {"night" if night else "day"}
    if max(abs(yaw_deg), abs(pitch_deg), abs(roll_deg)) > ANGLE_TAG_DEGREES:
        tags.add("angle")
    if "dirt" in effects and effects["dirt"]["strength"] >= DIRT_TAG_STRENGTH:
        tags.add("dirt")
    if "glare" in effects and effects["glare"]["on_plate"]:
        tags.add("glare")
    if "motion_blur" in effects and effects["motion_blur"]["length_px"] >= MOTION_BLUR_TAG_PX:
        tags.add("motion_blur")
    for weather in ("rain", "snow"):
        if weather in effects:
            tags.add(weather)
    return tuple(tag for tag in CONDITION_TAGS if tag in tags)
