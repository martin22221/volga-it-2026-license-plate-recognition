"""Plate geometry templates, in millimetres.

What is taken from GOST R 50577-2018 (text of sections 3.2, 3.3, 3.8 and
table 1, read 2026-09-13):

* type 1: 520 x 112 mm, white field, one line ``M 000 MM 55`` / ``... 555``;
* type 1A: two lines, ``M 000`` above ``MM 55`` / ``MM 555``, white field;
  the 290 x 170 mm size is the commonly cited figure for this type;
* type 1B: 520 x 112 mm, *yellow* field, one line;
* permitted character heights include 58 mm and 76 mm, with minimum stroke
  widths 9.0 mm and 11.0 mm respectively; border width 3.0 +/- 0.5 mm.

What is **approximated**, because the dimensioned figures A.1-A.5 were not
accessible: character widths, gaps, the exact baseline positions, the RUS
inscription and flag placement, and the corner radius.  Every such value is a
named constant below, so it can be corrected against a measured plate without
touching the renderer.

``type1a`` is laid out natively on two lines; it is never a squashed
``type1``.  ``type1b`` uses the ``type1`` geometry with a yellow field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .plate_text import PlateText

# ---------------------------------------------------------------- dimensions


@dataclass(frozen=True)
class CharMetrics:
    """Size of one character box and its stroke, in mm."""

    width: float
    height: float
    stroke: float


#: Main-field letters and digits (GOST heights 58 / 76 mm, strokes 9 / 11 mm).
LETTER: Final[CharMetrics] = CharMetrics(width=49.0, height=58.0, stroke=9.0)
DIGIT: Final[CharMetrics] = CharMetrics(width=49.0, height=76.0, stroke=11.0)
#: Region digits: 58 mm tall; narrower when the region has three digits.
REGION_2: Final[CharMetrics] = CharMetrics(width=40.0, height=58.0, stroke=9.0)
REGION_3: Final[CharMetrics] = CharMetrics(width=33.0, height=58.0, stroke=8.0)


@dataclass(frozen=True)
class OneLineTemplate:
    """Geometry shared by the one-line layouts (``type1``, ``type1b``)."""

    width: float = 520.0
    height: float = 112.0
    corner_radius: float = 9.0
    border_inset: float = 2.0
    border_width: float = 3.0
    #: Main field (series + number) spans this horizontal interval.
    main_x0: float = 8.0
    main_x1: float = 390.0
    #: Bottom edge shared by letters and digits (they are bottom-aligned).
    baseline: float = 94.0
    group_gap: float = 20.0
    char_gap: float = 9.0
    #: Vertical rule separating the region field.
    separator_x: float = 392.0
    separator_width: float = 2.0
    region_x1: float = 515.0
    region_top: float = 12.0
    region_gap_2: float = 8.0
    region_gap_3: float = 5.0
    #: "RUS" and flag row inside the region field.
    rus_top: float = 80.0
    rus_height: float = 15.0
    rus_char_width: float = 11.0
    rus_gap: float = 3.0
    rus_stroke: float = 2.6
    flag_width: float = 25.0
    flag_height: float = 15.0
    flag_gap: float = 5.0


@dataclass(frozen=True)
class TwoLineTemplate:
    """Geometry of the square ``type1a`` layout."""

    width: float = 290.0
    height: float = 170.0
    corner_radius: float = 10.0
    border_inset: float = 2.0
    border_width: float = 3.0
    #: Top line: letter + serial number, bottom-aligned on this baseline.
    top_baseline: float = 84.0
    top_letter_gap: float = 14.0
    #: Bottom line: two letters, region, then a flag-over-RUS column.
    bottom_top: float = 97.0
    bottom_letter_to_region_gap: float = 14.0
    char_gap: float = 9.0
    region_gap_2: float = 7.0
    region_gap_3: float = 5.0
    column_gap: float = 6.0
    column_width: float = 24.0
    flag_height: float = 14.0
    rus_height: float = 12.0
    rus_gap_below_flag: float = 30.0
    rus_stroke: float = 2.2


ONE_LINE: Final[OneLineTemplate] = OneLineTemplate()
TWO_LINE: Final[TwoLineTemplate] = TwoLineTemplate()

#: Physical plate size, per plate type.
PLATE_SIZE_MM: Final[dict[str, tuple[float, float]]] = {
    "type1": (ONE_LINE.width, ONE_LINE.height),
    "type1a": (TWO_LINE.width, TWO_LINE.height),
    "type1b": (ONE_LINE.width, ONE_LINE.height),
}

#: Field colour role per plate type; the renderer picks the actual shade.
FIELD_COLOUR: Final[dict[str, str]] = {"type1": "white", "type1a": "white", "type1b": "yellow"}


# ------------------------------------------------------------------ layouts


@dataclass(frozen=True)
class GlyphBox:
    """One character placed on the plate.

    ``position`` indexes the character in :attr:`PlateText.full`; it is
    ``None`` for the RUS inscription, which is not part of the label.
    """

    char: str
    x: float
    y: float
    width: float
    height: float
    stroke: float
    position: int | None
    line: int

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class PlateLayout:
    """Everything the renderer draws, positioned in plate millimetres."""

    plate_type: str
    width: float
    height: float
    corner_radius: float
    border_inset: float
    border_width: float
    field_colour: str
    glyphs: tuple[GlyphBox, ...]
    separators: tuple[Rect, ...]
    flag: Rect | None

    @property
    def label_glyphs(self) -> tuple[GlyphBox, ...]:
        return tuple(glyph for glyph in self.glyphs if glyph.position is not None)

    @property
    def line_count(self) -> int:
        return len({glyph.line for glyph in self.label_glyphs})


def _metrics(role: str, region_length: int) -> CharMetrics:
    if role == "letter":
        return LETTER
    if role == "digit":
        return DIGIT
    return REGION_2 if region_length <= 2 else REGION_3


def _place_row(
    chars: list[tuple[int | None, str, CharMetrics, float]],
    x_start: float,
    *,
    baseline: float | None = None,
    top: float | None = None,
    line: int,
) -> list[GlyphBox]:
    """Place ``(position, char, metrics, gap_before)`` left to right."""
    boxes: list[GlyphBox] = []
    x = x_start
    for index, (position, char, metrics, gap) in enumerate(chars):
        if index:
            x += gap
        y = baseline - metrics.height if baseline is not None else top  # type: ignore[operator]
        boxes.append(GlyphBox(char, x, float(y), metrics.width, metrics.height, metrics.stroke, position, line))
        x += metrics.width
    return boxes


def _row_width(chars: list[tuple[int | None, str, CharMetrics, float]]) -> float:
    return sum(m.width for _, _, m, _ in chars) + sum(g for i, (_, _, _, g) in enumerate(chars) if i)


def _rus_boxes(x: float, y: float, height: float, char_width: float, gap: float, stroke: float, line: int) -> list[GlyphBox]:
    return [
        GlyphBox(char, x + i * (char_width + gap), y, char_width, height, stroke, None, line)
        for i, char in enumerate("RUS")
    ]


def layout_one_line(text: PlateText, plate_type: str, t: OneLineTemplate = ONE_LINE) -> PlateLayout:
    """``type1`` / ``type1b``: main field, separator, region field."""
    main: list[tuple[int | None, str, CharMetrics, float]] = []
    position = 0
    for group_index, (chars, kind) in enumerate(text.main_groups()):
        for char_index, char in enumerate(chars):
            gap = t.group_gap if (char_index == 0 and group_index) else t.char_gap
            main.append((position, char, _metrics(kind, len(text.region)), gap))
            position += 1
    main_width = _row_width(main)
    x_start = t.main_x0 + ((t.main_x1 - t.main_x0) - main_width) / 2.0
    glyphs = _place_row(main, x_start, baseline=t.baseline, line=0)

    region_metrics = _metrics("region", len(text.region))
    region_gap = t.region_gap_2 if len(text.region) <= 2 else t.region_gap_3
    region = [(position + i, char, region_metrics, region_gap) for i, char in enumerate(text.region)]
    field_x0 = t.separator_x + t.separator_width
    region_x = field_x0 + ((t.region_x1 - field_x0) - _row_width(region)) / 2.0
    glyphs += _place_row(region, region_x, top=t.region_top, line=0)

    rus_width = 3 * t.rus_char_width + 2 * t.rus_gap
    row_width = rus_width + t.flag_gap + t.flag_width
    rus_x = field_x0 + ((t.region_x1 - field_x0) - row_width) / 2.0
    glyphs += _rus_boxes(rus_x, t.rus_top, t.rus_height, t.rus_char_width, t.rus_gap, t.rus_stroke, 0)
    flag = Rect(rus_x + rus_width + t.flag_gap, t.rus_top, t.flag_width, t.flag_height)

    inner_top = t.border_inset + t.border_width
    separator = Rect(t.separator_x, inner_top, t.separator_width, t.height - 2 * inner_top)
    return PlateLayout(
        plate_type=plate_type,
        width=t.width,
        height=t.height,
        corner_radius=t.corner_radius,
        border_inset=t.border_inset,
        border_width=t.border_width,
        field_colour=FIELD_COLOUR[plate_type],
        glyphs=tuple(glyphs),
        separators=(separator,),
        flag=flag,
    )


def layout_two_line(text: PlateText, t: TwoLineTemplate = TWO_LINE) -> PlateLayout:
    """``type1a``: ``M 000`` on the top line, ``MM 55`` + flag/RUS below."""
    if text.text_format != "competition":
        raise ValueError("type1a is defined only for the competition format")

    top = [(0, text.series1, LETTER, 0.0)]
    top += [(1 + i, char, DIGIT, t.top_letter_gap if i == 0 else t.char_gap) for i, char in enumerate(text.number)]
    top_x = (t.width - _row_width(top)) / 2.0
    glyphs = _place_row(top, top_x, baseline=t.top_baseline, line=0)

    region_metrics = _metrics("region", len(text.region))
    region_gap = t.region_gap_2 if len(text.region) <= 2 else t.region_gap_3
    bottom = [(4 + i, char, LETTER, t.char_gap) for i, char in enumerate(text.series2)]
    bottom += [
        (6 + i, char, region_metrics, t.bottom_letter_to_region_gap if i == 0 else region_gap)
        for i, char in enumerate(text.region)
    ]
    bottom_width = _row_width(bottom) + t.column_gap + t.column_width
    bottom_x = (t.width - bottom_width) / 2.0
    glyphs += _place_row(bottom, bottom_x, top=t.bottom_top, line=1)

    column_x = bottom_x + _row_width(bottom) + t.column_gap
    # Letters and region digits share the 58 mm height, so this row is flat.
    flag = Rect(column_x, t.bottom_top + 4.0, t.column_width, t.flag_height)
    rus_char = (t.column_width - 2 * 1.5) / 3.0
    glyphs += _rus_boxes(
        column_x, flag.y + t.flag_height + t.rus_gap_below_flag - t.rus_height,
        t.rus_height, rus_char, 1.5, t.rus_stroke, 1,
    )
    return PlateLayout(
        plate_type="type1a",
        width=t.width,
        height=t.height,
        corner_radius=t.corner_radius,
        border_inset=t.border_inset,
        border_width=t.border_width,
        field_colour=FIELD_COLOUR["type1a"],
        glyphs=tuple(glyphs),
        separators=(),
        flag=flag,
    )


def build_layout(plate_type: str, text: PlateText) -> PlateLayout:
    """The layout for ``plate_type`` -- each type has its own geometry."""
    if plate_type in ("type1", "type1b"):
        return layout_one_line(text, plate_type)
    if plate_type == "type1a":
        return layout_two_line(text)
    raise ValueError(f"unsupported plate type {plate_type!r}")
