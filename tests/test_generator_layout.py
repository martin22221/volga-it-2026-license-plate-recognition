"""Plate templates, glyphs and rendering: type1 / type1a / type1b geometry."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dataset.generator.fonts import StrokeFontProvider, TrueTypeFontProvider, get_font_provider
from dataset.generator.config import ConfigError
from dataset.generator.glyphs import GLYPHS, PLATE_CHARACTERS
from dataset.generator.plate_text import PlateText
from dataset.generator.render import PlateStyle, render_plate, sample_style
from dataset.generator.templates import DIGIT, LETTER, ONE_LINE, PLATE_SIZE_MM, TWO_LINE, build_layout

TYPE1 = PlateText("A", "123", "BC", "77", "competition")
TYPE1_3 = PlateText("M", "908", "XY", "799", "competition")
FONT = StrokeFontProvider()


def _inside(glyph, layout, margin: float = 0.0) -> bool:
    inner = layout.border_inset + layout.border_width - margin
    return (
        glyph.x >= inner
        and glyph.y >= inner
        and glyph.x1 <= layout.width - inner
        and glyph.y1 <= layout.height - inner
    )


# ------------------------------------------------------------------ glyphs


def test_every_plate_character_and_rus_has_a_glyph() -> None:
    for char in PLATE_CHARACTERS + "RUS":
        assert char in GLYPHS, char
        assert FONT.supports(char)


def test_glyph_points_lie_in_the_unit_box() -> None:
    for char, glyph in GLYPHS.items():
        for stroke in glyph:
            assert len(stroke) >= 2, char
            for u, v in stroke:
                assert -1e-6 <= u <= 1 + 1e-6 and -1e-6 <= v <= 1 + 1e-6, (char, u, v)


@pytest.mark.parametrize("char", list(PLATE_CHARACTERS))
def test_glyph_ink_stays_inside_its_box(char: str) -> None:
    mask = Image.new("L", (200, 200), 0)
    FONT.draw(ImageDraw.Draw(mask), char, (50, 40, 110, 160), stroke_px=10)
    ink = np.asarray(mask) > 0
    assert ink.sum() > 300, char
    ys, xs = np.nonzero(ink)
    assert xs.min() >= 49 and xs.max() <= 111 and ys.min() >= 39 and ys.max() <= 161, char


def test_glyphs_are_distinct() -> None:
    def raster(char: str) -> np.ndarray:
        mask = Image.new("L", (60, 90), 0)
        FONT.draw(ImageDraw.Draw(mask), char, (5, 5, 55, 85), stroke_px=7)
        return np.asarray(mask) > 127

    rasters = {char: raster(char) for char in PLATE_CHARACTERS}
    for a in PLATE_CHARACTERS:
        for b in PLATE_CHARACTERS:
            if a < b:
                assert (rasters[a] != rasters[b]).sum() > 50, (a, b)


def test_font_provider_registry_and_truetype_guard() -> None:
    assert isinstance(get_font_provider("stroke"), StrokeFontProvider)
    with pytest.raises(ConfigError):
        get_font_provider("some-downloaded-font")
    with pytest.raises(ConfigError):
        TrueTypeFontProvider("plate.ttf", "unknown")


# ------------------------------------------------------------------ type1


@pytest.mark.parametrize("text", [TYPE1, TYPE1_3])
def test_type1_is_a_single_line_with_region_field(text: PlateText) -> None:
    layout = build_layout("type1", text)
    assert (layout.width, layout.height) == (520.0, 112.0)
    assert layout.line_count == 1
    labels = sorted(layout.label_glyphs, key=lambda g: g.position)
    assert "".join(g.char for g in labels) == text.full
    # Reading order is left to right.
    assert [g.position for g in sorted(labels, key=lambda g: g.x)] == list(range(len(text.full)))
    main = [g for g in labels if g.position < 6]
    region = [g for g in labels if g.position >= 6]
    # Letters and digits are bottom-aligned; digits taller than letters (GOST 76 / 58 mm).
    assert {round(g.y1, 6) for g in main} == {ONE_LINE.baseline}
    assert {g.height for g in main if g.char.isdigit()} == {DIGIT.height}
    assert {g.height for g in main if g.char.isalpha()} == {LETTER.height}
    # Region sits right of the separator.
    separator = layout.separators[0]
    assert all(g.x > separator.x + separator.width for g in region)
    assert all(g.x1 < separator.x for g in main)
    assert all(_inside(g, layout) for g in layout.glyphs)
    assert layout.field_colour == "white"
    assert layout.flag is not None


def test_type1_boxes_do_not_overlap() -> None:
    for text in (TYPE1, TYPE1_3):
        boxes = sorted(build_layout("type1", text).glyphs, key=lambda g: (g.line, g.y, g.x))
        for a in boxes:
            for b in boxes:
                if a is not b and abs(a.y - b.y) < 1e-6:
                    assert a.x1 <= b.x or b.x1 <= a.x, (a.char, b.char)


# ------------------------------------------------------------------ type1a


@pytest.mark.parametrize("text", [TYPE1, TYPE1_3])
def test_type1a_is_a_genuine_two_line_layout(text: PlateText) -> None:
    layout = build_layout("type1a", text)
    assert (layout.width, layout.height) == (290.0, 170.0)
    assert layout.line_count == 2
    top = [g for g in layout.label_glyphs if g.line == 0]
    bottom = [g for g in layout.label_glyphs if g.line == 1]
    # GOST 3.3: "M 000" above "MM 55" / "MM 555".
    assert "".join(g.char for g in sorted(top, key=lambda g: g.x)) == text.series1 + text.number
    assert "".join(g.char for g in sorted(bottom, key=lambda g: g.x)) == text.series2 + text.region
    assert max(g.y1 for g in top) < min(g.y for g in bottom)
    assert all(_inside(g, layout) for g in layout.glyphs)
    assert layout.field_colour == "white"


def test_type1a_is_not_a_squashed_type1() -> None:
    one = build_layout("type1", TYPE1)
    two = build_layout("type1a", TYPE1)
    # Characters keep their physical size; only the arrangement changes.
    heights_one = sorted(g.height for g in one.label_glyphs if g.position < 6)
    heights_two = sorted(g.height for g in two.label_glyphs if g.position < 6)
    assert heights_one == heights_two
    widths_one = sorted(g.width for g in one.label_glyphs if g.position < 6)
    widths_two = sorted(g.width for g in two.label_glyphs if g.position < 6)
    assert widths_one == widths_two
    assert two.width / two.height < 2.0 < one.width / one.height


def test_type1a_rejects_gost_1b_text() -> None:
    with pytest.raises(ValueError):
        build_layout("type1a", PlateText("AB", "123", "", "77", "gost_1b"))


# ------------------------------------------------------------------ type1b


def test_type1b_uses_one_line_geometry_with_yellow_field() -> None:
    layout = build_layout("type1b", TYPE1)
    assert layout.field_colour == "yellow"
    assert (layout.width, layout.height) == (520.0, 112.0)
    assert layout.line_count == 1
    assert PLATE_SIZE_MM["type1b"] == PLATE_SIZE_MM["type1"]


def test_type1b_gost_format_lays_out_letters_first() -> None:
    text = PlateText("AB", "123", "", "77", "gost_1b")
    layout = build_layout("type1b", text)
    labels = sorted(layout.label_glyphs, key=lambda g: g.x)
    assert "".join(g.char for g in labels) == "AB12377"


def _field_colour(plate_type: str, seed: int) -> np.ndarray:
    layout = build_layout(plate_type, TYPE1)
    style = sample_style(np.random.default_rng(seed), layout.field_colour)
    plate = render_plate(layout, style, FONT, 1.0, supersample=2)
    field = (plate.ink < 0.02) & (plate.alpha > 0.99)
    # Exclude the flag (the only coloured artwork) by its location.
    if layout.flag is not None:
        f = layout.flag
        field[int(f.y) - 2 : int(f.y + f.height) + 3, int(f.x) - 2 : int(f.x + f.width) + 3] = False
    return plate.rgb[field].mean(axis=0)


@pytest.mark.parametrize("seed", range(8))
def test_type1b_renders_a_genuinely_yellow_field(seed: int) -> None:
    r, g, b = _field_colour("type1b", seed)
    assert r > 0.85 and 0.6 < g < 0.9 and b < 0.2, (r, g, b)
    assert r - b > 0.7


@pytest.mark.parametrize("plate_type", ["type1", "type1a"])
@pytest.mark.parametrize("seed", range(4))
def test_white_types_render_a_neutral_white_field(plate_type: str, seed: int) -> None:
    r, g, b = _field_colour(plate_type, seed)
    assert min(r, g, b) > 0.8 and max(r, g, b) - min(r, g, b) < 0.06, (r, g, b)


# ------------------------------------------------------------------ rendering


def test_rendered_plate_fills_its_canvas_exactly() -> None:
    layout = build_layout("type1", TYPE1)
    style = PlateStyle((0.95, 0.95, 0.95), (0.05, 0.05, 0.05), 1.0, 0.0)
    plate = render_plate(layout, style, FONT, 0.5, supersample=2)
    assert (plate.width, plate.height) == (260, 56)
    assert plate.rgb.shape == (56, 260, 3)
    # Opaque edges (bar the rounded corners): the plate touches every side.
    assert plate.alpha[28, 0] > 0.9 and plate.alpha[28, -1] > 0.9
    assert plate.alpha[0, 130] > 0.9 and plate.alpha[-1, 130] > 0.9
    assert len(plate.glyph_boxes) == len(TYPE1.full)


def test_glyph_boxes_contain_their_ink() -> None:
    layout = build_layout("type1a", TYPE1_3)
    style = PlateStyle((0.95, 0.95, 0.95), (0.05, 0.05, 0.05), 1.0, 0.0)
    plate = render_plate(layout, style, FONT, 1.0, supersample=2)
    for _position, _char, x0, y0, x1, y1 in plate.glyph_boxes:
        box_ink = plate.ink[int(y0) : int(np.ceil(y1)), int(x0) : int(np.ceil(x1))]
        assert box_ink.mean() > 0.1
