"""Font providers: how a character is drawn into a box.

The generator talks to a :class:`FontProvider`, never to a font file, so the
built-in :class:`StrokeFontProvider` can later be swapped for a real
plate-style TrueType font *once one with a verified redistributable licence
exists*.  No such font is bundled today; see ``README.md``.

A provider draws white ink onto a single-channel (``"L"``) mask; colour is
applied afterwards by the renderer.
"""

from __future__ import annotations

from typing import Protocol

from PIL import ImageDraw

from .config import ConfigError
from .glyphs import GLYPHS

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in canvas pixels


class FontProvider(Protocol):
    """Draws single characters into boxes on an ``"L"`` mask."""

    name: str
    license: str

    def supports(self, char: str) -> bool: ...

    def draw(self, draw: ImageDraw.ImageDraw, char: str, box: Box, stroke_px: float) -> None: ...


class StrokeFontProvider:
    """The built-in stroke font (:mod:`.glyphs`) -- our own work, CC BY 4.0."""

    name = "builtin-stroke"
    license = "CC BY 4.0 (own work, part of this generator)"

    def supports(self, char: str) -> bool:
        return char in GLYPHS

    def draw(self, draw: ImageDraw.ImageDraw, char: str, box: Box, stroke_px: float) -> None:
        glyph = GLYPHS.get(char)
        if glyph is None:
            raise KeyError(f"no glyph for {char!r}")
        x0, y0, x1, y1 = box
        stroke = max(1.0, float(stroke_px))
        half = stroke / 2.0
        span_x = max(0.0, (x1 - x0) - stroke)
        span_y = max(0.0, (y1 - y0) - stroke)
        width = max(1, int(round(stroke)))
        radius = width / 2.0

        for glyph_stroke in glyph:
            points = [(x0 + half + u * span_x, y0 + half + v * span_y) for u, v in glyph_stroke]
            if len(points) > 1:
                draw.line(points, fill=255, width=width, joint="curve")
            # Round caps and joins: PIL's line ends are square-cut and its
            # "curve" joint misses the stroke's own end points.
            for px, py in (points[0], points[-1]):
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)


class TrueTypeFontProvider:  # pragma: no cover - exercised only with a vetted font
    """Placeholder for a future vetted plate font.

    Deliberately refuses to construct: bundling a font requires its licence
    to be verified and registered in ``docs/data_sources.md`` first, and no
    such font has been cleared yet.  Implement drawing here only together with
    that registration.
    """

    def __init__(self, path: str, license: str) -> None:
        raise ConfigError(
            "no TrueType plate font has been cleared for redistribution; "
            "register one in docs/data_sources.md before enabling this provider"
        )


def get_font_provider(name: str) -> FontProvider:
    if name == "stroke":
        return StrokeFontProvider()
    raise ConfigError(f"unknown font provider {name!r}")
