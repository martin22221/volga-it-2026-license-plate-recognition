"""Font providers: how a character is drawn into a box.

The generator talks to a :class:`FontProvider`, never to a font file, so the
built-in :class:`StrokeFontProvider` can later be swapped for a real
plate-style TrueType font *once one with a verified redistributable licence
exists*.  No such font is bundled today; see ``README.md``.

A provider draws white ink onto a single-channel (``"L"``) image; colour is
applied afterwards by the renderer.
"""

from __future__ import annotations

import math
from typing import Protocol

from PIL import Image, ImageChops, ImageDraw

from .config import ConfigError
from .glyphs import GLYPHS

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in canvas pixels

#: A stroke end within this distance of the unit box edge counts as touching it.
EDGE_TOLERANCE = 0.02


class FontProvider(Protocol):
    """Draws single characters into boxes on an ``"L"`` image."""

    name: str
    license: str

    def supports(self, char: str) -> bool: ...

    def draw(self, target: Image.Image, char: str, box: Box, stroke_px: float) -> None: ...


def _on_edge(u: float, v: float) -> bool:
    return min(u, v) <= EDGE_TOLERANCE or max(u, v) >= 1.0 - EDGE_TOLERANCE


def _extend(points: list[tuple[float, float]], at_start: bool, length: float) -> None:
    """Push the first or last point outwards along its segment."""
    if len(points) < 2:
        return
    (ax, ay), (bx, by) = (points[1], points[0]) if at_start else (points[-2], points[-1])
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy) or 1.0
    extended = (bx + dx / norm * length, by + dy / norm * length)
    if at_start:
        points[0] = extended
    else:
        points[-1] = extended


class StrokeFontProvider:
    """The built-in stroke font (:mod:`.glyphs`) -- our own work, CC BY 4.0.

    Stroke ends that reach the character cell's edge are extended past it and
    clipped by the cell, which gives the straight horizontal/vertical cuts of
    the plate typeface (e.g. the feet of ``A`` and ``K``) rather than rounded
    ends.  Ends inside the cell (hooks of ``C``, ``S``, ``2``...) and joins
    keep a round cap.
    """

    name = "builtin-stroke-v2"
    license = "CC BY 4.0 (own work, part of this generator)"

    def supports(self, char: str) -> bool:
        return char in GLYPHS

    def draw(self, target: Image.Image, char: str, box: Box, stroke_px: float) -> None:
        glyph = GLYPHS.get(char)
        if glyph is None:
            raise KeyError(f"no glyph for {char!r}")
        x0, y0, x1, y1 = box
        stroke = max(1.0, float(stroke_px))
        half = stroke / 2.0
        width = max(1, int(round(stroke)))
        radius = width / 2.0

        # Draw into a local canvas covering the cell, then clip to the cell.
        cx0, cy0 = int(math.floor(x0)), int(math.floor(y0))
        cx1, cy1 = int(math.ceil(x1)), int(math.ceil(y1))
        pad = width + 2
        local = Image.new("L", (cx1 - cx0 + 2 * pad, cy1 - cy0 + 2 * pad), 0)
        draw = ImageDraw.Draw(local)
        span_x = max(0.0, (x1 - x0) - stroke)
        span_y = max(0.0, (y1 - y0) - stroke)
        ox, oy = x0 - cx0 + pad, y0 - cy0 + pad

        for glyph_stroke in glyph:
            points = [(ox + half + u * span_x, oy + half + v * span_y) for u, v in glyph_stroke]
            closed = len(glyph_stroke) > 2 and glyph_stroke[0] == glyph_stroke[-1]
            ends = [] if closed else [(True, glyph_stroke[0]), (False, glyph_stroke[-1])]
            for at_start, (u, v) in ends:
                if _on_edge(u, v):
                    _extend(points, at_start, stroke)
            if len(points) > 1:
                draw.line(points, fill=255, width=width, joint="curve")
            for at_start, (u, v) in ends:
                if not _on_edge(u, v):
                    px, py = points[0] if at_start else points[-1]
                    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)
            if closed:
                px, py = points[0]
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)

        cell = local.crop((pad, pad, pad + cx1 - cx0, pad + cy1 - cy0))
        clip = Image.new("L", cell.size, 0)
        ImageDraw.Draw(clip).rectangle((x0 - cx0, y0 - cy0, x1 - cx0 - 1, y1 - cy0 - 1), fill=255)
        cell = ImageChops.multiply(cell, clip)
        region = target.crop((cx0, cy0, cx1, cy1))
        target.paste(ImageChops.lighter(region, cell), (cx0, cy0))


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
