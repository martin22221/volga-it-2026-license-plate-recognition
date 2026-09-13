"""Built-in stroke font for plate characters -- original work of this project.

Each glyph is a set of strokes: centre-line polylines in a unit box, with
``u`` running right and ``v`` running down, both in ``[0, 1]``.  The renderer
insets the box by half the stroke width, so ink never leaves the character
box it is given.

The shapes are drawn from scratch to *resemble* the condensed, uniform-stroke
lettering on Russian plates (GOST R 50577 appendices B and C define it); they
are not traced from, and contain no data from, any font file or drawing.
That is the point: no third-party font means no redistribution question.

Covered: the ten digits, the twelve plate letters ``A B E K M H O P C T Y X``
(``Y`` is drawn as the Cyrillic ``У`` it stands for), and ``R U S`` for the
``RUS`` inscription.
"""

from __future__ import annotations

import math
from typing import Final, Iterable, Union

Point = tuple[float, float]
Stroke = tuple[Point, ...]
Glyph = tuple[Stroke, ...]


def _arc(
    cx: float, cy: float, rx: float, ry: float, start_deg: float, end_deg: float
) -> list[Point]:
    """Points on an ellipse; angles grow clockwise on screen (``v`` is down).

    0 degrees is the right-most point, 90 the bottom, 180 the left, 270 the top.
    """
    span = end_deg - start_deg
    steps = max(6, int(abs(span) / 6.0))
    points = []
    for i in range(steps + 1):
        angle = math.radians(start_deg + span * i / steps)
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    return points


def _stroke(*parts: Union[Point, Iterable[Point]]) -> Stroke:
    """Join points and point lists into one stroke, dropping repeats."""
    points: list[Point] = []
    for part in parts:
        items = [part] if isinstance(part[0], (int, float)) else list(part)  # type: ignore[index]
        for point in items:
            point = (round(float(point[0]), 4), round(float(point[1]), 4))
            if not points or points[-1] != point:
                points.append(point)
    return tuple(points)


def _stadium(top: float, bottom: float, cap: float) -> Stroke:
    """Closed rounded-rectangle outline filling the box width."""
    return _stroke(
        _arc(0.5, top + cap, 0.5, cap, 180, 360),
        (1.0, bottom - cap),
        _arc(0.5, bottom - cap, 0.5, cap, 0, 180),
        (0.0, top + cap),
    )


def _build() -> dict[str, Glyph]:
    g: dict[str, Glyph] = {}

    # ---- digits --------------------------------------------------------
    g["0"] = (_stadium(0.0, 1.0, 0.30),)
    g["1"] = (_stroke((0.12, 0.24), (0.64, 0.0), (0.64, 1.0)),)
    g["2"] = (_stroke(_arc(0.5, 0.28, 0.5, 0.28, 195, 380), (0.0, 1.0), (1.0, 1.0)),)
    g["3"] = (
        _stroke(_arc(0.5, 0.26, 0.46, 0.26, 205, 450), _arc(0.5, 0.75, 0.5, 0.25, 270, 520)),
    )
    g["4"] = (_stroke((0.78, 1.0), (0.78, 0.0), (0.0, 0.68), (1.0, 0.68)),)
    g["5"] = (
        _stroke((0.92, 0.0), (0.12, 0.0), (0.07, 0.47), _arc(0.5, 0.68, 0.5, 0.32, 215, 520)),
    )
    g["6"] = (
        _stroke(_arc(0.5, 0.30, 0.5, 0.30, 315, 180), (0.0, 0.68)),
        _stroke(_arc(0.5, 0.68, 0.5, 0.32, 0, 360)),
    )
    g["7"] = (_stroke((0.0, 0.0), (1.0, 0.0), (0.36, 1.0)),)
    g["8"] = (
        _stroke(_arc(0.5, 0.25, 0.44, 0.25, 0, 360)),
        _stroke(_arc(0.5, 0.74, 0.5, 0.26, 0, 360)),
    )
    g["9"] = (
        _stroke(_arc(0.5, 0.32, 0.5, 0.32, 0, 360)),
        _stroke((1.0, 0.32), (1.0, 0.70), _arc(0.5, 0.70, 0.5, 0.30, 0, 135)),
    )

    # ---- plate letters (Latin names of the Cyrillic look-alikes) ---------
    g["A"] = (_stroke((0.0, 1.0), (0.5, 0.0), (1.0, 1.0)), _stroke((0.2, 0.64), (0.8, 0.64)))
    g["B"] = (
        _stroke((0.0, 0.47), (0.0, 0.0), (0.58, 0.0), _arc(0.58, 0.235, 0.36, 0.235, 270, 450), (0.0, 0.47)),
        _stroke((0.0, 0.47), (0.60, 0.47), _arc(0.60, 0.735, 0.40, 0.265, 270, 450), (0.0, 1.0), (0.0, 0.47)),
    )
    g["E"] = (_stroke((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)), _stroke((0.0, 0.5), (0.82, 0.5)))
    g["K"] = (
        _stroke((0.0, 0.0), (0.0, 1.0)),
        _stroke((1.0, 0.0), (0.0, 0.56)),
        _stroke((0.32, 0.38), (1.0, 1.0)),
    )
    g["M"] = (_stroke((0.0, 1.0), (0.0, 0.0), (0.5, 0.62), (1.0, 0.0), (1.0, 1.0)),)
    g["H"] = (
        _stroke((0.0, 0.0), (0.0, 1.0)),
        _stroke((1.0, 0.0), (1.0, 1.0)),
        _stroke((0.0, 0.5), (1.0, 0.5)),
    )
    g["O"] = (_stadium(0.0, 1.0, 0.40),)
    g["P"] = (_stroke((0.0, 1.0), (0.0, 0.0), (0.58, 0.0), _arc(0.58, 0.27, 0.42, 0.27, 270, 450), (0.0, 0.54)),)
    g["C"] = (
        _stroke(_arc(0.5, 0.38, 0.5, 0.38, 335, 180), (0.0, 0.62), _arc(0.5, 0.62, 0.5, 0.38, 180, 25)),
    )
    g["T"] = (_stroke((0.0, 0.0), (1.0, 0.0)), _stroke((0.5, 0.0), (0.5, 1.0)))
    g["Y"] = (_stroke((1.0, 0.0), (0.30, 1.0)), _stroke((0.0, 0.0), (0.57, 0.60)))
    g["X"] = (_stroke((0.0, 0.0), (1.0, 1.0)), _stroke((1.0, 0.0), (0.0, 1.0)))

    # ---- extra glyphs for the "RUS" inscription --------------------------
    g["R"] = (
        _stroke((0.0, 1.0), (0.0, 0.0), (0.58, 0.0), _arc(0.58, 0.27, 0.42, 0.27, 270, 450), (0.0, 0.54)),
        _stroke((0.42, 0.54), (1.0, 1.0)),
    )
    g["U"] = (_stroke((0.0, 0.0), (0.0, 0.66), _arc(0.5, 0.66, 0.5, 0.34, 180, 0), (1.0, 0.0)),)
    g["S"] = (
        _stroke(_arc(0.5, 0.26, 0.48, 0.26, 340, 90), _arc(0.5, 0.75, 0.5, 0.25, 270, 520)),
    )
    return g


GLYPHS: Final[dict[str, Glyph]] = _build()

#: Characters that may appear in a plate string.
PLATE_CHARACTERS: Final[str] = "0123456789ABEKMHOPCTYX"
