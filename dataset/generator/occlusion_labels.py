"""Which characters an occluder leaves readable: the ``#`` labelling rule.

Up to 2.2.0 a character became ``#`` when an occluder covered more than 35 %
of its *bounding box*.  Glyph ink fills only part of that box, so the rule
failed both ways.  It kept the label on characters whose identifying stroke
was erased: production sample #9880 had 27 % of an ``O``'s ink hidden, all
of it the right stroke, so the ``O`` read as ``C`` but stayed labelled
``O``.  It also masked characters that were still legible.

Since 2.3.0 each character is judged by its own ink, against every character
that could stand in its place: the other letters for a letter position, the
other digits for a digit position.  A character keeps its label only if all
three hold:

1. **Enough ink is visible.**  At most ``MAX_INK_COVERED`` (50 %) of its ink
   is covered.  Beyond that it is unreadable, whatever remains.
2. **Its identifying strokes are visible.**  For every alternative ``a``, the
   evidence that separates the character from ``a`` is the absolute
   difference of their two ink masks, each blurred by ``EVIDENCE_BLUR``
   (0.5) stroke widths so that sub-stroke outline wobble does not count.  At
   least ``MIN_EVIDENCE_RETAINED`` (70 %) of that evidence must stay
   uncovered.  Covering ink the character has and ``a`` lacks (the right
   stroke of ``O`` against ``C``), or the gap where ``a`` has ink (the
   opening of ``C`` against ``O``), both remove evidence.
3. **It still reads as itself.**  A template reader compares the visible
   remainder with every candidate by normalised cross-correlation.  Templates
   and observation are blurred by ``READER_BLUR`` (0.35) stroke widths.  The
   covered pixels are shown twice, once as plate field (a snow clump on a
   white plate) and once as ink (a dark tow bar).  In both views, the true
   character must beat its strongest rival by at least ``MIN_READER_MARGIN``
   (half) of the margin it has when nothing covers it.

Evidence and reading are measured inside the character's own cell.  Anything
covered outside it changes nothing.  Otherwise, a character failing any test
is ``#``.  Nothing here depends on particular characters, samples or plate
types.  The rule sees only the stroke font's masks in the cell each character
was drawn in, so it applies unchanged at every plate scale and to every
layout.  It draws no random numbers.

The thresholds were calibrated on 53,181 random bar, blob and edge
occlusions: every plate character, all four cell sizes, 0.22-1.5 px/mm, two
seeds.  Two readers outside the rule were used as independent checks:
cross-correlation at 0.2 stroke widths, and an L1-distance reader, both
shown covered pixels as field and as ink.

- They misread none of the characters the rule keeps.
- None of the 4,009 clearly readable cases was masked.  "Clearly readable"
  means at most 30 % of the ink covered and both readers keeping at least
  60 % of their unoccluded margin.

An ink-only rule cannot do that.  Keeping every character with at most 10 %
of its ink covered still kept 67 that those readers misread.  See the
generator README (2.3.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np

from .fonts import FontProvider
from .plate_text import DIGITS, LETTERS
from .render import GlyphCell, glyph_ink

#: A character with more than this share of its ink covered is ``#``.
MAX_INK_COVERED: Final[float] = 0.50
#: Share of the evidence separating a character from each alternative that
#: must stay visible for its label to be kept.
MIN_EVIDENCE_RETAINED: Final[float] = 0.70
#: The reader's margin for the true character, as a share of its margin on
#: the unoccluded character, required in both views of the covered pixels.
MIN_READER_MARGIN: Final[float] = 0.50
#: Blur (in stroke widths) applied before measuring evidence and before reading.
EVIDENCE_BLUR: Final[float] = 0.5
READER_BLUR: Final[float] = 0.35


def alternatives(char: str) -> str:
    """The characters that could stand where ``char`` stands on a plate."""
    if char in DIGITS:
        pool = DIGITS
    elif char in LETTERS:
        pool = LETTERS
    else:
        raise ValueError(f"{char!r} is not a plate character")
    return "".join(c for c in pool if c != char)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with zero padding (float32, deterministic)."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    height, width = image.shape
    padded = np.pad(image, ((0, 0), (radius, radius)))
    rows = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        rows += weight * padded[:, i : i + width]
    padded = np.pad(rows, ((radius, radius), (0, 0)))
    out = np.zeros_like(image)
    for i, weight in enumerate(kernel):
        out += weight * padded[i : i + height]
    return out


def _standardise(image: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm vector: dot products of two are their NCC."""
    flat = image.ravel().astype(np.float64)
    flat = flat - flat.mean()
    norm = float(np.sqrt((flat * flat).sum()))
    return flat / norm if norm > 1e-12 else np.zeros_like(flat)


@dataclass
class GlyphReference:
    """Precomputed masks for judging one drawn character.

    Masks cover the character's cell plus a margin, so blurs are not cut off
    at the cell edge.  Evidence and reading are then measured inside the
    cell (``window``): an occluder beside a character, covering none of its
    cell, changes nothing about it.
    """

    cell: GlyphCell
    origin: tuple[int, int]  # plate pixel of the patch's top-left corner
    ink: np.ndarray  # the character's own ink over the patch
    window: tuple[slice, slice]  # the cell inside the patch
    rivals: str  # alternatives, in the order of ``evidence`` / ``templates[1:]``
    evidence: list[np.ndarray] = field(repr=False)  # per rival, over the window
    evidence_total: list[float] = field(repr=False)
    templates: np.ndarray = field(repr=False)  # (1 + len(rivals)) x window pixels, standardised
    baseline_margin: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.ink.shape  # type: ignore[return-value]


def _pad(cell: GlyphCell) -> int:
    return int(math.ceil(3.0 * EVIDENCE_BLUR * cell.stroke_px)) + 1


def glyph_reference(cell: GlyphCell, font: FontProvider) -> GlyphReference:
    stroke = cell.stroke_px
    pad = _pad(cell)
    ink, origin = glyph_ink(font, cell, cell.char, pad)
    f = cell.factor
    x0, y0, x1, y1 = cell.box
    window = (
        slice(math.floor(y0 / f) - origin[1], math.ceil(y1 / f) - origin[1]),
        slice(math.floor(x0 / f) - origin[0], math.ceil(x1 / f) - origin[0]),
    )
    rivals = alternatives(cell.char)
    rival_ink = [glyph_ink(font, cell, rival, pad)[0] for rival in rivals]
    own = _blur(ink, EVIDENCE_BLUR * stroke)[window]
    evidence = [np.abs(own - _blur(other, EVIDENCE_BLUR * stroke)[window]) for other in rival_ink]
    templates = np.stack([_standardise(_blur(mask, READER_BLUR * stroke)[window]) for mask in [ink, *rival_ink]])
    baseline = float(1.0 - (templates[1:] @ templates[0]).max())
    return GlyphReference(
        cell=cell,
        origin=origin,
        ink=ink,
        window=window,
        rivals=rivals,
        evidence=evidence,
        evidence_total=[float(e.sum()) for e in evidence],
        templates=templates,
        baseline_margin=max(baseline, 1e-6),
    )


@dataclass(frozen=True)
class CharacterVisibility:
    """How much of one character an occluder leaves, and the verdict."""

    position: int
    char: str
    ink_covered: float
    evidence_retained: float  # the least, over all alternatives
    closest: str  # the alternative with the least evidence left
    reader_margin: float  # the weaker of the two views, relative to unoccluded
    rival: str  # the reader's strongest rival in that view
    readable: bool

    def record(self) -> dict:
        return {
            "position": self.position,
            "char": self.char,
            "ink_covered": round(self.ink_covered, 4),
            "evidence_retained": round(self.evidence_retained, 4),
            "closest": self.closest,
            "reader_margin": round(self.reader_margin, 4),
            "rival": self.rival,
            "hidden": not self.readable,
        }


def assess_character(reference: GlyphReference, covered: np.ndarray) -> CharacterVisibility:
    """Judge one character under ``covered`` (occluder opacity over the patch)."""
    visible = 1.0 - np.clip(covered, 0.0, 1.0).astype(np.float32)
    ink = reference.ink
    ink_covered = 1.0 - float((ink * visible).sum() / max(float(ink.sum()), 1e-6))
    in_cell = visible[reference.window]
    retained = [
        float((e * in_cell).sum() / total) if total > 1e-9 else 1.0
        for e, total in zip(reference.evidence, reference.evidence_total)
    ]
    closest_index = int(np.argmin(retained))
    sigma = READER_BLUR * reference.cell.stroke_px
    reader_margin, rival = math.inf, ""
    remainder = ink * visible
    in_cell_cover = np.zeros_like(visible)
    in_cell_cover[reference.window] = 1.0 - in_cell
    for observed in (remainder, remainder + in_cell_cover):  # covered as field, as ink
        scores = reference.templates @ _standardise(_blur(observed, sigma)[reference.window])
        best = int(np.argmax(scores[1:]))
        margin = float(scores[0] - scores[1 + best]) / reference.baseline_margin
        if margin < reader_margin:
            reader_margin, rival = margin, reference.rivals[best]
    readable = (
        ink_covered <= MAX_INK_COVERED
        and retained[closest_index] >= MIN_EVIDENCE_RETAINED
        and reader_margin >= MIN_READER_MARGIN
    )
    return CharacterVisibility(
        position=reference.cell.position,
        char=reference.cell.char,
        ink_covered=ink_covered,
        evidence_retained=retained[closest_index],
        closest=reference.rivals[closest_index],
        reader_margin=reader_margin,
        rival=rival,
        readable=readable,
    )


def covered_patch(occluder: np.ndarray, reference: GlyphReference, plate_origin: tuple[int, int]) -> np.ndarray:
    """The occluder over a reference's patch (zero outside the canvas)."""
    height, width = reference.shape
    x0 = plate_origin[0] + reference.origin[0]
    y0 = plate_origin[1] + reference.origin[1]
    patch = np.zeros((height, width), np.float32)
    ya, yb = max(y0, 0), min(y0 + height, occluder.shape[0])
    xa, xb = max(x0, 0), min(x0 + width, occluder.shape[1])
    if yb > ya and xb > xa:
        patch[ya - y0 : yb - y0, xa - x0 : xb - x0] = occluder[ya:yb, xa:xb]
    return patch


class OcclusionJudge:
    """Judges the characters of one plate under candidate occluders.

    References are built lazily, only for characters an occluder touches,
    and reused across the redraws of one occlusion.
    """

    def __init__(self, cells: Sequence[GlyphCell], font: FontProvider) -> None:
        self._cells = list(cells)
        self._font = font
        self._references: dict[int, GlyphReference] = {}

    def reference(self, index: int) -> GlyphReference:
        if index not in self._references:
            self._references[index] = glyph_reference(self._cells[index], self._font)
        return self._references[index]

    def assess(self, occluder: np.ndarray, plate_origin: tuple[int, int]) -> list[CharacterVisibility]:
        """Every character the occluder touches, in position order."""
        results = []
        for index, cell in enumerate(self._cells):
            if not _touches(occluder, cell, plate_origin):
                continue
            reference = self.reference(index)
            results.append(assess_character(reference, covered_patch(occluder, reference, plate_origin)))
        return sorted(results, key=lambda r: r.position)


def _touches(occluder: np.ndarray, cell: GlyphCell, plate_origin: tuple[int, int]) -> bool:
    """Whether any occluder lies over the character's cell."""
    f = cell.factor
    x0, y0, x1, y1 = cell.box
    xa = max(plate_origin[0] + math.floor(x0 / f), 0)
    ya = max(plate_origin[1] + math.floor(y0 / f), 0)
    xb = min(plate_origin[0] + math.ceil(x1 / f), occluder.shape[1])
    yb = min(plate_origin[1] + math.ceil(y1 / f), occluder.shape[0])
    return xb > xa and yb > ya and bool(occluder[ya:yb, xa:xb].max() > 0.0)


__all__ = [
    "EVIDENCE_BLUR",
    "MAX_INK_COVERED",
    "MIN_EVIDENCE_RETAINED",
    "MIN_READER_MARGIN",
    "READER_BLUR",
    "CharacterVisibility",
    "GlyphReference",
    "OcclusionJudge",
    "alternatives",
    "assess_character",
    "glyph_reference",
]
