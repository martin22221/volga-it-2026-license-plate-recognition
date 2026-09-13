"""Plate identities: seeded, syntactically valid plate strings.

Two character structures, per GOST R 50577-2018 section 3.3:

* ``type1`` -- used by ``type1`` and ``type1a``: ``M 000 MM 55`` / ``M 000 MM 555``,
  normalised ``A123BC77`` / ``A123BC777``;
* ``type1b`` -- used by ``type1b`` only: ``MM 000 55``, normalised ``AB12377``.

The plate type decides the structure (:data:`FORMAT_BY_PLATE_TYPE`); it is not
configurable, so a ``type1b`` label can never be drawn in the type 1
structure.  (Generator V1 did exactly that and was superseded for it.)

Region codes:

* type 1 structure: ``01``-``99``, or three digits starting with ``1``, ``2``
  or ``7`` -- the task's rule, mirrored from ``src/validator.py``;
* type 1B: two digits only.  GOST shows 1B solely as ``MM 000 55`` and
  depicts it in one figure (A.5), while types 1 and 1A have separate two- and
  three-digit figures; see :data:`TYPE1B_REGION_LENGTHS`.

Deliberate *narrowings*, both subsets of the grammar and so never in conflict
with it: the serial number is ``001``-``999`` (``000`` is not issued), and
regions come from the grammar's space rather than a list of currently assigned
regions, so the OCR sees every digit in every position.

The grammar is re-implemented here, so the generator stays self-contained when
the dataset is submitted on its own; tests check it agrees with
``src/validator.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterator

import numpy as np

#: The twelve letters used on Russian plates, in their Latin form.
LETTERS: Final[str] = "ABEKMHOPCTYX"
DIGITS: Final[str] = "0123456789"
THREE_DIGIT_REGION_PREFIXES: Final[str] = "127"

STANDARD_FORMAT: Final[str] = "type1"
TYPE1B_FORMAT: Final[str] = "type1b"

#: Plate type -> character structure.  Fixed, not configurable.
FORMAT_BY_PLATE_TYPE: Final[dict[str, str]] = {
    "type1": STANDARD_FORMAT,
    "type1a": STANDARD_FORMAT,
    "type1b": TYPE1B_FORMAT,
}

#: Region lengths a type 1B plate may carry.
TYPE1B_REGION_LENGTHS: Final[tuple[int, ...]] = (2,)

_STANDARD_RE: Final[re.Pattern[str]] = re.compile(rf"^[{LETTERS}]\d{{3}}[{LETTERS}]{{2}}(\d{{2,3}})$")
_TYPE1B_RE: Final[re.Pattern[str]] = re.compile(rf"^[{LETTERS}]{{2}}\d{{3}}(\d{{2,3}})$")


@dataclass(frozen=True)
class PlateText:
    """A plate's characters, split into the groups the layouts position.

    ``series1`` precedes the serial number and ``series2`` follows it.  In the
    ``type1`` structure they hold one and two letters; in ``type1b`` two
    letters and nothing.
    """

    series1: str
    number: str
    series2: str
    region: str
    text_format: str

    def __post_init__(self) -> None:
        expected = (1, 2) if self.text_format == STANDARD_FORMAT else (2, 0)
        if self.text_format not in (STANDARD_FORMAT, TYPE1B_FORMAT):
            raise ValueError(f"unknown text format {self.text_format!r}")
        if (len(self.series1), len(self.series2)) != expected or len(self.number) != 3:
            raise ValueError(f"{self.full!r} does not have the {self.text_format} structure")

    @property
    def full(self) -> str:
        """The normalised plate string, e.g. ``A123BC77`` or ``AB12377``."""
        return f"{self.series1}{self.number}{self.series2}{self.region}"

    def main_groups(self) -> list[tuple[str, str]]:
        """Non-region groups in reading order, as ``(characters, kind)``."""
        groups = [(self.series1, "letter"), (self.number, "digit"), (self.series2, "letter")]
        return [(chars, kind) for chars, kind in groups if chars]

    def characters(self) -> Iterator[tuple[int, str, str]]:
        """``(position, char, role)`` for every character of :attr:`full`.

        ``role`` is ``letter``, ``digit`` or ``region``.
        """
        position = 0
        for chars, kind in self.main_groups():
            for char in chars:
                yield position, char, kind
                position += 1
        for char in self.region:
            yield position, char, "region"
            position += 1


def _region_ok(region: str) -> bool:
    if len(region) == 3 and region[0] not in THREE_DIGIT_REGION_PREFIXES:
        return False
    return set(region) != {"0"}


def is_standard_plate(text: str) -> bool:
    """Whether ``text`` has the type 1 / 1A structure."""
    match = _STANDARD_RE.fullmatch(text)
    return bool(match) and _region_ok(match.group(1))


def is_type1b_plate(text: str) -> bool:
    """Whether ``text`` has the type 1B structure."""
    match = _TYPE1B_RE.fullmatch(text)
    return bool(match) and len(match.group(1)) in TYPE1B_REGION_LENGTHS and _region_ok(match.group(1))


def is_plate_for_type(text: str, plate_type: str) -> bool:
    """Whether ``text`` has the structure ``plate_type`` requires."""
    fmt = FORMAT_BY_PLATE_TYPE[plate_type]
    return is_standard_plate(text) if fmt == STANDARD_FORMAT else is_type1b_plate(text)


def sample_region(rng: np.random.Generator, three_digit_probability: float) -> str:
    """A region code: ``01``-``99``, or ``[127]`` followed by ``00``-``99``."""
    if rng.random() < three_digit_probability:
        prefix = THREE_DIGIT_REGION_PREFIXES[int(rng.integers(len(THREE_DIGIT_REGION_PREFIXES)))]
        return f"{prefix}{int(rng.integers(0, 100)):02d}"
    return f"{int(rng.integers(1, 100)):02d}"


def _letters(rng: np.random.Generator, count: int) -> str:
    return "".join(LETTERS[int(i)] for i in rng.integers(0, len(LETTERS), size=count))


def sample_plate_text(
    rng: np.random.Generator, plate_type: str, three_digit_probability: float
) -> PlateText:
    """Draw one identity in the structure ``plate_type`` requires."""
    fmt = FORMAT_BY_PLATE_TYPE.get(plate_type)
    if fmt is None:
        raise ValueError(f"no plate structure for plate type {plate_type!r}")
    number = f"{int(rng.integers(1, 1000)):03d}"
    if fmt == STANDARD_FORMAT:
        return PlateText(
            series1=_letters(rng, 1),
            number=number,
            series2=_letters(rng, 2),
            region=sample_region(rng, three_digit_probability),
            text_format=fmt,
        )
    three = three_digit_probability if 3 in TYPE1B_REGION_LENGTHS else 0.0
    region = sample_region(rng, three)
    if len(region) not in TYPE1B_REGION_LENGTHS:  # pragma: no cover - guarded above
        raise RuntimeError("type1b region length outside TYPE1B_REGION_LENGTHS")
    return PlateText(series1=_letters(rng, 2), number=number, series2="", region=region, text_format=fmt)


class PlateIdentitySampler:
    """Draws plate identities, limiting how often any one string repeats."""

    #: Give up after this many consecutive rejections -- only reachable with a
    #: tiny identity space, never with the real grammar.
    MAX_ATTEMPTS: Final[int] = 10_000

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        three_digit_probability: float,
        max_images_per_plate: int = 1,
    ) -> None:
        self._rng = rng
        self._three_digit_probability = three_digit_probability
        self._max_uses = max_images_per_plate
        self._uses: dict[str, int] = {}

    def sample(self, plate_type: str) -> PlateText:
        for _ in range(self.MAX_ATTEMPTS):
            candidate = sample_plate_text(self._rng, plate_type, self._three_digit_probability)
            uses = self._uses.get(candidate.full, 0)
            if uses < self._max_uses:
                self._uses[candidate.full] = uses + 1
                return candidate
        raise RuntimeError("could not draw an unused plate identity")
