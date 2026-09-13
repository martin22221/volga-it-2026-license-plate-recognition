"""Plate identities: seeded, syntactically valid plate strings.

The grammar mirrors ``src/validator.py`` -- the competition mask
``[LETTER][3 DIGITS][2 LETTERS][REGION]`` -- but is re-implemented here so
the generator stays self-contained when the dataset is submitted on its own.
``tests/test_generator_plate_text.py`` checks both agree.

Two deliberate *narrowings* of that grammar, both subsets of it and so never in
conflict with it:

* the serial number is ``001``-``999``; ``000`` is not issued on real plates;
* region codes are drawn from the grammar's space (``01``-``99``, or three
  digits starting with ``1``, ``2`` or ``7``) rather than from a list of the
  regions currently assigned, so the OCR sees every digit in every position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterator

import numpy as np

from .config import ConfigError

#: The twelve letters used on Russian plates, in their Latin form.
LETTERS: Final[str] = "ABEKMHOPCTYX"
DIGITS: Final[str] = "0123456789"
THREE_DIGIT_REGION_PREFIXES: Final[str] = "127"

_COMPETITION_RE: Final[re.Pattern[str]] = re.compile(
    rf"^[{LETTERS}]\d{{3}}[{LETTERS}]{{2}}\d{{2,3}}$"
)


@dataclass(frozen=True)
class PlateText:
    """A plate's characters, split into the groups the layouts position.

    ``series1`` precedes the serial number and ``series2`` follows it.  In the
    ``competition`` format they hold one and two letters; in ``gost_1b`` they
    hold two letters and nothing.
    """

    series1: str
    number: str
    series2: str
    region: str
    text_format: str

    @property
    def full(self) -> str:
        """The normalised plate string, e.g. ``A123BC77``."""
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


def is_competition_plate(text: str) -> bool:
    """Whether ``text`` satisfies the competition grammar (see module doc)."""
    if not _COMPETITION_RE.fullmatch(text):
        return False
    region = text[6:]
    if len(region) == 3 and region[0] not in THREE_DIGIT_REGION_PREFIXES:
        return False
    return set(region) != {"0"}


def sample_region(rng: np.random.Generator, three_digit_probability: float) -> str:
    """A region code: ``01``-``99``, or ``[127]`` followed by ``00``-``99``."""
    if rng.random() < three_digit_probability:
        prefix = THREE_DIGIT_REGION_PREFIXES[int(rng.integers(len(THREE_DIGIT_REGION_PREFIXES)))]
        return f"{prefix}{int(rng.integers(0, 100)):02d}"
    return f"{int(rng.integers(1, 100)):02d}"


def _letters(rng: np.random.Generator, count: int) -> str:
    return "".join(LETTERS[int(i)] for i in rng.integers(0, len(LETTERS), size=count))


def sample_plate_text(
    rng: np.random.Generator, text_format: str, three_digit_probability: float
) -> PlateText:
    """Draw one plate identity in ``text_format``."""
    number = f"{int(rng.integers(1, 1000)):03d}"
    if text_format == "competition":
        return PlateText(
            series1=_letters(rng, 1),
            number=number,
            series2=_letters(rng, 2),
            region=sample_region(rng, three_digit_probability),
            text_format=text_format,
        )
    if text_format == "gost_1b":
        # GOST R 50577-2018 section 3.3 shows type 1B only with a 2-digit region.
        return PlateText(
            series1=_letters(rng, 2),
            number=number,
            series2="",
            region=sample_region(rng, 0.0),
            text_format=text_format,
        )
    raise ConfigError(f"unknown text format {text_format!r}")


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

    def sample(self, text_format: str) -> PlateText:
        for _ in range(self.MAX_ATTEMPTS):
            candidate = sample_plate_text(self._rng, text_format, self._three_digit_probability)
            uses = self._uses.get(candidate.full, 0)
            if uses < self._max_uses:
                self._uses[candidate.full] = uses + 1
                return candidate
        raise RuntimeError("could not draw an unused plate identity")
