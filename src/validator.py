"""Structural validator for standard Russian license plate numbers.

Target structure::

    [LETTER][DIGIT][DIGIT][DIGIT][LETTER][LETTER][REGION]

``REGION`` holds 2 or 3 digits; a 3-digit region must start with 1, 2 or 7.

Only the twelve letters that exist on Russian plates are accepted.  They are
the Cyrillic letters that are graphically identical to Latin ones, so both
alphabets are folded onto the Latin form during normalisation -- this is a
character-set normalisation, not an OCR correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

#: The twelve letters used on Russian plates, in their Latin form.
ALLOWED_LETTERS: Final[frozenset[str]] = frozenset("ABEKMHOPCTYX")

#: Cyrillic look-alikes folded onto the Latin letters above.
CYRILLIC_TO_LATIN: Final[dict[str, str]] = {
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0415": "E",  # Е
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u041d": "H",  # Н
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0421": "C",  # С
    "\u0422": "T",  # Т
    "\u0423": "Y",  # У
    "\u0425": "X",  # Х
}

#: Leading digits permitted for 3-digit region codes.
THREE_DIGIT_REGION_PREFIXES: Final[frozenset[str]] = frozenset("127")

_LETTERS_CLASS: Final[str] = "".join(sorted(ALLOWED_LETTERS))
_PLATE_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<series1>[{_LETTERS_CLASS}])"
    rf"(?P<number>\d{{3}})"
    rf"(?P<series2>[{_LETTERS_CLASS}]{{2}})"
    rf"(?P<region>\d{{2,3}})$"
)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a single plate string."""

    is_valid: bool
    normalized: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.is_valid


def normalize_plate(text: str) -> str:
    """Upper-case ``text``, drop separators and fold Cyrillic to Latin.

    Removes whitespace, hyphens and underscores only; no character is ever
    substituted for a "similar looking" one beyond the Cyrillic/Latin fold.
    """
    stripped = "".join(ch for ch in text if not ch.isspace() and ch not in "-_")
    upper = stripped.upper()
    return "".join(CYRILLIC_TO_LATIN.get(ch, ch) for ch in upper)


def validate_plate(text: str) -> ValidationResult:
    """Validate a plate number and explain the verdict.

    Returns a :class:`ValidationResult` whose ``normalized`` field holds the
    normalised text (useful even when the plate is rejected).
    """
    normalized = normalize_plate(text)

    if not normalized:
        return ValidationResult(False, normalized, "empty text")

    match = _PLATE_RE.fullmatch(normalized)
    if match is None:
        return ValidationResult(
            False, normalized, "does not match [LETTER][3 DIGITS][2 LETTERS][REGION]"
        )

    region = match.group("region")
    if len(region) == 3 and region[0] not in THREE_DIGIT_REGION_PREFIXES:
        return ValidationResult(
            False, normalized, f"3-digit region {region!r} must start with 1, 2 or 7"
        )
    if set(region) == {"0"}:
        return ValidationResult(False, normalized, f"region {region!r} is not assigned")

    return ValidationResult(True, normalized)


def is_valid_plate(text: str) -> bool:
    """Return whether ``text`` is a structurally valid Russian plate number."""
    return validate_plate(text).is_valid
