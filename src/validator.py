"""Structural validator for Russian license plate numbers.

Two character structures, per GOST R 50577-2018 section 3.3:

``type1`` / ``type1a`` -- ``M 000 MM 55`` or ``M 000 MM 555``::

    [LETTER][DIGIT][DIGIT][DIGIT][LETTER][LETTER][REGION]

``REGION`` holds 2 or 3 digits; a 3-digit region must start with 1, 2 or 7.
``type1a`` is the same registration arranged on two lines.

``type1b`` (yellow, passenger transport) -- ``MM 000 55``::

    [LETTER][LETTER][DIGIT][DIGIT][DIGIT][REGION]

The standard gives type 1B only with a two-digit region, and depicts it in a
single figure (A.5), whereas types 1 and 1A each have separate two- and
three-digit figures.  :data:`TYPE1B_REGION_LENGTHS` therefore allows two
digits only; widen it if official examples ever show otherwise.

:func:`validate_plate` without a plate type accepts either structure -- OCR
output is checked before, or independently of, the classifier's decision.
Pass ``plate_type`` to demand the structure of that type.

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
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "У": "Y",  # У
    "Х": "X",  # Х
}

#: Leading digits permitted for 3-digit region codes.
THREE_DIGIT_REGION_PREFIXES: Final[frozenset[str]] = frozenset("127")

#: Region lengths permitted on a type 1B plate (GOST shows ``MM 000 55`` only).
TYPE1B_REGION_LENGTHS: Final[tuple[int, ...]] = (2,)

#: Structure identifiers.
STANDARD_FORMAT: Final[str] = "type1"
TYPE1B_FORMAT: Final[str] = "type1b"

#: Which structure each plate type must follow.
FORMAT_BY_PLATE_TYPE: Final[dict[str, str]] = {
    "type1": STANDARD_FORMAT,
    "type1a": STANDARD_FORMAT,
    "type1b": TYPE1B_FORMAT,
}

_LETTERS_CLASS: Final[str] = "".join(sorted(ALLOWED_LETTERS))
_PLATE_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<series1>[{_LETTERS_CLASS}])"
    rf"(?P<number>\d{{3}})"
    rf"(?P<series2>[{_LETTERS_CLASS}]{{2}})"
    rf"(?P<region>\d{{2,3}})$"
)
_TYPE1B_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<series>[{_LETTERS_CLASS}]{{2}})"
    rf"(?P<number>\d{{3}})"
    rf"(?P<region>\d{{2,3}})$"
)

_STRUCTURE: Final[dict[str, str]] = {
    STANDARD_FORMAT: "[LETTER][3 DIGITS][2 LETTERS][REGION]",
    TYPE1B_FORMAT: "[2 LETTERS][3 DIGITS][REGION]",
}


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a single plate string.

    ``format`` names the structure the text matched (``type1`` or
    ``type1b``), or is empty when it matched none.
    """

    is_valid: bool
    normalized: str
    reason: str = ""
    format: str = ""

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


def _check_region(region: str) -> str:
    """Reason the region is invalid, or ``""``."""
    if len(region) == 3 and region[0] not in THREE_DIGIT_REGION_PREFIXES:
        return f"3-digit region {region!r} must start with 1, 2 or 7"
    if set(region) == {"0"}:
        return f"region {region!r} is not assigned"
    return ""


def _validate_format(normalized: str, fmt: str) -> ValidationResult:
    if fmt == STANDARD_FORMAT:
        match = _PLATE_RE.fullmatch(normalized)
    else:
        match = _TYPE1B_RE.fullmatch(normalized)
    if match is None:
        return ValidationResult(False, normalized, f"does not match {_STRUCTURE[fmt]}")

    region = match.group("region")
    if fmt == TYPE1B_FORMAT and len(region) not in TYPE1B_REGION_LENGTHS:
        allowed = " or ".join(str(n) for n in TYPE1B_REGION_LENGTHS)
        return ValidationResult(
            False, normalized, f"type1b region {region!r} must have {allowed} digits"
        )
    reason = _check_region(region)
    if reason:
        return ValidationResult(False, normalized, reason)
    return ValidationResult(True, normalized, format=fmt)


def validate_plate(text: str, plate_type: str | None = None) -> ValidationResult:
    """Validate a plate number and explain the verdict.

    With ``plate_type`` of ``type1``, ``type1a`` or ``type1b`` the text must
    follow that type's structure.  Without it (or for ``other``) any known
    structure is accepted; the standard one is tried first.

    Returns a :class:`ValidationResult` whose ``normalized`` field holds the
    normalised text (useful even when the plate is rejected).
    """
    normalized = normalize_plate(text)

    if not normalized:
        return ValidationResult(False, normalized, "empty text")

    required = FORMAT_BY_PLATE_TYPE.get(plate_type or "")
    if required is not None:
        return _validate_format(normalized, required)

    standard = _validate_format(normalized, STANDARD_FORMAT)
    if standard.is_valid or _PLATE_RE.fullmatch(normalized):
        return standard
    type1b = _validate_format(normalized, TYPE1B_FORMAT)
    if type1b.is_valid or _TYPE1B_RE.fullmatch(normalized):
        return type1b
    return standard


def is_valid_plate(text: str, plate_type: str | None = None) -> bool:
    """Return whether ``text`` is a structurally valid Russian plate number."""
    return validate_plate(text, plate_type).is_valid
