"""The character set the recogniser predicts, and how a plate becomes labels.

Russian plates use exactly twelve letters and ten digits, so the alphabet is 22
symbols plus a CTC blank. Keeping it that small is a real advantage: the
recogniser cannot emit a character that no plate can carry, which removes a
whole family of OCR errors before training starts.

``#`` -- the annotation guide's "exactly one character I cannot read" -- is
**not** in the alphabet. A model must never learn to predict uncertainty as a
character; uncertainty belongs in the confidence, and ``#`` is produced at
output time from a low per-character score. Training samples whose label
contains ``#`` are therefore handled explicitly rather than silently: see
:func:`encode`.

Reading order for the two-line ``type1a`` plate is the order the characters are
read, not the order they appear on a raster scan: ``M 000`` across the top,
then ``MM`` + region along the bottom (``dataset/generator/templates.py``).
That is the same string as the one-line types, which is why one sequence model
can read all three.
"""

from __future__ import annotations

from typing import Final, Iterable, Sequence

from src.validator import ALLOWED_LETTERS, normalize_plate

#: Digits, then the twelve letters, in a fixed order. The order is part of the
#: trained model: changing it invalidates every checkpoint.
DIGITS: Final[str] = "0123456789"
LETTERS: Final[str] = "".join(sorted(ALLOWED_LETTERS))
ALPHABET: Final[str] = DIGITS + LETTERS

#: CTC reserves index 0 for the blank; real symbols start at 1.
BLANK_INDEX: Final[int] = 0
NUM_CLASSES: Final[int] = len(ALPHABET) + 1

#: Marker for one unreadable character. Never predicted; only ever written out.
UNREADABLE: Final[str] = "#"

_INDEX_BY_CHAR: Final[dict[str, int]] = {
    char: index + 1 for index, char in enumerate(ALPHABET)
}
_CHAR_BY_INDEX: Final[dict[int, str]] = {
    index + 1: char for index, char in enumerate(ALPHABET)
}

#: The longest plate string the recogniser must produce: a type1/type1a
#: registration with a three-digit region is 9 characters.
MAX_LENGTH: Final[int] = 9


class AlphabetError(ValueError):
    """A plate string that cannot be turned into training labels."""


def is_trainable(text: str) -> bool:
    """Can this annotation be used as an OCR target at all?

    A label with an unreadable character has no ground truth for that position,
    so it cannot supervise a sequence loss. Such samples are still useful to
    the detector and the type classifier, and are excluded only from OCR.
    """
    return bool(text) and UNREADABLE not in text


def encode(text: str) -> list[int]:
    """Plate string to CTC target indices.

    Raises rather than guessing: an unknown character or a ``#`` means the
    caller has fed in a sample that should have been filtered by
    :func:`is_trainable`.
    """
    plate = normalize_plate(text)
    if not plate:
        raise AlphabetError("empty plate string")
    if UNREADABLE in plate:
        raise AlphabetError(
            f"{plate!r} contains {UNREADABLE!r}: an unreadable character has no target; "
            "filter the sample with is_trainable() instead of encoding it"
        )
    out: list[int] = []
    for char in plate:
        index = _INDEX_BY_CHAR.get(char)
        if index is None:
            raise AlphabetError(f"{char!r} in {plate!r} is not on a Russian plate")
        out.append(index)
    if len(out) > MAX_LENGTH:
        raise AlphabetError(f"{plate!r} is longer than {MAX_LENGTH} characters")
    return out


def decode(indices: Iterable[int]) -> str:
    """Target indices back to a plate string. The inverse of :func:`encode`."""
    return "".join(_CHAR_BY_INDEX[i] for i in indices if i != BLANK_INDEX)


def collapse(path: Sequence[int]) -> str:
    """Greedy CTC decode: drop repeats, then drop blanks.

    The standard best-path decode. It is here rather than in the model so that
    it can be tested without a framework, and so the inference pipeline and the
    training evaluation use exactly the same function.
    """
    out: list[str] = []
    previous = None
    for index in path:
        if index != previous and index != BLANK_INDEX:
            char = _CHAR_BY_INDEX.get(int(index))
            if char is not None:
                out.append(char)
        previous = index
    return "".join(out)


def mask_unconfident(text: str, char_scores: Sequence[float], threshold: float) -> str:
    """Replace characters the model is not confident about with ``#``.

    This is where ``#`` is allowed to appear, and the only place. It runs on the
    *output*, after decoding, so an uncertain read is reported as uncertain
    rather than rounded into a confident wrong plate.
    """
    if len(char_scores) != len(text):
        raise AlphabetError(
            f"{len(char_scores)} score(s) for {len(text)} character(s); they must correspond"
        )
    return "".join(
        char if score >= threshold else UNREADABLE
        for char, score in zip(text, char_scores)
    )


__all__ = [
    "ALPHABET",
    "AlphabetError",
    "BLANK_INDEX",
    "DIGITS",
    "LETTERS",
    "MAX_LENGTH",
    "NUM_CLASSES",
    "UNREADABLE",
    "collapse",
    "decode",
    "encode",
    "is_trainable",
    "mask_unconfident",
]
