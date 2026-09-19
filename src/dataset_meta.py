"""Dataset metadata: schema, parsing and validation of ``dataset/meta.csv``.

``meta.csv`` is the single source of truth for the training dataset we build
ourselves.  One row is one *annotation*, not one image: an image showing three
plates contributes three rows that share the same ``image`` value.

Format (UTF-8, semicolon separated, header required)::

    image;plate_num;plate_type;bbox_x;bbox_y;bbox_w;bbox_h;quad_x1;quad_y1;...

``image`` is a path relative to the dataset root and must live under
``images/real/`` or ``images/synthetic/``.  ``conditions`` holds zero or more
tags from :data:`ALLOWED_CONDITIONS`, separated by ``|``.

A *background row* -- an image deliberately kept as a negative example with no
plate on it at all -- is written as ``plate_type=other``, an empty
``plate_num`` and every geometry column set to ``0``.  Geometry checks are
skipped for such rows; see :meth:`MetaRow.is_background`.

This module only reads and checks metadata.  The human-readable report is
produced by ``scripts/validate_dataset_local.py``.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, Iterable, Iterator, Sequence

from src.validator import (
    ALLOWED_LETTERS,
    FORMAT_BY_PLATE_TYPE,
    TYPE1B_FORMAT,
    TYPE1B_REGION_LENGTHS,
    validate_plate,
)

logger = logging.getLogger(__name__)

CSV_DELIMITER: Final[str] = ";"
CSV_ENCODING: Final[str] = "utf-8-sig"

#: Separators accepted between individual tags inside ``conditions``.
#: The first one is canonical; the rest are tolerated on input.
CONDITION_SEPARATORS: Final[str] = "|,"

#: Every column ``meta.csv`` must declare, in canonical order.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "image",
    "plate_num",
    "plate_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "quad_x1",
    "quad_y1",
    "quad_x2",
    "quad_y2",
    "quad_x3",
    "quad_y3",
    "quad_x4",
    "quad_y4",
    "is_vehicle",
    "is_synthetic",
    "source",
    "license",
    "conditions",
)

#: Plate classes accepted in the ``plate_type`` column.
ALLOWED_PLATE_TYPES: Final[frozenset[str]] = frozenset(
    {"type1", "type1a", "type1b", "other"}
)

#: Plate classes whose ``plate_num`` must be filled in.
TEXT_BEARING_TYPES: Final[frozenset[str]] = frozenset({"type1", "type1a", "type1b"})

#: Controlled vocabulary for the ``conditions`` column.
ALLOWED_CONDITIONS: Final[tuple[str, ...]] = (
    "day",
    "night",
    "rain",
    "snow",
    "dirt",
    "glare",
    "motion_blur",
    "angle",
)

BOOLEAN_COLUMNS: Final[tuple[str, ...]] = ("is_vehicle", "is_synthetic")
BBOX_COLUMNS: Final[tuple[str, ...]] = ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
QUAD_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"quad_{axis}{corner}" for corner in (1, 2, 3, 4) for axis in ("x", "y")
)

#: Columns whose blank value is reported, but only as a warning.
PROVENANCE_COLUMNS: Final[tuple[str, ...]] = ("source", "license")

#: The license the finished dataset is submitted and published under.
DATASET_LICENSE: Final[str] = "CC BY 4.0"

#: Sources ruled out for the submission, mapped to the reason.  A row citing
#: one of these is an error even when its ``license`` column is otherwise
#: acceptable: the rejection is about the *provenance* of the images, which no
#: license string can repair.  Keep in step with ``docs/data_sources.md``.
REJECTED_SOURCES: Final[dict[str, str]] = {
    "roboflow_two_line_russian_license_plates": (
        "REJECTED_FOR_SUBMISSION_PROVENANCE -- the Roboflow project declares "
        "CC BY 4.0, but no evidence establishes the provenance, ownership or "
        "licensing chain of the underlying 27 photographs. Reference and "
        "visual review only; not competition training data"
    ),
}

#: Licenses under which a third-party image may be redistributed as part of a
#: CC BY 4.0 dataset.  Attribution-only and public-domain terms qualify;
#: everything else has to be confirmed by a human and added here explicitly.
REDISTRIBUTABLE_LICENSES: Final[frozenset[str]] = frozenset(
    {
        "own work",
        "public domain",
        "cc0",
        "cc0 1.0",
        "cc by 1.0",
        "cc by 2.0",
        "cc by 2.5",
        "cc by 3.0",
        "cc by 4.0",
    }
)

#: License tokens that rule an image out of a CC BY 4.0 dataset.  ``nc``
#: forbids commercial use, ``nd`` forbids the derivatives we make when we crop
#: and blur, and ``sa`` forces a ShareAlike license we cannot grant.
INCOMPATIBLE_LICENSE_TOKENS: Final[dict[str, str]] = {
    "nc": "non-commercial",
    "noncommercial": "non-commercial",
    "nd": "no-derivatives",
    "noderivatives": "no-derivatives",
    "noderivs": "no-derivatives",
    "sa": "share-alike",
    "sharealike": "share-alike",
    # Copyleft licences outside the Creative Commons family. They permit
    # redistribution and then require the result to carry the same licence,
    # which is the ShareAlike problem under another name: our dataset is
    # published under plain CC BY 4.0 and cannot carry them. Wikimedia Commons
    # serves a steady trickle of both, so naming them beats leaving them to
    # fall through as "unrecognised".
    "gfdl": "copyleft (GNU Free Documentation License)",
    "gpl": "copyleft (GNU General Public License)",
    "lgpl": "copyleft (GNU Lesser General Public License)",
    "fal": "copyleft (Free Art License)",
}

TRUE_VALUES: Final[frozenset[str]] = frozenset({"true", "1", "yes", "y"})
FALSE_VALUES: Final[frozenset[str]] = frozenset({"false", "0", "no", "n"})

#: Directory prefixes ``image`` may start with, one per ``is_synthetic`` value.
REAL_PREFIX: Final[str] = "images/real/"
SYNTHETIC_PREFIX: Final[str] = "images/synthetic/"

#: Character standing in for a plate glyph that cannot be read confidently.
UNREADABLE_CHAR: Final[str] = "#"

#: Deviation between the quad extent and the bbox tolerated before a warning,
#: as a fraction of the corresponding bbox side.
QUAD_BBOX_TOLERANCE: Final[float] = 0.25

#: File names written by synthetic generator V1 (``syn_<seed>_<index>.jpg``).
#: V1 was superseded on 2026-09-13: its ``type1b`` plates used the type 1
#: character structure instead of GOST's ``MM 000 55``.  Its output is a
#: development artefact and must never enter the dataset.  V2 and later name
#: files ``syn_v<major>_<seed>_<index>.jpg``.
SUPERSEDED_SYNTHETIC_NAME: Final[re.Pattern[str]] = re.compile(r"^syn_\d+_\d{5}\.jpg$")

_WILDCARD_LETTER: Final[str] = "[" + "".join(sorted(ALLOWED_LETTERS)) + re.escape(UNREADABLE_CHAR) + "]"
_WILDCARD_DIGIT: Final[str] = r"[\d" + re.escape(UNREADABLE_CHAR) + "]"
#: Structure of each format with ``#`` allowed in any position.
_MASKED_STRUCTURE: Final[dict[str, re.Pattern[str]]] = {
    "type1": re.compile(rf"^{_WILDCARD_LETTER}{_WILDCARD_DIGIT}{{3}}{_WILDCARD_LETTER}{{2}}{_WILDCARD_DIGIT}{{2,3}}$"),
    TYPE1B_FORMAT: re.compile(
        rf"^{_WILDCARD_LETTER}{{2}}{_WILDCARD_DIGIT}{{3}}"
        rf"{_WILDCARD_DIGIT}{{{min(TYPE1B_REGION_LENGTHS)},{max(TYPE1B_REGION_LENGTHS)}}}$"
    ),
}


class Severity(str, Enum):
    """How badly a finding compromises the dataset."""

    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    ``row`` is the 1-based line number in ``meta.csv`` as a spreadsheet shows
    it (the header is line 1), or ``None`` for file-level findings.
    """

    severity: Severity
    message: str
    row: int | None = None
    column: str | None = None

    def __str__(self) -> str:
        where = f"line {self.row}" if self.row is not None else "meta.csv"
        if self.column:
            where = f"{where}, column {self.column!r}"
        return f"[{self.severity}] {where}: {self.message}"


@dataclass(frozen=True)
class MetaRow:
    """One parsed annotation row.

    Cells are kept as raw strings; typed access goes through the helpers below
    so that a malformed row is reported rather than crashing the reader.

    ``values`` is always padded to the header, so ``cell_count`` records how
    many cells the line actually held -- ``None`` when it was not tracked.
    """

    line: int
    values: dict[str, str]
    cell_count: int | None = None

    def get(self, column: str) -> str:
        return self.values.get(column, "").strip()

    @property
    def image(self) -> str:
        """The image path, with Windows separators folded to ``/``."""
        return self.get("image").replace("\\", "/")

    @property
    def plate_num(self) -> str:
        return self.get("plate_num")

    @property
    def plate_type(self) -> str:
        return self.get("plate_type")

    @property
    def conditions(self) -> list[str]:
        return split_conditions(self.get("conditions"))

    @property
    def is_synthetic(self) -> bool | None:
        return parse_bool(self.get("is_synthetic"))

    @property
    def is_vehicle(self) -> bool | None:
        return parse_bool(self.get("is_vehicle"))

    def numbers(self, columns: Sequence[str]) -> list[float] | None:
        """Return ``columns`` as floats, or ``None`` if any of them is not."""
        parsed: list[float] = []
        for column in columns:
            value = parse_number(self.get(column))
            if value is None:
                return None
            parsed.append(value)
        return parsed

    def is_background(self) -> bool:
        """Whether this row is a negative example carrying no plate geometry."""
        if self.plate_type != "other":
            return False
        geometry = self.numbers(BBOX_COLUMNS + QUAD_COLUMNS)
        return geometry is not None and all(value == 0.0 for value in geometry)

    def identity(self) -> tuple[str, ...]:
        """The tuple used to detect fully duplicated rows."""
        return tuple(self.get(column) for column in REQUIRED_COLUMNS)


@dataclass
class DatasetStats:
    """Counts summarising a ``meta.csv``, used to build the report."""

    total_rows: int = 0
    total_images: int = 0
    real_images: int = 0
    synthetic_images: int = 0
    real_rows: int = 0
    synthetic_rows: int = 0
    background_rows: int = 0
    by_plate_type: dict[str, int] = field(default_factory=dict)
    by_condition: dict[str, int] = field(default_factory=dict)
    rows_without_conditions: int = 0
    unique_plates_by_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_license: dict[str, int] = field(default_factory=dict)
    rows_missing_source: int = 0
    rows_missing_license: int = 0
    missing_files: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Everything :func:`validate_meta` found."""

    meta_path: Path
    dataset_root: Path
    stats: DatasetStats = field(default_factory=DatasetStats)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: Severity,
        message: str,
        row: int | None = None,
        column: str | None = None,
    ) -> None:
        self.issues.append(Issue(severity, message, row, column))


class MetaFormatError(Exception):
    """``meta.csv`` is missing or cannot be parsed as a CSV file at all."""


def parse_bool(value: str) -> bool | None:
    """Parse a boolean cell, returning ``None`` when it is not recognised."""
    lowered = value.strip().lower()
    if lowered in TRUE_VALUES:
        return True
    if lowered in FALSE_VALUES:
        return False
    return None


def parse_number(value: str) -> float | None:
    """Parse a numeric cell, tolerating a decimal comma.

    Returns ``None`` for anything that is not a finite number.
    """
    text = value.strip().replace(",", ".")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def split_conditions(value: str) -> list[str]:
    """Split a ``conditions`` cell into its individual lower-cased tags."""
    canonical, *alternatives = CONDITION_SEPARATORS
    normalized = value
    for separator in alternatives:
        normalized = normalized.replace(separator, canonical)
    return [tag.strip().lower() for tag in normalized.split(canonical) if tag.strip()]


def normalize_license(value: str) -> str:
    """Reduce a license cell to a comparable form.

    Lower-cases, drops a ``creative commons`` prefix and the ``international``
    suffix, and turns punctuation into single spaces, so that
    ``"CC-BY-4.0"``, ``"cc by 4.0"`` and
    ``"Creative Commons Attribution 4.0 International"`` all collapse together.
    """
    text = value.strip().lower()
    if not text:
        return ""

    for old, new in (
        ("creative commons", "cc"),
        ("attribution", "by"),
        ("international", ""),
        ("public domain dedication", "public domain"),
    ):
        text = text.replace(old, new)

    for character in "-_/,()":
        text = text.replace(character, " ")

    return " ".join(text.split())


def license_tokens(value: str) -> list[str]:
    """The individual tokens of a normalised license string."""
    return normalize_license(value).split()


def license_incompatibility(value: str) -> str | None:
    """Name the clause that bars ``value`` from a CC BY 4.0 dataset.

    Returns ``None`` when no disqualifying clause is present; that is *not* a
    promise the license is compatible, only that it is not obviously not.
    """
    for token in license_tokens(value):
        clause = INCOMPATIBLE_LICENSE_TOKENS.get(token)
        if clause is not None:
            return clause
    return None


def is_redistributable_license(value: str) -> bool:
    """Whether ``value`` is a license we have confirmed we may redistribute."""
    return normalize_license(value) in REDISTRIBUTABLE_LICENSES


def polygon_signed_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace signed area of ``points``.

    Image coordinates run top-down, so a polygon listed *clockwise on screen*
    yields a positive area here.
    """
    total = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def read_meta(meta_path: Path) -> tuple[list[str], list[MetaRow]]:
    """Read ``meta_path`` into its header and rows.

    Raises :class:`MetaFormatError` if the file is absent or has no header.
    """
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        raise MetaFormatError(f"{meta_path} does not exist")

    with meta_path.open("r", encoding=CSV_ENCODING, newline="") as handle:
        reader = csv.reader(handle, delimiter=CSV_DELIMITER)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise MetaFormatError(f"{meta_path} is empty (no header row)") from None

        header = [name.strip() for name in raw_header]
        rows: list[MetaRow] = []
        for line, raw in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw):
                continue  # tolerate blank separator lines
            values = {
                name: raw[index] if index < len(raw) else ""
                for index, name in enumerate(header)
            }
            rows.append(MetaRow(line=line, values=values, cell_count=len(raw)))

    logger.info("Read %d row(s) from %s", len(rows), meta_path)
    return header, rows


def validate_meta(
    meta_path: Path,
    dataset_root: Path | None = None,
    *,
    check_files: bool = True,
) -> ValidationReport:
    """Validate ``meta.csv`` and return everything found.

    ``dataset_root`` defaults to the directory holding ``meta.csv``; image
    paths are resolved against it.  Set ``check_files`` to ``False`` to skip
    the filesystem existence check, e.g. when only the schema matters.
    """
    meta_path = Path(meta_path)
    root = Path(dataset_root) if dataset_root is not None else meta_path.parent
    report = ValidationReport(meta_path=meta_path, dataset_root=root)

    header, rows = read_meta(meta_path)
    if not _check_header(header, report):
        return report

    _check_rows(rows, header, root, report, check_files=check_files)
    _check_duplicates(rows, report)
    _collect_stats(rows, report)
    return report


def _check_header(header: Sequence[str], report: ValidationReport) -> bool:
    """Check the column set; returns whether per-row checks can proceed."""
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        report.add(Severity.ERROR, f"missing required column(s): {', '.join(missing)}")

    unknown = [column for column in header if column not in REQUIRED_COLUMNS]
    if unknown:
        report.add(
            Severity.WARNING, f"unrecognised column(s) ignored: {', '.join(unknown)}"
        )

    duplicated = sorted({name for name in header if header.count(name) > 1})
    if duplicated:
        report.add(
            Severity.ERROR,
            f"column(s) declared more than once: {', '.join(duplicated)}",
        )

    if not missing and list(header[: len(REQUIRED_COLUMNS)]) != list(REQUIRED_COLUMNS):
        report.add(
            Severity.WARNING,
            "columns are not in the canonical order defined by REQUIRED_COLUMNS",
        )

    return not missing


def _check_rows(
    rows: Iterable[MetaRow],
    header: Sequence[str],
    root: Path,
    report: ValidationReport,
    *,
    check_files: bool,
) -> None:
    for row in rows:
        _check_row_width(row, header, report)
        _check_image_path(row, root, report, check_files=check_files)
        _check_plate_type(row, report)
        _check_plate_num(row, report)
        _check_booleans(row, report)
        _check_conditions(row, report)
        _check_provenance(row, report)
        if not row.is_background():
            _check_geometry(row, report)


def _check_row_width(
    row: MetaRow, header: Sequence[str], report: ValidationReport
) -> None:
    if row.cell_count is None or row.cell_count == len(header):
        return
    report.add(
        Severity.ERROR,
        f"row has {row.cell_count} cell(s), header declares {len(header)}",
        row.line,
    )


def _check_image_path(
    row: MetaRow, root: Path, report: ValidationReport, *, check_files: bool
) -> None:
    image = row.image
    if not image:
        report.add(Severity.ERROR, "image path is empty", row.line, "image")
        return

    if Path(image).is_absolute() or ".." in Path(image).parts:
        report.add(
            Severity.ERROR,
            f"image path {image!r} must be relative to the dataset root "
            "and stay inside it",
            row.line,
            "image",
        )
        return

    in_real = image.startswith(REAL_PREFIX)
    in_synthetic = image.startswith(SYNTHETIC_PREFIX)
    if not (in_real or in_synthetic):
        report.add(
            Severity.ERROR,
            f"image path {image!r} must start with {REAL_PREFIX!r} "
            f"or {SYNTHETIC_PREFIX!r}",
            row.line,
            "image",
        )
        return

    synthetic = row.is_synthetic
    if in_synthetic and SUPERSEDED_SYNTHETIC_NAME.fullmatch(Path(image).name):
        report.add(
            Severity.ERROR,
            f"{image!r} was written by synthetic generator V1, which was superseded "
            "(its type1b plates used the wrong character structure); V1 output is a "
            "development artefact and must not enter the dataset",
            row.line,
            "image",
        )
    if synthetic is True and in_real:
        report.add(
            Severity.WARNING,
            f"{image!r} is stored as a real image but is_synthetic=true",
            row.line,
            "is_synthetic",
        )
    if synthetic is False and in_synthetic:
        report.add(
            Severity.WARNING,
            f"{image!r} is stored as a synthetic image but is_synthetic=false",
            row.line,
            "is_synthetic",
        )

    if check_files and not (root / image).is_file():
        report.add(Severity.ERROR, f"image file {image!r} not found", row.line, "image")
        report.stats.missing_files.append(image)


def _check_plate_type(row: MetaRow, report: ValidationReport) -> None:
    plate_type = row.plate_type
    if not plate_type:
        report.add(Severity.ERROR, "plate_type is empty", row.line, "plate_type")
    elif plate_type not in ALLOWED_PLATE_TYPES:
        allowed = ", ".join(sorted(ALLOWED_PLATE_TYPES))
        report.add(
            Severity.ERROR,
            f"plate_type {plate_type!r} is not one of: {allowed}",
            row.line,
            "plate_type",
        )


def _check_plate_num(row: MetaRow, report: ValidationReport) -> None:
    plate_num = row.plate_num

    if row.plate_type in TEXT_BEARING_TYPES and not plate_num:
        report.add(
            Severity.ERROR,
            f"plate_num is empty for plate_type {row.plate_type!r}",
            row.line,
            "plate_num",
        )
        return

    if plate_num and set(plate_num) == {UNREADABLE_CHAR}:
        report.add(
            Severity.WARNING,
            "every character of plate_num is unreadable; the plate is probably "
            "too small or too occluded to annotate",
            row.line,
            "plate_num",
        )

    if row.is_synthetic is True and plate_num and row.plate_type in FORMAT_BY_PLATE_TYPE:
        _check_synthetic_grammar(row, report)


def _check_synthetic_grammar(row: MetaRow, report: ValidationReport) -> None:
    """A generated label must follow its plate type's structure exactly.

    Only synthetic rows are checked: the generator knows the type it drew,
    so a mismatch is always a generator bug -- e.g. V1's ``type1b`` plates in
    the type 1 structure.  Real annotations record what a person saw and are
    not second-guessed here.
    """
    plate_num, plate_type = row.plate_num, row.plate_type
    fmt = FORMAT_BY_PLATE_TYPE[plate_type]
    if UNREADABLE_CHAR in plate_num:
        if _MASKED_STRUCTURE[fmt].fullmatch(plate_num):
            return
        reason = "does not fit the structure even allowing for '#'"
    else:
        result = validate_plate(plate_num, plate_type)
        if result.is_valid:
            return
        reason = result.reason
    report.add(
        Severity.ERROR,
        f"synthetic {plate_type} plate_num {plate_num!r} violates the {plate_type} "
        f"structure: {reason}",
        row.line,
        "plate_num",
    )


def _check_booleans(row: MetaRow, report: ValidationReport) -> None:
    for column in BOOLEAN_COLUMNS:
        raw = row.get(column)
        if not raw:
            report.add(Severity.ERROR, f"{column} is empty", row.line, column)
            continue
        if parse_bool(raw) is None:
            accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
            report.add(
                Severity.ERROR,
                f"{column}={raw!r} is not a boolean (accepted: {accepted})",
                row.line,
                column,
            )


def _check_conditions(row: MetaRow, report: ValidationReport) -> None:
    tags = row.conditions
    if not tags:
        return

    for tag in tags:
        if tag not in ALLOWED_CONDITIONS:
            allowed = ", ".join(ALLOWED_CONDITIONS)
            report.add(
                Severity.ERROR,
                f"condition {tag!r} is not in the controlled list: {allowed}",
                row.line,
                "conditions",
            )

    duplicated = sorted({tag for tag in tags if tags.count(tag) > 1})
    if duplicated:
        report.add(
            Severity.WARNING,
            f"condition(s) repeated: {', '.join(duplicated)}",
            row.line,
            "conditions",
        )

    if "day" in tags and "night" in tags:
        report.add(
            Severity.ERROR,
            "conditions cannot be both 'day' and 'night'",
            row.line,
            "conditions",
        )


def _check_provenance(row: MetaRow, report: ValidationReport) -> None:
    if row.is_synthetic is True:
        return  # synthetic images are ours; the generator config is the source

    for column in PROVENANCE_COLUMNS:
        if not row.get(column):
            report.add(
                Severity.WARNING,
                f"{column} is blank for a real image; every real image needs "
                "a documented source and license",
                row.line,
                column,
            )

    _check_source_not_rejected(row, report)
    _check_license_permits_redistribution(row, report)


def _check_source_not_rejected(row: MetaRow, report: ValidationReport) -> None:
    """Block any image from a source we have ruled out.

    A rejected source stays rejected regardless of what its ``license`` column
    says -- the objection is to the provenance of the images themselves, and a
    license string cannot answer it.  Without this the decision would rest on
    everyone remembering it.
    """
    reason = REJECTED_SOURCES.get(row.get("source"))
    if reason is not None:
        report.add(
            Severity.ERROR,
            f"source {row.get('source')!r} is rejected: {reason}",
            row.line,
            "source",
        )


def _check_license_permits_redistribution(
    row: MetaRow, report: ValidationReport
) -> None:
    """Check a real image's license against the dataset's own license.

    The dataset is submitted and published under :data:`DATASET_LICENSE`, so an
    image we may *use* but not *redistribute* cannot be in it.  A clause that
    definitely bars redistribution is an error; a license we simply do not
    recognise is a warning, because it needs a person to read the terms rather
    than an assumption either way.
    """
    license_name = row.get("license")
    if not license_name:
        return  # already reported as blank provenance

    clause = license_incompatibility(license_name)
    if clause is not None:
        report.add(
            Severity.ERROR,
            f"license {license_name!r} carries a {clause} clause and cannot be "
            f"redistributed as part of a {DATASET_LICENSE} dataset; "
            "remove the image or replace the source",
            row.line,
            "license",
        )
        return

    if not is_redistributable_license(license_name):
        report.add(
            Severity.WARNING,
            f"license {license_name!r} is not on the confirmed-redistributable "
            f"list; verify it allows redistribution under {DATASET_LICENSE}, "
            "record the check in docs/data_sources.md, and add it to "
            "REDISTRIBUTABLE_LICENSES",
            row.line,
            "license",
        )


def _check_geometry(row: MetaRow, report: ValidationReport) -> None:
    bbox = row.numbers(BBOX_COLUMNS)
    if bbox is None:
        report.add(
            Severity.ERROR,
            f"bbox columns must all be numeric: {', '.join(BBOX_COLUMNS)}",
            row.line,
        )
    else:
        x, y, width, height = bbox
        if x < 0 or y < 0:
            report.add(
                Severity.ERROR,
                f"bbox origin ({x}, {y}) must be non-negative",
                row.line,
            )
        if width <= 0 or height <= 0:
            report.add(
                Severity.ERROR,
                f"bbox size ({width} x {height}) must be strictly positive",
                row.line,
            )

    quad = row.numbers(QUAD_COLUMNS)
    if quad is None:
        report.add(
            Severity.ERROR,
            f"quad columns must all be numeric: {', '.join(QUAD_COLUMNS)}",
            row.line,
        )
        return

    corners = [(quad[index], quad[index + 1]) for index in range(0, 8, 2)]
    if any(value < 0 for value in quad):
        report.add(Severity.ERROR, "quad coordinates must be non-negative", row.line)

    if len(set(corners)) != len(corners):
        report.add(Severity.ERROR, "quad has repeated corners", row.line)
        return

    area = polygon_signed_area(corners)
    if area == 0.0:
        report.add(Severity.ERROR, "quad is degenerate (zero area)", row.line)
        return
    if area < 0:
        report.add(
            Severity.ERROR,
            "quad corners are counter-clockwise; list them clockwise starting "
            "at the top-left corner of the plate",
            row.line,
        )

    if bbox is not None:
        _check_quad_matches_bbox(corners, bbox, row, report)


def _check_quad_matches_bbox(
    corners: Sequence[tuple[float, float]],
    bbox: Sequence[float],
    row: MetaRow,
    report: ValidationReport,
) -> None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return  # already reported; the ratios below would be meaningless

    quad_x = [corner[0] for corner in corners]
    quad_y = [corner[1] for corner in corners]
    deviations = (
        abs(min(quad_x) - x) / width,
        abs(max(quad_x) - (x + width)) / width,
        abs(min(quad_y) - y) / height,
        abs(max(quad_y) - (y + height)) / height,
    )
    if max(deviations) > QUAD_BBOX_TOLERANCE:
        report.add(
            Severity.WARNING,
            "quad extent disagrees with bbox by more than "
            f"{QUAD_BBOX_TOLERANCE:.0%} of a side; one of them is likely wrong",
            row.line,
        )


def _check_duplicates(rows: Sequence[MetaRow], report: ValidationReport) -> None:
    seen_rows: dict[tuple[str, ...], int] = {}
    seen_annotations: dict[tuple[str, str], int] = {}
    seen_basenames: dict[str, str] = {}

    for row in rows:
        identity = row.identity()
        first_seen = seen_rows.get(identity)
        if first_seen is not None:
            report.add(
                Severity.ERROR,
                f"row is an exact duplicate of line {first_seen}",
                row.line,
            )
        else:
            seen_rows[identity] = row.line

        annotation = (row.image, row.plate_num)
        first_seen = seen_annotations.get(annotation)
        if first_seen is not None:
            report.add(
                Severity.ERROR,
                f"plate {row.plate_num!r} is annotated twice on {row.image!r} "
                f"(first at line {first_seen})",
                row.line,
            )
        else:
            seen_annotations[annotation] = row.line

        basename = Path(row.image).name
        if not basename:
            continue
        previous = seen_basenames.setdefault(basename, row.image)
        if previous != row.image:
            report.add(
                Severity.WARNING,
                f"file name {basename!r} is used by two different paths "
                f"({previous!r} and {row.image!r}); keep file names unique",
                row.line,
                "image",
            )


def _collect_stats(rows: Sequence[MetaRow], report: ValidationReport) -> None:
    stats = report.stats
    stats.total_rows = len(rows)

    real_images: set[str] = set()
    synthetic_images: set[str] = set()
    plates_by_type: dict[str, set[str]] = {}

    for row in rows:
        synthetic = row.is_synthetic
        if synthetic is True:
            stats.synthetic_rows += 1
            synthetic_images.add(row.image)
        elif synthetic is False:
            stats.real_rows += 1
            real_images.add(row.image)

        if row.is_background():
            stats.background_rows += 1

        plate_type = row.plate_type or "<blank>"
        stats.by_plate_type[plate_type] = stats.by_plate_type.get(plate_type, 0) + 1
        if row.plate_num:
            plates_by_type.setdefault(plate_type, set()).add(row.plate_num)

        tags = row.conditions
        if not tags:
            stats.rows_without_conditions += 1
        for tag in set(tags):
            stats.by_condition[tag] = stats.by_condition.get(tag, 0) + 1

        source = row.get("source")
        license_name = row.get("license")
        if source:
            stats.by_source[source] = stats.by_source.get(source, 0) + 1
        elif synthetic is not True:
            stats.rows_missing_source += 1
        if license_name:
            stats.by_license[license_name] = stats.by_license.get(license_name, 0) + 1
        elif synthetic is not True:
            stats.rows_missing_license += 1

    stats.real_images = len(real_images)
    stats.synthetic_images = len(synthetic_images)
    stats.total_images = len(real_images | synthetic_images)
    stats.unique_plates_by_type = {
        plate_type: len(plates) for plate_type, plates in sorted(plates_by_type.items())
    }


def iter_issues(report: ValidationReport, severity: Severity) -> Iterator[Issue]:
    """Yield ``report`` issues of exactly ``severity``, ordered by line."""
    matching = [issue for issue in report.issues if issue.severity is severity]
    yield from sorted(matching, key=lambda issue: (issue.row or 0, issue.message))
