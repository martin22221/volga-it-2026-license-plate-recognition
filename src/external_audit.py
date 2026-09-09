"""Read-only audit of an external dataset directory.

Inspects a downloaded third-party dataset and reports what is in it, so that a
human can decide whether it may be imported.  **Nothing here writes to the
audited directory, and nothing here imports anything.**  Importing is a
separate, manual step taken only after a license review and a plate-type
review; see ``docs/external_dataset_workflow.md``.

The audit is deliberately incurious about meaning.  It counts YOLO class ids
and reports them as raw integers; it never guesses that class ``0`` is a
license plate, and it never maps an external class onto one of our
``plate_type`` values.  That mapping is a human judgement, recorded when the
source is approved.

Image inspection is **structural**, using stdlib header parsing only, so the
project keeps its no-wheels-required property.  That catches the realistic
failure modes for a downloaded archive -- empty files, wrong or missing magic
bytes, unreadable headers, and truncation -- but it does not decode pixel data,
so corruption confined to the middle of an image stream is not detected.  This
is stated in the report rather than glossed over.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Sequence

logger = logging.getLogger(__name__)

#: Image extensions the audit recognises, lower-cased.
IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png"})

#: Extension treated as a candidate YOLO annotation file.
ANNOTATION_EXTENSION: Final[str] = ".txt"

CSV_EXTENSION: Final[str] = ".csv"

#: File names (lower-cased, without extension) that carry dataset paperwork.
METADATA_STEMS: Final[frozenset[str]] = frozenset(
    {"readme", "license", "licence", "copying", "copyright", "notice", "citation"}
)

#: File names that define YOLO class names rather than annotations.
CLASS_NAME_FILES: Final[frozenset[str]] = frozenset({"classes.txt", "obj.names"})

#: Dataset config files that may declare class names.
CONFIG_NAMES: Final[frozenset[str]] = frozenset(
    {"data.yaml", "dataset.yaml", "data.yml", "dataset.yml"}
)

#: How many bytes of a README/LICENSE to quote in the report, to speed up the
#: human license review without pulling whole files into it.
METADATA_PREVIEW_BYTES: Final[int] = 600

#: Tolerance when checking that a YOLO box lies inside the unit square.
YOLO_EPSILON: Final[float] = 1e-6

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE: Final[bytes] = b"\xff\xd8"

#: JPEG start-of-frame markers, which carry the image dimensions.  C4, C8 and
#: CC are Huffman/arithmetic tables, not frames.
_SOF_MARKERS: Final[frozenset[int]] = frozenset(
    set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
)

_HASH_CHUNK_BYTES: Final[int] = 1 << 20


class AuditError(Exception):
    """The dataset directory cannot be audited at all."""


@dataclass
class ImageRecord:
    """One discovered image file."""

    path: str
    extension: str
    format: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int = 0
    sha256: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        return self.width is not None and self.height is not None

    @property
    def is_corrupt(self) -> bool:
        return bool(self.problems)


@dataclass
class YoloBox:
    """One parsed YOLO annotation line."""

    class_id: int
    coordinates: list[float]


@dataclass
class YoloFile:
    """One candidate YOLO annotation file."""

    path: str
    line_count: int = 0
    boxes: list[YoloBox] = field(default_factory=list)
    invalid_lines: list[dict[str, Any]] = field(default_factory=list)
    is_yolo_format: bool = True

    @property
    def is_empty(self) -> bool:
        """An empty file is a valid YOLO negative (image with no objects)."""
        return self.line_count == 0


@dataclass
class CsvFile:
    """One discovered CSV file. The schema is reported, never assumed."""

    path: str
    header: list[str] = field(default_factory=list)
    row_count: int = 0
    delimiter: str | None = None
    error: str | None = None


@dataclass
class MetadataFile:
    """A README, LICENSE or similar."""

    path: str
    kind: str
    size_bytes: int = 0
    preview: str = ""


@dataclass
class DatasetConfig:
    """Class names declared by the dataset itself, if it declares any."""

    path: str
    class_names: dict[int, str] = field(default_factory=dict)
    declared_class_count: int | None = None
    parse_note: str = ""


@dataclass
class AuditReport:
    """Everything the audit found. Serialised to JSON by :meth:`to_dict`."""

    root: str
    source_id: str | None = None
    images: list[ImageRecord] = field(default_factory=list)
    annotations: list[YoloFile] = field(default_factory=list)
    csv_files: list[CsvFile] = field(default_factory=list)
    metadata_files: list[MetadataFile] = field(default_factory=list)
    configs: list[DatasetConfig] = field(default_factory=list)
    other_files: list[str] = field(default_factory=list)
    unreadable_files: list[dict[str, str]] = field(default_factory=list)

    images_without_annotations: list[str] = field(default_factory=list)
    annotations_without_images: list[str] = field(default_factory=list)
    duplicate_filenames: dict[str, list[str]] = field(default_factory=dict)
    duplicate_images: list[list[str]] = field(default_factory=list)
    ambiguous_stems: dict[str, list[str]] = field(default_factory=dict)

    # -- derived summaries ------------------------------------------------

    @property
    def corrupt_images(self) -> list[ImageRecord]:
        return [image for image in self.images if image.is_corrupt]

    @property
    def readable_images(self) -> list[ImageRecord]:
        return [image for image in self.images if image.is_readable]

    def format_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for image in self.images:
            key = image.format or f"unreadable ({image.extension})"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def dimension_stats(self) -> dict[str, Any]:
        """Width/height statistics over the readable images."""
        readable = self.readable_images
        if not readable:
            return {}

        widths = sorted(image.width for image in readable if image.width is not None)
        heights = sorted(image.height for image in readable if image.height is not None)
        resolutions: dict[str, int] = {}
        for image in readable:
            key = f"{image.width}x{image.height}"
            resolutions[key] = resolutions.get(key, 0) + 1

        return {
            "count": len(readable),
            "width": _numeric_summary(widths),
            "height": _numeric_summary(heights),
            "most_common_resolutions": dict(
                sorted(resolutions.items(), key=lambda item: (-item[1], item[0]))[:10]
            ),
            "distinct_resolutions": len(resolutions),
        }

    def class_counts(self) -> dict[int, int]:
        """How many annotation boxes carry each YOLO class id."""
        counts: dict[int, int] = {}
        for annotation in self.annotations:
            for box in annotation.boxes:
                counts[box.class_id] = counts.get(box.class_id, 0) + 1
        return dict(sorted(counts.items()))

    def class_names(self) -> dict[int, str]:
        """Class names, only if a config file actually declared them."""
        names: dict[int, str] = {}
        for config in self.configs:
            names.update(config.class_names)
        return names

    def undeclared_class_ids(self) -> list[int]:
        """Class ids that appear in annotations but not in any config.

        A non-empty result on a dataset that *has* a config means the config
        and the labels disagree -- treat both as unreliable.
        """
        names = self.class_names()
        if not names:
            return []
        return [class_id for class_id in self.class_counts() if class_id not in names]

    def unused_declared_class_ids(self) -> list[int]:
        """Class ids a config declares but no valid annotation uses."""
        counts = self.class_counts()
        return [class_id for class_id in sorted(self.class_names()) if class_id not in counts]

    def invalid_annotation_count(self) -> int:
        return sum(len(annotation.invalid_lines) for annotation in self.annotations)

    def total_boxes(self) -> int:
        return sum(len(annotation.boxes) for annotation in self.annotations)

    def to_dict(self) -> dict[str, Any]:
        """The machine-readable form of the audit."""
        return {
            "schema_version": 1,
            "root": self.root,
            "source_id": self.source_id,
            "inspection_only": True,
            "image_check": "structural header parsing only; pixel data not decoded",
            "summary": {
                "total_images": len(self.images),
                "readable_images": len(self.readable_images),
                "corrupt_images": len(self.corrupt_images),
                "annotation_files": len(self.annotations),
                "total_annotation_boxes": self.total_boxes(),
                "invalid_annotation_lines": self.invalid_annotation_count(),
                "csv_files": len(self.csv_files),
                "metadata_files": len(self.metadata_files),
                "config_files": len(self.configs),
                "other_files": len(self.other_files),
                "images_without_annotations": len(self.images_without_annotations),
                "annotations_without_images": len(self.annotations_without_images),
                "duplicate_filename_groups": len(self.duplicate_filenames),
                "duplicate_image_groups": len(self.duplicate_images),
            },
            "image_formats": self.format_counts(),
            "dimensions": self.dimension_stats(),
            "classes": {
                "counts_by_id": {str(k): v for k, v in self.class_counts().items()},
                "names_declared_by_dataset": {
                    str(k): v for k, v in self.class_names().items()
                },
                "ids_used_but_not_declared": self.undeclared_class_ids(),
                "ids_declared_but_unused": self.unused_declared_class_ids(),
                "note": (
                    "Class meanings are NOT inferred. Names appear here only if a "
                    "dataset config declared them. Mapping these onto our "
                    "plate_type values is a human decision made at approval time."
                ),
            },
            "images": [asdict(image) for image in self.images],
            "annotations": [asdict(annotation) for annotation in self.annotations],
            "csv_files": [asdict(csv_file) for csv_file in self.csv_files],
            "metadata_files": [asdict(meta) for meta in self.metadata_files],
            "configs": [
                {**asdict(config), "class_names": {str(k): v for k, v in config.class_names.items()}}
                for config in self.configs
            ],
            "other_files": self.other_files,
            "unreadable_files": self.unreadable_files,
            "pairing": {
                "images_without_annotations": self.images_without_annotations,
                "annotations_without_images": self.annotations_without_images,
                "ambiguous_stems": self.ambiguous_stems,
            },
            "duplicates": {
                "filenames": self.duplicate_filenames,
                "identical_images_sha256": self.duplicate_images,
            },
        }


def _numeric_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": values[0],
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 1),
        "max": values[-1],
    }


# --------------------------------------------------------------------------
# Image inspection
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_png_size(data: bytes) -> tuple[int, int]:
    """Width and height from a PNG IHDR chunk."""
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ValueError("PNG header is missing its IHDR chunk")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width == 0 or height == 0:
        raise ValueError("PNG declares a zero dimension")
    return width, height


def read_jpeg_size(data: bytes) -> tuple[int, int]:
    """Width and height from the first JPEG start-of-frame marker."""
    offset = 2
    end = len(data)
    while offset + 3 < end:
        if data[offset] != 0xFF:
            offset += 1  # resynchronise on padding or stray bytes
            continue

        marker = data[offset + 1]
        offset += 2
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue  # standalone markers carry no payload
        if offset + 1 >= end:
            break

        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2:
            raise ValueError("JPEG segment declares an impossible length")

        if marker in _SOF_MARKERS:
            if offset + 7 > end:
                raise ValueError("JPEG frame header is truncated")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width == 0 or height == 0:
                raise ValueError("JPEG declares a zero dimension")
            return width, height

        offset += segment_length

    raise ValueError("no JPEG start-of-frame marker found")


def inspect_image(path: Path, root: Path) -> ImageRecord:
    """Inspect one image file without decoding its pixels."""
    record = ImageRecord(
        path=_relative(path, root), extension=path.suffix.lower()
    )

    try:
        data = path.read_bytes()
    except OSError as error:
        record.problems.append(f"unreadable: {error.strerror or error}")
        return record

    record.size_bytes = len(data)
    record.sha256 = hashlib.sha256(data).hexdigest()

    if not data:
        record.problems.append("file is empty")
        return record

    if data.startswith(PNG_SIGNATURE):
        record.format = "png"
        try:
            record.width, record.height = read_png_size(data)
        except ValueError as error:
            record.problems.append(str(error))
        if not data.rstrip().endswith(b"IEND\xaeB`\x82"):
            record.problems.append("PNG is truncated (no IEND chunk at end of file)")
    elif data.startswith(JPEG_SIGNATURE):
        record.format = "jpeg"
        try:
            record.width, record.height = read_jpeg_size(data)
        except ValueError as error:
            record.problems.append(str(error))
        if not data.rstrip(b"\x00").endswith(b"\xff\xd9"):
            record.problems.append("JPEG is truncated (no EOI marker at end of file)")
    else:
        preview = data[:8].hex()
        record.problems.append(
            f"not a PNG or JPEG: unrecognised magic bytes {preview}"
        )
        return record

    expected = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}.get(record.extension)
    if expected is not None and record.format != expected:
        record.problems.append(
            f"extension {record.extension} does not match actual format {record.format}"
        )

    return record


# --------------------------------------------------------------------------
# YOLO annotations
# --------------------------------------------------------------------------


def parse_yolo_file(path: Path, root: Path) -> YoloFile:
    """Parse and validate a candidate YOLO annotation file.

    Accepts both the bounding-box form (``class cx cy w h``) and the
    segmentation form (``class x1 y1 x2 y2 ...``).  Every coordinate must lie
    in ``[0, 1]``; a bounding box must additionally have positive width and
    height and stay inside the unit square.
    """
    annotation = YoloFile(path=_relative(path, root))

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        annotation.invalid_lines.append(
            {"line": 0, "reason": f"unreadable: {error.strerror or error}", "text": ""}
        )
        annotation.is_yolo_format = False
        return annotation

    parsed_any = False
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        annotation.line_count += 1
        box, reason = _parse_yolo_line(line)
        if box is None:
            annotation.invalid_lines.append(
                {"line": number, "reason": reason, "text": line[:120]}
            )
        else:
            annotation.boxes.append(box)
            parsed_any = True

    # A file where nothing at all parsed is probably not YOLO -- prose, a
    # class list, or some other format -- rather than a broken annotation.
    if annotation.line_count > 0 and not parsed_any:
        annotation.is_yolo_format = False

    return annotation


def _parse_yolo_line(line: str) -> tuple[YoloBox | None, str]:
    fields = line.split()
    if len(fields) < 5:
        return None, f"expected at least 5 fields, found {len(fields)}"

    try:
        class_id = int(fields[0])
    except ValueError:
        return None, f"class id {fields[0]!r} is not an integer"
    if class_id < 0:
        return None, f"class id {class_id} is negative"

    try:
        coordinates = [float(value) for value in fields[1:]]
    except ValueError:
        return None, "coordinates are not all numeric"

    if any(value != value for value in coordinates):
        return None, "coordinates contain NaN"

    out_of_range = [value for value in coordinates if not -YOLO_EPSILON <= value <= 1 + YOLO_EPSILON]
    if out_of_range:
        return None, (
            "coordinates must be normalised to [0, 1]; "
            f"found {out_of_range[0]}"
        )

    if len(coordinates) == 4:
        centre_x, centre_y, width, height = coordinates
        if width <= 0 or height <= 0:
            return None, f"box size ({width} x {height}) must be positive"
        if centre_x - width / 2 < -YOLO_EPSILON or centre_x + width / 2 > 1 + YOLO_EPSILON:
            return None, "box extends beyond the image horizontally"
        if centre_y - height / 2 < -YOLO_EPSILON or centre_y + height / 2 > 1 + YOLO_EPSILON:
            return None, "box extends beyond the image vertically"
    elif len(coordinates) % 2 != 0:
        return None, (
            f"expected 4 box values or an even number of polygon values, "
            f"found {len(coordinates)}"
        )
    elif len(coordinates) < 6:
        return None, "a polygon needs at least 3 points"

    return YoloBox(class_id=class_id, coordinates=coordinates), ""


# --------------------------------------------------------------------------
# Dataset configuration
# --------------------------------------------------------------------------


def parse_dataset_config(path: Path, root: Path) -> DatasetConfig:
    """Extract ``names`` and ``nc`` from a YOLO ``data.yaml``.

    This is a deliberately narrow reader, not a YAML parser: the project has no
    third-party dependencies, and the audit needs exactly two keys.  Anything
    it cannot interpret is reported in ``parse_note`` rather than guessed at.
    """
    config = DatasetConfig(path=_relative(path, root))

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        config.parse_note = f"unreadable: {error.strerror or error}"
        return config

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue

        key, separator, value = stripped.partition(":")
        if not separator:
            index += 1
            continue

        key = key.strip()
        value = value.split("#")[0].strip()

        if key == "nc":
            try:
                config.declared_class_count = int(value)
            except ValueError:
                config.parse_note = f"could not read nc value {value!r}"
        elif key == "names":
            if value.startswith("[") :
                config.class_names = _parse_inline_names(value)
            elif value:
                config.parse_note = f"unsupported names form {value!r}"
            else:
                names, index = _parse_block_names(lines, index)
                config.class_names = names
                continue
        index += 1

    if not config.class_names and not config.parse_note:
        config.parse_note = "no class names declared"
    return config


def _parse_inline_names(value: str) -> dict[int, str]:
    inner = value.strip().lstrip("[").rstrip("]")
    names: dict[int, str] = {}
    for position, item in enumerate(inner.split(",")):
        cleaned = item.strip().strip("'\"")
        if cleaned:
            names[position] = cleaned
    return names


def _parse_block_names(lines: Sequence[str], start: int) -> tuple[dict[int, str], int]:
    """Read an indented ``names:`` block; returns the names and the next index."""
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    names: dict[int, str] = {}
    sequence_index = 0
    index = start + 1

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent:
            break

        item = line.strip()
        if item.startswith("- "):
            names[sequence_index] = item[2:].strip().strip("'\"")
            sequence_index += 1
        else:
            key, separator, value = item.partition(":")
            if separator and key.strip().isdigit():
                names[int(key.strip())] = value.strip().strip("'\"")
        index += 1

    return names, index


def parse_class_names_file(path: Path, root: Path) -> DatasetConfig:
    """Read a ``classes.txt`` / ``obj.names``: one class name per line."""
    config = DatasetConfig(path=_relative(path, root))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        config.parse_note = f"unreadable: {error.strerror or error}"
        return config

    for position, raw in enumerate(
        [line.strip() for line in text.splitlines() if line.strip()]
    ):
        config.class_names[position] = raw

    config.declared_class_count = len(config.class_names)
    if not config.class_names:
        config.parse_note = "file declares no class names"
    return config


# --------------------------------------------------------------------------
# CSV annotations
# --------------------------------------------------------------------------


def inspect_csv(path: Path, root: Path) -> CsvFile:
    """Report a CSV's header and size. The schema is never assumed."""
    import csv as csv_module

    record = CsvFile(path=_relative(path, root))
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as error:
        record.error = f"unreadable: {error.strerror or error}"
        return record

    if not text.strip():
        record.error = "file is empty"
        return record

    first_line = text.splitlines()[0]
    delimiter = max(";,\t|", key=first_line.count)
    if first_line.count(delimiter) == 0:
        delimiter = ","
    record.delimiter = delimiter

    reader = csv_module.reader(text.splitlines(), delimiter=delimiter)
    for position, row in enumerate(reader):
        if position == 0:
            record.header = [cell.strip() for cell in row]
        elif any(cell.strip() for cell in row):
            record.row_count += 1

    return record


# --------------------------------------------------------------------------
# Discovery and audit
# --------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return path.as_posix()


def _classify(path: Path) -> str:
    """Which audit bucket ``path`` belongs to."""
    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()

    if name in CONFIG_NAMES:
        return "config"
    if name in CLASS_NAME_FILES or suffix == ".names":
        return "class_names"
    if stem in METADATA_STEMS:
        return "metadata"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix == ANNOTATION_EXTENSION:
        return "annotation"
    if suffix == CSV_EXTENSION:
        return "csv"
    return "other"


def audit_dataset(
    root: Path,
    *,
    source_id: str | None = None,
    hash_images: bool = True,
) -> AuditReport:
    """Audit the dataset directory at ``root`` without modifying it.

    ``hash_images`` may be turned off to skip SHA-256 duplicate detection on a
    very large dataset; every other check still runs.
    """
    root = Path(root)
    if not root.exists():
        raise AuditError(f"{root} does not exist")
    if not root.is_dir():
        raise AuditError(f"{root} is not a directory")

    report = AuditReport(root=str(root), source_id=source_id)

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            bucket = _classify(path)
            if bucket == "image":
                report.images.append(inspect_image(path, root))
            elif bucket == "annotation":
                report.annotations.append(parse_yolo_file(path, root))
            elif bucket == "csv":
                report.csv_files.append(inspect_csv(path, root))
            elif bucket == "config":
                report.configs.append(parse_dataset_config(path, root))
            elif bucket == "class_names":
                report.configs.append(parse_class_names_file(path, root))
            elif bucket == "metadata":
                report.metadata_files.append(_read_metadata(path, root))
            else:
                report.other_files.append(_relative(path, root))
        except OSError as error:
            report.unreadable_files.append(
                {"path": _relative(path, root), "error": str(error)}
            )

    _pair_images_and_annotations(report)
    _find_duplicate_filenames(report)
    if hash_images:
        _find_duplicate_images(report)

    logger.info(
        "Audited %s: %d image(s), %d annotation file(s)",
        root,
        len(report.images),
        len(report.annotations),
    )
    return report


def _read_metadata(path: Path, root: Path) -> MetadataFile:
    stem = path.stem.lower()
    kind = "readme" if stem == "readme" else "license" if stem in {
        "license",
        "licence",
        "copying",
        "copyright",
    } else stem

    record = MetadataFile(path=_relative(path, root), kind=kind)
    record.size_bytes = path.stat().st_size
    text = path.read_text(encoding="utf-8", errors="replace")
    record.preview = text[:METADATA_PREVIEW_BYTES].strip()
    return record


def _pair_images_and_annotations(report: AuditReport) -> None:
    """Match images to annotations by file stem.

    YOLO datasets conventionally place ``images/x.jpg`` beside
    ``labels/x.txt``, so the stem is the reliable key.  A stem used by more
    than one image is reported as ambiguous rather than silently paired.
    """
    images_by_stem: dict[str, list[str]] = {}
    for image in report.images:
        images_by_stem.setdefault(Path(image.path).stem, []).append(image.path)

    annotations_by_stem: dict[str, list[str]] = {}
    for annotation in report.annotations:
        if annotation.is_yolo_format:
            annotations_by_stem.setdefault(
                Path(annotation.path).stem, []
            ).append(annotation.path)

    for stem, paths in sorted(images_by_stem.items()):
        if stem not in annotations_by_stem:
            report.images_without_annotations.extend(sorted(paths))
        if len(paths) > 1:
            report.ambiguous_stems[stem] = sorted(paths)

    for stem, paths in sorted(annotations_by_stem.items()):
        if stem not in images_by_stem:
            report.annotations_without_images.extend(sorted(paths))

    report.images_without_annotations.sort()
    report.annotations_without_images.sort()


def _find_duplicate_filenames(report: AuditReport) -> None:
    by_name: dict[str, list[str]] = {}
    for image in report.images:
        by_name.setdefault(Path(image.path).name, []).append(image.path)

    report.duplicate_filenames = {
        name: sorted(paths)
        for name, paths in sorted(by_name.items())
        if len(paths) > 1
    }


def _find_duplicate_images(report: AuditReport) -> None:
    by_hash: dict[str, list[str]] = {}
    for image in report.images:
        if image.sha256 is not None:
            by_hash.setdefault(image.sha256, []).append(image.path)

    report.duplicate_images = [
        sorted(paths) for _, paths in sorted(by_hash.items()) if len(paths) > 1
    ]


def write_json_report(report: AuditReport, destination: Path) -> None:
    """Write the machine-readable audit to ``destination``."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
