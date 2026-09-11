"""Tests for the external dataset audit tool."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from scripts import audit_external_dataset as cli
from src.external_audit import (
    AuditError,
    audit_dataset,
    parse_dataset_config,
    parse_yolo_file,
    read_bmp_size,
    read_jpeg_size,
    read_png_size,
    sha256_file,
)

# --------------------------------------------------------------------------
# Fixtures: real PNG and JPEG bytes, built with the standard library
# --------------------------------------------------------------------------


def png_bytes(width: int, height: int) -> bytes:
    """A structurally valid greyscale PNG of the given size."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def jpeg_bytes(width: int, height: int) -> bytes:
    """A JPEG with a valid SOI, APP0, SOF0 and EOI. Not decodable, but the
    audit only parses headers."""
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = (
        b"\xff\xc0"
        + struct.pack(">H", 11)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x01\x01\x11\x00"
    )
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00" + b"\xff\xd9"


def bmp_bytes(width: int, height: int, *, top_down: bool = False, core: bool = False) -> bytes:
    """A structurally valid 24-bit BMP.

    ``core`` emits the legacy 12-byte BITMAPCOREHEADER instead of the usual
    40-byte BITMAPINFOHEADER; ``top_down`` stores a negative height, which is a
    row order rather than a different size.
    """
    row_stride = (width * 3 + 3) & ~3
    pixels = b"\x00" * (row_stride * height)

    if core:
        dib = struct.pack("<IHHHH", 12, width, height, 1, 24)
    else:
        stored_height = -height if top_down else height
        dib = struct.pack(
            "<IiiHHIIiiII", 40, width, stored_height, 1, 24, 0, len(pixels), 0, 0, 0, 0
        )

    offset = 14 + len(dib)
    header = b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
    return header + dib + pixels


def write_image(path: Path, width: int = 640, height: int = 480, *, fmt: str = "jpg") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "png":
        data = png_bytes(width, height)
    elif fmt == "bmp":
        data = bmp_bytes(width, height)
    else:
        data = jpeg_bytes(width, height)
    path.write_bytes(data)
    return path


def write_label(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """A small, well-formed YOLO dataset: 3 images, 3 labels, a config."""
    root = tmp_path / "external"
    for index in range(1, 4):
        write_image(root / "images" / f"img{index}.jpg", 640 + index, 480)
        write_label(
            root / "labels" / f"img{index}.txt",
            [f"0 0.5 0.5 0.2 0.1", "1 0.25 0.25 0.1 0.1"],
        )
    (root / "data.yaml").write_text(
        "path: .\ntrain: images\nnc: 2\nnames: ['plate', 'vehicle']\n", encoding="utf-8"
    )
    (root / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International\nYou are free to share.\n",
        encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------
# Header parsing
# --------------------------------------------------------------------------


def test_png_size_is_read_from_ihdr() -> None:
    assert read_png_size(png_bytes(123, 45)) == (123, 45)


def test_jpeg_size_is_read_from_sof() -> None:
    assert read_jpeg_size(jpeg_bytes(321, 65)) == (321, 65)


def test_jpeg_without_frame_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="start-of-frame"):
        read_jpeg_size(b"\xff\xd8\xff\xd9")


# --------------------------------------------------------------------------
# BMP support
# --------------------------------------------------------------------------


def test_bmp_size_is_read_from_info_header() -> None:
    assert read_bmp_size(bmp_bytes(200, 120)) == (200, 120)


def test_bmp_size_is_read_from_legacy_core_header() -> None:
    assert read_bmp_size(bmp_bytes(64, 32, core=True)) == (64, 32)


def test_top_down_bmp_reports_positive_height() -> None:
    """A negative stored height is a row order, not a negative size."""
    assert read_bmp_size(bmp_bytes(80, 40, top_down=True)) == (80, 40)


def test_bmp_too_short_is_rejected() -> None:
    with pytest.raises(ValueError, match="too short"):
        read_bmp_size(b"BM" + b"\x00" * 8)


def test_bmp_with_unsupported_header_size_is_rejected() -> None:
    data = bytearray(bmp_bytes(10, 10))
    data[14:18] = (20).to_bytes(4, "little")  # neither 12 nor >= 40
    with pytest.raises(ValueError, match="unsupported BMP DIB header size"):
        read_bmp_size(bytes(data))


def test_bmp_is_discovered_and_measured(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.bmp", 320, 240, fmt="bmp")

    report = audit_dataset(root)

    assert len(report.images) == 1
    assert report.images[0].format == "bmp"
    assert (report.images[0].width, report.images[0].height) == (320, 240)
    assert report.corrupt_images == []


def test_bmp_counts_alongside_jpg_and_png(tmp_path: Path) -> None:
    """BMP must be added without disturbing the existing formats."""
    root = tmp_path / "d"
    write_image(root / "a.jpg", 100, 100, fmt="jpg")
    write_image(root / "b.jpeg", 100, 100, fmt="jpg")
    write_image(root / "c.png", 100, 100, fmt="png")
    write_image(root / "d.bmp", 100, 100, fmt="bmp")

    report = audit_dataset(root)

    assert len(report.images) == 4
    assert report.format_counts() == {"jpeg": 2, "bmp": 1, "png": 1}
    assert report.corrupt_images == []


def test_bmp_dimensions_feed_the_statistics(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "small.bmp", 100, 50, fmt="bmp")
    write_image(root / "large.bmp", 300, 150, fmt="bmp")
    write_image(root / "mid.jpg", 200, 100, fmt="jpg")

    stats = audit_dataset(root).dimension_stats()

    assert stats["count"] == 3
    assert stats["width"]["min"] == 100
    assert stats["width"]["max"] == 300
    assert stats["height"]["median"] == 100
    assert stats["distinct_resolutions"] == 3


def test_bmp_pairs_with_its_annotation(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "frame.bmp", fmt="bmp")
    write_label(root / "labels" / "frame.txt", ["0 0.5 0.5 0.2 0.2"])

    report = audit_dataset(root)

    assert report.images_without_annotations == []
    assert report.annotations_without_images == []


def test_bmp_without_annotation_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "images" / "orphan.bmp", fmt="bmp")

    report = audit_dataset(root)

    assert report.images_without_annotations == ["images/orphan.bmp"]


def test_identical_bmps_are_detected_by_hash(tmp_path: Path) -> None:
    root = tmp_path / "d"
    data = bmp_bytes(64, 64)
    for name in ("one.bmp", "two.bmp"):
        (root).mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    write_image(root / "other.bmp", 32, 32, fmt="bmp")

    report = audit_dataset(root)

    assert report.duplicate_images == [["one.bmp", "two.bmp"]]


def test_bmp_and_jpg_of_the_same_stem_are_ambiguous(tmp_path: Path) -> None:
    """A dataset holding both x.jpg and x.bmp must not silently pick one."""
    root = tmp_path / "d"
    write_image(root / "images" / "x.jpg", fmt="jpg")
    write_image(root / "images" / "x.bmp", fmt="bmp")
    write_label(root / "labels" / "x.txt", ["0 0.5 0.5 0.2 0.2"])

    report = audit_dataset(root)

    assert report.ambiguous_stems == {"x": ["images/x.bmp", "images/x.jpg"]}


def test_duplicate_bmp_filenames_across_directories(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "train" / "img.bmp", 10, 10, fmt="bmp")
    write_image(root / "val" / "img.bmp", 20, 20, fmt="bmp")

    report = audit_dataset(root)

    assert report.duplicate_filenames == {"img.bmp": ["train/img.bmp", "val/img.bmp"]}


def test_truncated_bmp_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "cut.bmp").write_bytes(bmp_bytes(64, 64)[:100])

    report = audit_dataset(root)

    assert len(report.corrupt_images) == 1
    assert "truncated" in " ".join(report.corrupt_images[0].problems)


def test_bmp_with_pixel_offset_past_end_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    data = bytearray(bmp_bytes(16, 16))
    data[10:14] = (999999).to_bytes(4, "little")
    (root / "bad.bmp").write_bytes(bytes(data))

    report = audit_dataset(root)

    assert "past the end of the file" in " ".join(report.corrupt_images[0].problems)


def test_empty_bmp_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "empty.bmp").write_bytes(b"")

    report = audit_dataset(root)

    assert "file is empty" in report.corrupt_images[0].problems[0]


def test_bmp_extension_on_a_jpeg_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "mislabelled.bmp").write_bytes(jpeg_bytes(32, 32))

    report = audit_dataset(root)

    assert report.images[0].format == "jpeg"
    assert "does not match actual format" in " ".join(report.images[0].warnings)
    assert report.corrupt_images == []


def test_jpg_extension_on_a_bmp_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "mislabelled.jpg").write_bytes(bmp_bytes(48, 24))

    report = audit_dataset(root)

    assert report.images[0].format == "bmp"
    assert (report.images[0].width, report.images[0].height) == (48, 24)
    assert "does not match actual format" in " ".join(report.images[0].warnings)
    assert report.corrupt_images == []


def test_bmp_appears_in_the_cli_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "d"
    write_image(root / "a.bmp", 128, 64, fmt="bmp")

    assert cli.main([str(root)]) == cli.EXIT_OK
    output = capsys.readouterr().out

    assert "bmp" in output
    assert "128x64" in output


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = write_image(tmp_path / "a.png", fmt="png")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# 1. Valid dataset
# --------------------------------------------------------------------------


def test_valid_dataset_audits_cleanly(valid_dataset: Path) -> None:
    report = audit_dataset(valid_dataset)

    assert len(report.images) == 3
    assert len(report.readable_images) == 3
    assert report.corrupt_images == []
    assert len(report.annotations) == 3
    assert report.total_boxes() == 6
    assert report.invalid_annotation_count() == 0
    assert report.images_without_annotations == []
    assert report.annotations_without_images == []
    assert report.duplicate_filenames == {}
    assert report.duplicate_images == []


def test_valid_dataset_reports_formats_and_dimensions(valid_dataset: Path) -> None:
    report = audit_dataset(valid_dataset)
    stats = report.dimension_stats()

    assert report.format_counts() == {"jpeg": 3}
    assert stats["count"] == 3
    assert stats["width"]["min"] == 641
    assert stats["width"]["max"] == 643
    assert stats["height"]["min"] == 480
    assert stats["distinct_resolutions"] == 3


def test_class_counts_and_declared_names(valid_dataset: Path) -> None:
    report = audit_dataset(valid_dataset)

    assert report.class_counts() == {0: 3, 1: 3}
    assert report.class_names() == {0: "plate", 1: "vehicle"}


def test_mixed_formats_are_counted(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg", fmt="jpg")
    write_image(root / "b.png", fmt="png")
    report = audit_dataset(root)

    assert report.format_counts() == {"jpeg": 1, "png": 1}


def test_audit_does_not_modify_the_dataset(valid_dataset: Path) -> None:
    before = {
        path: path.read_bytes()
        for path in sorted(valid_dataset.rglob("*"))
        if path.is_file()
    }
    audit_dataset(valid_dataset)
    after = {
        path: path.read_bytes()
        for path in sorted(valid_dataset.rglob("*"))
        if path.is_file()
    }

    assert before == after


# --------------------------------------------------------------------------
# 2. Missing annotation
# --------------------------------------------------------------------------


def test_image_without_annotation_is_reported(valid_dataset: Path) -> None:
    write_image(valid_dataset / "images" / "lonely.jpg")
    report = audit_dataset(valid_dataset)

    assert report.images_without_annotations == ["images/lonely.jpg"]
    assert report.annotations_without_images == []


# --------------------------------------------------------------------------
# 3. Orphan annotation
# --------------------------------------------------------------------------


def test_annotation_without_image_is_reported(valid_dataset: Path) -> None:
    write_label(valid_dataset / "labels" / "ghost.txt", ["0 0.5 0.5 0.2 0.2"])
    report = audit_dataset(valid_dataset)

    assert report.annotations_without_images == ["labels/ghost.txt"]
    assert report.images_without_annotations == []


def test_empty_annotation_file_is_a_valid_negative(valid_dataset: Path) -> None:
    write_image(valid_dataset / "images" / "bg.jpg")
    write_label(valid_dataset / "labels" / "bg.txt", [])
    report = audit_dataset(valid_dataset)

    assert report.images_without_annotations == []
    assert report.invalid_annotation_count() == 0


# --------------------------------------------------------------------------
# 4. Duplicate image
# --------------------------------------------------------------------------


def test_identical_images_are_detected_by_hash(tmp_path: Path) -> None:
    root = tmp_path / "d"
    data = jpeg_bytes(100, 50)
    for name in ("a/one.jpg", "b/two.jpg"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_image(root / "c" / "different.jpg", 200, 100)

    report = audit_dataset(root)

    assert report.duplicate_images == [["a/one.jpg", "b/two.jpg"]]


def test_duplicate_filenames_across_directories_are_detected(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "train" / "img.jpg", 100, 50)
    write_image(root / "val" / "img.jpg", 200, 100)  # same name, different content

    report = audit_dataset(root)

    assert report.duplicate_filenames == {"img.jpg": ["train/img.jpg", "val/img.jpg"]}
    assert report.duplicate_images == []


def test_hashing_can_be_skipped(tmp_path: Path) -> None:
    root = tmp_path / "d"
    data = jpeg_bytes(100, 50)
    for name in ("one.jpg", "two.jpg"):
        (root).mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)

    assert audit_dataset(root, hash_images=False).duplicate_images == []
    assert audit_dataset(root, hash_images=True).duplicate_images != []


# --------------------------------------------------------------------------
# 5. Corrupt image
# --------------------------------------------------------------------------


def test_empty_file_is_reported_as_corrupt(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "empty.jpg").write_bytes(b"")

    report = audit_dataset(root)

    assert len(report.corrupt_images) == 1
    assert "empty" in report.corrupt_images[0].problems[0]


def test_truncated_jpeg_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "cut.jpg").write_bytes(jpeg_bytes(640, 480)[:-2])  # drop the EOI

    report = audit_dataset(root)

    assert len(report.corrupt_images) == 1
    assert "truncated" in " ".join(report.corrupt_images[0].problems)


def test_truncated_png_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "cut.png").write_bytes(png_bytes(64, 64)[:-12])  # drop the IEND chunk

    report = audit_dataset(root)

    assert len(report.corrupt_images) == 1
    assert "truncated" in " ".join(report.corrupt_images[0].problems)


def test_file_with_wrong_magic_bytes_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "fake.jpg").write_bytes(b"this is not an image at all")

    report = audit_dataset(root)

    assert len(report.corrupt_images) == 1
    assert "not a PNG, JPEG or BMP" in report.corrupt_images[0].problems[0]
    assert report.corrupt_images[0].width is None


def test_extension_mismatch_is_a_warning_not_corruption(tmp_path: Path) -> None:
    """A wrongly named file is readable; counting it corrupt would mislead."""
    root = tmp_path / "d"
    root.mkdir()
    (root / "mislabelled.png").write_bytes(jpeg_bytes(32, 32))

    report = audit_dataset(root)

    assert report.corrupt_images == []
    assert len(report.mismatched_images) == 1
    assert "does not match actual format" in " ".join(report.images[0].warnings)
    assert report.images[0].format == "jpeg"
    assert report.images[0].width == 32  # still readable
    assert report.images[0] in report.readable_images


def test_mismatch_counts_group_by_extension_and_format(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "a.bmp").write_bytes(jpeg_bytes(10, 10))
    (root / "b.bmp").write_bytes(jpeg_bytes(20, 20))
    (root / "c.bmp").write_bytes(png_bytes(30, 30))
    write_image(root / "d.bmp", 40, 40, fmt="bmp")  # correctly named

    report = audit_dataset(root)

    assert report.mismatch_counts() == {".bmp -> jpeg": 2, ".bmp -> png": 1}
    assert report.corrupt_images == []


def test_corruption_and_mismatch_are_counted_separately(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "wrong_name.bmp").write_bytes(jpeg_bytes(10, 10))
    (root / "damaged.jpg").write_bytes(b"not an image")

    report = audit_dataset(root)

    assert [i.path for i in report.corrupt_images] == ["damaged.jpg"]
    assert [i.path for i in report.mismatched_images] == ["wrong_name.bmp"]


def test_corrupt_images_are_excluded_from_dimension_stats(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "good.jpg", 100, 50)
    (root / "bad.jpg").write_bytes(b"nope")

    report = audit_dataset(root)

    assert report.dimension_stats()["count"] == 1


# --------------------------------------------------------------------------
# 6. Invalid YOLO coordinates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("0 0.5 0.5 1.5 0.2", "normalised"),
        ("0 -0.1 0.5 0.2 0.2", "normalised"),
        ("0 0.5 0.5 0.0 0.2", "must be positive"),
        ("0 0.05 0.5 0.4 0.2", "beyond the image horizontally"),
        ("0 0.5 0.95 0.2 0.4", "beyond the image vertically"),
        ("0 0.5 0.5 0.2", "at least 5 fields"),
        ("x 0.5 0.5 0.2 0.2", "not an integer"),
        ("-1 0.5 0.5 0.2 0.2", "negative"),
        ("0 0.5 0.5 abc 0.2", "not all numeric"),
    ],
)
def test_invalid_yolo_lines_are_rejected(tmp_path: Path, line: str, reason: str) -> None:
    path = write_label(tmp_path / "a.txt", [line])
    annotation = parse_yolo_file(path, tmp_path)

    assert len(annotation.invalid_lines) == 1
    assert reason in annotation.invalid_lines[0]["reason"]
    assert annotation.boxes == []


@pytest.mark.parametrize(
    "line",
    [
        "0 0.5 0.5 0.2 0.1",
        "3 0.5 0.5 1.0 1.0",  # box filling the whole image
        "0 0.5 0.5 0.0001 0.0001",  # very small but well inside
        "0 0.05 0.5 0.1 0.2",  # left edge exactly at x=0
        "0 0.95 0.5 0.1 0.2",  # right edge exactly at x=1
        "2 0.1 0.1 0.2 0.2 0.3 0.3",  # polygon form
    ],
)
def test_valid_yolo_lines_are_accepted(tmp_path: Path, line: str) -> None:
    path = write_label(tmp_path / "a.txt", [line])
    annotation = parse_yolo_file(path, tmp_path)

    assert annotation.invalid_lines == []
    assert len(annotation.boxes) == 1


def test_box_centred_on_the_edge_overhangs_and_is_rejected(tmp_path: Path) -> None:
    """cx=0 with a positive width puts half the box outside the image."""
    path = write_label(tmp_path / "a.txt", ["0 0.0 0.0 0.0001 0.0001"])
    annotation = parse_yolo_file(path, tmp_path)

    assert "beyond the image" in annotation.invalid_lines[0]["reason"]


def test_invalid_line_records_number_and_text(tmp_path: Path) -> None:
    path = write_label(tmp_path / "a.txt", ["0 0.5 0.5 0.2 0.2", "0 9 9 9 9"])
    annotation = parse_yolo_file(path, tmp_path)

    assert len(annotation.boxes) == 1
    assert annotation.invalid_lines[0]["line"] == 2
    assert annotation.invalid_lines[0]["text"] == "0 9 9 9 9"


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = write_label(tmp_path / "a.txt", ["# comment", "", "0 0.5 0.5 0.2 0.2"])
    annotation = parse_yolo_file(path, tmp_path)

    assert annotation.line_count == 1
    assert annotation.invalid_lines == []


def test_prose_file_is_flagged_as_not_yolo(tmp_path: Path) -> None:
    path = write_label(tmp_path / "notes.txt", ["These are my collection notes."])
    annotation = parse_yolo_file(path, tmp_path)

    assert not annotation.is_yolo_format


def test_non_yolo_txt_is_not_paired_as_an_annotation(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "notes.txt", ["free text, not annotations"])

    report = audit_dataset(root)

    assert report.images_without_annotations == ["a.jpg"]
    assert report.annotations_without_images == []


# --------------------------------------------------------------------------
# 7. Unknown files
# --------------------------------------------------------------------------


def test_unknown_files_are_listed_separately(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    (root / "notes.docx").write_bytes(b"binary")
    (root / "archive.zip").write_bytes(b"PK\x03\x04")
    (root / "video.mp4").write_bytes(b"\x00")

    report = audit_dataset(root)

    assert sorted(report.other_files) == ["archive.zip", "notes.docx", "video.mp4"]
    assert len(report.images) == 1


def test_unsupported_image_format_counts_as_other(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "photo.webp").write_bytes(b"RIFF")
    (root / "photo.tiff").write_bytes(b"II*")
    (root / "photo.gif").write_bytes(b"GIF89a")

    report = audit_dataset(root)

    assert report.images == []
    assert sorted(report.other_files) == ["photo.gif", "photo.tiff", "photo.webp"]


# --------------------------------------------------------------------------
# Metadata and config discovery
# --------------------------------------------------------------------------


def test_metadata_files_are_found_with_a_preview(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "README.md").write_text("# Plates dataset\nCollected in 2024.\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC BY 4.0\n", encoding="utf-8")
    (root / "license.txt").write_text("see LICENSE\n", encoding="utf-8")

    report = audit_dataset(root)
    kinds = {meta.kind for meta in report.metadata_files}
    paths = {meta.path for meta in report.metadata_files}

    assert kinds == {"readme", "license"}
    assert paths == {"README.md", "LICENSE", "license.txt"}
    readme = next(m for m in report.metadata_files if m.kind == "readme")
    assert "Plates dataset" in readme.preview


@pytest.mark.parametrize("name", ["data.yaml", "dataset.yaml", "data.yml", "dataset.yml"])
def test_config_files_are_discovered(tmp_path: Path, name: str) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / name).write_text("nc: 1\nnames: ['plate']\n", encoding="utf-8")

    report = audit_dataset(root)

    assert len(report.configs) == 1
    assert report.configs[0].class_names == {0: "plate"}
    assert report.configs[0].declared_class_count == 1


def test_config_block_list_names(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("nc: 3\nnames:\n  - plate\n  - car\n  - truck\ntrain: x\n", encoding="utf-8")

    config = parse_dataset_config(path, tmp_path)

    assert config.class_names == {0: "plate", 1: "car", 2: "truck"}
    assert config.declared_class_count == 3


def test_config_mapping_names(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("names:\n  0: plate\n  1: car\nnc: 2\n", encoding="utf-8")

    config = parse_dataset_config(path, tmp_path)

    assert config.class_names == {0: "plate", 1: "car"}
    assert config.declared_class_count == 2


def test_config_without_names_says_so(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("train: images/train\nval: images/val\n", encoding="utf-8")

    config = parse_dataset_config(path, tmp_path)

    assert config.class_names == {}
    assert "no class names" in config.parse_note


def test_classes_txt_is_read_as_names_not_annotations(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "classes.txt").write_text("plate\nvehicle\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.annotations == []
    assert report.class_names() == {0: "plate", 1: "vehicle"}


@pytest.mark.parametrize(
    "name",
    [
        "README.dataset.txt",
        "README.roboflow.txt",
        "readme.md",
        "LICENSE.txt",
        "licence.md",
        "COPYING",
    ],
)
def test_multi_suffixed_paperwork_is_metadata_not_an_annotation(
    tmp_path: Path, name: str
) -> None:
    """Roboflow ships README.dataset.txt; Path.stem would read 'README.dataset'."""
    root = tmp_path / "d"
    root.mkdir()
    (root / name).write_text("License: CC BY 4.0\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.annotations == []
    assert report.invalid_annotation_count() == 0
    assert [meta.path for meta in report.metadata_files] == [name]


def test_paperwork_named_like_a_label_is_still_a_label(tmp_path: Path) -> None:
    """Only the leading segment counts -- 'license-plates.txt' is not a licence."""
    root = tmp_path / "d"
    write_image(root / "images" / "license-plates.jpg")
    write_label(root / "labels" / "license-plates.txt", ["0 0.5 0.5 0.2 0.2"])

    report = audit_dataset(root)

    assert report.metadata_files == []
    assert len(report.annotations) == 1


def test_readme_txt_is_metadata_not_an_annotation(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "readme.txt").write_text("hello\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.annotations == []
    assert len(report.metadata_files) == 1


def test_declared_but_unused_class_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "a.txt", ["0 0.5 0.5 0.2 0.2"])
    (root / "data.yaml").write_text("nc: 2\nnames: ['plate', 'vehicle']\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.unused_declared_class_ids() == [1]
    assert report.undeclared_class_ids() == []


def test_class_used_but_not_declared_is_reported(tmp_path: Path) -> None:
    """A config that disagrees with the labels makes both untrustworthy."""
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "a.txt", ["0 0.5 0.5 0.2 0.2", "7 0.5 0.5 0.2 0.2"])
    (root / "data.yaml").write_text("nc: 1\nnames: ['plate']\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.undeclared_class_ids() == [7]


def test_mismatch_is_not_claimed_when_no_config_exists(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "a.txt", ["4 0.5 0.5 0.2 0.2"])

    report = audit_dataset(root)

    assert report.undeclared_class_ids() == []
    assert report.unused_declared_class_ids() == []


def test_classes_are_not_named_without_a_config(tmp_path: Path) -> None:
    """The tool must never invent a meaning for a class id."""
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "a.txt", ["0 0.5 0.5 0.2 0.2"])

    report = audit_dataset(root)

    assert report.class_counts() == {0: 1}
    assert report.class_names() == {}


# --------------------------------------------------------------------------
# CSV discovery
# --------------------------------------------------------------------------


def test_csv_header_and_row_count_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "annotations.csv").write_text(
        "filename,x,y,w,h\na.jpg,1,2,3,4\nb.jpg,5,6,7,8\n", encoding="utf-8"
    )

    report = audit_dataset(root)

    assert len(report.csv_files) == 1
    assert report.csv_files[0].header == ["filename", "x", "y", "w", "h"]
    assert report.csv_files[0].row_count == 2
    assert report.csv_files[0].delimiter == ","


def test_semicolon_csv_is_detected(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "a.csv").write_text("image;plate\nx.jpg;A123BC77\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.csv_files[0].delimiter == ";"
    assert report.csv_files[0].header == ["image", "plate"]


def test_empty_csv_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "a.csv").write_text("", encoding="utf-8")

    report = audit_dataset(root)

    assert report.csv_files[0].error == "file is empty"


# --------------------------------------------------------------------------
# Audit entry point
# --------------------------------------------------------------------------


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(AuditError, match="does not exist"):
        audit_dataset(tmp_path / "nope")


def test_file_instead_of_directory_raises(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(AuditError, match="not a directory"):
        audit_dataset(path)


def test_empty_directory_audits_without_error(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    report = audit_dataset(root)

    assert report.images == []
    assert report.dimension_stats() == {}


def test_source_id_is_recorded(valid_dataset: Path) -> None:
    report = audit_dataset(valid_dataset, source_id="some-collection")
    assert report.source_id == "some-collection"
    assert report.to_dict()["source_id"] == "some-collection"


def test_nested_directories_are_discovered(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a" / "b" / "c" / "deep.jpg")

    report = audit_dataset(root)

    assert [image.path for image in report.images] == ["a/b/c/deep.jpg"]


# --------------------------------------------------------------------------
# CLI and reports
# --------------------------------------------------------------------------


def test_cli_runs_and_exits_zero(valid_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([str(valid_dataset)]) == cli.EXIT_OK

    output = capsys.readouterr().out
    assert "External dataset audit" in output
    assert "inspection only" in output


def test_cli_report_contains_every_section(
    valid_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main([str(valid_dataset)])
    output = capsys.readouterr().out

    for section in (
        "Totals",
        "Image formats",
        "Image dimensions",
        "Corrupt or unreadable images",
        "YOLO annotations",
        "Classes",
        "Image / annotation pairing",
        "Duplicates",
        "CSV files",
        "Dataset paperwork",
        "Next steps",
    ):
        assert section in output, section


def test_cli_missing_directory_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([str(tmp_path / "nope")]) == cli.EXIT_CANNOT_AUDIT
    assert "does not exist" in capsys.readouterr().err


def test_cli_writes_json_and_text_reports(valid_dataset: Path, tmp_path: Path) -> None:
    json_path = tmp_path / "out" / "audit.json"
    text_path = tmp_path / "out" / "audit.txt"

    assert cli.main(
        [str(valid_dataset), "--json", str(json_path), "--report", str(text_path)]
    ) == cli.EXIT_OK

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["total_images"] == 3
    assert payload["inspection_only"] is True
    assert payload["classes"]["counts_by_id"] == {"0": 3, "1": 3}
    assert "External dataset audit" in text_path.read_text(encoding="utf-8")


def test_json_report_is_serialisable_for_every_field(tmp_path: Path) -> None:
    root = tmp_path / "d"
    write_image(root / "a.jpg")
    write_label(root / "a.txt", ["0 0.5 0.5 0.2 0.2", "bad line"])
    (root / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "LICENSE").write_text("CC BY 4.0", encoding="utf-8")
    (root / "junk.bin").write_bytes(b"\x00")

    payload = json.loads(json.dumps(audit_dataset(root).to_dict()))

    assert payload["summary"]["invalid_annotation_lines"] == 1
    assert payload["other_files"] == ["junk.bin"]


def test_summary_json_omits_the_datasets_own_annotations(valid_dataset: Path) -> None:
    """A committable report must not reproduce third-party annotation data."""
    payload = audit_dataset(valid_dataset).to_dict(detail="summary")

    assert "annotations" not in payload
    assert "images" not in payload
    assert payload["detail"] == "summary"
    assert "0.5" not in json.dumps(payload)  # no box coordinates leaked
    # Statistics and findings survive.
    assert payload["summary"]["total_annotation_boxes"] == 6
    assert payload["classes"]["counts_by_id"] == {"0": 3, "1": 3}


def test_full_json_keeps_the_detail_arrays(valid_dataset: Path) -> None:
    payload = audit_dataset(valid_dataset).to_dict(detail="full")

    assert len(payload["annotations"]) == 3
    assert len(payload["images"]) == 3
    assert payload["detail"] == "full"


def test_unknown_detail_level_is_rejected(valid_dataset: Path) -> None:
    with pytest.raises(ValueError, match="detail must be one of"):
        audit_dataset(valid_dataset).to_dict(detail="partial")


def test_summary_still_lists_mismatched_paths(tmp_path: Path) -> None:
    root = tmp_path / "d"
    root.mkdir()
    (root / "wrong.bmp").write_bytes(jpeg_bytes(10, 10))

    payload = audit_dataset(root).to_dict(detail="summary")

    assert payload["mismatched_image_paths"] == [
        {"path": "wrong.bmp", "declared": ".bmp", "actual": "jpeg"}
    ]


def test_cli_json_detail_summary(valid_dataset: Path, tmp_path: Path) -> None:
    destination = tmp_path / "audit.json"

    assert cli.main(
        [str(valid_dataset), "--json", str(destination), "--json-detail", "summary"]
    ) == cli.EXIT_OK

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert "annotations" not in payload
    assert payload["summary"]["total_images"] == 3


def test_cli_source_id_flows_into_the_json(valid_dataset: Path, tmp_path: Path) -> None:
    json_path = tmp_path / "audit.json"
    cli.main([str(valid_dataset), "--source-id", "wikimedia-commons", "--json", str(json_path)])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source_id"] == "wikimedia-commons"


def test_cli_quiet_mode(valid_dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([str(valid_dataset), "--quiet"]) == cli.EXIT_OK

    output = capsys.readouterr().out.strip()
    assert output.startswith("3 image(s)")
    assert "Totals" not in output


def test_fail_on_findings_flag(tmp_path: Path, valid_dataset: Path) -> None:
    assert cli.main([str(valid_dataset), "--fail-on-findings"]) == cli.EXIT_OK

    write_image(valid_dataset / "images" / "orphan.jpg")
    assert cli.main([str(valid_dataset), "--fail-on-findings"]) == cli.EXIT_FINDINGS


def test_report_refuses_to_write_into_the_audited_directory(
    valid_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inside = valid_dataset / "audit.json"

    assert cli.main([str(valid_dataset), "--json", str(inside)]) == cli.EXIT_CANNOT_AUDIT
    assert "refusing to write" in capsys.readouterr().err
    assert not inside.exists()


def test_report_refuses_to_write_into_our_dataset_directory(
    valid_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = cli.REPO_ROOT / "dataset" / "audit.json"

    assert cli.main([str(valid_dataset), "--json", str(target)]) == cli.EXIT_CANNOT_AUDIT
    assert "competition dataset" in capsys.readouterr().err
    assert not target.exists()


def test_cli_never_creates_files_in_the_audited_directory(valid_dataset: Path) -> None:
    before = sorted(path.name for path in valid_dataset.rglob("*"))
    cli.main([str(valid_dataset)])
    after = sorted(path.name for path in valid_dataset.rglob("*"))

    assert before == after


def test_audit_does_not_touch_our_dataset_directory(valid_dataset: Path) -> None:
    dataset_dir = cli.REPO_ROOT / "dataset"
    before = sorted(p.relative_to(dataset_dir).as_posix() for p in dataset_dir.rglob("*"))

    cli.main([str(valid_dataset)])

    after = sorted(p.relative_to(dataset_dir).as_posix() for p in dataset_dir.rglob("*"))
    assert before == after
