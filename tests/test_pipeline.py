"""Tests for image discovery and the end-to-end pipeline wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from src.classifier import Classification, PlateType
from src.csv_writer import write_csv
from src.detector import BoundingBox, Detection
from src.ocr import OcrResult
from src.pipeline import Pipeline, iter_images

BOX = BoundingBox(0, 0, 100, 30)


class StubDetector:
    name = "stub-detector"

    def __init__(self, detections: Sequence[Detection]) -> None:
        self._detections = detections

    def detect(self, image_path: Path) -> Sequence[Detection]:
        return self._detections


class StubClassifier:
    name = "stub-classifier"

    def __init__(self, plate_type: PlateType, confidence: float) -> None:
        self._result = Classification(plate_type, confidence)

    def classify(self, image_path: Path, detection: Detection) -> Classification:
        return self._result


class StubOcr:
    name = "stub-ocr"

    def __init__(self, text: str, confidence: float) -> None:
        self._result = OcrResult(text, confidence)

    def read(self, image_path, detection, classification) -> OcrResult:
        return self._result


def make_images(root: Path, names: Sequence[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_iter_images_filters_extensions_and_recurses(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg", "b.JPEG", "c.png", "notes.txt", "sub/d.PNG", "sub/e.bmp"])
    found = [p.relative_to(tmp_path).as_posix() for p in iter_images(tmp_path)]
    assert found == ["a.jpg", "b.JPEG", "c.png", "sub/d.PNG"]


def test_iter_images_non_recursive(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg", "sub/b.jpg"])
    found = [p.name for p in iter_images(tmp_path, recursive=False)]
    assert found == ["a.jpg"]


def test_placeholder_pipeline_produces_no_rows(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg", "b.png"])
    pipeline = Pipeline()
    assert pipeline.process_directory(tmp_path) == []
    assert pipeline.stats.images == 2
    assert pipeline.stats.total_seconds >= 0.0


def test_full_chain_with_stubs(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg"])
    pipeline = Pipeline(
        detector=StubDetector([Detection(BOX, 0.8)]),
        classifier=StubClassifier(PlateType.TYPE1, 0.9),
        ocr=StubOcr(" a123bc 77 ", 1.0),
    )
    records = pipeline.process_directory(tmp_path)

    assert len(records) == 1
    record = records[0]
    assert record.image == "a.jpg"
    assert record.plate_num == "A123BC77"
    assert record.plate_type is PlateType.TYPE1
    assert record.confidence == pytest.approx(0.72)


def test_invalid_format_is_penalised_not_dropped(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg"])
    pipeline = Pipeline(
        detector=StubDetector([Detection(BOX, 1.0)]),
        classifier=StubClassifier(PlateType.TYPE1, 1.0),
        ocr=StubOcr("ZZ99", 1.0),
    )
    records = pipeline.process_directory(tmp_path)
    assert [r.confidence for r in records] == [pytest.approx(0.5)]
    assert pipeline.stats.invalid_format == 1


def test_empty_ocr_text_produces_no_row(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg"])
    pipeline = Pipeline(
        detector=StubDetector([Detection(BOX, 1.0)]),
        classifier=StubClassifier(PlateType.TYPE1, 1.0),
        ocr=StubOcr("   ", 1.0),
    )
    assert pipeline.process_directory(tmp_path) == []


def test_min_confidence_filters_rows(tmp_path: Path) -> None:
    make_images(tmp_path, ["a.jpg"])
    pipeline = Pipeline(
        detector=StubDetector([Detection(BOX, 0.3)]),
        classifier=StubClassifier(PlateType.TYPE1, 0.3),
        ocr=StubOcr("A123BC77", 1.0),
        min_confidence=0.5,
    )
    assert pipeline.process_directory(tmp_path) == []


def test_detector_failure_is_isolated(tmp_path: Path) -> None:
    class BrokenDetector:
        name = "broken"

        def detect(self, image_path: Path):
            raise RuntimeError("boom")

    make_images(tmp_path, ["a.jpg"])
    pipeline = Pipeline(detector=BrokenDetector())
    assert pipeline.process_directory(tmp_path) == []
    assert pipeline.stats.failed_images == 1


def test_pipeline_to_csv(tmp_path: Path) -> None:
    images = tmp_path / "images"
    make_images(images, ["car.jpg"])
    pipeline = Pipeline(
        detector=StubDetector([Detection(BOX, 1.0)]),
        classifier=StubClassifier(PlateType.TYPE1B, 1.0),
        ocr=StubOcr("M111MM102", 1.0),
    )
    output = tmp_path / "out.csv"
    write_csv(pipeline.process_directory(images), output)
    assert output.read_text(encoding="utf-8").splitlines() == [
        "image;plate_num;plate_type;confidence",
        "car.jpg;M111MM102;type1b;1.000",
    ]
