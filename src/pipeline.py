"""End-to-end inference pipeline.

    image -> detector -> crop -> classifier -> ocr -> validator -> confidence -> CSV

Each stage is injected, so replacing a placeholder with a trained model is a
constructor argument change and nothing else.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from src.classifier import PlaceholderClassifier, PlateClassifier
from src.csv_writer import PlateRecord
from src.detector import Detector, PlaceholderDetector
from src.ocr import OcrEngine, PlaceholderOcr
from src.validator import validate_plate

logger = logging.getLogger(__name__)

#: Image extensions accepted by the pipeline (compared lower-cased).
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

#: Confidence multiplier applied when the read text fails format validation.
INVALID_FORMAT_PENALTY: float = 0.5


@dataclass
class PipelineStats:
    """Counters collected over one directory run."""

    images: int = 0
    images_with_detections: int = 0
    detections: int = 0
    records: int = 0
    invalid_format: int = 0
    failed_images: int = 0
    total_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.images if self.images else 0.0


def iter_images(directory: Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield image files under ``directory`` in a stable, sorted order.

    Only ``.jpg``, ``.jpeg`` and ``.png`` files are returned; extension
    matching is case-insensitive.  Unreadable entries are skipped rather than
    raising.
    """
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


@dataclass
class Pipeline:
    """Runs the detection/classification/OCR chain over images.

    Defaults to the placeholder stages, which detect nothing and therefore
    produce no rows.
    """

    detector: Detector = field(default_factory=PlaceholderDetector)
    classifier: PlateClassifier = field(default_factory=PlaceholderClassifier)
    ocr: OcrEngine = field(default_factory=PlaceholderOcr)
    min_confidence: float = 0.0
    stats: PipelineStats = field(default_factory=PipelineStats)

    def process_image(self, image_path: Path, *, image_id: str | None = None) -> list[PlateRecord]:
        """Process a single image and return zero or more CSV records.

        ``image_id`` overrides the value written to the ``image`` column; it
        defaults to the file name.
        """
        image_path = Path(image_path)
        name = image_id if image_id is not None else image_path.name
        started = time.perf_counter()
        records: list[PlateRecord] = []

        try:
            detections = self.detector.detect(image_path)
            self.stats.detections += len(detections)
            if detections:
                self.stats.images_with_detections += 1

            for detection in detections:
                classification = self.classifier.classify(image_path, detection)
                ocr_result = self.ocr.read(image_path, detection, classification)

                if ocr_result.is_empty:
                    logger.debug("%s: OCR returned no text, dropping detection", name)
                    continue

                validation = validate_plate(ocr_result.text)
                confidence = (
                    detection.confidence
                    * classification.confidence
                    * ocr_result.confidence
                )
                if not validation.is_valid:
                    self.stats.invalid_format += 1
                    confidence *= INVALID_FORMAT_PENALTY
                    logger.debug(
                        "%s: %r fails format check (%s)",
                        name,
                        validation.normalized,
                        validation.reason,
                    )

                confidence = min(1.0, max(0.0, confidence))
                if confidence < self.min_confidence:
                    logger.debug(
                        "%s: dropping %r, confidence %.3f < %.3f",
                        name,
                        validation.normalized,
                        confidence,
                        self.min_confidence,
                    )
                    continue

                records.append(
                    PlateRecord(
                        image=name,
                        plate_num=validation.normalized,
                        plate_type=classification.plate_type,
                        confidence=confidence,
                    )
                )
        except Exception:  # noqa: BLE001 - one bad image must not stop the run
            self.stats.failed_images += 1
            logger.exception("Failed to process %s", image_path)
        finally:
            elapsed = time.perf_counter() - started
            self.stats.images += 1
            self.stats.records += len(records)
            self.stats.total_seconds += elapsed
            logger.info(
                "%s processed in %.2f ms -> %d record(s)", name, elapsed * 1000.0, len(records)
            )

        return records

    def process_directory(self, directory: Path, *, recursive: bool = True) -> list[PlateRecord]:
        """Process every supported image under ``directory``."""
        directory = Path(directory)
        images: Sequence[Path] = list(iter_images(directory, recursive=recursive))
        logger.info("Found %d image(s) in %s", len(images), directory)

        records: list[PlateRecord] = []
        for image_path in images:
            image_id = image_path.relative_to(directory).as_posix()
            records.extend(self.process_image(image_path, image_id=image_id))

        logger.info(
            "Processed %d image(s) in %.2f s (mean %.1f ms/image); "
            "%d detection(s), %d row(s), %d invalid format, %d failure(s)",
            self.stats.images,
            self.stats.total_seconds,
            self.stats.mean_seconds * 1000.0,
            self.stats.detections,
            self.stats.records,
            self.stats.invalid_format,
            self.stats.failed_images,
        )
        return records
