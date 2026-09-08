"""Plate type classification stage.

Decides which of the competition classes a detected crop belongs to.  The real
classifier will be a small CNN; the placeholder abstains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from src.detector import Detection

logger = logging.getLogger(__name__)


class PlateType(str, Enum):
    """Plate classes required by the competition CSV."""

    TYPE1 = "type1"      # standard white one-line plate
    TYPE1A = "type1a"    # white two-line / square plate
    TYPE1B = "type1b"    # yellow one-line passenger transport plate
    OTHER = "other"      # non-target plate or negative example

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Classes whose text is expected to follow the standard Russian plate format.
TARGET_TYPES: frozenset[PlateType] = frozenset(
    {PlateType.TYPE1, PlateType.TYPE1A, PlateType.TYPE1B}
)


@dataclass(frozen=True)
class Classification:
    """Result of the classification stage."""

    plate_type: PlateType
    confidence: float


class PlateClassifier(Protocol):
    """Contract every plate type classifier must satisfy."""

    name: str

    def classify(self, image_path: Path, detection: Detection) -> Classification:
        """Classify the crop described by ``detection`` inside ``image_path``."""
        ...


class PlaceholderClassifier:
    """Abstaining classifier: everything is ``other`` with zero confidence."""

    name = "placeholder-classifier"

    def classify(self, image_path: Path, detection: Detection) -> Classification:
        logger.debug("Placeholder classifier: abstaining for %s", image_path.name)
        return Classification(plate_type=PlateType.OTHER, confidence=0.0)
