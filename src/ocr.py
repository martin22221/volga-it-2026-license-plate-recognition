"""Optical character recognition stage.

The real engine will read the plate crop and return its characters.  The
placeholder returns an empty string so that no invented plate number can ever
reach the CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.classifier import Classification
from src.detector import Detection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    """Raw text read from a plate crop, before validation/normalisation."""

    text: str
    confidence: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class OcrEngine(Protocol):
    """Contract every OCR implementation must satisfy."""

    name: str

    def read(
        self,
        image_path: Path,
        detection: Detection,
        classification: Classification,
    ) -> OcrResult:
        """Read the plate text from the crop described by ``detection``.

        Implementations must return an empty :class:`OcrResult` rather than a
        guess when they cannot read the plate.
        """
        ...


class PlaceholderOcr:
    """No-op OCR used until a trained model is available."""

    name = "placeholder-ocr"

    def read(
        self,
        image_path: Path,
        detection: Detection,
        classification: Classification,
    ) -> OcrResult:
        logger.debug("Placeholder OCR: no text for %s", image_path.name)
        return OcrResult(text="", confidence=0.0)
