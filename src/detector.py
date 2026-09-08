"""Plate detection stage.

The real detector (a small object detector trained on Russian plates) will be
added later.  For now :class:`PlaceholderDetector` reports no detections, which
by design produces no CSV rows at all -- we never fabricate a plate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates of the source image."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    """One candidate plate found in an image."""

    box: BoundingBox
    confidence: float


class Detector(Protocol):
    """Contract every detector implementation must satisfy."""

    name: str

    def detect(self, image_path: Path) -> Sequence[Detection]:
        """Return candidate plate regions for ``image_path``.

        An empty sequence means "no plate in this image" and results in no CSV
        row being written.
        """
        ...


class PlaceholderDetector:
    """No-op detector used until a trained model is available.

    It intentionally returns nothing: an untrained pipeline must not emit
    guesses.
    """

    name = "placeholder-detector"

    def detect(self, image_path: Path) -> Sequence[Detection]:
        logger.debug("Placeholder detector: no detections for %s", image_path.name)
        return ()
