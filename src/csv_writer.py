"""Submission CSV writing.

Format (UTF-8, semicolon separated)::

    image;plate_num;plate_type;confidence
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.classifier import PlateType

logger = logging.getLogger(__name__)

CSV_DELIMITER: str = ";"
CSV_ENCODING: str = "utf-8"
CSV_HEADER: Sequence[str] = ("image", "plate_num", "plate_type", "confidence")

#: Number of decimals used for the confidence column.
CONFIDENCE_PRECISION: int = 3


@dataclass(frozen=True)
class PlateRecord:
    """One recognised plate, i.e. one row of the output CSV."""

    image: str
    plate_num: str
    plate_type: PlateType
    confidence: float

    def as_row(self) -> list[str]:
        confidence = min(1.0, max(0.0, self.confidence))
        return [
            self.image,
            self.plate_num,
            str(self.plate_type),
            f"{confidence:.{CONFIDENCE_PRECISION}f}",
        ]


def write_csv(
    records: Iterable[PlateRecord],
    output_path: Path,
    *,
    write_header: bool = True,
) -> int:
    """Write ``records`` to ``output_path`` and return the number of rows.

    The parent directory is created if missing.  An existing file is
    overwritten.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        if write_header:
            writer.writerow(CSV_HEADER)
        for record in records:
            writer.writerow(record.as_row())
            written += 1

    logger.info("Wrote %d row(s) to %s", written, output_path)
    return written
