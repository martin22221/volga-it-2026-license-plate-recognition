"""Command line entry point for offline license plate recognition.

Usage::

    python run.py --input <image_directory> --output <output_csv>
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.csv_writer import write_csv
from src.pipeline import Pipeline

logger = logging.getLogger("run")


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Send logs to stderr and, optionally, to ``log_file``."""
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Offline detection and recognition of Russian license plates.",
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="directory with .jpg/.jpeg/.png images"
    )
    parser.add_argument("--output", required=True, type=Path, help="path of the output CSV")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="only read images directly inside --input (default: walk subdirectories)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="drop rows below this confidence (default: 0.0)",
    )
    parser.add_argument(
        "--no-header", action="store_true", help="write the CSV without a header row"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level (default: INFO)",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="also write logs to this file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level, args.log_file)

    input_dir: Path = args.input
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        return 2

    pipeline = Pipeline(min_confidence=args.min_confidence)
    logger.info(
        "Stages: %s | %s | %s",
        pipeline.detector.name,
        pipeline.classifier.name,
        pipeline.ocr.name,
    )

    started = time.perf_counter()
    records = pipeline.process_directory(input_dir, recursive=not args.no_recursive)
    rows = write_csv(records, args.output, write_header=not args.no_header)
    logger.info(
        "Done: %d image(s), %d row(s) in %.2f s", pipeline.stats.images, rows,
        time.perf_counter() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
