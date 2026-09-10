#!/usr/bin/env python
"""Build a reproducible human-review sample from an audited external dataset.

Selects a seeded subset of images and writes two artefacts: a CSV for the
reviewer to fill in, and an HTML contact sheet to look at. The external dataset
is read only -- nothing is modified, copied or imported.

Usage::

    python scripts/sample_review_set.py <dataset-dir> --source-id <id> \\
        --csv data/audits/<id>_review/review_sample.csv \\
        --contact-sheet data/review/<id>/contact_sheet.html

Exit codes::

    0  the sample was built
    1  the dataset could not be read, or an output path was refused
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running the script directly
    sys.path.insert(0, str(REPO_ROOT))

from src.external_audit import AuditError  # noqa: E402
from src.review_sample import (  # noqa: E402
    HUMAN_ONLY_CHECKS,
    REVIEW_SAMPLE_SIZE,
    REVIEW_SEED,
    build_review_items,
    discover_pairs,
    flag_counts,
    select_sample,
    suggest_plate_type,
    write_contact_sheet,
    write_review_csv,
)

EXIT_OK: int = 0
EXIT_FAILED: int = 1

SEPARATOR: str = "-" * 72


def _refuse_unsafe_output(destination: Path, audited: Path) -> None:
    """Keep outputs out of the audited dataset and out of ``dataset/``."""
    resolved = destination.resolve()
    for forbidden, reason in (
        (audited.resolve(), "the audited dataset directory"),
        ((REPO_ROOT / "dataset").resolve(), "our competition dataset directory"),
    ):
        if resolved == forbidden or forbidden in resolved.parents:
            raise AuditError(
                f"refusing to write {destination} inside {reason}; "
                "this tool never modifies either"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sample_review_set.py",
        description=(
            "Build a reproducible review sample from an external dataset. "
            "Read-only: nothing is modified or imported."
        ),
    )
    parser.add_argument("directory", type=Path, help="the audited dataset directory")
    parser.add_argument(
        "--source-id", required=True, help="source_id this sample belongs to"
    )
    parser.add_argument("--csv", type=Path, required=True, help="review CSV to write")
    parser.add_argument(
        "--contact-sheet", type=Path, default=None, help="HTML contact sheet to write"
    )
    parser.add_argument(
        "--size", type=int, default=REVIEW_SAMPLE_SIZE, help="sample size (default: %(default)s)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REVIEW_SEED,
        help="random seed fixing the sample (default: %(default)s)",
    )
    parser.add_argument("--log-level", default="WARNING", help="logging level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    directory: Path = args.directory
    try:
        if not directory.is_dir():
            raise AuditError(f"{directory} is not a directory")
        for destination in (args.csv, args.contact_sheet):
            if destination is not None:
                _refuse_unsafe_output(destination, directory)

        pairs = discover_pairs(directory)
        if not pairs:
            raise AuditError(f"no images found under {directory}")
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED

    stems = select_sample(pairs, size=args.size, seed=args.seed)
    items = build_review_items(directory, stems, pairs)

    rows = write_review_csv(items, args.csv)
    print(f"Review CSV written to {args.csv} ({rows} rows)")

    if args.contact_sheet is not None:
        path = write_contact_sheet(
            items,
            directory,
            args.contact_sheet,
            source_id=args.source_id,
            seed=args.seed,
        )
        print(f"Contact sheet written to {path}")

    print()
    print(f"Review sample - {args.source_id}")
    print(SEPARATOR)
    print(f"  dataset      : {directory}")
    print(f"  population   : {len(pairs)} image(s)")
    print(f"  sample size  : {len(items)}")
    print(f"  seed         : {args.seed}")

    suggestions: dict[str, int] = {}
    for item in items:
        suggestion, _ = suggest_plate_type(item)
        suggestions[suggestion] = suggestions.get(suggestion, 0) + 1

    print()
    print("  Automated plate-type suggestions (NOT human-confirmed):")
    for suggestion, count in sorted(suggestions.items()):
        print(f"    {suggestion:<24} {count:>5}")

    print()
    print("  Geometry flags (machine-derived, from annotation boxes only):")
    for flag, count in flag_counts(items).items():
        print(f"    {flag:<32} {count:>5}")

    print()
    print("  Still requires a person, per image:")
    for check in HUMAN_ONLY_CHECKS:
        print(f"    - {check}")
    print()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
