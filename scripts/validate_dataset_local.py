#!/usr/bin/env python
"""Participant-side dataset checker.

Validates ``dataset/meta.csv`` against the schema in :mod:`src.dataset_meta`
and prints a human-readable report.

Usage::

    python scripts/validate_dataset_local.py
    python scripts/validate_dataset_local.py --dataset dataset --report report.txt
    python scripts/validate_dataset_local.py --skip-file-check

Exit codes::

    0  the dataset is valid (warnings may still be present)
    1  validation errors were found, or meta.csv could not be read
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running the script directly
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_meta import (  # noqa: E402  (path set up above)
    ALLOWED_CONDITIONS,
    ALLOWED_PLATE_TYPES,
    MetaFormatError,
    Severity,
    ValidationReport,
    iter_issues,
    validate_meta,
)

EXIT_OK: int = 0
EXIT_INVALID: int = 1

#: Default dataset directory, relative to the repository root.
DEFAULT_DATASET_DIR: Path = REPO_ROOT / "dataset"

#: How many issues of one severity to print before summarising the rest.
MAX_LISTED_ISSUES: int = 50

SEPARATOR: str = "-" * 72


def _format_counts(counts: Mapping[str, int], total: int, indent: str = "  ") -> list[str]:
    """Render a ``name: count (share)`` block, longest count first."""
    if not counts:
        return [f"{indent}(none)"]

    width = max(len(name) for name in counts)
    lines: list[str] = []
    for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        share = f"{count / total:6.1%}" if total else "     -"
        lines.append(f"{indent}{name:<{width}}  {count:>7}  {share}")
    return lines


def _format_issues(report: ValidationReport, severity: Severity) -> list[str]:
    issues = list(iter_issues(report, severity))
    if not issues:
        return ["  (none)"]

    lines = [f"  {issue}" for issue in issues[:MAX_LISTED_ISSUES]]
    if len(issues) > MAX_LISTED_ISSUES:
        lines.append(f"  ... and {len(issues) - MAX_LISTED_ISSUES} more")
    return lines


def _section(title: str) -> list[str]:
    return ["", title, SEPARATOR]


def build_report_text(report: ValidationReport) -> str:
    """Render ``report`` as the human-readable text report."""
    stats = report.stats
    lines: list[str] = [
        "Volga-IT 2026 - local dataset validation report",
        SEPARATOR,
        f"meta.csv     : {report.meta_path}",
        f"dataset root : {report.dataset_root}",
    ]

    lines += _section("Totals")
    lines += [
        f"  total images      : {stats.total_images}",
        f"  total annotations : {stats.total_rows}",
        f"  real images       : {stats.real_images}  ({stats.real_rows} annotations)",
        f"  synthetic images  : {stats.synthetic_images}  "
        f"({stats.synthetic_rows} annotations)",
        f"  background rows   : {stats.background_rows}  (negatives without geometry)",
    ]

    lines += _section("Annotations by plate_type")
    lines += _format_counts(stats.by_plate_type, stats.total_rows)
    missing_types = sorted(ALLOWED_PLATE_TYPES - set(stats.by_plate_type))
    if missing_types:
        lines.append(f"  not represented: {', '.join(missing_types)}")

    if stats.unique_plates_by_type:
        lines += _section("Unique plate numbers by plate_type")
        for plate_type, count in stats.unique_plates_by_type.items():
            lines.append(f"  {plate_type:<10} {count:>7}")

    lines += _section("Annotations by condition")
    lines += _format_counts(stats.by_condition, stats.total_rows)
    unused = [tag for tag in ALLOWED_CONDITIONS if tag not in stats.by_condition]
    if unused:
        lines.append(f"  never used: {', '.join(unused)}")
    lines.append(f"  rows with no condition tag: {stats.rows_without_conditions}")

    lines += _section("Source / license coverage")
    real_rows = stats.real_rows
    documented_source = real_rows - stats.rows_missing_source
    documented_license = real_rows - stats.rows_missing_license
    lines += [
        f"  real annotations with a source  : {documented_source}/{real_rows}",
        f"  real annotations with a license : {documented_license}/{real_rows}",
        "",
        "  sources:",
    ]
    lines += _format_counts(stats.by_source, stats.total_rows, indent="    ")
    lines += ["", "  licenses:"]
    lines += _format_counts(stats.by_license, stats.total_rows, indent="    ")

    lines += _section(f"Missing image files ({len(stats.missing_files)})")
    if stats.missing_files:
        for path in stats.missing_files[:MAX_LISTED_ISSUES]:
            lines.append(f"  {path}")
        if len(stats.missing_files) > MAX_LISTED_ISSUES:
            lines.append(f"  ... and {len(stats.missing_files) - MAX_LISTED_ISSUES} more")
    else:
        lines.append("  (none)")

    lines += _section(f"Errors ({len(report.errors)})")
    lines += _format_issues(report, Severity.ERROR)

    lines += _section(f"Warnings ({len(report.warnings)})")
    lines += _format_issues(report, Severity.WARNING)

    verdict = "VALID" if report.is_valid else "INVALID"
    lines += _section("Verdict")
    lines += [
        f"  {verdict} - {len(report.errors)} error(s), {len(report.warnings)} warning(s)",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_dataset_local.py",
        description="Validate the locally built dataset and print a report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="dataset root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="path to meta.csv (default: <dataset>/meta.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="also write the report to this file",
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="do not verify that the referenced image files exist",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors when choosing the exit code",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the verdict line",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root: Path = args.dataset
    meta_path: Path = args.meta if args.meta is not None else dataset_root / "meta.csv"

    try:
        report = validate_meta(
            meta_path, dataset_root, check_files=not args.skip_file_check
        )
    except MetaFormatError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_INVALID

    text = build_report_text(report)
    if args.quiet:
        print(
            f"{'VALID' if report.is_valid else 'INVALID'} - "
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        )
    else:
        print(text)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"Report written to {args.report}")

    failed = report.errors or (args.strict and report.warnings)
    return EXIT_INVALID if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
