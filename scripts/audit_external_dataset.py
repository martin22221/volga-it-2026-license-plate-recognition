#!/usr/bin/env python
"""Audit an external dataset directory. Inspection only -- imports nothing.

Produces a human-readable report on stdout and, optionally, a machine-readable
JSON report. The audited directory is never modified, and no image is ever
copied into ``dataset/``: importing happens manually, after a license review
and a plate-type review. See ``docs/external_dataset_workflow.md``.

Usage::

    python scripts/audit_external_dataset.py <directory>
    python scripts/audit_external_dataset.py <dir> --json audit.json --report audit.txt
    python scripts/audit_external_dataset.py <dir> --source-id wikimedia-commons

Exit codes::

    0  the audit ran (findings are reported, not treated as failure)
    1  the directory could not be audited
    2  --fail-on-findings was given and the audit found problems
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow running the script directly
    sys.path.insert(0, str(REPO_ROOT))

from src.external_audit import (  # noqa: E402  (path set up above)
    DETAIL_LEVELS,
    FULL_DETAIL,
    AuditError,
    AuditReport,
    audit_dataset,
    write_json_report,
)

EXIT_OK: int = 0
EXIT_CANNOT_AUDIT: int = 1
EXIT_FINDINGS: int = 2

#: Longest list printed in full before it is summarised.
MAX_LISTED: int = 25

SEPARATOR: str = "-" * 72


def _section(title: str) -> list[str]:
    return ["", title, SEPARATOR]


def _listing(items: Sequence[str], indent: str = "  ") -> list[str]:
    if not items:
        return [f"{indent}(none)"]
    lines = [f"{indent}{item}" for item in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f"{indent}... and {len(items) - MAX_LISTED} more")
    return lines


def _counts(counts: Mapping[Any, int], indent: str = "  ") -> list[str]:
    if not counts:
        return [f"{indent}(none)"]
    width = max(len(str(key)) for key in counts)
    return [f"{indent}{str(key):<{width}}  {value:>7}" for key, value in counts.items()]


def build_report_text(report: AuditReport) -> str:
    """Render the human-readable audit report."""
    lines: list[str] = [
        "External dataset audit (inspection only -- nothing was imported)",
        SEPARATOR,
        f"directory : {report.root}",
        f"source_id : {report.source_id or '(not assigned)'}",
    ]

    lines += _section("Totals")
    lines += [
        f"  images                : {len(report.images)}",
        f"  readable images       : {len(report.readable_images)}",
        f"  corrupt/unreadable    : {len(report.corrupt_images)}",
        f"  extension mismatches  : {len(report.mismatched_images)}  "
        "(readable, but wrongly named)",
        f"  annotation files      : {len(report.annotations)}",
        f"  annotation boxes      : {report.total_boxes()}",
        f"  invalid annotation rows: {report.invalid_annotation_count()}",
        f"  CSV files             : {len(report.csv_files)}",
        f"  metadata files        : {len(report.metadata_files)}",
        f"  config files          : {len(report.configs)}",
        f"  other files           : {len(report.other_files)}",
    ]

    lines += _section("Image formats")
    lines += _counts(report.format_counts())

    lines += _section("Image dimensions")
    stats = report.dimension_stats()
    if not stats:
        lines.append("  (no readable images)")
    else:
        for axis in ("width", "height"):
            summary = stats[axis]
            lines.append(
                f"  {axis:<7} min {summary['min']}  median {summary['median']}  "
                f"mean {summary['mean']}  max {summary['max']}"
            )
        lines.append(f"  distinct resolutions: {stats['distinct_resolutions']}")
        lines.append("  most common:")
        lines += _counts(stats["most_common_resolutions"], indent="    ")

    lines += _section(f"Corrupt or unreadable images ({len(report.corrupt_images)})")
    if report.corrupt_images:
        for image in report.corrupt_images[:MAX_LISTED]:
            lines.append(f"  {image.path}: {'; '.join(image.problems)}")
        if len(report.corrupt_images) > MAX_LISTED:
            lines.append(f"  ... and {len(report.corrupt_images) - MAX_LISTED} more")
    else:
        lines.append("  (none)")
    lines.append(
        "  Note: headers are parsed, pixel data is not decoded. Corruption in"
    )
    lines.append("  the middle of an image stream is not detected by this audit.")

    lines += _section(
        f"Extension / format mismatches ({len(report.mismatched_images)})"
    )
    if not report.mismatched_images:
        lines.append("  (none)")
    else:
        lines.append("  These files are readable; only their names are wrong.")
        lines += _counts(report.mismatch_counts(), indent="    ")
        lines.append("")
        lines.append("  examples:")
        lines += _listing(
            [
                f"{image.path}: {'; '.join(image.warnings)}"
                for image in report.mismatched_images
            ],
            indent="    ",
        )
        lines.append("")
        lines.append(
            "  On import, decide the format from the file's content, not its"
        )
        lines.append(
            "  extension, and rename to match -- some image loaders trust the"
        )
        lines.append("  extension and will fail on these.")

    lines += _section("YOLO annotations")
    if not report.annotations:
        lines.append("  (no .txt annotation files found)")
    else:
        not_yolo = [a.path for a in report.annotations if not a.is_yolo_format]
        empty = [a.path for a in report.annotations if a.is_yolo_format and a.is_empty]
        lines += [
            f"  files parsed as YOLO  : {len(report.annotations) - len(not_yolo)}",
            f"  files not in YOLO form: {len(not_yolo)}",
            f"  empty files (negatives): {len(empty)}",
        ]
        if not_yolo:
            lines.append("  not YOLO format:")
            lines += _listing(not_yolo, indent="    ")

        invalid = [
            f"{a.path}:{entry['line']}: {entry['reason']}"
            for a in report.annotations
            for entry in a.invalid_lines
        ]
        lines.append(f"  invalid annotation rows ({len(invalid)}):")
        lines += _listing(invalid, indent="    ")

    lines += _section("Classes")
    counts = report.class_counts()
    names = report.class_names()
    if not counts:
        lines.append("  (no annotation boxes found)")
    else:
        width = max(len(str(key)) for key in counts)
        for class_id, count in counts.items():
            label = names.get(class_id)
            suffix = f"  declared name: {label!r}" if label else ""
            lines.append(f"  class {str(class_id):<{width}}  {count:>7} box(es){suffix}")
    unused = report.unused_declared_class_ids()
    undeclared = report.undeclared_class_ids()
    if unused:
        lines.append("")
        lines.append(
            "  declared but never used: "
            + ", ".join(f"{i} ({names[i]!r})" for i in unused)
        )
    if undeclared:
        lines.append("")
        lines.append(
            "  WARNING: used but not declared: "
            + ", ".join(str(i) for i in undeclared)
            + " -- the config and the labels disagree, so treat both as unreliable."
        )

    if names:
        lines.append("")
        lines.append("  Names above come from the dataset's own config file.")
    else:
        lines.append("")
        lines.append("  No class names were declared by the dataset.")
    lines += [
        "  Class meanings are NOT inferred by this tool. Mapping these ids onto",
        "  our plate_type values is a human decision made when the source is",
        "  approved -- see docs/external_dataset_workflow.md.",
    ]

    lines += _section("Image / annotation pairing")
    lines += [
        f"  images without annotations ({len(report.images_without_annotations)}):"
    ]
    lines += _listing(report.images_without_annotations, indent="    ")
    lines += [
        f"  annotations without images ({len(report.annotations_without_images)}):"
    ]
    lines += _listing(report.annotations_without_images, indent="    ")
    if report.ambiguous_stems:
        lines.append(f"  ambiguous stems ({len(report.ambiguous_stems)}):")
        for stem, paths in list(report.ambiguous_stems.items())[:MAX_LISTED]:
            lines.append(f"    {stem}: {', '.join(paths)}")

    lines += _section("Duplicates")
    lines.append(f"  duplicate file names ({len(report.duplicate_filenames)}):")
    if report.duplicate_filenames:
        for name, paths in list(report.duplicate_filenames.items())[:MAX_LISTED]:
            lines.append(f"    {name}: {', '.join(paths)}")
    else:
        lines.append("    (none)")

    lines.append(f"  identical images by SHA-256 ({len(report.duplicate_images)}):")
    if report.duplicate_images:
        for group in report.duplicate_images[:MAX_LISTED]:
            lines.append(f"    {' == '.join(group)}")
    else:
        lines.append("    (none)")

    lines += _section("CSV files")
    if not report.csv_files:
        lines.append("  (none)")
    for csv_file in report.csv_files:
        if csv_file.error:
            lines.append(f"  {csv_file.path}: {csv_file.error}")
            continue
        lines.append(
            f"  {csv_file.path}: {csv_file.row_count} row(s), "
            f"delimiter {csv_file.delimiter!r}"
        )
        lines.append(f"    columns: {', '.join(csv_file.header) or '(none)'}")

    lines += _section("Dataset paperwork")
    if not report.metadata_files and not report.configs:
        lines.append("  (none found)")
        lines.append("  A dataset with no LICENSE cannot be approved. Find the")
        lines.append("  license at the source, or reject it.")
    for meta in report.metadata_files:
        lines.append(f"  [{meta.kind}] {meta.path} ({meta.size_bytes} bytes)")
        for preview_line in meta.preview.splitlines()[:8]:
            lines.append(f"    | {preview_line}")
        lines.append("")
    for config in report.configs:
        lines.append(f"  [config] {config.path}")
        if config.declared_class_count is not None:
            lines.append(f"    declared class count: {config.declared_class_count}")
        if config.class_names:
            for class_id, name in sorted(config.class_names.items()):
                lines.append(f"    {class_id}: {name}")
        if config.parse_note:
            lines.append(f"    note: {config.parse_note}")

    if report.other_files:
        lines += _section(f"Other files ({len(report.other_files)})")
        lines += _listing(report.other_files)

    if report.unreadable_files:
        lines += _section(f"Files that could not be read ({len(report.unreadable_files)})")
        lines += _listing([f"{e['path']}: {e['error']}" for e in report.unreadable_files])

    lines += _section("Next steps")
    lines += [
        "  This audit imports nothing. Before any image enters dataset/:",
        "    1. Human license review  -- may we redistribute under CC BY 4.0?",
        "    2. Human plate-type review -- what do these classes actually show?",
        "    3. Register the source in docs/data_sources.md.",
        "    4. Only then import, and re-run validate_dataset_local.py.",
        "",
    ]
    return "\n".join(lines)


def has_findings(report: AuditReport) -> bool:
    """Whether the audit found anything a person needs to look at."""
    return bool(
        report.corrupt_images
        or report.invalid_annotation_count()
        or report.images_without_annotations
        or report.annotations_without_images
        or report.duplicate_filenames
        or report.duplicate_images
        or report.unreadable_files
    )


def _refuse_writing_inside_dataset(destination: Path, audited: Path) -> None:
    """Guard the tool's inspection-only promise.

    Writing a report into the audited directory would modify it, and writing
    one into ``dataset/`` would put unreviewed material where the competition
    dataset lives. Both are refused.
    """
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
        prog="audit_external_dataset.py",
        description=(
            "Audit an external dataset directory. Inspection only: nothing is "
            "modified and nothing is imported."
        ),
    )
    parser.add_argument("directory", type=Path, help="the dataset directory to audit")
    parser.add_argument(
        "--source-id",
        default=None,
        help="source_id to record in the report, for later linking to docs/data_sources.md",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the JSON report here")
    parser.add_argument(
        "--json-detail",
        choices=sorted(DETAIL_LEVELS),
        default=FULL_DETAIL,
        help=(
            "how much the JSON report carries. '%(default)s' keeps per-image and "
            "per-annotation records; 'summary' keeps statistics and findings only, "
            "omitting the audited dataset's own annotation coordinates -- use it "
            "for a report you intend to commit"
        ),
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the text report here"
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="skip SHA-256 duplicate detection (faster on very large datasets)",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="exit 2 if the audit found anything needing review",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the totals")
    parser.add_argument("--log-level", default="WARNING", help="logging level")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    try:
        for destination in (args.json, args.report):
            if destination is not None:
                _refuse_writing_inside_dataset(destination, args.directory)

        report = audit_dataset(
            args.directory, source_id=args.source_id, hash_images=not args.no_hash
        )
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_CANNOT_AUDIT

    text = build_report_text(report)
    if args.quiet:
        print(
            f"{len(report.images)} image(s), {len(report.annotations)} annotation "
            f"file(s), {len(report.corrupt_images)} corrupt, "
            f"{report.invalid_annotation_count()} invalid annotation row(s)"
        )
    else:
        print(text)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
        print(f"Text report written to {args.report}")

    if args.json is not None:
        write_json_report(report, args.json, detail=args.json_detail)
        print(f"JSON report written to {args.json} (detail: {args.json_detail})")

    if args.fail_on_findings and has_findings(report):
        return EXIT_FINDINGS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
