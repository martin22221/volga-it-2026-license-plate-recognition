#!/usr/bin/env python
"""Build a full human review for a small candidate `type1a` dataset.

Every image is put forward for inspection -- no sampling, no filtering -- with
a large preview, a close-up of the annotated plate and a blank verdict area.

Read-only: the external dataset is never modified, copied or imported.

Usage::

    python scripts/build_type1a_review.py <dataset-dir> --source-id <id> \\
        --out data/review/<name>

Exit codes::

    0  the review was built
    1  the dataset could not be read, or an output path was refused
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.external_audit import AuditError, audit_dataset  # noqa: E402
from src.rare_review import (  # noqa: E402
    TYPE1A_CANDIDATE,
    TYPE1A_CRITERIA,
    TYPE1A_FLAG_PROMPTS,
    RareCandidate,
    build_page,
    write_type1a_csv,
)
from src.review_sample import build_review_items, discover_pairs  # noqa: E402

EXIT_OK: int = 0
EXIT_FAILED: int = 1

#: Bigger than the rare-class pages: the whole question here is whether the
#: characters sit on one line or two, which needs a genuinely large close-up.
PREVIEW_HEIGHT_PX: int = 360
CROP_HEIGHT_PX: int = 260
COLUMN_WIDTH_PX: int = 560

INTRO: str = """
<b>Nothing on this page is confirmed.</b>
The dataset is <i>named</i> "two-line-russian-license-plates" and its single
YOLO class is <code>license-plate</code>. Neither is evidence: the name is the
uploader's claim, and the class says only that a plate was annotated. Every
image here is a <code>type1a_candidate</code> and nothing more.
<p>Judge each one from the close-up, then record your verdict in
<code>review.csv</code>. The page itself is read-only.</p>
"""


def _refuse_unsafe_output(destination: Path, audited: Path) -> None:
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


def build_statistics(
    directory: Path, source_id: str, candidates: Sequence[RareCandidate], report
) -> str:
    """Render the plain-text statistics report."""
    line = "-" * 72
    flags: dict[str, int] = {}
    for candidate in candidates:
        for flag in candidate.item.flags:
            flags[flag] = flags.get(flag, 0) + 1

    aspects = sorted(
        candidate.item.largest_plate.aspect
        for candidate in candidates
        if candidate.item.largest_plate
    )

    out: list[str] = [
        f"type1a review statistics - {source_id}",
        line,
        f"dataset        : {directory}",
        f"images         : {len(report.images)}",
        f"annotation files: {len(report.annotations)}",
        f"annotation boxes: {report.total_boxes()}",
        f"invalid rows   : {report.invalid_annotation_count()}",
        f"corrupt images : {len(report.corrupt_images)}",
        f"duplicate images (SHA-256): {len(report.duplicate_images)} group(s)",
        f"images without labels     : {len(report.images_without_annotations)}",
        f"labels without images     : {len(report.annotations_without_images)}",
        "",
        "Declared classes",
        line,
    ]
    names = report.class_names()
    if names:
        for class_id, name in sorted(names.items()):
            count = report.class_counts().get(class_id, 0)
            out.append(f"  {class_id}: {name}  ({count} box(es))")
    else:
        out.append("  (none declared)")

    out += ["", "Candidates", line, f"  type1a candidates : {len(candidates)}"]
    out.append(f"  candidate_plate_type written: {TYPE1A_CANDIDATE}")
    out.append("  HUMAN-CONFIRMED type1a: 0   (nothing confirmed until reviewed)")

    out += ["", "Box aspect ratios", line]
    if aspects:
        middle = aspects[len(aspects) // 2]
        out += [
            f"  min {aspects[0]:.2f}  median {middle:.2f}  max {aspects[-1]:.2f}",
            "  Russian square/two-line plate is nominally 1.71; a one-line plate",
            "  is 4.64. A tight cluster near 1.71 is consistent with two-line",
            "  plates, but consistency is not confirmation -- a square crop or a",
            "  steeply angled one-line plate lands in the same place.",
        ]

    out += ["", "Geometry flags", line]
    for flag, count in sorted(flags.items(), key=lambda e: (-e[1], e[0])):
        out.append(f"  {flag:<32} {count:>4}")

    out += ["", "Requires a person, per image", line]
    for text in TYPE1A_CRITERIA:
        out.append(f"  - {text}")
    out.append("")
    for text in TYPE1A_FLAG_PROMPTS:
        out.append(f"  ! flag: {text}")
    out.append("")
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_type1a_review.py",
        description="Build a full human review for a candidate type1a dataset.",
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    directory: Path = args.directory
    try:
        if not directory.is_dir():
            raise AuditError(f"{directory} is not a directory")
        _refuse_unsafe_output(args.out, directory)
        pairs = discover_pairs(directory)
        if not pairs:
            raise AuditError(f"no images found under {directory}")
        report = audit_dataset(directory, source_id=args.source_id)
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED

    class_names = report.class_names()
    items = build_review_items(directory, sorted(pairs), pairs)
    candidates = [
        RareCandidate(
            item=item,
            reasons=list(item.flags),
            class_label=_class_label(item, class_names),
        )
        for item in items
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    sheet = args.out / "contact_sheet.html"
    csv_path = args.out / "review.csv"
    stats_path = args.out / "statistics.txt"

    sheet.write_text(
        build_page(
            candidates,
            directory,
            title=f"type1a candidate review - {args.source_id}",
            intro_html=INTRO,
            verdict=True,
            preview_height=PREVIEW_HEIGHT_PX,
            crop_height=CROP_HEIGHT_PX,
            column_width=COLUMN_WIDTH_PX,
        ),
        encoding="utf-8",
    )
    rows = write_type1a_csv(candidates, csv_path)
    statistics = build_statistics(directory, args.source_id, candidates, report)
    stats_path.write_text(statistics + "\n", encoding="utf-8")

    print(statistics)
    print(f"  contact sheet : {sheet}")
    print(f"  review CSV    : {csv_path} ({rows} rows)")
    print(f"  statistics    : {stats_path}")
    return EXIT_OK


def _class_label(item, class_names: dict[int, str]) -> str:
    """The dataset's own name for the annotated class, or the raw id."""
    ids = sorted({plate.class_id for plate in item.plates})
    if not ids:
        return ""
    return ", ".join(str(class_names.get(i, i)) for i in ids)


if __name__ == "__main__":
    raise SystemExit(main())
