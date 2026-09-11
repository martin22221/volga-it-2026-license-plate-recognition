#!/usr/bin/env python
"""Build focused review pages for the rare competition classes.

Collects shape candidates from the existing seeded review sample, scans the
whole split for yellow plate colour, and writes two large-preview HTML pages, a
CSV and a statistics report.

Read-only: the external dataset is never modified, copied or imported.

Usage::

    python scripts/build_rare_review.py <dataset-dir> --source-id <id> \\
        --out-html data/review/<id> --out-data data/audits/<id>_review

Exit codes::

    0  the pages were built
    1  the dataset could not be read, or an output path was refused
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.external_audit import AuditError  # noqa: E402
from src.plate_color import PlateColour, read_plate_colour  # noqa: E402
from src.rare_review import (  # noqa: E402
    RARE_INTRO,
    REASON_YELLOW,
    REASON_YELLOW_NEAR,
    YELLOW_INTRO,
    YELLOW_SHORTLIST,
    RareCandidate,
    build_page,
    collect_shape_candidates,
    merge_candidates,
    write_rare_csv,
)
from src.review_sample import (  # noqa: E402
    REVIEW_SAMPLE_SIZE,
    REVIEW_SEED,
    build_review_item,
    build_review_items,
    discover_pairs,
    select_sample,
)

EXIT_OK: int = 0
EXIT_FAILED: int = 1


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


def scan_colours(
    root: Path, pairs: dict[str, tuple[Path, Path | None]]
) -> dict[str, tuple[RareCandidate, PlateColour]]:
    """Measure plate colour for every image that has an annotation."""
    results: dict[str, tuple[RareCandidate, PlateColour]] = {}
    started = time.time()
    for index, stem in enumerate(sorted(pairs), start=1):
        image_path, label_path = pairs[stem]
        item = build_review_item(image_path, label_path, root)
        if not item.plates:
            continue
        plate = item.largest_plate
        assert plate is not None

        colour = read_plate_colour(
            image_path, plate.centre_x, plate.centre_y, plate.norm_width, plate.norm_height
        )
        candidate = RareCandidate(item=item, colour=colour)
        if colour.plate is not None:
            candidate.yellowness = colour.plate.yellowness
        results[item.image_path] = (candidate, colour)

        if index % 250 == 0:
            logger.warning(
                "colour scan %d/%d (%.0fs elapsed)", index, len(pairs), time.time() - started
            )

    logger.warning("colour scan finished in %.0fs", time.time() - started)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_rare_review.py",
        description="Build focused type1a / type1b review pages. Read-only.",
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out-html", type=Path, required=True, help="directory for HTML pages")
    parser.add_argument("--out-data", type=Path, required=True, help="directory for CSV/stats")
    parser.add_argument("--size", type=int, default=REVIEW_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=REVIEW_SEED)
    parser.add_argument(
        "--shortlist", type=int, default=YELLOW_SHORTLIST,
        help="how many most-yellow plates to show even below threshold",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


logger = logging.getLogger("build_rare_review")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")

    directory: Path = args.directory
    try:
        if not directory.is_dir():
            raise AuditError(f"{directory} is not a directory")
        for destination in (args.out_html, args.out_data):
            _refuse_unsafe_output(destination, directory)
        pairs = discover_pairs(directory)
        if not pairs:
            raise AuditError(f"no images found under {directory}")
    except AuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED

    # --- shape candidates, from the same seeded sample as before ----------
    sample_stems = select_sample(pairs, size=args.size, seed=args.seed)
    sample_items = build_review_items(directory, sample_stems, pairs)
    shape = collect_shape_candidates(sample_items)

    # --- colour, across the whole split -----------------------------------
    colours = scan_colours(directory, pairs)

    threshold_hits: dict[str, RareCandidate] = {}
    ranked: list[tuple[float, str]] = []
    unreliable = 0
    for path, (candidate, colour) in colours.items():
        if not colour.reliable:
            unreliable += 1
        if colour.is_yellow_candidate():
            candidate.reasons.append(REASON_YELLOW)
            threshold_hits[path] = candidate
        elif candidate.yellowness is not None and colour.reliable:
            ranked.append((candidate.yellowness, path))

    ranked.sort(reverse=True)
    shortlist: dict[str, RareCandidate] = {}
    for yellowness, path in ranked[: args.shortlist]:
        candidate = colours[path][0]
        candidate.reasons.append(REASON_YELLOW_NEAR)
        shortlist[path] = candidate

    yellow_all = {**threshold_hits, **shortlist}
    merged = merge_candidates(shape, yellow_all)

    # --- write artefacts ---------------------------------------------------
    args.out_html.mkdir(parents=True, exist_ok=True)
    args.out_data.mkdir(parents=True, exist_ok=True)

    rare_page = args.out_html / "rare_candidates.html"
    yellow_page = args.out_html / "yellow_candidates.html"
    csv_path = args.out_data / "rare_candidates.csv"
    stats_path = args.out_data / "rare_candidates_stats.json"

    shape_sorted = sorted(shape.values(), key=lambda c: c.image_path)
    yellow_sorted = sorted(
        yellow_all.values(), key=lambda c: -(c.yellowness or -999)
    )

    rare_page.write_text(
        build_page(
            shape_sorted, directory,
            title=f"type1a shape candidates - {args.source_id}",
            intro_html=RARE_INTRO,
        ),
        encoding="utf-8",
    )
    yellow_page.write_text(
        build_page(
            yellow_sorted, directory,
            title=f"type1b yellow candidates - {args.source_id}",
            intro_html=YELLOW_INTRO,
        ),
        encoding="utf-8",
    )

    merged_sorted = sorted(merged.values(), key=lambda c: c.image_path)
    rows = write_rare_csv(merged_sorted, csv_path)

    square = sum(1 for c in shape.values() if "square_or_two_line_candidate" in c.reasons)
    ambiguous = sum(1 for c in shape.values() if "ambiguous_shape" in c.reasons)
    overlap_shape_yellow = sorted(set(shape) & set(yellow_all))
    yellownesses = [
        candidate.yellowness
        for candidate, colour in colours.values()
        if candidate.yellowness is not None and colour.reliable
    ]

    stats = {
        "source_id": args.source_id,
        "dataset": str(directory),
        "sample_seed": args.seed,
        "sample_size": len(sample_items),
        "total_test_images_scanned": len(pairs),
        "colour_scanned": len(colours),
        "colour_unreliable": unreliable,
        "square_or_two_line_candidates": square,
        "ambiguous_shape_candidates": ambiguous,
        "shape_candidates_total": len(shape),
        "yellow_threshold_candidates": len(threshold_hits),
        "yellow_shortlist_shown": len(shortlist),
        "overlap_shape_and_yellow": len(overlap_shape_yellow),
        "overlap_images": overlap_shape_yellow,
        "unique_candidate_images": len(merged),
        "csv_rows": rows,
        "human_confirmed_type1a": 0,
        "human_confirmed_type1b": 0,
        "yellowness_percentiles": _percentiles(yellownesses),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    _print_report(stats, rare_page, yellow_page, csv_path, stats_path)
    return EXIT_OK


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    size = len(ordered)

    def at(fraction: float) -> float:
        return round(ordered[min(size - 1, int(size * fraction))], 2)

    return {
        "min": round(ordered[0], 2),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": round(ordered[-1], 2),
    }


def _print_report(
    stats: dict[str, object], rare: Path, yellow: Path, csv_path: Path, stats_path: Path
) -> None:
    line = "-" * 72
    print(f"Rare-class review - {stats['source_id']}")
    print(line)
    print(f"  total test images scanned     : {stats['total_test_images_scanned']}")
    print(f"  colour readings taken         : {stats['colour_scanned']}"
          f"  (unreliable: {stats['colour_unreliable']})")
    print(f"  square/two-line candidates    : {stats['square_or_two_line_candidates']}")
    print(f"  ambiguous-shape candidates    : {stats['ambiguous_shape_candidates']}")
    print(f"  yellow candidates (threshold) : {stats['yellow_threshold_candidates']}")
    print(f"  most-yellow shortlist shown   : {stats['yellow_shortlist_shown']}")
    print(f"  overlap shape & yellow        : {stats['overlap_shape_and_yellow']}")
    print(f"  unique candidate images       : {stats['unique_candidate_images']}")
    print()
    print(f"  HUMAN-CONFIRMED type1a        : {stats['human_confirmed_type1a']}")
    print(f"  HUMAN-CONFIRMED type1b        : {stats['human_confirmed_type1b']}")
    print("  (nothing is confirmed until a person fills human_plate_type)")
    print()
    print(f"  plate yellowness percentiles  : {stats['yellowness_percentiles']}")
    print()
    print(f"  rare page  : {rare}")
    print(f"  yellow page: {yellow}")
    print(f"  CSV        : {csv_path}")
    print(f"  stats      : {stats_path}")


if __name__ == "__main__":
    raise SystemExit(main())
