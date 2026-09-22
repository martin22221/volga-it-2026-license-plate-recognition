#!/usr/bin/env python
"""Fetch the photographs an online-discovery manifest has already approved.

This is the only tool in the repository that downloads a photograph, and it
does so under narrow conditions: the manifest row must already say
``ACCEPT_FOR_SUBMISSION``, and the file's licence at the source must still be
the licence that was approved. Anything else is **held**, per candidate, and
never replaced by a substitute.

It writes into ``data/real_staging/incoming/<source_id>/`` and nowhere else.
Nothing enters ``dataset/`` here; see ``docs/real_data_intake.md`` for what
follows.

Usage::

    python scripts/acquire_online_candidates.py \\
        data/real_staging/manifests/online_candidates_2026-09-19.csv \\
        --source-id wikimedia_commons_curated

    python scripts/acquire_online_candidates.py <manifest> --source-id <id> --dry-run

Exit codes::

    0  every approved candidate was acquired
    1  the manifest could not be read
    2  at least one candidate was held or failed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.online_acquisition import (  # noqa: E402
    ACQUISITION_FIELDS,
    acquire,
    approved_rows,
    fetch_commons_metadata,
    check_against_source,
    title_from_file_page,
)
from src.online_discovery import read_manifest  # noqa: E402

EXIT_OK, EXIT_UNUSABLE, EXIT_HELD = 0, 1, 2


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="candidate manifest CSV")
    parser.add_argument(
        "--source-id", required=True, help="staging source id, e.g. wikimedia_commons_curated"
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=REPO_ROOT / "data" / "real_staging" / "incoming",
        help="where the staging folder lives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify every approved candidate at the source, download nothing",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest.is_file():
        print(f"error: {args.manifest} is not a file", file=sys.stderr)
        return EXIT_UNUSABLE
    try:
        rows = read_manifest(args.manifest)
    except (OSError, UnicodeDecodeError) as error:
        print(f"error: cannot read {args.manifest}: {error}", file=sys.stderr)
        return EXIT_UNUSABLE

    approved = approved_rows(rows)
    print(f"manifest      : {args.manifest}")
    print(f"rows          : {len(rows)}")
    print(f"approved      : {len(approved)}  (ACCEPT_FOR_SUBMISSION only)")
    if not approved:
        print("nothing approved; nothing to do")
        return EXIT_OK

    staging = args.staging_root / args.source_id
    originals = staging / "originals"

    if args.dry_run:
        titles = [title_from_file_page(r.get("file_page_url", "")) for r in approved]
        live = fetch_commons_metadata(titles, timeout=args.timeout)
        held = 0
        for row, title in zip(approved, titles):
            problems = check_against_source(row, live.get(title))
            if problems:
                held += 1
                print(f"  [HOLD] {row['candidate_id']}: {'; '.join(problems)}")
            else:
                print(f"  [ok  ] {row['candidate_id']} {title[:64]}")
        print(f"\ndry run: {len(approved) - held} would be acquired, {held} held")
        return EXIT_HELD if held else EXIT_OK

    results = acquire(approved, originals_dir=originals, timeout=args.timeout)

    records = [r.record for r in results if r.record]
    out = staging / "acquisition_record.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ACQUISITION_FIELDS, delimiter=";", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)

    held = [r for r in results if not r.acquired]
    if held:
        (staging / "held.json").write_text(
            json.dumps(
                [{"candidate_id": r.candidate_id, "status": r.status, "reason": r.notes} for r in held],
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"\nacquired      : {len(records)}")
    print(f"held / failed : {len(held)}")
    print(f"originals     : {originals}")
    print(f"record        : {out}")
    return EXIT_HELD if held else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
