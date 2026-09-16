#!/usr/bin/env python
"""Audit one staged real-image source. Inspection only -- copies nothing.

Reads a folder under ``data/real_staging/incoming/`` together with its source
record, annotations and privacy review, and reports what is there and what
blocks promotion into ``dataset/images/real/``.

**It never approves a source and never infers legal rights.** The decision
lives in the source record, written by a person; this script only checks that
the decision is consistent with the rights recorded beside it.

Usage::

    python scripts/audit_real_source.py data/real_staging/incoming/<source_id>
    python scripts/audit_real_source.py <dir> --json audit.json --report audit.txt
    python scripts/audit_real_source.py <dir> --splits dataset/splits/real_splits.csv
    python scripts/audit_real_source.py --blank-record <source_id>   # print a record to fill in

Exit codes::

    0  the audit ran and found nothing blocking
    1  the folder could not be audited
    2  the audit found blocking problems
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.real_intake import (  # noqa: E402
    audit_staged_source,
    audit_text,
    blank_record,
    dataset_hashes,
)
from src.real_splits import read_manifest  # noqa: E402

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKING = 0, 1, 2


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path, help="staged source folder")
    parser.add_argument("--record", type=Path, default=None, help="source record JSON (default: <dir>/source_record.json)")
    parser.add_argument("--meta", type=Path, default=None, help="staged annotations (default: <dir>/meta.csv)")
    parser.add_argument("--privacy", type=Path, default=None, help="privacy review CSV (default: <dir>/privacy_review.csv)")
    parser.add_argument("--groups", type=Path, default=None, help="image;group CSV (default: <dir>/groups.csv)")
    parser.add_argument("--splits", type=Path, default=None, help="split manifest to check the staged images against")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset", help="dataset root, to spot images already in it")
    parser.add_argument("--no-near-duplicates", action="store_true", help="skip the thumbnail comparison")
    parser.add_argument("--json", type=Path, default=None, help="also write the machine-readable report here")
    parser.add_argument("--report", type=Path, default=None, help="also write the text report here")
    parser.add_argument("--blank-record", metavar="SOURCE_ID", default=None, help="print an empty source record and exit")
    return parser.parse_args(argv)


def _read_groups(path: Path | None) -> dict[str, str]:
    import csv

    if path is None or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    return {(r.get("image") or "").strip().replace("\\", "/"): (r.get("group") or "").strip() for r in rows if r.get("image")}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.blank_record:
        print(json.dumps(blank_record(args.blank_record), indent=1))
        return EXIT_OK
    if args.directory is None:
        print("error: a staged source folder is required (or --blank-record)", file=sys.stderr)
        return EXIT_UNUSABLE
    if not args.directory.is_dir():
        print(f"error: {args.directory} is not a directory", file=sys.stderr)
        return EXIT_UNUSABLE

    known: dict[str, str] = {}
    meta = args.dataset / "meta.csv"
    if meta.exists():
        known = dataset_hashes(meta, args.dataset)

    split_assignment: dict[str, str] = {}
    if args.splits and Path(args.splits).exists():
        split_assignment = {row.image: row.split for row in read_manifest(args.splits)}

    audit = audit_staged_source(
        args.directory,
        record_path=args.record,
        meta_path=args.meta,
        privacy_path=args.privacy,
        split_assignment=split_assignment or None,
        group_of=_read_groups(args.groups or (args.directory / "groups.csv")) or None,
        known_hashes=known or None,
        near_duplicates=not args.no_near_duplicates,
    )
    text = audit_text(audit)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(audit.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    return EXIT_BLOCKING if audit.blocking else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
