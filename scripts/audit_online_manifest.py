#!/usr/bin/env python
"""Audit an online candidate manifest. Inspection only -- acquires nothing.

Reads a candidate manifest written by ``discover_commons_candidates.py`` (or by
hand) and reports what it contains and what blocks it: rows whose recorded
rights do not support their decision, unverified provenance, ambiguous plate
classes, duplicate ids and duplicate source URLs.

**It never approves acquisition and never downloads an image.** A person
decides; this only checks that the evidence written beside a decision supports
it.

Usage::

    python scripts/audit_online_manifest.py data/real_staging/manifests/<name>.csv
    python scripts/audit_online_manifest.py <csv> --json out.json --report out.txt

Exit codes::

    0  the manifest is internally consistent
    1  the manifest could not be read
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

from src.online_discovery import audit_manifest, audit_text, read_manifest  # noqa: E402

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKING = 0, 1, 2


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="candidate manifest CSV")
    parser.add_argument("--json", type=Path, default=None, help="also write the audit as JSON")
    parser.add_argument("--report", type=Path, default=None, help="also write the text report")
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

    audit = audit_manifest(rows, path=str(args.manifest))
    text = audit_text(audit)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(audit.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8"
        )
    return EXIT_BLOCKING if audit.blocking else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
