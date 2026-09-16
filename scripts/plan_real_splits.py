#!/usr/bin/env python
"""Assign accepted real images to train / val / holdout, and audit for leakage.

The split is group-aware and deterministic: a group's split comes from a hash
of the group name and the seed, so the same inputs always give the same answer.
Groups already in the manifest keep their split -- new images never move an
existing one. That is what keeps the holdout honest.

Usage::

    # check the frozen manifest for leakage (no writes)
    python scripts/plan_real_splits.py --check

    # assign staged images, honouring everything already frozen
    python scripts/plan_real_splits.py --add data/real_staging/accepted/<source_id> --write

    # what would change, without writing
    python scripts/plan_real_splits.py --add <dir>

Exit codes::

    0  the manifest is clean (and was written, with --write)
    1  the inputs could not be read
    2  leakage or a conflict was found
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_meta import read_meta  # noqa: E402
from src.real_splits import (  # noqa: E402
    DEFAULT_WEIGHTS,
    SplitError,
    assign_rows,
    leakage_report,
    leakage_text,
    read_manifest,
    write_manifest,
)

DEFAULT_MANIFEST = REPO_ROOT / "dataset" / "splits" / "real_splits.csv"
EXIT_OK, EXIT_UNUSABLE, EXIT_LEAKAGE = 0, 1, 2


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="split manifest (default: dataset/splits/real_splits.csv)")
    parser.add_argument("--add", type=Path, action="append", default=[], metavar="DIR",
                        help="folder holding meta.csv and groups.csv of accepted images; repeatable")
    parser.add_argument("--seed", type=int, default=20260916, help="split seed; changing it re-rolls only unfrozen groups")
    parser.add_argument("--train", type=float, default=DEFAULT_WEIGHTS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_WEIGHTS["val"])
    parser.add_argument("--holdout", type=float, default=DEFAULT_WEIGHTS["holdout"])
    parser.add_argument("--write", action="store_true", help="write the manifest (otherwise report only)")
    parser.add_argument("--check", action="store_true", help="only audit the existing manifest for leakage")
    return parser.parse_args(argv)


def _read_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    return {(r.get("image") or "").strip().replace("\\", "/"): (r.get("group") or "").strip() for r in rows if r.get("image")}


def collect(directories: Sequence[Path]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """``(image, group, source)`` for every annotated image in the folders."""
    items: list[tuple[str, str, str]] = []
    problems: list[str] = []
    for directory in directories:
        meta = directory / "meta.csv"
        if not meta.exists():
            problems.append(f"{directory}: no meta.csv")
            continue
        groups = _read_groups(directory / "groups.csv")
        _, rows = read_meta(meta)
        for image in sorted({row.image for row in rows}):
            group = groups.get(image)
            if not group:
                problems.append(f"{directory}: {image} has no group in groups.csv")
                continue
            source = next((row.get("source") for row in rows if row.image == image), "")
            items.append((image, group, source))
    return items, problems


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frozen = read_manifest(args.manifest)
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_UNUSABLE

    if args.check or not args.add:
        findings = leakage_report(frozen)
        print(f"manifest: {args.manifest}")
        print(leakage_text(findings))
        return EXIT_OK if findings["clean"] else EXIT_LEAKAGE

    items, problems = collect(args.add)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return EXIT_UNUSABLE
    try:
        rows = assign_rows(
            items,
            seed=args.seed,
            weights={"train": args.train, "val": args.val, "holdout": args.holdout},
            frozen_rows=frozen,
            assigned_on=time.strftime("%Y-%m-%d"),
        )
    except SplitError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_LEAKAGE

    findings = leakage_report(rows)
    added = len(rows) - len(frozen)
    print(f"manifest: {args.manifest}  ({len(frozen)} frozen row(s), {added} new)")
    print(leakage_text(findings))
    if not findings["clean"]:
        return EXIT_LEAKAGE
    if args.write:
        write_manifest(args.manifest, rows)
        print(f"written: {args.manifest}")
    else:
        print("dry run: pass --write to update the manifest")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
