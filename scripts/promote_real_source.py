#!/usr/bin/env python
"""Move a reviewed source from staging into the dataset, PASS rows only.

Two steps, each refusing to run on anything a person has not cleared:

``--accept``   copies the images a human review marked ``PASS`` out of
               ``incoming/<source_id>/`` into ``accepted/<source_id>/``, with
               their annotations, groups and privacy rows. ``QUESTIONABLE`` and
               ``FAIL`` images are left behind, and the reason each was left is
               printed.

``--promote``  copies ``accepted/<source_id>/`` into ``dataset/images/real/``
               and appends its rows to ``dataset/meta.csv``.

Promotion refuses unless the source record says ``ACCEPT_FOR_SUBMISSION`` with
a ``decided_by``, and unless the staged-source audit of the accepted folder has
nothing blocking. The verdicts come from ``review_verdicts.json``, which a
person signs off against the contact sheets -- this script never decides a
verdict, and never promotes an image that is not on the PASS list.

Usage::

    python scripts/promote_real_source.py wikimedia_commons_curated --accept
    python scripts/promote_real_source.py wikimedia_commons_curated --promote
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_meta import CSV_DELIMITER, REQUIRED_COLUMNS  # noqa: E402
from src.real_intake import audit_staged_source, load_source_record  # noqa: E402

STAGING = REPO_ROOT / "data" / "real_staging"
DATASET = REPO_ROOT / "dataset"

#: The only verdict that may be promoted. A person writes it; we only read it.
PROMOTABLE_VERDICT = "PASS"

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKED = 0, 1, 2


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=CSV_DELIMITER))


def _write(path: Path, rows: list[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), delimiter=CSV_DELIMITER, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def accept(source_id: str) -> int:
    """Copy only the PASS images into accepted/<source_id>/."""
    incoming = STAGING / "incoming" / source_id
    accepted = STAGING / "accepted" / source_id
    verdict_file = incoming / "review_verdicts.json"
    if not verdict_file.is_file():
        print(f"error: no {verdict_file}; a human review must record its verdicts first",
              file=sys.stderr)
        return EXIT_UNUSABLE

    payload = json.loads(verdict_file.read_text(encoding="utf-8"))
    verdicts = {Path(k).name: v for k, v in payload.get("verdicts", {}).items()}
    notes = {Path(k).name: v for k, v in payload.get("notes", {}).items()}

    passed = {name for name, v in verdicts.items() if v == PROMOTABLE_VERDICT}
    held = {name: v for name, v in verdicts.items() if v != PROMOTABLE_VERDICT}

    print(f"verdicts: {len(passed)} {PROMOTABLE_VERDICT}, {len(held)} not promoted")
    for name, verdict in sorted(held.items()):
        print(f"  [{verdict:<12}] {name} -- left in incoming/  {notes.get(name,'')[:90]}")

    if accepted.exists():
        shutil.rmtree(accepted)
    dest_images = accepted / "images" / "real" / source_id
    dest_images.mkdir(parents=True)

    copied = []
    for name in sorted(passed):
        rel = f"images/real/{source_id}/{name}"
        src = incoming / rel
        if not src.is_file():
            print(f"error: {src} is missing", file=sys.stderr)
            return EXIT_UNUSABLE
        shutil.copy2(src, dest_images / name)
        copied.append(rel)

    keep = set(copied)
    meta = [r for r in _read(incoming / "meta.csv") if r["image"] in keep]
    groups = [r for r in _read(incoming / "groups.csv") if r["image"] in keep]
    privacy = [r for r in _read(incoming / "privacy_review.csv") if r["image"] in keep]
    acq = [r for r in _read(incoming / "acquisition_record.csv") if r["staged_image"] in keep]

    _write(accepted / "meta.csv", meta, REQUIRED_COLUMNS)
    _write(accepted / "groups.csv", groups, ["image", "group"])
    _write(accepted / "privacy_review.csv", privacy,
           ["image", "faces_present", "action", "reviewer", "date"])
    if acq:
        _write(accepted / "acquisition_record.csv", acq, list(acq[0].keys()))
    shutil.copy2(incoming / "source_record.json", accepted / "source_record.json")

    print(f"\naccepted: {len(copied)} image(s), {len(meta)} annotation row(s) -> {accepted}")
    for row in meta:
        print(f"  {Path(row['image']).name:<14} {row['plate_type']:<7} {row['plate_num']}")
    return EXIT_OK


def promote(source_id: str) -> int:
    """Copy accepted/<source_id>/ into dataset/ and append its meta rows."""
    accepted = STAGING / "accepted" / source_id
    record = load_source_record(accepted / "source_record.json")

    if record.decision != "ACCEPT_FOR_SUBMISSION":
        print(f"refusing: source decision is {record.decision!r}, not ACCEPT_FOR_SUBMISSION",
              file=sys.stderr)
        return EXIT_BLOCKED
    if record.problems:
        print("refusing: the source record does not support that decision:", file=sys.stderr)
        for problem in record.problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_BLOCKED

    audit = audit_staged_source(accepted)
    if audit.blocking:
        print("refusing: the accepted folder has blocking findings:", file=sys.stderr)
        for problem in audit.blocking:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_BLOCKED

    meta_rows = _read(accepted / "meta.csv")
    dest = DATASET / "images" / "real" / source_id
    dest.mkdir(parents=True, exist_ok=True)

    existing = _read(DATASET / "meta.csv")
    already = {r["image"] for r in existing}
    clash = already & {r["image"] for r in meta_rows}
    if clash:
        print(f"refusing: {len(clash)} row(s) already in dataset/meta.csv, e.g. {sorted(clash)[:3]}",
              file=sys.stderr)
        return EXIT_BLOCKED

    copied = 0
    for name in sorted({Path(r["image"]).name for r in meta_rows}):
        shutil.copy2(accepted / "images" / "real" / source_id / name, dest / name)
        copied += 1

    _write(DATASET / "meta.csv", existing + meta_rows, REQUIRED_COLUMNS)

    print(f"promoted {copied} image(s) into {dest}")
    print(f"dataset/meta.csv: {len(existing)} -> {len(existing) + len(meta_rows)} rows")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source_id")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--accept", action="store_true", help="stage the PASS images for promotion")
    group.add_argument("--promote", action="store_true", help="copy accepted images into dataset/")
    args = parser.parse_args(argv)
    return accept(args.source_id) if args.accept else promote(args.source_id)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
