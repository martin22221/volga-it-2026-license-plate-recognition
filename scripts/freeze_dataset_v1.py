#!/usr/bin/env python
"""Freeze Dataset V1: split the synthetic set, then record what V1 *is*.

Three outputs, none of which copies an image:

``dataset/splits/synthetic_splits.csv``  deterministic train/val/test for the
    12,000 synthetic images, stratified by ``(plate_type, difficulty)``.

``dataset/splits/dataset_v1.json``  the freeze itself: counts, class counts,
    the metadata digest, a per-file digest for every image, the combined
    digest, the generator version and production seal, the real-data
    provenance records, the git commit and the split seed.

``data/audits/dataset_v1_split_audit.{json,txt}``  the leakage audit.

The real split is **read, never recomputed**: it is frozen group-aware in
``dataset/splits/real_splits.csv`` and moving a group after the fact is what
quietly destroys a holdout.

Usage::

    python scripts/freeze_dataset_v1.py            # write the freeze
    python scripts/freeze_dataset_v1.py --check    # verify, write nothing

Exit codes::

    0  the freeze is written (or verifies)
    1  the dataset could not be read
    2  the audit found something blocking, or --check found a mismatch
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_meta import CSV_DELIMITER  # noqa: E402
from src.dataset_v1 import (  # noqa: E402
    DATASET_VERSION,
    SPLIT_SEED,
    SYNTHETIC_MANIFEST_COLUMNS,
    SYNTHETIC_WEIGHTS,
    assign_synthetic,
    audit_splits,
    audit_text,
    combined_digest,
    sha256_file,
)

DATASET = REPO_ROOT / "dataset"
SPLITS = DATASET / "splits"
AUDITS = REPO_ROOT / "data" / "audits"
PRODUCTION = REPO_ROOT / "data" / "synthetic_production" / "v2_3_seed2026091401_n12000"

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKED = 0, 1, 2


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=CSV_DELIMITER))


def _difficulty_by_image() -> dict[str, str]:
    """Per-image difficulty, from the generator's own record.

    ``dataset/meta.csv`` does not carry difficulty -- it is a generation
    property, not an annotation -- so it is read from ``generation.jsonl`` in
    the sealed production batch.
    """
    path = PRODUCTION / "generation.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[record["image"]] = record.get("difficulty", "?")
    return out


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except OSError:  # pragma: no cover - git absent
        return ""


def build(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true",
                        help="verify the committed freeze instead of writing it")
    args = parser.parse_args(argv)

    meta_rows = _read_csv(DATASET / "meta.csv")
    if not meta_rows:
        print(f"error: cannot read {DATASET / 'meta.csv'}", file=sys.stderr)
        return EXIT_UNUSABLE

    meta_by_image = {row["image"]: row for row in meta_rows}
    synthetic_rows = [r for r in meta_rows if r["is_synthetic"].lower() == "true"]
    real_rows_meta = [r for r in meta_rows if r["is_synthetic"].lower() != "true"]
    difficulty = _difficulty_by_image()

    missing_difficulty = [r["image"] for r in synthetic_rows if r["image"] not in difficulty]
    if missing_difficulty:
        print(f"error: {len(missing_difficulty)} synthetic image(s) have no difficulty "
              f"in generation.jsonl, e.g. {missing_difficulty[:3]}", file=sys.stderr)
        return EXIT_UNUSABLE

    assignment = assign_synthetic(
        ((r["image"], r["plate_type"], difficulty[r["image"]]) for r in synthetic_rows),
        seed=SPLIT_SEED,
    )

    real_split_rows = _read_csv(SPLITS / "real_splits.csv")

    # ---- hashes: one per image file actually present
    hashes: dict[str, str] = {}
    for image in sorted(meta_by_image):
        path = DATASET / image
        if path.is_file():
            hashes[image] = sha256_file(path)

    audit = audit_splits(
        assignment, real_split_rows, meta_by_image,
        difficulty_by_image=difficulty, hashes=hashes,
    )
    report = audit_text(audit)
    print(report)

    if args.check:
        frozen_path = SPLITS / "dataset_v1.json"
        if not frozen_path.is_file():
            print(f"\nerror: no freeze at {frozen_path}", file=sys.stderr)
            return EXIT_BLOCKED
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        problems = []
        if frozen.get("combined_image_digest") != combined_digest(hashes):
            problems.append("the image set no longer matches the frozen combined digest")
        if frozen.get("meta_sha256") != sha256_file(DATASET / "meta.csv"):
            problems.append("dataset/meta.csv no longer matches the frozen digest")
        if frozen.get("split_seed") != SPLIT_SEED:
            problems.append(f"split seed changed: frozen {frozen.get('split_seed')}, now {SPLIT_SEED}")
        committed = {r["image"]: r["split"] for r in _read_csv(SPLITS / "synthetic_splits.csv")}
        if committed != assignment:
            problems.append("the synthetic split does not reproduce from the seed")
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(f"\nfreeze check: {'OK' if not problems else 'MISMATCH'}")
        return EXIT_BLOCKED if (problems or audit.blocking) else EXIT_OK

    if audit.blocking:
        print("\nrefusing to freeze: the split audit has blocking findings", file=sys.stderr)
        return EXIT_BLOCKED

    # ---- write the synthetic split manifest
    SPLITS.mkdir(parents=True, exist_ok=True)
    with (SPLITS / "synthetic_splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SYNTHETIC_MANIFEST_COLUMNS),
                                delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writeheader()
        for image in sorted(assignment):
            row = meta_by_image[image]
            writer.writerow({
                "image": image,
                "split": assignment[image],
                "plate_type": row["plate_type"],
                "difficulty": difficulty[image],
                "stratum": f"{row['plate_type']}/{difficulty[image]}",
            })

    # ---- write the freeze
    seal = json.loads((PRODUCTION / "production_seal.json").read_text(encoding="utf-8"))
    counts = audit.counts
    freeze = {
        "dataset_version": DATASET_VERSION,
        "created_on": date.today().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Only the freeze's *inputs* can meaningfully be clean: the freeze
        # writes its own outputs into dataset/splits/, so a whole-tree check
        # would always report dirty and would say nothing about the data.
        "dataset_inputs_clean": _git(
            "status", "--porcelain", "--",
            "dataset/meta.csv", "dataset/LICENSE", "dataset/images",
            "dataset/splits/real_splits.csv",
        ) == "",
        "split_seed": SPLIT_SEED,
        "synthetic_split_weights": SYNTHETIC_WEIGHTS,
        "counts": {
            "images_total": len(meta_by_image),
            "rows_total": len(meta_rows),
            "synthetic_images": len(synthetic_rows),
            "real_images": len({r["image"] for r in real_rows_meta}),
            "real_rows": len(real_rows_meta),
        },
        "class_counts": {
            "all": dict(sorted(_count(meta_rows, "plate_type").items())),
            "synthetic": dict(sorted(_count(synthetic_rows, "plate_type").items())),
            "real": dict(sorted(_count(real_rows_meta, "plate_type").items())),
        },
        "splits": {
            "synthetic": counts["synthetic"]["by_split"],
            "synthetic_by_class": counts["synthetic"]["by_split_class"],
            "synthetic_by_difficulty": counts["synthetic"]["by_split_difficulty"],
            "real": counts["real"]["by_split"],
            "real_by_class": counts["real"]["by_split_class"],
            "real_groups": counts["real"]["groups"],
        },
        "meta_sha256": sha256_file(DATASET / "meta.csv"),
        "real_splits_sha256": sha256_file(SPLITS / "real_splits.csv"),
        "synthetic_splits_sha256": sha256_file(SPLITS / "synthetic_splits.csv"),
        "license_sha256": sha256_file(DATASET / "LICENSE"),
        "image_count_hashed": len(hashes),
        "combined_image_digest": combined_digest(hashes),
        "generator": {
            "name": "volga-synthetic-plate-generator",
            "version": seal.get("generator_version"),
            "seed": seal.get("seed"),
            "batch": PRODUCTION.name,
            "production_seal_sha256": sha256_file(PRODUCTION / "production_seal.json"),
            "seal_combined_digest_all_files": seal.get("combined_digest_all_files"),
            "seal_files": seal.get("files_sealed"),
        },
        "real_provenance": {
            "sources": sorted({r["source"] for r in real_rows_meta}),
            "source_records": sorted(
                str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                for p in (REPO_ROOT / "data" / "real_staging" / "source_records").glob("*.json")
            ),
            "acquisition_records": sorted(
                str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                for p in (REPO_ROOT / "data" / "real_staging" / "incoming").glob("*/acquisition_record.csv")
            ),
        },
        "how_to_verify": (
            "python scripts/freeze_dataset_v1.py --check  -- recomputes the synthetic split "
            "from split_seed, re-hashes every image and dataset/meta.csv, and compares with "
            "combined_image_digest and meta_sha256."
        ),
    }
    (SPLITS / "dataset_v1.json").write_text(
        json.dumps(freeze, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ---- per-image digests, kept beside the freeze rather than inside it
    with (SPLITS / "dataset_v1_files.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(["image", "sha256"])
        for image in sorted(hashes):
            writer.writerow([image, hashes[image]])

    AUDITS.mkdir(parents=True, exist_ok=True)
    (AUDITS / "dataset_v1_split_audit.txt").write_text(report + "\n", encoding="utf-8")
    (AUDITS / "dataset_v1_split_audit.json").write_text(
        json.dumps(audit.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nwritten:")
    for path in (SPLITS / "synthetic_splits.csv", SPLITS / "dataset_v1.json",
                 SPLITS / "dataset_v1_files.csv",
                 AUDITS / "dataset_v1_split_audit.txt", AUDITS / "dataset_v1_split_audit.json"):
        print(f"  {path.relative_to(REPO_ROOT)}")
    return EXIT_OK


def _count(rows, field: str) -> dict:
    from collections import Counter
    return dict(Counter(r[field] for r in rows))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(build())
