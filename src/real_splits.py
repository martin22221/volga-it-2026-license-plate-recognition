"""Deterministic, group-aware train / val / holdout splits for real images.

The split is decided by hashing a sample's **group**, never the file name, so
every image that shares a group lands in the same split.  A group is whatever
would leak if it were split: one capture session or burst, one video, one
vehicle, one plate.  Photographs of the same car from six angles are one group.

Two properties matter more than the exact ratios:

* **Deterministic.**  The same groups, weights and seed always give the same
  assignment, on any machine, in any order.
* **Frozen.**  Once a group is written into the manifest, its split never
  changes.  New groups are assigned around the existing ones.  This is what
  protects the holdout: an image cannot drift into training because the
  dataset grew.

The manifest (``dataset/splits/real_splits.csv``) is the record; this module
computes and checks it.  See ``docs/real_data_plan.md``.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SPLITS: tuple[str, ...] = ("train", "val", "holdout")
#: Default share of *groups* per split.  Real images are scarce, so the
#: holdout is sized to stay meaningful rather than to look generous.
DEFAULT_WEIGHTS: dict[str, float] = {"train": 0.70, "val": 0.15, "holdout": 0.15}
#: Columns of ``dataset/splits/real_splits.csv``.
MANIFEST_COLUMNS: tuple[str, ...] = ("image", "group", "split", "source", "assigned_on", "note")
#: The split that must never be trained or tuned on.
HOLDOUT: str = "holdout"


class SplitError(Exception):
    """A split manifest or request that cannot be honoured."""


@dataclass(frozen=True)
class SplitRow:
    """One image's frozen split assignment."""

    image: str
    group: str
    split: str
    source: str = ""
    assigned_on: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "image": self.image,
            "group": self.group,
            "split": self.split,
            "source": self.source,
            "assigned_on": self.assigned_on,
            "note": self.note,
        }


def group_score(group: str, seed: int) -> float:
    """A stable number in [0, 1) for one group: SHA-256 of ``seed:group``."""
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_groups(
    groups: Iterable[str],
    *,
    seed: int,
    weights: Mapping[str, float] | None = None,
    frozen: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each group to a split; groups in ``frozen`` keep their split.

    Deterministic in the groups, the weights and the seed, and independent of
    the order they arrive in.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    unknown = sorted(set(weights) - set(SPLITS))
    if unknown:
        raise SplitError(f"unknown split(s) in weights: {', '.join(unknown)}")
    total = sum(weights.values())
    if total <= 0 or any(value < 0 for value in weights.values()):
        raise SplitError(f"weights must be non-negative and sum above zero: {weights}")
    frozen = dict(frozen or {})
    for group, split in frozen.items():
        if split not in SPLITS:
            raise SplitError(f"frozen group {group!r} has unknown split {split!r}")

    edges: list[tuple[str, float]] = []
    running = 0.0
    for split in SPLITS:
        running += weights.get(split, 0.0) / total
        edges.append((split, running))

    assignment: dict[str, str] = {}
    for group in sorted(set(groups)):
        if group in frozen:
            assignment[group] = frozen[group]
            continue
        score = group_score(group, seed)
        for split, edge in edges:
            if score < edge or edge >= 1.0:
                assignment[group] = split
                break
    return assignment


def assign_rows(
    items: Sequence[tuple[str, str, str]],
    *,
    seed: int,
    weights: Mapping[str, float] | None = None,
    frozen_rows: Sequence[SplitRow] = (),
    assigned_on: str = "",
) -> list[SplitRow]:
    """Assign ``(image, group, source)`` triples, honouring a frozen manifest.

    An image already in ``frozen_rows`` keeps its row untouched, and its group
    keeps its split for every new image of that group.
    """
    frozen_by_image = {row.image: row for row in frozen_rows}
    frozen_by_group = {row.group: row.split for row in frozen_rows}
    for image, group, _source in items:
        row = frozen_by_image.get(image)
        if row and row.group != group:
            raise SplitError(
                f"{image} is already assigned in group {row.group!r}; it cannot move to {group!r} "
                "without breaking the frozen split"
            )
    assignment = assign_groups((group for _image, group, _source in items), seed=seed, weights=weights, frozen=frozen_by_group)
    rows: list[SplitRow] = []
    for image, group, source in sorted(items):
        if image in frozen_by_image:
            rows.append(frozen_by_image[image])
            continue
        rows.append(SplitRow(image=image, group=group, split=assignment[group], source=source, assigned_on=assigned_on))
    for row in frozen_rows:  # keep assignments whose image is no longer staged
        if row.image not in {image for image, _g, _s in items}:
            rows.append(row)
    return sorted(rows, key=lambda row: row.image)


def read_manifest(path: Path) -> list[SplitRow]:
    """Read a split manifest; an absent file is an empty manifest."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SplitError(f"{path}: missing column(s) {', '.join(missing)}")
        rows = []
        for number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            if not image:
                continue
            split = (row.get("split") or "").strip()
            if split not in SPLITS:
                raise SplitError(f"{path} line {number}: split {split!r} is not one of {', '.join(SPLITS)}")
            group = (row.get("group") or "").strip()
            if not group:
                raise SplitError(f"{path} line {number}: empty group")
            rows.append(SplitRow(image=image, group=group, split=split, source=(row.get("source") or "").strip(),
                                 assigned_on=(row.get("assigned_on") or "").strip(), note=(row.get("note") or "").strip()))
    return rows


def write_manifest(path: Path, rows: Sequence[SplitRow]) -> None:
    """Write a split manifest, sorted by image so diffs stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), delimiter=";", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda row: row.image):
            writer.writerow(row.as_dict())


def leakage_report(
    rows: Sequence[SplitRow],
    *,
    hashes: Mapping[str, str] | None = None,
    near_duplicate_pairs: Sequence[tuple[str, str, float]] = (),
) -> dict:
    """Everything that could leak between splits.

    ``hashes`` maps image path to its SHA-256, which catches the same picture
    stored under two names.  ``near_duplicate_pairs`` comes from the intake
    audit.
    """
    by_image: dict[str, SplitRow] = {}
    duplicate_images: list[str] = []
    for row in rows:
        if row.image in by_image:
            duplicate_images.append(row.image)
        by_image[row.image] = row

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row.group].add(row.split)
    spanning = sorted(group for group, splits in group_splits.items() if len(splits) > 1)

    identical_across: list[tuple[str, str, str, str]] = []
    if hashes:
        by_hash: dict[str, list[str]] = defaultdict(list)
        for image, digest in hashes.items():
            if image in by_image:
                by_hash[digest].append(image)
        for images in by_hash.values():
            for i, first in enumerate(sorted(images)):
                for second in sorted(images)[i + 1 :]:
                    if by_image[first].split != by_image[second].split:
                        identical_across.append((first, by_image[first].split, second, by_image[second].split))

    near_across = [
        (first, by_image[first].split, second, by_image[second].split, score)
        for first, second, score in near_duplicate_pairs
        if first in by_image and second in by_image and by_image[first].split != by_image[second].split
    ]

    counts = Counter(row.split for row in rows)
    groups = Counter()
    for group, splits in group_splits.items():
        groups[next(iter(splits))] += 1 if len(splits) == 1 else 0
    findings = {
        "images": len(by_image),
        "by_split": {split: counts.get(split, 0) for split in SPLITS},
        "groups": len(group_splits),
        "groups_by_split": {split: groups.get(split, 0) for split in SPLITS},
        "duplicate_manifest_rows": sorted(set(duplicate_images)),
        "groups_spanning_splits": spanning,
        "identical_images_across_splits": identical_across,
        "near_duplicates_across_splits": near_across,
    }
    findings["clean"] = not (spanning or identical_across or near_across or duplicate_images)
    return findings


def holdout_images(rows: Sequence[SplitRow]) -> set[str]:
    """The images that must never be trained or tuned on."""
    return {row.image for row in rows if row.split == HOLDOUT}


def leakage_text(findings: Mapping[str, object]) -> str:
    lines = [
        f"images {findings['images']} in {findings['groups']} group(s): {findings['by_split']}",
        f"groups per split: {findings['groups_by_split']}",
        f"duplicate manifest rows      : {len(findings['duplicate_manifest_rows'])}",
        f"groups spanning splits       : {len(findings['groups_spanning_splits'])}",
        f"identical images across split: {len(findings['identical_images_across_splits'])}",
        f"near-duplicates across splits: {len(findings['near_duplicates_across_splits'])}",
        f"verdict: {'CLEAN' if findings['clean'] else 'LEAKAGE FOUND'}",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_WEIGHTS",
    "HOLDOUT",
    "MANIFEST_COLUMNS",
    "SPLITS",
    "SplitError",
    "SplitRow",
    "assign_groups",
    "assign_rows",
    "group_score",
    "holdout_images",
    "leakage_report",
    "leakage_text",
    "read_manifest",
    "write_manifest",
]
