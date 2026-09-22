"""Dataset V1: the frozen, reproducible training set and its splits.

A freeze answers one question later: **exactly which data was Dataset V1?**
It does that with hashes and counts, not by copying 12,013 image files. The
manifest records the metadata digest, a per-file digest list, the generator
version and production seal, the real-data provenance records, the git commit
and the split seed, so a future session can prove membership byte for byte.

Two split populations, deliberately different:

**Synthetic** is split here, deterministically and stratified by
``(plate_type, difficulty)``. Every synthetic image carries a unique plate
number -- 12,000 images, 12,000 distinct strings -- so plate identity cannot
leak across a split no matter how the images are divided. The split is a pure
function of ``sha256("<seed>:<image>")``, so it reproduces anywhere.

**Real** is *not* split here. It was already frozen, group-aware, by
``src/real_splits.py`` into ``dataset/splits/real_splits.csv`` and must not be
recomputed: a group that has been written down keeps its split forever, which
is the only thing that keeps a holdout honest. This module reads that manifest
and audits it; it never reassigns it.

See ``docs/baseline_v1.md`` for how each split is used, and
``dataset/splits/README.md`` for the group policy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: The dataset version this module freezes.
DATASET_VERSION: str = "v1"

#: Seed for the synthetic split. Written down because the split is only
#: reproducible if the seed is; changing it silently re-splits the dataset.
SPLIT_SEED: int = 2026092201

#: Share of synthetic images per split, within every stratum.
SYNTHETIC_WEIGHTS: dict[str, float] = {"train": 0.85, "val": 0.10, "test": 0.05}

#: Split names used for synthetic data.
SYNTHETIC_SPLITS: tuple[str, ...] = ("train", "val", "test")

#: Columns of ``dataset/splits/synthetic_splits.csv``.
SYNTHETIC_MANIFEST_COLUMNS: tuple[str, ...] = (
    "image",
    "split",
    "plate_type",
    "difficulty",
    "stratum",
)


class DatasetV1Error(Exception):
    """A freeze or split that cannot be produced or trusted."""


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def image_score(image: str, seed: int) -> float:
    """A stable number in [0, 1) for one image: SHA-256 of ``seed:image``.

    Deliberately the same construction as ``real_splits.group_score``: no
    Python ``hash()``, no ordering dependence, no platform dependence.
    """
    digest = hashlib.sha256(f"{seed}:{image}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _check_weights(weights: Mapping[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(SYNTHETIC_SPLITS)
    if unknown:
        raise DatasetV1Error(f"unknown split(s): {', '.join(sorted(unknown))}")
    if any(w < 0 for w in weights.values()):
        raise DatasetV1Error("split weights cannot be negative")
    total = sum(weights.values())
    if total <= 0:
        raise DatasetV1Error("split weights sum to zero; nothing could be assigned")
    return {name: weights.get(name, 0.0) / total for name in SYNTHETIC_SPLITS}


# ---------------------------------------------------------------------------
# the synthetic split
# ---------------------------------------------------------------------------


def assign_synthetic(
    items: Iterable[tuple[str, str, str]],
    *,
    seed: int = SPLIT_SEED,
    weights: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Split synthetic images deterministically, stratified by class and difficulty.

    ``items`` are ``(image, plate_type, difficulty)``. Every stratum is cut at
    the same proportions, so rare combinations (a *hard* ``type1b``, say) keep
    their share in train, val and test instead of landing wherever chance puts
    them.

    Within a stratum the order is by :func:`image_score`, and the cuts are made
    by position. That makes the result a pure function of the seed and the set
    of images -- no RNG state, no insertion order.
    """
    share = _check_weights(weights or SYNTHETIC_WEIGHTS)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: set[str] = set()
    for image, plate_type, difficulty in items:
        if image in seen:
            raise DatasetV1Error(f"{image} appears twice in the synthetic split input")
        seen.add(image)
        strata[(plate_type, difficulty)].append(image)

    assignment: dict[str, str] = {}
    for key, images in strata.items():
        ordered = sorted(images, key=lambda name: (image_score(name, seed), name))
        n = len(ordered)
        # Cut by cumulative proportion, so every stratum is divided the same
        # way and the rounding error is at most one image per split per stratum.
        n_train = int(round(n * share["train"]))
        n_val = int(round(n * (share["train"] + share["val"]))) - n_train
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for index, name in enumerate(ordered):
            if index < n_train:
                assignment[name] = "train"
            elif index < n_train + n_val:
                assignment[name] = "val"
            else:
                assignment[name] = "test"
    return assignment


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@dataclass
class SplitAudit:
    """What the split actually contains, and anything wrong with it."""

    counts: dict = field(default_factory=dict)
    blocking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict:
        return {"counts": self.counts, "blocking": self.blocking,
                "notes": self.notes, "clean": self.clean}


def audit_splits(
    synthetic: Mapping[str, str],
    real_rows: Sequence[Mapping[str, str]],
    meta_by_image: Mapping[str, Mapping[str, str]],
    *,
    difficulty_by_image: Mapping[str, str] | None = None,
    hashes: Mapping[str, str] | None = None,
) -> SplitAudit:
    """Check the whole Dataset V1 split for leakage and for coverage.

    Blocking findings are the ones that would make a measured number a lie:
    an image in two splits, a plate string spanning splits, the same bytes in
    two splits, a real group spanning splits, or an image in the dataset that
    no split accounts for.
    """
    audit = SplitAudit()
    difficulty_by_image = difficulty_by_image or {}

    real_split = {row["image"]: row["split"] for row in real_rows}
    real_group = {row["image"]: row["group"] for row in real_rows}

    overlap = set(synthetic) & set(real_split)
    if overlap:
        audit.blocking.append(
            f"{len(overlap)} image(s) are in both the synthetic and the real split: "
            f"{sorted(overlap)[:3]}"
        )

    combined: dict[str, str] = {}
    for image, split in synthetic.items():
        combined[image] = f"synthetic:{split}"
    for image, split in real_split.items():
        combined[image] = f"real:{split}"

    # 1. every dataset image accounted for, exactly once
    unassigned = sorted(set(meta_by_image) - set(combined))
    if unassigned:
        audit.blocking.append(
            f"{len(unassigned)} dataset image(s) belong to no split: {unassigned[:3]}"
        )
    stale = sorted(set(combined) - set(meta_by_image))
    if stale:
        audit.blocking.append(
            f"{len(stale)} split entr(ies) name an image not in the dataset: {stale[:3]}"
        )

    # 2. plate identity must not span splits
    splits_by_plate: dict[str, set[str]] = defaultdict(set)
    for image, split in combined.items():
        row = meta_by_image.get(image)
        if not row:
            continue
        plate = (row.get("plate_num") or "").strip()
        if plate:
            splits_by_plate[plate].add(split)
    spanning = sorted(p for p, s in splits_by_plate.items() if len(s) > 1)
    if spanning:
        audit.blocking.append(
            f"{len(spanning)} plate number(s) appear in more than one split: {spanning[:5]}"
        )

    # 3. identical bytes must not span splits
    if hashes:
        splits_by_hash: dict[str, set[str]] = defaultdict(set)
        for image, split in combined.items():
            digest = hashes.get(image)
            if digest:
                splits_by_hash[digest].add(split)
        dup = [d for d, s in splits_by_hash.items() if len(s) > 1]
        if dup:
            audit.blocking.append(
                f"{len(dup)} identical image(s) appear in more than one split"
            )

    # 4. a real group must not span splits
    groups: dict[str, set[str]] = defaultdict(set)
    for image, group in real_group.items():
        groups[group].add(real_split[image])
    group_span = sorted(g for g, s in groups.items() if len(s) > 1)
    if group_span:
        audit.blocking.append(
            f"{len(group_span)} real group(s) span splits: {group_span[:3]}"
        )

    # ---- counts, for the human-readable report
    synth_counts: dict[str, Counter] = {s: Counter() for s in SYNTHETIC_SPLITS}
    synth_difficulty: dict[str, Counter] = {s: Counter() for s in SYNTHETIC_SPLITS}
    synth_conditions: dict[str, Counter] = {s: Counter() for s in SYNTHETIC_SPLITS}
    for image, split in synthetic.items():
        row = meta_by_image.get(image, {})
        synth_counts[split][row.get("plate_type", "?")] += 1
        synth_difficulty[split][difficulty_by_image.get(image, "?")] += 1
        for tag in (row.get("conditions") or "").split("|"):
            if tag.strip():
                synth_conditions[split][tag.strip()] += 1

    real_counts: dict[str, Counter] = defaultdict(Counter)
    for image, split in real_split.items():
        real_counts[split][meta_by_image.get(image, {}).get("plate_type", "?")] += 1

    audit.counts = {
        "total_images": len(meta_by_image),
        "synthetic": {
            "total": len(synthetic),
            "by_split": {s: sum(synth_counts[s].values()) for s in SYNTHETIC_SPLITS},
            "by_split_class": {s: dict(synth_counts[s]) for s in SYNTHETIC_SPLITS},
            "by_split_difficulty": {s: dict(synth_difficulty[s]) for s in SYNTHETIC_SPLITS},
            "by_split_conditions": {s: dict(synth_conditions[s]) for s in SYNTHETIC_SPLITS},
        },
        "real": {
            "total": len(real_split),
            "by_split": {s: sum(c.values()) for s, c in sorted(real_counts.items())},
            "by_split_class": {s: dict(c) for s, c in sorted(real_counts.items())},
            "groups": len(groups),
        },
    }

    if audit.counts["real"]["by_split"].get("holdout", 0) < 30:
        audit.notes.append(
            f"the real holdout holds {audit.counts['real']['by_split'].get('holdout', 0)} "
            "image(s): far too few for a stable metric. Report it as an indication with "
            "an explicit interval, never as an accuracy figure"
        )
    return audit


def audit_text(audit: SplitAudit) -> str:
    """The audit as a report a person reads."""
    counts = audit.counts
    lines = [
        f"Dataset {DATASET_VERSION} split audit",
        f"  seed              : {SPLIT_SEED}",
        f"  images            : {counts.get('total_images', 0)}",
    ]
    synth = counts.get("synthetic", {})
    lines.append(f"  synthetic         : {synth.get('total', 0)}  {synth.get('by_split', {})}")
    for split in SYNTHETIC_SPLITS:
        cls = synth.get("by_split_class", {}).get(split, {})
        dif = synth.get("by_split_difficulty", {}).get(split, {})
        lines.append(f"    {split:<5} class={cls} difficulty={dif}")
    real = counts.get("real", {})
    lines.append(
        f"  real              : {real.get('total', 0)}  {real.get('by_split', {})}"
        f"  groups={real.get('groups', 0)}"
    )
    for split, cls in real.get("by_split_class", {}).items():
        lines.append(f"    {split:<8} class={cls}")

    lines.append("")
    lines.append(f"Blocking ({len(audit.blocking)}):")
    lines += [f"  - {p}" for p in audit.blocking] or ["  (none)"]
    lines.append(f"Notes ({len(audit.notes)}):")
    lines += [f"  - {n}" for n in audit.notes] or ["  (none)"]
    lines.append("")
    lines.append(f"Verdict: {'CLEAN' if audit.clean else 'LEAKAGE OR GAPS FOUND'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the freeze
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(entries: Mapping[str, str]) -> str:
    """One digest over a whole file set: SHA-256 of sorted ``path sha`` lines.

    The same construction the synthetic production seal uses, so the two can be
    compared by eye and by script.
    """
    body = "\n".join(f"{path} {entries[path]}" for path in sorted(entries))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


__all__ = [
    "DATASET_VERSION",
    "DatasetV1Error",
    "SPLIT_SEED",
    "SYNTHETIC_MANIFEST_COLUMNS",
    "SYNTHETIC_SPLITS",
    "SYNTHETIC_WEIGHTS",
    "SplitAudit",
    "assign_synthetic",
    "audit_splits",
    "audit_text",
    "combined_digest",
    "image_score",
    "sha256_file",
]
