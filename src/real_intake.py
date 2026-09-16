"""Intake of real photographs: source records and staged-source audits.

Real images are staged under ``data/real_staging/`` and inspected there.  This
module reads a staged source and reports what it finds; it decides nothing.

**The tooling never approves a source and never infers rights.**  A source's
decision is written by a person into its source record
(``data/real_staging/source_records/<source_id>.json``); the audit only checks
that the decision is *consistent with the rights recorded next to it*, and
reports every gap that blocks promotion into ``dataset/images/real/``.

See ``docs/real_data_intake.md`` for the workflow and
``docs/real_data_plan.md`` for the targets and the split policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.dataset_meta import (
    is_redistributable_license,
    license_incompatibility,
    read_meta,
    validate_meta,
)
from src.external_audit import ImageRecord, inspect_image

#: What a person may write in a source record's ``decision`` field.
DECISIONS: tuple[str, ...] = (
    "PENDING",  # inspected, not decided
    "ACCEPT_FOR_SUBMISSION",  # may enter dataset/ and the submitted dataset
    "TRAINING_ONLY_IF_LEGAL",  # may train a model, never redistributed
    "REFERENCE_ONLY",  # look at it, never train on it, never redistribute
    "REJECT",  # not used at all
)
#: Only this decision lets an image reach ``dataset/images/real/``.
SUBMITTABLE: str = "ACCEPT_FOR_SUBMISSION"

#: Every field a source record must carry, even when the answer is "unknown".
RECORD_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_name",
    "source_reference",
    "original_creator",
    "upstream_source",
    "stated_license",
    "license_evidence_url",
    "redistribution_allowed",
    "modification_allowed",
    "commercial_use_allowed",
    "attribution_required",
    "attribution_text",
    "provenance_evidence",
    "privacy_review",
    "decision",
    "decided_by",
    "date_checked",
    "notes",
)
#: Permission fields are three-valued on purpose: "unclear" is not "no".
PERMISSIONS: tuple[str, ...] = ("yes", "no", "unclear")
PERMISSION_FIELDS: tuple[str, ...] = (
    "redistribution_allowed",
    "modification_allowed",
    "commercial_use_allowed",
)
#: Privacy review states; ``completed`` means every image has been looked at.
PRIVACY_STATES: tuple[str, ...] = ("not_started", "in_progress", "completed", "not_required")
#: What a person may write in a privacy review's ``action`` column.
PRIVACY_ACTIONS: tuple[str, ...] = ("none_needed", "blurred", "covered", "rejected")

IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
#: Thumbnail correlation above which two images are reported as near-duplicates.
NEAR_DUPLICATE_THRESHOLD: float = 0.97


# --------------------------------------------------------------------------
# source records
# --------------------------------------------------------------------------


@dataclass
class SourceRecord:
    """A person's provenance findings about one candidate source."""

    path: Path
    data: dict
    problems: list[str] = field(default_factory=list)

    def get(self, field_name: str) -> str:
        value = self.data.get(field_name, "")
        return value.strip() if isinstance(value, str) else str(value)

    @property
    def source_id(self) -> str:
        return self.get("source_id")

    @property
    def decision(self) -> str:
        return self.get("decision").upper()

    @property
    def submittable(self) -> bool:
        """Whether the record *claims* the source may enter the submitted dataset.

        Only meaningful together with :attr:`problems`: a claim that the
        recorded rights do not support is listed there.
        """
        return self.decision == SUBMITTABLE and not self.problems


def blank_record(source_id: str = "") -> dict:
    """An empty source record for a person to fill in."""
    record = {name: "" for name in RECORD_FIELDS}
    record["source_id"] = source_id
    record["decision"] = "PENDING"
    record["privacy_review"] = "not_started"
    return record


def load_source_record(path: Path) -> SourceRecord:
    """Read one source record and check it for completeness and consistency."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return SourceRecord(path=Path(path), data={}, problems=[f"unreadable source record: {error}"])
    if not isinstance(data, dict):
        return SourceRecord(path=Path(path), data={}, problems=["source record is not a JSON object"])
    return SourceRecord(path=Path(path), data=data, problems=record_problems(data))


def record_problems(data: Mapping[str, object]) -> list[str]:
    """Everything wrong with a source record, worst first.

    A decision is never *made* here.  ``ACCEPT_FOR_SUBMISSION`` is only
    accepted when the rights recorded beside it support redistribution under
    CC BY 4.0, which is what the submitted dataset requires.
    """
    problems: list[str] = []
    get = lambda name: str(data.get(name, "") or "").strip()

    for name in RECORD_FIELDS:
        if name not in data:
            problems.append(f"missing field {name!r}")
    for name in ("source_id", "source_name", "source_reference", "date_checked", "decided_by"):
        if name in data and not get(name):
            problems.append(f"{name} is empty")

    for name in PERMISSION_FIELDS:
        value = get(name).lower()
        if value and value not in PERMISSIONS:
            problems.append(f"{name}={value!r} is not one of {', '.join(PERMISSIONS)}")
        elif not value:
            problems.append(f"{name} is empty; record 'unclear' rather than leaving it blank")

    decision = get("decision").upper()
    if decision not in DECISIONS:
        problems.append(f"decision={decision!r} is not one of {', '.join(DECISIONS)}")

    privacy = get("privacy_review").lower()
    if privacy and privacy not in PRIVACY_STATES:
        problems.append(f"privacy_review={privacy!r} is not one of {', '.join(PRIVACY_STATES)}")

    if decision == SUBMITTABLE:
        for name in PERMISSION_FIELDS:
            if get(name).lower() != "yes":
                problems.append(
                    f"decision {SUBMITTABLE} but {name}={get(name)!r}: the submitted dataset is "
                    "published under CC BY 4.0, so redistribution, modification and commercial use "
                    "must all be permitted"
                )
        licence = get("stated_license")
        if not licence:
            problems.append(f"decision {SUBMITTABLE} but stated_license is empty")
        else:
            clash = license_incompatibility(licence)
            if clash:
                problems.append(f"decision {SUBMITTABLE} but the license is incompatible: {clash}")
            elif not is_redistributable_license(licence):
                problems.append(
                    f"decision {SUBMITTABLE} but license {licence!r} is not recognised as redistributable; "
                    "a person must resolve it before submission"
                )
        if not get("provenance_evidence"):
            problems.append(
                f"decision {SUBMITTABLE} but provenance_evidence is empty: a platform's license label is "
                "not evidence that the platform owns the photographs"
            )
        if not get("license_evidence_url"):
            problems.append(f"decision {SUBMITTABLE} but license_evidence_url is empty")
        if get("attribution_required").lower() == "yes" and not get("attribution_text"):
            problems.append(f"decision {SUBMITTABLE} but attribution is required and attribution_text is empty")
        if privacy not in ("completed", "not_required"):
            problems.append(f"decision {SUBMITTABLE} but privacy_review={privacy or 'empty'!r}")
    if decision == "TRAINING_ONLY_IF_LEGAL" and get("redistribution_allowed").lower() == "yes":
        problems.append(
            "decision TRAINING_ONLY_IF_LEGAL but redistribution is allowed; "
            f"reconsider {SUBMITTABLE} rather than holding the images back"
        )
    return problems


# --------------------------------------------------------------------------
# staged source audit
# --------------------------------------------------------------------------


@dataclass
class StagedSourceAudit:
    """What one staged source directory contains, and what blocks promotion."""

    source_id: str
    root: str
    record: SourceRecord | None = None
    images: list[ImageRecord] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    blocking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def promotable(self) -> bool:
        """Never a decision: only "a person's ACCEPT is recorded and nothing blocks it"."""
        return bool(self.record and self.record.submittable and not self.blocking)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "root": self.root,
            "decision": self.record.decision if self.record else "NO_RECORD",
            "record_problems": list(self.record.problems) if self.record else ["no source record"],
            "summary": self.summary,
            "blocking": self.blocking,
            "notes": self.notes,
            "promotable": self.promotable,
            "images": [vars(image) for image in self.images],
        }


def _thumbnail(path: Path, size: tuple[int, int] = (32, 24)) -> list[float] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            grey = image.convert("L").resize(size)
        values = [float(v) for v in grey.getdata()]
    except Exception:  # noqa: BLE001 - a corrupt file is reported elsewhere
        return None
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values) or 1.0
    return [(v - mean) / spread for v in values]


def near_duplicate_pairs(
    thumbs: Mapping[str, Sequence[float]], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[tuple[str, str, float]]:
    """Pairs whose normalised thumbnails correlate above ``threshold``."""
    names = sorted(thumbs)
    pairs: list[tuple[str, str, float]] = []
    for i, first in enumerate(names):
        a = thumbs[first]
        for second in names[i + 1 :]:
            b = thumbs[second]
            if len(a) != len(b):
                continue
            score = sum(x * y for x, y in zip(a, b)) / len(a)
            if score >= threshold:
                pairs.append((first, second, round(score, 4)))
    return pairs


def _annotation_state(meta_path: Path | None, root: Path, images: Sequence[str]) -> dict:
    """Annotation coverage and counts for a staged source."""
    state: dict = {
        "meta_csv": str(meta_path) if meta_path else None,
        "rows": 0,
        "annotated_images": 0,
        "images_without_annotation": [],
        "rows_without_image": [],
        "by_plate_type": {},
        "unique_plate_numbers": 0,
        "plate_numbers_with_hash": 0,
        "background_rows": 0,
        "validator_errors": [],
        "validator_warnings": [],
    }
    if meta_path is None or not Path(meta_path).exists():
        state["images_without_annotation"] = sorted(images)
        return state

    report = validate_meta(Path(meta_path), root, check_files=True)
    state["validator_errors"] = [issue.message for issue in report.issues if issue.severity.value == "error"]
    state["validator_warnings"] = [issue.message for issue in report.issues if issue.severity.value == "warning"]
    _, rows = read_meta(Path(meta_path))
    state["rows"] = len(rows)
    annotated = {row.image for row in rows}
    state["annotated_images"] = len(annotated)
    state["images_without_annotation"] = sorted(set(images) - annotated)
    state["rows_without_image"] = sorted(annotated - set(images))
    state["by_plate_type"] = dict(Counter(row.plate_type for row in rows))
    plates = {row.plate_num for row in rows if row.plate_num}
    state["unique_plate_numbers"] = len({p for p in plates if "#" not in p})
    state["plate_numbers_with_hash"] = len({p for p in plates if "#" in p})
    state["background_rows"] = sum(1 for row in rows if row.is_background())
    state["real_rows"] = sum(1 for row in rows if row.is_synthetic is False)
    state["synthetic_rows"] = sum(1 for row in rows if row.is_synthetic is True)
    return state


def _privacy_state(path: Path | None, images: Sequence[str]) -> dict:
    """Per-image privacy review coverage, read from ``privacy_review.csv``."""
    state: dict = {
        "file": str(path) if path else None,
        "reviewed": 0,
        "unreviewed": sorted(images),
        "actions": {},
        "unresolved": [],
        "rejected": [],
        "bad_rows": [],
    }
    if path is None or not Path(path).exists():
        return state
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    seen: set[str] = set()
    actions: Counter[str] = Counter()
    for number, row in enumerate(rows, start=2):
        image = (row.get("image") or "").strip().replace("\\", "/")
        action = (row.get("action") or "").strip().lower()
        faces = (row.get("faces_present") or "").strip().lower()
        if not image or action not in PRIVACY_ACTIONS:
            state["bad_rows"].append(f"line {number}: image={image!r} action={action!r}")
            continue
        seen.add(image)
        actions[action] += 1
        if action == "rejected":
            state["rejected"].append(image)
        elif faces in ("yes", "true", "1") and action == "none_needed":
            state["unresolved"].append(image)
    state["reviewed"] = len(seen)
    state["unreviewed"] = sorted(set(images) - seen)
    state["actions"] = dict(actions)
    return state


def audit_staged_source(
    directory: Path,
    *,
    record_path: Path | None = None,
    meta_path: Path | None = None,
    privacy_path: Path | None = None,
    split_assignment: Mapping[str, str] | None = None,
    group_of: Mapping[str, str] | None = None,
    known_hashes: Mapping[str, str] | None = None,
    near_duplicates: bool = True,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> StagedSourceAudit:
    """Inspect one staged source directory.  Reports; never approves, never writes.

    ``known_hashes`` maps a SHA-256 to a description of an image already in the
    dataset, so a restaged copy of an accepted image is reported rather than
    silently added twice.  ``split_assignment`` and ``group_of`` let the audit
    check the staged images against the frozen split manifest.
    """
    root = Path(directory)
    audit = StagedSourceAudit(source_id=root.name, root=str(root))
    if not root.is_dir():
        audit.blocking.append(f"{root} is not a directory")
        return audit

    record_path = record_path or (root / "source_record.json")
    audit.record = load_source_record(record_path) if Path(record_path).exists() else None
    if audit.record is None:
        audit.blocking.append(
            f"no source record at {record_path}: register the source before staging its images"
        )
    else:
        audit.blocking += [f"source record: {problem}" for problem in audit.record.problems]
        if audit.record.source_id and audit.record.source_id != root.name:
            audit.notes.append(f"source record id {audit.record.source_id!r} differs from the folder name {root.name!r}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    audit.images = [inspect_image(path, root) for path in files]
    names = [image.path for image in audit.images]

    by_hash: dict[str, list[str]] = {}
    for image in audit.images:
        if image.sha256:
            by_hash.setdefault(image.sha256, []).append(image.path)
    exact_duplicates = [group for group in by_hash.values() if len(group) > 1]
    already_present = sorted(
        (image.path, known_hashes[image.sha256])
        for image in audit.images
        if image.sha256 and known_hashes and image.sha256 in known_hashes
    )

    pairs: list[tuple[str, str, float]] = []
    if near_duplicates and len(audit.images) > 1:
        thumbs = {}
        for path, image in zip(files, audit.images):
            thumb = _thumbnail(path)
            if thumb is not None:
                thumbs[image.path] = thumb
        pairs = near_duplicate_pairs(thumbs, threshold)

    widths = [i.width for i in audit.images if i.width]
    heights = [i.height for i in audit.images if i.height]
    corrupt = [i.path for i in audit.images if i.is_corrupt]
    annotations = _annotation_state(meta_path or (root / "meta.csv"), root, names)
    privacy = _privacy_state(privacy_path or (root / "privacy_review.csv"), names)

    groups = {name: (group_of or {}).get(name) for name in names}
    assigned = {name: (split_assignment or {}).get(name) for name in names}
    split_counts = Counter(split for split in assigned.values() if split)
    group_spans: list[str] = []
    if split_assignment and group_of:
        per_group: dict[str, set[str]] = {}
        for name, group in group_of.items():
            split = split_assignment.get(name)
            if group and split:
                per_group.setdefault(group, set()).add(split)
        group_spans = sorted(group for group, splits in per_group.items() if len(splits) > 1)

    audit.summary = {
        "images": len(audit.images),
        "formats": dict(Counter(i.format or "unreadable" for i in audit.images)),
        "extensions": dict(Counter(i.extension for i in audit.images)),
        "width": _span(widths),
        "height": _span(heights),
        "megapixels_median": round(statistics.median([w * h / 1e6 for w, h in zip(widths, heights)]), 2) if widths else None,
        "total_mb": round(sum(i.size_bytes for i in audit.images) / 1e6, 2),
        "corrupt": corrupt,
        "warnings": {i.path: i.warnings for i in audit.images if i.warnings},
        "exact_duplicate_groups": exact_duplicates,
        "already_in_dataset": already_present,
        "near_duplicate_pairs": pairs,
        "annotations": annotations,
        "privacy": privacy,
        "splits": {
            "assigned": dict(split_counts),
            "unassigned": sorted(name for name, split in assigned.items() if not split),
            "images_without_group": sorted(name for name, group in groups.items() if not group),
            "groups_spanning_splits": group_spans,
        },
        "license": {
            "stated": audit.record.get("stated_license") if audit.record else None,
            "redistributable": is_redistributable_license(audit.record.get("stated_license")) if audit.record else None,
            "incompatibility": license_incompatibility(audit.record.get("stated_license")) if audit.record else None,
        },
    }

    if not audit.images:
        audit.blocking.append("no images found in the staging folder")
    if corrupt:
        audit.blocking.append(f"{len(corrupt)} corrupt image(s): {', '.join(corrupt[:5])}")
    if exact_duplicates:
        audit.blocking.append(f"{len(exact_duplicates)} group(s) of identical images staged")
    if already_present:
        audit.blocking.append(f"{len(already_present)} staged image(s) are already in the dataset")
    if annotations["validator_errors"]:
        audit.blocking.append(f"{len(annotations['validator_errors'])} annotation validator error(s)")
    if annotations["images_without_annotation"]:
        audit.blocking.append(f"{len(annotations['images_without_annotation'])} image(s) without an annotation row")
    if annotations["rows_without_image"]:
        audit.blocking.append(f"{len(annotations['rows_without_image'])} annotation row(s) without an image")
    if annotations.get("synthetic_rows"):
        audit.blocking.append("staged real images carry is_synthetic=true rows")
    if privacy["unreviewed"]:
        audit.blocking.append(f"{len(privacy['unreviewed'])} image(s) without a privacy review")
    if privacy["unresolved"]:
        audit.blocking.append(f"{len(privacy['unresolved'])} image(s) with faces and no privacy action")
    if privacy["rejected"]:
        audit.notes.append(f"{len(privacy['rejected'])} image(s) rejected by the privacy review; do not promote them")
    if privacy["bad_rows"]:
        audit.blocking.append(f"{len(privacy['bad_rows'])} unreadable privacy review row(s)")
    if group_spans:
        audit.blocking.append(f"{len(group_spans)} group(s) span more than one split")
    if pairs:
        audit.notes.append(f"{len(pairs)} near-duplicate pair(s) above {threshold}: keep them in one split or drop one of each")
    if annotations["validator_warnings"]:
        audit.notes.append(f"{len(annotations['validator_warnings'])} annotation validator warning(s)")
    audit.notes.append("the decision is a person's; this tool only checks the record against the rights written in it")
    return audit


def _span(values: Sequence[int]) -> dict | None:
    if not values:
        return None
    return {"min": min(values), "median": int(statistics.median(values)), "max": max(values)}


def audit_text(audit: StagedSourceAudit) -> str:
    """The audit as a human-readable report."""
    s = audit.summary
    lines = [
        f"Staged source: {audit.source_id}",
        f"  folder            : {audit.root}",
        f"  decision (person) : {audit.record.decision if audit.record else 'NO RECORD'}",
        f"  images            : {s.get('images', 0)}  formats {s.get('formats', {})}  {s.get('total_mb', 0)} MB",
    ]
    if s.get("width"):
        lines.append(f"  dimensions        : width {s['width']}  height {s['height']}  median {s['megapixels_median']} MP")
    annotations = s.get("annotations", {})
    privacy = s.get("privacy", {})
    splits = s.get("splits", {})
    lines += [
        f"  corrupt           : {len(s.get('corrupt', []))}",
        f"  exact duplicates  : {len(s.get('exact_duplicate_groups', []))} group(s); already in dataset: {len(s.get('already_in_dataset', []))}",
        f"  near duplicates   : {len(s.get('near_duplicate_pairs', []))} pair(s)",
        f"  annotations       : {annotations.get('rows', 0)} row(s) over {annotations.get('annotated_images', 0)} image(s); "
        f"types {annotations.get('by_plate_type', {})}; unique plates {annotations.get('unique_plate_numbers', 0)}",
        f"  annotation checks : {len(annotations.get('validator_errors', []))} error(s), {len(annotations.get('validator_warnings', []))} warning(s)",
        f"  privacy review    : {privacy.get('reviewed', 0)} reviewed, {len(privacy.get('unreviewed', []))} unreviewed, "
        f"{len(privacy.get('unresolved', []))} unresolved, actions {privacy.get('actions', {})}",
        f"  splits            : {splits.get('assigned', {})}; unassigned {len(splits.get('unassigned', []))}; "
        f"groups spanning splits {len(splits.get('groups_spanning_splits', []))}",
        f"  license           : {s.get('license', {}).get('stated')!r} redistributable={s.get('license', {}).get('redistributable')}",
    ]
    lines.append("")
    lines.append(f"Blocking ({len(audit.blocking)}):")
    lines += [f"  - {item}" for item in audit.blocking] or ["  (none)"]
    lines.append(f"Notes ({len(audit.notes)}):")
    lines += [f"  - {item}" for item in audit.notes]
    lines.append("")
    verdict = "READY FOR A PERSON'S PROMOTION DECISION" if audit.promotable else "NOT PROMOTABLE"
    lines.append(f"Verdict: {verdict} (this tool never promotes and never approves a source)")
    return "\n".join(lines)


def dataset_hashes(meta_path: Path, root: Path) -> dict[str, str]:
    """SHA-256 of every image already referenced by a ``meta.csv``."""
    _, rows = read_meta(Path(meta_path))
    hashes: dict[str, str] = {}
    for image in sorted({row.image for row in rows}):
        path = Path(root) / image
        if path.exists():
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()] = image
    return hashes


__all__ = [
    "DECISIONS",
    "IMAGE_SUFFIXES",
    "NEAR_DUPLICATE_THRESHOLD",
    "PERMISSIONS",
    "PRIVACY_ACTIONS",
    "PRIVACY_STATES",
    "RECORD_FIELDS",
    "SUBMITTABLE",
    "SourceRecord",
    "StagedSourceAudit",
    "audit_staged_source",
    "audit_text",
    "blank_record",
    "dataset_hashes",
    "load_source_record",
    "near_duplicate_pairs",
    "record_problems",
]
