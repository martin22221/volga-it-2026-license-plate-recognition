"""Candidate manifest for online real-photograph discovery.

Discovery finds *candidates*; it never acquires and never approves. Each row
records one photograph found on a public platform, what establishes its rights,
and what a person concluded about its plate class. The rules here decide when a
row's recorded evidence actually supports the decision written beside it.

Two ideas carry most of the weight:

**The gate is redistribution, not use.** The submitted dataset is published
under plain CC BY 4.0 (``dataset/README.md``), so a photograph may be accepted
only if its own licence permits redistribution, modification and commercial use
with attribution as the only condition. ShareAlike passes the first three and
fails anyway: it forbids relicensing the aggregate, which is exactly what
publication requires. ``src.dataset_meta`` already encodes that gate and is
reused here rather than restated.

**A platform is not a photographer.** A licence field on a hosting site says
what the uploader claimed. ``provenance_status`` records whether we actually
read the file's own page, and an unverified row can never be accepted.

Nothing here downloads an image. See ``docs/online_source_discovery.md``.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.dataset_meta import (
    is_redistributable_license,
    license_incompatibility,
    normalize_license,
)

#: Columns of a candidate manifest, in file order.
MANIFEST_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "source_platform",
    "original_url",
    "file_page_url",
    "creator",
    "license",
    "license_url",
    "attribution",
    "redistribution_status",
    "modification_status",
    "plate_class",
    "class_confidence",
    "is_real_photo",
    "full_scene_or_crop",
    "resolution",
    "source_group",
    "provenance_status",
    "decision",
    "notes",
)

#: What a reviewer may conclude about the plate in the photograph.
PLATE_CLASSES: tuple[str, ...] = (
    "CONFIRMED_TYPE1A",
    "CONFIRMED_TYPE1B",
    "CONFIRMED_TYPE1",
    "OTHER",
    "AMBIGUOUS",
    "REJECT",
)
#: Classes a person has actually confirmed, as opposed to parked or refused.
CONFIRMED_CLASSES: frozenset[str] = frozenset(
    {"CONFIRMED_TYPE1A", "CONFIRMED_TYPE1B", "CONFIRMED_TYPE1", "OTHER"}
)
#: The rights decision, same vocabulary as ``docs/real_data_intake.md``.
DECISIONS: tuple[str, ...] = (
    "ACCEPT_FOR_SUBMISSION",
    "TRAINING_ONLY_IF_LEGAL",
    "REFERENCE_ONLY",
    "REJECT",
    "PENDING",
)
#: How far the rights claim was actually checked.
PROVENANCE_STATES: tuple[str, ...] = (
    "VERIFIED_ON_SOURCE",  # we opened the file's own page and read the licence
    "PLATFORM_METADATA_ONLY",  # an index (e.g. Openverse) said so; not yet checked
    "UNRESOLVED",  # checking was attempted and did not settle it
    "UNKNOWN",
)
TRISTATE: tuple[str, ...] = ("yes", "no", "unclear")
#: Licences that need no credit line, so a blank ``attribution`` is honest.
NO_ATTRIBUTION_NEEDED: frozenset[str] = frozenset(
    {"cc0", "cc0 1.0", "public domain", "own work"}
)
UNKNOWN = "UNKNOWN"


@dataclass
class Candidate:
    """One discovered photograph and the evidence recorded about it."""

    candidate_id: str = ""
    source_platform: str = ""
    original_url: str = ""
    file_page_url: str = ""
    creator: str = UNKNOWN
    license: str = ""
    license_url: str = ""
    attribution: str = ""
    redistribution_status: str = "unclear"
    modification_status: str = "unclear"
    plate_class: str = "AMBIGUOUS"
    class_confidence: str = ""
    is_real_photo: str = "unclear"
    full_scene_or_crop: str = UNKNOWN
    resolution: str = ""
    source_group: str = ""
    provenance_status: str = "UNKNOWN"
    decision: str = "PENDING"
    notes: str = ""

    def to_row(self) -> dict[str, str]:
        data = asdict(self)
        return {name: str(data.get(name, "")) for name in MANIFEST_FIELDS}


def _get(data: Mapping[str, object], name: str) -> str:
    return str(data.get(name, "") or "").strip()


def candidate_problems(data: Mapping[str, object]) -> list[str]:
    """Everything inconsistent about one candidate row, worst first.

    This never *makes* a decision. It checks that the evidence recorded beside
    a decision supports it, and it refuses ``ACCEPT_FOR_SUBMISSION`` whenever
    the rights, the provenance or the class are not actually established.
    """
    problems: list[str] = []

    for name in MANIFEST_FIELDS:
        if name not in data:
            problems.append(f"missing column {name!r}")
    for name in ("candidate_id", "source_platform", "file_page_url"):
        if name in data and not _get(data, name):
            problems.append(f"{name} is empty")

    for name in ("redistribution_status", "modification_status", "is_real_photo"):
        value = _get(data, name).lower()
        if value and value not in TRISTATE:
            problems.append(f"{name}={value!r} is not one of {', '.join(TRISTATE)}")
        elif not value:
            problems.append(f"{name} is empty; record 'unclear' rather than leaving it blank")

    plate_class = _get(data, "plate_class").upper()
    if plate_class and plate_class not in PLATE_CLASSES:
        problems.append(f"plate_class={plate_class!r} is not one of {', '.join(PLATE_CLASSES)}")

    decision = _get(data, "decision").upper()
    if decision not in DECISIONS:
        problems.append(f"decision={decision!r} is not one of {', '.join(DECISIONS)}")

    provenance = _get(data, "provenance_status").upper()
    if provenance and provenance not in PROVENANCE_STATES:
        problems.append(
            f"provenance_status={provenance!r} is not one of {', '.join(PROVENANCE_STATES)}"
        )

    licence = _get(data, "license")
    clash = license_incompatibility(licence) if licence else None
    if clash and decision == "ACCEPT_FOR_SUBMISSION":
        problems.append(
            f"decision ACCEPT_FOR_SUBMISSION but the licence {licence!r} {clash}; the "
            "submitted dataset is published under plain CC BY 4.0"
        )

    if decision == "ACCEPT_FOR_SUBMISSION":
        if not licence:
            problems.append("decision ACCEPT_FOR_SUBMISSION but license is empty")
        elif not clash and not is_redistributable_license(licence):
            problems.append(
                f"decision ACCEPT_FOR_SUBMISSION but the licence {licence!r} is not on the "
                "confirmed redistributable list in src/dataset_meta.py; a person must add it "
                "there in the same commit or lower the decision"
            )
        if not _get(data, "license_url"):
            problems.append("decision ACCEPT_FOR_SUBMISSION but license_url is empty")
        for name in ("redistribution_status", "modification_status"):
            if _get(data, name).lower() != "yes":
                problems.append(
                    f"decision ACCEPT_FOR_SUBMISSION but {name}={_get(data, name)!r}"
                )
        if _get(data, "is_real_photo").lower() != "yes":
            problems.append(
                "decision ACCEPT_FOR_SUBMISSION but is_real_photo is not 'yes'; a rendering, "
                "diagram or plate graphic is not a photograph"
            )
        if provenance != "VERIFIED_ON_SOURCE":
            problems.append(
                f"decision ACCEPT_FOR_SUBMISSION but provenance_status={provenance!r}; a "
                "platform's licence label is not evidence until its own page has been read"
            )
        if plate_class not in CONFIRMED_CLASSES:
            problems.append(
                f"decision ACCEPT_FOR_SUBMISSION but plate_class={plate_class!r}; ambiguous "
                "candidates go to review, not into the dataset"
            )
        if normalize_license(licence) not in NO_ATTRIBUTION_NEEDED and not _get(data, "attribution"):
            problems.append(
                f"decision ACCEPT_FOR_SUBMISSION and the licence {licence!r} requires credit, "
                "but attribution is empty"
            )
        if not _get(data, "source_group"):
            problems.append(
                "decision ACCEPT_FOR_SUBMISSION but source_group is empty; the leakage unit "
                "has to be recorded before anything can be split"
            )
    return problems


def gate_verdict(licence: str) -> str:
    """``COMPATIBLE`` / ``INCOMPATIBLE`` / ``UNKNOWN`` for one licence string.

    ``UNKNOWN`` is not a soft yes. It means a person still has to decide, and
    ``candidate_problems`` will refuse an accept that rests on it.
    """
    if not licence.strip():
        return "UNKNOWN"
    if license_incompatibility(licence):
        return "INCOMPATIBLE"
    if is_redistributable_license(licence):
        return "COMPATIBLE"
    return "UNKNOWN"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=";")]


def write_manifest(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS), delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: str(row.get(name, "") or "") for name in MANIFEST_FIELDS})
            written += 1
    return written


@dataclass
class ManifestAudit:
    """What one manifest contains and what blocks it."""

    path: str = ""
    rows: int = 0
    by_class: dict[str, int] = field(default_factory=dict)
    by_decision: dict[str, int] = field(default_factory=dict)
    by_platform: dict[str, int] = field(default_factory=dict)
    by_licence: dict[str, int] = field(default_factory=dict)
    by_provenance: dict[str, int] = field(default_factory=dict)
    accepted_by_class: dict[str, int] = field(default_factory=dict)
    groups: int = 0
    largest_group: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_urls: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def audit_manifest(rows: Sequence[Mapping[str, object]], path: str = "") -> ManifestAudit:
    """Summarise a manifest and collect everything that blocks acquisition."""
    audit = ManifestAudit(path=path, rows=len(rows))
    audit.by_class = _count(_get(r, "plate_class").upper() or UNKNOWN for r in rows)
    audit.by_decision = _count(_get(r, "decision").upper() or UNKNOWN for r in rows)
    audit.by_platform = _count(_get(r, "source_platform") or UNKNOWN for r in rows)
    audit.by_licence = _count(_get(r, "license") or UNKNOWN for r in rows)
    audit.by_provenance = _count(_get(r, "provenance_status").upper() or UNKNOWN for r in rows)
    audit.accepted_by_class = _count(
        _get(r, "plate_class").upper()
        for r in rows
        if _get(r, "decision").upper() == "ACCEPT_FOR_SUBMISSION"
    )

    groups = _count(_get(r, "source_group") or UNKNOWN for r in rows)
    audit.groups = len(groups)
    audit.largest_group = max(groups.values(), default=0)

    seen_ids: dict[str, int] = {}
    seen_urls: dict[str, int] = {}
    for row in rows:
        cid = _get(row, "candidate_id")
        if cid:
            seen_ids[cid] = seen_ids.get(cid, 0) + 1
        url = _get(row, "file_page_url") or _get(row, "original_url")
        if url:
            seen_urls[url] = seen_urls.get(url, 0) + 1
    audit.duplicate_ids = sorted(k for k, n in seen_ids.items() if n > 1)
    audit.duplicate_urls = sorted(k for k, n in seen_urls.items() if n > 1)

    if audit.duplicate_ids:
        audit.blocking.append(f"duplicate candidate_id: {', '.join(audit.duplicate_ids[:5])}")
    if audit.duplicate_urls:
        audit.blocking.append(
            f"the same source URL appears more than once: {', '.join(audit.duplicate_urls[:5])}"
        )

    for row in rows:
        for problem in candidate_problems(row):
            audit.blocking.append(f"{_get(row, 'candidate_id') or '<no id>'}: {problem}")

    accepted = sum(audit.accepted_by_class.values())
    if accepted:
        audit.notes.append(
            f"{accepted} row(s) are ACCEPT_FOR_SUBMISSION; that is a rights verdict about "
            "the photograph, not permission to download in bulk"
        )
    audit.notes.append(
        "discovery records candidates; a person approves acquisition separately"
    )
    return audit


def audit_text(audit: ManifestAudit) -> str:
    """The audit as a report a person reads."""
    lines = [
        f"Online candidate manifest: {audit.path}",
        f"  rows              : {audit.rows}",
        f"  plate class       : {audit.by_class}",
        f"  decision          : {audit.by_decision}",
        f"  provenance        : {audit.by_provenance}",
        f"  platform          : {audit.by_platform}",
        f"  licences          : {audit.by_licence}",
        f"  accepted by class : {audit.accepted_by_class or '{}'}",
        f"  source groups     : {audit.groups} (largest {audit.largest_group})",
    ]
    lines.append("")
    lines.append(f"Blocking ({len(audit.blocking)}):")
    lines += [f"  - {item}" for item in audit.blocking[:60]] or ["  (none)"]
    if len(audit.blocking) > 60:
        lines.append(f"  ... and {len(audit.blocking) - 60} more")
    lines.append("")
    lines.append(f"Notes ({len(audit.notes)}):")
    lines += [f"  - {item}" for item in audit.notes] or ["  (none)"]
    lines.append("")
    verdict = "BLOCKED" if audit.blocking else "CONSISTENT"
    lines.append(f"Verdict: {verdict} (this tool never approves acquisition)")
    return "\n".join(lines)


__all__ = [
    "CONFIRMED_CLASSES",
    "Candidate",
    "DECISIONS",
    "MANIFEST_FIELDS",
    "ManifestAudit",
    "PLATE_CLASSES",
    "PROVENANCE_STATES",
    "audit_manifest",
    "audit_text",
    "candidate_problems",
    "gate_verdict",
    "read_manifest",
    "write_manifest",
]
