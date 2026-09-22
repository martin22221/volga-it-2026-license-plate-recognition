"""Acquire the photographs an online-discovery manifest has already approved.

Discovery (``src/online_discovery.py``) decides *whether* a photograph may be
used. This module does the one step that follows: fetching the approved files
and writing down, per file, what we took and where it came from.

Three rules shape everything here:

**Only what a person approved.** The input is a candidate manifest, and only
rows whose ``decision`` is ``ACCEPT_FOR_SUBMISSION`` are ever considered. There
is no discovery, no widening, no "while we are here".

**The licence is re-read at the source, not trusted from the manifest.** A file
on Wikimedia Commons can be relicensed, reverted, overwritten or deleted after
we recorded it. Before anything is downloaded, the file's current metadata is
read from the Commons API and compared with what the manifest recorded. A
mismatch **holds that candidate**; it never silently updates the record or
substitutes a different file.

**A hold is per candidate.** One unavailable or changed file does not stop the
other twenty-one, and it is never replaced by something else.

Nothing here writes into ``dataset/``. Acquisition ends in
``data/real_staging/incoming/<source_id>/``, and a person decides what happens
next -- see ``docs/real_data_intake.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from src.dataset_meta import is_redistributable_license

#: Wikimedia asks every automated client to identify itself and give a contact
#: route; an honest User-Agent is the condition under which the API is offered.
#: See https://meta.wikimedia.org/wiki/User-Agent_policy.
USER_AGENT: str = (
    "volga-it-2026-license-plate-recognition/1.0 "
    "(research dataset for the Volga-IT 2026 competition; "
    "https://github.com/martin22221/volga-it-2026-license-plate-recognition)"
)

COMMONS_API: str = "https://commons.wikimedia.org/w/api.php"

#: Columns of the per-file acquisition record. One row per photograph actually
#: fetched: what it is, where it came from, and what we are required to say
#: about it when we republish it.
ACQUISITION_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "staged_image",
    "original_filename",
    "original_url",
    "file_page_url",
    "creator",
    "license",
    "license_url",
    "attribution",
    "acquired_at",
    "sha256",
    "bytes",
    "width",
    "height",
    "source_group",
    "plate_class_expected",
    "status",
    "notes",
)

#: How a candidate ended up. ``ACQUIRED`` is the only one that produced a file.
ACQUISITION_STATES: tuple[str, ...] = ("ACQUIRED", "HELD", "FAILED")

#: Licence strings, normalised, that we accept as unchanged from one another.
#: Commons writes the same licence several ways ("CC0", "CC0 1.0",
#: "Public domain"), and a cosmetic difference is not a relicensing.
_LICENSE_ALIASES: dict[str, str] = {
    "cc0": "cc0",
    "cc01.0": "cc0",
    "cc010": "cc0",
    "publicdomain": "public domain",
    "pd": "public domain",
    "norestrictions": "public domain",
}


def normalise_license(value: str) -> str:
    """Collapse a licence string to a comparable form.

    Case, punctuation and spacing vary between Commons templates; the licence
    itself does not. ``CC BY 4.0``, ``cc-by-4.0`` and ``CC BY 4.0 `` are one
    licence, while ``CC BY 4.0`` and ``CC BY-SA 4.0`` are two.
    """
    squeezed = re.sub(r"[\s\-_]+", "", (value or "").strip().lower())
    return _LICENSE_ALIASES.get(squeezed, squeezed)


def licenses_match(recorded: str, live: str) -> bool:
    """Is the licence now at the source the one the manifest recorded?"""
    return normalise_license(recorded) == normalise_license(live)


@dataclasses.dataclass(slots=True)
class AcquisitionResult:
    """What happened to one candidate."""

    candidate_id: str
    status: str
    notes: str = ""
    record: dict[str, Any] | None = None

    @property
    def acquired(self) -> bool:
        return self.status == "ACQUIRED"


def approved_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows a person approved, and only those."""
    return [r for r in rows if (r.get("decision") or "").strip() == "ACCEPT_FOR_SUBMISSION"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so a large photograph is not held whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def title_from_file_page(url: str) -> str:
    """Recover the ``File:...`` title from a Commons file-page URL."""
    tail = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    return urllib.parse.unquote(tail).replace("_", " ")


def filename_from_url(url: str) -> str:
    """The original filename a download URL refers to.

    Only the URL *path* names the file. Commons appends tracking parameters
    (``?utm_source=...``) to the URLs it hands out, and a query string is not
    part of a filename -- on Windows ``?`` and ``&`` make the path invalid
    outright. Percent-escapes are decoded so the name on disk is the name the
    photographer gave it, and anything that could still redirect the write
    (separators, drive colons, traversal) is folded to an underscore.
    """
    path = urllib.parse.urlparse(url).path
    name = urllib.parse.unquote(path.rsplit("/", 1)[-1])
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "download.bin"


def _open(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_commons_metadata(
    titles: Sequence[str],
    *,
    timeout: float = 60.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> dict[str, dict[str, Any]]:
    """Read the current licence, author and file URL for each title.

    Returns a mapping of title to a flat record. A title Commons does not know
    is simply absent from the result, which the caller reports as a hold.
    """
    read = opener or _open
    out: dict[str, dict[str, Any]] = {}
    batch = 25  # the API accepts 50; 25 keeps each response small
    for start in range(0, len(titles), batch):
        chunk = list(titles[start : start + batch])
        query = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "iiprop": "extmetadata|url|size|mime|sha1",
            "titles": "|".join(chunk),
        }
        payload = json.loads(read(f"{COMMONS_API}?{urllib.parse.urlencode(query)}", timeout))
        pages = payload.get("query", {}).get("pages", {}) or {}
        for page in pages.values():
            if "missing" in page or not page.get("imageinfo"):
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {}) or {}

            def field(name: str) -> str:
                value = meta.get(name, {}).get("value", "")
                return re.sub(r"<[^>]+>", "", str(value)).strip()

            out[page["title"]] = {
                "title": page["title"],
                "url": info.get("url", ""),
                "descriptionurl": info.get("descriptionurl", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "size": info.get("size", 0),
                "mime": info.get("mime", ""),
                "sha1": info.get("sha1", ""),
                "license": field("LicenseShortName"),
                "license_url": field("LicenseUrl"),
                "artist": field("Artist"),
                "credit": field("Credit"),
                "attribution_required": field("AttributionRequired"),
                "restrictions": field("Restrictions"),
            }
    return out


def check_against_source(
    row: dict[str, Any], live: dict[str, Any] | None
) -> list[str]:
    """Problems that must hold this candidate back, comparing record to source.

    Returns an empty list when the file at the source still matches what was
    approved. Every entry is a reason to hold -- never a reason to substitute a
    different file or to quietly update the manifest.
    """
    problems: list[str] = []
    if live is None:
        return ["not found at the source now (deleted, renamed, or never resolved)"]

    recorded = (row.get("license") or "").strip()
    current = (live.get("license") or "").strip()
    if not current:
        problems.append("the source states no licence now")
    elif not licenses_match(recorded, current):
        problems.append(
            f"licence changed at the source: manifest {recorded!r}, source {current!r}"
        )
    elif not is_redistributable_license(current):
        problems.append(f"licence {current!r} is not redistributable")

    if (live.get("mime") or "") not in ("image/jpeg", "image/png"):
        problems.append(f"unexpected media type {live.get('mime')!r}")

    restrictions = (live.get("restrictions") or "").strip()
    if restrictions:
        problems.append(f"the source now records a usage restriction: {restrictions!r}")

    if not (live.get("url") or ""):
        problems.append("the source offers no download URL")

    return problems


def acquire_one(
    row: dict[str, Any],
    live: dict[str, Any] | None,
    *,
    originals_dir: Path,
    timeout: float = 180.0,
    opener: Callable[[str, float], bytes] | None = None,
) -> AcquisitionResult:
    """Verify one candidate at its source and, if unchanged, fetch the original."""
    candidate_id = str(row.get("candidate_id", ""))
    problems = check_against_source(row, live)
    if problems:
        return AcquisitionResult(candidate_id, "HELD", "; ".join(problems))

    assert live is not None  # check_against_source holds on None
    read = opener or _open
    filename = filename_from_url(live["url"])
    target = originals_dir / filename
    originals_dir.mkdir(parents=True, exist_ok=True)
    try:
        blob = read(live["url"], timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as error:
        return AcquisitionResult(candidate_id, "FAILED", f"download failed: {error}")

    expected = int(live.get("size") or 0)
    if expected and len(blob) != expected:
        return AcquisitionResult(
            candidate_id,
            "FAILED",
            f"size mismatch: the source announced {expected} bytes, {len(blob)} arrived",
        )
    target.write_bytes(blob)

    digest = sha256_file(target)
    sha1 = hashlib.sha1(blob).hexdigest()  # noqa: S324 - matching Commons' own checksum
    if live.get("sha1") and sha1 != live["sha1"]:
        target.unlink(missing_ok=True)
        return AcquisitionResult(
            candidate_id, "FAILED", "SHA-1 does not match the checksum Commons published"
        )

    record = {
        "candidate_id": candidate_id,
        "staged_image": "",  # filled in when the working copy is made
        "original_filename": filename,
        "original_url": live["url"],
        "file_page_url": live.get("descriptionurl") or row.get("file_page_url", ""),
        "creator": live.get("artist") or row.get("creator", ""),
        "license": live.get("license", ""),
        "license_url": live.get("license_url") or row.get("license_url", ""),
        "attribution": row.get("attribution", "") or live.get("artist", ""),
        "acquired_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest,
        "bytes": len(blob),
        "width": live.get("width", 0),
        "height": live.get("height", 0),
        "source_group": row.get("source_group", ""),
        "plate_class_expected": row.get("plate_class", ""),
        "status": "ACQUIRED",
        "notes": "",
    }
    return AcquisitionResult(candidate_id, "ACQUIRED", "", record)


def acquire(
    rows: Sequence[dict[str, Any]],
    *,
    originals_dir: Path,
    timeout: float = 180.0,
    opener: Callable[[str, float], bytes] | None = None,
    pause: float = 0.5,
    log: Callable[[str], None] = print,
) -> list[AcquisitionResult]:
    """Acquire every approved row, holding the ones the source no longer supports.

    ``pause`` spaces the requests out. We are a guest on someone else's
    infrastructure and there is no hurry.
    """
    approved = approved_rows(rows)
    titles = [title_from_file_page(r.get("file_page_url", "")) for r in approved]
    log(f"verifying {len(approved)} approved candidate(s) at the source")
    live_by_title = fetch_commons_metadata(titles, timeout=timeout, opener=opener)

    results: list[AcquisitionResult] = []
    for row, title in zip(approved, titles):
        live = live_by_title.get(title)
        result = acquire_one(
            row, live, originals_dir=originals_dir, timeout=timeout, opener=opener
        )
        results.append(result)
        mark = "ok  " if result.acquired else "HOLD"
        log(f"  [{mark}] {result.candidate_id} {title[:60]}" + (f" -- {result.notes}" if result.notes else ""))
        if pause and opener is None:
            time.sleep(pause)
    return results


__all__ = [
    "ACQUISITION_FIELDS",
    "ACQUISITION_STATES",
    "AcquisitionResult",
    "USER_AGENT",
    "acquire",
    "acquire_one",
    "approved_rows",
    "check_against_source",
    "fetch_commons_metadata",
    "filename_from_url",
    "licenses_match",
    "normalise_license",
    "sha256_file",
    "title_from_file_page",
]
