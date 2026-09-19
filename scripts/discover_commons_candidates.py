#!/usr/bin/env python
"""Find candidate real photographs of Russian plates on Wikimedia Commons.

Reads the MediaWiki API only: category membership, search results and per-file
metadata (licence, author, size, MIME). **No image is downloaded**, nothing is
imported, and every row it writes is ``PENDING`` -- the plate class and the
final rights decision are a person's, made by looking at the photograph.

What it does automatically:

- walks the categories that actually hold Russian vehicle photographs, and runs
  the English and Russian search terms;
- drops anything that is not a JPEG above a minimum size, because the Russian
  plate categories are dominated by SVG/PNG plate *graphics* and 131x110 region
  -code thumbnails, which are not photographs;
- runs each file's licence through the same CC BY 4.0 gate the dataset uses
  (``src.dataset_meta``), so ShareAlike, NC and ND are marked INCOMPATIBLE;
- records the file page URL, the uploader, the author field and the licence URL
  so a person can verify the claim on the file's own page.

Usage::

    python scripts/discover_commons_candidates.py --out data/real_staging/manifests/x.csv
    python scripts/discover_commons_candidates.py --out x.csv --max-files 500
    python scripts/discover_commons_candidates.py --dry-run     # print the plan only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.online_discovery import Candidate, gate_verdict, write_manifest  # noqa: E402

# The search terms are Russian, and a Windows console defaults to cp1252, which
# cannot encode them. Print replacement characters rather than crashing.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # not a real stream (pytest capture, a pipe)
        pass

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "VolgaIT2026-dataset-research/1.0 "
    "(https://github.com/martin22221/volga-it-2026-license-plate-recognition) "
    "metadata-only"
)
MIN_SIDE = 400

#: Categories that hold photographs of vehicles rather than plate graphics.
CATEGORIES: tuple[tuple[str, int], ...] = (
    ("Category:Vehicles with license plates of Russia", 3),
    ("Category:Automobiles with license plates of Russia", 3),
    ("Category:Buses with license plates of Russia", 3),
    ("Category:Trucks with license plates of Russia", 3),
    ("Category:Yandex.Taxi", 3),
    ("Category:Yellow taxis in Moscow", 2),
    ("Category:Taxis in Moscow", 2),
    ("Category:Taxis in Russia", 2),
    ("Category:Minibuses in Russia", 2),
    ("Category:Diplomatic license plates in Russia", 1),
)
#: Search terms, English and Russian. Commons' search is word-based, so these
#: are deliberately broad; the filters below do the narrowing.
SEARCHES: tuple[str, ...] = (
    "Russian square license plate",
    "Russia two-line license plate",
    "russian license plate type 1A",
    "japanese import Russia license plate",
    "Russian yellow license plate taxi",
    "yellow license plate Russia bus",
    "Russian license plate car",
    "russian registration plate vehicle",
    "тип 1А номер",
    "тип 1Б номер",
    "желтый номер такси",
    "такси Россия номер",
    "маршрутка номер",
    "регистрационный знак Россия",
)


def api(post: bool = False, **params) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    data = urllib.parse.urlencode(params)
    headers = {"User-Agent": USER_AGENT}
    if post:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(API, data=data.encode("utf-8"), headers=headers)
    else:
        request = urllib.request.Request(f"{API}?{data}", headers=headers)
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as handle:
                return json.load(handle)
        except Exception as error:  # network flake; back off and retry
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {last}")


def category_members(category: str, subcats: bool = False, limit: int = 600) -> list[str]:
    out: list[str] = []
    cont: dict[str, str] = {}
    while len(out) < limit:
        data = api(
            action="query",
            list="categorymembers",
            cmtitle=category,
            cmtype="subcat" if subcats else "file",
            cmlimit=min(500, limit - len(out)),
            **cont,
        )
        out += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        if "continue" not in data:
            break
        cont = {"cmcontinue": data["continue"]["cmcontinue"]}
    return out


def search_files(term: str, limit: int = 120) -> list[str]:
    out: list[str] = []
    offset = 0
    while len(out) < limit:
        data = api(
            action="query",
            list="search",
            srsearch=term,
            srnamespace=6,
            srlimit=min(50, limit - len(out)),
            sroffset=offset,
        )
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        out += [h["title"] for h in hits]
        if "continue" not in data:
            break
        offset = data["continue"]["sroffset"]
    return out


def file_info(titles: Sequence[str]) -> dict[str, dict]:
    """Licence, author and size per file. POSTed: title lists overflow a URL."""
    result: dict[str, dict] = {}
    for start in range(0, len(titles), 50):
        chunk = titles[start : start + 50]
        data = api(
            post=True,
            action="query",
            titles="|".join(chunk),
            prop="imageinfo",
            iiprop="url|size|mime|extmetadata|user",
        )
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata", {}) or {}
            value = lambda key: (meta.get(key, {}) or {}).get("value", "")
            result[page.get("title", "")] = {
                "title": page.get("title", ""),
                "pageid": page.get("pageid"),
                "url": info.get("url", ""),
                "descriptionurl": info.get("descriptionurl", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "mime": info.get("mime", ""),
                "uploader": info.get("user", ""),
                "license": value("LicenseShortName"),
                "license_url": value("LicenseUrl"),
                "artist": value("Artist"),
                "attribution": value("Attribution"),
                "usage_terms": value("UsageTerms"),
                "description": value("ImageDescription"),
            }
    return result


def strip_markup(text: str) -> str:
    """Commons returns small HTML fragments in author/description fields."""
    out, depth = [], 0
    for char in text:
        if char == "<":
            depth += 1
        elif char == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(char)
    return " ".join("".join(out).split())


def collect(max_files: int) -> dict[str, str]:
    """Titles to inspect, mapped to where they were found."""
    pool: dict[str, str] = {}
    for category, depth in CATEGORIES:
        seen, frontier = {category}, [(category, 0)]
        while frontier and len(pool) < max_files:
            current, level = frontier.pop(0)
            for title in category_members(current):
                pool.setdefault(title, current)
            if level < depth:
                for sub in category_members(current, subcats=True, limit=300):
                    if sub not in seen:
                        seen.add(sub)
                        frontier.append((sub, level + 1))
        print(f"  {category}: pool {len(pool)}", flush=True)
    for term in SEARCHES:
        if len(pool) >= max_files:
            break
        for title in search_files(term):
            pool.setdefault(title, f"search:{term}")
        print(f"  search {term!r}: pool {len(pool)}", flush=True)
    return pool


def to_candidate(info: dict, found_via: str) -> Candidate:
    licence = info.get("license", "")
    verdict = gate_verdict(licence)
    creator = strip_markup(info.get("artist", "")) or f"uploader:{info.get('uploader', '')}"
    permitted = "yes" if verdict == "COMPATIBLE" else ("no" if verdict == "INCOMPATIBLE" else "unclear")
    return Candidate(
        candidate_id=f"cw{info.get('pageid')}",
        source_platform="wikimedia_commons",
        original_url=info.get("url", ""),
        file_page_url=info.get("descriptionurl", ""),
        creator=creator or "UNKNOWN",
        license=licence or "UNKNOWN",
        license_url=info.get("license_url", ""),
        attribution=strip_markup(info.get("attribution", "")) or creator,
        redistribution_status=permitted,
        modification_status=permitted,
        plate_class="AMBIGUOUS",
        class_confidence="unreviewed",
        is_real_photo="unclear",
        full_scene_or_crop="UNKNOWN",
        resolution=f"{info.get('width', 0)}x{info.get('height', 0)}",
        source_group=f"commons:{info.get('pageid')}",
        provenance_status="PLATFORM_METADATA_ONLY",
        decision="PENDING",
        notes=f"found via {found_via}; licence gate {verdict}; "
        f"{strip_markup(info.get('description', ''))[:120]}",
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None, help="manifest CSV to write")
    parser.add_argument("--max-files", type=int, default=4000, help="stop collecting after this many titles")
    parser.add_argument("--min-side", type=int, default=MIN_SIDE, help="drop images smaller than this")
    parser.add_argument("--keep-incompatible", action="store_true",
                        help="keep rows whose licence fails the gate (for the record)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(f"{len(CATEGORIES)} categories, {len(SEARCHES)} searches, "
              f"max {args.max_files} files, min side {args.min_side}px")
        for category, depth in CATEGORIES:
            print(f"  category depth {depth}: {category}")
        for term in SEARCHES:
            print(f"  search: {term}")
        return 0
    if args.out is None:
        print("error: --out is required (or use --dry-run)", file=sys.stderr)
        return 1

    print("collecting titles ...", flush=True)
    pool = collect(args.max_files)
    print(f"pool: {len(pool)} titles; fetching metadata ...", flush=True)
    info = file_info(list(pool))

    rows, dropped = [], {"not_jpeg": 0, "small": 0, "incompatible": 0}
    for title, rec in info.items():
        if rec.get("mime") != "image/jpeg":
            dropped["not_jpeg"] += 1
            continue
        if min(rec.get("width", 0), rec.get("height", 0)) < args.min_side:
            dropped["small"] += 1
            continue
        verdict = gate_verdict(rec.get("license", ""))
        if verdict == "INCOMPATIBLE" and not args.keep_incompatible:
            dropped["incompatible"] += 1
            continue
        rows.append(to_candidate(rec, pool.get(title, "")).to_row())

    written = write_manifest(args.out, rows)
    print(f"dropped: {dropped}")
    print(f"wrote {written} PENDING candidate row(s) to {args.out}")
    print("Every row is PENDING: a person verifies the file page and the plate class.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
