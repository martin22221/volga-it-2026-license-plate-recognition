#!/usr/bin/env python
"""Contact sheets for a human review of a staged source.

One card per annotated plate: the full image with the bounding box drawn on it,
the plate crop beside it, and the facts a reviewer needs to accept or refuse the
row -- candidate id, staged file, proposed ``plate_type``, the ``plate_num`` as
annotated, the creator and licence, and the verdict carried from intake
(``PASS`` / ``QUESTIONABLE`` / ``FAIL``) with the reason.

Images rejected at intake get a card too, marked ``FAIL``, so a reviewer sees
what was refused and why rather than only what survived.

The sheets are review material: they are written under
``data/real_staging/review/<source_id>/`` and are never committed -- they embed
other people's photographs.

Usage::

    python scripts/build_intake_review.py wikimedia_commons_curated
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STAGING = REPO_ROOT / "data" / "real_staging"

CARD_W, SCENE_W, SCENE_H, CROP_H, PAD, LINE = 1500, 620, 380, 300, 12, 17
VERDICT_COLOUR = {
    "PASS": (28, 120, 48),
    "QUESTIONABLE": (176, 108, 8),
    "FAIL": (176, 32, 32),
}


#: Most of these photographers are Russian and their credit lines are Cyrillic.
#: Pillow's built-in bitmap font has no Cyrillic glyphs, so it would draw every
#: one of those names as a row of empty boxes -- and the creator is exactly what
#: a reviewer is checking. Find a TrueType face that covers it.
_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def _font(size: int = 13):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()  # pragma: no cover - last resort


FONT = _font()
FONT_BOLD = _font(14)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source_id")
    parser.add_argument("--per-sheet", type=int, default=4)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _card(scene_path: Path, row: dict | None, facts: list[tuple[str, str]],
          verdict: str) -> Image.Image:
    """One review card: scene with the box, the plate crop, and the facts."""
    height = max(SCENE_H, CROP_H) + LINE * (len(facts) + 1) + PAD * 2
    card = Image.new("RGB", (CARD_W, height), (252, 252, 252))
    draw = ImageDraw.Draw(card)

    scene = Image.open(scene_path).convert("RGB")
    box = None
    if row:
        x, y = int(row["bbox_x"]), int(row["bbox_y"])
        w, h = int(row["bbox_w"]), int(row["bbox_h"])
        box = (x, y, w, h)
        marked = scene.copy()
        pen = ImageDraw.Draw(marked)
        pen.rectangle([x, y, x + w, y + h], outline=(255, 40, 40),
                      width=max(3, scene.width // 400))
        scene_draw = marked
    else:
        scene_draw = scene

    k = min(SCENE_W / scene_draw.width, SCENE_H / scene_draw.height)
    thumb = scene_draw.resize((max(1, int(scene_draw.width * k)),
                               max(1, int(scene_draw.height * k))), Image.LANCZOS)
    card.paste(thumb, (PAD, PAD))

    if box:
        x, y, w, h = box
        px, py = int(w * 0.10) + 6, int(h * 0.45) + 6
        crop = scene.crop((max(0, x - px), max(0, y - py),
                           min(scene.width, x + w + px), min(scene.height, y + h + py)))
        ck = min((CARD_W - SCENE_W - PAD * 3) / crop.width, CROP_H / crop.height, 12.0)
        crop = crop.resize((max(1, int(crop.width * ck)), max(1, int(crop.height * ck))),
                           Image.LANCZOS)
        card.paste(crop, (SCENE_W + PAD * 2, PAD))
    scene.close()

    ty = max(SCENE_H, CROP_H) + PAD
    draw.text((PAD, ty), f"[{verdict}]", fill=VERDICT_COLOUR.get(verdict, (0, 0, 0)), font=FONT_BOLD)
    for i, (label, value) in enumerate(facts):
        draw.text((PAD + 115, ty + i * LINE), label, fill=(105, 105, 115), font=FONT)
        draw.text((PAD + 225, ty + i * LINE), str(value)[:170], fill=(20, 20, 20), font=FONT)
    draw.rectangle([0, 0, CARD_W - 1, height - 1], outline=(215, 215, 220))
    return card


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stage = STAGING / "incoming" / args.source_id
    out = STAGING / "review" / args.source_id
    out.mkdir(parents=True, exist_ok=True)

    meta = _read_csv(stage / "meta.csv")
    acq = {r["staged_image"]: r for r in _read_csv(stage / "acquisition_record.csv")}
    privacy = {r["image"]: r for r in _read_csv(stage / "privacy_review.csv")}
    groups = {r["image"]: r["group"] for r in _read_csv(stage / "groups.csv")}
    verdict_file = stage / "review_verdicts.json"
    verdicts = json.loads(verdict_file.read_text(encoding="utf-8")) if verdict_file.is_file() else {}
    by_name = {Path(k).name: v for k, v in verdicts.get("verdicts", {}).items()}
    notes = {Path(k).name: v for k, v in verdicts.get("notes", {}).items()}

    cards: list[Image.Image] = []
    annotated = {r["image"] for r in meta}

    for row in meta:
        rel = row["image"]
        name = Path(rel).name
        rec = acq.get(rel, {})
        pr = privacy.get(rel, {})
        verdict = by_name.get(name, "PASS")
        facts = [
            ("candidate", f"{rec.get('candidate_id','?')}   {name}"),
            ("plate_type", row["plate_type"]),
            ("plate_num", row["plate_num"]),
            ("bbox", f"x{row['bbox_x']} y{row['bbox_y']} w{row['bbox_w']} h{row['bbox_h']}"),
            ("creator", rec.get("creator", "?")),
            ("license", f"{rec.get('license','?')}   {rec.get('license_url','')}"),
            ("source page", rec.get("file_page_url", "")),
            ("group", groups.get(rel, "?")),
            ("privacy", f"faces={pr.get('faces_present','?')} action={pr.get('action','?')}"),
            ("note", notes.get(name, "")),
        ]
        cards.append(_card(stage / rel, row, facts, verdict))

    # Images that were staged but never annotated: rejected, or still open.
    for rel, rec in acq.items():
        if rel in annotated:
            continue
        name = Path(rel).name
        pr = privacy.get(rel, {})
        facts = [
            ("candidate", f"{rec.get('candidate_id','?')}   {name}"),
            ("plate_type", "-- not annotated --"),
            ("plate_num", "--"),
            ("expected", rec.get("plate_class_expected", "")),
            ("creator", rec.get("creator", "?")),
            ("license", rec.get("license", "?")),
            ("source page", rec.get("file_page_url", "")),
            ("privacy", f"faces={pr.get('faces_present','?')} action={pr.get('action','?')}"),
            ("reason", notes.get(name, "")),
        ]
        cards.append(_card(stage / rel, None, facts, by_name.get(name, "FAIL")))

    written = []
    for i in range(0, len(cards), args.per_sheet):
        chunk = cards[i : i + args.per_sheet]
        sheet = Image.new("RGB", (CARD_W + PAD * 2,
                                  sum(c.height for c in chunk) + PAD * (len(chunk) + 1)),
                          (232, 232, 236))
        y = PAD
        for card in chunk:
            sheet.paste(card, (PAD, y))
            y += card.height + PAD
        path = out / f"review_{args.source_id}_{i // args.per_sheet + 1:02d}.jpg"
        sheet.save(path, quality=88)
        written.append(path)
        print(f"  {path}")

    print(f"\ncards: {len(cards)}  sheets: {len(written)}")
    print(f"review material: {out}  (not committed)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
