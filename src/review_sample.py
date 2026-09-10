"""Build a reproducible human-review sample from an external dataset.

Selects a fixed, seeded subset of an audited dataset and prepares two things
for a person: a CSV to fill in, and an HTML contact sheet to look at.  The
originals are never touched.

**On automated plate typing.**  The competition types are distinguished by
colour as much as by shape -- ``type1`` and ``type1b`` differ *only* in the
colour of the plate -- and this project has no image-decoding dependency, so
pixel data is unavailable.  Colour therefore cannot be established, and neither
can blur, glare, dirt or the presence of a face.  Rather than guess, every row
is emitted as ``needs_human_review`` and :func:`suggest_plate_type` records the
reason.

What *is* reliable without pixels is geometry, taken from the annotation and
the image header: a plate's pixel size, its aspect ratio, how much of the frame
it fills, and how many plates an image holds.  A Russian one-line plate is
520x112 mm (aspect 4.64) and a square/two-line plate is 290x170 mm (aspect
1.71), so the box aspect ratio separates the two shapes cleanly enough to
*prioritise* a reviewer's attention -- which is what the flags below are for.
It is a candidate marker, never a determination: a one-line plate photographed
at a steep angle foreshortens towards square.
"""

from __future__ import annotations

import csv
import html
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Sequence

from src.external_audit import (
    ANNOTATION_EXTENSION,
    IMAGE_EXTENSIONS,
    ImageRecord,
    YoloBox,
    _classify,
    inspect_image,
    parse_yolo_file,
)

logger = logging.getLogger(__name__)

#: Seed fixing which images land in the review sample.  Changing it changes the
#: sample, so it stays constant once a review has started.
REVIEW_SEED: Final[int] = 20260910

REVIEW_SAMPLE_SIZE: Final[int] = 200

#: Columns of the review CSV, in order.
REVIEW_COLUMNS: Final[tuple[str, ...]] = (
    "image",
    "suggested_plate_type",
    "human_plate_type",
    "has_visible_face",
    "quality_notes",
    "review_status",
)

CSV_DELIMITER: Final[str] = ";"

#: Emitted whenever the plate type cannot be established from the evidence.
NEEDS_HUMAN_REVIEW: Final[str] = "needs_human_review"

REVIEW_STATUS_PENDING: Final[str] = "pending"

#: Aspect ratio (width / height) of a Russian one-line plate: 520 x 112 mm.
ONE_LINE_NOMINAL_ASPECT: Final[float] = 4.64

#: Aspect ratio of a Russian square / two-line plate: 290 x 170 mm.
SQUARE_NOMINAL_ASPECT: Final[float] = 1.71

#: At or above this, the box is one-line shaped.  Set well below the nominal
#: 4.64 so that moderate perspective does not push a one-line plate out.
ONE_LINE_ASPECT_MIN: Final[float] = 3.2

#: At or below this, the box is square/two-line shaped.  Set well above the
#: nominal 1.71 for the same reason, in the other direction.
SQUARE_ASPECT_MAX: Final[float] = 2.4

#: Plate width in pixels below which OCR is unlikely to be possible.
TINY_PLATE_WIDTH_PX: Final[float] = 30.0

#: Plate width in pixels below which a plate counts as small/distant.
SMALL_PLATE_WIDTH_PX: Final[float] = 80.0

#: How close to the frame edge (as a fraction) counts as touching it.
EDGE_MARGIN: Final[float] = 0.01

FLAG_SQUARE_CANDIDATE: Final[str] = "square_or_two_line_candidate"
FLAG_ONE_LINE_SHAPE: Final[str] = "one_line_shape"
FLAG_AMBIGUOUS_SHAPE: Final[str] = "ambiguous_shape"
FLAG_SMALL_PLATE: Final[str] = "small_plate"
FLAG_TINY_PLATE: Final[str] = "tiny_plate"
FLAG_MULTIPLE_PLATES: Final[str] = "multiple_plates"
FLAG_TOUCHES_EDGE: Final[str] = "plate_touches_edge"
FLAG_NO_ANNOTATION: Final[str] = "no_annotation"
FLAG_EXTENSION_MISMATCH: Final[str] = "extension_mismatch"
FLAG_UNREADABLE: Final[str] = "unreadable_image"

#: Conditions a person must judge, because they need pixel data we cannot read.
HUMAN_ONLY_CHECKS: Final[tuple[str, ...]] = (
    "plate colour (white vs yellow) -- decides type1 vs type1b",
    "Russian vs foreign plate",
    "visible faces",
    "blur",
    "glare",
    "dirt / occlusion",
    "severe angle (the annotation box is axis-aligned and carries no rotation)",
)


@dataclass(frozen=True)
class PlateGeometry:
    """Geometry of one annotated plate, in pixels of its own image."""

    class_id: int
    width_px: float
    height_px: float
    centre_x: float
    centre_y: float
    norm_width: float
    norm_height: float

    @property
    def aspect(self) -> float:
        return self.width_px / self.height_px if self.height_px else 0.0

    @property
    def area_fraction(self) -> float:
        return self.norm_width * self.norm_height

    def shape_flag(self) -> str:
        if self.aspect >= ONE_LINE_ASPECT_MIN:
            return FLAG_ONE_LINE_SHAPE
        if self.aspect <= SQUARE_ASPECT_MAX:
            return FLAG_SQUARE_CANDIDATE
        return FLAG_AMBIGUOUS_SHAPE

    def touches_edge(self) -> bool:
        left = self.centre_x - self.norm_width / 2
        right = self.centre_x + self.norm_width / 2
        top = self.centre_y - self.norm_height / 2
        bottom = self.centre_y + self.norm_height / 2
        return (
            left <= EDGE_MARGIN
            or top <= EDGE_MARGIN
            or right >= 1 - EDGE_MARGIN
            or bottom >= 1 - EDGE_MARGIN
        )


@dataclass
class ReviewItem:
    """One sampled image, with everything we could work out about it."""

    image_path: str
    label_path: str | None
    width: int | None
    height: int | None
    image_format: str | None
    plates: list[PlateGeometry] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def largest_plate(self) -> PlateGeometry | None:
        return max(self.plates, key=lambda p: p.area_fraction, default=None)

    def quality_notes(self) -> str:
        """Machine-derived facts only -- never an inference about type."""
        parts: list[str] = []
        if self.flags:
            parts.append("flags: " + ", ".join(self.flags))

        if self.width and self.height:
            parts.append(f"image {self.width}x{self.height}")

        if not self.plates:
            parts.append("no annotated plate")
        else:
            plate = self.largest_plate
            assert plate is not None
            parts.append(
                f"{len(self.plates)} plate(s), largest "
                f"{plate.width_px:.0f}x{plate.height_px:.0f}px, "
                f"aspect {plate.aspect:.2f}, "
                f"{plate.area_fraction * 100:.2f}% of frame"
            )
        # The CSV is semicolon separated and meant to be hand-edited, so no
        # note may contain the delimiter -- that would force quoting.
        return " | ".join(parts).replace(CSV_DELIMITER, ",")


def suggest_plate_type(item: ReviewItem) -> tuple[str, str]:
    """Return ``(suggestion, reason)`` for one item.

    Always abstains today.  ``type1`` and ``type1b`` are separated only by
    plate colour, and ``type1a`` additionally requires the plate to be white,
    so no honest suggestion can be made from geometry alone.  The shape flags
    travel in ``quality_notes`` instead, where they prioritise review without
    pretending to be a label.
    """
    if FLAG_UNREADABLE in item.flags:
        return NEEDS_HUMAN_REVIEW, "image header could not be read"
    if not item.plates:
        return NEEDS_HUMAN_REVIEW, "no annotated plate to judge"

    shape = item.largest_plate.shape_flag() if item.largest_plate else ""
    return (
        NEEDS_HUMAN_REVIEW,
        f"shape suggests {shape}, but plate colour cannot be determined "
        "without decoding pixels, and colour is what separates the types",
    )


def discover_pairs(root: Path) -> dict[str, tuple[Path, Path | None]]:
    """Map every image stem to its image and, when present, its YOLO label.

    Pairing is by file stem, the convention YOLO datasets use when they split
    ``images/`` from ``labels/``.  Stems carrying more than one image are
    skipped rather than paired arbitrarily.
    """
    root = Path(root)
    images: dict[str, list[Path]] = {}
    labels: dict[str, Path] = {}

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        bucket = _classify(path)
        if bucket == "image" and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem, []).append(path)
        elif bucket == "annotation" and path.suffix.lower() == ANNOTATION_EXTENSION:
            labels[path.stem] = path

    pairs: dict[str, tuple[Path, Path | None]] = {}
    for stem, found in images.items():
        if len(found) != 1:
            logger.warning("stem %r has %d images; skipped", stem, len(found))
            continue
        pairs[stem] = (found[0], labels.get(stem))
    return pairs


def select_sample(
    stems: Iterable[str], size: int = REVIEW_SAMPLE_SIZE, seed: int = REVIEW_SEED
) -> list[str]:
    """Choose ``size`` stems reproducibly.

    The candidate list is sorted before sampling so the result depends only on
    the *set* of files and the seed, never on filesystem ordering.  Re-running
    on the same dataset yields the same sample on any machine.
    """
    candidates = sorted(stems)
    if size >= len(candidates):
        return candidates
    return sorted(random.Random(seed).sample(candidates, size))


def build_review_item(
    image_path: Path, label_path: Path | None, root: Path
) -> ReviewItem:
    """Inspect one image and its label, without modifying either."""
    record: ImageRecord = inspect_image(image_path, root)
    item = ReviewItem(
        image_path=record.path,
        label_path=None,
        width=record.width,
        height=record.height,
        image_format=record.format,
    )

    if record.warnings:
        item.flags.append(FLAG_EXTENSION_MISMATCH)
    if not record.is_readable:
        item.flags.append(FLAG_UNREADABLE)
        return item

    if label_path is None:
        item.flags.append(FLAG_NO_ANNOTATION)
        return item

    annotation = parse_yolo_file(label_path, root)
    item.label_path = annotation.path
    item.plates = [
        _geometry(box, record.width, record.height)
        for box in annotation.boxes
        if len(box.coordinates) == 4
    ]

    if not item.plates:
        item.flags.append(FLAG_NO_ANNOTATION)
        return item

    if len(item.plates) > 1:
        item.flags.append(FLAG_MULTIPLE_PLATES)

    largest = item.largest_plate
    assert largest is not None
    item.flags.append(largest.shape_flag())
    if largest.width_px < TINY_PLATE_WIDTH_PX:
        item.flags.append(FLAG_TINY_PLATE)
    elif largest.width_px < SMALL_PLATE_WIDTH_PX:
        item.flags.append(FLAG_SMALL_PLATE)
    if any(plate.touches_edge() for plate in item.plates):
        item.flags.append(FLAG_TOUCHES_EDGE)

    return item


def _geometry(box: YoloBox, width: int, height: int) -> PlateGeometry:
    centre_x, centre_y, norm_width, norm_height = box.coordinates
    return PlateGeometry(
        class_id=box.class_id,
        width_px=norm_width * width,
        height_px=norm_height * height,
        centre_x=centre_x,
        centre_y=centre_y,
        norm_width=norm_width,
        norm_height=norm_height,
    )


def build_review_items(
    root: Path, stems: Sequence[str], pairs: dict[str, tuple[Path, Path | None]]
) -> list[ReviewItem]:
    """Build a :class:`ReviewItem` for each sampled stem."""
    items: list[ReviewItem] = []
    for stem in stems:
        image_path, label_path = pairs[stem]
        items.append(build_review_item(image_path, label_path, Path(root)))
    return items


def flag_counts(items: Sequence[ReviewItem]) -> dict[str, int]:
    """How many sampled images carry each flag."""
    counts: dict[str, int] = {}
    for item in items:
        for flag in item.flags:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda entry: (-entry[1], entry[0])))


def write_review_csv(items: Sequence[ReviewItem], destination: Path) -> int:
    """Write the review CSV and return the number of data rows.

    ``human_plate_type`` and ``has_visible_face`` are left empty on purpose:
    they are the reviewer's columns, and pre-filling them with a guess would
    defeat the point of the review.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(REVIEW_COLUMNS)
        for item in items:
            suggestion, _ = suggest_plate_type(item)
            writer.writerow(
                [
                    item.image_path,
                    suggestion,
                    "",  # human_plate_type
                    "",  # has_visible_face
                    item.quality_notes(),
                    REVIEW_STATUS_PENDING,
                ]
            )

    logger.info("Wrote %d review row(s) to %s", len(items), destination)
    return len(items)


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------

#: Width of the plate close-up, in CSS pixels.
CROP_WIDTH_PX: Final[int] = 260
CROP_HEIGHT_PX: Final[int] = 110


def _crop_style(item: ReviewItem, plate: PlateGeometry) -> str:
    """CSS placing the image so the plate box fills the crop viewport.

    The image is scaled so the plate spans ``CROP_WIDTH_PX``, then shifted so
    the plate's top-left lands at the viewport origin.  This shows a close-up
    of the annotated plate without decoding or writing a single pixel.
    """
    if not (item.width and item.height) or plate.norm_width <= 0:
        return "display:none"

    display_width = CROP_WIDTH_PX / plate.norm_width
    display_height = display_width * (item.height / item.width)
    left = -(plate.centre_x - plate.norm_width / 2) * display_width
    top = -(plate.centre_y - plate.norm_height / 2) * display_height
    # Centre the plate vertically in the viewport.
    top += (CROP_HEIGHT_PX - plate.norm_height * display_height) / 2

    return (
        f"width:{display_width:.1f}px;height:{display_height:.1f}px;"
        f"left:{left:.1f}px;top:{top:.1f}px"
    )


def build_contact_sheet(
    items: Sequence[ReviewItem],
    root: Path,
    *,
    source_id: str,
    seed: int = REVIEW_SEED,
) -> str:
    """Render the review contact sheet as a standalone HTML page.

    Images are referenced in place by ``file://`` URL -- nothing is copied, and
    the external dataset is not touched.  Each card shows the full frame with
    the annotation box drawn over it, a close-up of that box, the file name and
    the machine-derived flags.
    """
    root = Path(root)
    cards: list[str] = []

    for item in items:
        image_uri = html.escape((root / item.image_path).as_uri())
        name = html.escape(Path(item.image_path).name)
        flags = " ".join(
            f'<span class="flag {html.escape(flag)}">{html.escape(flag)}</span>'
            for flag in item.flags
        )

        boxes = "".join(
            f'<i style="left:{(p.centre_x - p.norm_width / 2) * 100:.2f}%;'
            f"top:{(p.centre_y - p.norm_height / 2) * 100:.2f}%;"
            f"width:{p.norm_width * 100:.2f}%;"
            f'height:{p.norm_height * 100:.2f}%"></i>'
            for p in item.plates
        )

        plate = item.largest_plate
        if plate is not None:
            crop = (
                f'<div class="crop"><img src="{image_uri}" alt="" '
                f'style="{_crop_style(item, plate)}"></div>'
            )
            geometry = (
                f"{plate.width_px:.0f}&times;{plate.height_px:.0f}px &middot; "
                f"aspect {plate.aspect:.2f}"
            )
        else:
            crop = '<div class="crop empty">no annotated plate</div>'
            geometry = "&mdash;"

        cards.append(
            f'<article class="card" data-flags="{html.escape(" ".join(item.flags))}">'
            f'<div class="frame"><img src="{image_uri}" alt="" loading="lazy">{boxes}</div>'
            f"{crop}"
            f'<div class="meta"><b>{name}</b><span>{geometry}</span></div>'
            f'<div class="flags">{flags}</div>'
            "</article>"
        )

    counts = flag_counts(items)
    summary = "".join(
        f"<li><code>{html.escape(flag)}</code> &mdash; {count}</li>"
        for flag, count in counts.items()
    )
    human_checks = "".join(f"<li>{html.escape(check)}</li>" for check in HUMAN_ONLY_CHECKS)
    filters = "".join(
        f'<button data-filter="{html.escape(flag)}">{html.escape(flag)} ({count})</button>'
        for flag, count in counts.items()
    )

    return _CONTACT_SHEET_TEMPLATE.format(
        source_id=html.escape(source_id),
        root=html.escape(str(root)),
        count=len(items),
        seed=seed,
        crop_h=CROP_HEIGHT_PX,
        summary=summary,
        human_checks=human_checks,
        filters=filters,
        cards="\n".join(cards),
    )


_CONTACT_SHEET_TEMPLATE: Final[str] = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Review sample - {source_id}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px;
          background: #f6f6f7; color: #16161a; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #17171a; color: #e8e8ea; }}
    .card {{ background: #212126 !important; }}
    header, .panel {{ background: #212126 !important; }}
  }}
  header, .panel {{ background: #fff; border-radius: 10px; padding: 16px 20px;
                    margin-bottom: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  .warn {{ border-left: 4px solid #d08700; padding-left: 12px; }}
  ul {{ margin: 6px 0; padding-left: 20px; }}
  .grid {{ display: grid; gap: 14px;
           grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }}
  .card {{ background: #fff; border-radius: 10px; overflow: hidden;
           padding-bottom: 8px; }}
  .frame {{ position: relative; line-height: 0; background: #000; }}
  .frame img {{ width: 100%; height: 170px; object-fit: contain; }}
  .frame i {{ position: absolute; border: 2px solid #ff3b30;
              box-shadow: 0 0 0 1px rgba(255,255,255,.6); }}
  .crop {{ position: relative; overflow: hidden; height: {crop_h}px;
           background: #000; }}
  .crop img {{ position: absolute; max-width: none; }}
  .crop.empty {{ display: flex; align-items: center; justify-content: center;
                 color: #888; font-size: 12px; line-height: {crop_h}px; }}
  .meta {{ display: flex; justify-content: space-between; gap: 8px;
           padding: 8px 10px 4px; font-size: 12px; }}
  .meta b {{ font-family: ui-monospace, monospace; }}
  .flags {{ padding: 0 10px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .flag {{ font-size: 10px; padding: 2px 6px; border-radius: 999px;
           background: #e8e8ed; color: #444; }}
  .flag.square_or_two_line_candidate {{ background: #ffd60a; color: #3a2f00; }}
  .flag.tiny_plate, .flag.small_plate {{ background: #ffb4a8; color: #4a1710; }}
  .flag.multiple_plates {{ background: #b8e0ff; color: #0b2c45; }}
  button {{ font: inherit; font-size: 12px; padding: 4px 10px; margin: 2px;
            border: 1px solid #bbb; border-radius: 999px; background: transparent;
            color: inherit; cursor: pointer; }}
  button.on {{ background: #16161a; color: #fff; border-color: #16161a; }}
  .card.broken {{ outline: 2px solid #ff3b30; }}
  .card.broken .frame::after {{ content: "preview failed - open the file directly";
      position: absolute; inset: 0; display: flex; align-items: center;
      justify-content: center; color: #ff8a80; font-size: 12px; text-align: center;
      padding: 8px; }}
  #render-status {{ font-weight: 600; }}
  #render-status.ok {{ color: #1a7f37; }}
  #render-status.bad {{ color: #c00; }}
</style></head><body>
<header>
  <h1>Review sample &mdash; {source_id}</h1>
  <p><b>{count}</b> images, selected with seed <code>{seed}</code>, from
     <code>{root}</code>.</p>
  <p>Images are referenced in place. Nothing here modifies, copies or imports
     the dataset.</p>
  <p>Preview status: <span id="render-status">checking&hellip;</span>
     <br><small>Many files in this dataset are named <code>.bmp</code> but hold
     JPEG or PNG data. Browsers normally sniff the content and render them
     anyway; this line reports whether yours did.</small></p>
</header>

<div class="panel warn">
  <b>The plate type is not filled in for you.</b>
  <code>type1</code> and <code>type1b</code> differ only by plate colour, and
  this project decodes no pixels, so every row is
  <code>needs_human_review</code>. The badges below come from the annotation
  box geometry alone and are there to prioritise your attention, not to label
  anything. A one-line plate seen at a steep angle foreshortens towards square,
  so <code>square_or_two_line_candidate</code> means &ldquo;look here
  first&rdquo;, not &ldquo;this is type1a&rdquo;.
  <p style="margin-bottom:0">You still have to judge, per image:</p>
  <ul>{human_checks}</ul>
</div>

<div class="panel">
  <b>Flags in this sample</b>
  <ul>{summary}</ul>
  <div><button data-filter="" class="on">all</button>{filters}</div>
</div>

<div class="grid">
{cards}
</div>

<script>
// Report broken previews rather than showing a silent blank box: the dataset
// is full of files whose extension disagrees with their real format.
(function () {{
  var failed = 0, total = document.querySelectorAll(".frame img").length;
  var status = document.getElementById("render-status");
  function refresh() {{
    if (failed === 0) {{
      status.textContent = total + " of " + total + " previews rendered";
      status.className = "ok";
    }} else {{
      status.textContent = failed + " of " + total + " previews FAILED to render";
      status.className = "bad";
    }}
  }}
  document.querySelectorAll(".frame img").forEach(function (img) {{
    img.addEventListener("error", function () {{
      failed++;
      img.closest(".card").classList.add("broken");
      refresh();
    }});
  }});
  window.addEventListener("load", refresh);
  refresh();
}})();

document.querySelectorAll("button[data-filter]").forEach(function (button) {{
  button.addEventListener("click", function () {{
    var wanted = button.dataset.filter;
    document.querySelectorAll("button[data-filter]").forEach(function (other) {{
      other.classList.toggle("on", other === button);
    }});
    document.querySelectorAll(".card").forEach(function (card) {{
      var flags = card.dataset.flags.split(" ");
      card.hidden = wanted !== "" && flags.indexOf(wanted) === -1;
    }});
  }});
}});
</script>
</body></html>
"""


def write_contact_sheet(
    items: Sequence[ReviewItem],
    root: Path,
    destination: Path,
    *,
    source_id: str,
    seed: int = REVIEW_SEED,
) -> Path:
    """Write the contact sheet to ``destination`` and return the path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_contact_sheet(items, root, source_id=source_id, seed=seed),
        encoding="utf-8",
    )
    logger.info("Wrote contact sheet for %d image(s) to %s", len(items), destination)
    return destination
