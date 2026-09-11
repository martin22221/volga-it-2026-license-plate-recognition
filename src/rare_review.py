"""Focused review pages for the rare competition classes.

Collects two kinds of candidate from an audited external dataset and renders
large-preview pages so a person can confirm or reject each one quickly:

``type1a`` candidates
    Images whose annotation box is square-ish or of ambiguous proportion.
    Shape is a *prompt to look*, never a label: a one-line plate photographed
    at a steep angle foreshortens towards square, and a square bounding box is
    not a square plate.

``type1b`` candidates
    Plates whose own colour reads yellow, measured by :mod:`src.plate_color`
    with a guard against yellow bodywork.

Nothing here assigns a competition type on its own.  ``suggested_plate_type``
is filled only where a heuristic is strong, and even then it is a suggestion
for a human to overrule.
"""

from __future__ import annotations

import csv
import html
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Sequence

from src.plate_color import PlateColour
from src.review_sample import (
    CSV_DELIMITER,
    FLAG_AMBIGUOUS_SHAPE,
    FLAG_SQUARE_CANDIDATE,
    NEEDS_HUMAN_REVIEW,
    REVIEW_STATUS_PENDING,
    ReviewItem,
)

logger = logging.getLogger(__name__)

#: Columns of the rare-candidate CSV, in order.
RARE_COLUMNS: Final[tuple[str, ...]] = (
    "image",
    "candidate_reason",
    "suggested_plate_type",
    "human_plate_type",
    "confidence_note",
    "review_status",
)

REASON_SQUARE: Final[str] = "square_or_two_line_candidate"
REASON_AMBIGUOUS: Final[str] = "ambiguous_shape"
REASON_YELLOW: Final[str] = "yellow_plate_colour"
REASON_YELLOW_NEAR: Final[str] = "most_yellow_in_dataset"

#: How many of the most-yellow plates to show even when none clears the
#: threshold, so a reviewer can see the tail and judge the calibration.
YELLOW_SHORTLIST: Final[int] = 40


@dataclass
class RareCandidate:
    """One image put forward for focused review."""

    item: ReviewItem
    reasons: list[str] = field(default_factory=list)
    colour: PlateColour | None = None
    yellowness: float | None = None

    @property
    def image_path(self) -> str:
        return self.item.image_path

    def suggested_plate_type(self) -> str:
        """A type is suggested only where a heuristic is genuinely strong.

        Shape alone never qualifies: it cannot separate a two-line plate from
        a foreshortened one-line plate, and it says nothing about colour.
        """
        if (
            REASON_YELLOW in self.reasons
            and self.colour is not None
            and self.colour.strength() == "strong"
        ):
            return "type1b"
        return NEEDS_HUMAN_REVIEW

    def confidence_note(self) -> str:
        parts: list[str] = []

        if REASON_YELLOW in self.reasons or REASON_YELLOW_NEAR in self.reasons:
            if self.colour is not None and self.colour.plate is not None:
                plate = self.colour.plate
                parts.append(
                    f"plate colour {plate.hex_colour} "
                    f"(Y {plate.luma:.0f}, Cb {plate.blue_diff:.0f}, "
                    f"Cr {plate.red_diff:.0f}), yellowness {plate.yellowness:+.1f}, "
                    f"{self.colour.margin:+.1f} vs surroundings"
                )
                if self.colour.strength() == "strong":
                    parts.append(
                        "strong yellow reading -- still needs a human to confirm "
                        "the plate is Russian and one-line"
                    )
                elif self.colour.strength() == "weak":
                    parts.append("weak yellow reading")
                else:
                    parts.append(
                        "below the yellow threshold, shown only as one of the "
                        "most yellow plates in the dataset"
                    )
            if self.colour is not None and not self.colour.reliable and self.colour.note:
                parts.append(self.colour.note)

        plate = self.item.largest_plate
        if plate is not None and (
            REASON_SQUARE in self.reasons or REASON_AMBIGUOUS in self.reasons
        ):
            parts.append(
                f"box {plate.width_px:.0f}x{plate.height_px:.0f}px, "
                f"aspect {plate.aspect:.2f} "
                "(one-line nominal 4.64, two-line nominal 1.71)"
            )
            parts.append(
                "shape only -- a one-line plate at a steep angle foreshortens "
                "towards square"
            )

        return " | ".join(parts).replace(CSV_DELIMITER, ",")


def collect_shape_candidates(items: Sequence[ReviewItem]) -> dict[str, RareCandidate]:
    """Every reviewed image flagged square-ish or ambiguous."""
    candidates: dict[str, RareCandidate] = {}
    for item in items:
        reasons = [
            reason
            for flag, reason in (
                (FLAG_SQUARE_CANDIDATE, REASON_SQUARE),
                (FLAG_AMBIGUOUS_SHAPE, REASON_AMBIGUOUS),
            )
            if flag in item.flags
        ]
        if reasons:
            candidates[item.image_path] = RareCandidate(item=item, reasons=reasons)
    return candidates


def merge_candidates(
    shape: dict[str, RareCandidate], yellow: dict[str, RareCandidate]
) -> dict[str, RareCandidate]:
    """Combine the two candidate sets, keeping every reason per image."""
    merged: dict[str, RareCandidate] = {}
    for path, candidate in shape.items():
        merged[path] = candidate
    for path, candidate in yellow.items():
        existing = merged.get(path)
        if existing is None:
            merged[path] = candidate
            continue
        for reason in candidate.reasons:
            if reason not in existing.reasons:
                existing.reasons.append(reason)
        existing.colour = candidate.colour
        existing.yellowness = candidate.yellowness
    return dict(sorted(merged.items()))


def write_rare_csv(candidates: Sequence[RareCandidate], destination: Path) -> int:
    """Write the rare-candidate CSV; returns the number of data rows."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(RARE_COLUMNS)
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.image_path,
                    ",".join(candidate.reasons),
                    candidate.suggested_plate_type(),
                    "",  # human_plate_type
                    candidate.confidence_note(),
                    REVIEW_STATUS_PENDING,
                ]
            )
    return len(candidates)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

#: Full-frame preview width on the focused pages, much larger than the
#: contact sheet's thumbnails.
PREVIEW_WIDTH_PX: Final[int] = 420
PREVIEW_HEIGHT_PX: Final[int] = 300
BIG_CROP_WIDTH_PX: Final[int] = 420
BIG_CROP_HEIGHT_PX: Final[int] = 180


def _crop_style(candidate: RareCandidate) -> str:
    item = candidate.item
    plate = item.largest_plate
    if plate is None or not (item.width and item.height) or plate.norm_width <= 0:
        return "display:none"

    display_width = BIG_CROP_WIDTH_PX / plate.norm_width
    display_height = display_width * (item.height / item.width)
    left = -(plate.centre_x - plate.norm_width / 2) * display_width
    top = -(plate.centre_y - plate.norm_height / 2) * display_height
    top += (BIG_CROP_HEIGHT_PX - plate.norm_height * display_height) / 2
    return (
        f"width:{display_width:.1f}px;height:{display_height:.1f}px;"
        f"left:{left:.1f}px;top:{top:.1f}px"
    )


def _card(candidate: RareCandidate, root: Path) -> str:
    item = candidate.item
    uri = html.escape((root / item.image_path).as_uri())
    name = html.escape(Path(item.image_path).name)

    boxes = "".join(
        f'<i style="left:{(p.centre_x - p.norm_width / 2) * 100:.2f}%;'
        f"top:{(p.centre_y - p.norm_height / 2) * 100:.2f}%;"
        f"width:{p.norm_width * 100:.2f}%;"
        f'height:{p.norm_height * 100:.2f}%"></i>'
        for p in item.plates
    )

    reasons = " ".join(
        f'<span class="flag">{html.escape(reason)}</span>' for reason in candidate.reasons
    )

    swatch = ""
    if candidate.colour is not None and candidate.colour.plate is not None:
        plate = candidate.colour.plate
        swatch = (
            f'<span class="swatch" style="background:{plate.hex_colour}" '
            f'title="measured mean plate colour"></span>'
            f"<code>{plate.hex_colour}</code> yellowness "
            f"<b>{plate.yellowness:+.1f}</b>"
        )

    return (
        '<article class="card">'
        f'<div class="head"><b>{name}</b>{swatch}</div>'
        '<div class="views">'
        f'<div class="frame"><img src="{uri}" alt="" loading="lazy">{boxes}</div>'
        f'<div class="crop"><img src="{uri}" alt="" loading="lazy" '
        f'style="{_crop_style(candidate)}"></div>'
        "</div>"
        f'<div class="flags">{reasons}</div>'
        f'<p class="note">{html.escape(candidate.confidence_note())}</p>'
        "</article>"
    )


def build_page(
    candidates: Sequence[RareCandidate],
    root: Path,
    *,
    title: str,
    intro_html: str,
) -> str:
    """Render a focused review page with large previews."""
    cards = "\n".join(_card(candidate, Path(root)) for candidate in candidates)
    return _PAGE_TEMPLATE.format(
        title=html.escape(title),
        intro=intro_html,
        count=len(candidates),
        root=html.escape(str(root)),
        preview_w=PREVIEW_WIDTH_PX,
        preview_h=PREVIEW_HEIGHT_PX,
        crop_w=BIG_CROP_WIDTH_PX,
        crop_h=BIG_CROP_HEIGHT_PX,
        cards=cards or '<p class="panel">No candidates in this group.</p>',
    )


_PAGE_TEMPLATE: Final[str] = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 system-ui, sans-serif; margin: 0; padding: 24px;
          background: #f5f5f7; color: #16161a; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #161619; color: #e9e9ec; }}
    .card, header, .panel {{ background: #212126 !important; }}
  }}
  header, .panel {{ background: #fff; border-radius: 12px; padding: 18px 22px;
                    margin-bottom: 18px; max-width: 1100px; }}
  h1 {{ font-size: 20px; margin: 0 0 10px; }}
  .warn {{ border-left: 4px solid #d08700; }}
  .grid {{ display: grid; gap: 18px;
           grid-template-columns: repeat(auto-fill, minmax({preview_w}px, 1fr)); }}
  .card {{ background: #fff; border-radius: 12px; padding: 12px; }}
  .head {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
           font-size: 13px; margin-bottom: 8px; }}
  .head b {{ font-family: ui-monospace, monospace; font-size: 14px; }}
  .swatch {{ width: 22px; height: 22px; border-radius: 5px; display: inline-block;
             border: 1px solid rgba(128,128,128,.5); vertical-align: middle; }}
  .views {{ display: flex; flex-direction: column; gap: 8px; }}
  .frame {{ position: relative; line-height: 0; background: #000;
            border-radius: 8px; overflow: hidden; }}
  .frame img {{ width: 100%; height: {preview_h}px; object-fit: contain; }}
  .frame i {{ position: absolute; border: 2px solid #ff3b30;
              box-shadow: 0 0 0 1px rgba(255,255,255,.7); }}
  .crop {{ position: relative; overflow: hidden; height: {crop_h}px;
           background: #000; border-radius: 8px; }}
  .crop img {{ position: absolute; max-width: none; }}
  .flags {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0 4px; }}
  .flag {{ font-size: 11px; padding: 3px 8px; border-radius: 999px;
           background: #ffd60a; color: #3a2f00; }}
  .note {{ font-size: 12px; color: #666; margin: 4px 0 0; }}
  @media (prefers-color-scheme: dark) {{ .note {{ color: #9a9aa2; }} }}
  code {{ font-size: 12px; }}
</style></head><body>
<header>
  <h1>{title}</h1>
  <p><b>{count}</b> candidate image(s) from <code>{root}</code>.
     Images are referenced in place; nothing was copied or imported.</p>
</header>
<div class="panel warn">{intro}</div>
<div class="grid">
{cards}
</div>
</body></html>
"""


RARE_INTRO: Final[str] = """
<b>These are prompts to look, not labels.</b>
Every image here has an annotation box that is square-ish or of ambiguous
proportion. That is <i>not</i> evidence of a two-line plate:
<ul>
  <li>a one-line plate photographed at a steep angle foreshortens towards
      square, and this dataset looks angle-heavy;</li>
  <li>a square bounding box is not a square plate.</li>
</ul>
<b>type1a</b> requires an actual Russian white two-line plate layout &mdash;
two rows of characters on the plate itself. Judge from the close-up, and record
your verdict in <code>rare_candidates.csv</code> under
<code>human_plate_type</code>.
"""

YELLOW_INTRO: Final[str] = """
<b>Colour was measured on the plate region only.</b>
Each plate's mean colour comes from the JPEG's DC coefficients, sampled inside
the annotation box and compared against a ring around it, so a yellow car with
a white plate is rejected rather than surfaced.
<ul>
  <li><b>type1b</b> requires the <i>plate itself</i> to be yellow, Russian, and
      one-line. A yellow vehicle, yellow background or yellow light does not
      qualify.</li>
  <li>Entries marked <code>most_yellow_in_dataset</code> did <b>not</b> clear
      the threshold. They are shown so you can see the tail and judge whether
      the heuristic is set sensibly.</li>
  <li>A mean colour is an average over the whole plate, black characters
      included, so it reads duller than the plate looks.</li>
</ul>
"""
