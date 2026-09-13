"""Self-contained HTML contact sheet for human review of a generated batch.

For every image the sheet shows the frame with the annotated quad drawn over
it, a crop *rectified from that quad* (so a wrong quad is immediately visible
as a skewed or misplaced crop), and the label, type, difficulty, seed,
conditions and effects.  Images are embedded as data URIs, so the single HTML
file can be opened or shared anywhere.

    python -m dataset.generator.contact_sheet <batch directory>
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .geometry import homography, pil_perspective_coefficients
from .templates import PLATE_SIZE_MM
from .writer import IMAGE_DIR, MANIFEST_NAME, META_NAME, RECORDS_NAME, REVIEW_DIR

CROP_HEIGHT = 90
THUMB_WIDTH = 520


def rectified_crop(image: Image.Image, quad: Sequence[tuple[float, float]], plate_type: str) -> Image.Image:
    """Undo the perspective using the annotated quad only."""
    width_mm, height_mm = PLATE_SIZE_MM.get(plate_type, (520.0, 112.0))
    crop_h = CROP_HEIGHT
    crop_w = int(round(crop_h * width_mm / height_mm))
    target = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], dtype=np.float64)
    matrix = homography(target, np.asarray(quad, dtype=np.float64))  # crop -> image
    coefficients = pil_perspective_coefficients(np.linalg.inv(matrix))  # PIL wants output -> input
    return image.transform((crop_w, crop_h), Image.PERSPECTIVE, coefficients, Image.BICUBIC)


def _data_uri(image: Image.Image, quality: int = 88) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_rows(root: Path) -> list[dict[str, str]]:
    with (root / META_NAME).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _read_records(root: Path) -> dict[str, dict]:
    records = {}
    for line in (root / RECORDS_NAME).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["image"]] = record
    return records


CSS = """
:root { --bg:#f4f4f2; --card:#fff; --ink:#1d1d1f; --muted:#666; --line:#ddd; --accent:#0a7d32; }
@media (prefers-color-scheme: dark) { :root { --bg:#161617; --card:#222224; --ink:#eee; --muted:#9a9a9a; --line:#3a3a3c; --accent:#3ddc84; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,sans-serif; }
h1 { font-size:20px; margin:0 0 4px; }
.meta { color:var(--muted); margin-bottom:12px; }
.filters { display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 16px; }
.filters button { border:1px solid var(--line); background:var(--card); color:var(--ink); padding:4px 10px; border-radius:14px; cursor:pointer; }
.filters button.on { background:var(--ink); color:var(--bg); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,360px),1fr)); gap:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; }
.frame { position:relative; }
.frame img, .frame svg { width:100%; display:block; }
.frame svg { position:absolute; inset:0; height:100%; }
.crop { margin-top:8px; background:#777; display:flex; justify-content:center; padding:4px; border-radius:4px; }
.crop img { max-width:100%; height:auto; image-rendering:auto; }
.plate { font:600 18px/1.2 ui-monospace,Consolas,monospace; letter-spacing:1px; margin-top:8px; }
.tags { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
.tag { font-size:12px; border:1px solid var(--line); border-radius:10px; padding:1px 8px; }
.t-type1 { border-color:#888; } .t-type1a { border-color:#2b6cb0; } .t-type1b { border-color:#c9a000; }
.small { color:var(--muted); font-size:12px; margin-top:4px; word-break:break-word; }
.stats { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; display:inline-block; }
.stats td { padding:1px 10px 1px 0; }
"""

JS = """
const state = {type: 'all', difficulty: 'all'};
function apply() {
  document.querySelectorAll('.card').forEach(c => {
    const ok = (state.type === 'all' || c.dataset.type === state.type) &&
               (state.difficulty === 'all' || c.dataset.difficulty === state.difficulty);
    c.hidden = !ok;
  });
}
document.querySelectorAll('.filters button').forEach(b => b.addEventListener('click', () => {
  state[b.dataset.key] = b.dataset.value;
  document.querySelectorAll(`.filters button[data-key="${b.dataset.key}"]`).forEach(x => x.classList.toggle('on', x === b));
  apply();
}));
"""


def build_contact_sheet(root: Path, output: Path | None = None) -> Path:
    """Write ``<root>/review/contact_sheet.html`` (or ``output``)."""
    root = Path(root)
    rows = _read_rows(root)
    records = _read_records(root)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    output = Path(output) if output is not None else root / REVIEW_DIR / "contact_sheet.html"
    output.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    for row in rows:
        record = records.get(row["image"], {})
        image = Image.open(root / row["image"]).convert("RGB")
        width, height = image.size
        quad = [(float(row[f"quad_x{i}"]), float(row[f"quad_y{i}"])) for i in range(1, 5)]
        crop = rectified_crop(image, quad, row["plate_type"])
        thumb = image.copy()
        thumb.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in quad)
        x1, y1 = quad[0]
        effects = ", ".join(sorted(record.get("effects", {}))) or "none"
        geometry = record.get("geometry", {})
        tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in row["conditions"].split("|") if t)
        plate_type = html.escape(row["plate_type"])
        cards.append(
            f"""<div class="card" data-type="{plate_type}" data-difficulty="{html.escape(record.get('difficulty', ''))}">
  <div class="frame"><img src="{_data_uri(thumb, 80)}" alt="{html.escape(row['image'])}">
    <svg viewBox="0 0 {width} {height}" preserveAspectRatio="none"><polygon points="{points}" fill="none" stroke="#00e676" stroke-width="{max(1.5, width / 400):.1f}"/>
    <circle cx="{x1:.1f}" cy="{y1:.1f}" r="{max(2.5, width / 250):.1f}" fill="#ff1744"/></svg></div>
  <div class="crop"><img src="{_data_uri(crop)}" alt="rectified plate crop"></div>
  <div class="plate">{html.escape(row['plate_num'])}</div>
  <div class="tags"><span class="tag t-{plate_type}">{plate_type}</span><span class="tag">{html.escape(record.get('difficulty', '?'))}</span>{tags}</div>
  <div class="small">#{record.get('index', '?')} &middot; {html.escape(row['image'])} &middot; {width}&times;{height} &middot; seed {record.get('sample_seed', '?')}</div>
  <div class="small">yaw {geometry.get('yaw_deg', 0):.1f}&deg; pitch {geometry.get('pitch_deg', 0):.1f}&deg; roll {geometry.get('roll_deg', 0):.1f}&deg; &middot; plate {float(row['bbox_w']):.0f}px wide &middot; effects: {html.escape(effects)} &middot; jpeg q{record.get('jpeg', {}).get('quality', '?')}</div>
</div>"""
        )

    counts = manifest.get("counts", {})

    def buttons(key: str, values: list[str]) -> str:
        items = [f'<button class="on" data-key="{key}" data-value="all">all {key}s</button>']
        items += [f'<button data-key="{key}" data-value="{v}">{v}</button>' for v in values]
        return "".join(items)

    stats = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(', '.join(f'{n} {c}' for n, c in v.items()))}</td></tr>"
        for k, v in counts.items()
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Synthetic plate review — seed {manifest.get('seed')}</title><style>{CSS}</style></head>
<body>
<h1>Synthetic plate generator {html.escape(str(manifest.get('generator_version')))} — human review</h1>
<div class="meta">{manifest.get('image_count')} images &middot; seed {manifest.get('seed')} &middot; all images are synthetic (is_synthetic=true, source {html.escape(str(manifest.get('source')))}).
Green outline = annotated quad (red dot = corner 1, plate top-left). The crop under each frame is rectified from that quad alone.</div>
<table class="stats">{stats}</table>
<div class="filters">{buttons('type', ['type1', 'type1a', 'type1b'])}</div>
<div class="filters">{buttons('difficulty', ['easy', 'medium', 'hard'])}</div>
<div class="grid">
{''.join(cards)}
</div>
<script>{JS}</script>
</body></html>
"""
    output.write_text(document, encoding="utf-8", newline="\n")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dataset.generator.contact_sheet", description=__doc__.splitlines()[0])
    parser.add_argument("batch", type=Path, help="directory written by the generator")
    parser.add_argument("--output", type=Path, default=None, help="HTML path (default: <batch>/review/contact_sheet.html)")
    args = parser.parse_args(argv)
    if not (args.batch / META_NAME).is_file() or not (args.batch / IMAGE_DIR).is_dir():
        print(f"error: {args.batch} is not a generated batch", file=sys.stderr)
        return 2
    print(build_contact_sheet(args.batch, args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
