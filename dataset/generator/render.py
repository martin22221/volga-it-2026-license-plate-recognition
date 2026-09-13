"""Plate renderer: a :class:`~.templates.PlateLayout` to an RGBA plate image.

Rendering happens at ``supersample`` times the target resolution and is then
box-filtered down, which antialiases the strokes without any font
rasteriser.  The plate occupies the canvas exactly: its outer corners are the
canvas corners ``(0, 0)``, ``(W, 0)``, ``(W, H)``, ``(0, H)`` in continuous
pixel coordinates.  Downstream geometry relies on that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .fonts import FontProvider
from .templates import PlateLayout

#: Canvas side limit at supersampled resolution.
MAX_SUPERSAMPLED_SIDE = 4096

FLAG_WHITE = (0.97, 0.97, 0.97)
FLAG_BLUE = (0.00, 0.22, 0.65)
FLAG_RED = (0.84, 0.17, 0.12)


@dataclass(frozen=True)
class PlateStyle:
    """Per-sample appearance of a plate (colours are linear-ish RGB in [0, 1])."""

    field_rgb: tuple[float, float, float]
    ink_rgb: tuple[float, float, float]
    stroke_scale: float
    emboss: float
    #: Retroreflective sheeting: a faint diagonal sheen and fine grain.
    sheen: float = 0.0
    grain: float = 0.0
    #: Mounting bolts: ``None`` for none, else the bolt head colour.
    bolt_rgb: tuple[float, float, float] | None = None


@dataclass
class RenderedPlate:
    """A rendered plate and where its characters are, in canvas pixels."""

    rgb: np.ndarray  # H x W x 3, float32
    alpha: np.ndarray  # H x W, float32: plate outline (rounded corners)
    ink: np.ndarray  # H x W, float32: characters, border, separator
    glyph_boxes: list[tuple[int, str, float, float, float, float]]  # position, char, x0, y0, x1, y1
    width: int
    height: int


def sample_style(rng: np.random.Generator, field_colour: str) -> PlateStyle:
    """Draw a plausible, slightly worn plate appearance."""
    if field_colour == "yellow":
        # Retroreflective yellow: red high, green well below red, blue low.
        field = (
            float(rng.uniform(0.90, 1.00)),
            float(rng.uniform(0.70, 0.84)),
            float(rng.uniform(0.00, 0.14)),
        )
    elif field_colour == "white":
        base = float(rng.uniform(0.88, 0.99))
        tint = rng.uniform(-0.02, 0.02, size=3)
        field = tuple(float(np.clip(base + t, 0.0, 1.0)) for t in tint)  # type: ignore[assignment]
    else:  # pragma: no cover - guarded by templates
        raise ValueError(f"unknown field colour {field_colour!r}")
    ink_level = float(rng.uniform(0.03, 0.12))
    bolt_draw = rng.random()
    bolt_rgb: tuple[float, float, float] | None
    if bolt_draw < 0.35:
        bolt_rgb = None
    elif bolt_draw < 0.65:
        tone = float(rng.uniform(0.55, 0.8))  # zinc / chrome
        bolt_rgb = (tone, tone, tone * 1.02)
    elif bolt_draw < 0.85:
        tone = float(rng.uniform(0.05, 0.15))  # black plastic cap
        bolt_rgb = (tone, tone, tone)
    else:
        bolt_rgb = tuple(float(c) * 0.92 for c in field)  # type: ignore[assignment]  # field-coloured cap
    return PlateStyle(
        field_rgb=field,  # type: ignore[arg-type]
        ink_rgb=(ink_level, ink_level, ink_level * float(rng.uniform(0.9, 1.1))),
        stroke_scale=float(rng.uniform(0.94, 1.06)),
        emboss=float(rng.uniform(0.03, 0.14)),
        sheen=float(rng.uniform(0.0, 0.08)),
        grain=float(rng.uniform(0.004, 0.018)),
        bolt_rgb=bolt_rgb,
    )


def _supersample(width: int, height: int, requested: int) -> int:
    factor = max(1, requested)
    while factor > 1 and max(width, height) * factor > MAX_SUPERSAMPLED_SIDE:
        factor -= 1
    return factor


def _to_float(mask: Image.Image, factor: int) -> np.ndarray:
    if factor > 1:
        mask = mask.reduce(factor)
    return np.asarray(mask, dtype=np.float32) / 255.0


def render_plate(
    layout: PlateLayout,
    style: PlateStyle,
    font: FontProvider,
    px_per_mm: float,
    *,
    supersample: int = 4,
    rng: np.random.Generator | None = None,
) -> RenderedPlate:
    """Render ``layout`` at ``px_per_mm`` (the plate's own resolution)."""
    width = max(16, int(round(layout.width * px_per_mm)))
    height = max(8, int(round(layout.height * px_per_mm)))
    factor = _supersample(width, height, supersample)
    big = (width * factor, height * factor)
    sx = big[0] / layout.width  # supersampled px per mm, horizontally
    sy = big[1] / layout.height

    def box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        return (x * sx, y * sy, (x + w) * sx, (y + h) * sy)

    radius = layout.corner_radius * sx
    shape = Image.new("L", big, 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, big[0] - 1, big[1] - 1), radius=radius, fill=255)

    ink = Image.new("L", big, 0)
    draw = ImageDraw.Draw(ink)
    inset = layout.border_inset
    draw.rounded_rectangle(
        box(inset, inset, layout.width - 2 * inset, layout.height - 2 * inset),
        radius=max(0.0, (layout.corner_radius - inset) * sx),
        outline=255,
        width=max(1, int(round(layout.border_width * sx))),
    )
    for rect in layout.separators:
        draw.rectangle(box(rect.x, rect.y, rect.width, rect.height), fill=255)

    glyph_boxes: list[tuple[int, str, float, float, float, float]] = []
    for glyph in layout.glyphs:
        stroke_px = glyph.stroke * style.stroke_scale * sx
        font.draw(ink, glyph.char, box(glyph.x, glyph.y, glyph.width, glyph.height), stroke_px)
        if glyph.position is not None:
            glyph_boxes.append(
                (
                    glyph.position,
                    glyph.char,
                    glyph.x * width / layout.width,
                    glyph.y * height / layout.height,
                    glyph.x1 * width / layout.width,
                    glyph.y1 * height / layout.height,
                )
            )

    flag_layers: list[tuple[np.ndarray, tuple[float, float, float]]] = []
    if layout.flag is not None:
        flag = layout.flag
        stripe = flag.height / 3.0
        for index, colour in enumerate((FLAG_WHITE, FLAG_BLUE, FLAG_RED)):
            mask = Image.new("L", big, 0)
            ImageDraw.Draw(mask).rectangle(box(flag.x, flag.y + index * stripe, flag.width, stripe), fill=255)
            flag_layers.append((_to_float(mask, factor), colour))
        # thin outline around the flag
        draw.rectangle(box(flag.x, flag.y, flag.width, flag.height), outline=255, width=max(1, int(round(0.8 * sx))))

    bolt_layers: list[tuple[np.ndarray, np.ndarray]] = []
    if style.bolt_rgb is not None and layout.bolt_sites:
        head = Image.new("L", big, 0)
        rim = Image.new("L", big, 0)
        for bx, by, br in layout.bolt_sites:
            cx, cy, r = bx * sx, by * sy, br * sx
            ImageDraw.Draw(rim).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
            inner = r * 0.78
            ImageDraw.Draw(head).ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=255)
        colour = np.asarray(style.bolt_rgb, dtype=np.float32)
        bolt_layers = [(_to_float(rim, factor), colour * 0.55), (_to_float(head, factor), colour)]

    alpha = _to_float(shape, factor)
    ink_mask = _to_float(ink, factor)

    rgb = np.empty((height, width, 3), dtype=np.float32)
    rgb[...] = np.asarray(style.field_rgb, dtype=np.float32)
    if style.sheen > 0 or style.grain > 0:
        # Retroreflective sheeting: a soft diagonal sheen, darker towards the
        # embossed rim, plus fine grain.
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        diagonal = (xs / max(width, 1) + ys / max(height, 1)) / 2.0
        rgb *= (1.0 + style.sheen * (0.5 - diagonal))[..., None]
        edge = np.minimum(np.minimum(xs, width - 1 - xs) / max(width, 1), np.minimum(ys, height - 1 - ys) / max(height, 1))
        rgb *= (1.0 - 0.06 * np.exp(-edge * 60.0))[..., None]
        if rng is not None and style.grain > 0:
            rgb += rng.normal(0.0, style.grain, size=(height, width, 1)).astype(np.float32)
    for mask, colour in flag_layers:
        rgb += mask[..., None] * (np.asarray(colour, dtype=np.float32) - rgb)
    if style.emboss > 0:
        # Characters are embossed: a faint light edge up-left, dark edge down-right.
        shift = max(1, int(round(0.004 * width)))
        lit = np.roll(ink_mask, (-shift, -shift), axis=(0, 1))
        shade = np.roll(ink_mask, (shift, shift), axis=(0, 1))
        rgb *= (1.0 - style.emboss * np.clip(shade - ink_mask, 0, 1))[..., None]
        rgb += style.emboss * np.clip(lit - ink_mask, 0, 1)[..., None] * (1.0 - rgb)
    rgb += ink_mask[..., None] * (np.asarray(style.ink_rgb, dtype=np.float32) - rgb)
    for mask, colour in bolt_layers:
        rgb += mask[..., None] * (colour - rgb)

    return RenderedPlate(
        rgb=np.clip(rgb, 0.0, 1.0),
        alpha=alpha,
        ink=ink_mask,
        glyph_boxes=glyph_boxes,
        width=width,
        height=height,
    )


def smooth_noise(rng: np.random.Generator, height: int, width: int, cells: tuple[int, int]) -> np.ndarray:
    """Low-frequency noise in [0, 1]: a random grid upsampled bicubically."""
    grid = rng.random((max(2, cells[1]), max(2, cells[0]))).astype(np.float32)
    image = Image.fromarray(grid).resize((width, height), Image.BICUBIC)  # float32 -> mode "F"
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def apply_dirt(plate: RenderedPlate, rng: np.random.Generator, strength: float) -> dict[str, float]:
    """Mud / road-salt blotches and faded ink, in plate space.

    Semi-transparent by construction: it lowers contrast but leaves every
    character legible, so the label stays complete.  Opaque coverage is the
    job of occlusion, which relabels hidden characters as ``#``.
    """
    h, w = plate.alpha.shape
    blotch = smooth_noise(rng, h, w, (int(rng.integers(4, 9)), int(rng.integers(2, 4))))
    speckle = smooth_noise(rng, h, w, (int(rng.integers(20, 40)), int(rng.integers(6, 12))))
    coverage = np.clip((blotch * 0.7 + speckle * 0.3 - (1.0 - strength)) / max(strength, 1e-3), 0.0, 1.0)
    coverage *= 0.2 + 0.6 * strength  # never opaque: at most ~0.6 at full strength
    tone = rng.uniform(0.25, 0.55)
    dirt_rgb = np.asarray([tone * 1.05, tone * 0.92, tone * 0.72], dtype=np.float32)
    plate.rgb += coverage[..., None] * (dirt_rgb - plate.rgb)
    # Faded ink: characters lose some of their contrast.
    fade = float(rng.uniform(0.0, 0.15 * strength))
    plate.rgb += (fade * plate.ink)[..., None] * (0.55 - plate.rgb)
    np.clip(plate.rgb, 0.0, 1.0, out=plate.rgb)
    return {"strength": round(strength, 4), "tone": round(float(tone), 4), "fade": round(fade, 4)}
