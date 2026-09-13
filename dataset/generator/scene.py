"""Procedural scenes: backgrounds and the vehicle panel a plate is mounted on.

Nothing here loads an image.  Backgrounds are gradients, smooth noise and
simple shapes; the "vehicle" is a body-coloured panel with a bumper, lamps and
an optional plate frame, drawn in the plate's own plane so it shares the
plate's perspective exactly.

:class:`BackgroundProvider` is the extension point: a provider serving
licensed real vehicle photographs can replace :class:`ProceduralBackground`
later without touching the rest of the pipeline.

Body colours are drawn from one palette for every plate type -- including
yellow cars carrying white plates -- so body colour can never become a
shortcut for the ``type1b`` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .render import RenderedPlate, smooth_noise

BODY_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.93, 0.93, 0.92),  # white
    (0.72, 0.73, 0.75),  # silver
    (0.45, 0.46, 0.48),  # grey
    (0.08, 0.08, 0.09),  # black
    (0.55, 0.06, 0.07),  # dark red
    (0.78, 0.12, 0.10),  # red
    (0.10, 0.20, 0.45),  # blue
    (0.12, 0.30, 0.18),  # green
    (0.90, 0.75, 0.12),  # yellow
    (0.55, 0.45, 0.32),  # beige / bronze
)


class BackgroundProvider(Protocol):
    name: str

    def render(self, rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
        """An ``H x W x 3`` float32 background and a record of how it was made."""
        ...


class ProceduralBackground:
    """Street, wall or car-park style backgrounds built from primitives."""

    name = "procedural"

    def render(self, rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
        width, height = size
        kind = ("street", "wall", "parking")[int(rng.integers(3))]
        ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]

        if kind == "street":
            horizon = float(rng.uniform(0.25, 0.55))
            sky_top = rng.uniform([0.35, 0.45, 0.60], [0.75, 0.82, 0.95]).astype(np.float32)
            sky_low = np.clip(sky_top + rng.uniform(0.05, 0.2), 0, 1).astype(np.float32)
            ground = rng.uniform(0.18, 0.45) * np.ones(3, np.float32) + rng.uniform(-0.03, 0.03, 3).astype(np.float32)
            t = np.clip(ys / max(horizon, 1e-3), 0, 1)
            sky = sky_top + (sky_low - sky_top) * t
            image = np.where(ys < horizon, sky, ground).astype(np.float32) * np.ones((1, width, 1), np.float32)
            image = self._blocks(rng, image, horizon)
        elif kind == "wall":
            grey = rng.uniform(0.28, 0.75)
            colour = np.clip(grey + rng.uniform(-0.08, 0.08, 3), 0, 1).astype(np.float32)
            image = np.ones((height, width, 3), np.float32) * colour
            image *= (0.85 + 0.3 * smooth_noise(rng, height, width, (6, 4)))[..., None]
            image = self._blocks(rng, image, 0.0, count=int(rng.integers(0, 4)))
        else:  # parking
            asphalt = rng.uniform(0.2, 0.42)
            image = np.ones((height, width, 3), np.float32) * asphalt
            image = self._parking_lines(rng, image)

        texture = smooth_noise(rng, height, width, (int(rng.integers(40, 90)), int(rng.integers(30, 70))))
        image *= (0.92 + 0.16 * texture)[..., None]
        grain = rng.normal(0.0, 0.015, size=(height, width, 1)).astype(np.float32)
        image = np.clip(image + grain, 0.0, 1.0)
        blur = float(rng.uniform(0.0, 1.5))
        if blur > 0.3:
            image = _gaussian(image, blur)
        return image.astype(np.float32), {"kind": kind, "blur": round(blur, 3)}

    @staticmethod
    def _blocks(rng: np.random.Generator, image: np.ndarray, horizon: float, count: int | None = None) -> np.ndarray:
        height, width, _ = image.shape
        canvas = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        count = int(rng.integers(3, 9)) if count is None else count
        for _ in range(count):
            w = int(rng.uniform(0.08, 0.4) * width)
            h = int(rng.uniform(0.1, 0.5) * height)
            x = int(rng.uniform(-0.1, 1.0) * width)
            y = int(max(0.0, horizon * height - h * rng.uniform(0.5, 1.0)))
            colour = tuple(int(c) for c in rng.integers(40, 220, 3))
            draw.rectangle((x, y, x + w, y + h), fill=colour)
            if rng.random() < 0.5:  # windows / signage stripes: text-like distractors
                stripe = tuple(int(c) for c in rng.integers(20, 250, 3))
                for row in range(int(rng.integers(1, 5))):
                    sy = y + int((row + 0.5) * h / 5)
                    draw.rectangle((x + w // 8, sy, x + w - w // 8, sy + max(2, h // 20)), fill=stripe)
        return np.asarray(canvas, dtype=np.float32) / 255.0

    @staticmethod
    def _parking_lines(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
        height, width, _ = image.shape
        canvas = Image.fromarray((image * 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas)
        paint = tuple(int(c) for c in rng.integers(180, 240, 3))
        skew = rng.uniform(-0.4, 0.4) * width
        for i in range(int(rng.integers(3, 8))):
            x = int((i + rng.uniform(0, 0.5)) * width / 5)
            draw.line((x, height, x + skew, 0), fill=paint, width=int(rng.integers(3, 9)))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def _gaussian(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


@dataclass
class PlaneCanvas:
    """The plate plane: vehicle panel with the plate pasted in.

    ``plate_origin`` is the plate's top-left in canvas pixels (integers), so
    the plate's corners are known exactly.  ``plate_alpha`` covers only the
    plate, for photometric effects that treat the plate specially.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    plate_alpha: np.ndarray
    plate_origin: tuple[int, int]
    plate_size: tuple[int, int]
    record: dict

    @property
    def plate_corners(self) -> np.ndarray:
        x, y = self.plate_origin
        w, h = self.plate_size
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def _rgb8(colour: np.ndarray | tuple[float, ...], factor: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 1) * 255) for c in colour)  # type: ignore[return-value]


def build_vehicle_panel(rng: np.random.Generator, plate: RenderedPlate, plate_type: str) -> PlaneCanvas:
    """Draw a car's rear or front in the plate's plane and mount the plate.

    Proportions follow a passenger car (about 1.8 m wide, 1.1-1.5 m tall)
    relative to the plate's physical size, so the plate sits where it would
    on a vehicle: low, centred, below a window and between the lamps.
    """
    pw, ph = plate.width, plate.height
    mm = pw / (290.0 if plate_type == "type1a" else 520.0)  # canvas px per mm
    panel_w = int(mm * rng.uniform(1650, 1950))
    panel_h = int(mm * rng.uniform(1150, 1500))
    ox = int(round((panel_w - pw) * rng.uniform(0.44, 0.56)))
    oy = int(round(panel_h - ph - mm * rng.uniform(250, 480)))

    body = np.asarray(BODY_PALETTE[int(rng.integers(len(BODY_PALETTE)))], np.float32)
    body = np.clip(body * rng.uniform(0.85, 1.1) + rng.uniform(-0.03, 0.03, 3), 0, 1).astype(np.float32)
    rear = bool(rng.random() < 0.6)

    canvas = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    shape = Image.new("L", (panel_w, panel_h), 0)
    shape_draw = ImageDraw.Draw(shape)

    # Silhouette: lower body plus a narrower cabin (roof and pillars).
    belt = int(panel_h * rng.uniform(0.40, 0.52))
    roof = int(panel_h * rng.uniform(0.0, 0.06))
    inset = int(panel_w * rng.uniform(0.10, 0.18))
    radius = int(mm * 120)
    wheel_h = int(mm * rng.uniform(90, 160))
    body_bottom = panel_h - wheel_h
    for d in (draw, shape_draw):
        fill = _rgb8(body) if d is draw else 255
        d.rounded_rectangle((0, belt - radius, panel_w - 1, body_bottom), radius=radius, fill=fill)
        d.polygon([(inset, roof), (panel_w - inset, roof), (panel_w - 1, belt), (0, belt)], fill=fill)
    # Tyres below the body.
    tyre_w = int(mm * rng.uniform(190, 240))
    for x in (int(mm * 90), panel_w - int(mm * 90) - tyre_w):
        for d in (draw, shape_draw):
            d.rectangle((x, body_bottom - wheel_h, x + tyre_w, panel_h - 1), fill=(18, 18, 20) if d is draw else 255)
    # Window glass.
    glass = int(mm * rng.uniform(55, 90))
    window = [
        (inset + glass, roof + glass),
        (panel_w - inset - glass, roof + glass),
        (panel_w - glass, belt - glass // 2),
        (glass, belt - glass // 2),
    ]
    tone = rng.uniform(0.06, 0.28)
    draw.polygon(window, fill=_rgb8((tone, tone * 1.05, tone * 1.15)))

    # Bumper band around the plate: body colour or black plastic.
    bumper = body * 0.8 if rng.random() < 0.5 else np.array([0.10, 0.10, 0.11], np.float32)
    band_top = int(oy - mm * rng.uniform(40, 140))
    band_bottom = min(body_bottom, int(oy + ph + mm * rng.uniform(60, 200)))
    draw.rectangle((0, band_top, panel_w, band_bottom), fill=_rgb8(bumper))
    # Plate recess.
    recess = int(mm * rng.uniform(5, 20))
    draw.rectangle((ox - recess * 2, oy - recess, ox + pw + recess * 2, oy + ph + recess), fill=_rgb8(bumper, 0.75))
    # Lamps at the corners, just under the belt line.
    lamp_colour = (190, 30, 25) if rear else (225, 225, 215)
    lamp_w, lamp_h = int(panel_w * rng.uniform(0.13, 0.2)), int(mm * rng.uniform(80, 160))
    lamp_y = int(belt + mm * rng.uniform(20, 80))
    for x in (int(panel_w * 0.03), int(panel_w * 0.97) - lamp_w):
        draw.rounded_rectangle((x, lamp_y, x + lamp_w, lamp_y + lamp_h), radius=max(1, lamp_h // 3), fill=lamp_colour)
    if not rear:  # grille between the headlamps
        gx0, gx1 = int(panel_w * 0.28), int(panel_w * 0.72)
        gy0, gy1 = lamp_y, max(lamp_y + 2, band_top - int(mm * 15))
        draw.rectangle((gx0, gy0, gx1, gy1), fill=(25, 25, 28))
        for gy in range(gy0 + 3, gy1, max(3, (gy1 - gy0) // 5)):
            draw.line((gx0, gy, gx1, gy), fill=(70, 70, 75), width=1)
    unit = ph
    # Plate frame / holder, drawn before the plate.
    frame = rng.random() < 0.55
    if frame:
        thickness = max(1, int(unit * rng.uniform(0.05, 0.14)))
        frame_colour = (15, 15, 17) if rng.random() < 0.8 else (170, 170, 175)
        draw.rounded_rectangle(
            (ox - thickness, oy - thickness, ox + pw + thickness, oy + ph + thickness),
            radius=thickness, fill=frame_colour,
        )

    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    shade = 0.8 + 0.35 * smooth_noise(rng, panel_h, panel_w, (3, 4))
    gradient = np.linspace(1.08, 0.82, panel_h, dtype=np.float32)[:, None]
    rgb = np.clip(rgb * (shade * gradient)[..., None], 0.0, 1.0)

    region = rgb[oy : oy + ph, ox : ox + pw]
    a = plate.alpha[..., None]
    rgb[oy : oy + ph, ox : ox + pw] = plate.rgb * a + region * (1.0 - a)

    alpha = np.asarray(shape, dtype=np.float32) / 255.0
    plate_alpha = np.zeros((panel_h, panel_w), np.float32)
    plate_alpha[oy : oy + ph, ox : ox + pw] = plate.alpha
    return PlaneCanvas(
        rgb=rgb.astype(np.float32),
        alpha=alpha,
        plate_alpha=plate_alpha,
        plate_origin=(ox, oy),
        plate_size=(pw, ph),
        record={"body_rgb": [round(float(c), 3) for c in body], "rear": rear, "frame": frame},
    )


def apply_occlusion(
    canvas: PlaneCanvas,
    glyph_boxes: list[tuple[int, str, float, float, float, float]],
    rng: np.random.Generator,
    extent: float,
    *,
    hidden_threshold: float = 0.35,
) -> tuple[set[int], dict]:
    """Cover part of the plate with an opaque object; return hidden positions.

    A character counts as hidden when more than ``hidden_threshold`` of its box
    is covered.  Hidden characters are labelled ``#`` -- the annotation guide's
    rule applied exactly, since here we *know* what is covered.
    """
    ox, oy = canvas.plate_origin
    pw, ph = canvas.plate_size
    height, width = canvas.alpha.shape
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    kind = ("bar", "blob", "corner")[int(rng.integers(3))]
    span = extent * pw
    if kind == "bar":  # tow bar / bike-rack upright crossing the plate
        cx = ox + rng.uniform(0.1, 0.9) * pw
        draw.rectangle((cx - span / 2, oy - ph, cx + span / 2, oy + 2 * ph), fill=255)
        colour = np.array([0.08, 0.08, 0.09]) * rng.uniform(0.8, 2.5)
    elif kind == "blob":  # snow or mud clump
        cx = ox + rng.uniform(0.0, 1.0) * pw
        cy = oy + rng.uniform(0.2, 0.9) * ph
        r = span / 2
        draw.ellipse((cx - r, cy - r * rng.uniform(0.6, 1.0), cx + r, cy + r * rng.uniform(0.6, 1.0)), fill=255)
        colour = np.array([0.92, 0.93, 0.95]) if rng.random() < 0.5 else np.array([0.30, 0.25, 0.18])
    else:  # corner covered by another object
        right = rng.random() < 0.5
        x_edge = ox + pw if right else ox
        x_in = x_edge - span if right else x_edge + span
        y_in = oy + ph * rng.uniform(0.3, 0.7)
        outside = x_edge + (ph if right else -ph)  # extend past the plate edge
        polygon = [(outside, oy + ph * 1.5), (outside, y_in), (x_edge, y_in), (x_in, oy + ph * 1.5)]
        draw.polygon(polygon, fill=255)
        colour = np.array([0.15, 0.15, 0.16]) * rng.uniform(0.7, 3.0)
    occluder = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255.0
    colour = np.clip(colour, 0, 1).astype(np.float32)
    canvas.rgb += occluder[..., None] * (colour - canvas.rgb)
    canvas.alpha = np.maximum(canvas.alpha, occluder)
    canvas.plate_alpha *= 1.0 - occluder

    hidden: set[int] = set()
    for position, _char, x0, y0, x1, y1 in glyph_boxes:
        xa, xb = int(ox + x0), int(np.ceil(ox + x1))
        ya, yb = int(oy + y0), int(np.ceil(oy + y1))
        covered = float(occluder[ya:yb, xa:xb].mean()) if xb > xa and yb > ya else 0.0
        if covered > hidden_threshold:
            hidden.add(position)
    return hidden, {"kind": kind, "extent": round(extent, 4), "hidden_positions": sorted(hidden)}
