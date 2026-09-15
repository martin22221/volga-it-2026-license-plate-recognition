"""Procedural scenes: backgrounds and the vehicle a plate is mounted on.

Nothing here loads an image.  Backgrounds are built from gradients, smooth
noise and simple shapes; vehicles are parametric silhouettes (sedan,
hatchback, SUV, van, estate) drawn *in the plate's plane*, so they share the
plate's perspective exactly.

:class:`BackgroundProvider` is the extension point: a provider serving
licensed real vehicle photographs can replace :class:`ProceduralBackground`
later without touching the rest of the pipeline.

**No shortcut for the classifier.**  Every scene and vehicle choice -- body
type, colour, view, mounting, frame, background -- is drawn from the same
distribution for every plate type.  Plate type influences the vehicle only
through the plate's physical size.  A test checks that ``type1`` and
``type1b`` plates get identical vehicles from the same random stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import FontProvider
from .occlusion_labels import OcclusionJudge
from .render import GlyphCell, RenderedPlate, smooth_noise

BODY_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.93, 0.93, 0.92),  # white
    (0.88, 0.88, 0.86),  # off-white
    (0.72, 0.73, 0.75),  # silver
    (0.45, 0.46, 0.48),  # grey
    (0.25, 0.26, 0.28),  # graphite
    (0.08, 0.08, 0.09),  # black
    (0.55, 0.06, 0.07),  # dark red
    (0.78, 0.12, 0.10),  # red
    (0.10, 0.20, 0.45),  # blue
    (0.35, 0.45, 0.60),  # light blue
    (0.12, 0.30, 0.18),  # green
    (0.90, 0.75, 0.12),  # yellow
    (0.55, 0.45, 0.32),  # beige / bronze
    (0.40, 0.25, 0.15),  # brown
)

#: (total height mm, cabin inset fraction, belt-line fraction from the top)
BODY_TYPES: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "sedan": ((1300, 1450), (0.13, 0.19), (0.46, 0.54)),
    "hatchback": ((1400, 1550), (0.11, 0.16), (0.42, 0.50)),
    "estate": ((1420, 1560), (0.08, 0.12), (0.44, 0.52)),
    "suv": ((1600, 1820), (0.06, 0.10), (0.40, 0.48)),
    "van": ((1850, 2200), (0.02, 0.05), (0.34, 0.44)),
}

#: Muted urban colours for backgrounds (walls, buildings, signage).
URBAN_PALETTE: tuple[tuple[float, float, float], ...] = (
    (0.62, 0.58, 0.52), (0.48, 0.45, 0.42), (0.70, 0.68, 0.64), (0.55, 0.36, 0.30),
    (0.40, 0.42, 0.45), (0.78, 0.74, 0.66), (0.33, 0.33, 0.35), (0.58, 0.60, 0.56),
    (0.66, 0.52, 0.40), (0.30, 0.36, 0.34),
)


def _rgb8(colour, factor: float = 1.0) -> tuple[int, int, int]:
    return tuple(int(np.clip(c * factor, 0, 1) * 255) for c in colour)  # type: ignore[return-value]


def _jitter(rng: np.random.Generator, colour, amount: float = 0.05) -> np.ndarray:
    return np.clip(np.asarray(colour, np.float32) * rng.uniform(1 - amount, 1 + amount) + rng.uniform(-amount / 2, amount / 2, 3), 0, 1).astype(np.float32)


def _gaussian(image: np.ndarray, radius: float) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


# ============================================================ backgrounds


class BackgroundProvider(Protocol):
    name: str

    def render(self, rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
        """An ``H x W x 3`` float32 background and a record of how it was made."""
        ...


class ProceduralBackground:
    """Street, facade, car-park, trees and garage scenes built from primitives."""

    name = "procedural-v2"
    KINDS = ("street", "facade", "parking", "trees", "garage")

    def render(self, rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, dict]:
        width, height = size
        kind = self.KINDS[int(rng.integers(len(self.KINDS)))]
        canvas = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        horizon = int(height * rng.uniform(0.28, 0.55))
        ground_y = horizon  # where vehicles may stand; set per kind below

        if kind in ("street", "trees", "parking"):
            self._sky(rng, canvas, horizon)
            ground = _jitter(rng, (0.32, 0.32, 0.33), 0.25)
            draw.rectangle((0, horizon, width, height), fill=_rgb8(ground))
            if kind == "street":
                self._buildings(rng, draw, width, horizon)
            elif kind == "trees":
                self._trees(rng, canvas, horizon)
            else:
                self._buildings(rng, draw, width, horizon, low=True)
            self._road_markings(rng, draw, width, height, horizon, parking=(kind == "parking"))
            if kind in ("parking", "street"):
                self._background_cars(rng, draw, width, height, horizon)
        elif kind == "facade":
            wall = _jitter(rng, URBAN_PALETTE[int(rng.integers(len(URBAN_PALETTE)))], 0.12)
            draw.rectangle((0, 0, width, height), fill=_rgb8(wall))
            self._windows(rng, draw, 0, 0, width, int(height * rng.uniform(0.6, 0.85)))
            base = int(height * rng.uniform(0.7, 0.88))
            ground_y = base
            draw.rectangle((0, base, width, height), fill=_rgb8(_jitter(rng, (0.3, 0.3, 0.31), 0.2)))
        else:  # garage / underground parking
            tone = rng.uniform(0.12, 0.3)
            draw.rectangle((0, 0, width, height), fill=_rgb8((tone, tone, tone * 1.05)))
            for _ in range(int(rng.integers(2, 6))):  # ceiling lights
                x = rng.uniform(0, width)
                draw.rectangle((x, height * 0.05, x + width * 0.08, height * 0.07), fill=(235, 235, 225))
            for i in range(int(rng.integers(2, 5))):  # pillars
                x = int(rng.uniform(0, width))
                draw.rectangle((x, 0, x + int(width * rng.uniform(0.03, 0.08)), height), fill=_rgb8((tone * 1.6,) * 3))
            ground_y = int(height * 0.45)
            self._road_markings(rng, draw, width, height, ground_y, parking=True)

        image = np.asarray(canvas, dtype=np.float32) / 255.0
        texture = smooth_noise(rng, height, width, (int(rng.integers(30, 80)), int(rng.integers(20, 60))))
        image *= (0.9 + 0.2 * texture)[..., None]
        shade = smooth_noise(rng, height, width, (3, 3))
        image *= (0.85 + 0.3 * shade)[..., None]
        image = np.clip(image + rng.normal(0.0, 0.012, size=(height, width, 1)).astype(np.float32), 0, 1)
        blur = float(rng.uniform(0.3, 2.2))  # backgrounds are rarely in focus
        image = _gaussian(image, blur)
        return image.astype(np.float32), {"kind": kind, "blur": round(blur, 3), "ground_y": int(ground_y)}

    @staticmethod
    def _sky(rng: np.random.Generator, canvas: Image.Image, horizon: int) -> None:
        width, _ = canvas.size
        top = _jitter(rng, (0.55, 0.65, 0.80), 0.25)
        low = np.clip(top + rng.uniform(0.05, 0.2), 0, 1)
        ys = np.linspace(0, 1, max(horizon, 1), dtype=np.float32)[:, None, None]
        sky = top + (low - top) * ys
        clouds = smooth_noise(rng, max(horizon, 1), width, (6, 3))
        sky = sky + (clouds[..., None] - 0.5) * rng.uniform(0.0, 0.3)
        canvas.paste(Image.fromarray((np.clip(np.broadcast_to(sky, (max(horizon, 1), width, 3)), 0, 1) * 255).astype(np.uint8)), (0, 0))

    def _buildings(self, rng: np.random.Generator, draw: ImageDraw.ImageDraw, width: int, horizon: int, low: bool = False) -> None:
        x = int(-rng.uniform(0, 0.2) * width)
        while x < width:
            w = int(rng.uniform(0.12, 0.35) * width)
            h = int(horizon * rng.uniform(0.2, 0.5 if low else 0.95))
            colour = _jitter(rng, URBAN_PALETTE[int(rng.integers(len(URBAN_PALETTE)))], 0.1)
            draw.rectangle((x, horizon - h, x + w, horizon), fill=_rgb8(colour))
            self._windows(rng, draw, x, horizon - h, x + w, horizon)
            x += w + int(rng.uniform(0, 0.05) * width)

    @staticmethod
    def _windows(rng: np.random.Generator, draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
        cols, rows = int(rng.integers(2, 7)), int(rng.integers(1, 6))
        cw, rh = (x1 - x0) / cols, (y1 - y0) / rows
        glass = _jitter(rng, (0.25, 0.3, 0.36), 0.3)
        for r in range(rows):
            for c in range(cols):
                gx, gy = x0 + c * cw + cw * 0.2, y0 + r * rh + rh * 0.2
                lit = rng.random() < 0.15
                draw.rectangle((gx, gy, gx + cw * 0.6, gy + rh * 0.55), fill=(210, 190, 140) if lit else _rgb8(glass))

    @staticmethod
    def _trees(rng: np.random.Generator, canvas: Image.Image, horizon: int) -> None:
        width, _ = canvas.size
        foliage = smooth_noise(rng, horizon, width, (int(rng.integers(8, 20)), 4))
        tree_line = np.linspace(0, 1, horizon, dtype=np.float32)[:, None] > (0.9 - foliage * 0.8)
        green = _jitter(rng, (0.18, 0.28, 0.14), 0.3)
        region = np.asarray(canvas, dtype=np.float32)[:horizon] / 255.0
        region[tree_line] = green * (0.7 + 0.5 * foliage[tree_line][:, None])
        canvas.paste(Image.fromarray((np.clip(region, 0, 1) * 255).astype(np.uint8)), (0, 0))

    @staticmethod
    def _road_markings(rng: np.random.Generator, draw: ImageDraw.ImageDraw, width: int, height: int, horizon: int, parking: bool) -> None:
        paint = _rgb8(_jitter(rng, (0.85, 0.85, 0.8), 0.1))
        vanish_x = rng.uniform(0.2, 0.8) * width
        lines = int(rng.integers(3, 8)) if parking else int(rng.integers(1, 4))
        for i in range(lines):
            bottom_x = (i + 0.5) / lines * width * 1.6 - width * 0.3
            t = rng.uniform(0.0, 0.35)
            top = (bottom_x + (vanish_x - bottom_x) * (1 - t), horizon + (height - horizon) * t)
            draw.line((bottom_x, height, *top), fill=paint, width=int(rng.integers(2, 6)))

    @staticmethod
    def _background_cars(rng: np.random.Generator, draw: ImageDraw.ImageDraw, width: int, height: int, horizon: int) -> None:
        """Distant plate-free vehicles: context, never an unlabelled plate."""
        for _ in range(int(rng.integers(0, 4))):
            w = int(width * rng.uniform(0.08, 0.2))
            h = int(w * rng.uniform(0.5, 0.8))
            x = int(rng.uniform(-0.1, 1.0) * width)
            y = int(horizon + (height - horizon) * rng.uniform(0.0, 0.25) - h)
            body = BODY_PALETTE[int(rng.integers(len(BODY_PALETTE)))]
            draw.rounded_rectangle((x, y + h * 0.35, x + w, y + h), radius=max(1, h // 6), fill=_rgb8(body))
            draw.polygon([(x + w * 0.18, y + h * 0.38), (x + w * 0.28, y), (x + w * 0.72, y), (x + w * 0.82, y + h * 0.38)], fill=_rgb8(body, 0.9))
            draw.polygon([(x + w * 0.24, y + h * 0.34), (x + w * 0.31, y + h * 0.06), (x + w * 0.69, y + h * 0.06), (x + w * 0.76, y + h * 0.34)], fill=(40, 44, 50))


# ============================================================ vehicle panel


@dataclass
class PlaneCanvas:
    """The plate plane: vehicle with the plate mounted.

    ``plate_origin`` is the plate's top-left in canvas pixels (integers), so
    the plate's corners are known exactly.  ``plate_alpha`` covers only the
    plate (minus anything occluding it) and ``ink`` its characters, for
    effects and measurements that treat the plate specially.
    """

    rgb: np.ndarray
    alpha: np.ndarray
    plate_alpha: np.ndarray
    ink: np.ndarray
    plate_origin: tuple[int, int]
    plate_size: tuple[int, int]
    record: dict

    @property
    def plate_corners(self) -> np.ndarray:
        x, y = self.plate_origin
        w, h = self.plate_size
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def build_vehicle_panel(rng: np.random.Generator, plate: RenderedPlate, plate_type: str) -> PlaneCanvas:
    """Draw a vehicle's rear or front in the plate's plane and mount the plate.

    Dimensions are physical (mm): the plate type only sets how many canvas
    pixels one millimetre is, so every plate type sees the same vehicles.
    """
    pw, ph = plate.width, plate.height
    mm = pw / (290.0 if plate_type == "type1a" else 520.0)  # canvas px per mm

    body_type = tuple(BODY_TYPES)[int(rng.integers(len(BODY_TYPES)))]
    height_range, inset_range, belt_range = BODY_TYPES[body_type]
    panel_w = int(mm * rng.uniform(1680, 1980))
    panel_h = int(mm * rng.uniform(*height_range))
    rear = bool(rng.random() < 0.65)
    body = _jitter(rng, BODY_PALETTE[int(rng.integers(len(BODY_PALETTE)))], 0.08)

    # Mounting: bumper (most common) or tailgate/boot lid (rear only).
    mount = "tailgate" if rear and rng.random() < 0.35 else "bumper"
    wheel_h = int(mm * rng.uniform(110, 170))
    plate_centre_above_ground = rng.uniform(650, 880) if mount == "tailgate" else rng.uniform(380, 560)
    oy = int(round(panel_h - mm * plate_centre_above_ground - ph / 2))
    oy = int(np.clip(oy, int(panel_h * 0.35), panel_h - wheel_h - ph - 2))
    ox = int(round((panel_w - pw) / 2 + mm * rng.uniform(-60, 60)))

    canvas = Image.new("RGB", (panel_w, panel_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    shape = Image.new("L", (panel_w, panel_h), 0)
    shape_draw = ImageDraw.Draw(shape)

    belt = int(panel_h * rng.uniform(*belt_range))
    inset = int(panel_w * rng.uniform(*inset_range))
    roof_radius = int(mm * rng.uniform(60, 200))
    body_radius = int(mm * rng.uniform(80, 160))
    body_bottom = panel_h - int(wheel_h * 0.55)
    for d, fill in ((draw, _rgb8(body)), (shape_draw, 255)):
        d.rounded_rectangle((0, belt - body_radius, panel_w - 1, body_bottom), radius=body_radius, fill=fill)
        d.polygon([(inset, roof_radius), (panel_w - inset, roof_radius), (panel_w - 1, belt), (0, belt)], fill=fill)
        d.rounded_rectangle((inset, 0, panel_w - inset, belt), radius=roof_radius, fill=fill)
    tyre_w = int(mm * rng.uniform(190, 245))
    for x in (int(mm * rng.uniform(80, 130)), panel_w - int(mm * rng.uniform(80, 130)) - tyre_w):
        for d, fill in ((draw, (20, 20, 22)), (shape_draw, 255)):
            d.rounded_rectangle((x, body_bottom - wheel_h, x + tyre_w, panel_h - 1), radius=max(1, tyre_w // 8), fill=fill)

    # Glass (vans sometimes have solid rear doors).
    glass_margin = int(mm * rng.uniform(50, 90))
    solid_doors = body_type == "van" and rear and rng.random() < 0.4
    window = [
        (inset + glass_margin, roof_radius // 2 + glass_margin),
        (panel_w - inset - glass_margin, roof_radius // 2 + glass_margin),
        (panel_w - glass_margin, belt - glass_margin // 2),
        (glass_margin, belt - glass_margin // 2),
    ]
    if solid_doors:
        draw.line((panel_w // 2, glass_margin, panel_w // 2, body_bottom - wheel_h), fill=_rgb8(body, 0.7), width=max(1, int(mm * 6)))
    else:
        tone = rng.uniform(0.05, 0.25)
        draw.polygon(window, fill=_rgb8((tone, tone * 1.05, tone * 1.18)))
        if rng.random() < 0.6:  # a sky reflection streak, placed and slanted at random
            reflection = rng.uniform(0.06, 0.25)
            wx0, wy0 = window[0]
            span = panel_w - 2 * wx0
            top_x = wx0 + span * rng.uniform(0.1, 0.8)
            width_top = span * rng.uniform(0.05, 0.2)
            slant = span * rng.uniform(-0.3, 0.3)
            bottom_y = belt - glass_margin // 2
            draw.polygon(
                [(top_x, wy0), (top_x + width_top, wy0), (top_x + width_top + slant, bottom_y), (top_x + slant, bottom_y)],
                fill=_rgb8((tone + reflection,) * 3),
            )

    # Bumper, plate recess and lamps.
    plastic = rng.random() < 0.5
    bumper = np.array([0.10, 0.10, 0.11], np.float32) if plastic else body * 0.9
    band_top = int(oy - mm * rng.uniform(60, 160)) if mount == "bumper" else int(panel_h - wheel_h - mm * rng.uniform(300, 420))
    draw.rectangle((0, band_top, panel_w, body_bottom - wheel_h // 3), fill=_rgb8(bumper))
    draw.rectangle((int(panel_w * 0.1), body_bottom - wheel_h // 3 - int(mm * 60), int(panel_w * 0.9), body_bottom - wheel_h // 3), fill=(28, 28, 30))
    if mount == "tailgate":  # boot-lid seam around the plate area
        seam = _rgb8(body, 0.72)
        draw.rounded_rectangle((int(panel_w * 0.2), belt - int(mm * 30), int(panel_w * 0.8), band_top - int(mm * 20)),
                               radius=int(mm * 40), outline=seam, width=max(1, int(mm * 4)))
    recess_colour = bumper * 0.7 if mount == "bumper" else body * 0.78
    recess = int(mm * rng.uniform(10, 30))
    draw.rounded_rectangle((ox - recess * 2, oy - recess, ox + pw + recess * 2, oy + ph + recess),
                           radius=max(1, recess), fill=_rgb8(recess_colour))
    lamp_w, lamp_h = int(panel_w * rng.uniform(0.12, 0.2)), int(mm * rng.uniform(90, 170))
    lamp_y = int(belt + mm * rng.uniform(10, 70))
    for x in (int(panel_w * 0.02), int(panel_w * 0.98) - lamp_w):
        if rear:
            draw.rounded_rectangle((x, lamp_y, x + lamp_w, lamp_y + lamp_h), radius=max(1, lamp_h // 3), fill=(170, 25, 22))
            draw.rounded_rectangle((x + lamp_w // 4, lamp_y + lamp_h // 3, x + lamp_w * 3 // 4, lamp_y + lamp_h * 2 // 3),
                                   radius=max(1, lamp_h // 6), fill=(225, 150, 60) if rng.random() < 0.5 else (230, 230, 225))
        else:
            draw.rounded_rectangle((x, lamp_y, x + lamp_w, lamp_y + lamp_h), radius=max(1, lamp_h // 3), fill=(205, 208, 210))
            r = lamp_h // 3
            cx, cy = x + lamp_w // 2, lamp_y + lamp_h // 2
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(245, 245, 240))
    if not rear:
        gx0, gx1 = int(panel_w * 0.3), int(panel_w * 0.7)
        gy0, gy1 = lamp_y, max(lamp_y + 2, min(oy - int(mm * 30), band_top))
        draw.rounded_rectangle((gx0, gy0, gx1, gy1), radius=int(mm * 20), fill=(22, 22, 25))
        for gy in range(gy0 + 3, gy1, max(3, (gy1 - gy0) // 6)):
            draw.line((gx0, gy, gx1, gy), fill=(62, 62, 68), width=max(1, int(mm * 5)))
    elif rng.random() < 0.6:  # exhaust
        ex = int(panel_w * rng.uniform(0.15, 0.3))
        r = int(mm * 35)
        ey = body_bottom - wheel_h // 3
        draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(35, 35, 38))

    frame_draw = rng.random()
    frame = "none" if frame_draw < 0.4 else ("black" if frame_draw < 0.85 else "chrome")
    if frame != "none":
        thickness = max(1, int(mm * rng.uniform(4, 12)))
        colour = (15, 15, 17) if frame == "black" else (175, 175, 180)
        draw.rounded_rectangle((ox - thickness, oy - thickness, ox + pw + thickness, oy + ph + thickness * 2),
                               radius=thickness, fill=colour)

    # Paint shading: vertical falloff, a specular band and soft noise.
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    ys = np.linspace(0.0, 1.0, panel_h, dtype=np.float32)[:, None]
    falloff = 1.08 - 0.28 * ys
    band_y = rng.uniform(0.45, 0.75)
    specular = rng.uniform(0.05, 0.22) * np.exp(-((ys - band_y) ** 2) / (2 * 0.03**2))
    shade = 0.88 + 0.24 * smooth_noise(rng, panel_h, panel_w, (3, 4))
    rgb = np.clip(rgb * (falloff * shade)[..., None] + specular[..., None], 0.0, 1.0)

    region = rgb[oy : oy + ph, ox : ox + pw]
    a = plate.alpha[..., None]
    mounted = plate.rgb.copy()
    if mount == "bumper" or rng.random() < 0.5:
        # The lip above a recessed plate casts a soft shadow on its top edge.
        depth = rng.uniform(0.05, 0.25)
        lip = np.exp(-np.linspace(0.0, 1.0, ph, dtype=np.float32) * rng.uniform(4, 10))[:, None, None]
        mounted *= 1.0 - depth * lip
    rgb[oy : oy + ph, ox : ox + pw] = mounted * a + region * (1.0 - a)

    alpha = np.asarray(shape, dtype=np.float32) / 255.0
    plate_alpha = np.zeros((panel_h, panel_w), np.float32)
    plate_alpha[oy : oy + ph, ox : ox + pw] = plate.alpha
    ink = np.zeros((panel_h, panel_w), np.float32)
    ink[oy : oy + ph, ox : ox + pw] = plate.ink * plate.alpha
    return PlaneCanvas(
        rgb=rgb.astype(np.float32),
        alpha=alpha,
        plate_alpha=plate_alpha,
        ink=ink,
        plate_origin=(ox, oy),
        plate_size=(pw, ph),
        record={
            "body_type": body_type,
            "body_rgb": [round(float(c), 3) for c in body],
            "rear": rear,
            "mount": mount,
            "frame": frame,
        },
    )


def _occluder_mask(canvas: PlaneCanvas, rng: np.random.Generator, extent: float) -> tuple[np.ndarray, str, np.ndarray]:
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
        outside = x_edge + (ph if right else -ph)
        draw.polygon([(outside, oy + ph * 1.5), (outside, y_in), (x_edge, y_in), (x_in, oy + ph * 1.5)], fill=255)
        colour = np.array([0.15, 0.15, 0.16]) * rng.uniform(0.7, 3.0)
    occluder = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32) / 255.0
    return occluder, kind, np.clip(colour, 0, 1).astype(np.float32)


def apply_occlusion(
    canvas: PlaneCanvas,
    glyph_cells: Sequence[GlyphCell],
    rng: np.random.Generator,
    extent: float,
    *,
    font: FontProvider,
    max_hidden: int = 2,
) -> tuple[set[int], dict]:
    """Cover part of the plate with an opaque object; return hidden positions.

    Whether a character stays readable is decided from its ink, not its box
    (:mod:`.occlusion_labels`): too much ink covered, identifying strokes
    covered, or a remainder that reads as another character makes it hidden,
    labelled ``#``.  An occluder hiding more than ``max_hidden`` characters
    is shrunk and redrawn, so occlusion stays a hard example rather than an
    unreadable one.  The occluder's random draws do not depend on the rule.
    """
    judge = OcclusionJudge(glyph_cells, font)
    attempts = 0
    while True:
        attempts += 1
        occluder, kind, colour = _occluder_mask(canvas, rng, extent)
        assessed = judge.assess(occluder, canvas.plate_origin)
        hidden = {character.position for character in assessed if not character.readable}
        if len(hidden) <= max_hidden or attempts >= 6:
            break
        extent *= 0.7
    if len(hidden) > max_hidden:  # pragma: no cover - six shrinks always suffice in practice
        return set(), {"kind": "skipped", "extent": 0.0, "hidden_positions": []}
    canvas.rgb += occluder[..., None] * (colour - canvas.rgb)
    canvas.alpha = np.maximum(canvas.alpha, occluder)
    canvas.plate_alpha *= 1.0 - occluder
    canvas.ink *= 1.0 - occluder
    return hidden, {
        "kind": kind,
        "extent": round(extent, 4),
        "attempts": attempts,
        "hidden_positions": sorted(hidden),
        "characters": [character.record() for character in assessed],
    }
