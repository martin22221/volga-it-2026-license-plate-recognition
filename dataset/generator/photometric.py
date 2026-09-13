"""Photometric degradations applied to the composed image.

Every effect takes the float image (``H x W x 3`` in [0, 1]), the plate mask
in output coordinates, the plate quad and its own random stream, and returns
the new image plus a record of the exact parameters it used.  The records go
into ``generation.jsonl``, so every image can be explained after the fact.

Order matters and is fixed: lighting, then night / low light, shadow, glare
(after darkening, so headlight glare stays bright at night), precipitation,
motion blur, defocus, sensor noise.  JPEG compression happens at save time.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EffectResult = tuple[np.ndarray, dict]


def _u(rng: np.random.Generator, pair: tuple[float, float]) -> float:
    return float(rng.uniform(pair[0], pair[1]))


def _gaussian(image: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur through PIL.

    PIL blurs 8-bit images only; the quantisation is harmless because the
    image ends as 8-bit JPEG and sensor noise is added after this step.
    """
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def lighting(image: np.ndarray, rng: np.random.Generator, profile) -> EffectResult:
    """Global exposure: brightness, contrast, gamma and white balance."""
    brightness = _u(rng, profile.brightness)
    contrast = _u(rng, profile.contrast)
    gamma = _u(rng, profile.gamma)
    wb = rng.uniform(-profile.white_balance, profile.white_balance, size=2)
    mean = float(image.mean())
    out = (image - mean) * contrast + mean
    out = np.clip(out * brightness, 0.0, 1.0) ** gamma
    out[..., 0] *= 1.0 + float(wb[0])
    out[..., 2] *= 1.0 + float(wb[1])
    record = {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "gamma": round(gamma, 4),
        "white_balance_rb": [round(float(v), 4) for v in wb],
    }
    return np.clip(out, 0.0, 1.0), record


def night(image: np.ndarray, plate_mask: np.ndarray, quad: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    """Dark scene with pools of artificial light; the retroreflective plate
    stays comparatively bright, as it does under headlights."""
    height, width = plate_mask.shape
    darkness = _u(rng, ranges["darkness"])
    plate_gain = _u(rng, ranges["plate_gain"])
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    gain = np.full((height, width), darkness * 0.5, np.float32)
    lights = int(rng.integers(1, 4))
    tint = np.array([1.0, 0.85, 0.6], np.float32) if rng.random() < 0.5 else np.array([0.85, 0.92, 1.0], np.float32)
    for _ in range(lights):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height * 0.7)
        radius = rng.uniform(0.2, 0.6) * max(width, height)
        gain += darkness * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * radius**2))
    gain = np.clip(gain, 0.0, 1.0)
    scene = image * gain[..., None] * tint
    plate = image * plate_gain
    out = scene * (1.0 - plate_mask[..., None]) + plate * plate_mask[..., None]
    return np.clip(out, 0.0, 1.0), {
        "darkness": round(darkness, 4),
        "plate_gain": round(plate_gain, 4),
        "lights": lights,
        "tint": [round(float(v), 3) for v in tint],
    }


def low_light(image: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    gain = _u(rng, ranges["gain"])
    return np.clip(image * gain, 0.0, 1.0), {"gain": round(gain, 4)}


def shadow(image: np.ndarray, quad: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    """A soft-edged shadow boundary passing near the plate."""
    height, width = image.shape[:2]
    strength = _u(rng, ranges["strength"])
    angle = rng.uniform(0, 2 * math.pi)
    centre = quad.mean(axis=0) + rng.normal(0, 0.3, 2) * (quad.max(axis=0) - quad.min(axis=0))
    softness = rng.uniform(0.01, 0.06) * max(width, height)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = (xs - centre[0]) * math.cos(angle) + (ys - centre[1]) * math.sin(angle)
    mask = 1.0 / (1.0 + np.exp(-distance / softness))
    out = image * (1.0 - strength * mask)[..., None]
    return out, {"strength": round(strength, 4), "angle_deg": round(math.degrees(angle), 2)}


def glare(image: np.ndarray, quad: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    """A bright hotspot -- sun or headlight reflection -- usually on the plate.

    Strength stays below ~0.65 in the default profiles: added to a white plate
    the field clips to 1.0 while black ink stays near the strength, so the
    characters keep at least ~0.35 of contrast and the label stays true.
    """
    height, width = image.shape[:2]
    strength = _u(rng, ranges["strength"])
    plate_width = float(np.linalg.norm(quad[1] - quad[0]))
    radius = max(3.0, _u(rng, ranges["radius"]) * plate_width)
    on_plate = bool(rng.random() < 0.75)
    if on_plate:
        weights = rng.dirichlet(np.ones(4))
        centre = weights @ quad  # a random point inside the (convex) quad
    else:
        centre = np.array([rng.uniform(0, width), rng.uniform(0, height)])
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    spot = np.exp(-((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2) / (2 * radius**2))
    colour = np.array([1.0, 0.97, 0.9], np.float32)
    out = image + strength * spot[..., None] * colour
    return np.clip(out, 0.0, 1.0), {
        "strength": round(strength, 4),
        "radius_px": round(radius, 2),
        "centre": [round(float(v), 2) for v in centre],
        "on_plate": on_plate,
    }


def precipitation(image: np.ndarray, rng: np.random.Generator, density: float, kind: str) -> EffectResult:
    """Rain streaks or snow flakes over the whole frame, plus a light haze."""
    height, width = image.shape[:2]
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    area = width * height
    if kind == "rain":
        count = int(density * area / 900)
        slant = math.radians(rng.uniform(-25, 25))
        for _ in range(count):
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            length = rng.uniform(0.02, 0.06) * height
            draw.line(
                (x, y, x + length * math.sin(slant), y + length * math.cos(slant)),
                fill=int(rng.integers(60, 150)), width=1,
            )
        streaks = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
        haze = rng.uniform(0.03, 0.10)
    else:
        count = int(density * area / 1400)
        for _ in range(count):
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            r = rng.uniform(0.6, 2.8)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=int(rng.integers(140, 250)))
        streaks = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
        haze = rng.uniform(0.05, 0.14)
    out = image * (1.0 - haze) + haze * 0.75
    out = out * (1.0 - streaks[..., None]) + streaks[..., None] * 0.9
    return np.clip(out, 0.0, 1.0), {"density": round(density, 4), "particles": count, "haze": round(float(haze), 4)}


def motion_blur(image: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    """Linear motion blur: the mean of copies shifted along one direction."""
    length = _u(rng, ranges["length_px"])
    # Vehicles mostly move across the frame or towards the camera.
    angle = rng.normal(0.0, 20.0) if rng.random() < 0.7 else rng.uniform(-90, 90)
    steps = max(2, int(round(length)))
    dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    pad = int(math.ceil(length)) + 1
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    height, width = image.shape[:2]
    accumulator = np.zeros_like(image)
    for i in range(steps):
        t = (i / (steps - 1) - 0.5) * length
        ox, oy = int(round(t * dx)), int(round(t * dy))
        accumulator += padded[pad + oy : pad + oy + height, pad + ox : pad + ox + width]
    return accumulator / steps, {"length_px": round(length, 3), "angle_deg": round(float(angle), 2)}


def defocus(image: np.ndarray, rng: np.random.Generator, ranges: dict) -> EffectResult:
    radius = _u(rng, ranges["radius_px"])
    return _gaussian(image, radius), {"radius_px": round(radius, 3)}


def sensor_noise(image: np.ndarray, rng: np.random.Generator, sigma: float, *, chroma: bool) -> EffectResult:
    """Signal-dependent Gaussian noise, optionally with colour noise."""
    luminance_noise = rng.normal(0.0, 1.0, size=image.shape[:2] + (1,)).astype(np.float32)
    scale = sigma * np.sqrt(np.clip(image, 0.0, 1.0) + 0.05) / math.sqrt(0.55)
    out = image + scale * luminance_noise
    if chroma:
        out += sigma * 0.5 * rng.normal(0.0, 1.0, size=image.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0), {"sigma": round(sigma, 5), "chroma": chroma}
