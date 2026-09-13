"""Planning and generating individual samples.

:func:`plan_dataset` fixes, up front, every sample's plate type, difficulty,
plate identity and seed.  :func:`generate_sample` then turns one plan into an
image and its annotation, drawing every further random value from named
streams of that sample's own seed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .annotations import Annotation, derive_conditions, masked_plate_number, round_quad
from .config import (
    DIFFICULTIES,
    EXCLUSIVE_EFFECTS,
    OPTIONAL_EFFECTS,
    PLATE_TYPES,
    DifficultyProfile,
    GeneratorConfig,
    largest_remainder,
)
from .fonts import FontProvider, get_font_provider
from .geometry import (
    local_scale,
    pil_perspective_coefficients,
    place_plate,
    project_plate_corners,
)
from .photometric import (
    defocus,
    glare,
    lighting,
    low_light,
    motion_blur,
    night,
    precipitation,
    sensor_noise,
    shadow,
)
from .plate_text import PlateIdentitySampler, PlateText
from .render import apply_dirt, render_plate, sample_style
from .rng import SampleRng, derive_seed, make_rng
from .scene import BackgroundProvider, ProceduralBackground, apply_occlusion, build_vehicle_panel
from .templates import PLATE_SIZE_MM, build_layout

#: Smallest label character on any layout is 58 mm tall.
SMALLEST_CHAR_MM = 58.0


@dataclass(frozen=True)
class SamplePlan:
    index: int
    sample_seed: int
    plate_type: str
    difficulty: str
    text: PlateText


@dataclass
class GeneratedSample:
    plan: SamplePlan
    image_bytes: bytes
    size: tuple[int, int]
    annotation: Annotation
    record: dict
    plate_mask: np.ndarray = field(repr=False)


def plan_dataset(config: GeneratorConfig) -> list[SamplePlan]:
    """Every sample's type, difficulty, identity and seed, in index order.

    Class counts are exact; difficulties are split within each class, so every
    class gets its own easy/medium/hard mix.  The order is then shuffled with
    the master seed.
    """
    entries: list[tuple[str, str]] = []
    for plate_type, count in config.resolved_class_counts().items():
        if count <= 0:
            continue
        split = largest_remainder(count, config.difficulty_weights, DIFFICULTIES)
        entries += [(plate_type, level) for level in DIFFICULTIES for _ in range(split[level])]
    order = make_rng(config.seed, "assignment").permutation(len(entries))
    entries = [entries[int(i)] for i in order]

    sampler = PlateIdentitySampler(
        make_rng(config.seed, "identity"),
        three_digit_probability=config.three_digit_region_probability,
        max_images_per_plate=config.max_images_per_plate,
    )
    plans = []
    for index, (plate_type, level) in enumerate(entries):
        text_format = config.type1b_text_format if plate_type == "type1b" else "competition"
        plans.append(
            SamplePlan(
                index=index,
                sample_seed=derive_seed(config.seed, "sample", index),
                plate_type=plate_type,
                difficulty=level,
                text=sampler.sample(text_format),
            )
        )
    return plans


def select_effects(profile: DifficultyProfile, rng: np.random.Generator) -> list[str]:
    """Which optional effects this sample gets.

    One uniform draw per effect, always, so the stream stays aligned whatever
    is enabled.  Exclusive pairs keep their first member, and at most
    ``max_effects`` survive -- difficulty raises the ceiling, it does not stack
    every degradation on every image.
    """
    draws = rng.random(len(OPTIONAL_EFFECTS))
    enabled = [name for name, u in zip(OPTIONAL_EFFECTS, draws) if u < profile.effect_probability[name]]
    for first, second in EXCLUSIVE_EFFECTS:
        if first in enabled and second in enabled:
            enabled.remove(second)
    cap_draw = rng.permutation(len(enabled))
    if len(enabled) > profile.max_effects:
        keep = sorted(int(i) for i in cap_draw[: profile.max_effects])
        enabled = [enabled[i] for i in keep]
    return enabled


def _u(rng: np.random.Generator, pair: tuple[float, float]) -> float:
    return float(rng.uniform(pair[0], pair[1]))


#: Blur is capped relative to the smallest character's height in pixels, so a
#: label never names characters the blur has erased.  The open space inside a
#: character (e.g. the counter of "0") is about 0.45 of its height; motion blur
#: stays well below that, defocus below a third of a stroke-and-gap.
MOTION_BLUR_LEGIBILITY = 0.25
DEFOCUS_LEGIBILITY = 0.12
#: Below this character height, JPEG quality is floored at SMALL_TEXT_JPEG_FLOOR.
SMALL_TEXT_PX = 12.0
SMALL_TEXT_JPEG_FLOOR = 55


def _text_scale_px(quad: np.ndarray, width_mm: float, height_mm: float) -> float:
    """Height of the smallest character in pixels, along the plate's weaker axis.

    Uses the shorter of each pair of opposite edges and the more compressed of
    the two directions, so a plate turned sideways (yaw) counts as smaller
    even though its characters are still tall.
    """
    horizontal = min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])) / width_mm
    vertical = min(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])) / height_mm
    return float(min(horizontal, vertical) * SMALLEST_CHAR_MM)


def _legible_ranges(profile: DifficultyProfile, char_px: float) -> dict[str, dict[str, tuple[float, float]]]:
    ranges = {effect: dict(values) for effect, values in profile.effect_range.items()}
    for effect, key, fraction in (
        ("motion_blur", "length_px", MOTION_BLUR_LEGIBILITY),
        ("defocus", "radius_px", DEFOCUS_LEGIBILITY),
    ):
        cap = fraction * char_px
        lo, hi = ranges[effect][key]
        ranges[effect][key] = (min(lo, cap), min(hi, cap))
    return ranges


def _warp_plane(canvas, matrix: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp the plane canvas (premultiplied) and its masks into the output."""
    coefficients = pil_perspective_coefficients(matrix)

    def warp(channel: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32))
        warped = image.transform(size, Image.PERSPECTIVE, coefficients, Image.BICUBIC)
        return np.asarray(warped, dtype=np.float32)

    alpha = np.clip(warp(canvas.alpha), 0.0, 1.0)
    premultiplied = np.stack([warp(canvas.rgb[..., c] * canvas.alpha) for c in range(3)], axis=-1)
    premultiplied = np.clip(premultiplied, 0.0, alpha[..., None])
    plate_mask = np.clip(warp(canvas.plate_alpha), 0.0, 1.0)
    return premultiplied, alpha, plate_mask


def _encode_jpeg(
    image: np.ndarray, rng: np.random.Generator, profile: DifficultyProfile, *, min_quality: int = 0
) -> tuple[bytes, dict]:
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))  # H x W x 3 uint8 -> RGB
    lo, hi = profile.jpeg_quality
    lo = max(lo, min(hi, min_quality))
    quality = int(rng.integers(lo, hi + 1))
    double = bool(rng.random() < profile.double_jpeg_probability)
    first_quality = int(rng.integers(lo, hi + 1))
    if double:
        buffer = io.BytesIO()
        pil.save(buffer, "JPEG", quality=first_quality)
        pil = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, "JPEG", quality=quality)
    record = {"quality": quality, "double": double}
    if double:
        record["first_quality"] = first_quality
    return buffer.getvalue(), record


def generate_sample(
    config: GeneratorConfig,
    plan: SamplePlan,
    *,
    font: FontProvider | None = None,
    background: BackgroundProvider | None = None,
    image_name: str = "",
) -> GeneratedSample:
    """Render one sample: plate -> panel -> perspective -> scene -> effects."""
    font = font if font is not None else get_font_provider(config.font)
    background = background if background is not None else ProceduralBackground()
    rngs = SampleRng(plan.sample_seed)
    profile = config.difficulties[plan.difficulty]
    width_mm, height_mm = PLATE_SIZE_MM[plan.plate_type]

    size_rng = rngs.stream("size")
    size = tuple(config.image_sizes[int(size_rng.integers(len(config.image_sizes)))])

    # ---- geometry --------------------------------------------------------
    g = rngs.stream("geometry")
    yaw = _u(g, (-profile.max_yaw_deg, profile.max_yaw_deg))
    pitch = _u(g, (-profile.max_pitch_deg, profile.max_pitch_deg))
    roll = _u(g, (-profile.max_roll_deg, profile.max_roll_deg))
    px_per_mm = _u(g, profile.px_per_mm)
    distance_ratio = _u(g, profile.distance_ratio)
    margin = max(3.0, 0.015 * min(size))

    def project(scale: float) -> np.ndarray:
        return project_plate_corners(
            width_mm, height_mm, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
            px_per_mm=scale, distance_ratio=distance_ratio,
        )

    relative = project(px_per_mm)
    adjustments: list[str] = []
    # Keep the smallest character legible (min_char_height_px) ...
    char_px = _text_scale_px(relative, width_mm, height_mm)
    if char_px < config.min_char_height_px:
        px_per_mm *= config.min_char_height_px / char_px * 1.001
        relative = project(px_per_mm)
        adjustments.append("enlarged_for_legibility")
    # ... and the whole plate inside the frame.
    extent = relative.max(axis=0) - relative.min(axis=0)
    room = np.array(size, dtype=np.float64) - 2 * margin - 2
    if np.any(extent > room):
        px_per_mm *= float(np.min(room / extent)) * 0.98
        relative = project(px_per_mm)
        adjustments.append("shrunk_to_fit")
    char_px = _text_scale_px(relative, width_mm, height_mm)
    ranges = _legible_ranges(profile, char_px)

    # ---- plate -----------------------------------------------------------
    layout = build_layout(plan.plate_type, plan.text)
    style = sample_style(rngs.stream("style"), layout.field_colour)
    render_scale = float(np.clip(1.3 * local_scale(relative, width_mm, height_mm), 0.08, 3.0))
    plate = render_plate(layout, style, font, render_scale, supersample=config.supersample)

    effects_enabled = select_effects(profile, rngs.stream("effects"))
    effects: dict[str, dict] = {}
    if "dirt" in effects_enabled:
        e_rng = rngs.stream("effect:dirt")
        effects["dirt"] = apply_dirt(plate, e_rng, _u(e_rng, ranges["dirt"]["strength"]))

    canvas = build_vehicle_panel(rngs.stream("panel"), plate, plan.plate_type)
    hidden: set[int] = set()
    if "occlusion" in effects_enabled:
        e_rng = rngs.stream("effect:occlusion")
        hidden, effects["occlusion"] = apply_occlusion(
            canvas, plate.glyph_boxes, e_rng, _u(e_rng, ranges["occlusion"]["extent"])
        )

    placement = place_plate(canvas.plate_corners, relative, size, rngs.stream("placement"), margin=margin)

    # ---- scene -----------------------------------------------------------
    scene, scene_record = background.render(rngs.stream("background"), size)
    premultiplied, alpha, plate_mask = _warp_plane(canvas, placement.matrix, size)
    image = premultiplied + scene * (1.0 - alpha[..., None])

    # ---- photometric -------------------------------------------------------
    image, light_record = lighting(image, rngs.stream("lighting"), profile)
    quad = placement.quad
    if "night" in effects_enabled:
        image, effects["night"] = night(image, plate_mask, quad, rngs.stream("effect:night"), ranges["night"])
    if "low_light" in effects_enabled:
        image, effects["low_light"] = low_light(image, rngs.stream("effect:low_light"), ranges["low_light"])
    if "shadow" in effects_enabled:
        image, effects["shadow"] = shadow(image, quad, rngs.stream("effect:shadow"), ranges["shadow"])
    if "glare" in effects_enabled:
        image, effects["glare"] = glare(image, quad, rngs.stream("effect:glare"), ranges["glare"])
    for weather in ("rain", "snow"):
        if weather in effects_enabled:
            w_rng = rngs.stream(f"effect:{weather}")
            image, effects[weather] = precipitation(
                image, w_rng, _u(w_rng, ranges[weather]["density"]), weather
            )
    if "motion_blur" in effects_enabled:
        image, effects["motion_blur"] = motion_blur(image, rngs.stream("effect:motion_blur"), ranges["motion_blur"])
    if "defocus" in effects_enabled:
        image, effects["defocus"] = defocus(image, rngs.stream("effect:defocus"), ranges["defocus"])
    n_rng = rngs.stream("noise")
    if "heavy_noise" in effects_enabled:
        sigma, chroma = _u(n_rng, ranges["heavy_noise"]["sigma"]), True
    else:
        sigma, chroma = _u(n_rng, profile.noise_sigma), False
    image, noise_record = sensor_noise(image, n_rng, sigma, chroma=chroma)
    jpeg_floor = SMALL_TEXT_JPEG_FLOOR if char_px < SMALL_TEXT_PX else 0
    image_bytes, jpeg_record = _encode_jpeg(image, rngs.stream("jpeg"), profile, min_quality=jpeg_floor)

    # ---- annotation --------------------------------------------------------
    rounded = round_quad(quad)
    conditions = derive_conditions(
        night="night" in effects_enabled, yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, effects=effects
    )
    annotation = Annotation(
        image=image_name,
        plate_num=masked_plate_number(plan.text.full, hidden),
        plate_type=plan.plate_type,
        quad=rounded,
        conditions=conditions,
    )
    record = {
        "index": plan.index,
        "image": image_name,
        "sample_seed": plan.sample_seed,
        "plate_type": plan.plate_type,
        "difficulty": plan.difficulty,
        "plate_num": annotation.plate_num,
        "plate_num_full": plan.text.full,
        "text_format": plan.text.text_format,
        "hidden_positions": sorted(hidden),
        "conditions": list(conditions),
        "image_size": list(size),
        "geometry": {
            "yaw_deg": round(yaw, 3),
            "pitch_deg": round(pitch, 3),
            "roll_deg": round(roll, 3),
            "px_per_mm": round(px_per_mm, 5),
            "distance_ratio": round(distance_ratio, 3),
            "render_px_per_mm": round(render_scale, 5),
            "adjustments": adjustments,
            "min_char_height_px": round(char_px, 2),
            "quad": [list(point) for point in rounded],
        },
        "plate_style": {
            "field_rgb": [round(c, 4) for c in style.field_rgb],
            "ink_rgb": [round(c, 4) for c in style.ink_rgb],
            "stroke_scale": round(style.stroke_scale, 4),
            "emboss": round(style.emboss, 4),
        },
        "vehicle": canvas.record,
        "background": scene_record,
        "lighting": light_record,
        "effects": effects,
        "noise": noise_record,
        "jpeg": jpeg_record,
    }
    return GeneratedSample(
        plan=plan,
        image_bytes=image_bytes,
        size=size,  # type: ignore[arg-type]
        annotation=annotation,
        record=record,
        plate_mask=plate_mask,
    )


__all__ = [
    "GeneratedSample",
    "PLATE_TYPES",
    "SamplePlan",
    "generate_sample",
    "plan_dataset",
    "select_effects",
]
