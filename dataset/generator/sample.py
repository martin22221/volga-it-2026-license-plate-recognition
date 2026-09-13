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
    EFFECT_SEVERITY,
    EXCLUSIVE_EFFECTS,
    OPTIONAL_EFFECTS,
    PLATE_TYPES,
    DifficultyProfile,
    GeneratorConfig,
    largest_remainder,
)
from .fonts import FontProvider, get_font_provider
from .geometry import (
    apply_homography,
    homography,
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

#: Redraws of a sample's effects when the plate contrast falls below the
#: level's ``min_plate_contrast``; after these, the sample is rendered clean.
MAX_LEGIBILITY_ATTEMPTS = 3


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
        plans.append(
            SamplePlan(
                index=index,
                sample_seed=derive_seed(config.seed, "sample", index),
                plate_type=plate_type,
                difficulty=level,
                text=sampler.sample(plate_type),
            )
        )
    return plans


def select_effects(profile: DifficultyProfile, rng: np.random.Generator) -> list[str]:
    """Which optional effects this sample gets.

    One uniform draw per effect, always, so the stream stays aligned whatever
    is enabled.  Exclusive pairs keep their first member; at most
    ``max_effects`` survive, and their summed severity must fit the level's
    ``severity_budget`` -- difficulty raises the ceiling, it does not stack
    every degradation on every image.
    """
    draws = rng.random(len(OPTIONAL_EFFECTS))
    enabled = [name for name, u in zip(OPTIONAL_EFFECTS, draws) if u < profile.effect_probability[name]]
    for first, second in EXCLUSIVE_EFFECTS:
        if first in enabled and second in enabled:
            enabled.remove(second)
    # Admit effects in a random order while they fit the count cap and the
    # severity budget; report them in pipeline order.
    kept: list[str] = []
    budget = profile.severity_budget
    for i in rng.permutation(len(enabled)):
        name = enabled[int(i)]
        if len(kept) < profile.max_effects and EFFECT_SEVERITY[name] <= budget + 1e-9:
            kept.append(name)
            budget -= EFFECT_SEVERITY[name]
    return [name for name in enabled if name in kept]


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


def _warp_plane(
    canvas, matrix: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Warp the plane canvas (premultiplied), its plate mask and ink into the output."""
    coefficients = pil_perspective_coefficients(matrix)

    def warp(channel: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.ascontiguousarray(channel, dtype=np.float32))
        warped = image.transform(size, Image.PERSPECTIVE, coefficients, Image.BICUBIC)
        return np.asarray(warped, dtype=np.float32)

    alpha = np.clip(warp(canvas.alpha), 0.0, 1.0)
    premultiplied = np.stack([warp(canvas.rgb[..., c] * canvas.alpha) for c in range(3)], axis=-1)
    premultiplied = np.clip(premultiplied, 0.0, alpha[..., None])
    plate_mask = np.clip(warp(canvas.plate_alpha), 0.0, 1.0)
    ink = np.clip(warp(canvas.ink), 0.0, 1.0)
    return premultiplied, alpha, plate_mask, ink


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

    # ---- plate, effects, scene: retried until the plate stays legible ------
    layout = build_layout(plan.plate_type, plan.text)
    style = sample_style(rngs.stream("style"), layout.field_colour)
    render_scale = float(np.clip(1.3 * local_scale(relative, width_mm, height_mm), 0.08, 3.0))
    scene, scene_record = background.render(rngs.stream("background"), size)
    light_record: dict = {}
    attempt_log: list[dict] = []
    for attempt in range(MAX_LEGIBILITY_ATTEMPTS + 1):
        suffix = "" if attempt == 0 else f"#retry{attempt}"
        final = attempt == MAX_LEGIBILITY_ATTEMPTS
        # The last attempt drops every optional effect: a clean render is
        # always legible, so the loop always ends with a truthful label.
        effects_enabled = [] if final else select_effects(profile, rngs.stream("effects" + suffix))
        rendered = _render_attempt(
            config, plan, profile, ranges, rngs, suffix, effects_enabled,
            layout=layout, style=style, font=font, render_scale=render_scale,
            relative=relative, size=size, margin=margin, scene=scene, scene_record=scene_record,
        )
        contrast = rendered["contrast"]
        attempt_log.append({"effects": sorted(rendered["effects"]), "plate_contrast": round(contrast, 4)})
        if contrast >= profile.min_plate_contrast or final:
            break
    image = rendered["image"]
    effects = rendered["effects"]
    hidden = rendered["hidden"]
    quad = rendered["quad"]
    canvas_record = rendered["vehicle"]
    light_record = rendered["lighting"]
    noise_record = rendered["noise"]
    plate_mask = rendered["plate_mask"]
    effects_enabled = rendered["enabled"]

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
            "sheen": round(style.sheen, 4),
            "bolts": style.bolt_rgb is not None,
        },
        "legibility": {
            "plate_contrast": attempt_log[-1]["plate_contrast"],
            "min_required": profile.min_plate_contrast,
            "attempts": attempt_log,
        },
        "vehicle": canvas_record,
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


def plate_contrast(image: np.ndarray, plate_mask: np.ndarray, ink: np.ndarray) -> float:
    """Ink-against-field contrast of the plate in the finished image.

    ``(field - ink) / (field + ink)`` of median luminance, over pixels well
    inside the plate: 1.0 is black on white, 0 means the characters have
    vanished into the field.  Blur, darkness, glare and dirt all lower it.
    """
    luminance = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    inside = plate_mask > 0.9
    ink_px = inside & (ink > 0.6)
    field_px = inside & (ink < 0.03)
    if ink_px.sum() < 5 or field_px.sum() < 20:
        return 0.0
    ink_level = float(np.median(luminance[ink_px]))
    field_level = float(np.median(luminance[field_px]))
    return (field_level - ink_level) / (field_level + ink_level + 1e-6)


def _render_attempt(
    config: GeneratorConfig,
    plan: SamplePlan,
    profile: DifficultyProfile,
    ranges: dict,
    rngs: SampleRng,
    suffix: str,
    effects_enabled: list[str],
    *,
    layout,
    style,
    font: FontProvider,
    render_scale: float,
    relative: np.ndarray,
    size: tuple[int, int],
    margin: float,
    scene: np.ndarray,
    scene_record: dict,
) -> dict:
    """One pass of plate -> vehicle -> warp -> photometric effects."""

    def stream(name: str) -> np.random.Generator:
        return rngs.stream(name + suffix)

    plate = render_plate(layout, style, font, render_scale, supersample=config.supersample, rng=rngs.stream("surface"))
    effects: dict[str, dict] = {}
    if "dirt" in effects_enabled:
        e_rng = stream("effect:dirt")
        effects["dirt"] = apply_dirt(plate, e_rng, _u(e_rng, ranges["dirt"]["strength"]))

    canvas = build_vehicle_panel(rngs.stream("panel"), plate, plan.plate_type)
    hidden: set[int] = set()
    if "occlusion" in effects_enabled:
        e_rng = stream("effect:occlusion")
        hidden, effects["occlusion"] = apply_occlusion(
            canvas, plate.glyph_boxes, e_rng, _u(e_rng, ranges["occlusion"]["extent"]),
            max_hidden=config.max_hidden_characters,
        )

    # Stand the vehicle on the ground: its wheels (panel bottom centre) must
    # project at or below the background's ground line.
    to_relative = homography(canvas.plate_corners, relative)
    panel_h, panel_w = canvas.alpha.shape
    wheels_dy = float(apply_homography(to_relative, np.array([[panel_w / 2.0, float(panel_h)]]))[0, 1])
    min_centre_y = scene_record.get("ground_y", 0) - wheels_dy + 0.02 * size[1]
    placement = place_plate(
        canvas.plate_corners, relative, size, rngs.stream("placement"), margin=margin, min_centre_y=min_centre_y
    )
    grounded = placement.quad.mean(axis=0)[1] >= min_centre_y - 1e-6
    premultiplied, alpha, plate_mask, ink = _warp_plane(canvas, placement.matrix, size)
    image = premultiplied + scene * (1.0 - alpha[..., None])

    image, light_record = lighting(image, rngs.stream("lighting"), profile)
    quad = placement.quad
    if "night" in effects_enabled:
        image, effects["night"] = night(image, plate_mask, quad, stream("effect:night"), ranges["night"])
    if "low_light" in effects_enabled:
        image, effects["low_light"] = low_light(image, stream("effect:low_light"), ranges["low_light"])
    if "shadow" in effects_enabled:
        image, effects["shadow"] = shadow(image, quad, stream("effect:shadow"), ranges["shadow"])
    if "glare" in effects_enabled:
        image, effects["glare"] = glare(image, quad, stream("effect:glare"), ranges["glare"])
    for weather in ("rain", "snow"):
        if weather in effects_enabled:
            w_rng = stream(f"effect:{weather}")
            image, effects[weather] = precipitation(image, w_rng, _u(w_rng, ranges[weather]["density"]), weather)
    if "motion_blur" in effects_enabled:
        image, effects["motion_blur"] = motion_blur(image, stream("effect:motion_blur"), ranges["motion_blur"])
    if "defocus" in effects_enabled:
        image, effects["defocus"] = defocus(image, stream("effect:defocus"), ranges["defocus"])
    n_rng = stream("noise")
    if "heavy_noise" in effects_enabled:
        sigma, chroma = _u(n_rng, ranges["heavy_noise"]["sigma"]), True
    else:
        sigma, chroma = _u(n_rng, profile.noise_sigma), False
    image, noise_record = sensor_noise(image, n_rng, sigma, chroma=chroma)
    return {
        "image": image,
        "effects": effects,
        "enabled": effects_enabled,
        "hidden": hidden,
        "quad": quad,
        "vehicle": {**canvas.record, "grounded": bool(grounded)},
        "lighting": light_record,
        "noise": noise_record,
        "plate_mask": plate_mask,
        "contrast": plate_contrast(image, plate_mask, ink),
    }

__all__ = [
    "GeneratedSample",
    "PLATE_TYPES",
    "SamplePlan",
    "generate_sample",
    "plan_dataset",
    "select_effects",
]
