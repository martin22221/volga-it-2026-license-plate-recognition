"""Generator configuration: built-in defaults, JSON loading and validation.

A configuration is plain data.  The built-in defaults below are the reference;
``configs/default.json`` mirrors them, and a JSON file passed with
``--config`` only needs the keys it changes -- everything else is inherited.

Unknown keys are an error rather than silently ignored: a misspelt option
that quietly does nothing would make a generated dataset misleading.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Final, Mapping

PLATE_TYPES: Final[tuple[str, ...]] = ("type1", "type1a", "type1b")
DIFFICULTIES: Final[tuple[str, ...]] = ("easy", "medium", "hard")

#: How a ``type1b`` plate's characters are composed.  ``competition`` is the
#: mask the task and ``src/validator.py`` use for every class
#: (``A123BC77``); ``gost_1b`` is the composition GOST R 50577-2018 section
#: 3.3 gives for type 1B (``AB12377``), which that validator rejects.
TEXT_FORMATS: Final[tuple[str, ...]] = ("competition", "gost_1b")

FONT_PROVIDERS: Final[tuple[str, ...]] = ("stroke",)

#: Optional effects, in the order the pipeline considers them.
OPTIONAL_EFFECTS: Final[tuple[str, ...]] = (
    "night",
    "low_light",
    "shadow",
    "glare",
    "dirt",
    "occlusion",
    "rain",
    "snow",
    "motion_blur",
    "defocus",
    "heavy_noise",
)

#: The parameter ranges each optional effect draws from.
EFFECT_RANGE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "night": ("darkness", "plate_gain"),
    "low_light": ("gain",),
    "shadow": ("strength",),
    "glare": ("strength", "radius"),
    "dirt": ("strength",),
    "occlusion": ("extent",),
    "rain": ("density",),
    "snow": ("density",),
    "motion_blur": ("length_px",),
    "defocus": ("radius_px",),
    "heavy_noise": ("sigma",),
}

#: Effects that cannot be combined; when both are drawn, the first is kept.
EXCLUSIVE_EFFECTS: Final[tuple[tuple[str, str], ...]] = (
    ("night", "low_light"),
    ("rain", "snow"),
    # Two blurs compound; each is capped for legibility on its own.
    ("motion_blur", "defocus"),
)

Range = tuple[float, float]


class ConfigError(ValueError):
    """The generator configuration is invalid."""


@dataclass(frozen=True)
class DifficultyProfile:
    """Parameter ranges for one difficulty level.

    ``px_per_mm`` sets the plate's size in the image: a 520 mm ``type1`` plate
    at 0.30 px/mm is 156 px wide.  Angles are maximum absolute values; each
    sample draws uniformly inside them.  ``distance_ratio`` is camera distance
    over plate width -- smaller values give stronger perspective
    (keystone) distortion at the same angle.
    """

    px_per_mm: Range
    max_yaw_deg: float
    max_pitch_deg: float
    max_roll_deg: float
    distance_ratio: Range
    brightness: Range
    contrast: Range
    gamma: Range
    white_balance: float
    noise_sigma: Range
    jpeg_quality: tuple[int, int]
    double_jpeg_probability: float
    max_effects: int
    effect_probability: Mapping[str, float]
    effect_range: Mapping[str, Mapping[str, Range]]


def _profile(**overrides: Any) -> DifficultyProfile:
    return DifficultyProfile(**overrides)


DEFAULT_DIFFICULTIES: Final[dict[str, DifficultyProfile]] = {
    "easy": _profile(
        px_per_mm=(0.30, 0.60),
        max_yaw_deg=12.0,
        max_pitch_deg=8.0,
        max_roll_deg=4.0,
        distance_ratio=(8.0, 20.0),
        brightness=(0.92, 1.08),
        contrast=(0.92, 1.08),
        gamma=(0.92, 1.08),
        white_balance=0.04,
        noise_sigma=(0.002, 0.008),
        jpeg_quality=(85, 95),
        double_jpeg_probability=0.0,
        max_effects=1,
        effect_probability={
            "night": 0.10,
            "low_light": 0.05,
            "shadow": 0.15,
            "glare": 0.05,
            "dirt": 0.10,
            "occlusion": 0.0,
            "rain": 0.03,
            "snow": 0.02,
            "motion_blur": 0.05,
            "defocus": 0.10,
            "heavy_noise": 0.0,
        },
        effect_range={
            "night": {"darkness": (0.40, 0.55), "plate_gain": (0.85, 1.0)},
            "low_light": {"gain": (0.65, 0.80)},
            "shadow": {"strength": (0.20, 0.35)},
            "glare": {"strength": (0.20, 0.35), "radius": (0.25, 0.50)},
            "dirt": {"strength": (0.15, 0.30)},
            "occlusion": {"extent": (0.05, 0.10)},
            "rain": {"density": (0.20, 0.40)},
            "snow": {"density": (0.20, 0.40)},
            "motion_blur": {"length_px": (2.0, 4.0)},
            "defocus": {"radius_px": (0.5, 1.0)},
            "heavy_noise": {"sigma": (0.02, 0.03)},
        },
    ),
    "medium": _profile(
        px_per_mm=(0.17, 0.38),
        max_yaw_deg=30.0,
        max_pitch_deg=15.0,
        max_roll_deg=8.0,
        distance_ratio=(5.0, 15.0),
        brightness=(0.80, 1.15),
        contrast=(0.80, 1.15),
        gamma=(0.85, 1.20),
        white_balance=0.08,
        noise_sigma=(0.004, 0.015),
        jpeg_quality=(60, 90),
        double_jpeg_probability=0.10,
        max_effects=2,
        effect_probability={
            "night": 0.25,
            "low_light": 0.15,
            "shadow": 0.30,
            "glare": 0.15,
            "dirt": 0.30,
            "occlusion": 0.05,
            "rain": 0.08,
            "snow": 0.06,
            "motion_blur": 0.20,
            "defocus": 0.20,
            "heavy_noise": 0.10,
        },
        effect_range={
            "night": {"darkness": (0.28, 0.45), "plate_gain": (0.70, 0.95)},
            "low_light": {"gain": (0.50, 0.70)},
            "shadow": {"strength": (0.30, 0.50)},
            "glare": {"strength": (0.30, 0.50), "radius": (0.30, 0.70)},
            "dirt": {"strength": (0.25, 0.50)},
            "occlusion": {"extent": (0.08, 0.20)},
            "rain": {"density": (0.35, 0.70)},
            "snow": {"density": (0.35, 0.70)},
            "motion_blur": {"length_px": (3.0, 7.0)},
            "defocus": {"radius_px": (0.8, 1.6)},
            "heavy_noise": {"sigma": (0.03, 0.05)},
        },
    ),
    "hard": _profile(
        px_per_mm=(0.12, 0.26),
        max_yaw_deg=50.0,
        max_pitch_deg=25.0,
        max_roll_deg=15.0,
        distance_ratio=(3.5, 10.0),
        brightness=(0.65, 1.25),
        contrast=(0.65, 1.20),
        gamma=(0.80, 1.35),
        white_balance=0.12,
        noise_sigma=(0.006, 0.022),
        jpeg_quality=(35, 75),
        double_jpeg_probability=0.25,
        max_effects=3,
        effect_probability={
            "night": 0.40,
            "low_light": 0.20,
            "shadow": 0.40,
            "glare": 0.30,
            "dirt": 0.45,
            "occlusion": 0.15,
            "rain": 0.15,
            "snow": 0.12,
            "motion_blur": 0.35,
            "defocus": 0.30,
            "heavy_noise": 0.25,
        },
        effect_range={
            "night": {"darkness": (0.18, 0.35), "plate_gain": (0.55, 0.90)},
            "low_light": {"gain": (0.38, 0.60)},
            "shadow": {"strength": (0.40, 0.65)},
            "glare": {"strength": (0.40, 0.62), "radius": (0.35, 0.80)},
            "dirt": {"strength": (0.35, 0.70)},
            "occlusion": {"extent": (0.12, 0.30)},
            "rain": {"density": (0.50, 1.00)},
            "snow": {"density": (0.50, 1.00)},
            "motion_blur": {"length_px": (4.0, 11.0)},
            "defocus": {"radius_px": (1.2, 2.4)},
            "heavy_noise": {"sigma": (0.04, 0.07)},
        },
    ),
}

DEFAULT_CLASS_WEIGHTS: Final[dict[str, float]] = {
    # docs/dataset_strategy.md: roughly type1a 40 % / type1b 40 % / rest 20 %.
    "type1": 0.20,
    "type1a": 0.40,
    "type1b": 0.40,
}

DEFAULT_DIFFICULTY_WEIGHTS: Final[dict[str, float]] = {
    "easy": 0.35,
    "medium": 0.40,
    "hard": 0.25,
}

DEFAULT_IMAGE_SIZES: Final[tuple[tuple[int, int], ...]] = (
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
)

#: Smallest character height, in output pixels, a plate may be rendered at.
#: Below this the label would claim text nobody could read.
DEFAULT_MIN_CHAR_HEIGHT_PX: Final[float] = 9.0


@dataclass(frozen=True)
class GeneratorConfig:
    """Everything that determines a generated dataset, apart from the code."""

    seed: int = 20260913
    count: int = 100
    class_counts: Mapping[str, int] | None = None
    class_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CLASS_WEIGHTS)
    )
    difficulty_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DIFFICULTY_WEIGHTS)
    )
    image_sizes: tuple[tuple[int, int], ...] = DEFAULT_IMAGE_SIZES
    three_digit_region_probability: float = 0.35
    type1b_text_format: str = "competition"
    max_images_per_plate: int = 1
    font: str = "stroke"
    supersample: int = 4
    min_char_height_px: float = DEFAULT_MIN_CHAR_HEIGHT_PX
    difficulties: Mapping[str, DifficultyProfile] = field(
        default_factory=lambda: dict(DEFAULT_DIFFICULTIES)
    )

    def resolved_class_counts(self) -> dict[str, int]:
        """Exact number of images per plate type."""
        if self.class_counts is not None:
            return {name: int(self.class_counts.get(name, 0)) for name in PLATE_TYPES}
        return largest_remainder(self.count, self.class_weights, PLATE_TYPES)

    @property
    def total_count(self) -> int:
        return sum(self.resolved_class_counts().values())


def largest_remainder(
    total: int, weights: Mapping[str, float], order: tuple[str, ...]
) -> dict[str, int]:
    """Split ``total`` into integers proportional to ``weights``.

    Largest-remainder rounding, ties broken by ``order``, so the result always
    sums to ``total`` and is deterministic.
    """
    weight_sum = sum(float(weights.get(name, 0.0)) for name in order)
    if weight_sum <= 0:
        raise ConfigError("weights must contain at least one positive value")
    exact = {name: total * float(weights.get(name, 0.0)) / weight_sum for name in order}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = total - sum(counts.values())
    by_remainder = sorted(order, key=lambda name: (-(exact[name] - counts[name]), order.index(name)))
    for name in by_remainder[:remaining]:
        counts[name] += 1
    return counts


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _check_range(name: str, value: Any, *, low: float, high: float) -> None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    ):
        raise ConfigError(f"{name} must be a [low, high] pair of numbers, got {value!r}")
    lo, hi = value
    if lo > hi:
        raise ConfigError(f"{name}: low {lo} is greater than high {hi}")
    if lo < low or hi > high:
        raise ConfigError(f"{name} must lie within [{low}, {high}], got {value!r}")


def _check_probability(name: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be a probability in [0, 1], got {value!r}")


def _check_weights(name: str, weights: Mapping[str, float], allowed: tuple[str, ...]) -> None:
    if not isinstance(weights, Mapping) or not weights:
        raise ConfigError(f"{name} must be a non-empty mapping")
    unknown = sorted(set(weights) - set(allowed))
    if unknown:
        raise ConfigError(f"{name}: unknown key(s) {unknown}; allowed: {list(allowed)}")
    for key, value in weights.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ConfigError(f"{name}[{key!r}] must be a non-negative number, got {value!r}")
    if sum(weights.values()) <= 0:
        raise ConfigError(f"{name} must contain at least one positive weight")


def validate_profile(level: str, profile: DifficultyProfile) -> None:
    prefix = f"difficulties.{level}"
    _check_range(f"{prefix}.px_per_mm", profile.px_per_mm, low=0.02, high=5.0)
    for name in ("max_yaw_deg", "max_pitch_deg", "max_roll_deg"):
        value = getattr(profile, name)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 70.0:
            raise ConfigError(f"{prefix}.{name} must be within [0, 70] degrees, got {value!r}")
    _check_range(f"{prefix}.distance_ratio", profile.distance_ratio, low=2.0, high=200.0)
    _check_range(f"{prefix}.brightness", profile.brightness, low=0.1, high=3.0)
    _check_range(f"{prefix}.contrast", profile.contrast, low=0.1, high=3.0)
    _check_range(f"{prefix}.gamma", profile.gamma, low=0.2, high=5.0)
    if not isinstance(profile.white_balance, (int, float)) or not 0 <= profile.white_balance <= 0.5:
        raise ConfigError(f"{prefix}.white_balance must be within [0, 0.5]")
    _check_range(f"{prefix}.noise_sigma", profile.noise_sigma, low=0.0, high=0.5)
    _check_range(f"{prefix}.jpeg_quality", profile.jpeg_quality, low=5, high=100)
    _check_probability(f"{prefix}.double_jpeg_probability", profile.double_jpeg_probability)
    if not isinstance(profile.max_effects, int) or profile.max_effects < 0:
        raise ConfigError(f"{prefix}.max_effects must be a non-negative integer")

    unknown = sorted(set(profile.effect_probability) - set(OPTIONAL_EFFECTS))
    if unknown:
        raise ConfigError(f"{prefix}.effect_probability: unknown effect(s) {unknown}")
    missing = sorted(set(OPTIONAL_EFFECTS) - set(profile.effect_probability))
    if missing:
        raise ConfigError(f"{prefix}.effect_probability: missing effect(s) {missing}")
    for effect, probability in profile.effect_probability.items():
        _check_probability(f"{prefix}.effect_probability.{effect}", probability)

    unknown = sorted(set(profile.effect_range) - set(OPTIONAL_EFFECTS))
    if unknown:
        raise ConfigError(f"{prefix}.effect_range: unknown effect(s) {unknown}")
    for effect, keys in EFFECT_RANGE_KEYS.items():
        ranges = profile.effect_range.get(effect)
        if ranges is None:
            raise ConfigError(f"{prefix}.effect_range.{effect} is missing")
        extra = sorted(set(ranges) - set(keys))
        if extra:
            raise ConfigError(f"{prefix}.effect_range.{effect}: unknown key(s) {extra}")
        for key in keys:
            if key not in ranges:
                raise ConfigError(f"{prefix}.effect_range.{effect}.{key} is missing")
            _check_range(f"{prefix}.effect_range.{effect}.{key}", ranges[key], low=0.0, high=100.0)


def validate_config(config: GeneratorConfig) -> GeneratorConfig:
    """Raise :class:`ConfigError` if ``config`` is unusable; return it otherwise."""
    if not isinstance(config.seed, int) or isinstance(config.seed, bool) or config.seed < 0:
        raise ConfigError(f"seed must be a non-negative integer, got {config.seed!r}")

    if config.class_counts is not None:
        if not isinstance(config.class_counts, Mapping) or not config.class_counts:
            raise ConfigError("class_counts must be a non-empty mapping")
        unknown = sorted(set(config.class_counts) - set(PLATE_TYPES))
        if unknown:
            raise ConfigError(
                f"class_counts: unknown plate type(s) {unknown}; allowed: {list(PLATE_TYPES)}"
            )
        for name, value in config.class_counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"class_counts[{name!r}] must be a non-negative integer")
        if sum(config.class_counts.values()) <= 0:
            raise ConfigError("class_counts must request at least one image")
    else:
        if not isinstance(config.count, int) or isinstance(config.count, bool) or config.count <= 0:
            raise ConfigError(f"count must be a positive integer, got {config.count!r}")
        _check_weights("class_weights", config.class_weights, PLATE_TYPES)

    _check_weights("difficulty_weights", config.difficulty_weights, DIFFICULTIES)

    if not config.image_sizes:
        raise ConfigError("image_sizes must list at least one [width, height]")
    for size in config.image_sizes:
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in size)
        ):
            raise ConfigError(f"image size must be [width, height] integers, got {size!r}")
        width, height = size
        if not (128 <= width <= 4096 and 96 <= height <= 4096):
            raise ConfigError(f"image size {size!r} outside 128..4096 x 96..4096")

    _check_probability("three_digit_region_probability", config.three_digit_region_probability)
    if config.type1b_text_format not in TEXT_FORMATS:
        raise ConfigError(
            f"type1b_text_format must be one of {list(TEXT_FORMATS)}, "
            f"got {config.type1b_text_format!r}"
        )
    if not isinstance(config.max_images_per_plate, int) or config.max_images_per_plate < 1:
        raise ConfigError("max_images_per_plate must be an integer >= 1")
    if config.font not in FONT_PROVIDERS:
        raise ConfigError(f"font must be one of {list(FONT_PROVIDERS)}, got {config.font!r}")
    if not isinstance(config.supersample, int) or not 1 <= config.supersample <= 8:
        raise ConfigError("supersample must be an integer within [1, 8]")
    if not isinstance(config.min_char_height_px, (int, float)) or not 2 <= config.min_char_height_px <= 100:
        raise ConfigError("min_char_height_px must be within [2, 100]")

    unknown = sorted(set(config.difficulties) - set(DIFFICULTIES))
    if unknown:
        raise ConfigError(f"difficulties: unknown level(s) {unknown}")
    for level in DIFFICULTIES:
        if level not in config.difficulties:
            raise ConfigError(f"difficulties.{level} is missing")
        validate_profile(level, config.difficulties[level])
    return config


# --------------------------------------------------------------------------
# Conversion to and from plain data
# --------------------------------------------------------------------------

_PROFILE_FIELDS: Final[frozenset[str]] = frozenset(f.name for f in fields(DifficultyProfile))
_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(f.name for f in fields(GeneratorConfig))
_PAIR_FIELDS: Final[frozenset[str]] = frozenset(
    {"px_per_mm", "distance_ratio", "brightness", "contrast", "gamma", "noise_sigma", "jpeg_quality"}
)


def _as_pair(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _merge_profile(base: DifficultyProfile, data: Mapping[str, Any], level: str) -> DifficultyProfile:
    if not isinstance(data, Mapping):
        raise ConfigError(f"difficulties.{level} must be an object")
    unknown = sorted(set(data) - _PROFILE_FIELDS)
    if unknown:
        raise ConfigError(f"difficulties.{level}: unknown key(s) {unknown}")
    changes: dict[str, Any] = {}
    for key, value in data.items():
        if key == "effect_probability":
            if not isinstance(value, Mapping):
                raise ConfigError(f"difficulties.{level}.effect_probability must be an object")
            changes[key] = {**base.effect_probability, **value}
        elif key == "effect_range":
            if not isinstance(value, Mapping):
                raise ConfigError(f"difficulties.{level}.effect_range must be an object")
            merged = {effect: dict(ranges) for effect, ranges in base.effect_range.items()}
            for effect, ranges in value.items():
                if not isinstance(ranges, Mapping):
                    raise ConfigError(f"difficulties.{level}.effect_range.{effect} must be an object")
                merged.setdefault(effect, {}).update(
                    {name: _as_pair(pair) for name, pair in ranges.items()}
                )
            changes[key] = merged
        elif key in _PAIR_FIELDS:
            changes[key] = _as_pair(value)
        else:
            changes[key] = value
    return replace(base, **changes)


def config_from_dict(data: Mapping[str, Any], base: GeneratorConfig | None = None) -> GeneratorConfig:
    """Overlay ``data`` onto ``base`` (the built-in defaults) and validate."""
    if not isinstance(data, Mapping):
        raise ConfigError("configuration must be a JSON object")
    base = base if base is not None else GeneratorConfig()
    data = {key: value for key, value in data.items() if not key.startswith("_")}
    unknown = sorted(set(data) - _CONFIG_FIELDS)
    if unknown:
        raise ConfigError(f"unknown configuration key(s): {unknown}")

    changes: dict[str, Any] = {}
    for key, value in data.items():
        if key == "difficulties":
            if not isinstance(value, Mapping):
                raise ConfigError("difficulties must be an object")
            unknown_levels = sorted(set(value) - set(DIFFICULTIES))
            if unknown_levels:
                raise ConfigError(f"difficulties: unknown level(s) {unknown_levels}")
            merged = dict(base.difficulties)
            for level, overrides in value.items():
                merged[level] = _merge_profile(base.difficulties[level], overrides, level)
            changes[key] = merged
        elif key == "image_sizes":
            if not isinstance(value, (list, tuple)):
                raise ConfigError("image_sizes must be a list of [width, height] pairs")
            changes[key] = tuple(_as_pair(size) for size in value)
        else:
            changes[key] = value
    return validate_config(replace(base, **changes))


def load_config(path: Path, base: GeneratorConfig | None = None) -> GeneratorConfig:
    """Read a JSON configuration file and overlay it onto ``base``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"configuration file {path} does not exist") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"configuration file {path} is not valid JSON: {error}") from None
    return config_from_dict(data, base)


def config_to_dict(config: GeneratorConfig) -> dict[str, Any]:
    """Plain JSON-serialisable form of ``config`` (recorded in the manifest)."""

    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    result: dict[str, Any] = {}
    for f in fields(GeneratorConfig):
        value = getattr(config, f.name)
        if f.name == "difficulties":
            result[f.name] = {
                level: {pf.name: plain(getattr(profile, pf.name)) for pf in fields(DifficultyProfile)}
                for level, profile in value.items()
            }
        else:
            result[f.name] = plain(value)
    return result
