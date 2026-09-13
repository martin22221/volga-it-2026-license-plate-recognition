"""Generator configuration: defaults, JSON loading, validation, distribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset.generator.config import (
    DEFAULT_DIFFICULTIES,
    EXCLUSIVE_EFFECTS,
    ConfigError,
    GeneratorConfig,
    config_from_dict,
    config_to_dict,
    largest_remainder,
    load_config,
    validate_config,
)
from dataset.generator.sample import plan_dataset, select_effects

DEFAULT_JSON = Path(__file__).resolve().parents[1] / "dataset" / "generator" / "configs" / "default.json"


def test_defaults_are_valid() -> None:
    assert validate_config(GeneratorConfig()) is not None


def test_default_json_mirrors_the_builtin_defaults() -> None:
    assert config_to_dict(load_config(DEFAULT_JSON)) == config_to_dict(GeneratorConfig())


def test_partial_json_overlays_the_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"seed": 5, "difficulties": {"hard": {"max_effects": 1}}}), encoding="utf-8")
    config = load_config(path)
    assert config.seed == 5
    assert config.difficulties["hard"].max_effects == 1
    assert config.difficulties["hard"].px_per_mm == DEFAULT_DIFFICULTIES["hard"].px_per_mm
    assert config.difficulties["easy"] == DEFAULT_DIFFICULTIES["easy"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"unknown_option": 1}, "unknown configuration key"),
        ({"count": 0}, "count"),
        ({"count": -5}, "count"),
        ({"seed": -1}, "seed"),
        ({"class_weights": {"type9": 1.0}}, "unknown key"),
        ({"class_weights": {"type1": 0.0, "type1a": 0.0, "type1b": 0.0}}, "positive"),
        ({"class_weights": {"type1": -1.0}}, "non-negative"),
        ({"class_counts": {"other": 5}}, "unknown plate type"),
        ({"class_counts": {"type1": 0}}, "at least one"),
        ({"class_counts": {"type1": 2.5}}, "integer"),
        ({"difficulty_weights": {"extreme": 1.0}}, "unknown key"),
        ({"image_sizes": []}, "image_sizes"),
        ({"image_sizes": [[10, 10]]}, "outside"),
        ({"image_sizes": [[640]]}, "width, height"),
        ({"three_digit_region_probability": 1.5}, "probability"),
        ({"type1b_text_format": "fancy"}, "type1b_text_format"),
        ({"font": "arial"}, "font"),
        ({"supersample": 0}, "supersample"),
        ({"max_images_per_plate": 0}, "max_images_per_plate"),
        ({"difficulties": {"insane": {}}}, "unknown level"),
        ({"difficulties": {"hard": {"px_per_mm": [0.5, 0.1]}}}, "greater than"),
        ({"difficulties": {"hard": {"max_yaw_deg": 95}}}, "max_yaw_deg"),
        ({"difficulties": {"hard": {"effect_probability": {"lasers": 0.5}}}}, "unknown effect"),
        ({"difficulties": {"hard": {"effect_probability": {"night": 2.0}}}}, "probability"),
        ({"difficulties": {"hard": {"effect_range": {"glare": {"strength": [0.9, 0.1]}}}}}, "greater than"),
        ({"difficulties": {"hard": {"effect_range": {"glare": {"colour": [0, 1]}}}}}, "unknown key"),
        ({"difficulties": {"hard": {"bogus": 1}}}, "unknown key"),
    ],
)
def test_invalid_configurations_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        config_from_dict(overrides)


def test_invalid_json_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(path)
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.json")


def test_largest_remainder_is_exact_and_deterministic() -> None:
    order = ("a", "b", "c")
    assert largest_remainder(60, {"a": 1, "b": 1, "c": 1}, order) == {"a": 20, "b": 20, "c": 20}
    assert largest_remainder(10, {"a": 0.2, "b": 0.4, "c": 0.4}, order) == {"a": 2, "b": 4, "c": 4}
    for total in range(1, 200, 7):
        split = largest_remainder(total, {"a": 0.35, "b": 0.4, "c": 0.25}, order)
        assert sum(split.values()) == total


def test_per_class_counts_are_exact() -> None:
    config = config_from_dict({"class_counts": {"type1": 20, "type1a": 20, "type1b": 20}})
    plans = plan_dataset(config)
    assert len(plans) == 60
    for plate_type in ("type1", "type1a", "type1b"):
        of_type = [p for p in plans if p.plate_type == plate_type]
        assert len(of_type) == 20
        levels = {level: sum(p.difficulty == level for p in of_type) for level in ("easy", "medium", "hard")}
        assert levels == largest_remainder(20, config.difficulty_weights, ("easy", "medium", "hard"))
        assert min(levels.values()) > 0


def test_class_weights_split_the_count() -> None:
    config = config_from_dict({"count": 100})
    plans = plan_dataset(config)
    counts = {t: sum(p.plate_type == t for p in plans) for t in ("type1", "type1a", "type1b")}
    assert counts == {"type1": 20, "type1a": 40, "type1b": 40}


def test_single_difficulty() -> None:
    config = config_from_dict({"count": 12, "difficulty_weights": {"hard": 1.0}})
    assert {p.difficulty for p in plan_dataset(config)} == {"hard"}


def test_plan_is_deterministic_and_seed_dependent() -> None:
    config = config_from_dict({"count": 40})
    first = [(p.plate_type, p.difficulty, p.text.full, p.sample_seed) for p in plan_dataset(config)]
    again = [(p.plate_type, p.difficulty, p.text.full, p.sample_seed) for p in plan_dataset(config)]
    other = [(p.plate_type, p.difficulty, p.text.full, p.sample_seed) for p in plan_dataset(config_from_dict({"count": 40, "seed": 1}))]
    assert first == again
    assert first != other
    assert len({entry[2] for entry in first}) == 40  # identities unique


def test_plan_structure_follows_the_plate_type() -> None:
    plans = plan_dataset(config_from_dict({"count": 60}))
    assert {p.text.text_format for p in plans if p.plate_type == "type1b"} == {"type1b"}
    assert {p.text.text_format for p in plans if p.plate_type != "type1b"} == {"type1"}


def test_the_v1_type1b_option_no_longer_exists() -> None:
    with pytest.raises(ConfigError, match="unknown configuration key"):
        config_from_dict({"type1b_text_format": "competition"})


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_effect_selection_respects_cap_and_exclusivity(level: str) -> None:
    profile = GeneratorConfig().difficulties[level]
    seen: set[str] = set()
    for seed in range(400):
        enabled = select_effects(profile, np.random.default_rng(seed))
        assert len(enabled) <= profile.max_effects
        for first, second in EXCLUSIVE_EFFECTS:
            assert not (first in enabled and second in enabled)
        seen.update(enabled)
    assert seen  # every level applies some effects


def test_difficulty_mix_contains_clean_examples() -> None:
    profile = GeneratorConfig().difficulties["easy"]
    counts = [len(select_effects(profile, np.random.default_rng(seed))) for seed in range(400)]
    assert counts.count(0) > 150  # most easy samples carry no optional effect
