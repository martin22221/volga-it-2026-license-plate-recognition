"""Regression: effect provenance is truthful, and adverse-condition coverage.

Pilot A (1,000 images, 2026-09-14) found heavy noise applied to 60 images
but never listed in their ``effects`` record: the noise step wrote only to a
separate ``noise`` field.  Pilot A also found night, rain, snow, glare on the
plate and strong motion blur under-represented; 2.2.0 raised their default
probabilities without changing severities, budgets or the difficulty mix.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from dataset.generator.config import (
    DIFFICULTIES,
    EFFECT_SEVERITY,
    OPTIONAL_EFFECTS,
    GeneratorConfig,
    largest_remainder,
)
from dataset.generator.sample import generate_sample, plan_dataset, select_effects
from tests.generator_helpers import force_effects, small_config

ALL_OFF = {effect: 0.0 for effect in OPTIONAL_EFFECTS}


def _samples(config):
    return [generate_sample(config, plan, image_name="x.jpg") for plan in plan_dataset(config)]


# ------------------------------------------------------------ heavy-noise provenance


def test_heavy_noise_applied_is_recorded_in_effects() -> None:
    config = small_config(seed=5, count=4, difficulty_weights={"hard": 1.0},
                          difficulties=force_effects("hard", {**ALL_OFF, "heavy_noise": 1.0}))
    for sample in _samples(config):
        record = sample.record
        assert record["noise"]["chroma"] is True
        assert "heavy_noise" in record["effects"]
        assert record["effects"]["heavy_noise"] == record["noise"]
        assert record["legibility"]["attempts"][-1]["effects"] == sorted(record["effects"])


def test_heavy_noise_not_applied_is_not_recorded() -> None:
    config = small_config(seed=6, count=4, difficulty_weights={"hard": 1.0},
                          difficulties=force_effects("hard", {**ALL_OFF, "dirt": 1.0}))
    for sample in _samples(config):
        assert sample.record["noise"]["chroma"] is False
        assert "heavy_noise" not in sample.record["effects"]
        assert "dirt" in sample.record["effects"]


def test_effect_provenance_is_consistent_across_a_mixed_batch() -> None:
    config = small_config(seed=21, count=30)
    for sample in _samples(config):
        record = sample.record
        effects = record["effects"]
        assert set(effects) <= set(OPTIONAL_EFFECTS)
        # heavy noise: recorded exactly when the noise step applied chroma noise
        assert ("heavy_noise" in effects) == record["noise"]["chroma"]
        # the final attempt's log is the effects actually applied
        assert record["legibility"]["attempts"][-1]["effects"] == sorted(effects)
        # condition tags agree with the effect records they are derived from
        tags = set(record["conditions"])
        assert ("night" in tags) == ("night" in effects)
        assert ("day" in tags) == ("night" not in effects)
        for weather in ("rain", "snow"):
            assert (weather in tags) == (weather in effects)
        assert ("glare" in tags) == bool(effects.get("glare", {}).get("on_plate"))
        assert ("motion_blur" in tags) == (effects.get("motion_blur", {}).get("length_px", 0.0) >= 3.0)
        assert sum(EFFECT_SEVERITY[e] for e in effects) <= config.difficulties[record["difficulty"]].severity_budget + 1e-9


# ------------------------------------------------------------ condition coverage


def _selection_shares(config: GeneratorConfig, n: int = 6000) -> dict[str, float]:
    """Share of samples whose first-attempt selection includes each effect,
    over the default difficulty mix."""
    split = largest_remainder(n, config.difficulty_weights, DIFFICULTIES)
    counts: collections.Counter[str] = collections.Counter()
    for level in DIFFICULTIES:
        profile = config.difficulties[level]
        for seed in range(split[level]):
            counts.update(select_effects(profile, np.random.default_rng([seed, DIFFICULTIES.index(level)])))
    return {effect: counts[effect] / n for effect in OPTIONAL_EFFECTS}


def test_default_mix_selects_adverse_conditions_in_the_target_ranges() -> None:
    shares = _selection_shares(GeneratorConfig())
    # First-attempt selection over the 35/45/20 mix (2.2.0 measures night
    # 17.9 %, rain 9.5 %, snow 9.1 %, glare 13.8 %, motion blur 18.3 %; Pilot A
    # defaults gave night ~11 %, rain ~5 %, snow ~4 %).  Legibility redraws
    # raise final night to ~21-22 % in trial batches.
    assert 0.15 <= shares["night"] <= 0.25, shares
    assert 0.07 <= shares["rain"] <= 0.12, shares
    assert 0.07 <= shares["snow"] <= 0.12, shares
    assert 0.10 <= shares["glare"] <= 0.18, shares       # ~75 % of glare lands on the plate
    assert 0.12 <= shares["motion_blur"] <= 0.24, shares  # ~45 % reach the 3 px tag after legibility caps
    assert shares["dirt"] >= 0.18, shares                # not crowded out


def test_difficulty_mix_and_budgets_are_unchanged() -> None:
    config = GeneratorConfig()
    assert dict(config.difficulty_weights) == {"easy": 0.35, "medium": 0.45, "hard": 0.20}
    assert {lvl: p.severity_budget for lvl, p in config.difficulties.items()} == {"easy": 1.25, "medium": 2.0, "hard": 2.75}
    assert {lvl: p.max_effects for lvl, p in config.difficulties.items()} == {"easy": 1, "medium": 2, "hard": 3}
    assert EFFECT_SEVERITY["night"] == 1.5


def test_night_never_fits_the_easy_budget() -> None:
    """Documents a property of the budgets: night comes from medium/hard only."""
    profile = GeneratorConfig().difficulties["easy"]
    assert EFFECT_SEVERITY["night"] > profile.severity_budget
    assert all("night" not in select_effects(profile, np.random.default_rng(s)) for s in range(2000))


def test_easy_and_medium_motion_blur_starts_at_the_tag_threshold() -> None:
    config = GeneratorConfig()
    for level in ("easy", "medium"):
        assert config.difficulties[level].effect_range["motion_blur"]["length_px"][0] >= 3.0


def test_glare_is_a_localised_hotspot() -> None:
    for profile in GeneratorConfig().difficulties.values():
        assert profile.effect_range["glare"]["radius"][1] <= 0.45


@pytest.mark.parametrize("level", DIFFICULTIES)
def test_clean_samples_remain(level: str) -> None:
    profile = GeneratorConfig().difficulties[level]
    none = sum(1 for s in range(1000) if not select_effects(profile, np.random.default_rng(s)))
    # 2.2.0 measures ~43 % / ~4.5 % / ~2 % effect-free (Pilot A: 57 / 14 / 4.5 %);
    # the drop in medium is the accepted cost of the adverse-condition coverage.
    assert none >= {"easy": 380, "medium": 30, "hard": 10}[level]
