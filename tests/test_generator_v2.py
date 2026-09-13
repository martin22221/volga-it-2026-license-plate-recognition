"""Generator V2 regressions: type1B structure, layouts, legibility, fairness.

V1 drew ``type1b`` labels in the type 1 structure (``A123BC77``).  GOST
R 50577-2018 section 3.3 gives type 1B as ``MM 000 55``.  These tests pin the
correction and the V2 realism/legibility guarantees.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from dataset.generator.config import EFFECT_SEVERITY, EXCLUSIVE_EFFECTS, GeneratorConfig, config_from_dict
from dataset.generator.fonts import StrokeFontProvider
from dataset.generator.plate_text import (
    LETTERS,
    PlateIdentitySampler,
    PlateText,
    is_standard_plate,
    is_type1b_plate,
    sample_plate_text,
)
from dataset.generator.render import render_plate, sample_style
from dataset.generator.rng import make_rng
from dataset.generator.sample import generate_sample, plan_dataset, plate_contrast, select_effects
from dataset.generator.scene import build_vehicle_panel
from dataset.generator.templates import TWO_LINE, build_layout
from dataset.generator.writer import image_name
from src.validator import is_valid_plate
from tests.generator_helpers import small_config

L = f"[{LETTERS}]"
STANDARD = re.compile(rf"^{L}\d{{3}}{L}{{2}}\d{{2,3}}$")
TYPE1B = re.compile(rf"^{L}{{2}}\d{{3}}\d{{2}}$")
#: The V1 mistake: the type 1 structure on a type1b plate.
A000AA = re.compile(rf"^{L}\d{{3}}{L}{{2}}\d+$")


def _plans(seed: int, n: int = 30):
    return plan_dataset(config_from_dict({"seed": seed, "class_counts": {"type1": n, "type1a": n, "type1b": n}}))


# ------------------------------------------------------------ grammar


@pytest.mark.parametrize("seed", [20260913, 1, 2, 3])
def test_type1_generates_the_type1_grammar(seed: int) -> None:
    for plan in _plans(seed):
        if plan.plate_type == "type1":
            assert STANDARD.fullmatch(plan.text.full), plan.text.full
            assert is_valid_plate(plan.text.full, "type1")


@pytest.mark.parametrize("seed", [20260913, 1, 2, 3])
def test_type1a_generates_the_type1_grammar(seed: int) -> None:
    for plan in _plans(seed):
        if plan.plate_type == "type1a":
            assert STANDARD.fullmatch(plan.text.full), plan.text.full
            assert is_valid_plate(plan.text.full, "type1a")


@pytest.mark.parametrize("seed", [20260913, 1, 2, 3])
def test_type1b_generates_the_dedicated_mm000_region_grammar(seed: int) -> None:
    for plan in _plans(seed):
        if plan.plate_type == "type1b":
            assert TYPE1B.fullmatch(plan.text.full), plan.text.full
            assert is_valid_plate(plan.text.full, "type1b")
            assert plan.text.text_format == "type1b"


def test_type1b_never_produces_the_a000aa_region_form() -> None:
    rng = make_rng(20260913, "identity")
    for _ in range(20_000):
        full = sample_plate_text(rng, "type1b", 0.9).full
        assert not A000AA.fullmatch(full), full
        assert not is_standard_plate(full)
        assert is_type1b_plate(full)


def test_type1b_region_is_two_digits_even_when_three_are_likely_elsewhere() -> None:
    sampler = PlateIdentitySampler(make_rng(5, "identity"), three_digit_probability=1.0)
    assert all(len(sampler.sample("type1b").region) == 2 for _ in range(500))
    assert all(len(sampler.sample("type1").region) == 3 for _ in range(500))


def test_type1b_structure_cannot_be_constructed_wrongly() -> None:
    with pytest.raises(ValueError):
        PlateText("A", "123", "BC", "77", "type1b")
    with pytest.raises(ValueError):
        PlateText("AB", "123", "", "77", "type1")


def test_v1_option_to_request_the_old_structure_is_gone() -> None:
    assert not hasattr(GeneratorConfig(), "type1b_text_format")


# ------------------------------------------------------------ appearance


@pytest.mark.parametrize("seed", range(6))
def test_type1b_is_yellow_and_type1_type1a_are_white(seed: int) -> None:
    font = StrokeFontProvider()
    texts = {
        "type1": PlateText("A", "123", "BC", "77", "type1"),
        "type1a": PlateText("A", "123", "BC", "77", "type1"),
        "type1b": PlateText("AB", "123", "", "77", "type1b"),
    }
    for plate_type, text in texts.items():
        layout = build_layout(plate_type, text)
        plate = render_plate(layout, sample_style(np.random.default_rng(seed), layout.field_colour), font, 0.8, supersample=2)
        field = plate.rgb[(plate.ink < 0.02) & (plate.alpha > 0.99)]
        r, g, b = np.median(field, axis=0)
        if plate_type == "type1b":
            assert r > 0.8 and b < 0.25 and r - b > 0.6, (plate_type, r, g, b)
        else:
            assert min(r, g, b) > 0.75 and max(r, g, b) - min(r, g, b) < 0.08, (plate_type, r, g, b)


def test_type1a_uses_the_two_line_layout_with_region_and_rus_bottom_right() -> None:
    text = PlateText("K", "456", "MH", "152", "type1")
    layout = build_layout("type1a", text)
    labels = layout.label_glyphs
    top = sorted((g for g in labels if g.line == 0), key=lambda g: g.x)
    bottom = sorted((g for g in labels if g.line == 1), key=lambda g: g.x)
    assert "".join(g.char for g in top) == "K456"
    assert "".join(g.char for g in bottom) == "MH152"
    letters = [g for g in bottom if g.char.isalpha()]
    region = [g for g in bottom if g.char.isdigit()]
    assert max(g.x1 for g in letters) < min(g.x for g in region)  # letters bottom-left
    rus = [g for g in layout.glyphs if g.position is None]
    assert "".join(g.char for g in rus) == "RUS"
    assert min(g.x for g in rus) > max(g.x1 for g in region)  # RUS area right of the region
    assert layout.flag is not None and layout.flag.x > max(g.x1 for g in region)
    assert (layout.width, layout.height) == (TWO_LINE.width, TWO_LINE.height)


def test_bolts_sit_in_gaps_never_on_characters() -> None:
    for plate_type, text in (
        ("type1", PlateText("A", "123", "BC", "77", "type1")),
        ("type1a", PlateText("A", "123", "BC", "777", "type1")),
        ("type1b", PlateText("AB", "123", "", "77", "type1b")),
    ):
        layout = build_layout(plate_type, text)
        assert layout.bolt_sites, plate_type
        for bx, by, br in layout.bolt_sites:
            for g in layout.glyphs:
                overlap_x = bx + br > g.x and bx - br < g.x1
                overlap_y = by + br > g.y and by - br < g.y1
                assert not (overlap_x and overlap_y), (plate_type, g.char)


# ------------------------------------------------------------ fairness


@pytest.mark.parametrize("seed", range(10))
def test_vehicle_and_scene_do_not_depend_on_plate_type(seed: int) -> None:
    """Same stream + same plate size => same vehicle for type1 and type1b."""
    font = StrokeFontProvider()
    records = {}
    for plate_type, text in (("type1", PlateText("A", "123", "BC", "77", "type1")), ("type1b", PlateText("AB", "123", "", "77", "type1b"))):
        layout = build_layout(plate_type, text)
        plate = render_plate(layout, sample_style(np.random.default_rng(0), layout.field_colour), font, 0.4, supersample=1)
        canvas = build_vehicle_panel(np.random.default_rng(seed), plate, plate_type)
        records[plate_type] = (canvas.record, canvas.plate_origin, canvas.alpha.shape)
    assert records["type1"] == records["type1b"]


# ------------------------------------------------------------ difficulty


def test_default_difficulty_mix_is_35_45_20() -> None:
    assert dict(GeneratorConfig().difficulty_weights) == {"easy": 0.35, "medium": 0.45, "hard": 0.20}
    plans = plan_dataset(config_from_dict({"class_counts": {"type1": 20, "type1a": 20, "type1b": 20}}))
    counts = {level: sum(p.difficulty == level for p in plans) for level in ("easy", "medium", "hard")}
    assert counts == {"easy": 21, "medium": 27, "hard": 12}


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_effects_fit_the_severity_budget(level: str) -> None:
    profile = GeneratorConfig().difficulties[level]
    for seed in range(1000):
        enabled = select_effects(profile, np.random.default_rng(seed))
        assert sum(EFFECT_SEVERITY[e] for e in enabled) <= profile.severity_budget + 1e-9
        assert len(enabled) <= profile.max_effects
        for first, second in EXCLUSIVE_EFFECTS:
            assert not (first in enabled and second in enabled)


def test_hard_never_stacks_night_with_two_other_severe_effects() -> None:
    profile = GeneratorConfig().difficulties["hard"]
    severe = {"night", "motion_blur", "defocus", "occlusion", "glare", "heavy_noise", "low_light"}
    for seed in range(2000):
        enabled = set(select_effects(profile, np.random.default_rng(seed)))
        if "night" in enabled:
            assert len(enabled & severe) <= 2


def test_plate_contrast_measure() -> None:
    image = np.full((40, 100, 3), 0.9, np.float32)
    ink = np.zeros((40, 100), np.float32)
    ink[10:30, 20:30] = 1.0
    image[ink > 0] = 0.1
    mask = np.ones((40, 100), np.float32)
    assert plate_contrast(image, mask, ink) == pytest.approx(0.8, abs=1e-3)
    assert plate_contrast(np.full_like(image, 0.5), mask, ink) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_every_sample_meets_its_legibility_floor(level: str) -> None:
    config = small_config(seed=11, count=15, difficulty_weights={level: 1.0})
    for plan in plan_dataset(config):
        sample = generate_sample(config, plan, image_name="x.jpg")
        legibility = sample.record["legibility"]
        clean_fallback = legibility["attempts"][-1]["effects"] == []
        assert legibility["plate_contrast"] >= legibility["min_required"] or clean_fallback
        assert len(sample.record["hidden_positions"]) <= config.max_hidden_characters


def test_retries_are_deterministic() -> None:
    config = small_config(seed=11, count=6, difficulty_weights={"hard": 1.0})
    plans = plan_dataset(config)
    first = [generate_sample(config, p, image_name="x.jpg") for p in plans]
    again = [generate_sample(config, p, image_name="x.jpg") for p in plans]
    assert [s.image_bytes for s in first] == [s.image_bytes for s in again]
    assert [s.record for s in first] == [s.record for s in again]


# ------------------------------------------------------------ V1 separation


def test_v2_file_names_carry_the_generator_major_version() -> None:
    assert image_name(20260913, 7) == "syn_v2_20260913_00007.jpg"


def test_vehicles_stand_on_the_ground() -> None:
    config = small_config(seed=21, count=24)
    records = [generate_sample(config, plan, image_name="x.jpg").record for plan in plan_dataset(config)]
    grounded = [r["vehicle"]["grounded"] for r in records]
    assert all("ground_y" in r["background"] for r in records)
    assert sum(grounded) >= 0.8 * len(grounded)
