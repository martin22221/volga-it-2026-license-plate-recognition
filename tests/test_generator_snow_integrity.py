"""Regression: precipitation must never change a character's identity.

V2 QA (2026-09-13) found sample #42 of the seed-20260913 review batch
(``O935TB97``, type1a, medium, snow): a near-opaque flake (alpha 0.74) erased
the right stroke of the ``O``, which then read as ``C`` while ``plate_num``
still said ``O``.  Snow and rain particles are now capped at
``TEXT_PARTICLE_ALPHA`` over plate ink, for every character.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import dataset.generator.sample as sample_module
from dataset.generator.config import config_from_dict
from dataset.generator.fonts import StrokeFontProvider
from dataset.generator.glyphs import PLATE_CHARACTERS
from dataset.generator.photometric import TEXT_PARTICLE_ALPHA, precipitation, text_guard

FIELD, INK = 0.9, 0.08


def _effective_alpha(before: np.ndarray, after: np.ndarray, haze: float) -> np.ndarray:
    """Particle opacity implied by one precipitation pass (inverts its compositing)."""
    hazed = before * (1.0 - haze) + haze * 0.75
    return (after - hazed) / np.maximum(0.9 - hazed, 1e-6)


def _glyph_plate(char: str, height_px: int = 16, copies: tuple[int, int] = (9, 7)) -> tuple[np.ndarray, np.ndarray]:
    """A white sheet tiled with one character at small-plate size (image, ink).

    Tiled so the frame is large enough for the snow density to place many
    flakes, and many of them land on strokes.
    """
    width = int(height_px * 0.85)
    cell_w, cell_h = width + 8, height_px + 8
    mask = Image.new("L", (cell_w * copies[0], cell_h * copies[1]), 0)
    font = StrokeFontProvider()
    for i in range(copies[0]):
        for j in range(copies[1]):
            x, y = i * cell_w + 4, j * cell_h + 4
            font.draw(mask, char, (x, y, x + width, y + height_px), stroke_px=max(1.5, height_px * 0.15))
    ink = np.asarray(mask, np.float32) / 255.0
    image = np.full(ink.shape + (3,), FIELD, np.float32)
    image += ink[..., None] * (INK - image)
    return image, ink


# ------------------------------------------------------------------ mechanism


def test_heavy_snow_without_protection_can_erase_strokes() -> None:
    """Sensitivity check: the scenario really does erase ink when unguarded."""
    erased = 0
    for char in PLATE_CHARACTERS:
        image, ink = _glyph_plate(char)
        core = ink > 0.6
        for seed in range(12):
            out, _ = precipitation(image, np.random.default_rng(seed), 1.0, "snow")
            erased += int((out[..., 0][core] > (FIELD + INK) / 2).any())
    assert erased > 0


@pytest.mark.parametrize("char", list(PLATE_CHARACTERS))
def test_every_character_keeps_its_strokes_under_heavy_snow(char: str) -> None:
    image, ink = _glyph_plate(char)
    core = ink > 0.6
    guard = text_guard(ink)
    for seed in range(12):
        out, record = precipitation(image, np.random.default_rng(seed), 1.0, "snow", protect=guard)
        assert record["text_protected"] is True
        alpha = _effective_alpha(image[..., 0], out[..., 0], record["haze"])
        assert alpha[core].max() <= TEXT_PARTICLE_ALPHA + 1e-4
        # every stroke pixel stays nearer the ink level than the field level
        assert out[..., 0][core].max() < (FIELD + INK) / 2


@pytest.mark.parametrize("kind", ["snow", "rain"])
def test_protection_changes_nothing_away_from_the_characters(kind: str) -> None:
    image, ink = _glyph_plate("B", height_px=30, copies=(3, 2))
    image = np.pad(image, ((0, 60), (0, 120), (0, 0)), constant_values=0.3)  # dark scene around the plate
    ink = np.pad(ink, ((0, 60), (0, 120)))
    guard = text_guard(ink)
    plain, record = precipitation(image, np.random.default_rng(3), 1.0, kind)
    guarded, _ = precipitation(image, np.random.default_rng(3), 1.0, kind, protect=guard)
    free = guard == 0.0
    np.testing.assert_array_equal(plain[free], guarded[free])
    # Precipitation is still drawn off the text: realistic snow/rain remains.
    hazed = image * (1.0 - record["haze"]) + record["haze"] * 0.75
    assert (plain[free] - hazed[free] > 0.05).any()


def test_text_guard_covers_antialiased_edges() -> None:
    ink = np.zeros((9, 9), np.float32)
    ink[4, 4] = 1.0
    guard = text_guard(ink)
    assert guard[3:6, 3:6].min() == pytest.approx(1.0, abs=1 / 255)
    assert guard[0, 0] == 0.0


# ------------------------------------------------------------------ the V2 failure


def _demo_config():
    return config_from_dict({"seed": 20260913, "class_counts": {"type1": 20, "type1a": 20, "type1b": 20}})


def _run_capturing(monkeypatch: pytest.MonkeyPatch, plan, *, guarded: bool) -> dict:
    captured: dict = {}
    real_precipitation = sample_module.precipitation
    real_guard = sample_module.text_guard

    def spy(image, rng, density, kind, *, protect=None):
        captured["before"] = image.copy()
        out, record = real_precipitation(image, rng, density, kind, protect=protect)
        captured.update(after=out, record=record)
        return out, record

    def guard(ink):
        captured["guard"] = real_guard(ink)
        return captured["guard"] if guarded else np.zeros_like(ink)

    monkeypatch.setattr(sample_module, "precipitation", spy)
    monkeypatch.setattr(sample_module, "text_guard", guard)
    captured["sample"] = sample_module.generate_sample(_demo_config(), plan, image_name="x.jpg")
    monkeypatch.undo()
    return captured


def test_the_v2_snow_case_no_longer_erases_character_strokes(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = sample_module.plan_dataset(_demo_config())[42]
    assert (plan.plate_type, plan.text.full) == ("type1a", "O935TB97")

    v2 = _run_capturing(monkeypatch, plan, guarded=False)   # V2 behaviour
    fixed = _run_capturing(monkeypatch, plan, guarded=True)
    assert "snow" in fixed["sample"].record["effects"]
    assert fixed["sample"].record["effects"]["snow"]["text_protected"] is True

    strokes = fixed["guard"] >= 0.999  # character ink (and plate border) in output pixels
    haze = fixed["record"]["haze"]
    v2_alpha = _effective_alpha(v2["before"][..., 0], v2["after"][..., 0], haze)[strokes]
    fixed_alpha = _effective_alpha(fixed["before"][..., 0], fixed["after"][..., 0], haze)[strokes]
    assert v2_alpha.max() > 0.6  # the flake that turned O into C
    assert fixed_alpha.max() <= TEXT_PARTICLE_ALPHA + 1e-3
    # The label is unchanged and now honest: nothing was hidden.
    assert fixed["sample"].annotation.plate_num == "O935TB97"


def test_samples_without_precipitation_are_unchanged_by_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _demo_config()
    checked = 0
    for plan in sample_module.plan_dataset(config)[:8]:
        a = sample_module.generate_sample(config, plan, image_name="x.jpg")
        if {"snow", "rain"} & set(a.record["effects"]):
            continue
        monkeypatch.setattr(sample_module, "text_guard", lambda ink: np.zeros_like(ink))
        b = sample_module.generate_sample(config, plan, image_name="x.jpg")
        monkeypatch.undo()
        assert a.image_bytes == b.image_bytes
        checked += 1
    assert checked >= 5
