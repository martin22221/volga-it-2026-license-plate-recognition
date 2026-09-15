"""Regression: occlusion labels must follow what the character's ink shows.

Production QA of the 2.2.0 batch (seed 2026091401, 12,000 images,
2026-09-14) found sample #9880, a hard type1a plate with the full text
``O366EO126``, labelled ``O#66EO126``.  A white snow clump covered 27 % of
the ``O``'s ink.  That was its whole right stroke, so the ``O`` read as
``C``, yet the label kept ``O``.  2.2.0 hid a character only when more than
35 % of its *bounding box* was covered, and glyph ink fills only part of
that box.  2.3.0 decides from the character's ink, against every character
that could stand in its place (:mod:`dataset.generator.occlusion_labels`).

Every family below also runs the 2.2.0 box rule as a negative control, so
the tests are shown to catch it.  A second reader, which is not part of the
rule, confirms what "reads as another character" means: cross-correlation at
0.2 stroke widths, with its own blur.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import PIL
import pytest

import dataset.generator.scene as scene_module
from dataset.generator.config import config_from_dict
from dataset.generator.fonts import StrokeFontProvider
from dataset.generator.glyphs import PLATE_CHARACTERS
from dataset.generator.occlusion_labels import (
    MAX_INK_COVERED,
    MIN_EVIDENCE_RETAINED,
    MIN_READER_MARGIN,
    CharacterVisibility,
    GlyphReference,
    OcclusionJudge,
    alternatives,
    assess_character,
    covered_patch,
    glyph_reference,
)
from dataset.generator.plate_text import LETTERS, PlateText
from dataset.generator.render import GlyphCell, PlateStyle, glyph_ink, render_plate
from dataset.generator.sample import generate_sample, plan_dataset
from dataset.generator.scene import PlaneCanvas, apply_occlusion, build_vehicle_panel
from dataset.generator.templates import DIGIT, LETTER, REGION_2, REGION_3, build_layout
from tests.generator_helpers import force_effects, small_config

FONT = StrokeFontProvider()
FACTOR = 4
#: Plate resolutions: production renders plates at 0.225-0.78 px/mm; 1.5 is a large close-up.
SCALES = (0.22, 0.5, 1.5)
CELLS = {"letter": LETTER, "digit": DIGIT, "region2": REGION_2, "region3": REGION_3}

#: Byte-level constants below were recorded with the pinned libraries.
PINNED = pytest.mark.skipif(
    (np.__version__, PIL.__version__) != ("2.5.3", "11.3.0"),
    reason="byte-level constants were recorded with numpy 2.5.3 / Pillow 11.3.0 (dataset/generator/requirements.txt)",
)


# ------------------------------------------------------------------ helpers


def _cell_kinds(char: str) -> list[str]:
    """Every cell a character can be drawn in: letters in letter cells, digits in the three digit cells."""
    return ["letter"] if char in LETTERS else ["digit", "region2", "region3"]


def _cell(char: str, kind: str, scale: float, position: int = 0) -> GlyphCell:
    metrics = CELLS[kind]
    x0 = y0 = 5.0 * scale
    box = (x0 * FACTOR, y0 * FACTOR, (x0 + metrics.width * scale) * FACTOR, (y0 + metrics.height * scale) * FACTOR)
    return GlyphCell(position, char, box, metrics.stroke * scale * FACTOR, FACTOR)


def _references():
    for char in PLATE_CHARACTERS:
        for kind in _cell_kinds(char):
            for scale in SCALES:
                yield glyph_reference(_cell(char, kind, scale), FONT)


def _ink_of(ref: GlyphReference, char: str) -> np.ndarray:
    """``char`` drawn in ``ref``'s cell, over the same patch as ``ref.ink``."""
    pad = math.floor(ref.cell.box[0] / ref.cell.factor) - ref.origin[0]
    ink, origin = glyph_ink(FONT, ref.cell, char, pad)
    assert origin == ref.origin and ink.shape == ref.shape
    return ink


def _inside(ref: GlyphReference, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    out[ref.window] = mask[ref.window]
    return out


def _grow(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.astype(np.float32)
    for _ in range(steps):
        p = np.pad(out, 1)
        out = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    return out


def _legacy_hides(ref: GlyphReference, covered: np.ndarray) -> bool:
    """2.2.0: hidden when the occluder covers more than 35 % of the glyph box."""
    return float(covered[ref.window].mean()) > 0.35


class LegacyJudge:
    """2.2.0's box rule behind the 2.3.0 judge interface (negative control)."""

    def __init__(self, cells, font) -> None:
        self.cells = list(cells)

    def assess(self, occluder, plate_origin):
        ox, oy = plate_origin
        out = []
        for cell in self.cells:
            x0, y0, x1, y1 = (v / cell.factor for v in cell.box)
            xa, xb = int(ox + x0), int(np.ceil(ox + x1))
            ya, yb = int(oy + y0), int(np.ceil(oy + y1))
            covered = float(occluder[ya:yb, xa:xb].mean()) if xb > xa and yb > ya else 0.0
            if covered > 0.0:
                out.append(CharacterVisibility(cell.position, cell.char, 0.0, 1.0, "", 0.0, "", covered <= 0.35))
        return out


def _gauss(image: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    rows = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, image.astype(np.float64))
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, rows)


def _independent_read(ref: GlyphReference, observed: np.ndarray) -> tuple[str, float]:
    """(best candidate, true character's score minus its best rival's) -- not the rule's reader."""
    sigma = 0.2 * ref.cell.stroke_px

    def unit(image: np.ndarray) -> np.ndarray:
        v = _gauss(image, sigma)[ref.window].ravel()
        v = v - v.mean()
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    obs = unit(observed)
    candidates = [ref.cell.char, *alternatives(ref.cell.char)]
    scores = {c: float(unit(ref.ink if c == ref.cell.char else _ink_of(ref, c)) @ obs) for c in candidates}
    rival = max(candidates[1:], key=scores.get)
    best = max(candidates, key=scores.get)
    return best, scores[ref.cell.char] - scores[rival]


# ------------------------------------------------------------------ the rule's inputs


@pytest.mark.parametrize("plate_type, text", [
    ("type1", PlateText("O", "386", "CE", "59", "type1")),
    ("type1a", PlateText("B", "704", "HK", "123", "type1")),
    ("type1b", PlateText("MX", "861", "", "47", "type1b")),
])
def test_glyph_ink_reproduces_the_rendered_characters(plate_type: str, text: PlateText) -> None:
    """The rule measures exactly the ink the renderer drew for each character."""
    style = PlateStyle((0.95, 0.95, 0.95), (0.05, 0.05, 0.05), 1.0, 0.0)
    for scale in (0.3, 0.78):
        plate = render_plate(build_layout(plate_type, text), style, FONT, scale, supersample=4)
        assert [c.position for c in plate.glyph_cells] == list(range(len(text.full)))
        assert "".join(c.char for c in plate.glyph_cells) == text.full
        for cell in plate.glyph_cells:
            ink, (px, py) = glyph_ink(FONT, cell, cell.char)
            region = plate.ink[py : py + ink.shape[0], px : px + ink.shape[1]]
            np.testing.assert_allclose(ink, region, atol=1.5 / 255)


def test_alternatives_follow_the_plate_grammar() -> None:
    for char in PLATE_CHARACTERS:
        pool = alternatives(char)
        assert char not in pool
        assert set(pool) | {char} == (set(LETTERS) if char in LETTERS else set("0123456789"))
    with pytest.raises(ValueError):
        alternatives("R")


# ------------------------------------------------------------------ the production failure


def _production_config():
    return config_from_dict({"seed": 2026091401, "class_counts": {"type1": 2400, "type1a": 4200, "type1b": 5400}})


PRODUCTION_9880_SHA256 = "fe7c249a03988440033d210c644de23071e8618bf650ee38203ebbc36134480a"


def test_production_sample_9880_is_relabelled() -> None:
    plan = plan_dataset(_production_config())[9880]
    assert (plan.plate_type, plan.difficulty, plan.text.full) == ("type1a", "hard", "O366EO126")
    sample = generate_sample(_production_config(), plan, image_name="x.jpg")
    occlusion = sample.record["effects"]["occlusion"]
    assert occlusion["kind"] == "blob" and occlusion["attempts"] == 1
    judged = {c["position"]: c for c in occlusion["characters"]}
    # The O: 27 % of its ink covered, all of it the right stroke that tells O from C.
    o = judged[0]
    assert 0.25 < o["ink_covered"] < 0.30
    assert o["closest"] == "C" and o["evidence_retained"] < 0.2 and o["rival"] == "C" and o["reader_margin"] < 0
    assert o["hidden"]
    # The 3: only 20 % of its ink is covered, but the clump covers the left of
    # the cell, which is where an 8 would differ.  It cannot be told from 8.
    three = judged[1]
    assert 0.15 < three["ink_covered"] < 0.25 and three["closest"] == "8" and three["hidden"]
    assert set(judged) == {0, 1}  # nothing else is under the clump
    assert sample.annotation.plate_num == "##66EO126"


@PINNED
def test_production_sample_9880_keeps_its_pixels_and_the_legacy_rule_reproduces_the_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same image bytes as the production file; only the label changes.

    Negative control: with the 2.2.0 box rule swapped back in, the generator
    reproduces the production label ``O#66EO126`` exactly.
    """
    config = _production_config()
    plan = plan_dataset(config)[9880]
    fixed = generate_sample(config, plan, image_name="x.jpg")
    assert hashlib.sha256(fixed.image_bytes).hexdigest() == PRODUCTION_9880_SHA256
    monkeypatch.setattr(scene_module, "OcclusionJudge", LegacyJudge)
    legacy = generate_sample(config, plan, image_name="x.jpg")
    assert legacy.annotation.plate_num == "O#66EO126"
    assert hashlib.sha256(legacy.image_bytes).hexdigest() == PRODUCTION_9880_SHA256


# ------------------------------------------------------------------ every character, every rival


def test_a_character_that_cannot_be_told_from_a_rival_is_hidden() -> None:
    """Cover exactly where a character and a rival differ: what remains is the same for both."""
    cases = legacy_kept = 0
    for ref in _references():
        for rival in alternatives(ref.cell.char):
            difference = np.abs(ref.ink - _ink_of(ref, rival)) > 0.05
            covered = _inside(ref, _grow(difference))
            verdict = assess_character(ref, covered)
            assert not verdict.readable, (ref.cell.char, rival, ref.cell.stroke_px, verdict)
            cases += 1
            legacy_kept += not _legacy_hides(ref, covered)
    assert cases == 3 * (12 * 11 + 3 * 10 * 9)
    assert legacy_kept >= 10  # negative control: the box rule keeps some of these labels (22 of 1,206 measured)


def test_removing_an_identifying_stroke_never_keeps_the_label() -> None:
    """Cover the strokes a character has and a rival lacks.

    Whenever the independent reader then reads another character, the label must be ``#``.
    """
    became_other = legacy_kept = 0
    pairs: set[tuple[str, str]] = set()
    for ref in _references():
        for rival in alternatives(ref.cell.char):
            covered = _inside(ref, _grow((ref.ink > 0.1) & (_grow(_ink_of(ref, rival)) < 0.1)))
            if (ref.ink * covered).sum() < 0.05 * ref.ink.sum():
                continue
            best, margin = _independent_read(ref, ref.ink * (1.0 - covered))
            if margin > 0:
                continue  # still reads as itself
            became_other += 1
            pairs.add((ref.cell.char, best))
            verdict = assess_character(ref, covered)
            assert not verdict.readable, (ref.cell.char, best, ref.cell.stroke_px, verdict)
            legacy_kept += not _legacy_hides(ref, covered)
    assert became_other >= 300  # 463 measured
    assert {("O", "C"), ("8", "3"), ("6", "5"), ("B", "P"), ("E", "C")} <= pairs  # among many
    assert legacy_kept >= 150  # negative control: the box rule kept most of them (292 of 463 measured)


@pytest.mark.parametrize("char", list(PLATE_CHARACTERS))
def test_small_partial_occlusion_keeps_a_readable_label(char: str) -> None:
    """Discs 0.75 stroke widths across, on a grid over the cell: most leave the character readable.

    Every kept label is confirmed by the independent reader.
    """
    kept_cover = []
    for kind in _cell_kinds(char):
        for scale in SCALES:
            ref = glyph_reference(_cell(char, kind, scale), FONT)
            ys, xs = ref.window
            yy, xx = np.mgrid[0 : ref.shape[0], 0 : ref.shape[1]]
            radius = 0.75 * ref.cell.stroke_px
            verdicts = []
            for cy in np.linspace(ys.start, ys.stop - 1, 7):
                for cx in np.linspace(xs.start, xs.stop - 1, 5):
                    covered = _inside(ref, ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2).astype(np.float32))
                    verdict = assess_character(ref, covered)
                    if verdict.ink_covered == 0.0:
                        continue
                    verdicts.append(verdict.readable)
                    assert not _legacy_hides(ref, covered)
                    if verdict.readable:
                        kept_cover.append(verdict.ink_covered)
                        assert _independent_read(ref, ref.ink * (1.0 - covered))[0] == char
            assert sum(verdicts) >= 0.8 * len(verdicts), (char, kind, scale)
    assert max(kept_cover) >= 0.07  # partial: real ink is covered, and the label is kept


@pytest.mark.parametrize("char", list(PLATE_CHARACTERS))
def test_heavy_occlusion_is_hidden(char: str) -> None:
    for kind in _cell_kinds(char):
        for scale in SCALES:
            ref = glyph_reference(_cell(char, kind, scale), FONT)
            ys, xs = ref.window
            my, mx = (ys.start + ys.stop) // 2, (xs.start + xs.stop) // 2
            for half in (np.s_[:, :mx], np.s_[:, mx:], np.s_[:my, :], np.s_[my:, :]):
                covered = np.zeros(ref.shape, np.float32)
                covered[half] = 1.0
                assert not assess_character(ref, _inside(ref, covered)).readable
            # Columns from the left until more than MAX_INK_COVERED of the ink is under them.
            column_ink = np.cumsum(ref.ink.sum(axis=0)) / ref.ink.sum()
            stop = int(np.searchsorted(column_ink, MAX_INK_COVERED + 0.02)) + 1
            covered = np.zeros(ref.shape, np.float32)
            covered[:, :stop] = 1.0
            verdict = assess_character(ref, covered)
            assert verdict.ink_covered > MAX_INK_COVERED and not verdict.readable


# ------------------------------------------------------------------ real layouts and occluders

TEXTS = {
    "type1": PlateText("O", "386", "CE", "59", "type1"),
    "type1a": PlateText("B", "704", "HK", "123", "type1"),
    "type1b": PlateText("MX", "861", "", "47", "type1b"),
}


def _panel(plate_type: str, scale: float, seed: int):
    text = TEXTS[plate_type]
    field = (0.93, 0.78, 0.05) if plate_type == "type1b" else (0.94, 0.94, 0.93)
    style = PlateStyle(field, (0.06, 0.06, 0.06), 1.0, 0.05)
    plate = render_plate(build_layout(plate_type, text), style, FONT, scale, supersample=4)
    return plate, build_vehicle_panel(np.random.default_rng(seed), plate, plate_type)


def _occlude(monkeypatch, plate, canvas, seed, extent, judge=None):
    """apply_occlusion with its final occluder captured (and optionally another judge)."""
    captured = {}
    real_mask = scene_module._occluder_mask

    def spy(canvas_, rng_, extent_):
        captured["occluder"], kind, colour = real_mask(canvas_, rng_, extent_)
        return captured["occluder"], kind, colour

    monkeypatch.setattr(scene_module, "_occluder_mask", spy)
    if judge is not None:
        monkeypatch.setattr(scene_module, "OcclusionJudge", judge)
    hidden, record = apply_occlusion(canvas, plate.glyph_cells, np.random.default_rng(seed), extent, font=FONT)
    monkeypatch.undo()
    return hidden, record, captured["occluder"]


@pytest.mark.parametrize("plate_type", ["type1", "type1a", "type1b"])
@pytest.mark.parametrize("scale", [0.25, 0.5, 0.78])
def test_generator_occluders_on_real_plates(monkeypatch: pytest.MonkeyPatch, plate_type: str, scale: float) -> None:
    """Bars, clumps and corner objects on every layout: every kept label reads as itself."""
    kinds, touched = set(), 0
    for seed in range(16):
        plate, canvas = _panel(plate_type, scale, seed)
        extent = 0.06 + 0.16 * (seed % 8) / 7
        hidden, record, occluder = _occlude(monkeypatch, plate, canvas, 1000 + seed, extent)
        kinds.add(record["kind"])
        judged = {c["position"]: c for c in record["characters"]}
        assert hidden == {p for p, c in judged.items() if c["hidden"]}
        assert record["hidden_positions"] == sorted(hidden) and len(hidden) <= 2
        for cell in plate.glyph_cells:
            ref = glyph_reference(cell, FONT)
            covered = covered_patch(occluder, ref, canvas.plate_origin)
            if covered[ref.window].max() <= 0.0:
                assert cell.position not in judged  # untouched characters are not judged
                continue
            touched += 1
            c = judged[cell.position]
            meets = (c["ink_covered"] <= MAX_INK_COVERED and c["evidence_retained"] >= MIN_EVIDENCE_RETAINED
                     and c["reader_margin"] >= MIN_READER_MARGIN)
            assert c["hidden"] is (not meets)
            if not c["hidden"]:
                for observed in (ref.ink * (1 - covered), ref.ink * (1 - covered) + _inside(ref, covered)):
                    assert _independent_read(ref, observed)[0] == cell.char, (plate_type, scale, seed, c)
    assert touched > 0 and len(kinds) >= 2


def test_the_legacy_rule_keeps_misread_characters_on_real_plates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control on real layouts: 2.2.0 kept labels the reader sees as other characters."""
    misread_kept = 0
    for plate_type in ("type1", "type1a", "type1b"):
        for seed in range(40):
            plate, canvas = _panel(plate_type, 0.3, seed)
            hidden, _record, occluder = _occlude(monkeypatch, plate, canvas, 2000 + seed, 0.12, judge=LegacyJudge)
            ox, oy = canvas.plate_origin
            for cell in plate.glyph_cells:
                x0, y0, x1, y1 = (int(round(v / cell.factor)) for v in cell.box)
                if cell.position in hidden or occluder[oy + y0 : oy + y1, ox + x0 : ox + x1].max() <= 0:
                    continue
                ref = glyph_reference(cell, FONT)
                covered = covered_patch(occluder, ref, canvas.plate_origin)
                if covered[ref.window].max() > 0 and _independent_read(ref, ref.ink * (1 - covered))[0] != cell.char:
                    misread_kept += 1
                    assert not assess_character(ref, covered).readable  # 2.3.0 hides it
    assert misread_kept >= 3  # 5 measured


def test_occluders_reaching_past_the_plate_and_canvas_edges() -> None:
    """Edge and corner objects: the first and last characters are the nearest to the plate border."""
    text = TEXTS["type1"]
    style = PlateStyle((0.94, 0.94, 0.93), (0.06, 0.06, 0.06), 1.0, 0.0)
    plate = render_plate(build_layout("type1", text), style, FONT, 0.4, supersample=4)
    # The plate fills the canvas exactly: nothing lies beyond its border.
    canvas = PlaneCanvas(plate.rgb.copy(), plate.alpha.copy(), plate.alpha.copy(), plate.ink.copy(), (0, 0),
                         (plate.width, plate.height), {})
    judge = OcclusionJudge(plate.glyph_cells, FONT)
    first, last = plate.glyph_cells[0], plate.glyph_cells[-1]

    def cover_x(x0: float, x1: float) -> np.ndarray:
        occluder = np.zeros(canvas.alpha.shape, np.float32)
        occluder[:, max(0, int(x0)) : min(canvas.alpha.shape[1], int(x1))] = 1.0
        return occluder

    # An object from the left edge over the whole first character.
    fx1 = first.box[2] / first.factor
    judged = {c.position: c for c in judge.assess(cover_x(0, fx1 + 1), canvas.plate_origin)}
    assert not judged[0].readable and set(judged) == {0}
    # From the right edge over the last region digit.
    lx0 = last.box[0] / last.factor
    judged = {c.position: c for c in judge.assess(cover_x(lx0 - 1, plate.width), canvas.plate_origin)}
    assert not judged[last.position].readable and set(judged) == {last.position}
    # A thin sliver along the left plate edge touches no character cell.
    assert judge.assess(cover_x(0, 3), canvas.plate_origin) == []
    # A plate hanging off the canvas: the part of a patch outside the canvas counts as uncovered.
    ref = glyph_reference(first, FONT)
    shift = -(ref.origin[0] + ref.shape[1] // 2)  # the canvas starts in the middle of the patch
    occluder = np.ones((plate.height, plate.width), np.float32)
    patch = covered_patch(occluder, ref, (shift, 0))
    columns = np.arange(ref.shape[1]) + ref.origin[0] + shift
    rows = np.arange(ref.shape[0]) + ref.origin[1]
    on_canvas = np.outer((rows >= 0) & (rows < plate.height), (columns >= 0) & (columns < plate.width))
    assert 0 < on_canvas.sum() < on_canvas.size
    np.testing.assert_array_equal(patch, on_canvas.astype(np.float32))
    assert not assess_character(ref, patch).readable  # half of it is gone
    # Occluders drawn by the generator, clipped at the canvas edge, run through the whole path.
    for seed in range(12):
        c = PlaneCanvas(plate.rgb.copy(), plate.alpha.copy(), plate.alpha.copy(), plate.ink.copy(), (0, 0),
                        (plate.width, plate.height), {})
        hidden, record = apply_occlusion(c, plate.glyph_cells, np.random.default_rng(seed), 0.15, font=FONT)
        assert hidden == {x["position"] for x in record["characters"] if x["hidden"]}


def test_the_rule_is_deterministic() -> None:
    plate, canvas_a = _panel("type1a", 0.4, 5)
    _, canvas_b = _panel("type1a", 0.4, 5)
    hidden_a, record_a = apply_occlusion(canvas_a, plate.glyph_cells, np.random.default_rng(9), 0.18, font=FONT)
    hidden_b, record_b = apply_occlusion(canvas_b, plate.glyph_cells, np.random.default_rng(9), 0.18, font=FONT)
    assert (hidden_a, record_a) == (hidden_b, record_b)
    np.testing.assert_array_equal(canvas_a.rgb, canvas_b.rgb)
    config = small_config(seed=41, count=3, difficulty_weights={"hard": 1.0},
                          difficulties=force_effects("hard", {"occlusion": 1.0}))
    for plan in plan_dataset(config):
        a = generate_sample(config, plan, image_name="x.jpg")
        b = generate_sample(config, plan, image_name="x.jpg")
        assert a.image_bytes == b.image_bytes and a.record == b.record


# ------------------------------------------------------------------ what the fix does not change

#: 2.2.0 (commit 7af77a1) SHA-256 of these samples.  Occlusion is off, every other effect is default.
UNOCCLUDED_2_2_0 = (
    "73e87e63da0e4617eabdecdca31a67b354932da94f11ecfdfa34d55f6135e395",
    "0bcae5ccaccfbfc574c9c9e9db78e23433e73ad0afcd8989e6ddaecb17656f91",
    "fac12d67ee7439389013f18b68ef627f2b9b813e8235b65ae3fa59a194b26450",
    "48edd9847b9f26d3677595895f5a2d6f8e43f510ec415b858cf656d8ab0a528e",
    "e9cb4ffea48bae49d7d312c9dda76df085f979a0dfc885a7be54990da6ba02c6",
    "d7fff76f9a0e583d0a61709f52d6c4068ca8af72ef62682e8dbbf21c04b29650",
    "ae173773315d743379d18d5ff0b7e050b7b30578ae1140b44035cbc1683b93e7",
    "3e01cdaf1da1282bf5ce39cd6aacc20a8962f702a88b9ce1d8a7db782622f9ea",
    "89911c9a5379cae3336d1bc1b9f335417ec742bfdabeae74f9e79a4f70c9863e",
    "46d5a127a852022cb530cba5ba5752fa947ae3a768e8f646609e8d1db812c582",
    "e735dc3329f7c422426eb84bfc8c2e72979b328aff410d6d2e6ece08ef8448e0",
    "b612c4c2e406170764da665a263c9a3b5ab440f3c27fd81e73173d4be58ddd4f",
)


@PINNED
def test_samples_without_occlusion_are_byte_identical_to_2_2_0() -> None:
    off = {level: {"effect_probability": {"occlusion": 0.0}} for level in ("easy", "medium", "hard")}
    config = small_config(seed=2026091501, class_counts={"type1": 4, "type1a": 4, "type1b": 4}, difficulties=off)
    plans = plan_dataset(config)
    assert {p.plate_type for p in plans} == {"type1", "type1a", "type1b"}
    assert {p.difficulty for p in plans} == {"easy", "medium", "hard"}
    digests = tuple(hashlib.sha256(generate_sample(config, p, image_name="x.jpg").image_bytes).hexdigest() for p in plans)
    assert digests == UNOCCLUDED_2_2_0


def test_the_same_occluder_gives_the_same_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule changes labels, not drawing: with the same number of redraws, images are identical."""
    forced = {lvl: {"effect_probability": {"occlusion": 1.0}, "max_effects": 11} for lvl in ("easy", "medium", "hard")}
    config = small_config(seed=77, class_counts={"type1": 6, "type1a": 6, "type1b": 6}, difficulties=forced)
    same = 0
    for plan in plan_dataset(config):
        fixed = generate_sample(config, plan, image_name="x.jpg")
        monkeypatch.setattr(scene_module, "OcclusionJudge", LegacyJudge)
        legacy = generate_sample(config, plan, image_name="x.jpg")
        monkeypatch.undo()
        a, b = fixed.record["effects"].get("occlusion"), legacy.record["effects"].get("occlusion")
        if a is None and b is None and fixed.record["legibility"] == legacy.record["legibility"]:
            assert fixed.image_bytes == legacy.image_bytes  # never occluded: untouched
        elif a and b and a["attempts"] == b["attempts"] and fixed.record["legibility"] == legacy.record["legibility"]:
            assert fixed.image_bytes == legacy.image_bytes
            same += 1
    assert same >= 8


def test_occlusion_records_explain_every_hash() -> None:
    config = small_config(seed=123, class_counts={"type1": 5, "type1a": 5, "type1b": 5}, difficulty_weights={"hard": 1.0},
                          difficulties=force_effects("hard", {"occlusion": 1.0}))
    judged_total = 0
    for plan in plan_dataset(config):
        sample = generate_sample(config, plan, image_name="x.jpg")
        occlusion = sample.record["effects"].get("occlusion")
        if occlusion is None:  # a legibility retry may drop the occlusion
            assert "#" not in sample.annotation.plate_num
            continue
        hashes = [i for i, ch in enumerate(sample.annotation.plate_num) if ch == "#"]
        assert hashes == occlusion["hidden_positions"] == sample.record["hidden_positions"]
        for c in occlusion["characters"]:
            judged_total += 1
            assert plan.text.full[c["position"]] == c["char"]
            assert c["hidden"] is (c["position"] in hashes)
    assert judged_total >= 10
