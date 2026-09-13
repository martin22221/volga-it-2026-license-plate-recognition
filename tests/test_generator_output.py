"""End-to-end: generated batches, their annotations and their metadata."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dataset.generator import OUTPUT_LICENSE, SOURCE_ID
from dataset.generator.annotations import CONDITION_TAGS, META_COLUMNS
from dataset.generator.generate import generate_dataset
from dataset.generator.plate_text import is_plate_for_type
from dataset.generator.sample import generate_sample, plan_dataset
from src.dataset_meta import (
    ALLOWED_CONDITIONS,
    REDISTRIBUTABLE_LICENSES,
    REJECTED_SOURCES,
    REQUIRED_COLUMNS,
    normalize_license,
    polygon_signed_area,
    validate_meta,
)
from src.validator import is_valid_plate
from tests.generator_helpers import force_effects, small_config

GENERATOR_DIR = Path(__file__).resolve().parents[1] / "dataset" / "generator"
REGISTRY = Path(__file__).resolve().parents[1] / "docs" / "data_sources.md"
EXTERNAL_SOURCE_IDS = ("autoria_numberplate_options", "roboflow_two_line_russian_license_plates")


def _rows(root: Path) -> list[dict[str, str]]:
    with (root / "meta.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def _records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "generation.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("batch")
    config = small_config(seed=123, class_counts={"type1": 4, "type1a": 4, "type1b": 4})
    generate_dataset(config, root)
    return root


# ------------------------------------------------------------ schema drift


def test_meta_columns_match_the_repository_schema() -> None:
    assert META_COLUMNS == REQUIRED_COLUMNS


def test_condition_vocabulary_matches_the_repository() -> None:
    assert CONDITION_TAGS == ALLOWED_CONDITIONS


# ------------------------------------------------------------ batch layout


def test_batch_layout(batch: Path) -> None:
    images = sorted((batch / "images" / "synthetic").glob("*.jpg"))
    assert len(images) == 12
    for name in ("meta.csv", "generation.jsonl", "manifest.json"):
        assert (batch / name).is_file()


def test_batch_passes_the_repository_validator_cleanly(batch: Path) -> None:
    report = validate_meta(batch / "meta.csv", batch)
    assert report.errors == []
    assert report.warnings == []
    assert report.stats.synthetic_images == 12
    assert report.stats.real_images == 0
    assert report.stats.by_plate_type == {"type1": 4, "type1a": 4, "type1b": 4}


def test_meta_rows_are_synthetic_with_our_own_provenance(batch: Path) -> None:
    for row in _rows(batch):
        assert row["image"].startswith("images/synthetic/")
        assert row["is_synthetic"] == "true"
        assert row["is_vehicle"] == "true"
        assert row["source"] == SOURCE_ID
        assert row["license"] == OUTPUT_LICENSE
        assert normalize_license(row["license"]) in REDISTRIBUTABLE_LICENSES
        assert row["source"] not in REJECTED_SOURCES
        assert row["source"] not in EXTERNAL_SOURCE_IDS


def test_generator_source_is_registered_as_own_work() -> None:
    registry = REGISTRY.read_text(encoding="utf-8")
    line = next(l for l in registry.splitlines() if l.startswith(f"| `{SOURCE_ID}`"))
    assert "own work" in line.lower()


def test_generator_ships_no_external_assets() -> None:
    forbidden = {".ttf", ".otf", ".woff", ".woff2", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg"}
    shipped = [p for p in GENERATOR_DIR.rglob("*") if p.suffix.lower() in forbidden]
    assert shipped == []


def test_manifest_records_provenance_and_file_hashes(batch: Path) -> None:
    manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["source"] == SOURCE_ID
    assert manifest["license"] == OUTPUT_LICENSE
    assert manifest["is_synthetic"] is True
    assert manifest["external_assets"] == []
    assert manifest["seed"] == 123
    assert manifest["counts"]["plate_type"] == {"type1": 4, "type1a": 4, "type1b": 4}
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((batch / relative).read_bytes()).hexdigest() == digest


def test_records_carry_seed_and_match_meta(batch: Path) -> None:
    rows = _rows(batch)
    records = _records(batch)
    assert [r["index"] for r in records] == list(range(12))
    assert len({r["sample_seed"] for r in records}) == 12
    for row, record in zip(rows, records):
        assert row["image"] == record["image"]
        assert row["plate_num"] == record["plate_num"]
        assert row["plate_type"] == record["plate_type"]
        assert record["difficulty"] in ("easy", "medium", "hard")
        assert row["conditions"].split("|") == record["conditions"]


# ------------------------------------------------------------ annotations


def test_plate_numbers_are_valid_or_honestly_masked(batch: Path) -> None:
    for row, record in zip(_rows(batch), _records(batch)):
        full = record["plate_num_full"]
        assert is_plate_for_type(full, row["plate_type"]) and is_valid_plate(full, row["plate_type"])
        label = row["plate_num"]
        assert len(label) == len(full)
        for index, (shown, actual) in enumerate(zip(label, full)):
            assert shown == ("#" if index in record["hidden_positions"] else actual)


def test_bbox_and_quad_are_consistent_and_inside_the_image(batch: Path) -> None:
    for row, record in zip(_rows(batch), _records(batch)):
        width, height = record["image_size"]
        x, y, w, h = (float(row[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        quad = [(float(row[f"quad_x{i}"]), float(row[f"quad_y{i}"])) for i in range(1, 5)]
        assert w > 0 and h > 0
        assert 0 <= x and x + w <= width and 0 <= y and y + h <= height
        xs, ys = [q[0] for q in quad], [q[1] for q in quad]
        assert min(xs) == pytest.approx(x, abs=0.011) and max(xs) == pytest.approx(x + w, abs=0.011)
        assert min(ys) == pytest.approx(y, abs=0.011) and max(ys) == pytest.approx(y + h, abs=0.011)
        assert polygon_signed_area(quad) > 0  # clockwise from the plate's top-left
        # Corner 1 is the plate's top-left: left of corner 2, above corner 4.
        assert quad[0][0] < quad[1][0] and quad[0][1] < quad[3][1]


@pytest.mark.parametrize("plate_type", ["type1", "type1a", "type1b"])
@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_transformed_quad_matches_the_rendered_plate_pixels(plate_type: str, level: str) -> None:
    """The quad comes from the transform; check it against the warped pixels."""
    no_occlusion = force_effects(level, {"occlusion": 0.0}, max_effects=3)
    config = small_config(seed=7, class_counts={plate_type: 3}, difficulty_weights={level: 1.0}, difficulties=no_occlusion)
    for plan in plan_dataset(config):
        sample = generate_sample(config, plan, image_name="x.jpg")
        mask = sample.plate_mask > 0.5
        ys, xs = np.nonzero(mask)
        x, y, w, h = sample.annotation.bbox
        # Rounded plate corners sit slightly inside the corner points.
        assert abs(xs.min() - x) <= 2.0 and abs(xs.max() + 1 - (x + w)) <= 2.0
        assert abs(ys.min() - y) <= 2.0 and abs(ys.max() + 1 - (y + h)) <= 2.0
        quad = sample.annotation.quad
        area = abs(polygon_signed_area(quad))
        assert abs(float(sample.plate_mask.sum()) - area) / area < 0.06


def test_occluded_characters_are_labelled_unreadable() -> None:
    config = small_config(
        seed=99, class_counts={"type1": 4, "type1a": 4}, difficulty_weights={"hard": 1.0},
        difficulties=force_effects("hard", {"occlusion": 1.0}),
    )
    hidden_total = 0
    for plan in plan_dataset(config):
        sample = generate_sample(config, plan, image_name="x.jpg")
        # A legibility retry may redraw the effects without occlusion.
        hidden = sample.record["effects"].get("occlusion", {}).get("hidden_positions", [])
        hidden_total += len(hidden)
        assert [i for i, ch in enumerate(sample.annotation.plate_num) if ch == "#"] == hidden
        assert len(hidden) < len(plan.text.full)  # never every character
    assert hidden_total > 0


def test_type1b_images_contain_a_yellow_plate() -> None:
    config = small_config(seed=3, class_counts={"type1b": 3, "type1": 3}, difficulty_weights={"easy": 1.0},
                          difficulties=force_effects("easy", {e: 0.0 for e in ("night", "low_light", "glare", "dirt", "shadow")}))
    from PIL import Image
    import io

    for plan in plan_dataset(config):
        sample = generate_sample(config, plan, image_name="x.jpg")
        image = np.asarray(Image.open(io.BytesIO(sample.image_bytes)).convert("RGB"), dtype=np.float32) / 255
        mask = sample.plate_mask > 0.95
        r, g, b = np.median(image[mask], axis=0)
        if plan.plate_type == "type1b":
            assert r - b > 0.35, (r, g, b)
        else:
            assert abs(r - b) < 0.2, (r, g, b)


# ------------------------------------------------------------ determinism


def test_same_seed_gives_byte_identical_batches(tmp_path: Path) -> None:
    config = small_config(seed=2024, class_counts={"type1": 2, "type1a": 2, "type1b": 2})
    first = generate_dataset(config, tmp_path / "a")
    second = generate_dataset(config, tmp_path / "b")
    assert first["files"] == second["files"]
    for name in ("meta.csv", "generation.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_different_seed_gives_a_different_batch(tmp_path: Path) -> None:
    a = generate_dataset(small_config(seed=1, class_counts={"type1a": 3}), tmp_path / "a")
    b = generate_dataset(small_config(seed=2, class_counts={"type1a": 3}), tmp_path / "b")
    assert (tmp_path / "a" / "meta.csv").read_bytes() != (tmp_path / "b" / "meta.csv").read_bytes()
    assert set(a["files"].values()).isdisjoint(set(b["files"].values()))


def test_parallel_workers_produce_identical_output(tmp_path: Path) -> None:
    config = small_config(seed=77, class_counts={"type1": 2, "type1a": 2, "type1b": 2})
    serial = generate_dataset(config, tmp_path / "serial")
    parallel = generate_dataset(config, tmp_path / "parallel", workers=2)
    assert serial["files"] == parallel["files"]
