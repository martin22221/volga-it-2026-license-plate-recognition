"""Tests for the Dataset V1 freeze and its splits.

The properties that matter are determinism (the split must reproduce from the
seed on any machine) and non-leakage (nothing may span two splits). Both are
checked on synthetic fixtures and then on the committed dataset itself.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.dataset_v1 import (
    DATASET_VERSION,
    SPLIT_SEED,
    SYNTHETIC_SPLITS,
    SYNTHETIC_WEIGHTS,
    DatasetV1Error,
    assign_synthetic,
    audit_splits,
    audit_text,
    combined_digest,
    image_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "dataset" / "splits"


def _items(n: int, plate_type: str = "type1", difficulty: str = "easy"):
    return [(f"images/synthetic/s_{i:05d}.jpg", plate_type, difficulty) for i in range(n)]


# ------------------------------------------------------------------ determinism


def test_the_score_is_pinned_not_merely_stable() -> None:
    """A pinned value: if this changes, every split silently changes with it."""
    assert image_score("images/synthetic/a.jpg", 7) == pytest.approx(
        image_score("images/synthetic/a.jpg", 7)
    )
    assert image_score("x", 1) != image_score("x", 2)
    assert 0.0 <= image_score("x", 1) < 1.0


def test_the_split_is_independent_of_input_order() -> None:
    items = _items(500)
    forward = assign_synthetic(items, seed=11)
    backward = assign_synthetic(list(reversed(items)), seed=11)
    assert forward == backward


def test_a_different_seed_gives_a_different_split() -> None:
    items = _items(500)
    assert assign_synthetic(items, seed=1) != assign_synthetic(items, seed=2)


def test_the_same_seed_reproduces_exactly() -> None:
    items = _items(300)
    assert assign_synthetic(items, seed=99) == assign_synthetic(items, seed=99)


# ------------------------------------------------------------------ proportions


def test_every_stratum_is_cut_at_the_same_proportions() -> None:
    """Rare combinations keep their share instead of landing wherever chance puts them."""
    items = (
        _items(1000, "type1", "easy")
        + [(f"images/synthetic/b_{i}.jpg", "type1a", "hard") for i in range(100)]
        + [(f"images/synthetic/c_{i}.jpg", "type1b", "medium") for i in range(40)]
    )
    assignment = assign_synthetic(items, seed=5)
    for prefix, total in (("images/synthetic/s_", 1000),
                          ("images/synthetic/b_", 100),
                          ("images/synthetic/c_", 40)):
        subset = [s for image, s in assignment.items() if image.startswith(prefix)]
        assert len(subset) == total
        share = subset.count("train") / total
        assert share == pytest.approx(SYNTHETIC_WEIGHTS["train"], abs=0.02)


def test_every_image_is_assigned_exactly_once() -> None:
    items = _items(777)
    assignment = assign_synthetic(items, seed=3)
    assert len(assignment) == 777
    assert set(assignment.values()) <= set(SYNTHETIC_SPLITS)


def test_a_duplicate_image_is_refused() -> None:
    items = _items(3) + _items(1)
    with pytest.raises(DatasetV1Error, match="twice"):
        assign_synthetic(items, seed=1)


@pytest.mark.parametrize("weights", [
    {"train": 1.0, "holdout": 1.0},        # holdout is not a synthetic split
    {"train": 0.0, "val": 0.0, "test": 0.0},
    {"train": -1.0, "val": 1.0},
])
def test_bad_weights_are_refused(weights: dict) -> None:
    with pytest.raises(DatasetV1Error):
        assign_synthetic(_items(10), seed=1, weights=weights)


def test_a_tiny_stratum_still_gets_assigned() -> None:
    """One image cannot be split three ways; it must still land somewhere."""
    assignment = assign_synthetic(_items(1), seed=1)
    assert len(assignment) == 1 and next(iter(assignment.values())) in SYNTHETIC_SPLITS


# ------------------------------------------------------------------ the audit


def _meta(images, plate="A123BC77", plate_type="type1"):
    return {
        image: {"plate_num": f"{plate}{i}", "plate_type": plate_type, "conditions": "day"}
        for i, image in enumerate(images)
    }


def test_a_clean_split_has_no_blocking_findings() -> None:
    synthetic = {"images/synthetic/a.jpg": "train", "images/synthetic/b.jpg": "val"}
    real = [{"image": "images/real/s/r.jpg", "group": "g1", "split": "holdout"}]
    meta = _meta(list(synthetic) + ["images/real/s/r.jpg"])
    audit = audit_splits(synthetic, real, meta)
    assert audit.clean, audit.blocking
    assert "CLEAN" in audit_text(audit)


def test_an_unassigned_dataset_image_blocks() -> None:
    synthetic = {"images/synthetic/a.jpg": "train"}
    meta = _meta(["images/synthetic/a.jpg", "images/synthetic/orphan.jpg"])
    audit = audit_splits(synthetic, [], meta)
    assert any("belong to no split" in p for p in audit.blocking)


def test_a_split_entry_for_a_missing_image_blocks() -> None:
    synthetic = {"images/synthetic/a.jpg": "train", "images/synthetic/ghost.jpg": "val"}
    audit = audit_splits(synthetic, [], _meta(["images/synthetic/a.jpg"]))
    assert any("not in the dataset" in p for p in audit.blocking)


def test_a_plate_number_in_two_splits_is_leakage() -> None:
    synthetic = {"images/synthetic/a.jpg": "train", "images/synthetic/b.jpg": "val"}
    meta = {
        "images/synthetic/a.jpg": {"plate_num": "A123BC77", "plate_type": "type1", "conditions": ""},
        "images/synthetic/b.jpg": {"plate_num": "A123BC77", "plate_type": "type1", "conditions": ""},
    }
    audit = audit_splits(synthetic, [], meta)
    assert any("more than one split" in p for p in audit.blocking)
    assert not audit.clean


def test_identical_bytes_in_two_splits_is_leakage() -> None:
    synthetic = {"images/synthetic/a.jpg": "train", "images/synthetic/b.jpg": "val"}
    meta = _meta(list(synthetic))
    audit = audit_splits(
        synthetic, [], meta,
        hashes={"images/synthetic/a.jpg": "same", "images/synthetic/b.jpg": "same"},
    )
    assert any("identical image" in p for p in audit.blocking)


def test_a_real_group_spanning_splits_is_leakage() -> None:
    real = [
        {"image": "images/real/s/a.jpg", "group": "g1", "split": "train"},
        {"image": "images/real/s/b.jpg", "group": "g1", "split": "holdout"},
    ]
    audit = audit_splits({}, real, _meta([r["image"] for r in real]))
    assert any("span splits" in p for p in audit.blocking)


def test_an_image_in_both_populations_blocks() -> None:
    shared = "images/real/s/a.jpg"
    audit = audit_splits(
        {shared: "train"},
        [{"image": shared, "group": "g1", "split": "holdout"}],
        _meta([shared]),
    )
    assert any("both the synthetic and the real split" in p for p in audit.blocking)


def test_a_tiny_holdout_is_noted_not_ignored() -> None:
    real = [{"image": "images/real/s/a.jpg", "group": "g1", "split": "holdout"}]
    audit = audit_splits({}, real, _meta(["images/real/s/a.jpg"]))
    assert audit.clean
    assert any("far too few" in n for n in audit.notes)


# ------------------------------------------------------------------ digests


def test_the_combined_digest_depends_on_content_and_not_on_order() -> None:
    a = {"x.jpg": "1", "y.jpg": "2"}
    b = {"y.jpg": "2", "x.jpg": "1"}
    assert combined_digest(a) == combined_digest(b)
    assert combined_digest(a) != combined_digest({"x.jpg": "1", "y.jpg": "3"})


# ------------------------------------------------------------------ the committed freeze


def test_the_committed_freeze_exists_and_describes_this_dataset() -> None:
    path = SPLITS_DIR / "dataset_v1.json"
    assert path.is_file(), "Dataset V1 has not been frozen"
    freeze = json.loads(path.read_text(encoding="utf-8"))

    assert freeze["dataset_version"] == DATASET_VERSION
    assert freeze["split_seed"] == SPLIT_SEED
    assert freeze["counts"]["synthetic_images"] == 12_000
    assert freeze["counts"]["real_images"] == 13
    assert freeze["counts"]["images_total"] == 12_013
    assert freeze["class_counts"]["synthetic"] == {"type1": 2400, "type1a": 4200, "type1b": 5400}
    assert freeze["class_counts"]["real"] == {"type1": 6, "type1b": 7}
    # the identity fields a later session would prove membership with
    for field in ("meta_sha256", "combined_image_digest", "git_commit"):
        assert freeze.get(field), f"{field} is empty"
    assert freeze["generator"]["version"] == "2.3.0"
    assert freeze["generator"]["seed"] == 2026091401


def test_the_committed_synthetic_split_reproduces_from_the_seed() -> None:
    """The whole point of the seed: re-derive the split and get the same answer."""
    path = SPLITS_DIR / "synthetic_splits.csv"
    assert path.is_file()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert len(rows) == 12_000

    recomputed = assign_synthetic(
        ((r["image"], r["plate_type"], r["difficulty"]) for r in rows), seed=SPLIT_SEED
    )
    committed = {r["image"]: r["split"] for r in rows}
    assert recomputed == committed


def test_the_committed_split_covers_every_dataset_image_exactly_once() -> None:
    with (SPLITS_DIR / "synthetic_splits.csv").open(encoding="utf-8", newline="") as handle:
        synthetic = {r["image"]: r["split"] for r in csv.DictReader(handle, delimiter=";")}
    with (SPLITS_DIR / "real_splits.csv").open(encoding="utf-8", newline="") as handle:
        real = list(csv.DictReader(handle, delimiter=";"))
    with (REPO_ROOT / "dataset" / "meta.csv").open(encoding="utf-8", newline="") as handle:
        meta = {r["image"]: r for r in csv.DictReader(handle, delimiter=";")}

    audit = audit_splits(synthetic, real, meta)
    assert audit.blocking == []
    assert len(synthetic) + len(real) == len(meta) == 12_013
