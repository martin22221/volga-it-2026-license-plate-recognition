"""Tests for the deterministic, group-aware real-image splits."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.real_splits import (
    DEFAULT_WEIGHTS,
    HOLDOUT,
    SPLITS,
    SplitError,
    SplitRow,
    assign_groups,
    assign_rows,
    group_score,
    holdout_images,
    leakage_report,
    leakage_text,
    read_manifest,
    write_manifest,
)


def _rows(*triples: tuple[str, str, str]) -> list[SplitRow]:
    return [SplitRow(image=i, group=g, split=s) for i, g, s in triples]


# ------------------------------------------------------------------ the hash


def test_group_score_is_stable_across_runs_and_machines() -> None:
    """The split must not depend on Python's hash randomisation or on the platform."""
    assert group_score("session-01", 7) == pytest.approx(
        group_score("session-01", 7)
    )
    # Pinned: SHA-256 of "7:session-01" -- a change here silently re-splits the dataset.
    assert group_score("session-01", 7) == pytest.approx(0.3769431752570218, abs=1e-12)
    assert group_score("session-01", 8) != group_score("session-01", 7)
    assert 0.0 <= group_score("x", 1) < 1.0


# ------------------------------------------------------------------ assignment


def test_assignment_is_deterministic_and_order_independent() -> None:
    groups = [f"g{i}" for i in range(50)]
    first = assign_groups(groups, seed=5)
    second = assign_groups(reversed(groups), seed=5)
    assert first == second
    assert set(first.values()) <= set(SPLITS)


def test_every_image_of_a_group_lands_in_one_split() -> None:
    items = [(f"images/real/s/{g}_{n}.jpg", g, "src") for g in (f"g{i}" for i in range(30)) for n in range(4)]
    rows = assign_rows(items, seed=11)
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(row.group, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert leakage_report(rows)["clean"]


def test_weights_are_respected_approximately() -> None:
    assignment = assign_groups([f"g{i}" for i in range(4000)], seed=3)
    shares = Counter(assignment.values())
    for split, weight in DEFAULT_WEIGHTS.items():
        assert shares[split] / 4000 == pytest.approx(weight, abs=0.03)


def test_zero_weight_split_gets_nothing() -> None:
    assignment = assign_groups([f"g{i}" for i in range(200)], seed=3,
                               weights={"train": 0.5, "val": 0.5, "holdout": 0.0})
    assert HOLDOUT not in set(assignment.values())


@pytest.mark.parametrize("weights", [
    {"train": 1.0, "test": 1.0},          # unknown split
    {"train": 0.0, "val": 0.0, "holdout": 0.0},  # nothing to assign to
    {"train": -1.0, "val": 1.0},          # negative
])
def test_bad_weights_are_refused(weights: dict[str, float]) -> None:
    with pytest.raises(SplitError):
        assign_groups(["a", "b"], seed=1, weights=weights)


# ------------------------------------------------------------------ freezing


def test_frozen_groups_keep_their_split_when_the_dataset_grows() -> None:
    frozen = _rows(("images/real/s/a.jpg", "g1", "holdout"), ("images/real/s/b.jpg", "g2", "train"))
    items = [
        ("images/real/s/a.jpg", "g1", "src"),
        ("images/real/s/b.jpg", "g2", "src"),
        ("images/real/s/c.jpg", "g1", "src"),  # new image of a frozen group
        ("images/real/s/d.jpg", "g9", "src"),  # new group
    ]
    rows = {row.image: row for row in assign_rows(items, seed=2, frozen_rows=frozen)}
    assert rows["images/real/s/a.jpg"].split == "holdout"
    assert rows["images/real/s/c.jpg"].split == "holdout"  # follows its group
    assert rows["images/real/s/b.jpg"].split == "train"
    assert rows["images/real/s/d.jpg"].split in SPLITS


def test_changing_the_seed_never_moves_a_frozen_group() -> None:
    frozen = _rows(("images/real/s/a.jpg", "g1", "holdout"))
    items = [("images/real/s/a.jpg", "g1", "src"), ("images/real/s/z.jpg", "g2", "src")]
    for seed in (1, 2, 3, 99):
        rows = {row.image: row.split for row in assign_rows(items, seed=seed, frozen_rows=frozen)}
        assert rows["images/real/s/a.jpg"] == "holdout"


def test_regrouping_an_assigned_image_is_refused() -> None:
    frozen = _rows(("images/real/s/a.jpg", "g1", "holdout"))
    with pytest.raises(SplitError, match="already assigned"):
        assign_rows([("images/real/s/a.jpg", "g2", "src")], seed=1, frozen_rows=frozen)


def test_frozen_rows_survive_when_their_image_is_not_restaged() -> None:
    frozen = _rows(("images/real/s/gone.jpg", "g1", "val"))
    rows = assign_rows([("images/real/s/new.jpg", "g2", "src")], seed=1, frozen_rows=frozen)
    assert {row.image for row in rows} == {"images/real/s/gone.jpg", "images/real/s/new.jpg"}


# ------------------------------------------------------------------ manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "real_splits.csv"
    rows = [SplitRow("images/real/s/a.jpg", "g1", "train", "src", "2026-09-16", "note")]
    write_manifest(path, rows)
    assert read_manifest(path) == rows
    assert read_manifest(tmp_path / "absent.csv") == []


@pytest.mark.parametrize("line, message", [
    ("images/real/s/a.jpg;g1;testing;src;2026-09-16;", "not one of"),
    ("images/real/s/a.jpg;;train;src;2026-09-16;", "empty group"),
])
def test_broken_manifest_rows_are_refused(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "m.csv"
    path.write_text("image;group;split;source;assigned_on;note\n" + line + "\n", encoding="utf-8")
    with pytest.raises(SplitError, match=message):
        read_manifest(path)


def test_manifest_missing_a_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text("image;split\nimages/real/s/a.jpg;train\n", encoding="utf-8")
    with pytest.raises(SplitError, match="missing column"):
        read_manifest(path)


# ------------------------------------------------------------------ leakage


def test_leakage_report_is_clean_for_a_sound_manifest() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/b.jpg", "g2", "holdout"))
    findings = leakage_report(rows)
    assert findings["clean"]
    assert findings["by_split"] == {"train": 1, "val": 0, "holdout": 1}
    assert findings["groups"] == 2
    assert "CLEAN" in leakage_text(findings)


def test_a_group_spanning_two_splits_is_leakage() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/b.jpg", "g1", "holdout"))
    findings = leakage_report(rows)
    assert findings["groups_spanning_splits"] == ["g1"]
    assert not findings["clean"]
    assert "LEAKAGE" in leakage_text(findings)


def test_the_same_picture_under_two_names_is_leakage() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/copy.jpg", "g2", "holdout"))
    findings = leakage_report(rows, hashes={"images/real/s/a.jpg": "d", "images/real/s/copy.jpg": "d"})
    assert findings["identical_images_across_splits"] == [
        ("images/real/s/a.jpg", "train", "images/real/s/copy.jpg", "holdout")
    ]
    assert not findings["clean"]


def test_near_duplicates_across_splits_are_leakage() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/b.jpg", "g2", "holdout"))
    pairs = [("images/real/s/a.jpg", "images/real/s/b.jpg", 0.99)]
    assert not leakage_report(rows, near_duplicate_pairs=pairs)["clean"]
    # the same pair inside one split is not leakage
    same = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/b.jpg", "g2", "train"))
    assert leakage_report(same, near_duplicate_pairs=pairs)["clean"]


def test_duplicate_manifest_rows_are_reported() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/a.jpg", "g1", "train"))
    findings = leakage_report(rows)
    assert findings["duplicate_manifest_rows"] == ["images/real/s/a.jpg"]
    assert not findings["clean"]


def test_holdout_images_lists_only_the_holdout() -> None:
    rows = _rows(("images/real/s/a.jpg", "g1", "train"), ("images/real/s/b.jpg", "g2", "holdout"))
    assert holdout_images(rows) == {"images/real/s/b.jpg"}


# ------------------------------------------------------------------ the CLI


def _accepted_folder(tmp_path: Path) -> Path:
    """A folder of accepted images: meta.csv plus groups.csv, as intake leaves it."""
    from tests.test_real_intake import PREFIX, _row, _write_csv
    from src.dataset_meta import REQUIRED_COLUMNS

    folder = tmp_path / "accepted" / "src"
    names = [f"{n}.jpg" for n in range(6)]
    _write_csv(folder / "meta.csv", list(REQUIRED_COLUMNS), [_row(name) for name in names])
    _write_csv(folder / "groups.csv", ["image", "group"],
               [{"image": f"{PREFIX}/{name}", "group": f"session-{i // 3}"} for i, name in enumerate(names)])
    return folder


def test_cli_checks_the_committed_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import plan_real_splits as cli

    assert cli.main(["--check"]) == cli.EXIT_OK
    assert "CLEAN" in capsys.readouterr().out


def test_cli_dry_run_does_not_write(tmp_path: Path) -> None:
    from scripts import plan_real_splits as cli

    manifest = tmp_path / "real_splits.csv"
    assert cli.main(["--manifest", str(manifest), "--add", str(_accepted_folder(tmp_path))]) == cli.EXIT_OK
    assert not manifest.exists()


def test_cli_writes_and_then_keeps_the_frozen_split(tmp_path: Path) -> None:
    from scripts import plan_real_splits as cli

    manifest = tmp_path / "real_splits.csv"
    folder = _accepted_folder(tmp_path)
    assert cli.main(["--manifest", str(manifest), "--add", str(folder), "--write"]) == cli.EXIT_OK
    first = {row.image: row.split for row in read_manifest(manifest)}
    assert len(first) == 6
    # a different seed must not move anything that is already frozen
    assert cli.main(["--manifest", str(manifest), "--add", str(folder), "--seed", "999", "--write"]) == cli.EXIT_OK
    assert {row.image: row.split for row in read_manifest(manifest)} == first
    assert leakage_report(read_manifest(manifest))["clean"]


def test_cli_refuses_images_without_a_group(tmp_path: Path) -> None:
    from scripts import plan_real_splits as cli
    from tests.test_real_intake import _write_csv

    folder = _accepted_folder(tmp_path)
    _write_csv(folder / "groups.csv", ["image", "group"], [])
    assert cli.main(["--manifest", str(tmp_path / "m.csv"), "--add", str(folder)]) == cli.EXIT_UNUSABLE


def test_the_committed_manifest_is_empty_and_clean() -> None:
    """No real image has been acquired yet; the manifest must say exactly that."""
    manifest = Path(__file__).resolve().parents[1] / "dataset" / "splits" / "real_splits.csv"
    rows = read_manifest(manifest)
    assert rows == []
    assert leakage_report(rows)["clean"]
