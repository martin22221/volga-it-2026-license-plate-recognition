"""The generator command line and the contact sheet."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dataset.generator.contact_sheet import build_contact_sheet
from dataset.generator.contact_sheet import main as contact_sheet_main
from dataset.generator.generate import DATASET_DIR, main

REPO = Path(__file__).resolve().parents[1]
SMALL = ["--image-size", "320x240"]


def run(*args: str) -> int:
    return main([*args, *SMALL, "--quiet"])


def test_cli_generates_a_valid_batch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    assert run("--output", str(out), "--per-class", "type1=1,type1a=2,type1b=1", "--seed", "5") == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["plate_type"] == {"type1": 1, "type1a": 2, "type1b": 1}
    assert manifest["config"]["image_sizes"] == [[320, 240]]
    assert "0 error(s), 0 warning(s)" in capsys.readouterr().out


def test_cli_count_and_difficulty(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert run("--output", str(out), "--count", "3", "--difficulty", "hard") == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_count"] == 3
    assert manifest["counts"]["difficulty"] == {"hard": 3}


def test_cli_with_config_file(tmp_path: Path) -> None:
    config = tmp_path / "c.json"
    config.write_text(json.dumps({"count": 2, "seed": 9, "class_weights": {"type1a": 1.0}}), encoding="utf-8")
    out = tmp_path / "run"
    assert run("--output", str(out), "--config", str(config)) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 9 and manifest["counts"]["plate_type"] == {"type1a": 2}


@pytest.mark.parametrize(
    "args",
    [
        ["--count", "0"],
        ["--count", "-3"],
        ["--per-class", "type7=3"],
        ["--per-class", "type1=x"],
        ["--per-class", "type1=2", "--count", "4"],
        ["--difficulty", "hard", "--difficulty-weights", "easy=1"],
        ["--difficulty-weights", "brutal=1"],
        ["--image-size", "big"],
        ["--difficulty", "extreme"],
        ["--type1b-format", "competition"],  # removed in V2: type1b structure is fixed
    ],
)
def test_cli_rejects_invalid_arguments(tmp_path: Path, args: list[str]) -> None:
    assert main(["--output", str(tmp_path / "x"), *args, "--quiet"]) == 2
    assert not (tmp_path / "x" / "meta.csv").exists()


def test_cli_rejects_a_missing_config_file(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path / "x"), "--config", str(tmp_path / "nope.json")]) == 2


def test_cli_refuses_the_competition_dataset_directory(tmp_path: Path) -> None:
    target = DATASET_DIR / "images" / "synthetic" / "should_not_exist"
    assert run("--output", str(target), "--count", "1") == 3
    assert not target.exists()


def test_cli_refuses_a_foreign_non_empty_directory(tmp_path: Path) -> None:
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "notes.txt").write_text("keep me", encoding="utf-8")
    assert run("--output", str(out), "--count", "1") == 3
    assert run("--output", str(out), "--count", "1", "--overwrite") == 3
    assert (out / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_cli_overwrites_only_its_own_batch(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert run("--output", str(out), "--count", "3", "--seed", "1") == 0
    assert run("--output", str(out), "--count", "2", "--seed", "2") == 3  # needs --overwrite
    assert run("--output", str(out), "--count", "2", "--seed", "2", "--overwrite") == 0
    assert len(list((out / "images" / "synthetic").glob("*.jpg"))) == 2
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2


def test_cli_can_overwrite_an_interrupted_batch(tmp_path: Path) -> None:
    out = tmp_path / "run"
    (out / "images" / "synthetic").mkdir(parents=True)
    (out / "images" / "synthetic" / "syn_1_00000.jpg").write_bytes(b"partial")
    (out / "manifest.json").write_text(json.dumps({"generator": "volga-synthetic-plate-generator", "complete": False}))
    assert run("--output", str(out), "--count", "1", "--seed", "4", "--overwrite") == 0
    assert not (out / "images" / "synthetic" / "syn_1_00000.jpg").exists()


def test_contact_sheet_shows_every_image(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert run("--output", str(out), "--per-class", "type1=1,type1a=1,type1b=1", "--contact-sheet") == 0
    sheet = out / "review" / "contact_sheet.html"
    html = sheet.read_text(encoding="utf-8")
    assert html.count('class="card"') == 3
    for plate_type in ("type1", "type1a", "type1b"):
        assert f'data-type="{plate_type}"' in html
    assert "data:image/jpeg;base64," in html
    assert build_contact_sheet(out, tmp_path / "again.html").is_file()
    assert contact_sheet_main([str(tmp_path / "nothing")]) == 2


def test_module_entry_point(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dataset.generator", "--output", str(tmp_path / "m"), "--count", "1", *SMALL, "--quiet"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "m" / "meta.csv").is_file()
