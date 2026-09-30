"""scripts/kaggle_launch.py: the decisions and the plumbing that must never hang (no GPU, no Kaggle)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from scripts import kaggle_launch as kl

COMMIT = "a" * 40


def test_choose_start() -> None:
    ok = {"resumable": True, "kind": "baseline", "git_commit": COMMIT, "finished": False}
    assert kl.choose_start(None, COMMIT) == "fresh"
    assert kl.choose_start(ok, COMMIT) == "resume"
    assert kl.choose_start({**ok, "finished": True}, COMMIT) == "finished"
    assert kl.choose_start({**ok, "git_commit": "b" * 40}, COMMIT) == "incompatible"
    assert kl.choose_start({**ok, "kind": "smoke"}, COMMIT) == "incompatible"
    assert kl.choose_start({"resumable": False}, COMMIT) == "incompatible"  # weights-only, pre-resume last.pt
    assert kl.choose_start({"error": "corrupt"}, COMMIT) == "incompatible"


def test_stream_echoes_output_and_returns_the_exit_code(capsys) -> None:
    code = kl.stream([sys.executable, "-c", "print('one'); print('two'); raise SystemExit(4)"],
                     env=None, cwd=Path.cwd(), stall_seconds=60)
    assert code == 4
    out = capsys.readouterr().out
    assert "one" in out and "two" in out


def test_stream_kills_a_silent_process() -> None:
    started = time.monotonic()
    code = kl.stream([sys.executable, "-c", "print('started', flush=True); import time; time.sleep(60)"],
                     env=None, cwd=Path.cwd(), stall_seconds=1)
    assert code == -kl.EXIT_STALLED
    assert time.monotonic() - started < 30


def _bundle(root: Path) -> Path:
    for name in ("scripts/train_baseline.py", "dataset/meta.csv", "dataset/images/synthetic/a.jpg"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name, encoding="utf-8")
    (root / "scripts" / "__pycache__").mkdir()
    (root / "scripts" / "__pycache__" / "x.pyc").write_bytes(b"x")
    return root


def test_copy_verifies_reuses_and_never_deletes(tmp_path) -> None:
    bundle = _bundle(tmp_path / "input")
    work = tmp_path / "working" / "baseline_v1"
    work.parent.mkdir()
    kl.stage_copy(bundle, work, COMMIT)
    assert (work / "dataset" / "images" / "synthetic" / "a.jpg").read_text() == "dataset/images/synthetic/a.jpg"
    assert not (work / "scripts" / "__pycache__").exists()
    assert (work / kl.COPY_MARKER).read_text().strip() == COMMIT

    (work / "runs").mkdir()
    (work / "runs" / "keep.txt").write_text("run")
    kl.stage_copy(bundle, work, COMMIT)  # same commit: reused, runs kept
    assert (work / "runs" / "keep.txt").exists()

    kl.stage_copy(bundle, work, "b" * 40)  # another commit: old copy moved aside, not deleted
    stale = [p for p in work.parent.iterdir() if p.name.startswith("baseline_v1.stale-")]
    assert len(stale) == 1 and (stale[0] / "runs" / "keep.txt").exists()
    assert (work / kl.COPY_MARKER).read_text().strip() == "b" * 40


def test_bundle_stage_refuses_an_incomplete_bundle(tmp_path) -> None:
    with pytest.raises(kl.StageFailed, match="lacks"):
        kl.stage_bundle(_bundle(tmp_path))


def test_local_runs_ignore_smoke_snapshots_and_moved_aside(tmp_path) -> None:
    root = tmp_path / "runs" / "detector"
    for name in ("baseline_v1_20260930_2026092201", "baseline_v1_20260930_2026092201_smoke",
                 ".baseline_v1_20260930_2026092201.sync", "_incompatible_old", "no_checkpoint"):
        (root / name).mkdir(parents=True)
        if name != "no_checkpoint":
            (root / name / "last.pt").write_bytes(b"x")
    assert [p.name for p in kl.local_runs(tmp_path)] == ["baseline_v1_20260930_2026092201"]


def test_package_pins() -> None:
    pinned = {"torch": "2.14.0+cu126", "torchvision": "0.29.0+cu126", "numpy": "2.5.3", "PIL": "11.3.0",
              "cuda_build": "12.6"}
    assert kl._mismatches(pinned, need_cuda=True) == []
    assert kl._mismatches({**pinned, "cuda_build": None}, need_cuda=True) == ["torch"]  # CPU-only build
    assert kl._mismatches({**pinned, "cuda_build": None}, need_cuda=False) == []
    assert kl._mismatches({**pinned, "numpy": "2.1.0", "torch": None}, need_cuda=True) == ["torch", "numpy"]
