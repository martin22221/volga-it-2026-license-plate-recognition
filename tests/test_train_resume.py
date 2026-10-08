"""Interruption safety of scripts/train_baseline.py: --resume, atomic checkpoints, --sync-command.

The central test trains a tiny smoke run twice on CPU -- once straight through,
once killed after an epoch and resumed -- and requires the two to end with
bit-identical weights, optimizer state and metrics. Needs torch and the
Dataset V1 synthetic images; skipped otherwise.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="training stack not installed")
pytest.importorskip("torchvision", reason="training stack not installed")

from scripts import train_baseline as tb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
SYNTHETIC = REPO_ROOT / "dataset" / "images" / "synthetic"

needs_images = pytest.mark.skipif(
    not any(SYNTHETIC.glob("*.jpg")), reason="Dataset V1 synthetic images not present"
)

#: (train, val, epochs) -- small enough for CPU, three epochs so a resume lands mid-run.
TINY = {"detector": (8, 4, 3), "recognizer": (64, 16, 3)}
BATCH = {"detector": 4, "recognizer": 16}
TIMING = ("train_seconds", "val_seconds")


_REAL_EPOCH = tb.RunWriter.epoch


class Crash(Exception):
    """Stands in for the machine dying."""


@pytest.fixture
def fast(monkeypatch):
    """Tiny subsets, no preflight re-hash, no pretrained download; log handlers cleaned up."""
    from src.training import models

    real_build = models.build_detector
    monkeypatch.setattr(tb, "preflight", lambda component, config: ([], []))
    monkeypatch.setattr(tb, "SMOKE", dict(TINY))
    monkeypatch.setattr(models, "build_detector",
                        lambda section, load_pretrained=None: real_build(section, load_pretrained=False))
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()


def _config(root: Path, component: str, name: str = "config.json", **changes) -> Path:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["experiment"]["output_root"] = str(root / "runs")
    config[component]["batch_size"] = BATCH[component]
    config[component].update(changes)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _args(component: str, config: Path, *extra: str) -> list[str]:
    return ["--component", component, "--config", str(config), "--i-have-approval", "test",
            "--smoke", "--device", "cpu", "--workers", "0", *extra]


def _run_dir(root: Path, component: str) -> Path:
    (found,) = (root / "runs" / component).iterdir()
    return found


def _metrics(run_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    return [{k: v for k, v in row.items() if k not in TIMING} for row in rows]


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _same_tensors(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_tensors(x, y) for x, y in zip(a, b))
    return a == b


def _crash_during_epoch(monkeypatch, epoch: int) -> None:
    """Die after metrics.jsonl gets `epoch` but before its last.pt -- the worst moment."""
    real = tb.RunWriter.epoch

    def epoch_then_crash(self, payload):
        real(self, payload)
        if payload["epoch"] == epoch:
            raise Crash

    monkeypatch.setattr(tb.RunWriter, "epoch", epoch_then_crash)


@pytest.fixture
def crashed(tmp_path, fast, monkeypatch):
    """A detector smoke run killed during epoch 2: last.pt holds epoch 1."""
    config = _config(tmp_path / "b", "detector")
    _crash_during_epoch(monkeypatch, 2)
    with pytest.raises(Crash):
        tb.main(_args("detector", config))
    monkeypatch.setattr(tb.RunWriter, "epoch", _REAL_EPOCH)
    return config, _run_dir(tmp_path / "b", "detector")


# ---------------------------------------------------------------------------
# the property that matters: resumed == uninterrupted
# ---------------------------------------------------------------------------


@needs_images
@pytest.mark.parametrize("component", ["detector", "recognizer"])
def test_resumed_run_is_identical_to_an_uninterrupted_one(component, tmp_path, fast, monkeypatch) -> None:
    sync = tmp_path / "sync.py"
    sync.write_text(
        "import os\n"
        "d = os.environ['RUN_DIR']\n"
        "line = '%s %s %s' % (os.environ['RUN_EPOCH'], os.path.exists(os.path.join(d, 'last.pt')),"
        " os.path.exists(os.path.join(d, 'last.pt.tmp')))\n"
        "open(os.path.join(d, 'synced.txt'), 'a').write(line + '\\n')\n",
        encoding="utf-8",
    )
    straight_cfg = _config(tmp_path / "a", component)
    assert tb.main(_args(component, straight_cfg, "--sync-command", f'"{sys.executable}" "{sync}"')) == 0
    straight = _run_dir(tmp_path / "a", component)
    assert (straight / "synced.txt").read_text().split("\n") == [
        "1 True False", "2 True False", "3 True False", "final True False", ""]

    resumed_cfg = _config(tmp_path / "b", component)
    _crash_during_epoch(monkeypatch, 2)
    with pytest.raises(Crash):
        tb.main(_args(component, resumed_cfg))
    monkeypatch.setattr(tb.RunWriter, "epoch", _REAL_EPOCH)
    resumed = _run_dir(tmp_path / "b", component)
    assert _load(resumed / "last.pt")["epoch"] == 1
    assert [row["epoch"] for row in _metrics(resumed)] == [1, 2]  # one line ahead of last.pt

    assert tb.main(_args(component, resumed_cfg, "--resume", str(resumed / "last.pt"))) == 0

    a, b = _load(straight / "last.pt"), _load(resumed / "last.pt")
    assert a["epoch"] == b["epoch"] == 3
    assert _same_tensors(a["model"], b["model"])
    assert _same_tensors(a["resume"]["optimizer"], b["resume"]["optimizer"])
    assert a["resume"]["step"] == b["resume"]["step"] == 3 * a["resume"]["steps_per_epoch"]
    assert a["resume"]["best_epoch"] == b["resume"]["best_epoch"]
    assert a["resume"]["finished"] and b["resume"]["finished"]
    assert _same_tensors(_load(straight / "best.pt")["model"], _load(resumed / "best.pt")["model"])
    assert _metrics(straight) == _metrics(resumed)

    record = json.loads((resumed / "run.json").read_text(encoding="utf-8"))
    assert record["result"] == json.loads((straight / "run.json").read_text(encoding="utf-8"))["result"]
    (entry,) = record["resumes"]
    assert entry["from_checkpoint_epoch"] == 1
    assert [row["epoch"] for row in entry["discarded_metrics"]] == [2]
    assert not list(resumed.glob("*.tmp"))


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def _refused(capsys, argv: list[str], expected: str) -> None:
    assert tb.main(argv) == tb.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert "BLOCKED" in out and expected in out, out


@needs_images
def test_resume_refuses_what_cannot_continue_this_run(crashed, tmp_path, monkeypatch, capsys) -> None:
    config, run_dir = crashed
    last = str(run_dir / "last.pt")

    _refused(capsys, _args("detector", config, "--resume", str(run_dir / "best.pt")), "weights only")
    _refused(capsys, _args("detector", config, "--resume", str(tmp_path / "nope.pt")), "no checkpoint at")
    _refused(capsys, _args("recognizer", config, "--resume", last), "checkpoint component is 'detector'")
    _refused(capsys, [a for a in _args("detector", config, "--resume", last) if a != "--smoke"],
             "checkpoint kind is 'smoke'")
    changed = _config(run_dir.parents[2], "detector", "changed.json", lr0=0.02)  # same output_root
    _refused(capsys, _args("detector", changed, "--resume", last), "different configuration (differs in: detector)")

    real_revision = tb.source_revision
    monkeypatch.setattr(tb, "source_revision", lambda: {**real_revision(), "git_commit": "0" * 40})
    _refused(capsys, _args("detector", config, "--resume", last), "checkpoint git_commit")
    monkeypatch.setattr(tb, "source_revision", real_revision)

    # a fresh start must not silently overwrite a run it could resume
    _refused(capsys, _args("detector", config), "already holds a run; pass --resume")

    # resume format from the future
    future = tmp_path / "future" / "last.pt"
    future.parent.mkdir()
    for name in ("run.json", "metrics.jsonl", "best.pt"):
        shutil.copy2(run_dir / name, future.parent / name)
    checkpoint = _load(run_dir / "last.pt")
    checkpoint["resume"]["format"] = tb.RESUME_FORMAT + 1
    torch.save(checkpoint, future)
    _refused(capsys, _args("detector", config, "--resume", str(future)), "resume format")

    # a run directory that lost its history
    (future.parent / "metrics.jsonl").write_text("", encoding="utf-8")
    checkpoint["resume"]["format"] = tb.RESUME_FORMAT
    torch.save(checkpoint, future)
    _refused(capsys, _args("detector", config, "--resume", str(future)), "metrics.jsonl does not hold exactly epochs 1..1")

    # nothing touched the real run directory
    assert _load(run_dir / "last.pt")["epoch"] == 1


@needs_images
def test_resume_refuses_a_finished_run(tmp_path, fast, capsys) -> None:
    config = _config(tmp_path / "a", "detector")
    assert tb.main(_args("detector", config)) == 0
    run_dir = _run_dir(tmp_path / "a", "detector")
    _refused(capsys, _args("detector", config, "--resume", str(run_dir / "last.pt")), "already finished")


@needs_images
def test_trainer_refuses_when_the_data_no_longer_matches(crashed, monkeypatch, capsys) -> None:
    config, run_dir = crashed
    monkeypatch.setattr(tb, "SMOKE", {**TINY, "detector": (12, 4, 3)})  # 12 train images, not 8
    assert tb.main(_args("detector", config, "--resume", str(run_dir / "last.pt"))) == tb.EXIT_BLOCKED
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert not record.get("resumes"), "a refused resume must not be recorded as one"
    assert _load(run_dir / "last.pt")["epoch"] == 1


@needs_images
def test_best_pt_one_write_behind_is_rebuilt_from_last_pt(crashed, tmp_path) -> None:
    config, run_dir = crashed
    copy = tmp_path / "copy" / run_dir.name
    shutil.copytree(run_dir, copy)
    (copy / "best.pt").unlink()  # died between last.pt and best.pt of epoch 1
    plan, problems = tb.plan_resume(copy / "last.pt", "detector", json.loads(config.read_text()), smoke=True)
    assert not problems and plan.repair_best
    tb.apply_resume_plan(plan)
    rebuilt, last = _load(copy / "best.pt"), _load(copy / "last.pt")
    assert rebuilt["epoch"] == 1 and "resume" not in rebuilt
    assert _same_tensors(rebuilt["model"], last["model"])
    assert [json.loads(x)["epoch"] for x in (copy / "metrics.jsonl").read_text().splitlines()] == [1]


@needs_images
def test_checkpoint_layout_stays_compatible(crashed) -> None:
    """best.pt keeps its weights-only layout (export_onnx reads it); last.pt adds `resume`."""
    _, run_dir = crashed
    best, last = _load(run_dir / "best.pt"), _load(run_dir / "last.pt")
    assert set(best) == {"model", "epoch", "config", "selection"}
    assert set(last) == {"model", "epoch", "config", "selection", "resume"}
    assert set(last["resume"]) >= {"format", "identity", "optimizer", "rng", "step", "steps_per_epoch",
                                   "epochs_planned", "sizes", "best_key", "best_epoch", "finished"}
    assert set(last["resume"]["rng"]) == {"python", "numpy", "torch", "cuda", "loader"}


def test_a_failing_sync_is_logged_not_raised(tmp_path) -> None:
    failing = tb.RunWriter(tmp_path, {}, sync_command=f'"{sys.executable}" -c "raise SystemExit(3)"')
    silent = tb.RunWriter(tmp_path / "x", {})
    try:
        assert failing.sync(1) is False
        assert silent.sync(1) is True  # no command: nothing to do
    finally:
        failing.close()
        silent.close()
