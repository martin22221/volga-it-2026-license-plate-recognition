"""scripts/kaggle_persist_run.py against a fake Kaggle CLI: what it stages, what it runs, what it restores.

Nothing here talks to Kaggle; the real upload/download is exercised only on Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import kaggle_persist_run as kp

DATASET = "someone/baseline-v1-detector-ckpt"


def _run_dir(root: Path, epoch: int = 3) -> Path:
    run = root / "runs" / "detector" / "baseline_v1_20260929_2026092201"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"experiment_id": "detector/baseline_v1_20260929_2026092201"}))
    (run / "metrics.jsonl").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in range(1, epoch + 1)))
    (run / "train.log").write_text("log\n")
    (run / "last.pt").write_bytes(b"last")
    (run / "best.pt").write_bytes(b"best")
    (run / "last.pt.tmp").write_bytes(b"half-written")  # must never be uploaded
    return run


class FakeKaggle:
    def __init__(self, code: int = 0, download_from: Path | None = None) -> None:
        self.calls: list[list[str]] = []
        self.staged: dict[str, bytes] = {}
        self.code = code
        self.download_from = download_from

    def __call__(self, args: list[str]) -> int:
        self.calls.append(args)
        if args[:2] in (["datasets", "version"], ["datasets", "create"]):
            folder = Path(args[args.index("-p") + 1])
            self.staged = {p.name: p.read_bytes() for p in folder.iterdir()}
        if args[:2] == ["datasets", "download"] and self.download_from is not None:
            folder = Path(args[args.index("-p") + 1])
            for p in self.download_from.iterdir():
                (folder / p.name).write_bytes(p.read_bytes())
        return self.code


def test_push_uploads_exactly_the_run_files_as_a_new_private_version(tmp_path, monkeypatch) -> None:
    run = _run_dir(tmp_path)
    fake = FakeKaggle()
    monkeypatch.setenv("RUN_DIR", str(run))
    monkeypatch.setenv("RUN_EPOCH", "3")
    assert kp.main(["push", "--dataset", DATASET], runner=fake) == 0
    (call,) = fake.calls
    assert call[:2] == ["datasets", "version"] and call[call.index("-m") + 1] == "epoch 3"
    assert set(fake.staged) == {"run.json", "metrics.jsonl", "train.log", "last.pt", "best.pt",
                                "dataset-metadata.json"}
    assert fake.staged["last.pt"] == b"last"
    assert json.loads(fake.staged["dataset-metadata.json"])["id"] == DATASET
    assert "--delete-old-versions" not in call and "-d" not in call  # history kept for rollback


def test_init_creates_a_private_dataset(tmp_path) -> None:
    fake = FakeKaggle()
    assert kp.main(["init", "--dataset", DATASET], runner=fake) == 0
    (call,) = fake.calls
    assert call[:2] == ["datasets", "create"] and "--public" not in call and "-u" not in call
    meta = json.loads(fake.staged["dataset-metadata.json"])
    assert meta["id"] == DATASET and 6 <= len(meta["title"]) <= 50


def test_push_refuses_without_a_resumable_run_and_reports_cli_failure(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    fake = FakeKaggle()
    assert kp.main(["push", "--dataset", DATASET, "--run-dir", str(empty)], runner=fake) == kp.EXIT_REFUSED
    assert not fake.calls
    run = _run_dir(tmp_path)
    assert kp.main(["push", "--dataset", DATASET, "--run-dir", str(run)], runner=FakeKaggle(code=1)) == kp.EXIT_CLI
    assert kp.main(["push", "--dataset", "no-slash", "--run-dir", str(run)], runner=fake) == kp.EXIT_REFUSED
    assert kp.main(["check", "--dataset", DATASET], runner=FakeKaggle(code=1)) == kp.EXIT_CLI


def test_restore_downloads_and_puts_the_run_back_under_its_own_name(tmp_path, capsys) -> None:
    run = _run_dir(tmp_path / "origin")
    fake = FakeKaggle(download_from=run)
    dest_root = tmp_path / "work" / "runs"
    assert kp.main(["restore", "--dataset", DATASET, "--dest-root", str(dest_root)], runner=fake) == 0
    assert fake.calls[0][:2] == ["datasets", "download"] and "--unzip" in fake.calls[0]
    restored = dest_root / "detector" / "baseline_v1_20260929_2026092201"
    assert capsys.readouterr().out.strip() == str(restored / "last.pt")
    assert (restored / "last.pt").read_bytes() == b"last"
    assert not (restored / "last.pt.tmp").exists()

    # an existing run is never overwritten by accident
    assert kp.main(["restore", "--source", str(run), "--dest-root", str(dest_root)], runner=fake) == kp.EXIT_REFUSED
    assert kp.main(["restore", "--source", str(run), "--dest-root", str(dest_root), "--overwrite"],
                   runner=fake) == 0


def test_restore_refuses_ambiguous_or_incomplete_sources(tmp_path) -> None:
    fake = FakeKaggle()
    _run_dir(tmp_path / "one")
    _run_dir(tmp_path / "two")
    assert kp.main(["restore", "--source", str(tmp_path), "--dest-root", str(tmp_path / "d")],
                   runner=fake) == kp.EXIT_REFUSED
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / "last.pt").write_bytes(b"x")
    (lonely / "run.json").write_text(json.dumps({"experiment_id": "../escape"}))
    (lonely / "metrics.jsonl").write_text("")
    assert kp.main(["restore", "--source", str(lonely), "--dest-root", str(tmp_path / "d")],
                   runner=fake) == kp.EXIT_REFUSED
    assert kp.main(["restore", "--dest-root", str(tmp_path / "d")], runner=fake) == kp.EXIT_REFUSED
