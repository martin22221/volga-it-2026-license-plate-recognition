#!/usr/bin/env python
"""Keep a training run alive across Kaggle session loss, in a private Kaggle Dataset.

``/kaggle/working`` disappears with the session that wrote it: an interactive
session that dies, times out or is restarted takes ``last.pt`` with it, and a
"Save & Run All" version only publishes its output if the run gets to the end.
The one place a notebook can write that outlives it is Kaggle itself, through
the Kaggle API -- which needs an API token. This script is that path; nothing
here works without the token, and nothing here pretends to.

One-time setup (kaggle.com -> Settings -> API -> Create New Token gives
``kaggle.json``; add ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` from it as notebook
*Secrets*, then export them in the notebook, see docs/baseline_v1_runbook.md)::

    python scripts/kaggle_persist_run.py init  --dataset <user>/<slug>

Then, from ``train_baseline.py --sync-command``, after every epoch::

    python scripts/kaggle_persist_run.py push  --dataset <user>/<slug>

which uploads ``run.json``, ``metrics.jsonl``, ``train.log``, ``last.pt`` and
``best.pt`` of ``$RUN_DIR`` as a new version of the private dataset (older
versions are kept, so a bad upload can be rolled back). On a new session::

    python scripts/kaggle_persist_run.py restore --dataset <user>/<slug>

downloads the latest version and puts the run back at
``runs/<component>/<run>/``, printing the ``last.pt`` to pass to ``--resume``.
``--source <dir>`` restores from an already attached ``/kaggle/input`` copy
instead (beware: an attached dataset is pinned to the version current when it
was attached; the download always gets the latest).

``check --dataset`` proves the CLI, the token and the dataset all work -- run
it before training so a missing secret fails in seconds, not after epoch 20.

Exit codes: 0 ok, 1 the Kaggle CLI failed or is missing, 2 refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The files that make a run directory resumable (train_baseline.py --resume).
RUN_FILES = ("run.json", "metrics.jsonl", "train.log", "last.pt", "best.pt")
REQUIRED = ("run.json", "metrics.jsonl", "last.pt")

EXIT_OK, EXIT_CLI, EXIT_REFUSED = 0, 1, 2

Runner = Callable[[list[str]], int]


#: Seconds each Kaggle CLI call may take before it is killed: a stalled network
#: must end in an error, never in a silent hang.
TIMEOUTS = {"status": 60, "create": 600, "version": 900, "download": 900}


def _default_runner(args: list[str]) -> int:
    exe = shutil.which("kaggle")
    if exe is None:
        print("error: the kaggle CLI is not installed (pip install kaggle)", file=sys.stderr)
        return 127
    timeout = TIMEOUTS.get(args[1] if len(args) > 1 else "", 600)
    try:
        return subprocess.run([exe, *args], check=False, stdin=subprocess.DEVNULL, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"error: `kaggle {' '.join(args[:2])}` gave no result within {timeout} s", file=sys.stderr)
        return 124


def _metadata(dataset: str, title: str | None = None) -> dict:
    owner, _, slug = dataset.partition("/")
    return {
        "id": dataset,
        "title": (title or f"ckpt {slug}")[:50].ljust(6, "_"),
        "licenses": [{"name": "other"}],
    }


def stage_run(run_dir: Path, staging: Path, dataset: str) -> list[str]:
    """Copy the run's files flat into ``staging`` with the dataset metadata. Returns the files copied."""
    missing = [name for name in REQUIRED if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} has no {', '.join(missing)}; nothing resumable to push")
    copied = []
    for name in RUN_FILES:
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, staging / name)
            copied.append(name)
    (staging / "dataset-metadata.json").write_text(json.dumps(_metadata(dataset)), encoding="utf-8")
    return copied


def _valid_dataset(dataset: str) -> bool:
    owner, sep, slug = dataset.partition("/")
    return bool(owner and sep and slug and "/" not in slug)


def cmd_init(dataset: str, run: Runner) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        (staging / "README.md").write_text(
            "Private checkpoint store for a Baseline V1 training run (scripts/kaggle_persist_run.py).\n",
            encoding="utf-8",
        )
        (staging / "dataset-metadata.json").write_text(json.dumps(_metadata(dataset)), encoding="utf-8")
        # Private unless --public is given; it is never given here.
        code = run(["datasets", "create", "-p", str(staging), "-q"])
    return EXIT_OK if code == 0 else EXIT_CLI


def cmd_check(dataset: str, run: Runner) -> int:
    code = run(["datasets", "status", dataset])
    if code != 0:
        print(f"check failed: `kaggle datasets status {dataset}` exited {code}. Is the token exported "
              "(KAGGLE_USERNAME / KAGGLE_KEY) and was `init` run?", file=sys.stderr)
        return EXIT_CLI
    return EXIT_OK


def cmd_push(dataset: str, run_dir: Path, message: str, run: Runner) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            copied = stage_run(run_dir, Path(tmp), dataset)
        except FileNotFoundError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        code = run(["datasets", "version", "-p", tmp, "-m", message, "-q"])
    if code != 0:
        print(f"push failed: kaggle exited {code}", file=sys.stderr)
        return EXIT_CLI
    print(f"pushed {', '.join(copied)} to {dataset} ({message})")
    return EXIT_OK


def find_run(source: Path) -> list[Path]:
    """Directories under ``source`` holding a resumable run."""
    return sorted({p.parent for p in source.rglob("last.pt") if (p.parent / "run.json").is_file()})


def restore_from(source: Path, dest_root: Path, *, overwrite: bool) -> tuple[Path | None, str]:
    """Copy the single run found under ``source`` to ``dest_root/<experiment_id>/``."""
    found = find_run(source)
    if len(found) != 1:
        listed = ", ".join(str(p) for p in found) or "none"
        return None, f"expected exactly one run (run.json + last.pt) under {source}, found {len(found)}: {listed}"
    src = found[0]
    missing = [name for name in REQUIRED if not (src / name).is_file()]
    if missing:
        return None, f"{src} lacks {', '.join(missing)}"
    experiment_id = json.loads((src / "run.json").read_text(encoding="utf-8")).get("experiment_id", "")
    parts = experiment_id.split("/")
    if len(parts) != 2 or not all(parts) or any(part in (".", "..") for part in parts):
        return None, f"run.json has no usable experiment_id ({experiment_id!r})"
    dest = dest_root / parts[0] / parts[1]
    if (dest / "last.pt").exists() and not overwrite:
        return None, f"{dest} already holds a run; pass --overwrite to replace it"
    dest.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dest / name)
    return dest / "last.pt", ""


def cmd_restore(dataset: str | None, source: Path | None, dest_root: Path, overwrite: bool, run: Runner) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        if source is None:
            code = run(["datasets", "download", dataset, "-p", tmp, "--unzip", "-q"])
            if code != 0:
                print(f"download failed: kaggle exited {code}", file=sys.stderr)
                return EXIT_CLI
            source = Path(tmp)
        last, problem = restore_from(source, dest_root, overwrite=overwrite)
    if last is None:
        print(f"refused: {problem}", file=sys.stderr)
        return EXIT_REFUSED
    print(last)
    return EXIT_OK


def main(argv: Sequence[str] | None = None, runner: Runner = _default_runner) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "check", "push", "restore"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=name != "restore", help="<user>/<slug> of the private dataset")
        if name == "push":
            p.add_argument("--run-dir", type=Path, default=None, help="default: $RUN_DIR (set by --sync-command)")
            p.add_argument("--message", default=None, help="default: 'epoch $RUN_EPOCH'")
        if name == "restore":
            p.add_argument("--source", type=Path, default=None, help="restore from this directory, not a download")
            p.add_argument("--dest-root", type=Path, default=REPO_ROOT / "runs")
            p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset is not None and not _valid_dataset(args.dataset):
        print(f"refused: --dataset must be <user>/<slug>, got {args.dataset!r}", file=sys.stderr)
        return EXIT_REFUSED
    if args.command == "init":
        return cmd_init(args.dataset, runner)
    if args.command == "check":
        return cmd_check(args.dataset, runner)
    if args.command == "push":
        run_dir = args.run_dir or (Path(os.environ["RUN_DIR"]) if os.environ.get("RUN_DIR") else None)
        if run_dir is None:
            print("refused: no --run-dir and no RUN_DIR in the environment", file=sys.stderr)
            return EXIT_REFUSED
        message = args.message or f"epoch {os.environ.get('RUN_EPOCH', '?')}"
        return cmd_push(args.dataset, run_dir, message, runner)
    if (args.dataset is None) == (args.source is None):
        print("refused: restore takes exactly one of --dataset or --source", file=sys.stderr)
        return EXIT_REFUSED
    return cmd_restore(args.dataset, args.source, args.dest_root, args.overwrite, runner)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
