#!/usr/bin/env python
"""Baseline V1 training -- refuses unless the preflight passes and a person approved it.

This script is the boundary between "ready to train" and "training". It checks
everything that must be true before a run is worth starting, prints exactly what
would happen, and then **stops** unless ``--i-have-approval`` is passed with the
approving person's name.

That guard is the point. A training run consumes hours of borrowed GPU, writes
weights, and -- if it is pointed at the wrong split -- destroys the value of a
holdout that took an acquisition, a human review and a promotion to build.

Preflight checks, all of which must pass:

* the Dataset V1 freeze exists and still verifies;
* the split audit is clean;
* the requested component's training and validation splits contain no real
  photograph;
* the real holdout is not reachable from the training configuration;
* the framework is installed (``requirements-train.txt``).

What a run does (``docs/baseline_v1.md``, ``configs/baseline_v1.json``):

* trains on ``synthetic:train`` only, and selects the checkpoint on
  ``synthetic:val`` only -- no real image is read anywhere in this file;
* seeds Python, NumPy and torch from the config, and asks torch for
  deterministic kernels (warn-only: CTC and some CUDA kernels have none);
* writes ``runs/<component>/<experiment>_<YYYYMMDD>_<seed>[_smoke]/`` with
  ``run.json`` (identity, versions, hardware, config, result), ``metrics.jsonl``
  (one line per epoch), ``train.log``, ``best.pt`` and ``last.pt``;
* ``best.pt`` holds the selected weights; ``last.pt`` holds the latest weights
  **and** everything needed to continue the same run: optimizer state, the
  shuffle generator and the Python / NumPy / torch / CUDA RNG states, the
  global step (which is the LR schedule's position), the best selection key
  and epoch (which are also the early-stopping state), and the run's identity.
  Both are written atomically (temporary file, then rename), so an
  interruption mid-write leaves the previous file intact.

``--resume runs/<component>/<run>/last.pt`` continues that run in its own
directory from the next epoch. It refuses a checkpoint written before resume
support, one from another component, kind (smoke/baseline), configuration,
dataset or code commit, a run that already finished, and a run directory whose
``run.json`` / ``metrics.jsonl`` / ``best.pt`` do not agree with it.

``--sync-command CMD`` runs a shell command after every epoch's checkpoints
are on disk (and once more at the end), with ``RUN_DIR`` and ``RUN_EPOCH`` in
its environment -- the hook that copies the run somewhere that outlives the
machine (``scripts/kaggle_persist_run.py`` on Kaggle). A failing sync is logged
loudly but does not stop training.

``--smoke`` trains on a small deterministic subset for a couple of epochs and
writes to a ``_smoke`` directory. It proves the loop runs end to end; it is
**not** a baseline and its numbers mean nothing.

Usage::

    python scripts/train_baseline.py --component detector                       # preflight only
    python scripts/train_baseline.py --component recognizer --i-have-approval "<name>"
    python scripts/train_baseline.py --component detector --i-have-approval "<name>" --smoke
    python scripts/train_baseline.py --component detector --i-have-approval "<name>" \
        --resume runs/detector/<run>/last.pt

Exit codes::

    0  preflight passed (and, if approved, the run finished)
    1  the configuration or dataset could not be read
    2  a preflight check failed, the resume checkpoint was refused, or the run
       is not authorised
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.data import load_samples, load_split, training_pool  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
FREEZE = REPO_ROOT / "dataset" / "splits" / "dataset_v1.json"
COMPONENTS = ("detector", "recognizer")

#: Frameworks a real run needs (``requirements-train.txt``).
REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    # torchvision, not a third-party detector repository: the selected detector
    # is torchvision's own SSDlite (BSD-3), so the detector needs nothing the
    # recogniser does not already need. See docs/baseline_v1.md section 3.
    "detector": ("torch", "torchvision"),
    "recognizer": ("torch", "torchvision"),
}

#: Licences of every third-party component a run would pull in, so the check is
#: mechanical rather than remembered. Copyleft is refused: the project's source
#: is Apache-2.0 (LICENSE), and an AGPL dependency would override that choice
#: at submission time. torch's wheel metadata lists BSD-3-Clause for PyTorch
#: itself plus permissive bundled components (Apache-2.0, MIT, BSL-1.0, BSD-2).
COMPONENT_LICENCES: dict[str, str] = {
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
}
REFUSED_LICENCE_TOKENS: tuple[str, ...] = ("agpl", "gpl-3", "gplv3")

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKED = 0, 1, 2

#: --smoke subset sizes. Deliberately tiny; the smoke run proves plumbing only.
SMOKE = {"detector": (64, 32, 2), "recognizer": (1024, 256, 2)}  # train, val, epochs

#: Layout version of the ``resume`` block inside last.pt. Bump it when the
#: layout changes; --resume refuses any other version.
RESUME_FORMAT = 1

#: Seconds a --sync-command may take before it is abandoned (training goes on).
SYNC_TIMEOUT = 1800

#: Seconds a data-loader worker may take to deliver a batch before the run
#: fails loudly instead of hanging (only with --workers > 0).
LOADER_TIMEOUT = 600

#: Seconds the COCO-pretrained weight download may stall before it fails.
DOWNLOAD_TIMEOUT = 120

#: Log a progress line every this many training steps (and on the first).
PROGRESS_EVERY = 50

logger = logging.getLogger("train")


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - odd import states
        return False


def training_bundle_mode() -> bool:
    """True on a GPU machine unpacked from scripts/pack_training_bundle.py.

    The bundle deliberately carries no real photograph (so the holdout cannot
    leak into a training machine), which means the full freeze check -- it
    re-hashes all 12,013 images -- cannot run there. The bundle marks itself
    with ``TRAINING_BUNDLE.json``.
    """
    return (REPO_ROOT / "TRAINING_BUNDLE.json").is_file()


def verify_training_bundle() -> list[str]:
    """Everything the full freeze check proves about the data a run can touch.

    meta.csv and both split manifests against their frozen digests; the
    per-file list against the frozen combined digest; every synthetic image
    against that list; and no real image present at all.
    """
    import csv

    from src.dataset_v1 import combined_digest, sha256_file

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    dataset = REPO_ROOT / "dataset"
    problems: list[str] = []
    for name, key in (("meta.csv", "meta_sha256"), ("splits/synthetic_splits.csv", "synthetic_splits_sha256"),
                      ("splits/real_splits.csv", "real_splits_sha256")):
        if sha256_file(dataset / name) != freeze[key]:
            problems.append(f"dataset/{name} differs from the Dataset V1 freeze")
    with (dataset / "splits" / "dataset_v1_files.csv").open(encoding="utf-8", newline="") as handle:
        files = {r["image"]: r["sha256"] for r in csv.DictReader(handle, delimiter=";")}
    if combined_digest(files) != freeze["combined_image_digest"]:
        problems.append("dataset_v1_files.csv does not reproduce the frozen combined image digest")
    real_present = [p for p in (dataset / "images" / "real").rglob("*") if p.is_file() and p.name != ".gitkeep"]         if (dataset / "images" / "real").is_dir() else []
    if real_present:
        problems.append(f"{len(real_present)} real image(s) present in a training bundle; there must be none")
    synthetic = {k: v for k, v in files.items() if k.startswith("images/synthetic/")}
    bad = [k for k, v in synthetic.items() if not (dataset / k).is_file() or sha256_file(dataset / k) != v]
    if len(synthetic) != 12_000 or bad:
        problems.append(f"synthetic images: {len(synthetic)} listed, {len(bad)} missing or altered")
    return problems


def preflight(component: str, config: dict) -> tuple[list[str], list[str]]:
    """Everything that must hold before a run starts. Returns (blocking, notes)."""
    blocking: list[str] = []
    notes: list[str] = []

    if not FREEZE.is_file():
        blocking.append(f"no Dataset V1 freeze at {FREEZE.relative_to(REPO_ROOT)}")
    elif training_bundle_mode():
        print("  ...     verifying the training bundle (re-hashes 12,000 images, about a minute)", flush=True)
        problems = verify_training_bundle()
        if problems:
            blocking += problems
        else:
            notes.append(
                "training bundle: no real photograph present; meta.csv, both split manifests and "
                "every synthetic image verified against the Dataset V1 freeze"
            )
    else:
        check = subprocess.run(
            [sys.executable, "scripts/freeze_dataset_v1.py", "--check"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if check.returncode != 0:
            blocking.append("the Dataset V1 freeze does not verify; run --check and read the report")
        else:
            notes.append("Dataset V1 freeze verifies")

    section = config.get(component, {})
    train_split = section.get("train_split")
    val_split = section.get("val_split")
    if not train_split:
        blocking.append(f"config has no {component}.train_split")

    samples = load_samples()
    for name in (train_split, val_split):
        if not name:
            continue
        rows = load_split(name, samples=samples)
        if not rows:
            blocking.append(f"split {name!r} is empty")
        real = [r for r in rows if not r.is_synthetic]
        if real:
            blocking.append(
                f"split {name!r} contains {len(real)} real photograph(s); baseline v1 "
                "fits on synthetic only, and the real images are the evaluation set"
            )
        else:
            notes.append(f"{name}: {len(rows)} sample(s), all synthetic")

    if "holdout" in json.dumps({k: v for k, v in section.items()}):
        blocking.append(f"{component} configuration mentions a holdout split; it must not")

    refused = [
        f"{name} ({licence})"
        for name, licence in COMPONENT_LICENCES.items()
        if name in REQUIRED_MODULES[component]
        and any(token in licence.lower() for token in REFUSED_LICENCE_TOKENS)
    ]
    if refused:
        blocking.append(f"copyleft dependency refused: {', '.join(refused)}")
    else:
        declared = ", ".join(
            f"{n}={COMPONENT_LICENCES.get(n, '?')}" for n in REQUIRED_MODULES[component]
        )
        notes.append(f"dependency licences: {declared}")

    missing = [m for m in REQUIRED_MODULES[component] if not _installed(m)]
    if missing:
        blocking.append(
            f"missing framework(s): {', '.join(missing)}. Install requirements-train.txt "
            "into an isolated environment first"
        )
    return blocking, notes


# ---------------------------------------------------------------------------
# run bookkeeping
# ---------------------------------------------------------------------------


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    except OSError:  # git not installed
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def source_revision() -> dict:
    """Which commit this code is. An unpacked training bundle has no .git, so
    its commit comes from the bundle's own marker, written by the packer from a
    clean tree."""
    if training_bundle_mode():
        marker = json.loads((REPO_ROOT / "TRAINING_BUNDLE.json").read_text(encoding="utf-8"))
        return {"git_commit": marker["git_commit"], "git_dirty": False, "source": "TRAINING_BUNDLE.json"}
    return {
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
        "source": "git",
    }


def environment() -> dict:
    import numpy
    import PIL
    import torch
    import torchvision

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
    }


def dataset_identity() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    return {
        "dataset_version": freeze["dataset_version"],
        "freeze_file_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
        "meta_sha256": freeze["meta_sha256"],
        "combined_image_digest": freeze["combined_image_digest"],
        "synthetic_splits_sha256": freeze["synthetic_splits_sha256"],
        "real_splits_sha256": freeze["real_splits_sha256"],
        "generator_seal_sha256": freeze["generator"]["production_seal_sha256"],
    }


def seed_everything(seed: int, deterministic: bool) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:  # pragma: no cover - runs in loader workers
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def probe_dataset(dataset, name: str) -> None:
    """Decode and augment one training sample before any model exists, so a missing or
    unreadable image fails in seconds, with the file named, not after model start-up."""
    started = time.perf_counter()
    try:
        item = dataset[0]
    except Exception as exc:
        raise RuntimeError(f"{name} dataset: the first training sample cannot be loaded: {exc}") from exc
    shape = tuple(item[0].shape) if hasattr(item[0], "shape") else "?"
    logger.info("stage: %s data probe ok: sample 0 -> %s in %.2f s", name, shape, time.perf_counter() - started)


def _subset(rows: list, n: int) -> list:
    """A deterministic, evenly strided subset -- keeps every class represented."""
    if n >= len(rows):
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def _lr_at(step: int, total: int, warmup: int, base: float, final_fraction: float) -> float:
    if warmup and step < warmup:
        return base * (0.1 + 0.9 * step / warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * (final_fraction + (1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))


class RunWriter:
    def __init__(self, run_dir: Path, record: dict, sync_command: str | None = None) -> None:
        self.dir = run_dir
        self.record = record
        self.sync_command = sync_command
        self._sync: subprocess.Popen | None = None
        self._sync_epoch: int | str | None = None
        self._sync_started = 0.0
        self._sync_ok = True
        run_dir.mkdir(parents=True, exist_ok=True)
        self._handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(self._handler)
        self.save()

    def save(self) -> None:
        _atomic_write_text(self.dir / "run.json", json.dumps(self.record, indent=1, default=str) + "\n")

    def epoch(self, payload: dict) -> None:
        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def identity(self) -> dict:
        """What a checkpoint must agree with to continue this run."""
        keys = ("experiment_id", "component", "kind", "config", "dataset", "git_commit")
        return {key: self.record.get(key) for key in keys}

    @property
    def snapshot_dir(self) -> Path:
        """What a sync uploads: a copy, so training can overwrite the run while it uploads."""
        return self.dir.parent / f".{self.dir.name}.sync"

    def _reap(self, wait: float = 0.0) -> bool:
        """True when no sync is in flight any more (reporting its outcome once)."""
        if self._sync is None:
            return True
        try:
            code = self._sync.wait(timeout=wait) if wait else self._sync.poll()
        except subprocess.TimeoutExpired:
            code = None
        if code is None and time.monotonic() - self._sync_started > SYNC_TIMEOUT:
            self._sync.kill()
            code = self._sync.wait()
            logger.warning("SYNC of epoch %s killed after %d s", self._sync_epoch, SYNC_TIMEOUT)
        if code is None:
            return False
        self._sync_ok = code == 0
        if self._sync_ok:
            logger.info("synced epoch %s", self._sync_epoch)
        else:
            logger.warning("SYNC FAILED for epoch %s (exit %s, see sync.log): that state is NOT persisted "
                           "off this machine; training continues", self._sync_epoch, code)
        self._sync = None
        return True

    def sync(self, epoch: int | str) -> bool:
        """Start --sync-command in the background on a snapshot of the finished epoch.

        Never raises and never waits: if the previous upload is still running,
        this epoch is skipped (the next sync carries newer state anyway), so the
        network can never stall training.
        """
        if not self.sync_command:
            return True
        if not self._reap():
            logger.warning("sync of epoch %s still running; epoch %s not synced (the next sync carries it)",
                           self._sync_epoch, epoch)
            return False
        try:
            snapshot = self.snapshot_dir
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in ("run.json", "metrics.jsonl", "train.log", "last.pt", "best.pt"):
                if (self.dir / name).is_file():
                    shutil.copy2(self.dir / name, snapshot / name)
            env = {**os.environ, "RUN_DIR": str(snapshot), "RUN_EPOCH": str(epoch)}
            with (self.dir / "sync.log").open("ab") as log:
                self._sync = subprocess.Popen(self.sync_command, shell=True, env=env, stdout=log,
                                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        except OSError as exc:
            logger.warning("SYNC not started for epoch %s (%s); training continues", epoch, exc)
            self._sync_ok = False
            return False
        self._sync_epoch, self._sync_started = epoch, time.monotonic()
        logger.info("sync of epoch %s started in the background (sync.log)", epoch)
        return True

    def finish_sync(self) -> bool:
        """Wait (bounded by SYNC_TIMEOUT) for the last sync. True when it succeeded or there was none."""
        if not self.sync_command:
            return True
        while not self._reap(wait=5.0):
            pass
        return self._sync_ok

    def close(self) -> None:
        logging.getLogger().removeHandler(self._handler)
        self._handler.close()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save(payload: dict, path: Path) -> None:
    """torch.save to a temporary file, then rename: a kill mid-write keeps the old file."""
    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _checkpoint(path: Path, model, meta: dict, resume: dict | None = None) -> None:
    payload = {"model": model.state_dict(), **meta}
    if resume is not None:
        payload["resume"] = resume
    _atomic_save(payload, path)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------


class ResumeMismatch(RuntimeError):
    """The checkpoint cannot continue this run."""


def capture_rng(loader_generator) -> dict:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": loader_generator.get_state(),
    }


def restore_rng(state: dict, loader_generator) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
        else:
            logger.warning("CUDA RNG state saved for %d device(s), %d present; not restored",
                           len(cuda), torch.cuda.device_count())
    loader_generator.set_state(state["loader"])


def resume_block(run: RunWriter, optimizer, loader_generator, *, step: int, steps_per_epoch: int,
                 epochs: int, sizes: dict, best, best_epoch: int, finished: bool) -> dict:
    return {
        "format": RESUME_FORMAT,
        "identity": run.identity(),
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng(loader_generator),
        "step": step,
        "steps_per_epoch": steps_per_epoch,
        "epochs_planned": epochs,
        "sizes": sizes,
        "best_key": list(best) if best is not None else None,
        "best_epoch": best_epoch,
        "finished": finished,
    }


def restore_training(checkpoint: dict, model, optimizer, loader_generator, *, steps_per_epoch: int,
                     epochs: int, sizes: dict) -> tuple[int, tuple | None, int, int]:
    """Load a last.pt into a freshly built trainer. Returns (epochs done, best key, best epoch, step)."""
    state = checkpoint["resume"]
    done = int(checkpoint["epoch"])
    problems = []
    if state["steps_per_epoch"] != steps_per_epoch:
        problems.append(f"steps per epoch {state['steps_per_epoch']} in the checkpoint, {steps_per_epoch} now")
    if state["epochs_planned"] != epochs:
        problems.append(f"epochs planned {state['epochs_planned']} in the checkpoint, {epochs} now")
    if state["sizes"] != sizes:
        problems.append(f"split sizes {state['sizes']} in the checkpoint, {sizes} now")
    if state["step"] != done * state["steps_per_epoch"]:
        problems.append(f"step {state['step']} does not match {done} completed epoch(s)")
    if problems:
        raise ResumeMismatch("; ".join(problems))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], loader_generator)
    best = tuple(state["best_key"]) if state["best_key"] is not None else None
    logger.info("resumed after epoch %d (step %d, best epoch %d)", done, state["step"], state["best_epoch"])
    return done, best, int(state["best_epoch"]), int(state["step"])


def _save_epoch(run: RunWriter, model, optimizer, loader_generator, *, epoch: int, section: dict, key,
                improved: bool, best, best_epoch: int, stop: bool, step: int, steps_per_epoch: int,
                epochs: int, sizes: dict) -> None:
    """last.pt (with resume state), then best.pt if improved, then the sync hook.

    The order matters: if the machine dies between the two writes, last.pt
    already names this epoch as the best and holds its weights, so --resume
    rebuilds best.pt from it.
    """
    meta = {"epoch": epoch, "config": section, "selection": list(key)}
    resume = resume_block(run, optimizer, loader_generator, step=step, steps_per_epoch=steps_per_epoch,
                          epochs=epochs, sizes=sizes, best=best, best_epoch=best_epoch,
                          finished=stop or epoch == epochs)
    _checkpoint(run.dir / "last.pt", model, meta, resume)
    if improved:
        _checkpoint(run.dir / "best.pt", model, meta)
    run.sync(epoch)


@dataclass
class ResumePlan:
    checkpoint: dict
    run_dir: Path
    record: dict
    kept_metrics: list[str]
    discarded_metrics: list[str]
    repair_best: bool


def _epoch_of(line: str) -> int | None:
    try:
        return int(json.loads(line)["epoch"])
    except (ValueError, KeyError, TypeError):
        return None


def plan_resume(path: Path, component: str, config: dict, *, smoke: bool) -> tuple[ResumePlan | None, list[str]]:
    """Check that ``path`` can continue a run of exactly this invocation. Returns (plan, problems)."""
    import torch

    if not path.is_file():
        return None, [f"no checkpoint at {path}"]
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a refusal
        return None, [f"{path} could not be read: {exc}"]
    state = checkpoint.get("resume") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        return None, [f"{path} holds weights only (a best.pt, or a last.pt written before resume support); "
                      "it cannot continue a run"]
    if state.get("format") != RESUME_FORMAT:
        return None, [f"{path} has resume format {state.get('format')!r}; this script reads {RESUME_FORMAT}"]

    problems: list[str] = []
    identity = state["identity"]
    expected = {
        "component": component,
        "kind": "smoke" if smoke else "baseline",
        "dataset": dataset_identity(),
        "git_commit": source_revision()["git_commit"],
    }
    for name, want in expected.items():
        if identity.get(name) != want:
            problems.append(f"checkpoint {name} is {identity.get(name)!r}, this invocation is {want!r}")
    if identity.get("config") != config:
        saved = identity.get("config") or {}
        differing = sorted(k for k in set(saved) | set(config) if saved.get(k) != config.get(k))
        problems.append(f"checkpoint was trained with a different configuration (differs in: {', '.join(differing)})")
    done = int(checkpoint["epoch"])
    if state.get("finished"):
        problems.append(f"the run already finished at epoch {done} (early stop or last epoch); nothing to resume")

    run_dir = path.resolve().parent
    record: dict = {}
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        problems.append(f"no run.json beside {path.name}; resume needs the whole run directory")
    else:
        record = json.loads(run_json.read_text(encoding="utf-8"))
        if record.get("experiment_id") != identity.get("experiment_id"):
            problems.append(f"run.json is {record.get('experiment_id')!r}, "
                            f"the checkpoint is {identity.get('experiment_id')!r}")

    # metrics.jsonl is written before last.pt, so it may be one epoch ahead.
    metrics = run_dir / "metrics.jsonl"
    lines = [ln for ln in metrics.read_text(encoding="utf-8").splitlines() if ln.strip()] if metrics.is_file() else []
    epochs = [_epoch_of(ln) for ln in lines]
    kept = [ln for ln, e in zip(lines, epochs) if e is not None and e <= done]
    discarded = [ln for ln, e in zip(lines, epochs) if e is None or e > done]
    if [e for e in epochs if e is not None and e <= done] != list(range(1, done + 1)):
        problems.append(f"metrics.jsonl does not hold exactly epochs 1..{done}")

    # best.pt is written after last.pt, so it may be one epoch behind.
    repair_best = False
    best_epoch = int(state["best_epoch"])
    best_path = run_dir / "best.pt"
    on_disk = None
    if best_path.is_file():
        try:
            on_disk = torch.load(best_path, map_location="cpu", weights_only=False).get("epoch")
        except Exception:  # noqa: BLE001
            on_disk = None
    if on_disk != best_epoch:
        if best_epoch == done:
            repair_best = True  # last.pt holds exactly the best weights
        else:
            problems.append(f"best.pt holds epoch {on_disk}, but the run's best is epoch {best_epoch}; "
                            "the selected weights are missing")

    if problems:
        return None, problems
    return ResumePlan(checkpoint, run_dir, record, kept, discarded, repair_best), []


def apply_resume_plan(plan: ResumePlan) -> None:
    """Bring the run directory back to the checkpoint's epoch before training continues."""
    _atomic_write_text(plan.run_dir / "metrics.jsonl", "".join(line + "\n" for line in plan.kept_metrics))
    if plan.repair_best:
        ckpt = plan.checkpoint
        _atomic_save({"model": ckpt["model"], "epoch": ckpt["epoch"], "config": ckpt["config"],
                      "selection": ckpt["selection"]}, plan.run_dir / "best.pt")


# ---------------------------------------------------------------------------
# recogniser
# ---------------------------------------------------------------------------


def evaluate_recognizer_torch(model, samples, regions, device, char_threshold: float, batch: int = 256):
    """The deployed two-pass read, with the torch model standing in for ONNX."""
    import numpy as np
    import torch

    from src.recognition import read_plates

    model.eval()

    def run(x: np.ndarray):
        with torch.no_grad():
            out = model(torch.from_numpy(x).to(device))
        return tuple(o.float().cpu().numpy() for o in out)

    readings = []
    for start in range(0, len(samples), batch):
        chunk = regions[start : start + batch]
        readings += read_plates(run, [(r.image(), r.box) for r in chunk], char_threshold=char_threshold)
    return readings


def train_recognizer(config: dict, run: RunWriter, device, *, smoke: bool, workers: int,
                     resume: dict | None = None) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.training.evaluation import evaluate_readings, selection_score
    from src.training.models import build_recognizer, count_parameters
    from src.training.torch_data import RecognizerDataset, cut_regions

    section = config["recognizer"]
    seed = int(config["experiment"]["seed"])
    char_threshold = float(config["evaluation"]["char_confidence_for_hash"])
    samples = load_samples()
    train = training_pool(samples)
    val = load_split(section["val_split"], samples=samples)
    epochs = int(section["epochs"])
    if smoke:
        n_train, n_val, epochs = SMOKE["recognizer"]
        train, val = _subset(train, n_train), _subset(val, n_val)
    assert all(s.is_synthetic for s in train + val), "a real photograph reached the recogniser"

    started = time.perf_counter()
    cut_workers = max(1, min(8, os.cpu_count() or 1))
    train_regions = cut_regions(train, workers=cut_workers)
    val_regions = cut_regions(val, workers=cut_workers)
    logger.info("regions cut: %d train, %d val in %.1f s", len(train), len(val), time.perf_counter() - started)

    dataset = RecognizerDataset(train, train_regions, train=True, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(section["batch_size"]), shuffle=True, generator=generator,
        num_workers=workers, worker_init_fn=_worker_init if workers else None, drop_last=True,
    )
    model = build_recognizer(section).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(section["lr"]), weight_decay=float(section["weight_decay"]))
    total_steps = epochs * len(loader)
    patience = int(section["patience"])
    weights = (float(section["ctc_loss_weight"]), float(section["type_loss_weight"]), float(section["corner_loss_weight"]))
    run.record.update({
        "parameters": count_parameters(model),
        "train_samples": len(train), "val_samples": len(val),
        "epochs_planned": epochs, "steps_planned": total_steps,
    })
    run.save()

    sizes = {"train": len(train), "val": len(val)}
    start, best_key, best_epoch, step = 0, None, -1, 0
    if resume is not None:
        start, best_key, best_epoch, step = restore_training(
            resume, model, optimizer, generator, steps_per_epoch=len(loader), epochs=epochs, sizes=sizes,
        )
    for epoch in range(start, epochs):
        dataset.epoch = epoch
        model.train()
        sums = {"loss": 0.0, "ctc": 0.0, "type": 0.0, "corner": 0.0}
        seen = 0
        t0 = time.perf_counter()
        for corner_view, strip, corners, type_index, targets, lengths in loader:
            lr = _lr_at(step, total_steps, 0, float(section["lr"]), 0.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            corner_view, strip = corner_view.to(device), strip.to(device)
            corners, type_index = corners.to(device), type_index.to(device)
            log_probs, type_logits, _ = model(strip)
            _, _, corner_pred = model(corner_view)

            readable = lengths > 0
            if readable.any():
                lp = log_probs[readable.to(device)].permute(1, 0, 2)
                tl = lengths[readable]
                flat = torch.cat([t[:n] for t, n in zip(targets[readable], tl)])
                ctc = F.ctc_loss(
                    lp.float(), flat.to(device), torch.full((lp.shape[1],), lp.shape[0], dtype=torch.long),
                    tl, blank=0, zero_infinity=True,
                )
            else:
                ctc = log_probs.sum() * 0.0
            type_loss = F.cross_entropy(type_logits, type_index)
            corner_loss = F.smooth_l1_loss(corner_pred, corners, beta=0.02)
            loss = weights[0] * ctc + weights[1] * type_loss + weights[2] * corner_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            n = strip.shape[0]
            seen += n
            for key, value in (("loss", loss), ("ctc", ctc), ("type", type_loss), ("corner", corner_loss)):
                sums[key] += float(value.detach()) * n
        train_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = evaluate_recognizer_torch(model, val, val_regions, device, char_threshold)
        report = evaluate_readings(val, result)
        key = selection_score(report)
        improved = best_key is None or key > best_key
        payload = {
            "epoch": epoch + 1, "lr": lr, "train_seconds": round(train_seconds, 1),
            "val_seconds": round(time.perf_counter() - t1, 1),
            **{f"train_{k}": round(v / max(seen, 1), 5) for k, v in sums.items()},
            "val_type_and_string": report["type_and_string_correct"]["value"],
            "val_ocr_exact_raw": report["ocr_raw"]["exact"]["value"],
            "val_cer_raw": report["ocr_raw"]["cer"],
            "val_type_accuracy": report["classification"]["accuracy"]["value"],
            "val_type_macro_f1": report["classification"]["per_class"]["macro"]["f1"],
            "improved": improved,
        }
        run.epoch(payload)
        logger.info("epoch %s", json.dumps(payload))
        if improved:
            best_key, best_epoch = key, epoch + 1
        stop = not improved and epoch + 1 - best_epoch >= patience
        _save_epoch(run, model, optimizer, generator, epoch=epoch + 1, section=section, key=key,
                    improved=improved, best=best_key, best_epoch=best_epoch, stop=stop, step=step,
                    steps_per_epoch=len(loader), epochs=epochs, sizes=sizes)
        if stop:
            logger.info("early stop: no improvement for %d epochs", patience)
            break

    return {"best_epoch": best_epoch, "best_selection_key": list(best_key or ()), "epochs_run": epoch + 1}


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def detector_predictions(model, images, device, section: dict, confidence_floor: float, batch: int = 16):
    """Validation predictions through the deployed postprocessing (src.onnx_backend)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from src.onnx_backend import postprocess
    from src.training.models import DetectorExport
    from src.training.torch_data import DetectorDataset, detector_collate

    size = int(section["input_size"])
    model.eval()
    export = DetectorExport(model.cpu(), size).to(device)
    model.to(device)
    dataset = DetectorDataset(images, size, augment=None, seed=0)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, collate_fn=detector_collate)
    out: dict[str, list] = {}
    index = 0
    with torch.no_grad():
        for tensors, _, sizes in loader:
            x = (torch.stack(tensors).to(device) - 0.5) / 0.5
            for i in range(x.shape[0]):
                boxes, scores = export(x[i : i + 1])
                w, h = sizes[i].tolist()
                out[images[index][1][0].image] = postprocess(
                    boxes[0].cpu().numpy(), scores[0].cpu().numpy(), size=size,
                    scale_x=w / size, scale_y=h / size, score_threshold=confidence_floor,
                    nms_threshold=float(section["nms_thresh"]), topk=int(section["topk_candidates"]),
                    max_detections=int(section["detections_per_img"]),
                )
                index += 1
    return out


def train_detector(config: dict, run: RunWriter, device, *, smoke: bool, workers: int,
                   resume: dict | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from src.training.evaluation import evaluate_detections, load_difficulty
    from src.training.models import build_detector, count_parameters
    from src.training.torch_data import DetectorDataset, detector_collate, group_by_image

    section = config["detector"]
    seed = int(config["experiment"]["seed"])
    confidence = float(config["evaluation"]["detector_confidence"])
    logger.info("stage: loading Dataset V1 annotations and splits")
    samples = load_samples()
    train_images = group_by_image(training_pool(samples))
    val_images = group_by_image(load_split(section["val_split"], samples=samples))
    epochs = int(section["epochs"])
    if smoke:
        n_train, n_val, epochs = SMOKE["detector"]
        train_images, val_images = _subset(train_images, n_train), _subset(val_images, n_val)
    assert all(r.is_synthetic for _, rows in train_images + val_images for r in rows)
    val_truth = {rows[0].image: rows for _, rows in val_images}
    difficulty = load_difficulty()
    logger.info("stage: %d train / %d val images", len(train_images), len(val_images))

    dataset = DetectorDataset(train_images, int(section["input_size"]), augment=section["augmentation"], seed=seed)
    probe_dataset(dataset, "detector")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(section["batch_size"]), shuffle=True, generator=generator,
        num_workers=workers, worker_init_fn=_worker_init if workers else None,
        collate_fn=detector_collate, drop_last=True, timeout=LOADER_TIMEOUT if workers else 0,
    )
    logger.info("stage: building the detector (COCO weights: %s)",
                "cached or downloaded now" if section.get("pretrained", True) else "not used")
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(DOWNLOAD_TIMEOUT)  # no loader worker exists yet; restored at once
    try:
        model = build_detector(section)
    finally:
        socket.setdefaulttimeout(previous)
    model = model.to(device)
    logger.info("stage: detector on %s; %d steps per epoch, %d worker(s)", device, len(loader), workers)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(section["lr0"]), momentum=float(section["momentum"]),
        weight_decay=float(section["weight_decay"]),
    )
    total_steps = epochs * len(loader)
    warmup = int(section["warmup_epochs"]) * len(loader)
    patience = int(section["patience"])
    run.record.update({
        "parameters": count_parameters(model),
        "train_images": len(train_images), "val_images": len(val_images),
        "epochs_planned": epochs, "steps_planned": total_steps,
    })
    run.save()

    sizes = {"train": len(train_images), "val": len(val_images)}
    start, best, best_epoch, step = 0, None, -1, 0
    if resume is not None:
        start, best, best_epoch, step = restore_training(
            resume, model, optimizer, generator, steps_per_epoch=len(loader), epochs=epochs, sizes=sizes,
        )
    for epoch in range(start, epochs):
        dataset.epoch = epoch
        model.train()
        sums = {"loss": 0.0, "bbox_regression": 0.0, "classification": 0.0}
        batches = 0
        t0 = time.perf_counter()
        for tensors, targets, _ in loader:
            lr = _lr_at(step, total_steps, warmup, float(section["lr0"]), float(section["lrf"]))
            for group in optimizer.param_groups:
                group["lr"] = lr
            tensors = [t.to(device) for t in tensors]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(tensors, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            batches += 1
            sums["loss"] += float(loss.detach())
            for k, v in losses.items():
                sums[k] += float(v.detach())
            if batches == 1 or batches % PROGRESS_EVERY == 0 or batches == len(loader):
                elapsed = time.perf_counter() - t0
                logger.info("epoch %d step %d/%d loss %.4f lr %.5f %.1f img/s", epoch + 1, batches, len(loader),
                            sums["loss"] / batches, lr, batches * len(tensors) / max(elapsed, 1e-9))
        train_seconds = time.perf_counter() - t0
        logger.info("epoch %d: validating on %d images", epoch + 1, len(val_images))

        t1 = time.perf_counter()
        predictions = detector_predictions(model, val_images, device, section, confidence_floor=0.01)
        report = evaluate_detections(val_truth, predictions, confidence=confidence, difficulty=difficulty)
        key = (report["mAP50"], report["mAP50_95"])
        improved = best is None or key > best
        payload = {
            "epoch": epoch + 1, "lr": lr, "train_seconds": round(train_seconds, 1),
            "val_seconds": round(time.perf_counter() - t1, 1),
            **{f"train_{k}": round(v / max(batches, 1), 5) for k, v in sums.items()},
            "val_mAP50": report["mAP50"], "val_mAP50_95": report["mAP50_95"],
            "val_precision": report["precision"], "val_recall": report["recall"],
            "val_small_recall": (report["fallback_trigger"]["small_plate_recall"] or {}).get("value"),
            "improved": improved,
        }
        run.epoch(payload)
        logger.info("epoch %s", json.dumps(payload))
        if improved:
            best, best_epoch = key, epoch + 1
        stop = not improved and epoch + 1 - best_epoch >= patience
        _save_epoch(run, model, optimizer, generator, epoch=epoch + 1, section=section, key=key,
                    improved=improved, best=best, best_epoch=best_epoch, stop=stop, step=step,
                    steps_per_epoch=len(loader), epochs=epochs, sizes=sizes)
        if stop:
            logger.info("early stop: no improvement for %d epochs", patience)
            break

    return {"best_epoch": best_epoch, "best_selection_key": list(best or ()), "epochs_run": epoch + 1}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--i-have-approval",
        metavar="NAME",
        default="",
        help="the person who authorised this run; without it the script refuses",
    )
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs: plumbing check, not a baseline")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="data-loader workers (default: 0 on Windows, where workers re-pickle the dataset; 4 elsewhere)",
    )
    parser.add_argument(
        "--resume", type=Path, metavar="LAST_PT", default=None,
        help="continue the run whose last.pt this is, in its own directory, from the next epoch",
    )
    parser.add_argument(
        "--sync-command", metavar="CMD", default=None,
        help="shell command run after every epoch's checkpoints are written and once at the end, "
             "with RUN_DIR and RUN_EPOCH set (e.g. scripts/kaggle_persist_run.py push ...)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # a pipe (Kaggle, tee) sees every line at once

    if not args.config.is_file():
        print(f"error: no config at {args.config}", file=sys.stderr)
        return EXIT_UNUSABLE
    config = json.loads(args.config.read_text(encoding="utf-8"))

    print(f"train_baseline: {args.component} preflight starting", flush=True)
    blocking, notes = preflight(args.component, config)
    section = config.get(args.component, {})
    experiment = config.get("experiment", {})
    run_name = (
        f"{experiment.get('name', 'baseline')}_"
        f"{datetime.now().strftime('%Y%m%d')}_{experiment.get('seed')}"
        + ("_smoke" if args.smoke else "")
    )
    run_dir = REPO_ROOT / experiment.get("output_root", "runs") / args.component / run_name

    plan = None
    if args.resume is not None and not blocking:  # a blocked preflight may mean torch is missing
        plan, problems = plan_resume(args.resume, args.component, config, smoke=args.smoke)
        if plan is None:
            blocking += [f"resume refused: {p}" for p in problems]
        else:
            run_dir = plan.run_dir
            notes.append(f"resume checkpoint verified: epoch {plan.checkpoint['epoch']} done, "
                         f"best epoch {plan.checkpoint['resume']['best_epoch']}; continues at epoch "
                         f"{plan.checkpoint['epoch'] + 1}")
            if plan.discarded_metrics:
                notes.append(f"metrics.jsonl: {len(plan.discarded_metrics)} line(s) past the checkpoint "
                             "will be moved to run.json and the epoch(s) retrained")
            if plan.repair_best:
                notes.append("best.pt is one write behind last.pt; it will be rebuilt from last.pt")
    elif args.resume is None and (run_dir / "last.pt").exists():
        blocking.append(f"{_display(run_dir)} already holds a run; pass --resume {_display(run_dir / 'last.pt')} "
                        "to continue it, or move it away to start afresh")

    print(f"component        : {args.component}  ({section.get('family')})")
    print(f"dataset          : {experiment.get('dataset_version')}  seed {experiment.get('seed')}")
    print(f"train / val      : {section.get('train_split')} / {section.get('val_split')}")
    print(f"{'resumes in' if plan else 'would write to':<17}: {_display(run_dir)}/")
    print()
    for note in notes:
        print(f"  ok      {note}")
    for problem in blocking:
        print(f"  BLOCKED {problem}")

    if blocking:
        print(f"\npreflight: {len(blocking)} blocking finding(s); not starting")
        return EXIT_BLOCKED

    if not args.i_have_approval.strip():
        print(
            "\npreflight passed. NOT training: no approval given.\n"
            "Re-run with --i-have-approval \"<name>\" to start."
        )
        return EXIT_OK

    print("stage: importing torch", flush=True)
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    workers = args.workers if args.workers is not None else (0 if os.name == "nt" else 4)
    seed_everything(int(experiment["seed"]), bool(experiment.get("deterministic", True)))

    session = {
        "approved_by": args.i_have_approval.strip(),
        "device": str(device),
        "workers": workers,
        "environment": environment(),
    }
    if plan is None:
        record = {
            "experiment_id": f"{args.component}/{run_name}",
            "component": args.component,
            "kind": "smoke" if args.smoke else "baseline",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **source_revision(),
            "dataset": dataset_identity(),
            "training_bundle_mode": training_bundle_mode(),
            "seed": experiment["seed"],
            **session,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "config": config,
        }
    else:
        apply_resume_plan(plan)
        record = plan.record
        record.setdefault("component", args.component)
        record.setdefault("resumes", []).append({
            "resumed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "from_checkpoint_epoch": plan.checkpoint["epoch"],
            **source_revision(),
            **session,
            "discarded_metrics": [json.loads(line) for line in plan.discarded_metrics if _epoch_of(line)],
            "rebuilt_best_pt": plan.repair_best,
        })
    run = RunWriter(run_dir, record, sync_command=args.sync_command)
    verb = f"resuming at epoch {plan.checkpoint['epoch'] + 1}" if plan else "starting"
    print(f"\napproved by {session['approved_by']} -- {verb} {args.component} "
          f"{'SMOKE ' if args.smoke else ''}training on {device}")
    trainer = train_detector if args.component == "detector" else train_recognizer
    started = time.perf_counter()
    try:
        result = trainer(config, run, device, smoke=args.smoke, workers=workers,
                         resume=plan.checkpoint if plan else None)
    except ResumeMismatch as exc:
        logger.error("resume refused: %s", exc)
        run.record["resumes"].pop()  # it did not happen
        run.save()
        run.close()
        return EXIT_BLOCKED
    wall = round(time.perf_counter() - started, 1)
    run.record.update({
        "result": result,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": wall if plan is None else run.record.get("wall_seconds"),
    })
    if plan is not None:
        run.record["resumes"][-1]["wall_seconds"] = wall
    run.save()
    run.finish_sync()
    run.sync("final")
    if not run.finish_sync():
        logger.warning("the final state was NOT synced; %s is the only copy", run_dir)
    run.close()
    print(f"\ndone: {json.dumps(result)}\nrun directory: {run_dir}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
