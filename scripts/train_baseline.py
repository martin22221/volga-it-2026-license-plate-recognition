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
  (one line per epoch), ``train.log``, ``best.pt`` and ``last.pt``.

``--smoke`` trains on a small deterministic subset for a couple of epochs and
writes to a ``_smoke`` directory. It proves the loop runs end to end; it is
**not** a baseline and its numbers mean nothing.

Usage::

    python scripts/train_baseline.py --component detector                       # preflight only
    python scripts/train_baseline.py --component recognizer --i-have-approval "<name>"
    python scripts/train_baseline.py --component detector --i-have-approval "<name>" --smoke

Exit codes::

    0  preflight passed (and, if approved, the run finished)
    1  the configuration or dataset could not be read
    2  a preflight check failed, or the run is not authorised
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
import subprocess
import sys
import time
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
    def __init__(self, run_dir: Path, record: dict) -> None:
        self.dir = run_dir
        self.record = record
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(handler)
        self.save()

    def save(self) -> None:
        (self.dir / "run.json").write_text(json.dumps(self.record, indent=1, default=str) + "\n", encoding="utf-8")

    def epoch(self, payload: dict) -> None:
        with (self.dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def _checkpoint(path: Path, model, meta: dict) -> None:
    import torch

    torch.save({"model": model.state_dict(), **meta}, path)


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


def train_recognizer(config: dict, run: RunWriter, device, *, smoke: bool, workers: int) -> dict:
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

    best_key, best_epoch, step = None, -1, 0
    for epoch in range(epochs):
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
        meta = {"epoch": epoch + 1, "config": section, "selection": list(key)}
        _checkpoint(run.dir / "last.pt", model, meta)
        if improved:
            best_key, best_epoch = key, epoch + 1
            _checkpoint(run.dir / "best.pt", model, meta)
        elif epoch + 1 - best_epoch >= patience:
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


def train_detector(config: dict, run: RunWriter, device, *, smoke: bool, workers: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from src.training.evaluation import evaluate_detections, load_difficulty
    from src.training.models import build_detector, count_parameters
    from src.training.torch_data import DetectorDataset, detector_collate, group_by_image

    section = config["detector"]
    seed = int(config["experiment"]["seed"])
    confidence = float(config["evaluation"]["detector_confidence"])
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

    dataset = DetectorDataset(train_images, int(section["input_size"]), augment=section["augmentation"], seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=int(section["batch_size"]), shuffle=True, generator=generator,
        num_workers=workers, worker_init_fn=_worker_init if workers else None,
        collate_fn=detector_collate, drop_last=True,
    )
    model = build_detector(section).to(device)
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

    best, best_epoch, step = None, -1, 0
    for epoch in range(epochs):
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
        train_seconds = time.perf_counter() - t0

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
        meta = {"epoch": epoch + 1, "config": section, "selection": list(key)}
        _checkpoint(run.dir / "last.pt", model, meta)
        if improved:
            best, best_epoch = key, epoch + 1
            _checkpoint(run.dir / "best.pt", model, meta)
        elif epoch + 1 - best_epoch >= patience:
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
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"error: no config at {args.config}", file=sys.stderr)
        return EXIT_UNUSABLE
    config = json.loads(args.config.read_text(encoding="utf-8"))

    blocking, notes = preflight(args.component, config)
    section = config.get(args.component, {})
    experiment = config.get("experiment", {})
    run_name = (
        f"{experiment.get('name', 'baseline')}_"
        f"{datetime.now().strftime('%Y%m%d')}_{experiment.get('seed')}"
        + ("_smoke" if args.smoke else "")
    )
    run_dir = REPO_ROOT / experiment.get("output_root", "runs") / args.component / run_name

    print(f"component        : {args.component}  ({section.get('family')})")
    print(f"dataset          : {experiment.get('dataset_version')}  seed {experiment.get('seed')}")
    print(f"train / val      : {section.get('train_split')} / {section.get('val_split')}")
    print(f"would write to   : {run_dir.relative_to(REPO_ROOT).as_posix()}/")
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

    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    workers = args.workers if args.workers is not None else (0 if os.name == "nt" else 4)
    seed_everything(int(experiment["seed"]), bool(experiment.get("deterministic", True)))

    record = {
        "experiment_id": f"{args.component}/{run_name}",
        "kind": "smoke" if args.smoke else "baseline",
        "approved_by": args.i_have_approval.strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **source_revision(),
        "dataset": dataset_identity(),
        "training_bundle_mode": training_bundle_mode(),
        "seed": experiment["seed"],
        "device": str(device),
        "workers": workers,
        "environment": environment(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config": config,
    }
    run = RunWriter(run_dir, record)
    print(f"\napproved by {record['approved_by']} -- starting {args.component} "
          f"{'SMOKE ' if args.smoke else ''}training on {device}")
    trainer = train_detector if args.component == "detector" else train_recognizer
    started = time.perf_counter()
    result = trainer(config, run, device, smoke=args.smoke, workers=workers)
    run.record.update({
        "result": result,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - started, 1),
    })
    run.save()
    print(f"\ndone: {json.dumps(result)}\nrun directory: {run_dir}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
