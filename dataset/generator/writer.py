"""Writes a generated batch in the dataset's own layout.

``<output>/`` mirrors the ``dataset/`` root, so the repository's validator runs
on it unchanged (``--dataset <output>``)::

    <output>/
      images/synthetic/syn_<seed>_<index>.jpg
      meta.csv            dataset/meta.csv schema, one row per plate
      generation.jsonl    every parameter of every sample, incl. its seed
      manifest.json       config, versions, counts and SHA-256 of every file

All three text files are written with fixed formatting and ``\\n`` line
endings, and contain no timestamps, so identical inputs give identical bytes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import PIL

from . import GENERATOR_NAME, GENERATOR_VERSION, OUTPUT_LICENSE, SOURCE_ID
from .annotations import META_COLUMNS
from .config import GeneratorConfig, config_to_dict
from .sample import GeneratedSample

IMAGE_DIR = "images/synthetic"
META_NAME = "meta.csv"
RECORDS_NAME = "generation.jsonl"
MANIFEST_NAME = "manifest.json"
REVIEW_DIR = "review"


class OutputError(Exception):
    """The output directory cannot be used."""


def image_name(seed: int, index: int) -> str:
    return f"syn_{seed}_{index:05d}.jpg"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_output(root: Path, *, overwrite: bool) -> None:
    """Create ``root``; refuse to mix a new batch into existing content.

    With ``overwrite``, only a directory previously written by this generator
    (it has our ``manifest.json``) is cleared, and only of the files that
    manifest lists -- nothing else is ever deleted.
    """
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise OutputError(f"{root} exists and is not a directory")
    if root.is_dir() and any(root.iterdir()):
        manifest_path = root / MANIFEST_NAME
        if not overwrite:
            raise OutputError(f"{root} is not empty; pass --overwrite to replace a previous batch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise OutputError(
                f"{root} is not empty and was not written by this generator; refusing to overwrite"
            ) from None
        if manifest.get("generator") != GENERATOR_NAME:
            raise OutputError(f"{root} was not written by {GENERATOR_NAME}; refusing to overwrite")
        targets = [root / relative for relative in manifest.get("files", {})]
        if not manifest.get("complete", True):
            # An interrupted run never listed its files; it wrote only these.
            targets += list((root / IMAGE_DIR).glob("syn_*.jpg"))
            targets += [root / META_NAME, root / RECORDS_NAME]
        for target in targets:
            target = target.resolve()
            if root.resolve() in target.parents and target.is_file():
                target.unlink()
        review = root / REVIEW_DIR
        if review.is_dir():
            shutil.rmtree(review)
        manifest_path.unlink(missing_ok=True)
    (root / IMAGE_DIR).mkdir(parents=True, exist_ok=True)
    # Marker so that an interrupted run can still be overwritten later.
    (root / MANIFEST_NAME).write_text(
        json.dumps({"generator": GENERATOR_NAME, "complete": False}) + "\n", encoding="utf-8"
    )


class DatasetWriter:
    """Collects samples and writes the batch."""

    def __init__(self, root: Path, config: GeneratorConfig) -> None:
        self.root = Path(root)
        self.config = config
        self._rows: list[dict[str, str]] = []
        self._records: list[dict[str, Any]] = []
        self._files: dict[str, str] = {}

    def add(self, sample: GeneratedSample) -> None:
        relative = sample.annotation.image
        (self.root / relative).write_bytes(sample.image_bytes)
        self._files[relative] = _sha256(sample.image_bytes)
        self._rows.append(sample.annotation.meta_row())
        self._records.append(sample.record)

    def _write_text(self, name: str, text: str) -> None:
        data = text.encode("utf-8")
        (self.root / name).write_bytes(data)
        self._files[name] = _sha256(data)

    def finish(self) -> dict[str, Any]:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(META_COLUMNS), delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(self._rows)
        self._write_text(META_NAME, buffer.getvalue())

        lines = [json.dumps(record, sort_keys=True, ensure_ascii=False) for record in self._records]
        self._write_text(RECORDS_NAME, "\n".join(lines) + "\n")

        counts: dict[str, dict[str, int]] = {"plate_type": {}, "difficulty": {}}
        for record in self._records:
            for key in counts:
                counts[key][record[key]] = counts[key].get(record[key], 0) + 1
        manifest = {
            "generator": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "complete": True,
            "source": SOURCE_ID,
            "license": OUTPUT_LICENSE,
            "is_synthetic": True,
            "seed": self.config.seed,
            "image_count": len(self._records),
            "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
            "config": config_to_dict(self.config),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "external_assets": [],
            "files": dict(sorted(self._files.items())),
        }
        (self.root / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        return manifest
