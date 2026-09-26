#!/usr/bin/env python
"""Pack everything a GPU machine needs to train Baseline V1 -- and nothing real.

Contents of the zip:

* every git-tracked file at ``HEAD`` (code, configs, docs, meta.csv, the split
  manifests, the Dataset V1 freeze) -- the working tree must be clean, so the
  bundle is exactly a commit;
* the 12,000 synthetic images under ``dataset/images/synthetic/``;
* the sealed batch's ``generation.jsonl`` (character heights and occlusion,
  used only for evaluation breakdowns);
* ``TRAINING_BUNDLE.json``: the commit, the Dataset V1 identity, and the file
  count, which also switches the trainer's preflight to bundle verification.

**No real photograph is packed.** Baseline V1 trains and selects on synthetic
data only, so a training machine has no use for them, and a file that is not
there cannot leak. Real-image evaluation runs where the real images live, on
the exported ONNX models.

Usage::

    python scripts/pack_training_bundle.py --out ../baseline_v1_training_bundle.zip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION = Path("data/synthetic_production/v2_3_seed2026091401_n12000/generation.jsonl")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if _git("status", "--porcelain").strip():
        print("refusing: the working tree is not clean; commit first so the bundle is a commit", file=sys.stderr)
        return 2
    tracked = [line for line in _git("ls-files").splitlines() if line]
    if any(t.startswith("dataset/images/real/") and not t.endswith(".gitkeep") for t in tracked):
        print("refusing: a real image is tracked by git", file=sys.stderr)
        return 2
    synthetic = sorted((REPO_ROOT / "dataset" / "images" / "synthetic").glob("*.jpg"))
    if len(synthetic) != 12_000:
        print(f"refusing: expected 12,000 synthetic images, found {len(synthetic)}", file=sys.stderr)
        return 2

    freeze = json.loads((REPO_ROOT / "dataset" / "splits" / "dataset_v1.json").read_text(encoding="utf-8"))
    marker = {
        "bundle": "baseline_v1_training",
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "dataset_version": freeze["dataset_version"],
        "combined_image_digest": freeze["combined_image_digest"],
        "meta_sha256": freeze["meta_sha256"],
        "synthetic_images": len(synthetic),
        "real_images": 0,
        "note": "No real photograph is included, by design. See scripts/pack_training_bundle.py.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name in tracked:
            bundle.write(REPO_ROOT / name, name)
        for path in synthetic:
            bundle.write(path, path.relative_to(REPO_ROOT).as_posix())
        if (REPO_ROOT / GENERATION).is_file():
            bundle.write(REPO_ROOT / GENERATION, GENERATION.as_posix())
        bundle.writestr("TRAINING_BUNDLE.json", json.dumps(marker, indent=1) + "\n")
    print(json.dumps(marker, indent=1))
    print(f"written: {args.out}  ({args.out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
