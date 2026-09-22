#!/usr/bin/env python
"""Baseline training entry point -- inert until a person authorises a run.

This script is the boundary between "ready to train" and "training". It checks
everything that must be true before a run is worth starting, prints exactly what
would happen, and then **stops** unless ``--i-have-approval`` is passed with the
approving person's name.

That guard is the point. A training run consumes hours of borrowed GPU, writes
weights, and -- if it is pointed at the wrong split -- destroys the value of a
holdout that took an acquisition, a human review and a promotion to build. The
default behaviour of this file is therefore to refuse.

Preflight checks, all of which must pass:

* the Dataset V1 freeze exists and still verifies;
* the split audit is clean;
* the requested component's training split contains no real photograph;
* the real holdout is not reachable from the training configuration;
* a framework is actually installed (it is not, today).

Usage::

    python scripts/train_baseline.py --component detector            # preflight only
    python scripts/train_baseline.py --component detector --i-have-approval "<name>"

Exit codes::

    0  preflight passed (and the run would start, if approved and possible)
    1  the configuration or dataset could not be read
    2  a preflight check failed, or the run is not authorised
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.data import load_samples, load_split  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
FREEZE = REPO_ROOT / "dataset" / "splits" / "dataset_v1.json"
COMPONENTS = ("detector", "recognizer")

#: Frameworks a real run needs. Absent today, and deliberately not installed:
#: the current task authorises readiness, not training.
REQUIRED_MODULES: dict[str, tuple[str, ...]] = {
    # torchvision, not a third-party detector repository: the selected detector
    # is torchvision's own SSDlite (BSD-3), so the detector needs nothing the
    # recogniser does not already need. See docs/baseline_v1.md section 3.
    "detector": ("torch", "torchvision"),
    "recognizer": ("torch",),
}

#: Licences of every third-party component a run would pull in, so the check is
#: mechanical rather than remembered. AGPL is refused: this project has not
#: chosen a source licence (docs/baseline_v1.md section 8a), and an AGPL
#: dependency would choose one for it at submission time.
COMPONENT_LICENCES: dict[str, str] = {
    "torch": "BSD-3-Clause",
    "torchvision": "BSD-3-Clause",
}
REFUSED_LICENCE_TOKENS: tuple[str, ...] = ("agpl", "gpl-3", "gplv3")

EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKED = 0, 1, 2


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - odd import states
        return False


def preflight(component: str, config: dict) -> tuple[list[str], list[str]]:
    """Everything that must hold before a run starts. Returns (blocking, notes)."""
    blocking: list[str] = []
    notes: list[str] = []

    if not FREEZE.is_file():
        blocking.append(f"no Dataset V1 freeze at {FREEZE.relative_to(REPO_ROOT)}")
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

    # The licence of what a run would pull in, checked rather than remembered.
    refused = [
        f"{name} ({licence})"
        for name, licence in COMPONENT_LICENCES.items()
        if name in REQUIRED_MODULES[component]
        and any(token in licence.lower() for token in REFUSED_LICENCE_TOKENS)
    ]
    if refused:
        blocking.append(
            f"copyleft dependency refused: {', '.join(refused)}. This project has not chosen a "
            "source licence, and such a dependency would choose one for it at submission time"
        )
    else:
        declared = ", ".join(
            f"{n}={COMPONENT_LICENCES.get(n, '?')}" for n in REQUIRED_MODULES[component]
        )
        notes.append(f"dependency licences: {declared}")

    missing = [m for m in REQUIRED_MODULES[component] if not _installed(m)]
    if missing:
        blocking.append(
            f"missing framework(s): {', '.join(missing)}. Training is not authorised in the "
            "current task, so they are deliberately not installed"
        )
    return blocking, notes


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
    )

    print(f"component        : {args.component}  ({section.get('family')})")
    print(f"dataset          : {experiment.get('dataset_version')}  seed {experiment.get('seed')}")
    print(f"train / val      : {section.get('train_split')} / {section.get('val_split')}")
    print(f"would write to   : {experiment.get('output_root')}/{args.component}/{run_name}/")
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

    print(f"\napproved by {args.i_have_approval.strip()} -- starting {args.component} training")
    raise SystemExit(
        "the training loop is not implemented yet: this task built the readiness, "
        "not the run. Implement it here once a framework is installed and a GPU is available."
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
