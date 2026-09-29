#!/usr/bin/env python
"""Alias of scripts/kaggle_persist_run.py -- same commands, same arguments, same exit codes.

Kept so the Kaggle checkpoint helper can be found by name. The implementation
and its documentation live in kaggle_persist_run.py::

    python scripts/kaggle_checkpoint.py init    --dataset <user>/<slug>
    python scripts/kaggle_checkpoint.py check   --dataset <user>/<slug>
    python scripts/kaggle_checkpoint.py push    --dataset <user>/<slug>
    python scripts/kaggle_checkpoint.py restore --dataset <user>/<slug>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kaggle_persist_run import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
