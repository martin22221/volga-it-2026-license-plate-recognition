"""Training-side scaffolding for the baseline.

Everything here that can work without a deep-learning framework, does: the
alphabet and CTC coding, the Dataset V1 split reader, the label exporters and
the metrics are plain Python and NumPy, so they are testable today, before any
framework is installed and before any model is trained.

The parts that genuinely need a framework live in ``scripts/train_*.py`` and
import it lazily, so importing this package never pulls in torch.

No module here trains anything.
"""

from __future__ import annotations

__all__ = ["alphabet", "data", "metrics"]
