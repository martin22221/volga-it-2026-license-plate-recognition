"""Reading Dataset V1 for training, and exporting labels a trainer can eat.

One rule governs this module: **the holdout is not reachable from here.**
:func:`load_split` refuses to hand back the real holdout unless the caller asks
for it by name *and* says why, which makes an accidental "train on everything"
impossible to write by mistake rather than merely discouraged in a comment.

Nothing here imports a deep-learning framework. It produces plain records and
plain label files; the trainer turns those into tensors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.dataset_meta import CSV_DELIMITER
from src.training.alphabet import is_trainable

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "dataset"
SPLITS = DATASET / "splits"

#: Plate classes the models predict, in a fixed index order. The order is part
#: of a trained checkpoint; changing it silently relabels every prediction.
CLASSES: tuple[str, ...] = ("type1", "type1a", "type1b", "other")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

#: The split that must never be trained on, tuned on, or looked at until the
#: end. Named here so the guard below and the docs cannot drift apart.
REAL_HOLDOUT: str = "holdout"


class SplitAccessError(Exception):
    """An attempt to read a split that the caller is not entitled to."""


@dataclass(frozen=True)
class Sample:
    """One annotated plate, with everything a trainer needs and nothing else."""

    image: str          # path relative to dataset/
    plate_num: str
    plate_type: str
    # Pixels, and floating point on purpose: the generator writes sub-pixel
    # corners (the plate is rendered through a homography, so its corners do
    # not land on integers), while hand annotation writes whole pixels. The
    # schema has always been float; rounding here would quietly degrade every
    # synthetic label.
    bbox: tuple[float, float, float, float]        # x, y, w, h
    quad: tuple[tuple[float, float], ...]          # 4 corners, clockwise from plate top-left
    conditions: tuple[str, ...]
    is_synthetic: bool
    source: str
    split: str

    @property
    def path(self) -> Path:
        return DATASET / self.image

    @property
    def class_index(self) -> int:
        return CLASS_INDEX[self.plate_type]

    @property
    def ocr_trainable(self) -> bool:
        """Usable as an OCR target: a readable string, and a target class."""
        return self.plate_type != "other" and is_trainable(self.plate_num)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=CSV_DELIMITER))


def _split_map() -> dict[str, tuple[str, str]]:
    """image -> (population, split), where population is 'synthetic' or 'real'."""
    out: dict[str, tuple[str, str]] = {}
    for row in _read_csv(SPLITS / "synthetic_splits.csv"):
        out[row["image"]] = ("synthetic", row["split"])
    for row in _read_csv(SPLITS / "real_splits.csv"):
        out[row["image"]] = ("real", row["split"])
    return out


def load_samples() -> list[Sample]:
    """Every annotated plate in Dataset V1, with its split attached."""
    splits = _split_map()
    samples: list[Sample] = []
    for row in _read_csv(DATASET / "meta.csv"):
        population, split = splits.get(row["image"], ("?", "?"))
        samples.append(
            Sample(
                image=row["image"],
                plate_num=row["plate_num"],
                plate_type=row["plate_type"],
                bbox=(float(row["bbox_x"]), float(row["bbox_y"]),
                      float(row["bbox_w"]), float(row["bbox_h"])),
                quad=tuple(
                    (float(row[f"quad_x{i}"]), float(row[f"quad_y{i}"]))
                    for i in (1, 2, 3, 4)
                ),
                conditions=tuple(t for t in (row["conditions"] or "").split("|") if t),
                is_synthetic=row["is_synthetic"].lower() == "true",
                source=row["source"],
                split=f"{population}:{split}",
            )
        )
    return samples


def load_split(
    name: str,
    *,
    samples: Sequence[Sample] | None = None,
    unlock_holdout: str = "",
) -> list[Sample]:
    """Samples of one split, e.g. ``"synthetic:train"`` or ``"real:holdout"``.

    Asking for the real holdout raises unless ``unlock_holdout`` carries a
    written reason. This is not ceremony: the holdout is 2 images, it is the
    only measurement we will have that the model has never influenced, and it
    is spent the first time it is used for anything but a final report.
    """
    if name == f"real:{REAL_HOLDOUT}" and not unlock_holdout.strip():
        raise SplitAccessError(
            "real:holdout is the final, once-only evaluation set. Pass "
            "unlock_holdout='<why>' to read it, and never from a training or "
            "hyperparameter-selection path."
        )
    pool = samples if samples is not None else load_samples()
    return [s for s in pool if s.split == name]


def training_pool(samples: Sequence[Sample] | None = None) -> list[Sample]:
    """Everything the baseline may fit on: synthetic train only.

    Baseline v1 trains on **no real photograph at all**. 13 real images against
    12,000 synthetic would move no weights, and spending them on training would
    cost the only real-world measurement the project has. They are all
    evaluation in v1; ``real:train`` is reserved for a later fine-tune.
    """
    pool = samples if samples is not None else load_samples()
    return [s for s in pool if s.split == "synthetic:train"]


# ---------------------------------------------------------------------------
# label export
# ---------------------------------------------------------------------------


def yolo_line(sample: Sample, image_width: int, image_height: int, class_index: int) -> str:
    """One YOLO label line: ``cls cx cy w h``, normalised to [0, 1].

    Coordinates are clamped to the image, because a box that runs off the edge
    is legal in our annotations (the guide says clip it) but is not legal here.
    """
    x, y, w, h = sample.bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_width, x + w), min(image_height, y + h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{sample.image}: degenerate box after clipping")
    cx = (x1 + x2) / 2 / image_width
    cy = (y1 + y2) / 2 / image_height
    nw = (x2 - x1) / image_width
    nh = (y2 - y1) / image_height
    return f"{class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def summarise(samples: Iterable[Sample]) -> dict:
    """Counts a person actually checks before starting a run."""
    from collections import Counter

    samples = list(samples)
    images = {s.image for s in samples}
    conditions: Counter = Counter()
    for sample in samples:
        conditions.update(sample.conditions)
    return {
        "rows": len(samples),
        "images": len(images),
        "by_class": dict(Counter(s.plate_type for s in samples)),
        "by_split": dict(Counter(s.split for s in samples)),
        "ocr_trainable": sum(1 for s in samples if s.ocr_trainable),
        "unique_plates": len({s.plate_num for s in samples if s.plate_num}),
        "conditions": dict(conditions),
    }


__all__ = [
    "CLASSES",
    "CLASS_INDEX",
    "DATASET",
    "REAL_HOLDOUT",
    "Sample",
    "SplitAccessError",
    "load_samples",
    "load_split",
    "summarise",
    "training_pool",
    "yolo_line",
]
