"""Tensors for the two trainers, built from :mod:`src.training.data` samples.

Imports torch; only the training scripts import this module.

**Which samples reach here is not decided here.** Both datasets take a list of
:class:`~src.training.data.Sample` and train on exactly that list. The list is
built by ``scripts/train_baseline.py`` from ``training_pool()`` and the
configured validation split, and that script refuses to start if either
contains a real photograph. The real holdout cannot be loaded without an
explicit written reason (:func:`src.training.data.load_split`), and nothing on
the training path supplies one.

Augmentation follows ``configs/baseline_v1.json``. Hue is never shifted: the
one thing separating ``type1b`` from ``type1`` is the yellow field.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.recognition import (
    BOX_MARGIN,
    box_crop,
    expand_box,
    is_two_line,
    quad_to_crop,
    rectify,
    to_tensor,
)
from src.training.alphabet import MAX_LENGTH, encode
from src.training.data import Sample


# ---------------------------------------------------------------------------
# recogniser: a cached region around every plate
# ---------------------------------------------------------------------------

from src.training.regions import Region, cut_region, cut_regions  # noqa: E402,F401  (re-exported)


def _photometric(image: Image.Image, rng: random.Random) -> Image.Image:
    """Brightness, contrast, saturation and occasional blur. Never hue."""
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    return image


def jitter_box(box, rng: random.Random, amount: float):
    """Move each edge by up to ``amount`` of the box size: a detector's error."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (
        x1 + rng.gauss(0, amount) * w,
        y1 + rng.gauss(0, amount) * h,
        x2 + rng.gauss(0, amount) * w,
        y2 + rng.gauss(0, amount) * h,
    )


def jitter_quad(quad, rng: random.Random, amount: float):
    """Move each corner by ``amount`` of the plate's height: a corner head's error."""
    tl, tr, br, bl = quad
    scale = (math.dist(tl, bl) + math.dist(tr, br)) / 2
    return tuple((x + rng.gauss(0, amount) * scale, y + rng.gauss(0, amount) * scale) for x, y in quad)


class RecognizerDataset(Dataset):
    """Two views of every plate: a box crop for the corner head, a strip for reading.

    ``train=False`` turns every random perturbation off, which is what the
    loss-on-validation and the tests use. Model *selection* does not use this
    class: it runs the deployed two-pass read (:func:`src.recognition.read_plates`)
    on the validation regions, so the checkpoint is chosen on exactly the
    procedure that will ship.
    """

    def __init__(self, samples: Sequence[Sample], regions: Sequence[Region], *, train: bool, seed: int) -> None:
        if len(samples) != len(regions):
            raise ValueError("one region per sample")
        self.samples = list(samples)
        self.regions = list(regions)
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample, region = self.samples[index], self.regions[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}") if self.train else None
        image = region.image()

        box = jitter_box(region.box, rng, 0.04) if rng else region.box
        margin = BOX_MARGIN + (rng.uniform(-0.03, 0.06) if rng else 0.0)
        crop = expand_box(box, margin, image.width, image.height)
        corner_view = box_crop(image, crop)
        corners = quad_to_crop(region.quad, crop)

        quad = jitter_quad(region.quad, rng, 0.04) if rng else region.quad
        strip = rectify(image, quad, is_two_line(quad))
        if rng:
            corner_view = _photometric(corner_view, rng)
            strip = _photometric(strip, rng)

        target = [0] * MAX_LENGTH
        length = 0
        if sample.ocr_trainable:
            encoded = encode(sample.plate_num)
            target[: len(encoded)] = encoded
            length = len(encoded)
        return (
            torch.from_numpy(to_tensor([corner_view])[0]),
            torch.from_numpy(to_tensor([strip])[0]),
            torch.tensor(corners, dtype=torch.float32),
            torch.tensor(sample.class_index, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def _affine_matrix(size: int, degrees: float, scale: float, tx: float, ty: float) -> np.ndarray:
    """Forward 3x3 affine about the image centre: rotate, scale, translate."""
    c = size / 2
    a = math.radians(degrees)
    cos, sin = math.cos(a) * scale, math.sin(a) * scale
    return np.array(
        [[cos, -sin, c - cos * c + sin * c + tx], [sin, cos, c - sin * c - cos * c + ty], [0, 0, 1]],
        dtype=np.float64,
    )


class DetectorDataset(Dataset):
    """Full images stretched to ``size`` x ``size``, boxes from the plate corners.

    Stretching rather than letterboxing is torchvision SSD's own convention
    (``GeneralizedRCNNTransform`` with ``fixed_size``), and inference does the
    same (:class:`src.onnx_backend.OnnxDetector`).

    Boxes are taken from the four corners, transformed with the image, so a
    rotated plate keeps a tight box rather than the loose box of a rotated box.
    """

    def __init__(self, images: Sequence[tuple[str, list[Sample]]], size: int, *, augment: dict | None, seed: int) -> None:
        if augment and augment.get("fliplr", 0):
            raise ValueError("fliplr must be 0: a mirrored plate is not a plate")
        self.images = list(images)
        self.size = size
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.images)

    def load(self, index: int) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
        path, rows = self.images[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        original = image.size
        sx, sy = self.size / image.width, self.size / image.height
        image = image.resize((self.size, self.size), Image.BILINEAR)
        quads = np.array([[(x * sx, y * sy) for x, y in r.quad] for r in rows], dtype=np.float64).reshape(-1, 4, 2)
        return image, quads, original

    def __getitem__(self, index: int):
        image, quads, original = self.load(index)
        augment = self.augment
        if augment:
            rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
            degrees = rng.uniform(-augment["degrees"], augment["degrees"])
            scale = rng.uniform(1 - augment["scale"], 1 + augment["scale"])
            tx = rng.uniform(-augment["translate"], augment["translate"]) * self.size
            ty = rng.uniform(-augment["translate"], augment["translate"]) * self.size
            forward = _affine_matrix(self.size, degrees, scale, tx, ty)
            inverse = np.linalg.inv(forward)
            image = image.transform(
                image.size, Image.AFFINE, tuple(inverse[:2].reshape(-1)), Image.BILINEAR,
                fillcolor=(114, 114, 114),
            )
            if len(quads):
                homogeneous = np.concatenate([quads, np.ones((*quads.shape[:2], 1))], axis=2)
                quads = (homogeneous @ forward.T)[..., :2]
            image = ImageEnhance.Color(image).enhance(rng.uniform(1 - augment["hsv_s"], 1 + augment["hsv_s"]))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(1 - augment["hsv_v"], 1 + augment["hsv_v"]))

        boxes = []
        for quad in quads:
            x1, y1 = quad[:, 0].min(), quad[:, 1].min()
            x2, y2 = quad[:, 0].max(), quad[:, 1].max()
            cx1, cy1 = max(0.0, x1), max(0.0, y1)
            cx2, cy2 = min(float(self.size), x2), min(float(self.size), y2)
            full = (x2 - x1) * (y2 - y1)
            kept = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
            # A plate pushed mostly out of frame is dropped rather than kept as a
            # sliver the model would be punished for missing.
            if full > 0 and kept / full >= 0.4 and cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
                boxes.append([cx1, cy1, cx2, cy2])

        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }
        return tensor, target, torch.tensor(original)


def group_by_image(samples: Sequence[Sample]) -> list[tuple[str, list[Sample]]]:
    """One entry per image, every plate on it, in a stable order."""
    by_image: dict[str, list[Sample]] = {}
    for sample in samples:
        by_image.setdefault(sample.image, []).append(sample)
    return [(str(rows[0].path), rows) for _, rows in sorted(by_image.items())]


def detector_collate(batch):
    images, targets, sizes = zip(*batch)
    return list(images), list(targets), torch.stack(sizes)


__all__ = [
    "DetectorDataset",
    "RecognizerDataset",
    "Region",
    "cut_region",
    "cut_regions",
    "detector_collate",
    "group_by_image",
]
