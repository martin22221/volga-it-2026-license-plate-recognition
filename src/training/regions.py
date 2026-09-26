"""A cached region around every plate -- no framework needed.

Cutting a plate and its surroundings out of the full image once, instead of
decoding a 1280x720 (or 26-megapixel) JPEG for every read, is what makes the
recogniser's epochs and its evaluations cheap. Used by the trainer
(:mod:`src.training.torch_data`) and by ``scripts/evaluate_recognizer.py``,
which must run on the inference stack alone.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from src.training.data import Sample


#: Context kept around each plate in the cache, as a fraction of the plate's
#: size per side. Must exceed the widest box jitter plus the crop margin.
REGION_CONTEXT: float = 0.6
#: Plates wider than this are downscaled in the cache. The reading strip is 192
#: px wide, so nothing above ~2x that carries information the model can use.
REGION_MAX_PLATE_WIDTH: float = 384.0


@dataclass
class Region:
    """A plate and its surroundings, cut once from the full image."""

    pixels: np.ndarray                      # H x W x 3 uint8
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in region pixels
    quad: tuple[tuple[float, float], ...]   # in region pixels

    def image(self) -> Image.Image:
        return Image.fromarray(self.pixels)


def cut_region(sample: Sample) -> Region:
    with Image.open(sample.path) as handle:
        image = handle.convert("RGB")
    x, y, w, h = sample.bbox
    rx1 = max(0.0, x - w * REGION_CONTEXT)
    ry1 = max(0.0, y - h * REGION_CONTEXT)
    rx2 = min(float(image.width), x + w * (1 + REGION_CONTEXT))
    ry2 = min(float(image.height), y + h * (1 + REGION_CONTEXT))
    scale = min(1.0, REGION_MAX_PLATE_WIDTH / max(w, 1.0))
    ix1, iy1, ix2, iy2 = int(rx1), int(ry1), int(math.ceil(rx2)), int(math.ceil(ry2))
    out_w = max(1, round((ix2 - ix1) * scale))
    out_h = max(1, round((iy2 - iy1) * scale))
    region = image.resize((out_w, out_h), Image.BILINEAR, box=(ix1, iy1, ix2, iy2))
    sx, sy = out_w / (ix2 - ix1), out_h / (iy2 - iy1)
    return Region(
        pixels=np.asarray(region, dtype=np.uint8),
        box=((x - ix1) * sx, (y - iy1) * sy, (x + w - ix1) * sx, (y + h - iy1) * sy),
        quad=tuple(((qx - ix1) * sx, (qy - iy1) * sy) for qx, qy in sample.quad),
    )


def cut_regions(samples: Sequence[Sample], workers: int = 0) -> list[Region]:
    """Every sample's region, in order. Parallel, because it is 12,000 JPEG decodes."""
    if workers <= 1:
        return [cut_region(s) for s in samples]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(cut_region, samples, chunksize=64))


__all__ = ["REGION_CONTEXT", "REGION_MAX_PLATE_WIDTH", "Region", "cut_region", "cut_regions"]
