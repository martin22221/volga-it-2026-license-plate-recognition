"""Plate geometry, rectification and the two-pass read -- framework-free.

Everything between "the detector drew a box" and "here is a string" lives here,
in NumPy and Pillow only, so that **training-time validation and deployed
inference run literally the same code**. The only thing that differs between
them is the function that evaluates the network: a torch module while
training, an ONNX Runtime session when deployed. Both are passed in as a
``run_model`` callable, and nothing else is allowed to vary.

The read is two passes of one network (``docs/baseline_v1.md`` section 3.5):

1. **Corner pass.** The detector box, widened by a margin, is resized to the
   recogniser's 48x192 input. Only the corner head's output is used: four plate
   corners, normalised to that crop.
2. **Reading pass.** The plate is perspective-warped from those corners into a
   flat 48x192 strip. A two-line plate (``type1a``) is warped into a square and
   its two bands are laid side by side, so one left-to-right CTC reader handles
   every class. Only the type head and the CTC head are used.

Whether a plate is two-line is decided from the *shape of the predicted quad*,
not from the type head: ``type1a`` is 290x170 mm and the one-line plates are
520x112 mm, and in all 10,200 training quads the two populations are separated
by a wide gap (max 1.83 vs min 3.31, see :data:`TWO_LINE_ASPECT`).

Uncertainty is never learned as a glyph. Per-character scores come from the
CTC posteriors, and :func:`src.training.alphabet.mask_unconfident` turns a
low-scoring character into ``#`` at output time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from src.training.alphabet import BLANK_INDEX, collapse, mask_unconfident
from src.training.data import CLASSES

#: Recogniser input, fixed by ``configs/baseline_v1.json``.
INPUT_HEIGHT: int = 48
INPUT_WIDTH: int = 192

#: The detector box is widened by this fraction of its size on every side
#: before the corner pass, so a slightly tight box still contains the corners.
BOX_MARGIN: float = 0.08

#: Quad aspect (mean horizontal edge / mean vertical edge) below which a plate
#: is read as two-line. Training quads: type1a 1.21-1.83, type1/type1b
#: 3.31-5.00. 2.5 is the geometric midpoint of the gap.
TWO_LINE_ASPECT: float = 2.5

#: Inside the rectified strip, the plate occupies the canvas minus this margin
#: on each side, so a corner estimate that is slightly inside the true plate
#: does not cut the first or last character.
RECTIFY_MARGIN_X: float = 0.03
RECTIFY_MARGIN_Y: float = 0.06

#: Pixel normalisation shared by both networks: ``(x / 255 - 0.5) / 0.5``.
MEAN: float = 0.5
STD: float = 0.5

#: ``run_model(batch)`` -> ``(log_probs [N,T,C], type_logits [N,4], corners [N,8])``.
#: ``batch`` is float32 ``[N, 3, 48, 192]``, already normalised.
RunModel = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]

Box = tuple[float, float, float, float]            # x1, y1, x2, y2
Quad = tuple[tuple[float, float], ...]             # TL, TR, BR, BL


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def expand_box(box: Box, margin: float, width: int, height: int) -> Box:
    """``box`` widened by ``margin`` of its size on each side, clipped to the image."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(width), x2 + mx),
        min(float(height), y2 + my),
    )


def quad_aspect(quad: Quad) -> float:
    """Mean horizontal edge over mean vertical edge. Robust to moderate perspective."""
    tl, tr, br, bl = quad
    horizontal = math.dist(tl, tr) + math.dist(bl, br)
    vertical = math.dist(tl, bl) + math.dist(tr, br)
    return horizontal / vertical if vertical > 0 else float("inf")


def is_two_line(quad: Quad) -> bool:
    return quad_aspect(quad) < TWO_LINE_ASPECT


def quad_to_crop(quad: Quad, crop: Box) -> list[float]:
    """Image-space corners to the 8 numbers the corner head is trained on."""
    x1, y1, x2, y2 = crop
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    out: list[float] = []
    for x, y in quad:
        out += [(x - x1) / w, (y - y1) / h]
    return out


def crop_to_quad(values: Sequence[float], crop: Box) -> Quad:
    """The corner head's 8 numbers back to image-space corners."""
    x1, y1, x2, y2 = crop
    w, h = x2 - x1, y2 - y1
    return tuple(
        (x1 + float(values[2 * i]) * w, y1 + float(values[2 * i + 1]) * h) for i in range(4)
    )


def perspective_coefficients(destination: Quad, source: Quad) -> tuple[float, ...]:
    """Pillow ``PERSPECTIVE`` coefficients mapping ``destination`` points to ``source``.

    Pillow's transform is inverse: for every output pixel it asks where to
    sample the input, so the system is solved output -> input.
    """
    rows, rhs = [], []
    for (dx, dy), (sx, sy) in zip(destination, source):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        rhs += [sx, sy]
    solution = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return tuple(float(v) for v in solution)


def _warp(image: Image.Image, quad: Quad, width: int, height: int) -> Image.Image:
    mx, my = width * RECTIFY_MARGIN_X, height * RECTIFY_MARGIN_Y
    destination = ((mx, my), (width - mx, my), (width - mx, height - my), (mx, height - my))
    coefficients = perspective_coefficients(destination, quad)
    return image.transform((width, height), Image.PERSPECTIVE, coefficients, Image.BILINEAR)


def rectify(image: Image.Image, quad: Quad, two_line: bool) -> Image.Image:
    """The flat 48x192 strip the reading pass sees.

    One-line plates are warped straight into it. A two-line plate is warped
    into a 96x96 square whose upper and lower halves are then placed side by
    side, which puts ``M 000`` before ``MM 55`` in reading order.
    """
    if not two_line:
        return _warp(image, quad, INPUT_WIDTH, INPUT_HEIGHT)
    half = INPUT_WIDTH // 2
    square = _warp(image, quad, half, 2 * INPUT_HEIGHT)
    strip = Image.new("RGB", (INPUT_WIDTH, INPUT_HEIGHT))
    strip.paste(square.crop((0, 0, half, INPUT_HEIGHT)), (0, 0))
    strip.paste(square.crop((0, INPUT_HEIGHT, half, 2 * INPUT_HEIGHT)), (half, 0))
    return strip


def box_crop(image: Image.Image, crop: Box) -> Image.Image:
    """The corner pass's input: ``crop`` resized to 48x192, whatever its aspect."""
    return image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR, box=tuple(crop))


def to_tensor(images: Sequence[Image.Image]) -> np.ndarray:
    """RGB images to a normalised float32 ``[N, 3, H, W]`` batch."""
    batch = np.stack([np.asarray(im.convert("RGB"), dtype=np.float32) for im in images])
    batch = (batch / 255.0 - MEAN) / STD
    return np.ascontiguousarray(batch.transpose(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def ctc_greedy(log_probs: np.ndarray) -> tuple[str, list[float]]:
    """Best-path decode of one ``[T, C]`` sequence, with a score per character.

    A character's score is the highest posterior it reaches over the frames of
    its run -- the probability the model assigned to that glyph where it was
    most sure of it. Uses :func:`alphabet.collapse` for the string, so the
    decode cannot disagree with the one the tests pin.
    """
    probs = np.exp(log_probs)
    path = probs.argmax(axis=1)
    text = collapse(path.tolist())
    scores: list[float] = []
    previous = BLANK_INDEX
    for t, index in enumerate(path):
        index = int(index)
        if index != BLANK_INDEX and index != previous:
            scores.append(float(probs[t, index]))
        elif index != BLANK_INDEX and index == previous:
            scores[-1] = max(scores[-1], float(probs[t, index]))
        previous = index
    return text, scores


# ---------------------------------------------------------------------------
# the read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """Everything the recogniser says about one plate."""

    raw_text: str                 # the decode, before any masking
    text: str                     # low-confidence characters replaced by '#'
    char_scores: tuple[float, ...]
    plate_type: str
    type_confidence: float
    type_probs: tuple[float, ...]
    quad: Quad
    two_line: bool

    @property
    def mean_char_score(self) -> float:
        return float(np.mean(self.char_scores)) if self.char_scores else 0.0


def read_plates(
    run_model: RunModel,
    items: Sequence[tuple[Image.Image, Box]],
    *,
    char_threshold: float,
    margin: float = BOX_MARGIN,
) -> list[Reading]:
    """Two-pass read of every ``(image, detector_box)`` in ``items``, batched."""
    if not items:
        return []
    crops = [expand_box(box, margin, im.width, im.height) for im, box in items]
    _, _, corners = run_model(to_tensor([box_crop(im, c) for (im, _), c in zip(items, crops)]))
    quads = [crop_to_quad(corners[i], crops[i]) for i in range(len(items))]
    two_line = [is_two_line(q) for q in quads]
    strips = [rectify(im, q, tl) for (im, _), q, tl in zip(items, quads, two_line)]
    log_probs, type_logits, _ = run_model(to_tensor(strips))
    type_probs = softmax(type_logits, axis=1)

    out: list[Reading] = []
    for i in range(len(items)):
        raw, scores = ctc_greedy(log_probs[i])
        k = int(type_probs[i].argmax())
        out.append(
            Reading(
                raw_text=raw,
                text=mask_unconfident(raw, scores, char_threshold),
                char_scores=tuple(scores),
                plate_type=CLASSES[k],
                type_confidence=float(type_probs[i, k]),
                type_probs=tuple(float(p) for p in type_probs[i]),
                quad=quads[i],
                two_line=two_line[i],
            )
        )
    return out


__all__ = [
    "BOX_MARGIN",
    "INPUT_HEIGHT",
    "INPUT_WIDTH",
    "Reading",
    "TWO_LINE_ASPECT",
    "box_crop",
    "crop_to_quad",
    "ctc_greedy",
    "expand_box",
    "is_two_line",
    "perspective_coefficients",
    "quad_aspect",
    "quad_to_crop",
    "read_plates",
    "rectify",
    "softmax",
    "to_tensor",
]
