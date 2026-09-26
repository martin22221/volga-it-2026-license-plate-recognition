"""Deployed inference: the two exported ONNX graphs behind the pipeline's stages.

Runtime dependencies are ONNX Runtime (MIT), NumPy and Pillow -- no torch, no
network, no external API. Everything a model needs is in its ``.onnx`` file and
the ``.json`` sidecar written beside it by ``scripts/export_onnx.py``.

Stage mapping onto :mod:`src.pipeline`:

* :class:`OnnxDetector` is the ``Detector``;
* :class:`OnnxRecognizer` is **both** the ``PlateClassifier`` and the
  ``OcrEngine`` -- one network with a type head and a CTC head, as decided in
  ``docs/baseline_v1.md`` section 4. It reads each plate once and answers both
  stages from that one reading.

Detector postprocessing (score threshold, top-k, NMS, rescale) is here, in
NumPy, and training-time validation calls the same :func:`postprocess`, so the
checkpoint is selected on exactly the postprocessing that ships.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from src.classifier import Classification, PlateType
from src.detector import BoundingBox, Detection
from src.ocr import OcrResult
from src.recognition import MEAN, STD, Reading, read_plates

DETECTOR_FILE = "detector.onnx"
RECOGNIZER_FILE = "recognizer.onnx"


# ---------------------------------------------------------------------------
# detector postprocessing -- shared with training-time validation
# ---------------------------------------------------------------------------


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression; indices kept, highest score first."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def preprocess_detector(image: Image.Image, size: int) -> tuple[np.ndarray, float, float]:
    """Stretch to ``size`` x ``size`` and normalise. Returns the batch and the x/y scale back."""
    resized = image.convert("RGB").resize((size, size), Image.BILINEAR)
    array = (np.asarray(resized, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]), image.width / size, image.height / size


def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    size: int,
    scale_x: float,
    scale_y: float,
    score_threshold: float,
    nms_threshold: float,
    topk: int,
    max_detections: int,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Decoded anchors -> final ``((x1, y1, x2, y2), score)`` in original pixels."""
    candidates = np.nonzero(scores > score_threshold)[0]
    if candidates.size == 0:
        return []
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")[:topk]]
    kept_boxes = np.clip(boxes[candidates], 0, size)
    kept_scores = scores[candidates]
    keep = nms(kept_boxes, kept_scores, nms_threshold)[:max_detections]
    out = []
    for i in keep:
        x1, y1, x2, y2 = kept_boxes[i]
        out.append(((float(x1 * scale_x), float(y1 * scale_y), float(x2 * scale_x), float(y2 * scale_y)), float(kept_scores[i])))
    return out


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


def _session(path: Path, providers: Sequence[str] | None):
    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = [p for p in (providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]) if p in available]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=wanted or ["CPUExecutionProvider"])


def _sidecar(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


@dataclass
class StageTimer:
    """Milliseconds per stage, accumulated, so the benchmark can split the total."""

    samples: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, stage: str, seconds: float) -> None:
        self.samples[stage].append(seconds * 1000.0)


class _ImageCache:
    """The most recently decoded image, so detector and recogniser decode once."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.image: Image.Image | None = None

    def get(self, path: Path, timer: StageTimer | None = None) -> Image.Image:
        path = Path(path)
        if path != self.path:
            started = time.perf_counter()
            with Image.open(path) as handle:
                self.image = handle.convert("RGB")
            self.path = path
            if timer:
                timer.add("decode", time.perf_counter() - started)
        return self.image


class OnnxDetector:
    """The exported SSDlite detector as a pipeline ``Detector``."""

    name = "ssdlite-mnv3-640-onnx"

    def __init__(
        self,
        model_path: Path,
        *,
        confidence: float | None = None,
        providers: Sequence[str] | None = None,
        cache: _ImageCache | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.meta = _sidecar(Path(model_path))
        self.session = _session(Path(model_path), providers)
        self.size = int(self.meta["input_size"])
        self.confidence = float(self.meta["confidence_threshold"] if confidence is None else confidence)
        self.cache = cache or _ImageCache()
        self.timer = timer

    def detect_image(self, image: Image.Image) -> list[tuple[tuple[float, float, float, float], float]]:
        started = time.perf_counter()
        batch, sx, sy = preprocess_detector(image, self.size)
        t1 = time.perf_counter()
        boxes, scores = self.session.run(None, {"image": batch})
        t2 = time.perf_counter()
        found = postprocess(
            boxes[0], scores[0], size=self.size, scale_x=sx, scale_y=sy,
            score_threshold=self.confidence,
            nms_threshold=float(self.meta["nms_threshold"]),
            topk=int(self.meta["topk_candidates"]),
            max_detections=int(self.meta["max_detections"]),
        )
        if self.timer:
            self.timer.add("detector_preprocess", t1 - started)
            self.timer.add("detector", t2 - t1)
            self.timer.add("detector_postprocess", time.perf_counter() - t2)
        return found

    def detect(self, image_path: Path) -> Sequence[Detection]:
        image = self.cache.get(image_path, self.timer)
        out = []
        for (x1, y1, x2, y2), score in self.detect_image(image):
            box = BoundingBox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            if box.width > 0 and box.height > 0:
                out.append(Detection(box=box, confidence=score))
        return out


class OnnxRecognizer:
    """The exported recogniser, serving as both the classifier and the OCR stage."""

    name = "crnn-3head-onnx"

    def __init__(
        self,
        model_path: Path,
        *,
        char_threshold: float | None = None,
        providers: Sequence[str] | None = None,
        cache: _ImageCache | None = None,
        timer: StageTimer | None = None,
    ) -> None:
        self.meta = _sidecar(Path(model_path))
        self.session = _session(Path(model_path), providers)
        self.char_threshold = float(self.meta["char_confidence_for_hash"] if char_threshold is None else char_threshold)
        self.cache = cache or _ImageCache()
        self.timer = timer
        self._last: tuple[tuple, Reading] | None = None

    def run_model(self, batch: np.ndarray):
        started = time.perf_counter()
        out = self.session.run(None, {"crop": batch.astype(np.float32)})
        if self.timer:
            self.timer.add("recognizer_network", time.perf_counter() - started)
        return out

    def read_boxes(self, image: Image.Image, boxes: Sequence[tuple[float, float, float, float]]) -> list[Reading]:
        started = time.perf_counter()
        readings = read_plates(self.run_model, [(image, b) for b in boxes], char_threshold=self.char_threshold)
        if self.timer:
            self.timer.add("recognizer_total", time.perf_counter() - started)
        return readings

    def _reading(self, image_path: Path, detection: Detection) -> Reading:
        key = (str(image_path), detection.box)
        if self._last is None or self._last[0] != key:
            image = self.cache.get(image_path, self.timer)
            b = detection.box
            self._last = (key, self.read_boxes(image, [(b.x1, b.y1, b.x2, b.y2)])[0])
        return self._last[1]

    # PlateClassifier
    def classify(self, image_path: Path, detection: Detection) -> Classification:
        reading = self._reading(image_path, detection)
        return Classification(plate_type=PlateType(reading.plate_type), confidence=reading.type_confidence)

    # OcrEngine
    def read(self, image_path: Path, detection: Detection, classification: Classification) -> OcrResult:
        reading = self._reading(image_path, detection)
        return OcrResult(text=reading.text, confidence=reading.mean_char_score)


def load_models(
    models_dir: Path,
    *,
    providers: Sequence[str] | None = None,
    timer: StageTimer | None = None,
) -> tuple[OnnxDetector, OnnxRecognizer]:
    """Both stages from one directory, sharing one decoded-image cache."""
    models_dir = Path(models_dir)
    cache = _ImageCache()
    detector = OnnxDetector(models_dir / DETECTOR_FILE, providers=providers, cache=cache, timer=timer)
    recognizer = OnnxRecognizer(models_dir / RECOGNIZER_FILE, providers=providers, cache=cache, timer=timer)
    return detector, recognizer


__all__ = [
    "DETECTOR_FILE",
    "OnnxDetector",
    "OnnxRecognizer",
    "RECOGNIZER_FILE",
    "StageTimer",
    "load_models",
    "nms",
    "postprocess",
    "preprocess_detector",
]
