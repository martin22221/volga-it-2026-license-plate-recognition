"""Tests for the Baseline V1 implementation: geometry, decoding, data, models, export, CLI.

The framework-free parts (geometry, CTC decoding, NMS, evaluation) always run.
The parts that need torch / onnxruntime are skipped when those are not
installed, so the inference-only environment still runs the rest.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.classifier import Classification, PlateType
from src.detector import BoundingBox, Detection
from src.ocr import OcrResult
from src.onnx_backend import nms, postprocess
from src.pipeline import INVALID_FORMAT_PENALTY, Pipeline
from src.recognition import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TWO_LINE_ASPECT,
    crop_to_quad,
    ctc_greedy,
    expand_box,
    is_two_line,
    perspective_coefficients,
    quad_aspect,
    quad_to_crop,
    read_plates,
    rectify,
    to_tensor,
)
from src.training.alphabet import encode
from src.training.data import CLASSES, load_split
from src.training.evaluation import evaluate_detections, evaluate_readings, failure_category, selection_score

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ geometry


def test_corner_normalisation_round_trips() -> None:
    quad = ((10.5, 20.0), (110.0, 18.0), (112.0, 45.0), (11.0, 47.5))
    crop = (5.0, 10.0, 120.0, 55.0)
    back = crop_to_quad(quad_to_crop(quad, crop), crop)
    for (a, b), (c, d) in zip(quad, back):
        assert a == pytest.approx(c) and b == pytest.approx(d)


def test_perspective_coefficients_map_destination_onto_source() -> None:
    dst = ((0, 0), (100, 0), (100, 20), (0, 20))
    src = ((13, 7), (140, 2), (150, 40), (10, 44))
    a, b, c, d, e, f, g, h = perspective_coefficients(dst, src)
    for (x, y), (sx, sy) in zip(dst, src):
        w = g * x + h * y + 1
        assert (a * x + b * y + c) / w == pytest.approx(sx, abs=1e-6)
        assert (d * x + e * y + f) / w == pytest.approx(sy, abs=1e-6)


def test_expand_box_is_clipped_to_the_image() -> None:
    assert expand_box((0, 0, 100, 20), 0.1, 105, 50) == (0.0, 0.0, 105.0, 22.0)


def test_the_two_line_rule_separates_every_training_quad_by_its_class() -> None:
    """The unroll decision is geometric; on all 10,200 training quads it must be exact."""
    wrong = [
        s.image for s in load_split("synthetic:train")
        if is_two_line(s.quad) != (s.plate_type == "type1a")
    ]
    assert wrong == []


def test_quad_aspect_of_a_rectangle() -> None:
    assert quad_aspect(((0, 0), (520, 0), (520, 112), (0, 112))) == pytest.approx(520 / 112)
    assert 290 / 170 < TWO_LINE_ASPECT < 520 / 112


def test_rectify_yields_the_recogniser_input_for_both_layouts() -> None:
    image = Image.new("RGB", (300, 200), "white")
    one = rectify(image, ((10, 10), (250, 10), (250, 60), (10, 60)), two_line=False)
    two = rectify(image, ((10, 10), (110, 10), (110, 70), (10, 70)), two_line=True)
    assert one.size == two.size == (INPUT_WIDTH, INPUT_HEIGHT)


def test_two_line_unroll_puts_the_top_band_first() -> None:
    """A square plate whose top half is black and bottom half white unrolls black|white."""
    image = Image.new("RGB", (100, 100), "white")
    image.paste((0, 0, 0), (0, 0, 100, 50))
    strip = np.asarray(rectify(image, ((0, 0), (100, 0), (100, 100), (0, 100)), two_line=True))
    assert strip[:, : INPUT_WIDTH // 2 - 8].mean() < 60
    assert strip[:, INPUT_WIDTH // 2 + 8 :].mean() > 200


def test_to_tensor_normalises_to_minus_one_one() -> None:
    batch = to_tensor([Image.new("RGB", (INPUT_WIDTH, INPUT_HEIGHT), (0, 255, 0))])
    assert batch.shape == (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    assert batch[0, 0].max() == -1.0 and batch[0, 1].min() == 1.0


# ------------------------------------------------------------------ decoding


def _log_probs_for(path: list[int], confidence: float = 0.9) -> np.ndarray:
    classes = 23
    probs = np.full((len(path), classes), (1 - confidence) / (classes - 1))
    for t, index in enumerate(path):
        probs[t, index] = confidence
    return np.log(probs)


def test_ctc_greedy_gives_one_score_per_emitted_character() -> None:
    a, one = encode("A1")
    text, scores = ctc_greedy(_log_probs_for([0, a, a, 0, one, 0, one, 0]))
    assert text == "A11"
    assert len(scores) == 3 and all(s == pytest.approx(0.9) for s in scores)


def test_read_plates_masks_unconfident_characters_and_never_learns_hash() -> None:
    path = [0] + [i for i in encode("A123BC77") for i in (i, 0)]

    def run(batch):
        n = batch.shape[0]
        lp = np.stack([_log_probs_for(path + [0] * (48 - len(path)), 0.4)] * n)
        corners = np.tile(np.array([0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], dtype=np.float32), (n, 1))
        return lp, np.tile(np.array([3.0, 0.0, 0.0, 0.0]), (n, 1)), corners

    image = Image.new("RGB", (400, 200), "white")
    (reading,) = read_plates(run, [(image, (100, 80, 300, 125))], char_threshold=0.5)
    assert reading.raw_text == "A123BC77"
    assert reading.text == "#" * 8
    assert reading.plate_type == "type1" and not reading.two_line


# ------------------------------------------------------------------ detector postprocessing


def test_nms_suppresses_overlaps_and_keeps_the_best() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.8, 0.9, 0.5]), 0.5) == [1, 2]


def test_postprocess_thresholds_and_rescales() -> None:
    boxes = np.array([[10, 10, 20, 20], [100, 100, 200, 150]], dtype=np.float32)
    out = postprocess(boxes, np.array([0.2, 0.9]), size=640, scale_x=2.0, scale_y=1.5,
                      score_threshold=0.25, nms_threshold=0.55, topk=300, max_detections=100)
    assert out == [((200.0, 150.0, 400.0, 225.0), pytest.approx(0.9))]


# ------------------------------------------------------------------ evaluation


def test_perfect_detections_score_full_map() -> None:
    rows = load_split("synthetic:val")[:20]
    truth = {r.image: [r] for r in rows}
    preds = {r.image: [((r.bbox[0], r.bbox[1], r.bbox[0] + r.bbox[2], r.bbox[1] + r.bbox[3]), 0.9)] for r in rows}
    report = evaluate_detections(truth, preds, confidence=0.25)
    assert report["mAP50"] == 1.0 and report["mAP50_95"] == 1.0
    assert report["fp"] == 0 and report["fn"] == 0


def test_a_missed_small_plate_counts_against_the_fallback_trigger() -> None:
    small = [r for r in load_split("synthetic:val") if r.bbox[2] < 80][:5]
    report = evaluate_detections({r.image: [r] for r in small}, {}, confidence=0.25)
    assert report["fallback_trigger"]["triggered"] is True
    assert len(report["missed"]) == 5


class _Reading:
    def __init__(self, raw: str, plate_type: str) -> None:
        self.raw_text = raw
        self.text = raw
        self.plate_type = plate_type
        self.two_line = plate_type == "type1a"


def test_raw_ocr_is_scored_raw_and_rare_classes_stay_visible() -> None:
    rows = [r for r in load_split("synthetic:val") if r.ocr_trainable]
    rows = [next(r for r in rows if r.plate_type == t) for t in ("type1", "type1a", "type1b")]
    readings = [_Reading(rows[0].plate_num, "type1"), _Reading(rows[1].plate_num, "type1"),
                _Reading(rows[2].plate_num[:-1], "type1b")]
    report = evaluate_readings(rows, readings)
    assert report["ocr_raw"]["exact"]["successes"] == 2
    assert report["type_and_string_correct"]["successes"] == 1
    assert report["classification"]["per_class"]["type1a"]["recall"] == 0.0
    assert report["classification"]["confusion"]["type1a"]["type1"] == 1
    assert selection_score(report)[0] == pytest.approx(1 / 3)


def test_failure_categories() -> None:
    row = next(r for r in load_split("synthetic:val") if r.ocr_trainable and r.plate_type == "type1")
    assert failure_category(row, row.plate_num, "type1") == "correct"
    assert failure_category(row, row.plate_num, "type1b") == "type_wrong"
    assert failure_category(row, row.plate_num[:-1], "type1").startswith("text_")


# ------------------------------------------------------------------ pipeline integration


class _Stage:
    name = "stub"

    def __init__(self, text: str, plate_type: PlateType) -> None:
        self.text, self.plate_type = text, plate_type

    def detect(self, image_path):
        return [Detection(BoundingBox(0, 0, 10, 5), 1.0)]

    def classify(self, image_path, detection):
        return Classification(self.plate_type, 1.0)

    def read(self, image_path, detection, classification):
        return OcrResult(self.text, 1.0)


def test_validation_uses_the_predicted_type(tmp_path: Path) -> None:
    """A type1-shaped string classified as type1b is not a valid type1b plate."""
    (tmp_path / "a.jpg").write_bytes(b"")
    stage = _Stage("A123BC77", PlateType.TYPE1B)
    (record,) = Pipeline(detector=stage, classifier=stage, ocr=stage).process_directory(tmp_path)
    assert record.confidence == pytest.approx(INVALID_FORMAT_PENALTY)


def test_an_unreadable_character_is_never_validated_into_a_confident_plate(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"")
    stage = _Stage("A12#BC77", PlateType.TYPE1)
    (record,) = Pipeline(detector=stage, classifier=stage, ocr=stage).process_directory(tmp_path)
    assert record.plate_num == "A12#BC77"
    assert record.confidence == pytest.approx(INVALID_FORMAT_PENALTY)


# ------------------------------------------------------------------ holdout guards on the new scripts


@pytest.mark.parametrize("script", ["predict_baseline", "evaluate_recognizer"])
def test_new_evaluation_scripts_refuse_the_holdout_without_a_reason(script: str, tmp_path: Path) -> None:
    module = __import__(f"scripts.{script}", fromlist=["main"])
    code = module.main(["--models", str(tmp_path), "--population", "real_holdout", "--out", str(tmp_path / "x.json")])
    assert code == 2


def test_training_bundle_verification_rejects_a_tree_holding_real_images() -> None:
    from scripts.train_baseline import verify_training_bundle

    problems = verify_training_bundle()
    assert any("real image" in p for p in problems)
    assert not any("synthetic" in p or "differs" in p or "digest" in p for p in problems)
