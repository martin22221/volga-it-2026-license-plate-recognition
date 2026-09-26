"""Scoring whole populations: detector boxes, and recogniser reads on known boxes.

Built on :mod:`src.training.metrics` (the metric definitions fixed before any
training) and free of any framework, so the training scripts, the evaluation
scripts and the tests all compute every number with the same code.

**Raw OCR is scored raw.** The OCR figures use the decoder's string before the
``#`` mask and before the validator sees it. A validator can only make a read
look better (by rejecting it into a lower confidence) or stay silent; scoring
after it would let format checking inflate the number. The masked string is
scored separately and labelled as such.

Breakdowns are reported only where the metadata supports them: difficulty from
``dataset/splits/synthetic_splits.csv`` (committed); condition tags from
``meta.csv`` (committed); character height and occluded positions from the
sealed batch's ``generation.jsonl``, when it is present on this machine.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.dataset_meta import CSV_DELIMITER
from src.training.data import CLASSES, DATASET, REPO_ROOT, Sample
from src.training.metrics import (
    average_precision,
    confusion,
    detection_scores,
    levenshtein,
    match_detections,
    ocr_scores,
    per_class_scores,
    proportion,
)

GENERATION_JSONL = (
    REPO_ROOT / "data" / "synthetic_production" / "v2_3_seed2026091401_n12000" / "generation.jsonl"
)

#: A plate narrower than this (original pixels) is "small"; the pre-agreed
#: detector fallback trigger is recall below 0.85 on these (config ``detector.fallback``).
SMALL_PLATE_WIDTH: float = 80.0

#: mAP50-95: the COCO IoU ladder.
IOU_LADDER: tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))

BoxXYXY = tuple[float, float, float, float]


def to_xywh(box: BoxXYXY) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


@lru_cache(maxsize=1)
def load_difficulty() -> dict[str, str]:
    path = DATASET / "splits" / "synthetic_splits.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["image"]: r["difficulty"] for r in csv.DictReader(handle, delimiter=CSV_DELIMITER)}


@lru_cache(maxsize=2)
def load_generation_info(path: Path = GENERATION_JSONL) -> dict[str, dict]:
    """Per-image render facts from the sealed batch, or ``{}`` if it is not here."""
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            out[record["image"]] = {
                "min_char_height_px": record["geometry"]["min_char_height_px"],
                "hidden_positions": len(record.get("hidden_positions") or []),
            }
    return out


def char_height_bin(height: float | None) -> str:
    if height is None:
        return "unknown"
    for limit, name in ((10, "<10px"), (14, "10-14px"), (20, "14-20px"), (30, "20-30px")):
        if height < limit:
            return name
    return ">=30px"


def width_bin(width: float) -> str:
    if width < SMALL_PLATE_WIDTH:
        return "<80px"
    if width < 150:
        return "80-150px"
    return ">=150px"


def _groups_for(sample: Sample, difficulty: Mapping[str, str], generation: Mapping[str, dict]) -> dict[str, str]:
    groups = {"type": sample.plate_type, "width": width_bin(sample.bbox[2])}
    if sample.image in difficulty:
        groups["difficulty"] = difficulty[sample.image]
    info = generation.get(sample.image)
    if info:
        groups["char_height"] = char_height_bin(info["min_char_height_px"])
        groups["occluded"] = "yes" if info["hidden_positions"] else "no"
    return groups


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def evaluate_detections(
    truth: Mapping[str, Sequence[Sample]],
    predictions: Mapping[str, Sequence[tuple[BoxXYXY, float]]],
    *,
    confidence: float,
    difficulty: Mapping[str, str] | None = None,
) -> dict:
    """Detector quality over one population.

    AP is computed over every prediction the detector returns (it is a ranking
    metric); precision, recall, FP and FN are at the operating ``confidence``.
    Recall is broken down by condition tag, difficulty and plate width.
    """
    difficulty = difficulty or {}
    ap_scored: dict[float, list[tuple[float, bool]]] = {t: [] for t in IOU_LADDER}
    n_truth = 0
    tp = fp = fn = 0
    images_with_fp = 0
    recall_groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    missed: list[dict] = []
    false_positives: list[dict] = []

    for image, rows in truth.items():
        truth_boxes = [r.bbox for r in rows]
        preds = [(to_xywh(b), s) for b, s in predictions.get(image, [])]
        n_truth += len(rows)
        for threshold in IOU_LADDER:
            matches, _, unmatched = match_detections(truth_boxes, preds, threshold)
            ap_scored[threshold] += [(preds[p][1], True) for _, p in matches]
            ap_scored[threshold] += [(preds[p][1], False) for p in unmatched]

        operating = [p for p in preds if p[1] >= confidence]
        matches, unmatched_truth, unmatched_pred = match_detections(truth_boxes, operating, 0.5)
        tp, fn, fp = tp + len(matches), fn + len(unmatched_truth), fp + len(unmatched_pred)
        if unmatched_pred:
            images_with_fp += 1
            false_positives += [
                {"image": image, "score": round(operating[p][1], 4), "box_xywh": [round(v, 1) for v in operating[p][0]]}
                for p in unmatched_pred
            ]
        hit = {t for t, _ in matches}
        for index, row in enumerate(rows):
            found = index in hit
            keys = {"all": "all", "width": width_bin(row.bbox[2])}
            if row.image in difficulty:
                keys["difficulty"] = difficulty[row.image]
            for name, value in keys.items():
                recall_groups[name][value][0] += found
                recall_groups[name][value][1] += 1
            for tag in row.conditions:
                recall_groups["condition"][tag][0] += found
                recall_groups["condition"][tag][1] += 1
            if not found:
                missed.append({"image": image, "plate_num": row.plate_num, "type": row.plate_type,
                               "width_px": round(row.bbox[2], 1), "conditions": list(row.conditions)})

    ap = {t: average_precision(ap_scored[t], n_truth) for t in IOU_LADDER}
    recall_by = {
        name: {value: proportion(*counts) for value, counts in sorted(values.items())}
        for name, values in recall_groups.items()
    }
    small = recall_by.get("width", {}).get("<80px")
    return {
        "operating_confidence": confidence,
        "images": len(truth),
        "plates": n_truth,
        **detection_scores(tp, fp, fn),
        "mAP50": round(ap[0.5], 4),
        "mAP75": round(ap[0.75], 4),
        "mAP50_95": round(sum(ap.values()) / len(ap), 4),
        "images_with_false_positive": images_with_fp,
        "recall_by": recall_by,
        "fallback_trigger": {
            "rule": f"recall on plates narrower than {SMALL_PLATE_WIDTH:.0f} px below 0.85",
            "small_plate_recall": small,
            "triggered": bool(small and small["total"] and small["value"] < 0.85),
        },
        "missed": missed,
        "false_positives": false_positives,
    }


# ---------------------------------------------------------------------------
# recognition on known boxes
# ---------------------------------------------------------------------------


def failure_category(sample: Sample, raw_text: str, plate_type: str) -> str:
    type_ok = plate_type == sample.plate_type
    if not sample.ocr_trainable:
        return "type_ok_unscored_text" if type_ok else "type_wrong_unscored_text"
    text_ok = raw_text == sample.plate_num
    if text_ok and type_ok:
        return "correct"
    if text_ok:
        return "type_wrong"
    distance = levenshtein(sample.plate_num, raw_text)
    kind = "text_1_edit" if distance == 1 else ("text_length" if len(raw_text) != len(sample.plate_num) else "text_multi_edit")
    return kind if type_ok else f"{kind}+type_wrong"


def evaluate_readings(samples: Sequence[Sample], readings: Sequence, *, with_rows: bool = False) -> dict:
    """Type classification and OCR over plates read from their ground-truth boxes.

    ``readings`` are :class:`src.recognition.Reading`, one per sample.
    """
    difficulty = load_difficulty()
    generation = load_generation_info()
    type_pairs = [(s.plate_type, r.plate_type) for s, r in zip(samples, readings)]
    matrix = confusion(type_pairs, CLASSES)
    scored = [(s, r) for s, r in zip(samples, readings) if s.ocr_trainable]
    raw_pairs = [(s.plate_num, r.raw_text) for s, r in scored]
    masked_pairs = [(s.plate_num, r.text) for s, r in scored]
    both = sum(1 for s, r in scored if s.plate_num == r.raw_text and s.plate_type == r.plate_type)

    grouped_ocr: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    grouped_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for sample, reading in zip(samples, readings):
        groups = _groups_for(sample, difficulty, generation)
        tags = [("condition", t) for t in sample.conditions]
        for name, value in list(groups.items()) + tags:
            grouped_type[name][value][0] += reading.plate_type == sample.plate_type
            grouped_type[name][value][1] += 1
            if sample.ocr_trainable:
                grouped_ocr[name][value].append((sample.plate_num, reading.raw_text))

    def _ocr_block(pairs):
        scores = ocr_scores(pairs)
        return {"exact": scores["exact"], "cer": None if scores["cer"] is None else round(scores["cer"], 4)}

    categories = Counter(failure_category(s, r.raw_text, r.plate_type) for s, r in zip(samples, readings))
    result = {
        "plates": len(samples),
        "ocr_scored_plates": len(scored),
        "ocr_excluded_unreadable_labels": len(samples) - len(scored),
        "classification": {
            "accuracy": proportion(sum(t == p for t, p in type_pairs), len(type_pairs)),
            "per_class": per_class_scores(matrix),
            "confusion": matrix,
        },
        "ocr_raw": {**ocr_scores(raw_pairs), "character_accuracy": None},
        "ocr_masked": ocr_scores(masked_pairs),
        "type_and_string_correct": proportion(both, len(scored)),
        "ocr_by": {n: {v: _ocr_block(p) for v, p in sorted(vals.items())} for n, vals in grouped_ocr.items()},
        "type_accuracy_by": {n: {v: proportion(*c) for v, c in sorted(vals.items())} for n, vals in grouped_type.items()},
        "failure_categories": dict(categories.most_common()),
        "two_line_decision": dict(Counter(
            f"{s.plate_type}->{'two' if r.two_line else 'one'}_line" for s, r in zip(samples, readings)
        )),
    }
    cer = result["ocr_raw"]["cer"]
    result["ocr_raw"]["character_accuracy"] = None if cer is None else round(1 - cer, 4)
    if with_rows:
        result["rows"] = [
            {
                "image": s.image, "truth": s.plate_num, "truth_type": s.plate_type,
                "raw": r.raw_text, "masked": r.text, "type": r.plate_type,
                "type_conf": round(r.type_confidence, 4),
                "char_scores": [round(c, 3) for c in r.char_scores],
                "two_line": r.two_line,
                "category": failure_category(s, r.raw_text, r.plate_type),
            }
            for s, r in zip(samples, readings)
        ]
    return result


def selection_score(result: Mapping) -> tuple[float, float]:
    """Recogniser checkpoint key: type-and-string exact on synthetic:val, then -CER.

    Fixed here, before the first run, so the choice of what "best" means cannot
    follow the results.
    """
    value = result["type_and_string_correct"]["value"] or 0.0
    cer = result["ocr_raw"]["cer"]
    return (value, -(cer if cer is not None else 1.0))


__all__ = [
    "IOU_LADDER",
    "SMALL_PLATE_WIDTH",
    "evaluate_detections",
    "evaluate_readings",
    "failure_category",
    "load_difficulty",
    "load_generation_info",
    "selection_score",
    "to_xywh",
]
