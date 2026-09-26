#!/usr/bin/env python
"""Score a baseline's predictions against Dataset V1, population by population.

Takes a predictions file and produces the metrics defined in
``src/training/metrics.py``. It works today, before any model exists, because
it scores a file rather than a model -- which also means the evaluation path
can be tested and reviewed before it is ever used in anger.

Two refusals are built in:

* **Synthetic and real are never merged.** Each population is scored and
  printed separately; there is no flag that produces one combined number.
* **The real holdout needs a reason.** ``--population real_holdout`` requires
  ``--final-report "<why>"``. It is 2 images, and it is worth exactly one use.

Predictions file: JSON, a list of objects::

    {"image": "images/real/.../wcc_0001.jpg",
     "detections": [{"bbox": [x, y, w, h], "confidence": 0.9,
                     "plate_type": "type1", "plate_num": "K362HH977",
                     "char_scores": [0.99, ...]}]}

Usage::

    python scripts/evaluate_baseline.py --predictions runs/.../preds.json \\
        --population synthetic_val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.data import CLASSES, load_samples, load_split  # noqa: E402
from src.training.metrics import (  # noqa: E402
    EvaluationReport,
    average_precision,
    confusion,
    detection_scores,
    match_detections,
    ocr_scores,
    per_class_scores,
    proportion,
    report_text,
)

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
EXIT_OK, EXIT_UNUSABLE, EXIT_BLOCKED = 0, 1, 2


def _splits_for(population: str, config: dict) -> list[str]:
    mapping = config["evaluation"]["populations"]
    if population not in mapping:
        raise KeyError(population)
    value = mapping[population]
    return list(value) if isinstance(value, list) else [value]


def evaluate(
    samples_by_image: dict,
    predictions: list[dict],
    iou_threshold: float,
    operating_confidence: float = 0.0,
) -> dict:
    """Detection, classification, OCR and end-to-end, over one population.

    Detections flagged ``below_operating`` (kept only so AP can be computed)
    or scoring under ``operating_confidence`` are excluded from everything but
    the ``detection_full`` block's ranking metrics.
    """
    from collections import Counter

    from src.training.evaluation import evaluate_detections, load_difficulty

    all_predictions = {
        p["image"]: [
            ((d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]),
             float(d.get("confidence", 0.0)))
            for d in p.get("detections", [])
        ]
        for p in predictions
    }
    predictions = [
        {**p, "detections": [
            d for d in p.get("detections", [])
            if not d.get("below_operating") and float(d.get("confidence", 0.0)) >= operating_confidence
        ]}
        for p in predictions
    ]
    failures: Counter = Counter()
    tp = fp = fn = 0
    scored: list[tuple[float, bool]] = []
    class_pairs: list[tuple[str, str]] = []
    ocr_pairs: list[tuple[str, str]] = []
    end_to_end_correct = 0
    end_to_end_total = 0
    images_with_no_truth_predicted_on = 0

    predictions_by_image = {p["image"]: p.get("detections", []) for p in predictions}

    for image, truth_rows in samples_by_image.items():
        truth_boxes = [r.bbox for r in truth_rows]
        preds = predictions_by_image.get(image, [])
        pred_boxes = [(p["bbox"], float(p.get("confidence", 0.0))) for p in preds]

        matches, unmatched_truth, unmatched_pred = match_detections(
            truth_boxes, pred_boxes, iou_threshold
        )
        tp += len(matches)
        fn += len(unmatched_truth)
        fp += len(unmatched_pred)
        for _, pred_index in [(t, p) for t, p in matches]:
            scored.append((pred_boxes[pred_index][1], True))
        for pred_index in unmatched_pred:
            scored.append((pred_boxes[pred_index][1], False))

        if not truth_rows and preds:
            images_with_no_truth_predicted_on += 1

        for truth_index, pred_index in matches:
            truth_row = truth_rows[truth_index]
            pred = preds[pred_index]
            end_to_end_total += 1
            predicted_type = pred.get("plate_type", "other")
            predicted_num = (pred.get("plate_num") or "").strip()
            class_pairs.append((truth_row.plate_type, predicted_type))
            if truth_row.ocr_trainable:
                ocr_pairs.append((truth_row.plate_num, predicted_num))
            type_ok = predicted_type == truth_row.plate_type
            text_ok = predicted_num == truth_row.plate_num
            if type_ok and text_ok:
                end_to_end_correct += 1
                failures["correct"] += 1
            elif not truth_row.ocr_trainable:
                failures["unreadable_label" + ("" if type_ok else "+type_wrong")] += 1
            else:
                failures[
                    ("string_wrong" if not text_ok else "")
                    + ("+" if not text_ok and not type_ok else "")
                    + ("type_wrong" if not type_ok else "")
                ] += 1
        # a plate the detector missed is an end-to-end failure, not an absence
        end_to_end_total += len(unmatched_truth)
        failures["not_detected"] += len(unmatched_truth)
        failures["false_positive_detection"] += len(unmatched_pred)

    matrix = confusion(class_pairs, CLASSES)
    return {
        "detection": {
            **detection_scores(tp, fp, fn),
            "average_precision": round(average_precision(scored, tp + fn), 4),
            "images_with_no_plate_predicted_on": images_with_no_truth_predicted_on,
        },
        "classification": {
            "per_class": per_class_scores(matrix),
            "confusion": matrix,
            "matched_plates": len(class_pairs),
        },
        "ocr": ocr_scores(ocr_pairs),
        "end_to_end": {
            "correct_detection_class_and_string": proportion(
                end_to_end_correct, end_to_end_total
            ),
            "outcomes": dict(failures.most_common()),
        },
        "detection_full": {
            k: v
            for k, v in evaluate_detections(
                samples_by_image, all_predictions,
                confidence=operating_confidence, difficulty=load_difficulty(),
            ).items()
            if k not in ("missed", "false_positives")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    parser.add_argument(
        "--final-report",
        metavar="WHY",
        default="",
        help="required to score real_holdout; it is a once-only measurement",
    )
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"error: no config at {args.config}", file=sys.stderr)
        return EXIT_UNUSABLE
    config = json.loads(args.config.read_text(encoding="utf-8"))

    if args.population == "real_holdout" and not args.final_report.strip():
        print(
            "refusing: real_holdout is the final, once-only evaluation set.\n"
            "Pass --final-report \"<why>\" and only for a report, never to choose a model.",
            file=sys.stderr,
        )
        return EXIT_BLOCKED

    try:
        split_names = _splits_for(args.population, config)
    except KeyError:
        known = ", ".join(config["evaluation"]["populations"])
        print(f"error: unknown population {args.population!r}; known: {known}", file=sys.stderr)
        return EXIT_UNUSABLE

    if not args.predictions.is_file():
        print(f"error: no predictions at {args.predictions}", file=sys.stderr)
        return EXIT_UNUSABLE
    loaded = json.loads(args.predictions.read_text(encoding="utf-8"))
    # predict_baseline.py wraps the list with its provenance; a bare list is accepted too
    predictions = loaded["predictions"] if isinstance(loaded, dict) else loaded
    operating = (
        float(loaded.get("operating_confidence", 0.0)) if isinstance(loaded, dict)
        else float(config["evaluation"]["detector_confidence"])
    )

    all_samples = load_samples()
    rows = []
    for name in split_names:
        rows += load_split(name, samples=all_samples, unlock_holdout=args.final_report)
    by_image: dict = defaultdict(list)
    for row in rows:
        by_image[row.image].append(row)

    payload = evaluate(dict(by_image), predictions, config["evaluation"]["iou_match"], operating)
    report = EvaluationReport(name=f"{args.population} ({', '.join(split_names)})")
    report.add(args.population, payload)

    text = report_text(report)
    print(text)
    print(f"images scored: {len(by_image)}   plates: {len(rows)}")
    if len(rows) < 30:
        print(
            "\nNOTE: fewer than 30 plates. Every figure above is an indication with a wide\n"
            "interval, not a measurement. Report the interval, never the point estimate alone."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwritten: {args.out}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
