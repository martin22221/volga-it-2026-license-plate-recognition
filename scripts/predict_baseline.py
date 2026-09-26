#!/usr/bin/env python
"""Run the exported Baseline V1 over one Dataset V1 population and save predictions.

The output is the predictions file ``scripts/evaluate_baseline.py`` scores.
Inference is the deployed path -- ONNX Runtime graphs, NumPy postprocessing,
:mod:`src.recognition` -- so what is scored is what would be submitted.

Each detection records the detector box and score, the recogniser's raw and
masked strings, per-character scores, plate type, the validator's verdict and
the final confidence computed exactly as :mod:`src.pipeline` does. Detections
are kept down to a low floor so average precision can be computed; only those
at or above the model's operating confidence are read by the recogniser.

``--population real_holdout`` requires ``--final-report "<why>"``: the holdout
is spent the first time it is looked at.

Usage::

    python scripts/predict_baseline.py --models models/baseline_v1 \\
        --population synthetic_val --out runs/eval/<run>/synthetic_val.predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import INVALID_FORMAT_PENALTY  # noqa: E402
from src.training.data import load_samples, load_split  # noqa: E402
from src.validator import validate_plate  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
AP_FLOOR = 0.01


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--final-report", metavar="WHY", default="")
    parser.add_argument("--limit", type=int, default=0, help="first N images only (smoke checks)")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.population == "real_holdout" and not args.final_report.strip():
        print("refusing: real_holdout needs --final-report \"<why>\"; it is a once-only measurement",
              file=sys.stderr)
        return 2
    mapping = config["evaluation"]["populations"]
    if args.population not in mapping:
        print(f"error: unknown population {args.population!r}", file=sys.stderr)
        return 1
    splits = mapping[args.population]
    splits = splits if isinstance(splits, list) else [splits]

    from PIL import Image

    from src.onnx_backend import load_models

    detector, recognizer = load_models(args.models)
    operating = detector.confidence
    samples = load_samples()
    images = sorted({r.image for name in splits for r in load_split(name, samples=samples, unlock_holdout=args.final_report)})
    if args.limit:
        images = images[: args.limit]

    out = []
    for image_name in images:
        with Image.open(REPO_ROOT / "dataset" / image_name) as handle:
            image = handle.convert("RGB")
        detector.confidence = AP_FLOOR
        found = detector.detect_image(image)
        detector.confidence = operating
        readable = [(b, s) for b, s in found if s >= operating]
        readings = recognizer.read_boxes(image, [b for b, _ in readable]) if readable else []
        detections = []
        for (box, score), reading in zip(readable, readings):
            expected = reading.plate_type if reading.plate_type != "other" else None
            verdict = validate_plate(reading.text, expected)
            confidence = score * reading.type_confidence * reading.mean_char_score
            if not verdict.is_valid:
                confidence *= INVALID_FORMAT_PENALTY
            detections.append({
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "confidence": score,
                "plate_type": reading.plate_type,
                "type_confidence": round(reading.type_confidence, 4),
                "plate_num": verdict.normalized,
                "raw_text": reading.raw_text,
                "char_scores": [round(c, 4) for c in reading.char_scores],
                "format_valid": verdict.is_valid,
                "format_reason": verdict.reason,
                "final_confidence": round(min(1.0, max(0.0, confidence)), 4),
                "two_line": reading.two_line,
            })
        below = [{"bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "confidence": s, "below_operating": True}
                 for b, s in found if s < operating]
        out.append({"image": image_name, "detections": detections + below})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "population": args.population,
        "splits": splits,
        "models": {"detector": detector.meta, "recognizer": recognizer.meta},
        "operating_confidence": operating,
        "ap_floor": AP_FLOOR,
        "predictions": out,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"{len(out)} image(s) -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
