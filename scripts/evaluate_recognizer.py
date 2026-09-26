#!/usr/bin/env python
"""Score the exported recogniser on ground-truth boxes: type classification and OCR.

This isolates the recogniser from the detector. Each annotated plate's box is
handed to the deployed two-pass read (:func:`src.recognition.read_plates`
through ONNX Runtime), and the result is scored by
:func:`src.training.evaluation.evaluate_readings`:

* type: accuracy, per-class precision/recall/F1, 4x4 confusion, macro average;
* OCR: exact plate-string accuracy and CER on the **raw** decode (before the
  ``#`` mask and before the validator), and separately on the masked string;
* breakdowns by plate type, difficulty (easy/medium/hard), condition tag,
  character height and occlusion where the metadata exists.

``real_holdout`` requires ``--final-report "<why>"``.

Usage::

    python scripts/evaluate_recognizer.py --models models/baseline_v1 \\
        --population synthetic_val --out runs/eval/<run>/recognizer_synthetic_val.json
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

from src.training.data import load_samples, load_split  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--final-report", metavar="WHY", default="")
    parser.add_argument("--rows", action="store_true", help="include one row per plate (always on for real populations)")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.population == "real_holdout" and not args.final_report.strip():
        print("refusing: real_holdout needs --final-report \"<why>\"", file=sys.stderr)
        return 2
    splits = config["evaluation"]["populations"][args.population]
    splits = splits if isinstance(splits, list) else [splits]

    from src.onnx_backend import OnnxRecognizer
    from src.recognition import read_plates
    from src.training.evaluation import evaluate_readings
    from src.training.regions import cut_regions

    recognizer = OnnxRecognizer(args.models / "recognizer.onnx")
    samples = load_samples()
    rows = [r for name in splits for r in load_split(name, samples=samples, unlock_holdout=args.final_report)]
    regions = cut_regions(rows, workers=8)
    readings = []
    for start in range(0, len(rows), 256):
        chunk = regions[start : start + 256]
        readings += read_plates(
            recognizer.run_model, [(r.image(), r.box) for r in chunk], char_threshold=recognizer.char_threshold
        )
    real = any(not r.is_synthetic for r in rows)
    report = evaluate_readings(rows, readings, with_rows=args.rows or real)
    report = {"population": args.population, "splits": splits, "model": recognizer.meta, **report}
    if len(rows) < 30:
        report["note"] = "fewer than 30 plates: every proportion is an indication with a wide interval"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    c, o = report["classification"], report["ocr_raw"]
    print(f"{args.population}: {len(rows)} plates")
    print(f"  type accuracy       {c['accuracy']['value']:.4f}  ci95 {c['accuracy']['ci95']}")
    print(f"  type macro F1       {c['per_class']['macro']['f1']}")
    print(f"  OCR exact (raw)     {o['exact']['value']}  ci95 {o['exact']['ci95']}   CER {o['cer']}")
    print(f"  type+string correct {report['type_and_string_correct']['value']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
