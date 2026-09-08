# Volga-IT 2026 — Offline Russian License Plate Recognition

Semifinal project for the discipline *Artificial Intelligence and Data Analysis*.

## Purpose

Detect and recognise Russian vehicle license plates in still images **fully
offline** — no APIs, no cloud services, no network access at inference time —
and write the results to a semicolon-separated CSV.

Target plate classes:

| class    | description                                     |
|----------|-------------------------------------------------|
| `type1`  | standard white one-line Russian plate           |
| `type1a` | white two-line / square Russian plate           |
| `type1b` | yellow one-line passenger transport plate       |
| `other`  | non-target plates and negative examples         |

## Current status

**Skeleton + working end-to-end plumbing. No ML models yet.**

* The full chain runs: image discovery → detector → classifier → OCR →
  format validator → confidence → CSV.
* `detector`, `classifier` and `ocr` are **placeholders**. The detector
  reports no detections, so a run currently produces a valid CSV containing
  only the header. This is deliberate: an untrained pipeline must never
  fabricate plate numbers.
* The **format validator is real** and fully tested.
* No models are trained, downloaded or bundled; no synthetic dataset has been
  generated.

## Folder structure

```
volga-it-2026-license-plate-recognition/
  src/
    __init__.py
    pipeline.py      # orchestration, image discovery, timing, stats
    detector.py      # Detector protocol + PlaceholderDetector
    classifier.py    # PlateType enum, PlateClassifier protocol + placeholder
    ocr.py           # OcrEngine protocol + PlaceholderOcr
    validator.py     # real Russian plate format validator
    csv_writer.py    # PlateRecord + submission CSV writer
  training/          # (empty) training scripts for detector/classifier/OCR
  generator/         # (empty) synthetic plate dataset generator
  scripts/           # (empty) helper/utility scripts
  configs/           # (empty) model and run configuration files
  models/            # (empty) trained weights, git-ignored
  data/
    official_debug/  # official debug images from the organisers
    outputs/         # generated CSVs
  tests/             # pytest suite
  docs/              # notes and design documents
  run.py             # CLI entry point
  requirements.txt
  .gitignore
  README.md
```

## Environment setup

Python 3.10+ (developed on 3.12). The pipeline itself needs only the standard
library; `pytest` is required to run the tests.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run command

```bash
python run.py --input <image_directory> --output <output_csv>
```

Example:

```bash
python run.py --input data/official_debug --output data/outputs/result.csv
```

Optional flags: `--no-recursive`, `--min-confidence <float>`, `--no-header`,
`--log-level DEBUG`, `--log-file <path>`.

Images are read from the directory (recursively by default); `.jpg`, `.jpeg`
and `.png` are accepted, case-insensitively.

### Output format

UTF-8, semicolon separated, one row per recognised plate:

```
image;plate_num;plate_type;confidence
car_0001.jpg;A123BC77;type1;0.910
```

* `image` — path of the file relative to `--input` (the file name for a flat
  directory).
* `plate_type` — one of `type1`, `type1a`, `type1b`, `other`.
* `confidence` — float in `[0.0, 1.0]`, the product of the detector,
  classifier and OCR confidences, halved when the text fails format
  validation.
* Images with no detection produce **no row**.

## Plate format validator

Implemented in `src/validator.py`:

```
[LETTER][DIGIT][DIGIT][DIGIT][LETTER][LETTER][REGION]
```

* Allowed letters: `A B E K M H O P C T Y X`.
* `REGION` is 2 or 3 digits; a 3-digit region must start with `1`, `2` or `7`;
  an all-zero region is rejected.
* Normalisation upper-cases the text, removes spaces/hyphens/underscores and
  folds the twelve Cyrillic look-alike letters (А, В, Е, К, М, Н, О, Р, С, Т,
  У, Х) onto their Latin forms. No other substitutions are made — OCR
  confusion handling (`0`↔`O`, `8`↔`B`, …) is deliberately **not** implemented
  yet.

## Tests

```bash
python -m pytest -q
```

Covers valid plates, invalid letters, malformed numbers, 2- and 3-digit
regions, normalisation, CSV generation, image discovery, pipeline wiring and
the CLI.

## Replacing the placeholders

Each stage is a `typing.Protocol`; implement it and inject it into `Pipeline`:

```python
from src.pipeline import Pipeline

pipeline = Pipeline(detector=MyYoloDetector(...), classifier=..., ocr=...)
```

`src/pipeline.py` does not need to change.

## Current limitations

* No detection, classification or text recognition happens — every run yields
  an empty result set.
* Image pixels are never decoded; the placeholder stages only receive the
  file path. Real stages will add an image-loading dependency
  (OpenCV/Pillow) to `requirements.txt`.
* Confidence combination is a simple product and will be recalibrated once
  real models exist.
* No OCR error correction, no plate-type-specific post-processing, no
  duplicate-detection/NMS logic yet.

## Next steps

1. Synthetic plate generator (`generator/`).
2. Detector training (`training/`) and integration.
3. Plate type classifier for `type1` / `type1a` / `type1b` / `other`.
4. OCR model + calibrated confidences.
5. Evaluation scripts against `data/official_debug`.

**Actual models will be added later — nothing in this repository is trained
yet.**
