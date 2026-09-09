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
    dataset_meta.py  # meta.csv schema + dataset validation
    external_audit.py# read-only audit of third-party datasets
  training/          # (empty) training scripts for detector/classifier/OCR
  dataset/           # the training dataset we build ourselves
    images/real/     # collected photographs, git-ignored
    images/synthetic/# generated images, git-ignored
    labels/          # derived per-image labels, git-ignored
    generator/       # generator configs, fonts, templates
    meta.csv         # the annotation source of truth
    README.md        # format and legal rules
    LICENSE          # dataset license (CC BY 4.0)
  scripts/
    validate_dataset_local.py   # dataset checker, prints a report
    audit_external_dataset.py   # inspect a third-party dataset, imports nothing
  configs/           # (empty) model and run configuration files
  models/            # (empty) trained weights, git-ignored
  data/
    official_debug/  # official debug images from the organisers
    external/        # staging for third-party datasets, git-ignored
    outputs/         # generated CSVs
  tests/             # pytest suite
  docs/
    dataset_strategy.md   # what to collect and how much
    data_sources.md       # source registry (licenses, provenance)
    annotation_guide.md   # how to annotate
    external_dataset_workflow.md  # audit -> review -> approve -> import
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

## Training dataset

The dataset we build ourselves lives in `dataset/`. Annotations go in
`dataset/meta.csv` (UTF-8, semicolon separated, one row per *plate*, not per
image); the schema and all validation rules are in `src/dataset_meta.py`.

```bash
python scripts/validate_dataset_local.py
python scripts/validate_dataset_local.py --report reports/dataset.txt --strict
```

Exit code `0` = valid, `1` = validation errors. The report gives image and
annotation totals, real/synthetic split, counts by plate type and by condition,
missing files, duplicates, and source/license coverage.

No images have been collected yet. The dataset is submitted and published under
**CC BY 4.0**, so every real image needs a documented source and a license that
permits redistribution on those terms — NC, ND, SA and unclear licenses are
rejected, and the validator enforces this. The official 30-image debug set is
never copied into `dataset/`. See
[`dataset/README.md`](dataset/README.md),
[`docs/dataset_strategy.md`](docs/dataset_strategy.md),
[`docs/data_sources.md`](docs/data_sources.md) and
[`docs/annotation_guide.md`](docs/annotation_guide.md).

## Auditing an external dataset

Before any third-party dataset is considered, it is inspected in place:

```bash
python scripts/audit_external_dataset.py <directory> --source-id candidate     --json reports/candidate.json --report reports/candidate.txt
```

The tool is **inspection-only**: it never modifies the audited directory and
never imports anything into `dataset/` — it refuses to write even its own
reports into either location. It reports image counts, formats, dimension
statistics, corrupt files, YOLO coordinate validity, class-id counts,
image/annotation pairing, duplicate names, byte-identical images (SHA-256),
CSV headers and any README/LICENSE it finds.

It does not interpret what a class id *means* and does not judge licenses.
Those are human steps: see
[`docs/external_dataset_workflow.md`](docs/external_dataset_workflow.md).

## Tests

```bash
python -m pytest -q
```

Covers valid plates, invalid letters, malformed numbers, 2- and 3-digit
regions, normalisation, CSV generation, image discovery, pipeline wiring,
the CLI, the dataset metadata schema and validator, and the external
dataset audit tool.

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

1. Collect and annotate the real dataset per `docs/dataset_strategy.md`.
2. Synthetic plate generator (`dataset/generator/`).
3. Detector training (`training/`) and integration.
4. Plate type classifier for `type1` / `type1a` / `type1b` / `other`.
5. OCR model + calibrated confidences.
6. Evaluation scripts against `data/official_debug`.

**Actual models will be added later — nothing in this repository is trained
yet.**
