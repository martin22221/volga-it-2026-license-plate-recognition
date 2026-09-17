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
* The **synthetic plate generator 2.3.0** (`dataset/generator/`) renders
  `type1`, `type1a` and `type1b` plates with exact quad annotations. Its
  audited 12,000-image batch (seed 2026091401) was reviewed and **promoted into
  `dataset/`**: 0 validator errors, 0 QA FAILs, 0 provenance and reproduction
  mismatches. V1 was superseded because its `type1b` plates used the wrong
  character structure, and its output must never enter the dataset. See
  [`dataset/generator/README.md`](dataset/generator/README.md).
* **Real images: none yet.** The real-data stage is set up — targets, split
  policy, source-acceptance rules, staging and intake tooling — but nothing has
  been collected, downloaded or imported. See
  [`docs/real_data_plan.md`](docs/real_data_plan.md) and
  [`docs/real_data_intake.md`](docs/real_data_intake.md).
* No models are trained, downloaded or bundled.

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
    review_sample.py # seeded review sample + contact sheet
    jpeg_dc.py       # baseline-JPEG DC reader (1/8-scale, stdlib only)
    plate_color.py   # conservative plate colour / yellow heuristic
    rare_review.py   # type1a / type1b candidate pages
    real_intake.py   # real-image source records + staged-source audit
    real_splits.py   # deterministic, group-aware train/val/holdout splits
  training/          # (empty) training scripts for detector/classifier/OCR
  dataset/           # the training dataset we build ourselves
    images/real/     # collected photographs, git-ignored
    images/synthetic/# generated images, git-ignored
    labels/          # derived per-image labels, git-ignored
    splits/          # frozen real train/val/holdout manifest
    generator/       # generator configs, fonts, templates
    meta.csv         # the annotation source of truth
    README.md        # format and legal rules
    LICENSE          # dataset license (CC BY 4.0)
  scripts/
    validate_dataset_local.py   # dataset checker, prints a report
    audit_external_dataset.py   # inspect a third-party dataset, imports nothing
    sample_review_set.py        # build a reproducible human-review sample
    build_rare_review.py        # focused type1a / type1b candidate review
    audit_real_source.py        # inspect a staged real source, approves nothing
    plan_real_splits.py         # freeze and audit the real split manifest
  configs/           # (empty) model and run configuration files
  models/            # (empty) trained weights, git-ignored
  data/
    official_debug/  # official debug images from the organisers
    external/        # staging for third-party datasets, git-ignored
    real_staging/    # real-image intake; photos git-ignored, records committed
    synthetic_production/  # sealed generator batches, git-ignored
    outputs/         # generated CSVs
  tests/             # pytest suite
  docs/
    dataset_strategy.md   # what to collect and how much
    data_sources.md       # source registry (licenses, provenance)
    annotation_guide.md   # how to annotate
    real_data_plan.md     # real targets, splits, mixing, readiness checklist
    real_data_source_research.md  # public real-image sources: findings + decisions
    real_data_intake.md   # staging, source acceptance, privacy, workflow
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

Implemented in `src/validator.py`. There are two character structures, per
GOST R 50577-2018 §3.3:

```
type1 / type1a:  [LETTER][DIGIT][DIGIT][DIGIT][LETTER][LETTER][REGION]   A123BC77
type1b:          [LETTER][LETTER][DIGIT][DIGIT][DIGIT][REGION]           AB12377
```

* Allowed letters: `A B E K M H O P C T Y X`.
* Type 1 / 1A `REGION` is 2 or 3 digits; a 3-digit region must start with `1`,
  `2` or `7`; an all-zero region is rejected. Type 1B regions have 2 digits
  (GOST shows 1B only as `MM 000 55`).
* `validate_plate(text)` accepts either structure;
  `validate_plate(text, plate_type)` demands the structure of that type.
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

It currently holds **12,000 synthetic images and no real ones**: `type1` 2,400,
`type1a` 4,200, `type1b` 5,400, from generator 2.3.0 (seed 2026091401). The
image files are not committed — only `meta.csv`, the structure and the docs —
and are rebuilt with the command in [`dataset/README.md`](dataset/README.md).

Real-image work is staged, never collected straight into `dataset/`:

```bash
python scripts/audit_real_source.py data/real_staging/incoming/<source_id>
python scripts/plan_real_splits.py --check
```

The dataset is submitted and published under **CC BY 4.0**, so every real image
needs a documented source and a license that permits redistribution on those
terms — NC, ND, SA and unclear licenses are rejected, and the validator
enforces this. The official 30-image debug set is never copied into `dataset/`.
See [`dataset/README.md`](dataset/README.md),
[`docs/real_data_plan.md`](docs/real_data_plan.md),
[`docs/real_data_intake.md`](docs/real_data_intake.md),
[`docs/dataset_strategy.md`](docs/dataset_strategy.md),
[`docs/data_sources.md`](docs/data_sources.md) and
[`docs/annotation_guide.md`](docs/annotation_guide.md).

## Auditing an external dataset

Before any third-party dataset is considered, it is inspected in place:

```bash
python scripts/audit_external_dataset.py <directory> --source-id candidate \
    --json data/audits/candidate/audit.json --json-detail summary \
    --report data/audits/candidate/audit.txt
```

The tool is **inspection-only**: it never modifies the audited directory and
never imports anything into `dataset/` — it refuses to write even its own
reports into either location. It reports image counts, formats, dimension
statistics, corrupt files, YOLO coordinate validity, class-id counts,
image/annotation pairing, duplicate names, byte-identical images (SHA-256),
CSV headers and any README/LICENSE it finds.

Images are recognised as **JPEG, PNG or BMP by magic bytes**, not by
extension, so a file whose name lies about its format is still measured — and
reported as a naming mismatch rather than as corruption. Use
`--json-detail summary` for any report you intend to commit: the default
`full` JSON embeds the audited dataset's own annotation coordinates.

Completed audits are kept under `data/audits/`.

For the human plate-type review that follows, `scripts/sample_review_set.py`
builds a seeded, reproducible sample plus an HTML contact sheet. It never
guesses a plate type: `type1` and `type1b` differ only by colour, and this
project decodes no pixels, so every row is `needs_human_review` and the
geometry flags exist only to prioritise a reviewer's attention.

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

## Synthetic data

```bash
pip install -r requirements.txt
python -m dataset.generator --output data/synthetic_dev/v2_demo \
    --per-class type1=20,type1a=20,type1b=20 --seed 20260913 --contact-sheet
```

The generator loads no external asset. The font, templates, backgrounds and
vehicles are all drawn in code, so its output is ours to release under CC BY
4.0. Development batches go under `data/synthetic_dev/`, which is git-ignored.
The generator refuses to write into `dataset/` unless asked explicitly.

## Next steps

1. Acquire and annotate real images per `docs/real_data_plan.md` and
   `docs/real_data_intake.md`; rare classes (`type1a`, `type1b`) first.
2. Freeze the real splits and pass the readiness checklist.
3. Detector training (`training/`) and integration.
4. Plate type classifier for `type1` / `type1a` / `type1b` / `other`.
5. OCR model + calibrated confidences.
6. Evaluation scripts against `data/official_debug`.

**Actual models will be added later — nothing in this repository is trained
yet.**
