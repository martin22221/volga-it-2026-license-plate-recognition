# Roboflow `two-line-russian-license-plates` — audit and review record

Candidate source for the rare `type1a` class (white square / two-line Russian
plates). Audited 2026-09-12.

**Status: `REJECTED_FOR_SUBMISSION_PROVENANCE` (2026-09-12). Not imported,
not trained on, nothing human-confirmed. Files retained for reference.**

## Identification

Selected by contents (`data.yaml` + `README.dataset.txt` + `README.roboflow.txt`
+ `train/`), not by name. Exactly one directory in `C:\Users\User\Downloads`
matched, so there was no ambiguity:

```
C:\Users\User\Downloads\two-line-russian-license-plates.v1i.yolov8
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
└── train/
    ├── images/   27 .jpg
    └── labels/   27 .txt
```

| Field | Value |
| --- | --- |
| Platform | Roboflow Universe |
| Project | `two-line-russian-license-plates` |
| Workspace | `fverwfgerwf` |
| Version | v1, exported 2026-02-10 12:17 GMT |
| URL | https://universe.roboflow.com/fverwfgerwf/two-line-russian-license-plates/dataset/1 |
| Declared licence | CC BY 4.0 (in both `data.yaml` and `README.dataset.txt`) |
| Attribution | Required; no author named, only the workspace handle |
| Format | YOLOv8, single class `license-plate` |
| README image count | 27 — matches what is on disk |

`data.yaml` also lists `val: ../valid/images` and `test: ../test/images`, which
do not exist. That is a Roboflow export quirk when every image is assigned to
train; it is not missing data.

## Technical audit

| Metric | Value |
| --- | --- |
| Images | **27** (all JPEG, all readable) |
| Annotation files | **27** |
| Annotation boxes | 27 (one per image) |
| Invalid YOLO rows | **0** |
| Boxes outside [0, 1] | **0** |
| Empty annotation files | 0 |
| Images without labels | 0 |
| Labels without images | 0 |
| Corrupt images | 0 |
| Extension/format mismatches | 0 |
| **Duplicate images (SHA-256)** | **0** |
| Duplicate filenames | 0 |
| Declared classes | 1 — `license-plate`, 27 boxes |
| Image dimensions | 469×302 to 1908×1371 (median 932×705), 27 distinct |

Technically this is a clean little dataset: perfect pairing, no corruption, no
duplicates, no invalid coordinates.

### Box geometry is unusually consistent

All 27 box aspect ratios fall between **0.87 and 1.85**, median **1.64** —
a tight unimodal cluster essentially on the Russian square/two-line nominal of
**1.71**. For contrast, AUTO.RIA's plates form a continuous smear from 1.11 to
5.73.

This is the strongest geometric signal we have seen for genuine two-line
plates. It is still **not confirmation**: a square crop, or a one-line plate
photographed at a steep angle, lands in the same place. That is what the visual
review is for.

## Licensing — `REJECTED_FOR_SUBMISSION_PROVENANCE`

**Decided 2026-09-12 after external provenance research.** The Roboflow project
declares CC BY 4.0 and publicly presents the 27 images, but no sufficient
evidence establishes the provenance, ownership or licensing chain of the
underlying photographs. The source is therefore **not approved** for the
competition dataset or for training.

We hold to a stricter standard than the platform's declaration because the
submitted dataset may be redistributed and we must be able to justify rights to
the underlying images. A declaration is not a provenance chain.

**The files are retained, not deleted.** They stay outside `dataset/` and
remain available for visual and reference review only — no training, no copying
into the repository, not competition training data. The decision is reversible
if reliable original-provenance evidence is found.

The rejection is enforced in code: `REJECTED_SOURCES` in `src/dataset_meta.py`
makes any `meta.csv` row citing this source a validation error, whatever its
`license` column says.

### The evidence behind it

A licence is only as good as the licensor's right to grant it, and there is no
evidence of that here:

- The uploader is an anonymous handle; the README says only *"Provided by a
  Roboflow user"*. No photographer, collection or upstream source is named.
- **All 27 files look like screen captures rather than photographs:**
  - every filename embeds a capture timestamp, and all 27 fall within an
    **18½-minute window on 2026-02-10 (12:16:44 – 12:35:21)**;
  - every original was **PNG** (`_png.rf.` in the Roboflow-rewritten names) —
    what a screenshot tool emits, not a camera;
  - image aspect ratios are arbitrary (0.75 … 2.52), only 5 of 27 near a
    standard camera ratio — the shape of arbitrary crops of a screen.

Twenty minutes of screenshotting plates from a website would produce exactly
this. If that is what happened, the uploader had no rights to grant and the
CC BY 4.0 declaration is void however sincerely it was made.

Full reasoning and what would resolve it:
[`../../../docs/data_sources.md`](../../../docs/data_sources.md).

This is what the decision rests on. Reopening it would need the upstream origin
of the photographs identified with its own licence, or confirmation from the
uploader that they created the images themselves.

## Human review

Generated for **all 27 images** — no sampling:

```bash
python scripts/build_type1a_review.py \
    "C:\Users\User\Downloads\two-line-russian-license-plates.v1i.yolov8" \
    --source-id roboflow_two_line_russian_license_plates \
    --out data/review/roboflow_type1a
```

| Artefact | Path | Committed |
| --- | --- | --- |
| Contact sheet | `data/review/roboflow_type1a/contact_sheet.html` | no — absolute local paths |
| Review CSV | `data/review/roboflow_type1a/review.csv` | yes |
| Statistics | `data/review/roboflow_type1a/statistics.txt` | yes |

Each card shows the filename, the full vehicle image with the annotation box
drawn over it, a large close-up of that box (560 px wide), the YOLO class, and
a blank verdict area listing the `type1a` criteria and the conditions to flag.

The review work is **kept** despite the rejection: it costs nothing now and
would matter immediately if provenance were ever established.

All 27 rows carry `candidate_plate_type = type1a_candidate`.
`human_plate_type`, `russian_plate`, `two_line_physical_plate` and
`visible_face` are **empty**.

**Human-confirmed `type1a`: 0.**

## A bug this dataset exposed in our auditor

The first audit reported *"metadata files: 0"* and 20 invalid annotation rows on
a dataset whose READMEs carry the licence — it had classified
`README.dataset.txt` and `README.roboflow.txt` as broken YOLO label files,
because the classifier used `Path.stem`, which reads `README.dataset`, not
`readme`. Since the licence review depends on the audit surfacing exactly these
files, that was worth fixing properly rather than working around.

`src/external_audit.py` now classifies on the leading name segment, so
multi-suffixed paperwork (`README.dataset.txt`, `LICENSE.md`) is recognised
while a label genuinely named `license-plates.txt` still is not. Both cases are
covered by tests.
