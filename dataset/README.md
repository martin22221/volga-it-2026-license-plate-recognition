# Dataset

The training dataset we build ourselves for the Volga-IT 2026 semifinal
(Russian license plate detection and recognition).

Nothing in `images/` is committed to git — only the folder structure,
`meta.csv` and the documentation. The dataset is rebuilt from `meta.csv` plus
the source registry in [`../docs/data_sources.md`](../docs/data_sources.md).

## Folder structure

```
dataset/
├── images/
│   ├── real/        photographs collected from documented sources
│   └── synthetic/    images produced by our own generator
├── labels/           per-image label files (optional mirror of meta.csv)
├── generator/        synthetic image generator: configs, fonts, templates
├── meta.csv          the single source of truth for every annotation
├── README.md         this file
└── LICENSE           the license this dataset is released under
```

`meta.csv` is authoritative. `labels/` may hold per-image files in a trainer's
native format (for example one `.txt` per image), but they are *derived* from
`meta.csv` and must be regenerated rather than edited by hand.

## Annotation format

`meta.csv` is UTF-8 and **semicolon separated**, matching the submission CSV
format used by `src/csv_writer.py`. The first line is the header.

**One row is one annotation, not one image.** An image showing three plates
contributes three rows sharing the same `image` value. An image with no plate
at all still gets exactly one row — a *background row*, see below.

Coordinates are in **pixels of the original image**, with the origin at the
top-left corner and the y axis pointing down. They are not normalised.

Each plate is described twice:

- **`bbox_*`** — the axis-aligned bounding box (`x`, `y` of the top-left
  corner, then width and height). Used to train the detector.
- **`quad_*`** — the four plate corners, listed **clockwise starting from the
  top-left corner of the plate itself**. Used to rectify skewed plates before
  OCR. For a plate photographed straight on, the quad is simply the bbox.

The validator checks that the quad winds clockwise and that its extent agrees
with the bbox to within 25 % of a side.

### Background rows

An image deliberately kept as a negative example — no plate on it at all — is
recorded as one row with `plate_type=other`, an empty `plate_num`, and **every**
`bbox_*` and `quad_*` column set to `0`. Geometry checks are skipped for these
rows. This is the only case in which zero geometry is accepted.

## Metadata fields

| Field | Type | Meaning |
| --- | --- | --- |
| `image` | path | Path relative to this directory. Must start with `images/real/` or `images/synthetic/`. |
| `plate_num` | text | Plate characters in Latin transliteration, no spaces or hyphens. `#` stands for one unreadable character. Empty for background rows. |
| `plate_type` | enum | `type1`, `type1a`, `type1b` or `other`. |
| `bbox_x`, `bbox_y` | number | Top-left corner of the bounding box, in pixels. Non-negative. |
| `bbox_w`, `bbox_h` | number | Width and height of the bounding box, in pixels. Strictly positive. |
| `quad_x1..quad_y4` | number | Four plate corners, clockwise from the plate's top-left. Non-negative. |
| `is_vehicle` | bool | Whether the plate is mounted on an actual vehicle in the scene. `false` for a detached plate, a plate on a screen, a poster, or a shop window. |
| `is_synthetic` | bool | Whether the image came from our generator. Must agree with the folder the image is stored in. |
| `source` | text | `source_id` from `docs/data_sources.md`. Required for every real image. |
| `license` | text | The exact license the image is used under, e.g. `CC BY 4.0`. Required for every real image, and must permit redistribution — see Legal and source tracking. |
| `conditions` | tags | Zero or more tags from the controlled list, separated by `\|`. |

Booleans accept `true`/`false`, `1`/`0`, `yes`/`no`, `y`/`n` in any case.

### Plate types

| Value | Description |
| --- | --- |
| `type1` | Standard white one-line passenger car plate. Supported by the pipeline so recognition quality on ordinary plates is preserved, but not a real-data acquisition target for now. |
| `type1a` | **Target.** Square / two-line white plate. |
| `type1b` | **Target.** Yellow one-line plate for passenger transport and taxis. |
| `other` | Any other plate (trailer, motorcycle, military, diplomatic, foreign, transit), an unrecognisable plate, or a background image with no plate. |

### Conditions

The controlled list — anything else is a validation error:

`day`, `night`, `rain`, `snow`, `dirt`, `glare`, `motion_blur`, `angle`

Tag what is actually visible, not what you assume. `day` and `night` are
mutually exclusive. `angle` means the plate is noticeably rotated or viewed
obliquely rather than photographed straight on.

## Validating

```bash
python scripts/validate_dataset_local.py
python scripts/validate_dataset_local.py --report reports/dataset.txt
python scripts/validate_dataset_local.py --strict   # warnings fail too
```

Exit code `0` means valid, `1` means validation errors. Run it before every
commit that touches `meta.csv`.

## Legal and source tracking

These rules are not optional. The competition rules state that the submitted
dataset is **published under CC BY 4.0** and may be used and published by the
organizers with attribution. Everything here follows from that.

Our own contributions — the annotations, the synthetic images, the generator,
and this documentation — are released under CC BY 4.0. That grant covers only
what we made; it does **not** and cannot alter the terms of any third-party
image. See [`LICENSE`](LICENSE).

- **The test for a third-party image is redistribution, not use.** Because the
  whole dataset gets published under CC BY 4.0, an image may be included only
  if its own license permits redistribution, commercial use and modification
  (we crop, resize and blur faces), with attribution as the only condition.
  A license that allows us to *train on* an image but not to *republish* it
  disqualifies it.
  - Rejected without exception: **NonCommercial (NC)**, **NoDerivatives (ND)**,
    **ShareAlike (SA)** — SA permits redistribution but forbids the relicensing
    under plain CC BY that publication requires — "no redistribution" terms,
    and anything **unclear or unverifiable**.
  - The validator enforces this: an NC/ND/SA license is an **error**, and an
    unrecognised license is a **warning** for a person to resolve before
    submission.
- **Every real image must have a documented `source` and `license`.** The
  `source` value is a `source_id` registered in
  [`../docs/data_sources.md`](../docs/data_sources.md), which also records the
  redistribution check and its date. The `license` column holds the exact
  license. Blank provenance is a validation warning today and a blocker before
  submission.
- **Faces must be blurred or covered** wherever a person is identifiable and
  the source's license or applicable privacy rules require it. Do this before
  the image enters `images/real/`, never afterwards.
- **The official 30-image debug set must NOT be included in the submitted
  dataset.** It is evaluation material provided by the organizers, it stays in
  `data/official_debug/`, and it is neither copied into `dataset/` nor
  referenced from `meta.csv`. The validator rejects any path outside
  `images/real/` and `images/synthetic/`, which keeps this mistake from
  passing silently.
- **Do not use closed or private sources**, and do not scrape any site in a way
  that bypasses its access restrictions, paywall, login wall, rate limits, or
  `robots.txt`. If getting an image requires circumventing something, the
  image does not go in the dataset.
- **Generator assets count as sources.** A font, plate template, texture or
  background used by the synthetic generator is registered in
  `docs/data_sources.md` and passes the same gate: a synthetic image built on a
  non-redistributable asset is not redistributable either.
- Attribution requirements from a source's license must be recorded in
  `docs/data_sources.md` and reproduced in `LICENSE`.

## Related documents

- [`../docs/dataset_strategy.md`](../docs/dataset_strategy.md) — what to collect and how much.
- [`../docs/data_sources.md`](../docs/data_sources.md) — the source registry.
- [`../docs/annotation_guide.md`](../docs/annotation_guide.md) — how to annotate.
