# External dataset workflow

How a third-party dataset gets from "someone mentioned it" to "its images are
in `dataset/`". Every arrow that changes our repository is crossed by a human,
deliberately.

Related: [`data_sources.md`](data_sources.md) (the registry the approval is
recorded in), [`../dataset/README.md`](../dataset/README.md) (our format and
legal rules), [`annotation_guide.md`](annotation_guide.md).

## The pipeline

```
external source
    |
    v
download to a temporary/external location   <- NEVER into dataset/
    |
    v
scripts/audit_external_dataset.py           <- read-only inspection
    |
    v
human license review                        <- may we redistribute it?
    |
    v
human plate-type review                     <- what do the labels actually mean?
    |
    v
approved  (recorded in docs/data_sources.md)
    |
    v
only then import into our competition dataset
```

The audit tool sits in the middle on purpose. It tells you what a dataset
*contains*; it never decides whether you may *use* it, and it never moves a
file. Both remaining questions are human judgement, and both can sink a
dataset that looks technically perfect.

## Step 1 — Download to a temporary location

Put the archive somewhere outside the repository, or under `data/external/`
(git-ignored). **Never unpack a third-party dataset into `dataset/`.** Once
unreviewed images are mixed into our tree, separating them again is guesswork.

Nothing has been downloaded yet. When the time comes, download only from the
source's own official distribution channel, and never by bypassing a login
wall, paywall or rate limit.

## Step 2 — Audit

```bash
python scripts/audit_external_dataset.py <directory>
python scripts/audit_external_dataset.py <directory> \
    --source-id candidate-name \
    --json  reports/candidate-audit.json \
    --report reports/candidate-audit.txt
```

The tool is **inspection-only**. It does not modify the audited directory, and
it refuses to write its own reports into either the audited directory or
`dataset/` — that refusal is enforced in code, not just documented.

`--source-id` stamps the report with the id you intend to register in
[`data_sources.md`](data_sources.md), so an audit artefact can be traced back
to a source later. It does not register anything by itself.

Supported image formats: **JPEG, PNG and BMP**, identified by magic bytes
rather than by file extension. Anything else is listed under "other files".

### Committing an audit report

Audit reports live in `data/audits/<source>/` and are committed as a record of
what we inspected. Use `--json-detail summary` for anything committed:

```bash
python scripts/audit_external_dataset.py <directory> \
    --source-id candidate \
    --json  data/audits/candidate/audit.json --json-detail summary \
    --report data/audits/candidate/audit.txt
```

The default `full` JSON embeds every parsed bounding box, which is a verbatim
copy of the dataset's annotations — fine locally, not something to commit
before the source is approved. `summary` keeps the statistics and findings and
drops the per-image and per-annotation arrays. The text report never contains
annotation data.

What the report covers:

| Section | Tells you |
| --- | --- |
| Totals | Images, annotation files, boxes, CSVs, paperwork, unknown files. |
| Image formats | JPEG / PNG / BMP split by **actual content**, plus anything unreadable. |
| Image dimensions | min/median/mean/max and the most common resolutions. Tiny images mean tiny plates. |
| Corrupt or unreadable images | Empty files, wrong magic bytes, truncation. Genuine damage only. |
| Extension / format mismatches | Readable files whose name lies about their format, e.g. a `.bmp` that is really a JPEG. Reported separately from corruption, because the file is fine and only its name is wrong. |
| YOLO annotations | Files that parsed, files that are not YOLO at all, and every invalid row with its line number and reason. |
| Classes | Raw class ids with box counts, and any names the dataset itself declared. |
| Image / annotation pairing | Images with no label, labels with no image, ambiguous stems. |
| Duplicates | Repeated file names, and byte-identical images by SHA-256. |
| CSV files | Header and row count of any CSV, schema reported not assumed. |
| Dataset paperwork | README/LICENSE/config files, with a preview of the first few lines. |

Exit codes: `0` the audit ran, `1` the directory could not be audited, `2` with
`--fail-on-findings` when anything needs review.

### What the audit deliberately does not do

- **It does not decode pixels.** Image checks parse headers only, which catches
  empty files, wrong magic bytes and truncated downloads — the realistic
  failure modes for an archive — but not corruption in the middle of a stream.
  The report says so rather than implying a clean bill of health.
- **It does not interpret class ids.** If a dataset ships `names: ['plate']`,
  the report repeats that claim and attributes it to the dataset. It never
  assumes class `0` means a license plate, and it never maps an external class
  onto one of our `plate_type` values.
- **It does not judge licenses.** It surfaces the LICENSE file so a person can
  read it.

## Step 3 — Human license review

Read the actual license text the audit surfaced, then answer the question from
[`data_sources.md`](data_sources.md):

> May this image be redistributed to the public, commercially, in modified
> form, as part of a dataset published under CC BY 4.0, with attribution as the
> only condition?

Accept only an unambiguous yes. **Reject** NonCommercial, NoDerivatives,
ShareAlike, "no redistribution" terms, and anything unclear. A dataset with no
LICENSE file at all cannot be approved — find the license at the source or
reject it.

Two traps worth naming:

- **A permissive-looking aggregate license does not cover its contents.** Many
  dataset repositories license the *compilation* while the images inside carry
  their own, sometimes unknown, terms. If the images' own provenance is not
  documented, the dataset is unclear and therefore rejected.
- **A README claiming images were scraped** is a rejection on its own, whatever
  license the packager attached — they could not grant rights they never had.

## Step 4 — Human plate-type review

A dataset can be perfectly licensed and still useless or harmful to us.

Build a reproducible review sample rather than clicking through folders:

```bash
python scripts/sample_review_set.py <directory> \
    --source-id <id> \
    --csv data/audits/<id>_review/review_sample.csv \
    --contact-sheet data/review/<id>/contact_sheet.html
```

This selects a seeded, deterministic subset (200 images by default), writes a
CSV for you to fill in, and builds an HTML contact sheet that references the
images in place — nothing is copied. Each card shows the frame with the
annotation box overlaid, a close-up of that box, the filename and geometry
flags, with filter buttons per flag. The CSV is committed as a record; the
contact sheet is not, since it embeds absolute local paths.

**The tool does not fill in plate types.** `type1` and `type1b` differ only by
plate colour, and this project decodes no pixels, so every row comes back
`needs_human_review`. The geometry flags — plate size, box aspect ratio, plate
count — exist to prioritise your attention, not to label anything. Treat
`square_or_two_line_candidate` as "look here first": a one-line plate seen at a
steep angle foreshortens towards square.

For the rare classes specifically, `scripts/build_rare_review.py` narrows the
split further: it collects the shape candidates and measures plate colour
across every image, then writes large-preview pages for `type1a` and `type1b`
candidates plus a CSV and statistics.

Plate colour is measured from the JPEG's own DC coefficients
(`src/jpeg_dc.py`) -- no image-decoding dependency, no model, no network. It
samples inside the annotation box and compares against a ring around it, so a
yellow car with a white plate is rejected rather than surfaced, and it abstains
on plates too small to judge. A type is suggested only on a strong reading, and
even then the reviewer decides.

Then open the sheet and check:

- **What do the classes actually contain?** "plate" in someone else's dataset
  may mean any plate of any country. Our `type1a` / `type1b` distinction almost
  certainly does not exist there, so imported labels need re-checking against
  [`annotation_guide.md`](annotation_guide.md), not trusted wholesale.
- **Are these Russian plates?** A dataset of EU or US plates trains the
  detector but actively harms OCR.
- **Is the annotation quality acceptable?** Check the invalid-row list from the
  audit, and eyeball whether boxes are tight.
- **Are faces visible?** They must be blurred or covered before import.
- **Does it overlap the official debug set?** If any image looks like it came
  from the organizers' material, stop.

Record the mapping you decide on — external class id to our `plate_type` — in
the source's `notes` in the registry. It is a decision, so it gets written
down.

## Step 5 — Approve and register

Add the source to [`data_sources.md`](data_sources.md) with all gate columns
answered, the `date_checked`, and the class mapping in `notes`. If the license
is compatible but not yet recognised by our validator, add it to
`REDISTRIBUTABLE_LICENSES` in `src/dataset_meta.py` in the same commit.

A rejected candidate is recorded as rejected rather than deleted, so nobody
re-evaluates it in three weeks.

## Step 6 — Import

Only now do images move. Copy them into `dataset/images/real/<source_id>/`,
blur faces, and write one `meta.csv` row per plate per
[`annotation_guide.md`](annotation_guide.md), with `source` set to the
registered `source_id` and `license` to the exact license.

Then:

```bash
python scripts/validate_dataset_local.py
```

Fix every error before committing. The import is not finished until the
validator exits `0`.

## Status

Nothing has been imported into `dataset/`.

Per-source status lives in [`data_sources.md`](data_sources.md). As of
2026-09-13:

- **`autoria_numberplate_options`**
  - The test split has been audited, and licensing is cleared.
  - The rare-class review is complete and found no `type1a` or `type1b`.
  - The split is not approved wholesale.
- **`roboflow_two_line_russian_license_plates`** was audited and then rejected
  on provenance.
