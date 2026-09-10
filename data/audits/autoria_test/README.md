# AUTO.RIA test split — audit record

Audit of the **TEST split only** of the AUTO.RIA Russian license plate
detection dataset (candidate `source_id`: `autoria_numberplate_options`), run
on 2026-09-10.

- Audited directory: `C:\Users\User\Downloads\test\test` (external, outside
  this repository)
- Command: `scripts/audit_external_dataset.py ... --json-detail summary`
- `audit.txt` — the human-readable report
- `audit.json` — statistics and findings only. The per-image and
  per-annotation arrays are deliberately omitted: at `--json-detail full` they
  reproduce every bounding box, which would copy the dataset's annotations into
  this repository before the source is approved.

**Nothing was imported.** No file under `dataset/` was created or changed, and
no file in the audited directory was modified.

## Headline numbers

| Metric | Value |
| --- | --- |
| Images | 2564 (1820 `.jpg`, 744 `.bmp` by extension) |
| Actual formats | 2558 JPEG, 4 PNG, 2 BMP |
| Extension mismatches | 742 — `.bmp` files that are really JPEG (738) or PNG (4) |
| Corrupt / unreadable | 0 |
| Annotation files | 2564, all valid YOLO |
| Annotation boxes | 2685, all class `0` |
| Invalid annotation rows | 0 |
| Images without annotations | 0 |
| Annotations without images | 0 |
| Identical images (SHA-256) | 9 groups, 20 files, 11 redundant copies |
| README / LICENSE / config | none in the extracted split (see licensing below) |

## Licensing — resolved 2026-09-10

**The source is licensed CC BY 4.0 and passes our redistribution gate.**

- Copyright 2018–2024 **ARS Online OU**
- **CC BY 4.0**, per `license.txt` in the source repository
- Attribution required; the exact credit line is in
  [`../../../docs/data_sources.md`](../../../docs/data_sources.md)

An earlier note in this file described the source as effectively unlicensed.
That was wrong and has been corrected. What the audit actually established is
narrower and still worth knowing: **`license.txt` is not present in the
extracted split**, which contains images and labels only. The licence evidence
lives in the source repository. When importing, archive a copy of `license.txt`
with the images so the provenance chain does not rely on a file we never kept.

The Hugging Face mirror's repository-level metadata may declare a different
license; that describes the mirror repository, while `license.txt` is the
rights holder's statement about the dataset itself.

## Status

**Licensing cleared. Content review open. Not approved, not imported.**

The remaining blocker is the plate-type question: the single class `0` is
unnamed, nothing in the data says what it marks, and it certainly does not
encode our `type1a` / `type1b` distinction. A human visual review is under way
— see [`../autoria_test_review/`](../autoria_test_review/).

Next steps are in
[`../../../docs/external_dataset_workflow.md`](../../../docs/external_dataset_workflow.md).
