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
| README / LICENSE / config | **none found** |

## Status

**Not approved.** The technical audit is clean, but two blocking questions are
untouched by it:

1. **No LICENSE, README or config file ships with this split.** Licensing must
   be established from the source itself before anything can be imported. Per
   `docs/data_sources.md`, an unclear license is a rejection.
2. **The single class `0` is unnamed.** Nothing in the data says what it marks,
   and it certainly does not encode our `type1a` / `type1b` distinction. A
   human plate-type review is required.

Next steps are in
[`../../../docs/external_dataset_workflow.md`](../../../docs/external_dataset_workflow.md).
