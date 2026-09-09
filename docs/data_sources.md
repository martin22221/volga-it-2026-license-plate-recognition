# Data source registry

Every real image in `dataset/images/real/` traces back to a row in the table
below through the `source` column of `dataset/meta.csv`.

**The table is intentionally empty.** No source has been checked yet. Rows are
added only after someone has actually read that source's license — never from
memory, never from an assumption about what a site "probably" allows.

## The gate: redistribution, not just use

The competition rules state that the submitted dataset is **published under
CC BY 4.0** and may be used and published by the organizers with attribution.
So the question to ask of a source is not "may we use these images?" but:

> May this image be **redistributed to the public**, **commercially**, **in
> modified form** (we crop, resize and blur faces), as part of a dataset
> published under CC BY 4.0, with attribution as the only condition?

A source is accepted **only if the answer is an unambiguous yes**:

```
redistribution_allowed = yes
AND commercial_use_allowed = yes
AND modification_allowed = yes
```

`attribution_required = yes` is fine — CC BY requires attribution anyway; the
exact credit line goes in `notes`.

### Rejected without exception

- **Use-but-no-redistribution** terms. A license that lets us train on images
  but not republish them is useless here, however generous it otherwise is.
- **NonCommercial (NC)** — CC BY permits commercial use; we cannot grant that.
- **NoDerivatives (ND)** — cropping and face blurring are derivatives.
- **ShareAlike (SA)** — permits redistribution but forbids relicensing under
  plain CC BY, which is exactly what publication requires. SA is the trap in
  this list: it looks permissive and is not compatible.
- **Unclear or unverifiable** terms. "Unclear" is a reason to stop, not a
  reason to proceed carefully.

`scripts/validate_dataset_local.py` enforces this: an NC, ND or SA license in
`meta.csv` is a **validation error**, and a license not on the confirmed list
in `src/dataset_meta.py` is a **warning** that a person must resolve before
submission.

## How to use this file

1. Before collecting anything from a source, add a row here.
2. Fill in `license` from the source's own terms, quoting the version
   (`CC BY 4.0`, not "Creative Commons").
3. Answer the redistribution question above. If any of the three gate columns
   is not `yes`, the source is rejected — record it as rejected in `notes`
   rather than deleting the row, so nobody re-evaluates it later.
4. Set `date_checked` to the day you read the terms. Re-check anything older
   than six months.
5. Use the `source_id` as the `source` value in `meta.csv`.
6. If a column cannot be answered from the source's own text, the answer is
   **no**, not "probably".
7. If the license is compatible but not yet recognised by the validator, add it
   to `REDISTRIBUTABLE_LICENSES` in `src/dataset_meta.py` in the same commit,
   with the check recorded here.

### `source_id` convention

Lowercase, hyphenated, stable: `own-photos-ulyanovsk`, `wikimedia-commons`,
`openimages-v7`. Once used in `meta.csv`, an id is never renamed.

### Field meanings

| Column | Meaning |
| --- | --- |
| `source_id` | Short stable identifier, used verbatim in `meta.csv`. |
| `source_name` | Human-readable name of the collection, site or person. |
| `source_url_or_description` | URL, or a description for material not on the web (e.g. "photographed by team member, public street"). |
| `license` | Exact license name and version, or "own work". |
| `redistribution_allowed` | yes / no / unclear. **Gate column.** May the image be republished as part of a public CC BY 4.0 dataset? |
| `commercial_use_allowed` | yes / no / unclear. **Gate column.** |
| `modification_allowed` | yes / no / unclear. **Gate column.** Cropping, resizing and face blurring all count as modification. |
| `attribution_required` | yes / no. If yes, the exact required credit line goes in `notes`. |
| `date_checked` | ISO date the license text was actually read. |
| `notes` | Required credit line, restrictions, caveats, contact for takedown; or the reason a source was rejected. |

## Registry

| source_id | source_name | source_url_or_description | license | redistribution_allowed | commercial_use_allowed | modification_allowed | attribution_required | date_checked | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _(none registered yet)_ | | | | | | | | | |

## Rules

- **Our own photographs** are registered like anything else, with
  `license` = "own work" and a note on where they were taken. Photograph from
  publicly accessible space only. These are the cleanest source available to
  us, since we hold the rights outright and can grant CC BY 4.0 directly.
- **Closed or private sources are not used.** Neither is any material obtained
  by bypassing a paywall, login wall, rate limit, access restriction or
  `robots.txt`.
- **Faces are blurred or covered** before the image enters the dataset wherever
  individuals are identifiable and the license or applicable privacy rules
  require it.
- **Generator assets count as sources.** A font, plate template, texture or
  background image used by the synthetic generator is registered here and must
  pass the same gate — a synthetic image built on a non-redistributable asset
  is not redistributable either.
- **The official 30-image debug set is never registered here** and never enters
  `dataset/`. It is evaluation material and stays in `data/official_debug/`.
- A source removed from this table means its images must be removed from
  `dataset/` in the same commit.

## Pending

- The organizer's official debug set is unavailable (ownCloud share empty as of
  2026-09-09); we have emailed them. It is **not** a dataset source and will not
  be added here when it arrives.
