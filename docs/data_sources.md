# Data source registry

Every real image in `dataset/images/real/` traces back to a row in the table
below through the `source` column of `dataset/meta.csv`.

**The table is intentionally empty.** No source has been checked yet. Rows are
added only after someone has actually read that source's license — never from
memory, never from an assumption about what a site "probably" allows.

## How to use this file

1. Before collecting anything from a source, add a row here.
2. Fill in `license` from the source's own terms, quoting the version
   (`CC BY 4.0`, not "Creative Commons").
3. Set `date_checked` to the day you read the terms. Re-check anything older
   than six months.
4. Use the `source_id` as the `source` value in `meta.csv`.
5. If a column cannot be answered from the source's own text, the answer is
   **no**, not "probably".

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
| `commercial_use_allowed` | yes / no / unclear. |
| `modification_allowed` | yes / no / unclear. Cropping, resizing and face blurring all count as modification. |
| `attribution_required` | yes / no. If yes, the exact required credit line goes in `notes`. |
| `date_checked` | ISO date the license text was actually read. |
| `notes` | Required credit line, restrictions, caveats, contact for takedown. |

## Registry

| source_id | source_name | source_url_or_description | license | commercial_use_allowed | modification_allowed | attribution_required | date_checked | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| _(none registered yet)_ | | | | | | | | |

## Rules

- A source whose terms are **unclear** is not used until clarified. "Unclear"
  is a reason to stop, not a reason to proceed carefully.
- **Closed or private sources are not used.** Neither is any material obtained
  by bypassing a paywall, login wall, rate limit, access restriction or
  `robots.txt`.
- **Our own photographs** are registered like anything else, with
  `license` = "own work" and a note on where they were taken. Photograph from
  publicly accessible space only.
- **Faces are blurred or covered** before the image enters the dataset wherever
  individuals are identifiable and the license or applicable privacy rules
  require it.
- **The official 30-image debug set is never registered here** and never enters
  `dataset/`. It is evaluation material and stays in `data/official_debug/`.
- A source removed from this table means its images must be removed from
  `dataset/` in the same commit.

## Pending

- The organizer's official debug set is unavailable (ownCloud share empty as of
  2026-09-09); we have emailed them. It is **not** a dataset source and will not
  be added here when it arrives.
