# Data source registry

Every real image in `dataset/images/real/` traces back to a row in the table
below through the `source` column of `dataset/meta.csv`.

Rows are added only after someone has actually read that source's license —
never from memory, never from an assumption about what a site "probably"
allows. A registered source is a *candidate cleared on licensing*; it is not
imported until its content review passes too.

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

For a *third-party dataset* rather than a single image, the audit and review
steps that come before registration are in
[`external_dataset_workflow.md`](external_dataset_workflow.md).

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
| `autoria_numberplate_options` | AUTO.RIA Numberplate Options Dataset | Public dataset published by ARS Online OU; test split extracted locally for audit | CC BY 4.0 | yes | yes | yes | yes | 2026-09-10 | Copyright 2018–2024 ARS Online OU. Licence evidence: `license.txt` in the source repository (see below). **Licensing cleared. Rare-class review completed 2026-09-13 for the generated candidate sets: 0 confirmed `type1a`, 0 confirmed `type1b`.** Candidate for generic plate detection / robustness only; **not approved wholesale for OCR or `type1` training** — country/type filtering still required. Not imported. See below and `data/audits/autoria_test_review/`. |
| `roboflow_two_line_russian_license_plates` | two-line-russian-license-plates (Roboflow Universe) | https://universe.roboflow.com/fverwfgerwf/two-line-russian-license-plates/dataset/1 | CC BY 4.0 *as declared by the project* | **no** | unclear | unclear | yes | 2026-09-12 | **REJECTED_FOR_SUBMISSION_PROVENANCE** (2026-09-12). Project declares CC BY 4.0, but no evidence establishes the provenance chain of the underlying 27 photographs. Not approved for the competition dataset or for training. Files kept locally for reference/visual review only. Reconsiderable on new evidence. |

### `roboflow_two_line_russian_license_plates` — licensing detail

**Status: `REJECTED_FOR_SUBMISSION_PROVENANCE`** — decided 2026-09-12.

Not approved for inclusion in the competition dataset, and not approved for
training. **The source is retained, not deleted**, and the decision may be
revisited if reliable provenance evidence appears.

#### Decision

| | |
| --- | --- |
| Decision | `REJECTED_FOR_SUBMISSION_PROVENANCE` |
| Decided | 2026-09-12 |
| Applies to | Inclusion in the submitted dataset **and** training of any kind |
| Does **not** apply to | Keeping the downloaded files locally; visual and reference review |
| Reversible | Yes, on reliable original-provenance evidence |

**Reasons of record:**

1. The Roboflow project declares **CC BY 4.0**.
2. The 27 images are publicly presented by that project.
3. **No sufficient evidence was found** establishing the original provenance,
   ownership or licensing chain of the underlying photographs.
4. Therefore the source is **not approved** for the competition dataset or for
   training at this time.

We apply a stricter standard than the platform's own declaration because the
submitted dataset may be redistributed, and we must be able to *justify rights
to the underlying images* — not merely point at a licence field someone else
filled in. A declaration is not a provenance chain.

#### What this means in practice

- The downloaded files **stay where they are**, outside `dataset/`. Nothing is
  deleted.
- They remain available for **visual and reference review only** — the audit
  and contact sheet in `data/audits/roboflow_type1a/` and
  `data/review/roboflow_type1a/` are kept and still useful.
- **No training**, no copying into the repository, and the images are **not**
  competition training data.
- The rejection is enforced mechanically: `REJECTED_SOURCES` in
  `src/dataset_meta.py` makes any `meta.csv` row citing this `source` a
  validation **error**, regardless of what its `license` column says — because
  the objection is to provenance, which no licence string can cure.

#### What would reopen it

- The upstream origin of the photographs identified, with its own licence; or
- confirmation from the uploader that they created the images themselves.

Either would be recorded here with its date, and the row moved out of
`REJECTED_SOURCES` in the same commit.

#### Evidence gathered (retained for the record)

| Field | Value |
| --- | --- |
| Platform | Roboflow Universe |
| Project name | `two-line-russian-license-plates` |
| Workspace | `fverwfgerwf` |
| Version | v1, exported 2026-02-10 |
| Source URL | https://universe.roboflow.com/fverwfgerwf/two-line-russian-license-plates/dataset/1 |
| Declared licence | CC BY 4.0 |
| Local licence evidence | `data.yaml` (`license: CC BY 4.0`) and `README.dataset.txt` (`Provided by a Roboflow user` / `License: CC BY 4.0`) |
| Attribution requirement | CC BY 4.0 requires attribution; no author name is given anywhere in the download — only the workspace handle `fverwfgerwf` |
| Provenance status | **Undocumented, with contrary evidence** |
| Redistribution status | **Not established** — the basis of the rejection |

**Why this is not cleared despite a CC BY 4.0 declaration.**

The declaration is real and consistently stated in two files. But a licence is
only as good as the licensor's right to grant it, and here there is no evidence
the uploader held any rights in the images:

- The uploader is an anonymous handle (`fverwfgerwf`) and the README says only
  *"Provided by a Roboflow user"*. No photographer, no collection, no upstream
  source is named.
- **All 27 files carry the signature of screen captures, not photographs.**
  Every filename embeds a capture timestamp, and all 27 fall inside an
  **18½-minute window on 2026-02-10 (12:16:44 – 12:35:21)**. Every original was
  **PNG** (`_png.rf.` in the Roboflow-rewritten names) — the format a screenshot
  tool produces, not a camera. Image aspect ratios are arbitrary (0.75 … 2.52),
  with only 5 of 27 near a standard camera ratio, which is what arbitrary crops
  of a screen look like.

The most plausible reading is that someone spent twenty minutes screenshotting
Russian two-line plates from a website and uploaded the crops. If so, the
uploader could not grant CC BY 4.0 over them, and the declaration is void
regardless of good faith.

This is exactly the trap this document already warns about: *a
permissive-looking aggregate licence does not cover its contents*, and *if the
images' own provenance is not documented, the dataset is unclear*. Our rule for
unclear provenance is to stop.

Twenty-seven images were never worth a provenance risk in a dataset we must
publish under CC BY 4.0, which is where the decision above landed. The audit
and contact-sheet work is kept: knowing whether these are usable `type1a`
examples costs nothing now and would matter immediately if provenance were
ever established.

### `autoria_numberplate_options` — licensing detail

**Copyright holder:** ARS Online OU
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Evidence:** `license.txt` in the source repository, which states:

> AUTO.RIA Numberplate Options Dataset
> Copyright 2018-2024 by ARS Online OU
>
> AUTO.RIA Numberplate Dataset is licensed under a
> Creative Commons Attribution 4.0 International License.

**Gate result: passes.** CC BY 4.0 permits redistribution, commercial use and
modification, with attribution as the only condition — exactly the terms our
own dataset is published under, so this material can be carried through a
CC BY 4.0 release without relicensing trouble.

**Attribution requirement.** Every redistribution of our dataset that includes
these images must credit:

> AUTO.RIA Numberplate Options Dataset © 2018–2024 ARS Online OU, licensed
> under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

This line goes into `dataset/LICENSE` at the moment the first image is
imported, not before.

**Two caveats to keep on the record:**

1. **The Hugging Face mirror's repository-level metadata may declare a
   different license from `license.txt`.** Mirror metadata describes the
   *repository*; `license.txt` is the upstream rights holder's own statement
   about the *dataset*, and that is what we rely on. If the two ever conflict
   in a way that matters, the upstream statement governs and the discrepancy
   is noted here.
2. **`license.txt` is not present in the extracted test split.** The audit of
   `C:\Users\User\Downloads\test\test` found no LICENSE, README or config file
   — the split ships images and labels only. The licence evidence comes from
   the source repository, not from our local copy. **When importing, archive a
   verbatim copy of `license.txt` alongside the imported images**, so our
   provenance chain does not depend on a file we never kept.

#### Content review status — recorded 2026-09-13

| | |
| --- | --- |
| Licence / provenance | **Cleared** on the CC BY 4.0 evidence above |
| Rare-class review | **Completed** for the generated candidate sets (test split only) |
| Confirmed `type1a` from review | **0** |
| Confirmed `type1b` from review | **0** |
| Country / type classification of the whole split | **Not done** |
| Import status | **Not imported** |

**What was reviewed.** A human reviewed the rare-class contact sheets built by
`scripts/build_rare_review.py`: the 89 shape candidates (46
`square_or_two_line_candidate` + 43 `ambiguous_shape`, drawn from the seeded
200-image sample) and the 54 colour candidates (14 over the yellow threshold +
40 most-yellow shown anyway, scanned across all 2564 test images).

**Findings.**

- No reviewed yellow candidate was confirmed as `type1b`.
- No reviewed square/ambiguous candidate was confirmed as `type1a`.
- Most reviewed candidates appeared to be ordinary one-line plates whose
  apparent shape or colour came from perspective, cropping, lighting, dirt or a
  colour cast.

The verdict was reached at the level of the candidate sets. Per-row verdicts
were not recorded, so `human_plate_type` in `rare_candidates.csv` is
deliberately left empty rather than back-filled.

**What this does *not* establish.** It does **not** establish that the 2564
test images are Russian `type1` plates. AUTO.RIA is a Ukrainian marketplace,
and the split has not been classified by country or plate type. Images outside
the candidate sets were not reviewed individually.

**Permitted next uses (each still requires its own step before import):**

| Use | Status |
| --- | --- |
| Generic plate detection / robustness | Plausible candidate |
| `type1` / OCR training | **Not approved wholesale.** Only after country/type filtering |
| Foreign / out-of-scope negatives for `other` | To be evaluated separately |
| `type1a` / `type1b` | **Not a useful source.** None found |

Faces must still be checked and blurred before any image is imported.

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
- **A rejected source is kept in this table, never deleted**, so the decision
  and its reasons survive and nobody re-evaluates it from scratch. Add its
  `source_id` to `REJECTED_SOURCES` in `src/dataset_meta.py` in the same
  commit, so the validator refuses any `meta.csv` row citing it.
- A rejection on **provenance** is not cured by a licence string. Record the
  decision, the date, what evidence was missing, and what would reopen it.

## Pending

- The organizer's official debug set is unavailable (ownCloud share empty as of
  2026-09-09); we have emailed them. It is **not** a dataset source and will not
  be added here when it arrives.
