# Real-data intake

How a real photograph gets from "someone has it" to `dataset/images/real/`,
and what stops it on the way.

**Status, 2026-09-22.** One source has now been through acquisition and intake
and is **waiting for a human review**; nothing has been promoted, so
`dataset/images/real/` is still empty.

| Source | Acquisition | State |
| --- | --- | --- |
| `wikimedia_commons_curated` | #2, online discovery | **22 acquired, 21 annotated, 1 rejected on privacy.** `decision: PENDING` — staged, audited, contact sheets built, awaiting review. See [`acquisition_wikimedia_commons.md`](acquisition_wikimedia_commons.md). |
| `team-capture-2026-09` | #1, targeted `type1a` capture | `decision: PENDING`, **DEFERRED** — no photographs; the participant is in Bulgaria. See [`capture_session_type1a.md`](capture_session_type1a.md). |

Related: [`real_data_plan.md`](real_data_plan.md) (targets, splits, training
mix, readiness), [`real_data_source_research.md`](real_data_source_research.md)
(the source survey and its decisions),
[`capture_session_type1a.md`](capture_session_type1a.md) (the field checklist
for acquisition #1), [`data_sources.md`](data_sources.md) (the registry),
[`annotation_guide.md`](annotation_guide.md) (labelling),
[`../dataset/README.md`](../dataset/README.md) (format and legal rules).

## The one rule

**Unreviewed images never touch `dataset/`.** They live in
`data/real_staging/`, are audited there, and move only when a person decides
they may. The tooling reports; it never approves.

## Staging structure

```
data/real_staging/
  incoming/<source_id>/       images as received, plus the paperwork below
  source_records/<id>.json    provenance record per source   (committed)
  accepted/<source_id>/       reviewed, annotated, cleared for promotion
  rejected/<source_id>/       kept with the reason, never deleted silently
  review/                     contact sheets and human review artefacts
  audits/                     intake audit reports            (committed)
```

**A staging folder mirrors the dataset layout**, so every path is already the
path the image will have after promotion and the same validator runs on both:

```
data/real_staging/incoming/<source_id>/
  images/real/<source_id>/*.jpg    the photographs
  source_record.json               provenance record (copy of source_records/<id>.json)
  meta.csv                         annotations, dataset/meta.csv schema
  groups.csv                       image;group -- the leakage unit per image
  privacy_review.csv               image;faces_present;action;reviewer;date
```

`meta.csv` rows use `images/real/<source_id>/<file>` exactly as
`dataset/meta.csv` will, and `groups.csv` and `privacy_review.csv` use the same
paths. Promotion is then a straight copy: no path rewriting, nothing to get
wrong. This is how the synthetic batches were promoted.

**Git policy.** Photographs, their annotations and their privacy reviews are
**not committed** — they are personal data under review whose rights are not
settled. The source records and the intake audit reports **are** committed:
they are metadata, and they are the paper trail for every accept and reject.
The rules are in `.gitignore`.

## Source acceptance policy

Register a source **before** collecting from it. One record per source, in
`data/real_staging/source_records/<source_id>.json`; the accepted ones are
also added to the registry table in [`data_sources.md`](data_sources.md).

```bash
python scripts/audit_real_source.py --blank-record <source_id> > \
    data/real_staging/source_records/<source_id>.json
```

Every field must be filled in, including with "unclear" — a blank is not an
answer:

| Field | Notes |
| --- | --- |
| `source_id`, `source_name`, `source_reference` | Identity and where it came from (URL, licence document, or "photographed by us at …"). |
| `original_creator`, `upstream_source` | Who actually took the photographs, and where the host got them. "Unknown" is a valid and important answer. |
| `stated_license`, `license_evidence_url` | The exact licence, and where that text can be read. |
| `redistribution_allowed`, `modification_allowed`, `commercial_use_allowed` | `yes` / `no` / `unclear`. |
| `attribution_required`, `attribution_text` | The exact credit line if one is required. |
| `provenance_evidence` | What actually establishes the chain of rights. |
| `privacy_review` | `not_started` / `in_progress` / `completed` / `not_required`. |
| `decision`, `decided_by`, `date_checked`, `notes` | The verdict, who made it, when, and why. |

### The four decisions

| Decision | Meaning |
| --- | --- |
| `PENDING` | Inspected, not decided. The default. |
| `ACCEPT_FOR_SUBMISSION` | May enter `dataset/` and be published under CC BY 4.0. |
| `TRAINING_ONLY_IF_LEGAL` | May train a model if that use is lawful, but is **never** redistributed and never enters `dataset/`. Requires an explicit note on why it is lawful. |
| `REFERENCE_ONLY` | May be looked at. Never trained on, never redistributed. |
| `REJECT` | Not used at all. |

`ACCEPT_FOR_SUBMISSION` is only consistent when **all** of these hold, and
`scripts/audit_real_source.py` refuses the record otherwise:

- redistribution, modification and commercial use are all `yes`;
- the licence is one that permits redistribution (NC, ND and SA are rejected
  outright — the dataset is published under plain CC BY 4.0);
- `provenance_evidence` is not empty and `license_evidence_url` is present;
- the required attribution text is recorded, if attribution is required;
- the privacy review is `completed` or `not_required`.

### Contributed photographs: the grant

Someone outside the team offering us their photographs is the one case where
the rights are clean but the *evidence* of them is easy to get wrong. The
photographer holds the copyright, so they can grant exactly what we need — and
a chat message saying "yeah, use them" grants nothing we could show anyone.

Send this, and keep the reply verbatim:

> I'd like to include your photograph(s) in a public dataset for the Volga-IT
> 2026 competition. The dataset is **published under CC BY 4.0**, which means
> anyone may copy, modify and use the images commercially, with credit. We
> crop, resize and blur faces where needed. Please reply confirming:
>
> 1. You took these photographs yourself.
> 2. You grant them under **CC BY 4.0**, understanding they will be published
>    and cannot be withdrawn once released.
> 3. The credit line you want (name, handle, or "anonymous").
> 4. No identifiable person is the subject of the images.

Each answer maps onto a field of the source record — the point of asking in
this order:

| Their answer | Source-record field |
| --- | --- |
| "I took them" | `original_creator`, and `upstream_source` = "photographed by the contributor" |
| "I grant CC BY 4.0" | `stated_license` = `CC BY 4.0`; the three gate fields all `yes` |
| The credit line | `attribution_required` = `yes`, `attribution_text` = their exact wording |
| "no identifiable person" | `privacy_review` — their statement, not a substitute for our own pass |
| Where the reply is kept | `license_evidence_url`, e.g. "email from the contributor, 2026-09-20, archived at …" |

Rules that decide the awkward cases:

- **A partial grant is not a grant.** If any of the four is missing, the
  decision stays `PENDING` and the images stay in `incoming/`.
- **"Anonymous" is a valid credit line**; a missing one is not. CC BY requires
  attribution, so `attribution_text` must say something, even if it is
  "anonymous contributor".
- **Their privacy answer does not replace ours.** We still review every image
  and still fill in `privacy_review.csv`. They are stating intent, and we are
  the ones publishing.
- **A photograph they found is not a photograph they took.** If the answer to
  (1) is anything other than a plain yes — "a friend sent it", "it's from a
  group chat" — the source is `REJECT`, whatever the friend would say.
- **Archive the reply itself**, not a summary of it. The record cites where it
  is kept; the paper trail is the message.

### Rules that decide most cases

- **A platform's licence label is not ownership.** A dataset marked CC BY on a
  hosting site says what the uploader claims, not what the photographer
  granted. Without evidence of the chain from the photographer, provenance is
  unclear.
- **Unclear provenance is rejected for submission.** It may at most be
  `REFERENCE_ONLY`. We do not resolve doubt in our own favour.
- **The test is redistribution, not use.** A licence that lets us train but not
  republish cannot enter the submitted dataset.
- **No scraping** that bypasses access restrictions, paywalls, login walls,
  rate limits or `robots.txt`. If getting the image requires circumventing
  something, the image does not go in.
- **Our own photographs are the cleanest source.** We are the creator, the
  rights are ours to grant, and the provenance record says exactly that.
- A decision is reversible on new evidence. Record the change and the date;
  never rewrite the history.

## Privacy and submission compliance

- **Faces are blurred or covered before an image enters `dataset/images/real/`**,
  wherever a person is identifiable and the licence or applicable privacy rules
  require it. Never afterwards, and never only in a derived copy.
- **Images centred on identifiable people are rejected.** We photograph
  vehicles. A plate in a street scene is fine; a portrait with a car behind it
  is not, and no blur makes it fine.
- **The provenance survives the transformation.** Blurring produces a new file;
  its row keeps the original `source` and `license`, and the privacy review
  records what was changed. We never launder an image's origin by editing it.
- **Every real image gets a `privacy_review.csv` row**, even when the answer is
  `none_needed`. `faces_present=yes` with `action=none_needed` is a blocking
  finding.
- **The official debug set never enters the submitted dataset.** It stays in
  `data/official_debug/`, is never copied into `dataset/`, and is never
  referenced from `meta.csv`. The validator rejects any path outside
  `images/real/` and `images/synthetic/`, which catches the mistake mechanically.
- **Unclear-rights images never enter the submitted dataset**, whatever their
  quality and however much we need that class.

## The workflow

1. **Register.** Write the source record. Read the licence text now, not later.
2. **Stage.** Put the images in `incoming/<source_id>/`. Do not touch
   `dataset/`.
3. **Privacy pass.** Blur or cover faces; reject people-centred images; fill in
   `privacy_review.csv`.
4. **Annotate** per [`annotation_guide.md`](annotation_guide.md) into the
   folder's `meta.csv`, and record the leakage groups in `groups.csv`.
5. **Audit.**

   ```bash
   python scripts/audit_real_source.py data/real_staging/incoming/<source_id> \
       --json data/real_staging/audits/<source_id>.json \
       --report data/real_staging/audits/<source_id>.txt
   ```

   It reports counts, formats, dimensions, corrupt files, exact and near
   duplicates, images already in the dataset, annotation coverage and validator
   findings, class and unique-plate counts, privacy coverage, and the split
   state. Exit code 2 means something blocks promotion.
6. **Decide.** A person sets the decision in the source record. The audit only
   checks that the rights recorded support it.
7. **Move** cleared sources to `accepted/<source_id>/`, rejected ones to
   `rejected/<source_id>/` with the reason in the record.
8. **Freeze the split.**

   ```bash
   python scripts/plan_real_splits.py --add data/real_staging/accepted/<source_id> --write
   ```

9. **Promote** the accepted images into `dataset/images/real/<source_id>/`,
   append their rows to `dataset/meta.csv`, and register the source in
   `data_sources.md`.
10. **Validate.** `python scripts/validate_dataset_local.py --strict` must exit
    0, and `scripts/plan_real_splits.py --check` must report CLEAN.

Steps 9 and 10 are the only ones that write into `dataset/`, and they happen
after a human decision — exactly as the synthetic batch was promoted.

## What the tooling will not do

- It does not approve a source, and it does not infer rights from a licence
  string. Only a person writes a decision.
- It does not auto-label images. Detector output is a suggestion for a human
  annotator, never an annotation, and a model-labelled image is never described
  as manually annotated.
- It does not download anything. Acquisition is a separate, explicitly approved
  step.
- It does not move or delete staged files.
