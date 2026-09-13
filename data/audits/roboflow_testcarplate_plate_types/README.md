# Roboflow `russian-license-plates-classification-by-this-type` — desk research

Time-boxed provenance and content investigation, 2026-09-13. This was the
final broad external-dataset search before building the synthetic generator.

**Decision: `REJECT_FOR_SUBMISSION`** (provenance). Secondary finding:
**`NOT_USEFUL_FOR_RARE_CLASSES`** even if provenance were cured.

**Nothing was downloaded, audited or imported.** The source has **not** been
added to the registry in `docs/data_sources.md`. Registry rows require the
licence text to have been read first-hand, and it could not be (see "Access").

## Identification

| Field | Value | Verified how |
| --- | --- | --- |
| Platform | Roboflow Universe | Search results |
| Project | *Russian license plates classification by this type* (`russian-license-plates-classification-by-this-type`) | URL slug in search results |
| Workspace / owner | `testcarplate` (anonymous handle; no person or organisation named) | URL slug |
| URL | https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type | Search results |
| Versions seen | dataset **v3** (`/dataset/3`), model **v8** (`/model/8`) | URL slugs |
| Published | ~October 2024 | Search summary only |
| Task | **Object detection** (YOLOv8 tags), not image classification, despite the name | Page titles: "Object Detection Model" |
| Image count | **7,879** on the project page; one summary gave **4,965**, probably a single version's count | Search summaries only — **unconfirmed** |
| Classes | **2: `n_p`, `p_p`** | Model workflow parameters quoted in search summaries |
| Class counts | Not found | — |
| Class meaning | Not defined in any verbatim text. One summary said `p_p` = police and `n_p` = ordinary civilian plates; another said the meaning is not stated | **Unconfirmed** |
| Scenes vs crops | Not determinable without the page | Unknown |
| Declared licence | CC BY 4.0 | Search summaries only |

The author's stated roadmap, as repeated by several summaries: the data is
"currently annotated into 2 classes", with plans to add **taxis and
motorcycles** and possibly all Russian plate types later. In other words,
**taxi / passenger-transport (our `type1b`) is not annotated in this dataset.**
No square / two-line (`type1a`) class is reported.

## Provenance of the underlying photographs

Search summaries restricted to `universe.roboflow.com` stated that the project
uses photographs from **platesmania.com** and **migalki.net**. Both are
plate-spotting sites where members upload their own photos of vehicles.

- **Confidence in this claim: moderate, not verbatim-verified.** It came from a
  search engine's summary of the Roboflow page, not from a quote read
  directly. It reappeared when the search was limited to Roboflow's domain,
  which makes it unlikely to be a mix-up with the unrelated OSINT article that
  appeared in an earlier result.
- If accurate, the images belong to many individual contributors on those
  sites. They were not taken by `testcarplate`, and the uploader had no right to
  relicense them under CC BY 4.0.
- **No evidence was found to the contrary.** No photographer, no own-work
  statement and no upstream licence was found.

## Decision reasoning

We apply the same standard as for
`roboflow_two_line_russian_license_plates`: a platform licence field is not a
provenance chain.

1. **Provenance fails.** The only provenance information found points to
   third-party plate-spotting sites. Even setting that aside, the uploader is
   an anonymous handle with no own-work claim. Either way the rights in the
   underlying photographs cannot be justified for a CC BY 4.0 redistribution.
2. **Rare-class value is low even if provenance were cured.**
   - There is no taxi / yellow passenger-transport class; the author lists it
     as future work.
   - No two-line / square class is reported.
   - The two classes that do exist appear to be ordinary plates and one
     special category, which at best help `type1` and `other`. Those are not
     our bottleneck.
3. **The ~7.8k size does not change this.** Size is irrelevant if the rights
   cannot be justified and the rare classes are absent.

## Access limitations (recorded honestly)

- `universe.roboflow.com` returns **HTTP 403** (Cloudflare bot challenge) to
  both WebFetch and plain HTTP clients. It was not bypassed.
- The Wayback Machine was unreachable (the fetch tool is blocked; the API
  returned 429).
- The browser extension was not connected.
- A Hugging Face model that appeared in the results
  (`Garon16/rtdetr_r50vd_russia_plate_detector_lightning_test`) returned
  **401**, so it is private or removed.

So nothing above is a verbatim quote from the project page. The licence, image
count, class meanings and the platesmania/migalki statement should all be
read as **search-derived**.

## What would reopen it

All of the following, read first-hand:

- the project page stating the images are the uploader's own work, or naming an
  upstream source whose own licence permits CC BY 4.0 redistribution;
- **and** class definitions showing genuine `type1a` or `type1b` content.

Absent both, do not re-investigate.

## Search trail

- https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type
- https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type/dataset/3/download
- https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type/model/8
