# Acquisition #2 — `wikimedia_commons_curated`

The 22 photographs the online-discovery pass approved, fetched on **2026-09-22**
and taken through intake. A person reviewed the contact sheets the same day and
**promoted the 13 marked PASS**; `decision: ACCEPT_FOR_SUBMISSION`. The other 9
stay in staging and are not in the dataset.

Related: [`online_source_discovery.md`](online_source_discovery.md) (where the
22 came from and why), [`real_data_intake.md`](real_data_intake.md) (the
workflow), [`annotation_guide.md`](annotation_guide.md) (the labelling rules),
[`data_sources.md`](data_sources.md) (the registry).

## What was acquired

22 of 22 approved candidates. **Nothing was held and nothing failed to
download.** Before any byte was fetched, each file's licence, author, media
type and size were read again from the Commons API and compared with the
approved record; all 22 still matched. Each download was then checked against
the SHA-1 Commons publishes, and its own SHA-256 recorded.

| | |
| --- | --- |
| Requested | 22 |
| Acquired | **22** |
| Held / failed | **0** |
| Bytes | 74.9 MB, all JPEG, 1024×682 to 6432×4112 |
| Licences | CC0 ×17, CC BY 4.0 ×5 |
| Photographers | 6 |
| Leakage groups | 20 |

The per-file paper trail is
`data/real_staging/incoming/wikimedia_commons_curated/acquisition_record.csv`
— original URL, file page, creator, exact licence, licence URL, required
attribution, acquisition timestamp, SHA-256, byte count, dimensions, source
group. It is committed; the photographs are not.

Originals are kept untouched in `originals/`, under the filename the
photographer gave them. The working copies — the files that were blurred,
annotated and would be promoted — are `images/real/wikimedia_commons_curated/wcc_0001.jpg`
upward, already on the path they will have inside `dataset/`.

## What intake found

**Every plate was located and read by eye on the full-resolution file.** A
colour-and-geometry pass proposed regions to look at and nothing more: no label
here comes from a detector, per the "never auto-label" rule in the annotation
guide.

| | |
| --- | --- |
| Images annotated | **21** |
| Plates annotated | **24** (24 distinct plate numbers) |
| `type1` | **12** |
| `type1b` | **12** |
| `type1a` | **0** |
| PASS | **13 images** (13 plates) |
| QUESTIONABLE | **8 images** (11 plates) |
| FAIL | **1 image** |
| Exact duplicates | 0 |
| Near duplicates | 0 |
| Corrupt files | 0 |
| Annotation validator | 0 errors, 0 warnings |

### One image was rejected

**`wcc_0018`** (`cw89232390`, the Rostov-on-Don depot inspection) is a
**people-centred photograph**: about eleven identifiable people walk towards the
camera and occupy the centre of the frame, and the buses are context. The intake
policy is explicit — *"Images centred on identifiable people are rejected. We
photograph vehicles... and no blur makes it fine."* It is rejected rather than
blurred. Its `type1b` plate was also only 60 px wide.

This is worth noting beyond the one file: it was the single
government/institutional hit in the entire discovery pass. A municipal press
service publishes under a good licence, but it photographs *events*, and events
have people in them. That vein is thinner than its licence suggests.

### One image disagreed with what discovery expected

**`wcc_0011`** (`cw169321860`) was recorded as `type1b`. At full resolution the
subject vehicle — a Moskvich 3 taxi — carries a **white one-line plate**,
`A095CX797`, which is `type1`. The yellow `type1b` plate that discovery saw
belongs to **the car behind it** (`KX 943 77`).

Both plates are annotated, each as what it is. The expectation was **not**
forced onto the subject plate. Discovery worked from a 960 px review thumbnail,
where the two cars' plates sit close together; this is exactly the error a
full-resolution intake pass exists to catch.

### Why eight are QUESTIONABLE

None of these is a defect in the photograph; they are things a reviewer should
decide rather than have decided for them.

- **Unannotated background plates** (`wcc_0002`, `0004`, `0008`, `0012`,
  `0015`, `0016`) — further plates are visible but too small or too soft to
  read. The guide would have them annotated as `other` with a box; that second
  pass is **outstanding**, and saying so beats inventing boxes.
- **Very small plates** (`wcc_0015` at 69 px wide, `wcc_0021` at 132 px) —
  legible at zoom, marginal as OCR training data.
- **The class disagreement** (`wcc_0011`, above).

### Privacy

Every image has a `privacy_review.csv` row. Faces were blurred **before**
anything could be promoted, never afterwards, and the blur never changes where
an image came from — each row keeps its original `source` and `license`.

| Action | Images |
| --- | --- |
| `none_needed` | 12 |
| `blurred` | 9 (22 face regions) |
| `rejected` | 1 |

Of the 13 promoted, 8 needed nothing and **5 had faces blurred**.

## Known approximation: the quads

`quad_x1..quad_y4` are currently the **four corners of the bounding box**. The
annotation guide says that is correct for a plate photographed straight on, and
most of these are. Four plates are visibly oblique and tagged `angle` in
`conditions`: `wcc_0003`, `wcc_0012` and both plates in `wcc_0016`.

**Only one of those, `wcc_0003`, was promoted** — the other three are on images
the review held back. So exactly one row in `dataset/meta.csv` carries a quad
that is a bounding-box approximation rather than the plate's true corners. It is
recorded here rather than left for someone to discover during training, and it
should be corrected on the next pass over this source.

## Rights

The rights here sit on **each file**, not on the collection, so the source
record lists every licence present (`CC0; CC BY 4.0`) and the per-file licence
travels with the image in the acquisition record. The intake gate was taught to
read such a list, and it passes only if **every** licence in it would pass on
its own — one ShareAlike entry would fail the whole source.

Both licences permit redistribution, modification and commercial use, so both
can sit inside a dataset published under CC BY 4.0. CC0 imposes no legal
attribution requirement; the creator is recorded and credited anyway. At
promotion, each photograph's credit line goes into `dataset/LICENSE`.

## What a reviewer should open

`data/real_staging/review/wikimedia_commons_curated/review_wikimedia_commons_curated_01..07.jpg`
— 25 cards, one per annotated plate plus the rejected image. Each card shows
the full photograph with the bounding box drawn on it, the plate crop beside it,
and the candidate id, `plate_type`, `plate_num`, creator, licence, source page,
leakage group, privacy action and verdict.

Not committed: they embed other people's photographs. Rebuild them with

```bash
python scripts/build_intake_review.py wikimedia_commons_curated
```

## The human review, and what it promoted

Reviewed **2026-09-22** against the seven contact sheets. The decision was
**PASS only**: 13 of the 22 images, carrying 13 plates.

| | Images | Plates |
| --- | --- | --- |
| **PASS — promoted** | **13** | **13** (6 `type1`, 7 `type1b`, 0 `type1a`) |
| QUESTIONABLE — not promoted | 8 | 11 |
| FAIL — not promoted | 1 | 0 |

The reviewer restated the class rule, and it was checked against every promoted
plate before promotion: **a yellow taxi *vehicle* does not make a `type1b`
plate.** The registration plate itself must be the yellow format. Of the 13, six
are **white plate panels** — several of them bolted to bright yellow taxi
bodywork — and stay `type1`; seven are **yellow plate panels** and are `type1b`.
No plate was moved into `type1b` to make a target.

Promoted, in order:

| File | Class | Plate | Licence |
| --- | --- | --- | --- |
| `wcc_0001` | `type1` | `K362HH977` | CC BY 4.0 |
| `wcc_0003` | `type1` | `C838CK797` | CC0 |
| `wcc_0005` | `type1b` | `KM34350` | CC0 |
| `wcc_0006` | `type1` | `P884TH797` | CC0 |
| `wcc_0007` | `type1` | `A512EE799` | CC0 |
| `wcc_0009` | `type1b` | `MP00577` | CC0 |
| `wcc_0010` | `type1b` | `YY75777` | CC0 |
| `wcc_0013` | `type1` | `A332HT797` | CC0 |
| `wcc_0014` | `type1` | `C155OP797` | CC0 |
| `wcc_0017` | `type1b` | `KM38814` | CC0 |
| `wcc_0019` | `type1b` | `YX34377` | CC0 |
| `wcc_0020` | `type1b` | `KE11714` | CC BY 4.0 |
| `wcc_0022` | `type1b` | `KM58214` | CC0 |

Split, frozen group-aware in `dataset/splits/real_splits.csv`: **train 10, val
1, holdout 2** over 11 groups. Leakage check CLEAN — no group spans a split, no
identical or near-duplicate image crosses one.

Each promoted photograph keeps its own licence in `dataset/meta.csv` and its own
credit line in section 2a of `dataset/LICENSE` — author, licence, licence URL
and Commons file page per file. **We do not relicense them**; our CC BY 4.0
grant covers our annotations, not other people's photographs.

## What stayed behind

The 9 unpromoted images remain in
`data/real_staging/incoming/wikimedia_commons_curated/` with their annotations
and review notes intact. They are not deleted and not rejected outright (except
`wcc_0018`): each carries the reason it was held, and a later pass can resolve
the unannotated background plates and re-submit them for review. That pass would
also be the moment to replace the approximate quads on the oblique plates.
