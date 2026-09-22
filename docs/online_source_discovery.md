# Online real-photograph discovery

Targeted discovery of **already-published, individually licensed photographs**
of real Russian vehicles, run 2026-09-19, after self-capture was deferred.

**This pass acquired nothing.** Its output is a **candidate manifest**, not a
dataset: what it fetched was metadata, plus small review thumbnails kept in
staging so a person could judge a plate class by eye. No photograph entered
`dataset/` and no model was trained.

> **Superseded in part, 2026-09-22.** A person then approved acquisition of the
> 22 candidates this pass marked `ACCEPT_FOR_SUBMISSION`, and they were
> downloaded and taken through intake — see
> [`acquisition_wikimedia_commons.md`](acquisition_wikimedia_commons.md) and
> "What happened next" below. The numbers in this document remain those of the
> discovery pass on 2026-09-19; intake revised one of its class calls. Still
> nothing in `dataset/`.

Related: [`real_data_source_research.md`](real_data_source_research.md) (the
2026-09-17 survey this continues), [`real_data_intake.md`](real_data_intake.md)
(what happens to a photograph after it is approved),
[`capture_session_type1a.md`](capture_session_type1a.md) (the deferred
self-capture), [`data_sources.md`](data_sources.md) (the registry).

## Why this pass exists

Acquisition #1 was a targeted team capture session for `type1a`, prepared in
full on 2026-09-17 and stopped at `WAITING_FOR_HUMAN_CAPTURE`. It is now
**DEFERRED**, for a reason no amount of effort changes: the participant is in
**Bulgaria**, and Russian `type1a` and `type1b` plates are not on the street
here. It is kept, not cancelled, and would revive the moment someone can shoot
to its checklist.

That left one honest route: photographs already taken by other people in
Russia, published under a licence that lets us republish them. The 2026-09-17
survey had looked for *datasets* and found none that supplied the rare classes.
This pass looked for **individual photographs**, which is a different search and
a much better one.

## The licence question, answered first

Before accepting anything, the question the repository already answers
(`dataset/README.md`, "Legal and source tracking"):

> The competition rules state that the submitted dataset is **published under
> CC BY 4.0** and may be used and published by the organizers with attribution.

So this is **arrangement B**: the dataset *as a whole* must be distributable
under plain CC BY 4.0. Individual items may not keep an incompatible licence of
their own.

What that means per licence, decided explicitly rather than assumed:

| Licence | In a CC BY 4.0 dataset? | Why |
| --- | --- | --- |
| **CC0 1.0** | **Yes** | A public-domain dedication imposes no downstream condition. No credit is legally required; we record the creator anyway. |
| **Public domain / PD-self** | **Yes** | Same: nothing to carry forward. |
| **CC BY 4.0** | **Yes** | Identical terms to the dataset's own. Attribution must travel with the image. |
| **CC BY 3.0 / 2.5 / 2.0 / 1.0** | **Yes** | Attribution-only, and each permits redistribution and modification. The *aggregate* is published under CC BY 4.0 while each photograph keeps its own licence and credit — CC BY does not require the result to be relicensed, only attributed. |
| **CC BY-SA (any version)** | **No** | ShareAlike is the trap: it permits republication and then forbids releasing the result under plain CC BY 4.0. It fails on the arrangement, not on the permissions. |
| **CC BY-NC / -ND (any)** | **No** | Non-commercial fails the commercial-use requirement; no-derivatives fails cropping and face blurring. |
| **GFDL, FAL, GPL, LGPL** | **No** | Copyleft outside Creative Commons — the ShareAlike problem under another name. **Added to the gate in this pass**; see below. |
| Anything unrecognised | **Hold** | `UNKNOWN` is not a soft yes. It blocks acceptance until a person resolves it. |

**We never relicense someone else's photograph as our own CC BY 4.0.** The
image keeps its licence, its creator and its credit line; the dataset is the
thing published under CC BY 4.0, and it can only contain images whose own terms
allow that.

### A gap this pass closed

`src/dataset_meta.py` knew NC, ND and SA but not **GFDL, FAL, GPL or LGPL**, so
those fell through as merely "unrecognised". They are copyleft and cannot sit in
a CC BY 4.0 aggregate, and Commons serves a steady trickle of them. They are now
named in `INCOMPATIBLE_LICENSE_TOKENS`, with tests. A side effect worth noting:
the gate now rejects **LGPL-3.0** on its own — the exact licence that got the
Kaggle Nomeroff mirror rejected by hand on 2026-09-17.

## Where we looked

| Platform | Result |
| --- | --- |
| **Wikimedia Commons** | **The productive source.** Per-file licence, per-file author, a real file page to verify against, and a machine-readable API. Everything below comes from here. |
| **Openverse** | Usable as an index, but the anonymous tier rate-limited us out (HTTP 429) after a handful of queries. Partially probed; **no candidate was accepted from it**. It surfaced Flickr and Commons results with licence labels, which is a lead, never evidence. An API key would make it viable for the acquisition stage. |
| **Flickr** | Not searchable first-hand without an API key. Openverse showed CC BY 2.0 Flickr material for Russian transport exists, but **nothing was verified on a Flickr photo page**, so nothing is claimed. Recorded as unresolved, not as a yield. |
| **Government / institutional** | One genuine hit reached the manifest: a **Rostov-on-Don city administration** photograph (CC BY 4.0, credit `Rostov-gorod.ru`) of a municipal bus depot inspection. Russian municipal press services do publish under CC BY; this is a thin but real vein. |
| **Academic / dataset repositories** | Re-searched; nothing new and clean. Hits were the already-rejected Roboflow projects and commercial vendors. The 2026-09-17 conclusion stands. |
| **AUTO.RIA / Roboflow** | Not used. AUTO.RIA has 0 confirmed rare-class images and is not a rare-class source; the rejected Roboflow projects stay rejected. |

### Queries

English and Russian, in both the file-search and category namespaces. Commons'
search is word-based and noisy, so the narrowing is done by the filters, not the
query.

- `type1a`: *Russian square license plate*, *Russia two-line license plate*,
  *russian license plate type 1A*, *japanese import Russia license plate*,
  *тип 1А номер*, *квадратный номер Россия автомобиль*, *ГОСТ 50577 квадратный*.
- `type1b`: *Russian yellow license plate taxi*, *yellow license plate Russia
  bus*, *тип 1Б номер*, *желтый номер такси*, *такси Россия номер*,
  *маршрутка номер*, *желтые номера автобус Россия*.
- General: *Russian license plate car*, *russian registration plate vehicle*,
  *регистрационный знак Россия*, *госномер России автомобиль*.

**The categories mattered far more than the search terms.** The decisive find
was `Category:Vehicles with license plates of Russia` and its siblings — photos
of vehicles *categorised as carrying Russian plates* — together with the Russian
taxi operator trees (`Category:Yandex.Taxi`, `Category:Yellow taxis in Moscow`,
`Category:Taxis in Moscow`). Searching for the word "plate" mostly returns
pictures *of plates*; the vehicle categories return photographs of traffic.

## What the numbers actually are

| Stage | Files |
| --- | --- |
| Distinct titles collected (category walks + searches) | **5,813** |
| Surviving format, size and licence screening | **2,169** |
| — of those, licence COMPATIBLE | **2,143** |
| — INCOMPATIBLE (copyleft, caught by the new tokens) | 16 |
| — UNKNOWN, held for a person | 10 |
| Linked to Russia | **1,908** |
| Post-2019, the window in which a yellow plate can exist | 774 |
| Review thumbnails fetched, so a person could look | **1,053** |
| Yellow plate-region crops reviewed by eye | **140** |
| Two-line white-region crops reviewed by eye | **105** (64 files) |
| Scenes reviewed by eye in the targeted `type1a` square hunt | **179** |

Across the three pools the screening dropped 1,030 non-JPEG files, 309 under
400 px and 2,622 on licence. The first two are the same finding the 2026-09-17
survey measured: most Commons files tagged for Russian plates are **graphics,
not photographs** — region-code illustrations and renderings of the standard.

### Licences among the 1,908 usable Russian photographs

CC0 587 · CC BY 4.0 461 · CC BY 3.0 461 · Public domain 197 · CC BY 2.0 195 ·
CC BY 2.5 7. All six are compatible with publication under CC BY 4.0.
1,870 further files were dropped for ShareAlike, NC or ND.

## Four things a person had to see

**1. A yellow car is not a yellow plate.** Moscow taxis are painted yellow by
livery convention, and most of the ones photographed still carry **white**
plates. Of four Moscow taxi photographs checked at full review resolution, three
were `type1` (`В383СВ797`, `С838СК797`, `К362НН977`) and one was `type1b`
(`УУ 757 77`). Any pipeline that scored "yellow pixels near a car" would have
mislabelled three quarters of them. This is why every accepted row was looked at.

**2. Buses are not automatically `type1b` either.** A 2025 municipal city bus in
Naberezhnye Chelny carries the white plate `Р970УТ716`. The yellow plate marks
*licensed commercial passenger carriage*, which not every bus in a photograph is
doing, and re-plating happens on re-registration rather than on a date.

**3. There are two yellow-plate eras, and only one of them is `type1b`.**
Russia issued yellow commercial-transport plates from **1 March 2002**, phased
them out around **2008**, then reintroduced them under **GOST R 50577-2018,
effective 1 January 2019**. A yellow Russian plate photographed in 2008 or 2011
is a *legacy* plate under the older standard, not the `type1b` the competition
defines. Two otherwise-good candidates were held as `AMBIGUOUS` for exactly this
reason rather than being counted.

**4. A blank plate looks square, and Moscow has a lot of blank plates.** Every
candidate the geometry filter offered as a "square two-line plate" turned out,
at review zoom, to be an ordinary **one-line** plate — wide character field plus
the narrow right-hand `RUS` region block — whose *characters were blank or
unreadable*. A Lincoln Corsair (`cw169547179`), a Neta GT (`cw175011132`) and a
Toyota Voxy (`cw175243961`) all show an empty white panel with only `977`/`797
RUS` legible. A recent Commons upload is even titled "Daihatsu with **shadow
plates**". Cropped to the plate and scaled down, a blank one-line plate is a
white rectangle of roughly plate-square proportions, which is exactly what an
aspect-ratio filter is built to find. Two consequences: the square hunt's whole
yield was false positives, and those photographs would be **useless as training
data anyway** — a plate with no characters carries no label.

## Verified candidates

Every row below was checked twice: its licence and author read from the file's
own Commons metadata, and its plate looked at in a crop.

| Class | Confirmed | Accepted | Decision |
| --- | --- | --- | --- |
| `CONFIRMED_TYPE1B` | **14** | **12** | ACCEPT_FOR_SUBMISSION; 2 held on rights |
| `CONFIRMED_TYPE1` | **14** | **10** | ACCEPT_FOR_SUBMISSION; 4 held, rights unverified |
| `CONFIRMED_TYPE1A` | **0** | **0** | — |
| `REJECT` | 5 | — | REJECT |
| `AMBIGUOUS`, flagged for a second look | 7 | — | PENDING |
| Licence-screened, class unreviewed | 2,129 | — | PENDING |

The 14 `type1b` photographs are Moscow and regional taxis (Chery Tiggo 4,
Kia K5 ×2, Moskvich 3), Yakutsk and Tver-region buses (LiAZ 5292, PAZ-3205,
PAZ Vector Next ×3, KAvZ-4270 ×2), a Bryansk PAZ and a Rostov-on-Don depot
inspection. Licences: CC0 ×8, CC BY 4.0 ×3, Public domain ×2, CC BY 3.0 ×1.

**Two of them are held, not accepted.** `cw128647044` and `cw128647042`
(PAZ Vector Next, Tver Oblast, `Mungojerrie (Андрей Кузнецов)`) carry a bare
`Public domain` self-dedication with **no licence URL**, and their Commons
`Credit` field points at **`fotobus.msk.ru`** rather than the uploader. So the
public-domain claim is asserted by an uploader over a third-party mirror and
cannot be verified at an original source. The plate class stands — both are
genuinely yellow one-line `type1b` — but the rights do not, and the audit gate
refused them. They are `PENDING` until a person resolves the claim with the
photographer. That is the gate working, not a defect.

Five candidates were **rejected outright** on review: three blank-plate false
positives (above), a Toyota Alphard on **Armenian** plates, and a tractor whose
yellow two-line plate is a *Gostekhnadzor self-propelled-machinery* registration
— a different plate type from the competition's `type1b`.

### `type1a` — still zero, and that is the finding

**No confirmed `type1a` photograph was found**, and this was looked for three
separate ways before it was concluded:

1. **Search and category walk.** Commons has no category for square or two-line
   Russian plates, and the search terms return plate *graphics*.
2. **Geometry filter over the thumbnails.** 105 two-line white-region crops from
   64 files, reviewed on three contact sheets. Every one was sky, snow, a white
   building face, signage or vehicle bodywork — the filter had latched onto
   whole-frame bright regions, not plates. Two of the 64 were vintage postcards
   and posters, i.e. graphics, and are rejected as such.
3. **A targeted square hunt.** 179 scenes chosen for the vehicle classes where a
   square plate is actually plausible — Japanese kei imports and grey-import
   vans (Daihatsu Tanto, Toyota Voxy/Alphard/HiAce, Honda Freed Spike), plus
   recent Chinese and US imports — reviewed on twelve scene sheets and then at
   plate zoom. Every readable plate was **one-line**; the square-looking ones
   were blank plates (above); the Alphard's plate was **Armenian**.

This matches, and hardens, the 2026-09-17 conclusion. `type1a` is rare on the
street by nature, so it is rare in other people's photographs too — being a
photograph rather than a dataset does not make it less rare. The one honest
surprise is that Japanese imports, the most likely `type1a` carriers, are
overwhelmingly photographed wearing **ordinary one-line plates**: a Russian
registration is issued to the owner, not to the car's plate recess.

Two candidates are left genuinely **unresolved** rather than rejected:
`cw186512094` (second frame of the Daihatsu) and `cw176100848` (Toyota HiAce),
whose plates cannot be resolved at review resolution. Settling them needs the
full-resolution file, which is an acquisition, not a discovery.

## The concentration problem

Two photographers account for roughly **1,000 of the 1,908** usable Russian
photographs:

- **Artyom Svetlov** — ~712 files, CC BY 4.0 / CC BY 3.0, a prolific Russian
  transport photographer (his name appears in four spellings in the metadata);
- **"Retired electrician"** — ~300 files, CC0, the source of most of the recent
  Moscow taxi photographs.

This is good news for rights — the chain is short and the licence is clear —
and a real limit on **domain diversity**: one photographer is one camera, one
city and one habit of framing, which is the same problem
`real_data_plan.md` already guards against with its "≥ 3 distinct cameras, none
above 60 % of a class" rule. Any acquisition from here must count photographers,
not just images.

It is also a **leakage** hazard, and the manifest handles it: `source_group` is
`commons:<creator>:<date>`, so one photographer on one day is one group and
cannot be split across train and holdout. The grouping already caught a real
case — two files whose capture timestamps are identical to the second
(`2022-05-15 15:24:51`) are the same photograph or one burst, and are held
pending a duplicate check rather than counted twice.

## The manifest

`data/real_staging/manifests/online_candidates_2026-09-19.csv` — one row per
candidate, in the schema defined by `src/online_discovery.py`:

`candidate_id`, `source_platform`, `original_url`, `file_page_url`, `creator`,
`license`, `license_url`, `attribution`, `redistribution_status`,
`modification_status`, `plate_class`, `class_confidence`, `is_real_photo`,
`full_scene_or_crop`, `resolution`, `source_group`, `provenance_status`,
`decision`, `notes`.

Unknown fields say `UNKNOWN`; none is invented. `provenance_status` separates
**`VERIFIED_ON_SOURCE`** (the file's own Commons metadata was read, and a
sample was cross-checked against the rendered file page) from
**`PLATFORM_METADATA_ONLY`** (licence screened, photograph not yet looked at).
`scripts/audit_online_manifest.py` refuses `ACCEPT_FOR_SUBMISSION` on any row
that is not verified, not a photograph, ambiguous in class, missing a credit
line a licence requires, or missing a source group.

Review material — thumbnails and the contact sheets — is in
`data/real_staging/review/commons_discovery/`. It is **not committed**: it is
copies of other people's photographs, held only so a person can check a class.
The sheets a person should actually open, in `sheets/`:

| File(s) | What it shows |
| --- | --- |
| `crops_yellow_01..04.jpg` | 140 yellow plate-region crops — the `type1b` yield |
| `crops_two_01..03.jpg` | 105 two-line white-region crops — all false positives |
| `scene_type1a_hunt_01..12.jpg` | the 179 targeted square-plate scenes |
| `zoom_squarehunt.jpg` | the six square-looking plates at zoom; settles them |
| `crops_yellow.json`, `crops_two.json` | crop → file title, licence, region |

## Honest gaps

- **`type1a` is not solved and this pass did not solve it.** Zero confirmed,
  after three independent attempts. Self-capture remains the only route that has
  ever been credible for it, and it is deferred for geography.
- **Two `type1b` photographs are held on rights**, not counted as accepted.
- **Two candidates are unresolved** and cannot be settled without the
  full-resolution files.
- **2,129 candidates are licence-screened but unreviewed.** The `type1b` yield
  among reviewed crops suggests more are there; that is an expectation, not a
  count, and it is not reported as one.
- **Flickr is unverified.** Nothing accepted, nothing claimed.
- **Privacy has not been done.** At least one accepted candidate shows an
  identifiable passenger. Faces are handled at intake, before anything enters
  `dataset/images/real/`, per `real_data_intake.md`.
- **`plate_reading` in the notes is a reading for review, not an annotation.**
  Annotation happens at intake, on the full-resolution file, by hand.

## What happened next

**Acquisition #2 was approved on 2026-09-22 and the 22 ACCEPT_FOR_SUBMISSION
candidates were acquired.** All 22 downloaded, all still matching the licence
they were approved under. Intake then annotated 21 of them (24 plates: 12
`type1`, 12 `type1b`) and rejected one on privacy, and found that **`wcc_0011`
was not the class this pass recorded** — its subject plate is white `type1`,
and the `type1b` belongs to the car behind it. That is the review-resolution
limit of this pass showing up, and the reason a full-resolution intake exists.
Nothing is promoted. See [`acquisition_wikimedia_commons.md`](acquisition_wikimedia_commons.md).

The sequence that was followed:

1. Register `wikimedia_commons_curated` in [`data_sources.md`](data_sources.md)
   — one `source_id`, with per-image creator and licence carried in the
   manifest, since the rights here are per file rather than per collection.
2. Download only the `ACCEPT_FOR_SUBMISSION` rows, recording SHA-256, original
   filename and acquisition date per file.
3. Stage under `data/real_staging/incoming/`, then the ordinary intake: privacy
   pass, manual annotation, groups, audit, human review, decision.
4. Carry each photograph's own credit line into `dataset/LICENSE` at import.

Nothing is promoted into `dataset/images/real/` before that whole sequence, and
no bulk download happens before a person says so.
