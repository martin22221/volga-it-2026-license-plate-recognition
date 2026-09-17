# Real-data source research

Desk research into public sources of **real** Russian license-plate photographs,
run 2026-09-16 and continued 2026-09-17, with the rare competition classes as
the priority.

**Nothing was downloaded, imported or trained on.** No file under `dataset/`
changed. Every finding below is desk research: pages read, licence text read
where reachable, and prior audit records re-used rather than repeated.

Related: [`real_data_plan.md`](real_data_plan.md) (targets and the split
policy), [`real_data_intake.md`](real_data_intake.md) (the decisions and the
workflow), [`data_sources.md`](data_sources.md) (the registry — a source only
gets a row there once its licence text has been read first-hand).

## Method and its limits

- Searched Hugging Face, Kaggle, Roboflow Universe, GitHub, academic venues,
  Wikimedia Commons, Flickr and Russian-language sources.
- Content claims were checked against the dataset page itself where the page
  could be read. Two pages refused automated reading (HTTP 403/404), and that
  is recorded rather than filled in from a search summary.
- **Second pass, 2026-09-17.** Three of the first pass's open questions were
  closed with first-hand evidence, and two of its conclusions changed as a
  result:
  - the Kaggle Nomeroff mirror's licence was read directly and moved the source
    from REFERENCE_ONLY to **REJECT** (§8);
  - the Wikimedia Commons categories were measured through the MediaWiki API —
    licence, format and pixel size per file — which corrected the licence claim
    in our favour and the **usable-image claim sharply against it** (§6);
  - one new source (NeuroCore) was found and rejected (§17).
- **Measurement beats impression.** The Commons numbers below are counted, not
  estimated, and the count is what demoted the source. Where a figure is still
  an estimate it is marked as one.
- **Where evidence is a search summary rather than a page read first-hand, the
  confidence column says so.** A title is not evidence: none of these sources
  is called useful for `type1a` / `type1b` because its name suggests it.
- Prior decisions were re-read, not re-litigated: the AUTO.RIA audit
  (2026-09-10/13) and the two Roboflow rejections (2026-09-12/13) stand.

## Summary

17 sources investigated.

| # | Source | Status | Rare classes? |
| --- | --- | --- | --- |
| 1 | AUTO.RIA Numberplate (Nomeroff, `ria-com`) — already in the registry | **ACCEPT_FOR_SUBMISSION** | No (0 `type1a`, 0 `type1b` confirmed) |
| 2 | HF `AY000554/Car_plate_detecting_dataset` (25,632, YOLO) | **ACCEPT_FOR_SUBMISSION** (conditional) | No |
| 3 | HF `AY000554/Car_plate_OCR_dataset` (~45.5k crops) | **ACCEPT_FOR_SUBMISSION** (conditional) | No |
| 4 | Our own photographs (team capture) | **ACCEPT_FOR_SUBMISSION** | **Yes — the only reliable route** |
| 5 | Contributed photographs with a written CC BY 4.0 grant | **ACCEPT_FOR_SUBMISSION** | **Yes, if contributors deliver** |
| 6 | Wikimedia Commons, per file | **ACCEPT_FOR_SUBMISSION** (per file — **~14 usable photographs, measured**) | Marginal (1 possible `type1b`) |
| 7 | Flickr CC BY / CC0, per photo | **ACCEPT_FOR_SUBMISSION** (per photo, unverified yield) | Marginal |
| 8 | Kaggle `evgrafovmaxim/nomeroff-russian-license-plates` | **REJECT** *(changed 2026-09-17 on first-hand licence evidence)* | No |
| 9 | Roboflow `ru-anrp/russian-license-plates-detector` (2,787) | REFERENCE_ONLY | Unknown — Cloudflare-blocked, not bypassed |
| 10 | Roboflow `ru-anrp/russian-license-plate-characters-detector` (807) | REFERENCE_ONLY | No (character boxes) |
| 11 | Roboflow `new-workspace-fvqqv/russian-car-plates` (351) | REJECT | Unknown |
| 12 | Roboflow `carplates/russian-plate-kuabh` (262) | REJECT | Unknown |
| 13 | Roboflow `testcarplate/…-classification-by-this-type` (7,879) | REJECT *(prior decision, 2026-09-13)* | No |
| 14 | Roboflow `fverwfgerwf/two-line-russian-license-plates` (27) | REJECT *(prior decision, 2026-09-12)* | Claimed `type1a`, rejected on provenance |
| 15 | platesmania.com / avto-nomer.ru | REJECT (REFERENCE_ONLY for looking) | Rich, but unusable |
| 16 | ICPR26-LRLPR, UniDataPro, Mapillary, Kaggle competitions | REJECT | — |
| 17 | NeuroCore `neuro-core.ru` ANPR/OCR dataset (312,740) | REJECT | Unknown — not disclosed |

- **ACCEPT_FOR_SUBMISSION: 7** (three datasets on one licence chain, four acquisition routes).
- **TRAINING_ONLY_IF_LEGAL: 0.** Nothing landed here, and that is deliberate — see below.
- **REFERENCE_ONLY: 2** (both Roboflow `ru-anrp` projects).
- **REJECT: 8** (plus the four grouped in row 16).

**Only rows 1 and 14 appear in [`data_sources.md`](data_sources.md), and the
rest are deliberately absent.** The registry is for sources whose licence text
we have read first-hand and from which we may actually collect — AUTO.RIA
because it is cleared, the two-line Roboflow project because we downloaded it
before rejecting it. A desk-research rejection of something we never touched is
recorded here instead, following the precedent set for `testcarplate` in
`data/audits/roboflow_testcarplate_plate_types/README.md`. A source moves into
the registry at the moment it is approved for collection, not before — so a
short registry is a sign the gate is working, not a sign of missing paperwork.

## The sources

### 1–3. The AUTO.RIA / Nomeroff chain — accepted, but not for the rare classes

The one source whose rights we have already established first-hand.

| | |
| --- | --- |
| Rights holder | ARS Online OU, copyright 2018–2024 |
| Licence | **CC BY 4.0**, per `license.txt` in the source repository (read 2026-09-10) |
| Attribution | Required; the credit line is in the registry |
| Our audit | 2,564-image test split audited 2026-09-10; 0 corrupt, 2,685 YOLO boxes, 9 groups of identical images |
| Rare classes | Human review 2026-09-13: **0 `type1a`, 0 `type1b` confirmed** out of 89 shape and 54 colour candidates |
| Confidence | High on the licence; **moderate** on the underlying photographs — they are vehicle-listing images, and the chain from each photographer to ARS Online OU is asserted by the licence file, not independently evidenced |

Two Hugging Face derivatives extend the same chain:

- `AY000554/Car_plate_detecting_dataset` — **25,632 images** (20,505/2,563/2,564 train/val/test), YOLO boxes, page states CC BY 4.0 via `license.txt` and states it is based on the AUTO.RIA Numberplate Options Dataset. Page read first-hand.
- `AY000554/Car_plate_OCR_dataset` — ~45.5k **cropped plates** with text transcriptions, same stated basis. Crops, so it trains OCR but not detection.

Both say "Russian plates **of one type**" — consistent with our own finding
that the rare classes are absent.

**Conditions on the accept.** The mirror grants nothing by itself: it inherits.
Before import, pull the data from the upstream Nomeroff distribution, archive
`license.txt` beside the images (our audit noted it is absent from the
extracted split), and record ARS Online OU as the rights holder. If the
upstream file cannot be archived, the source drops to REFERENCE_ONLY.

**What this chain is good for:** `type1` at volume, detector training, and OCR
pre-training. **What it cannot do:** the two classes the competition scores us
on hardest.

### 4–5. Our own and contributed photographs — the only reliable rare-class route

We are the rights holder, so the provenance record is a statement of fact
rather than an inference, and the licence is ours to grant. This is the
cleanest possible source and the only one that can be aimed deliberately at
square/two-line and yellow plates.

Contributed photographs need the grant in writing, recorded per contributor in
`data/real_staging/source_records/`. One contributor equals one `source_id`.
**A verbal "sure, use it" is not a grant** — not because we distrust anyone,
but because the record has to stand up without the contributor present to
explain it.

The wording to send a contributor, and how each answer lands in the source
record, is in
[`real_data_intake.md`](real_data_intake.md#contributed-photographs-the-grant).

### 6–7. Wikimedia Commons and Flickr — per-file, small, but genuinely clean

Both host individually licensed photographs, so provenance is per file and the
licence is the uploader's own statement.

**Measured 2026-09-17 via the MediaWiki API** — licence, file type and pixel
size for every file in the four relevant categories. No image was downloaded;
only metadata was read.

| Category | Files | Gate-compatible licence | Of those, real photographs >200 px |
| --- | --- | --- | --- |
| `Category:License plates of Russia` | 179 | **143** | **9** |
| `Category:Diplomatic license plates in Russia` | 21 | 8 | **5** |
| `Category:Military license plates of Russia` | 7 | 4 | 0 |
| `Category:Trailer license plates of Russia` | 4 | **0** | 0 |
| **Total** | **211** | **155** | **14** |

Two findings, pulling in opposite directions:

- **The licensing is much better than assumed.** The main category is
  CC0-dominated, not ShareAlike-dominated: 123 CC0, 14 public domain, 6 CC BY,
  against 36 CC BY-SA. 80 % of it clears our gate. The first pass's claim that
  "Commons' most common licence is CC BY-SA" was wrong for this category and is
  corrected here.
- **The usable yield is far worse than assumed, and this is what decides it.**
  Of the 143 gate-compatible files in the main category, **114 are 131×110
  region-code graphics** — the `NN регион. (Россия). <oblast>.jpg` series, one
  per region. They are illustrations of region codes, not photographs of
  vehicles. Most of the rest are PNG renderings of the plate standard. Only
  **9 files** are gate-compatible photographs above 200 px wide.

**The whole Commons Russian-plate tree yields roughly 14 usable real
photographs.** The good news inside that number: the 5 diplomatic-vehicle
images are CC0 full vehicle scenes at 3,600–4,386 px (`Moscow, Fiat Ducato,
Embassy of Italy…` and similar) — genuinely excellent `other` material, and CC0
asks nothing of us, not even attribution. One main-category file, *Ситроен
Маршрутное такси в Омске* (CC0, 2,421×1,127), is a route taxi and may carry a
`type1b` plate; it is worth one person's look.

Note the trap this measurement caught: `Category:Trailer license plates of
Russia` has 4 files and **all 4 are CC BY-SA**, so a category that reads as
tailor-made for `other` contributes nothing.

Flickr's CC BY and CC0 pools are searchable by licence and usable on the same
per-photo basis. Yield for Russian taxis and square plates is **unverified** —
it was not measured, and after the Commons result it should not be assumed.
Plate legibility in tourist photographs is the real limit.

A dozen clean images is worth having, and worth ten minutes, not a day. It is
not a path to 300 `type1a`.

### 8. Kaggle `evgrafovmaxim/nomeroff-russian-license-plates` — REJECT

**Status changed 2026-09-17: REFERENCE_ONLY → REJECT**, on evidence read
first-hand. The first pass could not render the page; the page's own embedded
description was retrieved on the second pass and says, in full:

> \# Dataset
> This dataset was created by Evgrafov Maxim
>
> Released under GNU Lesser General Public License 3.0
>
> \# Contents

Two independent grounds for rejection, either sufficient:

1. **The licence is incompatible and inapplicable.** LGPL-3.0 is a *software*
   licence; applied to a photograph collection it is a category error, and it
   is copyleft — it cannot be relicensed into a plain CC BY 4.0 dataset. This
   is the ShareAlike trap in a different costume, and `src/dataset_meta.py`
   would not recognise it as redistributable either.
2. **The provenance contradicts itself.** The dataset is *titled* for Nomeroff
   — the AUTO.RIA project owned by ARS Online OU and released under CC BY 4.0
   — while the page claims it was "created by Evgrafov Maxim" and relicenses it
   under LGPL-3.0. Both cannot be true. If it is a Nomeroff re-upload, the
   uploader relicensed someone else's CC BY 4.0 data under a licence he had no
   right to apply; if it is genuinely his own work, the name misattributes it.
   Either way the rights chain is broken at the point we would rely on it.

The "# Contents" heading is followed by nothing: the page documents no image
count, no format and no annotation scheme. There is no version of this source
worth taking over the upstream Nomeroff distribution (§1–3), which is properly
licensed and which we have already audited.

### 9–10. Reference only — could not be verified first-hand

- **Roboflow `ru-anrp/russian-license-plates-detector`** (reported 2,787
  images) and **`ru-anrp/russian-license-plate-characters-detector`** (reported
  807 images, 22 character classes) — the dataset pages returned HTTP 403 to
  automated reading. Retried 2026-09-17: the 403 is a **Cloudflare bot
  challenge** (`__cf_chl_tk` token in the response body), the same barrier the
  `testcarplate` investigation hit on 2026-09-13.
  **We did not bypass it**, because `real_data_intake.md` forbids obtaining
  material by circumventing an access restriction — a rule that is worth
  nothing if we suspend it the moment it costs us something.
  The workspace handle names no person or organisation, and no upstream
  photograph source is stated anywhere we could read. Even if the page declares
  CC BY, that is the uploader's claim about images whose origin is not
  established. Both are detector/character-box datasets, so neither offers
  rare-class value: reading the page could at best move them to REJECT, never
  to ACCEPT. That is why this is left unresolved rather than pursued — the
  answer cannot change what we do next.

### 11–14. Rejected Roboflow projects

- `new-workspace-fvqqv/russian-car-plates` (351) and `carplates/russian-plate-kuabh`
  (262): anonymous workspaces, no named creator, no upstream source, small.
  Rejected for submission on provenance.
- `testcarplate/russian-license-plates-classification-by-this-type` (7,879):
  **prior decision 2026-09-13, REJECT** on provenance — the project's images
  are reported to come from platesmania.com and migalki.net, both sites of
  user-uploaded photographs. Secondary finding then and now: its two classes
  (`n_p`, `p_p`) are not our rare classes, and the author's own roadmap listed
  taxi plates as *future* work. Nothing new changes this.
- `fverwfgerwf/two-line-russian-license-plates` (27 images): **prior decision
  2026-09-12, REJECT** on provenance. It is the only public dataset that
  actually claims `type1a`, and 27 images would not meet the minimum anyway.
  Unchanged.

### 15. Plate-spotting sites — rich and unusable

platesmania.com (and its Russian sibling avto-nomer.ru) hold exactly what we
lack: tens of thousands of well-framed Russian plates including yellow
transport and square plates. The site's terms permit copying **only with an
active link back**, and the photographs belong to individual uploaders, not the
site. There is no CC grant and no chain to the photographers. That is a
rejection for submission, and we do not "train only" on it either: the images
are not offered under any licence that contemplates it.

Useful for one thing: looking, to learn what the rare plates look like in the
wild before a capture session.

### 16. Other rejections

| Source | Why |
| --- | --- |
| **ICPR26-LRLPR** (ICPR 2026 competition dataset) | Academic, non-commercial, access by signed licence agreement; redistribution and modification are explicitly forbidden. Fails the gate outright. |
| **UniDataPro `license-plate-detection`** (HF) | **CC BY-NC-ND 4.0** — non-commercial *and* no derivatives; the full 1.9M set is a paid product. Fails twice over. |
| **Mapillary** | CC BY-**SA** (our gate rejects SA), and Mapillary automatically blurs license plates. The one thing we need is removed by design. |
| **Kaggle competition data** (e.g. ABBYY plate-recognition hackathon) | Competition datasets are licensed for that competition; no redistribution right. |

### 17. NeuroCore ANPR/OCR dataset — REJECT

Found 2026-09-17 through a Russian-language search; page read first-hand.
<https://neuro-core.ru/services/data-labeling/anpr-ocr-data>

A Russian commercial data-labelling lab offering **312,740 images** (156,370
plate crops and 156,354 full vehicle frames), stated to be collected from real
surveillance cameras. On size and composition it is the most substantial thing
this survey found.

It is rejected anyway, on terms rather than content:

- **No licence is published at all** — not a permissive one, not a restrictive
  one. The dataset is request-gated behind a "Запросить датасет" enquiry form,
  which makes it a commercial product negotiated per customer.
- **No redistribution grant exists to inherit.** A negotiated commercial
  licence would have to grant public CC BY 4.0 redistribution of 312,740
  third-party camera images; a vendor whose asset is the data has every reason
  not to, and no page suggests they would.
- **The plate types are not disclosed.** Nothing states whether `type1a` or
  `type1b` appear, so even the commercial route is a gamble on our actual
  bottleneck.

Reopenable only on a written CC BY 4.0 redistribution grant from NeuroCore.
Given the deadline, this is not a route to pursue — it is recorded so nobody
re-finds it and wonders.

## Rare-class candidate tables

### TYPE1A — square / two-line white plates

| Candidate | Usable real images | Unique plates | Scenes or crops | Annotation | Provenance | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| Team capture | as many as we shoot; **300 is realistic over 2–3 targeted sessions** | 1 per vehicle | full scenes | ours to make, exact | **ours — highest** | medium: imports, SUVs/pickups, trailers, car markets, ferry/rail terminals |
| Contributed photos | unknown | unknown | full scenes | must be annotated by us | high **if** the grant is recorded | low per image, high coordination |
| Commons / Flickr per file | **~0 measured** — the Commons tree holds 14 usable photographs and none is a confirmed `type1a` | — | mixed | none, we annotate | high per file | medium: per-file licence checks |
| Roboflow `two-line-…` (27) | 0 usable | — | crops/scenes unknown | single class | **rejected** | — |
| AUTO.RIA chain | **0 confirmed** | — | scenes | boxes only | accepted | — |

**No public, redistributable dataset supplies real `type1a` at the required
scale.** Saying otherwise would require lowering the provenance standard, and
the one dataset that claims the class is 27 images with unclear rights.

### TYPE1B — yellow one-line transport plates

| Candidate | Usable real images | Unique plates | Scenes or crops | Annotation | Provenance | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| Team capture | **500 is realistic**; taxi ranks and bus stations give high density | 1 per vehicle | full scenes | ours, exact | **ours — highest** | low–medium: stations, ranks, termini, from public space |
| Contributed photos | unknown | unknown | full scenes | ours to annotate | high if granted | low per image |
| Commons / Flickr per file | **1 candidate, measured** (a CC0 Omsk route-taxi photograph); Flickr unmeasured | ~1 | scenes | none | high per file | low: one file to check |
| Roboflow `testcarplate` | 0 (class not annotated; rejected anyway) | — | — | — | **rejected** | — |
| AUTO.RIA chain | **0 confirmed** | — | scenes | boxes only | accepted | — |

Same conclusion, slightly less severe: `type1b` is easier to photograph than
`type1a`, but no public source delivers it under a licence we can republish.

## Submission data vs training-only data

- **A — may go into the submitted CC BY 4.0 dataset:** our own photographs,
  properly granted contributions, the AUTO.RIA chain (with `license.txt`
  archived), and individually verified CC0/CC BY files from Commons or Flickr.
- **B — might be lawful to train on but cannot be redistributed:** nothing is
  currently placed here. Every candidate that fails redistribution also fails
  on a second ground — no licence contemplating training (plate-spotting
  sites), an explicit no-derivatives or non-commercial clause (UniDataPro), a
  signed academic agreement we are not party to (ICPR26-LRLPR), or blurred
  plates (Mapillary). Keeping this category empty also keeps the dataset
  simple: everything we train on, we can ship.
- **Not to be used at all:** platesmania/avto-nomer scraping, the official
  30-image debug set as training data, and anything whose provenance we cannot
  state in one sentence.

## Gap analysis against the targets

| Class | Target (min / preferred) | Covered by verified sources | Gap |
| --- | --- | --- | --- |
| `type1` | no documented minimum / 400 | **AUTO.RIA chain: thousands** (after filtering to Russian one-line plates) | **None** — this class is solved |
| `type1a` | **150 / 300** | ~0 | **150–300 images, 50–120 unique plates** |
| `type1b` | **300 / 500** | ~0 | **300–500 images, 100–200 unique plates** |
| `other` | **50 / 150** | **5 CC0 Commons vehicle photographs** (measured, not 32) + trivially self-captured negatives | **~45–145 images**, low risk |

**The biggest risk is `type1a`.** It is a competition-scored class, the hardest
to photograph (the plates are rare on the street by nature), and the only
public dataset that claims it is 27 images we rejected on provenance. `type1b`
is the same problem one notch easier. Everything else is either solved or cheap.

The second pass **hardened this conclusion rather than softening it**. The only
candidate that looked like a fallback — Wikimedia Commons, on a file count of
179 — turned out to hold 14 usable photographs once measured, and the Kaggle
mirror that might have been a second licensed route turned out to be LGPL-3.0
with a broken provenance chain. Both open questions closed against us. Nothing
found in two passes changes the answer: **the rare classes come from our own
camera or they do not come at all.**

## Recommendation

**Start with our own targeted capture for `type1a`, one `source_id`, one
session.** It is the only route that is simultaneously real, aimed at the
scarce class, unambiguously redistributable, and annotatable to our own
standard. The AUTO.RIA chain is already accepted and can be imported later for
`type1` bulk and OCR pre-training — it is not urgent, because that class is not
where we are short.

Legal basis: we are the photographer and the copyright holder; the grant is
ours to make; the record states exactly that, with no third-party chain to
verify. Photographing vehicles from public space is the ordinary case the
privacy rules in `real_data_intake.md` already cover.

Expected yield for a first session, from the scene list in
`dataset_strategy.md`: **40–80 `type1a` images across 15–30 unique vehicles**
at a large shopping-centre car park or a car market, plus incidental `type1`
and `other` frames at no extra cost. Reaching 300 needs roughly four to six
such sessions, which is why this should start now rather than after the
model work.

## Next action, once approved

1. Write the source record (this is the only step that touches the repository):

   ```bash
   python scripts/audit_real_source.py --blank-record team-capture-2026-09 > \
       data/real_staging/source_records/team-capture-2026-09.json
   ```

   Fill in: creator = the team member who shoots; licence = CC BY 4.0;
   evidence = "original camera files retained"; `privacy_review: not_started`;
   `decision: PENDING` until the session is reviewed.

2. Shoot one session targeting square/two-line plates — imports, pickups, SUVs,
   trailers, agricultural and construction vehicles — with deliberate variety in
   angle, distance and lighting, and both plate lines readable in most frames.

3. Stage into
   `data/real_staging/incoming/team-capture-2026-09/images/real/team-capture-2026-09/`,
   privacy pass, annotate, record capture groups, then
   `python scripts/audit_real_source.py data/real_staging/incoming/team-capture-2026-09`.

No download is part of this. The AUTO.RIA import is a separate decision for
later, and it needs its own approval.

## Sources consulted

- AUTO.RIA / Nomeroff Net: <https://github.com/ria-com/nomeroff-net>
- <https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset>
- <https://huggingface.co/datasets/AY000554/Car_plate_OCR_dataset>
- <https://huggingface.co/datasets/UniDataPro/license-plate-detection>
- <https://www.kaggle.com/datasets/evgrafovmaxim/nomeroff-russian-license-plates> (description read 2026-09-17: LGPL-3.0)
- <https://universe.roboflow.com/ru-anrp/russian-license-plates-detector/dataset/1> (HTTP 403 — Cloudflare challenge, not bypassed)
- <https://neuro-core.ru/services/data-labeling/anpr-ocr-data>
- Wikimedia Commons MediaWiki API, `categoryinfo` and `imageinfo/extmetadata`
  for `Category:License plates of Russia` and its subcategories (measured
  2026-09-17; metadata only, no image downloaded)
- <https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type>
- <https://universe.roboflow.com/fverwfgerwf/two-line-russian-license-plates/dataset/1>
- <https://universe.roboflow.com/new-workspace-fvqqv/russian-car-plates>, <https://universe.roboflow.com/carplates/russian-plate-kuabh>
- <https://commons.wikimedia.org/wiki/Category:License_plates_of_Russia>
- <https://platesmania.com/about>, <https://platesmania.com/rules>
- <https://icpr26lrlpr.github.io/>, <https://icpr26lrlpr.github.io/license-agreement.pdf>
- <https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data>, <https://help.mapillary.com/hc/en-us/articles/115001663705-Blurring-images-on-Mapillary>
- <https://www.flickr.com/creativecommons/by-2.0>
