# Capture session 1 — `type1a`

> ## Status: DEFERRED — `WAITING_FOR_HUMAN_CAPTURE` (2026-09-19)
>
> **This acquisition is not the active one, and nothing here has been
> cancelled.** It was prepared in full on 2026-09-17 and stopped at the point
> where a camera is required. It stays available as a fallback.
>
> **Why it was deferred.** The competition scores us on genuine Russian
> `type1a` and `type1b` plates, and the participant is in **Bulgaria**. Local
> self-capture cannot realistically collect Russian rare-class plates at any
> useful scale: the vehicles are not on the street here. The constraint is
> geography, not effort, and no amount of shooting in Sofia fixes it.
>
> **What replaced it.** Online discovery of already-published, per-image
> licensed photographs — see
> [`online_source_discovery.md`](online_source_discovery.md). That pass found
> real, verified Russian `type1b` photographs under CC0 and CC BY, which is
> exactly what a local camera could not reach.
>
> **What would revive it.** A team member travelling to Russia, or a
> contributor there who can shoot to this checklist. The source record
> `team-capture-2026-09` stays `PENDING`, its staging folder stays in place,
> and everything below still applies the day someone can use it.

The field checklist for **real-data acquisition #1**, approved 2026-09-17 on the
recommendation in [`real_data_source_research.md`](real_data_source_research.md)
(commit `0dafaa7`): one targeted team capture session for square / two-line
white plates, staged as `source_id` **`team-capture-2026-09`**.

**No photograph exists yet.** The source record, the staging folder and the
annotation templates are prepared and waiting. This page is what turns them into
data.

Related: [`real_data_plan.md`](real_data_plan.md) (targets and condition
quotas), [`real_data_intake.md`](real_data_intake.md) (the 10-step workflow),
[`annotation_guide.md`](annotation_guide.md) (labelling),
[`dataset_strategy.md`](dataset_strategy.md) (the scene list this is drawn from).

## Why this class, and why now

`type1a` is the project's single biggest risk. It is scored by the competition,
it is the hardest plate to find on the street, and two passes of source research
found **no public, redistributable dataset that supplies it**: the one project
that claims the class is 27 images rejected on provenance. The class comes from
our own camera or it does not come at all. Four to six sessions like this one
reach the preferred target of 300 images, so the first one should happen before
the model work, not after it.

## What a `type1a` plate looks like

**White plate, square or two-line.** Both lines belong to one plate.

```
    ┌──────────────┐
    │   А 123 ВС   │     ← characters on the upper line
    │    ▌ 116     │     ← region code on the lower line
    └──────────────┘
```

Fitted where a long plate does not fit. Look for:

- Japanese and American **imports** — the most reliable single indicator;
- **SUVs and pickups**, especially with a bull bar, winch or front-mounted spare;
- **trailers** with a square mount, and caravans;
- **agricultural and construction vehicles** — tractors, excavators, loaders;
- older stock and re-registered vehicles.

**Not `type1a`:** an ordinary long white plate (`type1`), a yellow plate
(`type1b`), a yellow *two-line* plate (`other`), motorcycle plates (`other`).
Photograph those anyway when they are in front of you — they annotate as `type1`
or `other` and cost nothing — but they are not what the session is hunting.

## Where to go

From the scene list in `dataset_strategy.md`, best hit rate first:

1. **Shopping-centre and hypermarket car parks** — highest yield per hour, and
   legal to photograph from public space. Start here.
2. **Car markets, import dealerships, tuning and off-road shops.**
3. **Residential courtyards and long-stay street parking.**
4. **Truck stops and industrial estates** — the trailer and agricultural
   variants.
5. **Ferry and rail terminals**, where imports queue.

One large car park plus one car market is a realistic single session.

## How much to shoot

| | Target |
| --- | --- |
| Unique vehicles with a `type1a` plate | **15–30** |
| `type1a` images | **40–80** |
| Frames per vehicle | **3–5**, each genuinely different (see below) |
| Incidental `type1` / `other` frames | as many as turn up — free, and they count |

Three to five frames per vehicle is the number that matters. Ten near-identical
frames of one car add one plate string to the OCR and one group to the split;
they do not add ten images of value.

**Stop at 30 vehicles even if the car park is generous.** A second session
elsewhere, on another day, in other light, is worth more than a longer first one.

## The frames, per vehicle

Vary these deliberately — the variation is the point, not the vehicle count:

| Axis | What to do |
| --- | --- |
| **Angle** | One straight-on, then two or three oblique — roughly 20–45° left and right. Aim for **40–50 %** of all frames to be noticeably angled. A two-line plate at an oblique angle loses its lower line first: that is the hard case, so shoot it on purpose. |
| **Distance** | Near (plate > 200 px wide) ~30 %, mid ~45 %, far (plate 30–80 px) ~25 %. The far frames are where detection fails first, so do not skip them because they look bad. |
| **Height** | Not everything from standing eye level. Crouch for some, shoot down over a bonnet for others. |
| **Tilt / roll** | A few frames with the camera deliberately rolled 10–20°. |

Both plate lines readable in **most** frames — but not all. Some hard frames are
the point.

### Conditions to hunt

Per-axis targets from `real_data_plan.md`, applied to this class:

| Condition | Share | How to get it |
| --- | --- | --- |
| `day` | 60–70 % | The default. |
| `night` | **30–40 %, never below 25 %** | Come back to the same car park after dark. **This is the axis most likely to be skipped and the one most likely to decide the result.** |
| `angle` | 40–50 % | Covered above. |
| `dirt` | 15–20 % | Mud, road salt, faded paint, bent plates. Industrial estates and truck stops are full of it. |
| `glare` | 15–20 % | Direct sun on the retroreflective surface; headlights at night. Shoot into low sun on purpose. |
| `motion_blur` | 10–15 % | A moving vehicle, or a slow shutter handheld. |
| Partial occlusion | 10–15 % | Tow bars, bike racks, plate frames, another vehicle, a thick plate holder. These become `#` characters and validate that our occlusion policy transfers to real photos. |
| `rain` / `snow` | opportunistic | If the weather offers it, take it. Do not wait for it in September. |

### Cameras

- Use **at least two different devices** in this session if two are available.
  The class-level target is **≥ 3 distinct cameras, none above 60 % of the
  class** — one phone's sensor, lens and JPEG pipeline is one domain, and a
  model tuned to it fails on the jury's images.
- Shoot **at least two capture resolutions**. Do not let the plate's pixel
  height be constant.
- **Shoot JPEG, not HEIC.** If the phone only produces HEIC, convert to JPEG on
  import and keep the HEIC originals — the intake tooling reads
  `.jpg/.jpeg/.png/.bmp`.
- **Turn off** beautification, HDR-heavy "scene optimisation", and any auto
  filter. We want the sensor, not the marketing.
- Leave EXIF intact. It is provenance.

## What NOT to photograph

Hard rules. Any one of these costs more than the images are worth.

- **No person as the subject.** We photograph vehicles. A passer-by in the frame
  is fine and will be blurred at intake; a portrait with a car behind it is
  rejected, and no blur makes it acceptable.
- **No children as a subject**, in any framing.
- **Nothing that requires getting past an access control.** Public space and
  publicly accessible car parks only. No fences, no barriers, no "private
  property" signs, no restricted yards, no depots without permission. The same
  rule that made us leave a Cloudflare challenge alone applies with a camera.
- **Do not photograph where photography is posted as prohibited**, and stop if
  anyone in authority asks. An image is never worth an argument.
- **No plate-spotting sites, no screenshots, no re-photographing someone else's
  photo.** The whole point of this source is that we hold the rights.
- **Do not photograph people's homes, windows or interiors** to reach a plate.
- **No editing in the field or after.** No crops, no filters, no straightening,
  no "cleaning up" a plate. The only permitted transformation is the privacy
  blur, applied at intake and recorded.
- **Do not delete anything**, including frames you think are bad. Blurred, dark
  and distant frames are training data; that judgement is made at intake, not in
  the car park.

A plate on a poster, a screen, a shop window or a vehicle wrap **is** worth a
frame — deliberately, as a negative. It annotates `is_vehicle=false`,
`plate_type=other`. Just do not mistake it for a real plate.

## Tracking capture groups in the field

**This is the one piece of bookkeeping that cannot be reconstructed afterwards.**

A group is whatever would leak if it were split across train and holdout: **one
vehicle, one physical plate, one burst.** Six angles of the same import are
**one** group, not six. Two vehicles that happen to be photographed a minute
apart are two groups.

In the field, keep it simple: **shoot each vehicle's frames consecutively, and
never interleave two vehicles.** Then the grouping is recoverable from the
filename order at intake, and `groups.csv` is quick to fill in honestly.

If you double back to a vehicle you already shot — a second visit, better light —
it is **still the same group**. Note it in the capture log.

## Afterwards: where the files go

Copy the card, do not move it, and do not rename in the camera.

**1. Originals, untouched:**

```
data/real_staging/incoming/team-capture-2026-09/originals/
```

Straight off the device, original filenames, EXIF intact. **Never edited, never
deleted.** This is the provenance evidence the source record points at.

**2. Working copies, for annotation:**

```
data/real_staging/incoming/team-capture-2026-09/images/real/team-capture-2026-09/
```

JPEG, named `tc202609_0001.jpg` upward in capture order. This path is already
the path the image will have inside `dataset/`, so promotion is a straight copy
with no path rewriting.

**3. Fill in the capture log** (`capture_log.csv`, already created with its
header) — one row per group:

```
group;vehicle;place;date;time_of_day;camera;notes
```

Then tell me the files are there, and I will run the privacy pass, the
annotation, the grouping and the intake audit.

## What happens after that — the rest of the workflow

Steps 3–10 of [`real_data_intake.md`](real_data_intake.md), none of which have
run yet:

1. **Privacy pass** — every image reviewed, faces blurred or covered, people-centred
   images rejected, one `privacy_review.csv` row per image even when the answer is
   `none_needed`.
2. **Annotation** — manually, into `meta.csv`, per `annotation_guide.md`. Never
   auto-labelled: a detector's output is a suggestion to a human, never an
   annotation.
3. **Groups** — `groups.csv` from the capture log.
4. **Intake audit** —
   `python scripts/audit_real_source.py data/real_staging/incoming/team-capture-2026-09`
   with its JSON and text reports written to `data/real_staging/audits/`.
5. **Human review** — a contact sheet and a PASS / QUESTIONABLE / FAIL summary
   for a person to read. **Promotion is held on any class-label, privacy,
   provenance or licensing doubt.**
6. **Decide** — the source record moves from `PENDING` to
   `ACCEPT_FOR_SUBMISSION` only when a person writes it there.
7. **Freeze the split**, then promote into `dataset/images/real/`, then
   `validate_dataset_local.py --strict` and `plan_real_splits.py --check`.

Nothing enters `dataset/` before step 6. That is the one rule the whole intake
process exists to enforce.
