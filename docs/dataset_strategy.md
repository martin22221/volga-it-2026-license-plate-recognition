# Dataset strategy

Acquisition plan for the training dataset. Written 2026-09-09, before any
image has been collected.

Related: [`../dataset/README.md`](../dataset/README.md) (format and legal
rules), [`data_sources.md`](data_sources.md) (source registry),
[`annotation_guide.md`](annotation_guide.md) (how to label).

## Standing constraints

- The official 30-image debug set is **unavailable** — the organizer's
  ownCloud share is empty and we have emailed them. We do not wait for it and
  we do not substitute unofficial data for it. When it arrives it is used for
  *checking* our pipeline, never for training, and it is never copied into
  `dataset/`.
- Every real image needs a documented source and license before it counts
  towards any target below. An image without provenance is not "collected
  but undocumented" — it is not collected.
- No closed or private sources, no scraping that bypasses access restrictions.
- Faces blurred or covered before the image enters `images/real/`.

## Targets

The competition minimums, and the targets we actually plan for. The recommended
figures carry the headroom needed for a clean validation split and for images
that get discarded on review.

| Category | Minimum (real images) | Minimum (unique plates) | **Recommended (real images)** |
| --- | --- | --- | --- |
| `type1a` — square / two-line white | 150 | 50 | **220–300** |
| `type1b` — yellow one-line transport | 300 | 100 | **400–500** |
| `other` — negatives and out-of-scope | 50 | — | **100–150** |
| Synthetic (all types) | 5 000 | — | **8 000–12 000** |

Unique-plate targets scale with the image targets: aim for **≥ 80 unique
plates** for `type1a` and **≥ 150 unique plates** for `type1b`. Unique plates
matter more than raw image count — 300 photographs of 20 vehicles teach the OCR
those 20 strings, not the alphabet.

Practical caps to keep the set honest:

- **No more than 4–5 images of the same physical plate**, and only when they
  differ meaningfully (different angle, light, distance).
- **No near-duplicate frames** pulled from consecutive video frames.
- Hold out roughly **15 % of real images as a validation split**, split by
  *plate*, not by image, so the same plate never appears on both sides.

## Condition distribution

The same distribution applies to `type1a` and `type1b`. Percentages are of the
real images in that category and overlap — one image can carry several tags.

| Axis | Target | Notes |
| --- | --- | --- |
| `day` | 60–70 % | Includes overcast, which is the easy majority case. |
| `night` | 30–40 % | Artificial light, headlights, retroreflective glare. This is where most solutions fail; do not let it fall below 25 %. |
| `angle` | 40–50 % | Yaw beyond ±15°, or pitch from a high/low camera. |
| Straight-on | ~50 % | Untagged. The baseline case, and what the generator produces by default. |
| Distance: near | ~30 % | Plate ≥ 200 px wide. |
| Distance: mid | ~45 % | Plate 80–200 px wide. |
| Distance: far | ~25 % | Plate 30–80 px wide. Below 30 px, annotate only if characters are still legible; otherwise it is an `other`/`#` case. |
| `dirt` | 15–20 % | Mud, road salt, snow crust, faded paint, bent plates. |
| `glare` | 15–20 % | Direct sun, headlights, flash on the retroreflective surface. |
| `rain` | 10–15 % | Wet plate, droplets on lens, spray. Opportunistic — collect when the weather offers it. |
| `snow` | 10–15 % | Same. Realistically this fills in during winter; do not block on it. |
| `motion_blur` | 10–15 % | Moving vehicle or handheld camera at a slow shutter. |
| Partial occlusion | 10–15 % | Tow bar, bicycle rack, another vehicle, a person, plate frame, snow covering 1–2 characters. Annotate with `#` for the covered characters. |

If a real condition is genuinely unavailable (a season we are not in), close the
gap with **synthetic** images tagged for that condition, and record the shortfall
here rather than quietly leaving the axis empty.

## Per-category plan

### `type1a` — square / two-line white plates

The scarce class. These are fitted where a long plate does not fit: many
Japanese and American imports, some SUVs and pickups, trailers with a square
mount, agricultural and construction vehicles, and older stock.

Scenes to work:

- Parking lots of shopping centres and hypermarkets — the highest hit rate per
  hour, and legal to photograph from public space.
- Residential courtyards and long-stay street parking.
- Car markets, dealerships for imports, tuning and off-road shops.
- Truck stops and industrial estates for the trailer/agricultural variants.
- Ferry and rail terminals, where imports queue.

Notes:

- Both plate lines must be readable. A two-line plate at an oblique angle loses
  the lower line first — that is exactly the hard case, so collect some
  deliberately, tagged `angle`.
- Expect this class to be the bottleneck. If field collection stalls below 150,
  the gap is closed with synthetic `type1a` images, and the shortfall in real
  images is reported here.

### `type1b` — yellow one-line passenger transport plates

Taxis, minibuses, route buses, and other licensed passenger transport. Higher
density than `type1a` and much easier to reach the target.

Scenes to work:

- Bus stations, terminals, layover and turnaround points — many vehicles,
  stationary, plates at a consistent height.
- Taxi ranks at airports, railway stations and hotels.
- Route termini and depots, photographed from public space only.
- Ordinary street traffic for the moving, blurred, distant cases.

Notes:

- The yellow background changes contrast behaviour: at night under sodium or
  LED light the yellow can wash out towards white, and in strong sun it
  saturates. Collect both — this is the main reason `type1b` needs a *higher*
  count than `type1a` despite being easier to find.
- Vary the vehicle body colour. A depot visit yields 40 plates on 40 identical
  yellow minibuses; that is one background, not forty.
- Bus and minibus plates are often mounted low and get dirtier than car plates.
  Good source of natural `dirt` samples.

### `other` — negatives and out-of-scope

Two distinct jobs, both needed:

1. **Out-of-scope plates** (~60 % of the category): standard white one-line
   `type1` plates, motorcycle plates, trailer plates, military, diplomatic,
   transit, and foreign plates. These teach the classifier the boundary of the
   target classes.
2. **True negatives** (~40 %): images with no plate at all but with things that
   look like one — road signs, advertising panels, house numbers, shop signage,
   text on vehicle bodies, printed banners, and plates displayed on screens,
   posters or storefronts. These are recorded as background rows (see
   `dataset/README.md`). This is what stops the detector firing on rectangles
   of text.

Collect these opportunistically while working the scenes above — they cost
almost no extra time.

### Synthetic

The generator lives in `dataset/generator/` and is **not built yet** (no images
are generated at this milestone). When it is, it should cover:

- All three target layouts, with `type1a` deliberately over-represented
  relative to its real-world frequency, since that is where real data is
  thinnest.
- Plate strings sampled across the full legal character and region space, not
  just the regions we happen to photograph. This is the main thing synthetic
  data buys us for OCR.
- The same condition axes as the table above, applied as augmentations:
  perspective warp, motion blur, defocus, noise, JPEG artefacts, brightness and
  white-balance shifts, simulated dirt and partial occlusion, glare hotspots,
  rain and snow overlays.
- Composited onto varied real backgrounds at varied scales, not rendered on a
  flat colour — a plate on a blank field teaches the detector nothing.

Split the budget roughly `type1a` 40 % / `type1b` 40 % / `type1` and other
layouts 20 %.

Synthetic images always carry `is_synthetic=true` and live under
`images/synthetic/`; they need no `source` or `license`.

## Workflow

1. Register the source in `docs/data_sources.md` **before** collecting from it,
   and check its license terms then, not later.
2. Collect. Blur faces. Store under `dataset/images/real/<source_id>/`.
3. Annotate per `docs/annotation_guide.md`, appending rows to `meta.csv`.
4. Run `python scripts/validate_dataset_local.py` and fix every error.
5. Review the report's condition and plate-type tables against the targets
   above, and steer the next collection session at whatever is thinnest.

## Open questions

- The license for our own contributions (`dataset/LICENSE`) is drafted as
  CC BY 4.0 and needs team confirmation.
- Whether the competition rules cap dataset size or require the dataset to be
  published — this affects how aggressively we can use sources whose licenses
  forbid redistribution.
