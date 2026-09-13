# Annotation guide

How to turn an image into rows of `dataset/meta.csv`. Read
[`../dataset/README.md`](../dataset/README.md) first for the field list.

Run `python scripts/validate_dataset_local.py` after every annotation session.
It catches most mistakes described here mechanically.

## The one rule that decides everything

**Annotate what you can see, not what you know.** If you recognise the vehicle
and remember its plate, but three characters are hidden behind a tow bar, those
three characters are `#`. The model has to learn from the pixels.

## Bounding box

`bbox_x`, `bbox_y`, `bbox_w`, `bbox_h` — pixels, origin at the image's
top-left, y increasing downwards.

- Draw the **tightest axis-aligned rectangle that contains the entire plate**,
  including its border and the blue region panel on the right. Exclude the
  frame or holder the plate is mounted in, and exclude the bolts.
- If the plate is rotated, the box still stays axis-aligned, so it will contain
  some surrounding bodywork. That is correct — the quad carries the rotation.
- If the plate runs past the image edge, clip the box to the image. Never use
  negative coordinates or extend past the width/height.
- A plate more than ~50 % outside the frame is not annotated as a target; treat
  the image as `other` unless another plate in it qualifies.
- Width and height must be strictly positive.

## The four corners

`quad_x1..quad_y4` — the plate's actual corners, used to rectify perspective
before OCR.

**Order: clockwise, starting from the plate's own top-left corner.**

```
        1 ─────────────► 2
        ▲                │
        │   А 123 ВС  77 │
        │                ▼
        4 ◄───────────── 3
```

- Corner 1 is the top-left **of the plate**, not of the image. On a plate
  rotated 30° clockwise, corner 1 is still the corner that is top-left when the
  plate is viewed upright. Follow the plate, not the screen.
- Corners 2, 3, 4 then follow clockwise: top-right, bottom-right, bottom-left.
- For a plate photographed straight on, the quad is the four corners of the
  bbox.
- For a **two-line `type1a`** plate, the quad encloses the whole plate — both
  lines together — not one line each.
- Place corners on the plate's outer edge, the same edge the bbox touches.
- Coordinates must be non-negative and no two corners may coincide.

The validator rejects a counter-clockwise winding, so if it complains about
"counter-clockwise", you have listed the corners in the wrong direction —
reverse corners 2 and 4. It also warns when the quad's extent disagrees with
the bbox by more than 25 % of a side, which almost always means one of the two
was drawn on a different plate.

## Assigning `plate_type`

Decide on **layout and colour**, not on what the vehicle is.

| See this | Use |
| --- | --- |
| White plate, single line, standard car layout | `type1` |
| White plate, **square or two-line** | `type1a` |
| **Yellow** plate, single line | `type1b` |
| Anything else, or unrecognisable | `other` |

`type1a` and `type1b` are the competition's target classes; `type1` and `other`
are context that teaches the classifier the boundary.

**Annotate every `type1` plate you photograph.** The class is supported by the
pipeline and stays in the schema, so ordinary white plates keep their
recognition quality — they are just not a class we go hunting for, since they
turn up for free in every scene. Label them `type1`, never `other`.

Goes to `other`: motorcycle plates, trailer plates, military (black),
diplomatic, transit, police, foreign plates, and any plate too small, too
blurred or too occluded to classify confidently.

A yellow **two-line** plate is `other` — `type1b` is the one-line form. When
you are torn between two classes, choose `other` and add a note in the commit
message rather than guessing.

## Writing `plate_num`

- **Latin transliteration, uppercase, no spaces or hyphens.** `А123ВС77`
  becomes `A123BC77`. Only the twelve letters `A B E K M H O P C T Y X` occur
  on Russian plates; they are the Cyrillic letters that look identical to Latin
  ones, so the transliteration is unambiguous. `src/validator.py` enforces this.
- **Include the region code**, 2 or 3 digits, with no separator.
- Do not include the country code, the flag, or the small `RUS`.

### The `#` character

`#` stands for **exactly one character you cannot read confidently**.

- `A12#BC77` — one digit hidden or illegible; the rest is certain.
- `A123BC7#` — the last region digit is cut off by the image edge.
- Use one `#` per unreadable character. If you cannot tell **how many**
  characters are hidden, you cannot annotate the plate: make it `other`.
- Never guess a character to avoid a `#`. A wrong character is worse than a
  `#`, because it trains the OCR on a lie.
- If **every** character is `#`, the plate carries no OCR signal. Keep it only
  if it is still useful as a detection example, and expect a validator warning;
  otherwise use `other`.
- For `type1`, `type1a` and `type1b`, `plate_num` must not be empty. Empty
  `plate_num` is allowed only for `other`.

## Images with several plates

**One row per plate**, all sharing the same `image` value.

- Annotate every plate that is legible enough to classify, including the ones
  that turn out to be `type1` or `other`. A visible plate left unannotated
  teaches the detector that it is background.
- Each row gets its own bbox, quad, `plate_type`, `plate_num` and `conditions`
  — conditions are per plate, so the near plate may be clean while the distant
  one is `motion_blur`.
- The same plate number may legitimately appear on two different images. The
  same plate number twice **on the same image** is a duplicate and the
  validator rejects it.
- Plates too small or too blurred to classify: annotate as `other` with a bbox
  and quad if you can see where they are. This is honest and useful.

## Images with no target plate

Still worth keeping — they teach the detector what is *not* a plate.

Write **one** row:

- `plate_type` = `other`
- `plate_num` = empty
- `is_vehicle` = `false`
- **every** `bbox_*` and `quad_*` column = `0`
- `conditions` as usual

This is the *background row* form; the validator skips geometry checks for it.
Zero geometry on any type other than `other` is an error.

Prioritise negatives that resemble plates: road signs, advertising panels,
house numbers, shop signage, text printed on vehicle bodies.

## Plates on screens, adverts and storefronts

A plate shown **on a screen, poster, banner, advertisement, shop window,
printed photo or vehicle wrap** is not a plate on a vehicle. It is a picture of
one.

- Set **`is_vehicle` = `false`**. This is exactly what that column is for.
- Set `plate_type` = `other`, whatever layout the pictured plate has. We do not
  want the detector confidently reading plates off billboards.
- You may still record its bbox and quad — a well-localised negative is more
  useful than a background row, because it tells the detector "here, and it is
  not a target".
- Leave `plate_num` empty.

The same applies to a **detached plate**: one lying on a table, held in a hand,
hanging on a wall, or displayed at a shop. `is_vehicle` = `false`.

A plate on a vehicle that is itself *inside* a photo on a billboard is still a
picture of a plate: `is_vehicle` = `false`, `plate_type` = `other`.

Reflections — a plate visible in a window, mirror or puddle — follow the same
rule: `is_vehicle` = `false`, `plate_type` = `other`. Annotate the real plate
separately if it is also visible.

## Tagging `conditions`

Only these tags, separated by `|`:

`day` · `night` · `rain` · `snow` · `dirt` · `glare` · `motion_blur` · `angle`

- Tag what is **visible in this image**, per plate. Leave the cell empty for a
  clean, straight-on, daylight shot — that is the default case and needs no tag.
- `day` and `night` are mutually exclusive; the validator rejects both together.
  Dusk and dawn: pick whichever the artificial lighting resembles more.
- `angle` — the plate is noticeably rotated or viewed obliquely, roughly beyond
  ±15°, rather than photographed straight on.
- `dirt` — mud, salt, snow crust, faded paint, a bent or damaged plate.
- `glare` — a bright reflection on the plate itself (sun, headlights, flash),
  not merely a bright scene.
- `motion_blur` — the plate is smeared by movement. Out-of-focus blur is not
  motion blur; leave it untagged.
- `rain` / `snow` — visible precipitation, wetness or droplets, on the plate or
  the lens.

## Provenance

- `source` — the `source_id` from [`data_sources.md`](data_sources.md).
  Register the source there *before* annotating its images.
- `license` — the **exact** license, e.g. `CC BY 4.0`, not "Creative Commons"
  and not "free". The validator reads this cell: an NC, ND or SA license is an
  error, and a license it does not recognise is a warning for a person to
  resolve. The dataset is published under CC BY 4.0, so an image we may use but
  not redistribute cannot be in it.
- Both are required for every real image. Synthetic rows written by the
  generator carry `source=volga_synthetic_generator` and `license=CC BY 4.0`
  (our own work); the validator does not require them for synthetic rows.
- `is_synthetic` must match the folder: `images/real/` → `false`,
  `images/synthetic/` → `true`.

## Before you commit

1. `python scripts/validate_dataset_local.py`
2. Fix every **error**. Exit code must be `0`.
3. Read the **warnings** — most are real mistakes: blank provenance, an
   unverified license, a quad that disagrees with its bbox, a file name reused
   under two paths.
4. Check the report's plate-type and condition tables against the targets in
   [`dataset_strategy.md`](dataset_strategy.md).
