# Synthetic plate generator (V2.2)

Renders synthetic Russian registration plates — `type1`, `type1a`, `type1b` —
mounted on procedural vehicles, with seeded geometric and photometric
degradation, and writes images plus annotations in the `dataset/meta.csv`
schema.

**Status: 2.2.0, awaiting Pilot B review.** Do not generate the full
training set until Pilot B has been approved.

## V1 was superseded — read this first

Generator V1 (commit `0cd33e1`, 2026-09-13) generated **`type1b` plates with
the wrong character structure.** It printed the type 1 pattern (`A123BC77`,
`A123BC777`) on a yellow field. GOST R 50577-2018 §3.3 gives type 1B as
`MM 000 55`: two letters, three digits, region.

- **V1 yellow plates are not valid `type1b` examples.** All V1 output, including
  the 60-image demo in `data/synthetic_dev/v1_demo_seed20260913/`, is a
  development artefact. It is kept unchanged for reproducibility and **must
  never enter the training or submitted dataset.**
- **Two mechanical guards** in `src/dataset_meta.py` enforce that:
  1. Any synthetic image named in V1's scheme (`syn_<seed>_<index>.jpg`) is a
     validation error. V2 names files `syn_v2_<seed>_<index>.jpg`.
  2. Every synthetic row's `plate_num` must follow its `plate_type`'s
     structure; `#` is allowed as a wildcard. A `type1b` row reading
     `A123BC77` is an error.
- The V1 option `--type1b-format` / `type1b_text_format` has been removed.
  The structure now follows from the plate type, and old configs that set it
  fail loudly.

## 2.2.0 — provenance fix and adverse-condition coverage

This release follows the Pilot A review (1,000 images, 2026-09-14).

**Provenance.** Heavy chroma noise was applied but never listed in a record's
`effects`; the noise step wrote only to `noise`. It is now recorded in
`effects["heavy_noise"]` whenever it is applied. Images are byte-identical,
and no random draws changed.

**Coverage.** Only default effect probabilities and two ranges changed.
Severities, budgets, caps and the 35/45/20 mix are untouched. The targets for
the full default mix are night 20–25 %, and rain, snow, glare on the plate and
tagged (≥ 3 px) motion blur 7–10 % each.

- Night now comes from medium (0.50) and hard (0.55). Night's severity (1.5)
  never fits the easy budget (1.25), so easy has no night.
- Rain, snow, glare and motion blur probabilities were raised.
- Dirt was raised slightly to keep its share against the extra effects.
- Glare radii were narrowed to localised hotspots. Whole-plate washes were
  mostly rejected by the legibility check, so glare rarely survived.
- Easy and medium motion-blur lengths now start at the 3 px tag threshold.

**Trade-off.** Medium images now carry more effects: 4.5 % are effect-free
(14 % before), and mean severity rose from 1.17 to 1.45. The legibility
floors are unchanged.

## V2.1 — precipitation label-integrity fix

**Found:** focused QA of the V2 demo found that a single snowflake could erase
a character's stroke on a small plate. In sample #42, `O935TB97`, the `O`
read as `C`, but `plate_num` was unchanged.

**Why the existing guards missed it:** snow and rain particles are drawn with
an absolute pixel size and up to about 0.98 opacity after the plate is
composited. They were not covered by the `#` occlusion labelling or by the
plate-level contrast check.

**Fix** (2.1.0):

- Particle opacity is capped at `TEXT_PARTICLE_ALPHA = 0.35` over the plate's
  character ink, which is warped into the image and grown by one pixel
  (`photometric.text_guard`).
- A capped particle can veil a stroke but never erase it, so every character
  keeps its identity. No character is special-cased.
- Snow and rain are otherwise unchanged: the same size, density and
  placement, and fully visible off the characters.
- The guard draws no random numbers. Samples without precipitation are
  byte-identical to V2.
- Regression tests are in `tests/test_generator_snow_integrity.py`.

## Quick start

```bash
pip install -r dataset/generator/requirements.txt   # numpy + Pillow, pinned

# 60-image review batch, 20 per class, default 35/45/20 difficulty mix
python -m dataset.generator --output data/synthetic_dev/v2_demo_seed20260913 \
    --per-class type1=20,type1a=20,type1b=20 --seed 20260913 --contact-sheet

# Proportional batch with the default class mix, in parallel
python -m dataset.generator --output data/synthetic_dev/run --count 1000 --workers 4

# Validate any batch with the repository's own checker
python scripts/validate_dataset_local.py --dataset data/synthetic_dev/v2_demo_seed20260913
```

| Option | Meaning |
| --- | --- |
| `--output DIR` | Batch directory (required). Mirrors the `dataset/` root layout. |
| `--count N` | Total images, split by `--class-weights` (default type1 20 % / type1a 40 % / type1b 40 %, per `docs/dataset_strategy.md`). |
| `--per-class type1=N,...` | Exact images per plate type (instead of `--count`). |
| `--seed S` | Master seed (default `20260913`). |
| `--difficulty easy\|medium\|hard` | Generate a single level. |
| `--difficulty-weights easy=W,...` | Level mix (default 35 / 45 / 20 %), split *within* each class. |
| `--image-size WxH` | Output frame size; repeatable (default 640×480, 800×600, 1024×768, 1280×720). |
| `--config FILE` | JSON overlay on the defaults; only changed keys are needed. `configs/default.json` lists every key. |
| `--workers N` | Parallel processes; output is byte-identical to `--workers 1`. |
| `--contact-sheet` | Also write `review/contact_sheet.html`. |
| `--overwrite` | Replace a batch *this generator* wrote there. Anything else is refused. |
| `--allow-dataset-dir` | Required to write inside `dataset/` — reserved for final generation. |

Exit codes: `0` success, `1` written but fails validation, `2` bad arguments or
config, `3` output directory unusable.

## Plate types and character structures

| Type | Structure (GOST §3.3) | Example | Size (mm) | Layout | Field |
| --- | --- | --- | --- | --- | --- |
| `type1` | `M 000 MM 55` / `M 000 MM 555` | `A123BC77`, `A123BC777` | 520 × 112 | one line; separator; region over `RUS` + flag | white |
| `type1a` | same registration as type 1 | `A123BC77` | 290 × 170 | `A 123` on top; `BC` bottom-left, region, then flag over `RUS` bottom-right | white |
| `type1b` | `MM 000 55` | `AB12377` | 520 × 112 | one line: `AB 123`, separator, region over `RUS` + flag | yellow |

- Letters: `A B E K M H O P C T Y X` (the Latin twins of the Cyrillic plate
  letters). Serial numbers run `001`–`999`.
- Type 1 / 1A regions: `01`–`99`, or three digits starting with `1`, `2` or
  `7`, following the task rule.
- **Type 1B regions: two digits only.** GOST §3.3 writes type 1B solely as
  `MM 000 55`. The standard also has separate figures for two- and
  three-digit regions for types 1 (A.1/A.2) and 1A (A.3/A.4), but a single
  figure for 1B (A.5). No source consulted shows a three-digit type 1B
  region. This is `TYPE1B_REGION_LENGTHS` in `src/validator.py` and
  `plate_text.py`; widen it there if official examples show otherwise.
- `src/validator.py` now knows both structures. `validate_plate(text)`
  without a type accepts either, so the inference pipeline's untyped check
  keeps working. `validate_plate(text, plate_type)` demands the structure of
  that type.

Taken from the GOST text: the layouts (§3.3), the yellow field of type 1B
(table 2), character heights of 58 and 76 mm with minimum strokes of 9 and 11
mm (§3.8, table 1), and the 3 mm border. The dimensioned figures A.1–A.5 were
not accessible. Character widths, gaps, baselines, the separator position,
the RUS and flag placement, corner radii and bolt sites are therefore
approximations, kept as named constants in `templates.py`.

## What V2 changed (after human review of V1)

**Plate fidelity**
- Stroke ends that meet a character's cell edge are cut flat by the cell
  (e.g. the feet of `A`, `K`, `M`, `X`) instead of rounded. Hooks and joins
  inside a cell keep round caps.
- Retuned proportions:
  - letters 47 × 58 mm and digits 48 × 76 mm, with tighter 7 mm character gaps
    and 18 mm group gaps;
  - wider margins at the plate ends;
  - a 1.5 mm border inset and 3 mm border;
  - a larger region field (region digits 41 mm wide, or 33 mm for three digits);
  - `RUS` raised from 15 to 17 mm with a bolder stroke, and a 25.5 × 17 mm flag.
- `type1b` has its own `LL DDD` main-field layout, with wider group spacing
  for its five characters.
- Retroreflective surface: a faint diagonal sheen, fine grain and a darker
  embossed rim.
- Optional mounting bolts (zinc, black caps or field-coloured caps). They are
  placed in the widest gaps, never on a character, and a test enforces that.

**Mounting and vehicles**
- Five parametric body types: sedan, hatchback, estate, SUV and van (vans can
  have solid rear doors).
- Rear or front views.
- Plate on the bumper (centre 380–560 mm above ground) or on the tailgate
  (650–880 mm), with a recess, a soft shadow from the lip above it, and an
  optional black or chrome frame.
- Paint shading: vertical falloff, a specular band and a random window
  reflection. Lamp clusters, a grille and exhaust.

**Scenes**
- Street, facade, car park, trees and garage backgrounds in a muted urban
  palette, with window grids, perspective road markings and plate-free
  background cars.
- **No classifier shortcut:** every scene and vehicle choice is drawn from the
  same distribution for every plate type. A test checks that `type1` and
  `type1b` get identical vehicles from the same stream.

**Difficulty and legibility**
- Default mix: easy 35 % / medium 45 % / hard 20 %.
- **Severity budget.** Each effect has a severity (night 1.5, motion blur and
  defocus 1.25, glare, occlusion and low light 1.0, …). Enabled effects must
  fit the level's budget: easy 1.25, medium 2.0, hard 2.75. Hard can combine
  night + motion blur, or glare + dirt + shadow, but never
  night + dirt + defocus.
- **Measured legibility.** On the finished image, the median luminance of the
  plate's ink is compared with its field: `(field − ink) / (field + ink)`.
  Minimums are easy 0.50, medium 0.38, hard 0.30. A sample below its
  minimum gets its effects redrawn from a new deterministic stream, up to
  3 times, and is then rendered clean. Attempts and the final contrast are
  recorded in `generation.jsonl`.
- Softer hard ranges: plate scale 0.14–0.28 px/mm, the smallest character at
  least 10 px, yaw up to 45°, JPEG quality 45–80.
- Occlusion hides at most 2 characters; larger occluders are shrunk and
  redrawn.
- Unchanged from V1: blur is capped relative to character size, motion blur
  and defocus are mutually exclusive, glare stays below clipping strength,
  and dirt is never opaque.

## Output

```
<output>/
  images/synthetic/syn_v2_<seed>_<index>.jpg
  meta.csv            dataset/meta.csv schema, one row per plate
  generation.jsonl    every parameter of every sample, including its seed
                      and legibility attempts
  manifest.json       config, library versions, counts, SHA-256 of each file
  review/contact_sheet.html   (with --contact-sheet)
```

`meta.csv` rows carry:

- `plate_num`: normalised Latin text, with `#` for an occluded character;
- `plate_type`: `type1`, `type1a` or `type1b`, never `other`;
- `bbox_*`: the tight box of the quad;
- `quad_*`: the plate's outer corners, clockwise from its own top-left;
- `is_vehicle=true`, `is_synthetic=true`;
- `source=volga_synthetic_generator`, `license=CC BY 4.0`;
- `conditions`: `day` or `night` plus whichever of `angle`, `dirt`, `glare`,
  `motion_blur`, `rain`, `snow` was actually applied.

The quad comes from the same homography that warps the pixels (camera model →
four corner correspondences → one matrix), so it is exact. Tests check it
against the warped plate mask.

## Architecture

| File | Responsibility |
| --- | --- |
| `config.py` | Defaults, difficulty profiles, effect severities, JSON overlay loading, validation. |
| `rng.py` | Named SHA-256-derived random streams; no global random state. |
| `plate_text.py` | Plate identities; the structure is fixed per plate type. |
| `glyphs.py` | Built-in stroke font: 10 digits, 12 plate letters, `R U S`. Own work. |
| `fonts.py` | `FontProvider` protocol; the stroke provider with cell-clipped flat terminals; a guarded placeholder for a future vetted TrueType font. |
| `templates.py` | Explicit millimetre geometry per layout; bolt sites. |
| `render.py` | Layout → antialiased RGBA plate with sheeting texture and bolts; plate-space dirt. |
| `scene.py` | `BackgroundProvider` protocol, procedural backgrounds, parametric vehicles, occlusion. |
| `geometry.py` | Camera model, homographies, placement, quad checks. |
| `photometric.py` | Lighting, night, low light, shadow, glare, rain/snow, motion blur, defocus, noise. |
| `sample.py` | Plans the batch; renders one sample, with the severity budget and legibility retries. |
| `annotations.py` | `meta.csv` row model, condition derivation, `#` masking. |
| `writer.py` | Batch layout, versioned file names, manifest, safe overwrite. |
| `contact_sheet.py` | Self-contained HTML review page; crops are rectified from the annotated quad. |
| `generate.py` | CLI and `generate_dataset()`. |

## Determinism

Every random value comes from a stream named by the master seed, the sample
index and the stage (`geometry`, `effect:glare`, `effects#retry1`, …). The
same configuration and seed reproduce identical plate identities, annotations
and image bytes, given the same NumPy and Pillow versions (recorded in the
manifest). `--workers N` does not change the output.

## Licensing

The generator loads **no external asset**: no font file, texture, template
image or photograph. Every manifest records `external_assets: []`. The
generator and its output are released under CC BY 4.0 with the dataset, and
the source is registered as `volga_synthetic_generator` in
`docs/data_sources.md`.

`fonts.TrueTypeFontProvider` refuses to construct on purpose. A real
GOST-style plate font may be added only after its redistribution licence has
been verified and registered.

## Open issues (V2)

1. **Glyph fidelity.** The stroke font approximates the plate typeface. It is
   not the GOST drawing, and no redistributable GOST-style font has been
   found. The font provider is modular so a vetted font can be dropped in.
2. **Approximate geometry.** See above. The type 1A RUS/flag placement and all
   widths and gaps are unverified against the dimensioned figures.
3. **Type 1B three-digit regions.** These are excluded on the GOST evidence;
   revisit if official examples show one.
4. **Procedural scenes.** Better than V1, but still visibly synthetic. Detector
   realism needs licensed real backgrounds through `BackgroundProvider`.
5. **Throughput.** About 0.1–0.4 s per image per process. Use `--workers`.
