# Synthetic plate generator (V1)

Renders synthetic Russian registration plates — `type1`, `type1a`, `type1b` —
mounted on procedural vehicles, with seeded geometric and photometric
degradation, and writes images plus annotations in the `dataset/meta.csv`
schema.

**Status: V1, awaiting human visual review.** Do not generate the full
training set until the demo contact sheet has been approved.

## Quick start

```bash
pip install -r dataset/generator/requirements.txt   # numpy + Pillow, pinned

# 60-image review batch, 20 per class, mixed difficulty
python -m dataset.generator --output data/synthetic_dev/demo \
    --per-class type1=20,type1a=20,type1b=20 --seed 20260913 --contact-sheet

# Proportional batch with the default class and difficulty mix
python -m dataset.generator --output data/synthetic_dev/run --count 1000 --workers 4

# Validate any batch with the repository's own checker
python scripts/validate_dataset_local.py --dataset data/synthetic_dev/demo
```

| Option | Meaning |
| --- | --- |
| `--output DIR` | Batch directory (required). Mirrors the `dataset/` root layout. |
| `--count N` | Total images, split by `--class-weights` (default type1 20 % / type1a 40 % / type1b 40 %, per `docs/dataset_strategy.md`). |
| `--per-class type1=N,...` | Exact images per plate type (instead of `--count`). |
| `--seed S` | Master seed (default `20260913`). |
| `--difficulty easy\|medium\|hard` | Generate a single level. |
| `--difficulty-weights easy=W,...` | Level mix (default 35 / 40 / 25 %). Split *within* each class. |
| `--image-size WxH` | Output frame size; repeatable (default 640×480, 800×600, 1024×768, 1280×720). |
| `--type1b-format competition\|gost_1b` | Character layout of `type1b` (see *Open issues*). |
| `--config FILE` | JSON overlay on the defaults; only changed keys are needed. `configs/default.json` lists every key. |
| `--workers N` | Parallel processes; output is byte-identical to `--workers 1`. |
| `--contact-sheet` | Also write `review/contact_sheet.html`. |
| `--overwrite` | Replace a batch *this generator* wrote there. Anything else is refused. |
| `--allow-dataset-dir` | Required to write inside `dataset/` — reserved for final generation. |

Exit codes: `0` success, `1` written but fails validation, `2` bad arguments or
config, `3` output directory unusable.

## Output

```
<output>/
  images/synthetic/syn_<seed>_<index>.jpg
  meta.csv            dataset/meta.csv schema, one row per plate
  generation.jsonl    every parameter of every sample, including its seed
  manifest.json       config, library versions, counts, SHA-256 of each file
  review/contact_sheet.html   (with --contact-sheet)
```

`meta.csv` rows:

| Column | Value |
| --- | --- |
| `plate_num` | Normalised Latin text, e.g. `A123BC77`; an occluded character is `#`. |
| `plate_type` | `type1`, `type1a` or `type1b` — never `other`. |
| `bbox_*` | Tight axis-aligned box of the quad, in pixels. |
| `quad_*` | The plate's outer corners, clockwise from the plate's own top-left. |
| `is_vehicle` / `is_synthetic` | `true` / `true`. |
| `source` / `license` | `volga_synthetic_generator` / `CC BY 4.0` — never an external dataset's id. |
| `conditions` | `day` or `night`, plus `angle`, `dirt`, `glare`, `motion_blur`, `rain`, `snow` when present. |

Conditions are derived from the parameters actually applied: `angle` above 15°
of yaw, pitch or roll; `dirt` from strength 0.25; `glare` only when the
hotspot lands on the plate; `motion_blur` from 3 px. Effects outside the
controlled vocabulary (defocus, noise, shadow, JPEG) are recorded only in
`generation.jsonl`.

### How the annotation is computed

The plate is rendered so that it fills its canvas exactly, and is mounted in a
vehicle panel drawn *in the plate's plane*. A pinhole camera model (yaw, pitch,
roll, distance) gives the image positions of the plate's four corners. A
homography maps the canvas onto those points, and the same matrix both warps
the pixels and yields the quad. The quad is computed from the transform, not
estimated from pixels. Tests check it against the warped plate mask: the
measured deviation is under 1 px, due only to the rounded corners.

## Architecture

| File | Responsibility |
| --- | --- |
| `config.py` | Defaults, difficulty profiles, JSON overlay loading, validation (`ConfigError`). |
| `rng.py` | Named SHA-256-derived random streams; no global random state. |
| `plate_text.py` | Plate identities in the competition grammar; unique by default. |
| `glyphs.py` | Built-in stroke font: 10 digits, 12 plate letters, `R U S`. Own work. |
| `fonts.py` | `FontProvider` protocol; the stroke provider; a guarded placeholder for a future vetted TrueType font. |
| `templates.py` | Explicit millimetre geometry for the one-line and two-line layouts. |
| `render.py` | Layout → antialiased RGBA plate; plate-space dirt. |
| `scene.py` | `BackgroundProvider` protocol, procedural backgrounds, car-shaped vehicle panel, occlusion. |
| `geometry.py` | Camera model, homographies, placement, quad checks. |
| `photometric.py` | Lighting, night, low light, shadow, glare, rain/snow, motion blur, defocus, noise. |
| `sample.py` | Plans the batch; renders one sample end to end. |
| `annotations.py` | `meta.csv` row model, condition derivation, `#` masking. |
| `writer.py` | Batch layout, manifest, safe overwrite. |
| `contact_sheet.py` | Self-contained HTML review page; crops are rectified from the annotated quad. |
| `generate.py` | CLI and `generate_dataset()`. |

## Plate types

| Type | Size (mm) | Layout | Field |
| --- | --- | --- | --- |
| `type1` | 520 × 112 | `A 123 BC` · separator · region over `RUS` + flag | white |
| `type1a` | 290 × 170 | `A 123` over `BC 77`, flag over `RUS` at the right | white |
| `type1b` | 520 × 112 | as `type1` | yellow |

Taken from the text of GOST R 50577-2018: the layouts of types 1 and 1A (§3.3),
the yellow field of type 1B (table 2), character heights of 58 and 76 mm with
minimum strokes of 9 and 11 mm (§3.8, table 1), and a 3 mm border. The
dimensioned figures A.1–A.5 were not available. Character widths, gaps,
baselines, corner radius and the RUS/flag placement are therefore
approximations, kept as named constants in `templates.py`.

`type1a` has its own two-line layout, with characters at full physical size;
it is not a squashed `type1`. Body colours come from one palette for every
type, yellow included, so car colour cannot act as a shortcut for `type1b`.

## Difficulty and degradations

| | easy | medium | hard |
| --- | --- | --- | --- |
| Plate scale (px/mm) | 0.30–0.60 | 0.17–0.38 | 0.12–0.26 |
| Max yaw / pitch / roll | 12° / 8° / 4° | 30° / 15° / 8° | 50° / 25° / 15° |
| Optional effects per image | ≤ 1 | ≤ 2 | ≤ 3 |
| JPEG quality | 85–95 | 60–90 | 35–75 |

Optional effects, each with its own probability per level: night, low light,
shadow, glare, dirt, occlusion, rain, snow, motion blur, defocus, heavy
(chroma) noise. Always applied: exposure/contrast/gamma/white balance, sensor
noise and JPEG (hard adds double compression). Night and low light, rain and
snow, and motion blur and defocus are mutually exclusive. Most easy images
carry no optional effect at all.

**Legibility guards** keep labels truthful. Every label names characters that
can actually be read:

- the smallest character is at least 9 px tall, measured along the plate's
  more compressed axis, so strong yaw counts;
- motion blur is capped at 0.25 × that height and defocus at 0.12 ×;
- JPEG quality is floored at 55 when characters are under 12 px;
- glare stays below the strength that would clip the characters away;
- dirt is never opaque;
- characters an occluder covers by more than 35 % are labelled `#`.

## Determinism

Every random value comes from a stream named by the master seed, the sample
index and the stage (`geometry`, `effect:glare`, …). The same configuration
and seed therefore reproduce identical plate identities, annotations and image
bytes, given the same NumPy and Pillow versions (which the manifest records).
`--workers N` does not change the output.

## Licensing

The generator loads **no external asset**: no font file, texture, template
image or photograph. The font, templates, backgrounds and vehicles are all
drawn in code in this directory, and every manifest records
`external_assets: []`. The generator and its output are released under CC BY
4.0 with the dataset; the source is registered as `volga_synthetic_generator`
in `docs/data_sources.md`.

`fonts.TrueTypeFontProvider` refuses to construct on purpose. A real
plate-style font may be added only after its redistribution licence has been
verified and registered in `docs/data_sources.md`.

## Open issues (V1)

1. **`type1b` character composition.** GOST R 50577-2018 §3.3 gives type 1B as
   `ММ 000 55` (two letters, three digits, region). The competition mask and
   `src/validator.py` use `M 000 MM 55` for every class, and this generator
   follows that by default. If the official `type1b` examples use the GOST
   form, switch with `--type1b-format gost_1b`, and extend the pipeline
   validator in the same change. It currently rejects that form.
2. **Glyph fidelity.** The stroke font resembles the plate typeface but is not
   the GOST drawing. Real photographs should be used to check that OCR
   generalises.
3. **Approximate geometry.** See *Plate types*; the type 1A RUS/flag
   placement in particular is unverified.
4. **Procedural scenes.** Backgrounds and vehicles are deliberately simple and
   are aimed at plate appearance, OCR, classification and perspective.
   Detector realism needs licensed real backgrounds through `BackgroundProvider`.
5. **Throughput.** About 0.4 s per image per process. Use `--workers`.
