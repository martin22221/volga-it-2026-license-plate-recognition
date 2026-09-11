# AUTO.RIA test split — visual / type review sample

Reproducible 200-image sample drawn from the AUTO.RIA test split
(`source_id: autoria_numberplate_options`) for human plate-type review.
Prepared 2026-09-10. **Nothing has been imported.**

## How the sample was selected

```bash
python scripts/sample_review_set.py "C:\Users\User\Downloads\test\test" \
    --source-id autoria_numberplate_options \
    --csv data/audits/autoria_test_review/review_sample.csv \
    --contact-sheet data/review/autoria_test/contact_sheet.html
```

- Population: **2564** image/label pairs (the whole test split)
- Sample: **200** images
- Seed: **20260910**, fixed in `src/review_sample.py` as `REVIEW_SEED`

Selection is deterministic and machine-independent: candidate stems are sorted
before sampling, so the result depends only on the set of files and the seed,
never on filesystem ordering. Re-running reproduces the same 200 images. The
seed stays fixed once review starts — changing it changes the sample.

## Artefacts

| What | Where | Committed |
| --- | --- | --- |
| Review CSV | `review_sample.csv` (this directory) | yes — metadata only |
| Contact sheet | `data/review/autoria_test/contact_sheet.html` | no — git-ignored |

The contact sheet references the external images **in place** by `file://`
URL. It copies nothing, so it only works on the machine holding the download.
Open it in a browser; each card shows the full frame with the annotation box
drawn over it, a close-up of that box, the filename, and the geometry flags.
Buttons at the top filter by flag.

## Automated suggestions — all 200 abstain

| suggested_plate_type | Count |
| --- | --- |
| `needs_human_review` | **200** |
| `type1` / `type1a` / `type1b` / `other` | 0 |

**This is deliberate, not a failure.** `type1` and `type1b` are distinguished
*only* by plate colour, and `type1a` additionally requires the plate to be
white. This sampling tool reads no pixels, so it cannot judge colour and any
type it emitted would be fabricated.

> **Updated 2026-09-11.** Plate colour *is* now measurable, via the DC
> coefficients of the JPEG itself (`src/jpeg_dc.py`) — still stdlib-only, no
> model, no network. It is used in the separate rare-class pass
> ([`RARE_CLASSES.md`](RARE_CLASSES.md)) to surface `type1b` candidates. It
> narrows the field; it does not assign types, and this 200-image sample is
> unchanged.

`human_plate_type` and `has_visible_face` are left empty for the reviewer.

## Geometry flags (machine-derived, from annotation boxes only)

Computed from the YOLO box and the image header — no pixels read.

| Flag | Count | Meaning |
| --- | --- | --- |
| `one_line_shape` | 111 | Box aspect ≥ 3.2 — consistent with a one-line plate |
| `extension_mismatch` | 67 | Named `.bmp`, actually JPEG or PNG |
| `square_or_two_line_candidate` | 46 | Box aspect ≤ 2.4 — **see the caveat below** |
| `ambiguous_shape` | 43 | Aspect between the two thresholds |
| `small_plate` | 13 | Plate < 80 px wide |
| `multiple_plates` | 9 | More than one annotated plate |
| `plate_touches_edge` | 3 | Plate runs to the frame edge |
| `tiny_plate` | 0 | Plate < 30 px wide — none in this sample |

### Caveat: the square flag is weak on this dataset

The aspect ratios of the 200 sampled plates form a **continuous smear from
1.11 to 5.73** (median 3.33), not the two clusters you would expect if the
dataset genuinely mixed one-line plates (nominal aspect 4.64) with square /
two-line plates (nominal 1.71):

```
1.0-1.5  #####                              5
1.5-2.0  ###################               19
2.0-2.5  ###########################       27
2.5-3.0  ############################      28
3.0-3.5  #################################  33
3.5-4.0  #################################  33
4.0-4.5  #########################         25
4.5-5.0  #########################         25
5.0-5.5  ###                                3
5.5-6.0  ##                                 2
```

The most likely reading: **this is an angle-heavy dataset**, and perspective
foreshortening — not plate shape — drives the low aspect ratios. A one-line
plate photographed from a steep angle compresses towards square. Marketplace
photography of parked cars from oblique angles would produce exactly this.

So treat `square_or_two_line_candidate` as **"look here first"**, not as
"type1a". Most of those 46 are probably one-line plates seen at an angle. If
genuine two-line plates exist here they will concentrate in the lowest band —
the **24 images with aspect below 2.0** are the highest-value ones to open
first.

The threshold was not tuned to make this number look better; it stays at the
nominal geometry plus margin.

## What the machine could not assess

None of these are in the CSV, because none can be computed without decoding
pixels. Every one needs eyes:

- **Plate colour (white vs yellow)** — decides `type1` vs `type1b`. Now
  *measured* for the rare-class pass (see [`RARE_CLASSES.md`](RARE_CLASSES.md)),
  but the measurement flags candidates rather than confirming them.
- **Russian vs foreign plate** — AUTO.RIA is a Ukrainian marketplace, so a
  large share of these plates may be Ukrainian and therefore out of scope
- **Visible faces** — **0 detected, because no detection was run.** All 200
  need a human check. Faces must be blurred before any import.
- **Blur, glare, dirt / occlusion**
- **Severe angle** — the YOLO box is axis-aligned and carries no rotation, so
  angle is not recoverable from the annotation

## Rare-class follow-up

A focused review of the rare classes now exists, adding plate-colour
measurement across all 2564 images: see [`RARE_CLASSES.md`](RARE_CLASSES.md).
143 unique candidate images, 0 human-confirmed.

## Related

- Technical audit: [`../autoria_test/`](../autoria_test/) — including the 9
  SHA-256 duplicate groups (20 files), which stand unchanged
- Licensing: [`../../../docs/data_sources.md`](../../../docs/data_sources.md)
  — CC BY 4.0, ARS Online OU, gate passed
- Workflow: [`../../../docs/external_dataset_workflow.md`](../../../docs/external_dataset_workflow.md)
