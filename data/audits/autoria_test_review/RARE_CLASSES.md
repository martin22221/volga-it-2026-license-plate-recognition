# AUTO.RIA test split — rare-class candidate review

Focused review of the two rare competition classes, `type1a` (white square /
two-line) and `type1b` (yellow one-line transport). Prepared 2026-09-11.

**Nothing has been imported.**

## Human review outcome — recorded 2026-09-13

A human reviewed both candidate pages (`rare_candidates.html`, 89 shape
candidates, and `yellow_candidates.html`, 54 colour candidates).

| Result | Value |
| --- | --- |
| Human-confirmed `type1a` | **0** |
| Human-confirmed `type1b` | **0** |
| Typical finding | Ordinary one-line plates whose shape or colour came from perspective, crop, lighting, dirt or a colour cast |

- **Yellow candidates:** none confirmed as `type1b`. This matches the warning
  below that the measured colours were warm tan/khaki rather than saturated
  yellow. The single strong `type1b` suggestion (`images/25202.jpg`) was not
  confirmed either.
- **Square / ambiguous candidates:** none confirmed as `type1a`. This matches
  the angle-heavy aspect-ratio smear.

**Scope and limits.**

- The verdict was reached for the candidate sets as a whole. Per-row verdicts
  were not recorded, so the `human_plate_type` column in `rare_candidates.csv`
  is left empty on purpose and not back-filled.
- The stats file's `human_confirmed_type1a` / `human_confirmed_type1b` = 0
  remain accurate.
- This review does **not** show that the split consists of Russian `type1`
  plates. Country and plate type have not been classified across the 2564
  images, and AUTO.RIA is a Ukrainian marketplace.

**Consequence.**

- AUTO.RIA is **not a rare-class source**.
- It remains a candidate for generic plate detection and robustness.
- It is a candidate for `type1` / OCR only after country/type filtering.
- Its possible use as foreign/out-of-scope `other` examples is a separate
  evaluation.

The registry entry in
[`../../../docs/data_sources.md`](../../../docs/data_sources.md) carries the
same status.

---

*The rest of this document was written before the review and is kept
unchanged.*

```bash
python scripts/build_rare_review.py "C:\Users\User\Downloads\test\test" \
    --source-id autoria_numberplate_options \
    --out-html data/review/autoria_test \
    --out-data data/audits/autoria_test_review
```

## Statistics

| Metric | Value |
| --- | --- |
| Total test images scanned | **2564** |
| Plate colour readings taken | 2564 (9 unreliable) |
| `square_or_two_line_candidate` | **46** |
| `ambiguous_shape` | **43** |
| `yellow_plate_colour` (over threshold) | **14** |
| `most_yellow_in_dataset` (below threshold, shown anyway) | 40 |
| Overlap between shape and yellow groups | **0** |
| **Unique candidate images** | **143** |
| Human-confirmed `type1a` | **0** |
| Human-confirmed `type1b` | **0** |

Shape candidates come from the seeded 200-image sample (seed `20260910`);
colour was scanned across all 2564 images.

## Review pages

| Page | Contents |
| --- | --- |
| `data/review/autoria_test/rare_candidates.html` | 89 shape candidates (46 square + 43 ambiguous) |
| `data/review/autoria_test/yellow_candidates.html` | 54 colour candidates (14 over threshold + 40 most-yellow) |

Both use large previews: the full vehicle image with the annotation box drawn
over it, and a wide close-up of the plate beside it, with the filename and the
heuristic reason. Images are referenced in place by `file://` — nothing copied.
The pages are git-ignored because they embed absolute local paths.

Record verdicts in `rare_candidates.csv` under `human_plate_type`.

## How plate colour was measured

No external API, no pretrained model, no internet. Plate colour comes from the
JPEG's own **DC coefficients** — each 8×8 block's DC coefficient is that
block's average, so decoding DC alone yields a 1/8-scale thumbnail without an
inverse DCT and without an image-decoding dependency (`src/jpeg_dc.py`).

Two guards keep it conservative (`src/plate_color.py`):

1. **The surroundings must not be yellow too.** A ring around the plate is
   sampled and its contribution unmixed, and a plate only counts when it is at
   least 12 yellowness points more yellow than what surrounds it. A yellow car
   with a white plate is rejected — there is a test for exactly that case.
2. **Small plates abstain.** Below 3 DC blocks (24 px) of plate width the
   reading is contaminated by surrounding bodywork and is reported unreliable
   rather than used.

Thresholds: Cb ≤ 115 **and** Cr ≥ 133 **and** luma ≥ 55 **and** margin ≥ 12.
Yellow is the one hue that pushes Cb down and Cr up together, which separates
it from red (Cr only), blue (Cb up) and neutral greys.

### Decoder validation

The DC reader was checked before its output was trusted:

- Across 60 random images, plate regions read **Cb 129 / Cr 125 with tight
  IQRs** — near-neutral, exactly what white plates give. Random or
  mis-mapped regions would show a much wider, off-neutral spread, so this
  validates both the colour decoding and the bounding-box → block mapping.
- Whole-image chroma medians land on 127–128 (neutral) across mixed scenes.
- Top-of-frame reads bluer than bottom-of-frame in 20/29 outdoor photos,
  matching the sky-above-road prior.
- Unit tests confirm the detector **fires** on a synthetic yellow plate and
  **does not** fire on a white plate, a yellow car with a white plate, a
  uniformly yellow scene, a red patch, a dark patch, or a plate too small to
  judge.

## Yellowness distribution (all 2564)

| Percentile | Yellowness |
| --- | --- |
| min | −75.6 |
| p50 | −2.2 |
| p90 | +8.5 |
| p99 | +26.1 |
| max | +48.6 |

The distribution centres on zero — i.e. neutral, white plates — with a thin
warm tail. The 14 threshold candidates sit in roughly the top 0.5 %.

### Read this before reviewing the yellow page

The measured colours of the 14 candidates are **warm tan / khaki**
(`#c9ae7a`, `#9b8254`, `#804f3d`), not saturated yellow. Two explanations fit
equally well, and only a person can separate them:

- a genuinely yellow plate, averaged over 8×8 blocks together with its black
  characters, which pulls the mean towards a muted gold; or
- a **white plate under warm light** — sunset, tungsten, or a warm colour cast
  — or a dirty/beige plate.

Expect a meaningful share of the 14 to be the second case. The heuristic's job
was to narrow 2564 images down to 54 worth opening, not to decide.

## Shape candidates — the same caveat as before

The 89 shape candidates are **prompts to look, not evidence of two-line
plates**. Aspect ratios across the sample form a continuous smear (1.11–5.73,
median 3.33) rather than two clusters, which points at an angle-heavy dataset
where perspective foreshortening — not plate layout — produces low aspect
ratios. `type1a` requires an actual two-line layout visible on the plate.

## Suggested types in the CSV

| suggested_plate_type | Count |
| --- | --- |
| `needs_human_review` | 142 |
| `type1b` | 1 (`images/25202.jpg`, strongest yellow reading in the dataset) |

The single `type1b` suggestion is the one case clearing the "strong" bar. It is
still only a suggestion: `human_plate_type` is empty, and the confidence note
says a person must confirm the plate is Russian and one-line.

## Related

- Seeded sample and contact sheet: [`README.md`](README.md)
- Technical audit, including the 9 SHA-256 duplicate groups (20 files, kept):
  [`../autoria_test/`](../autoria_test/)
- Licensing (CC BY 4.0, ARS Online OU):
  [`../../../docs/data_sources.md`](../../../docs/data_sources.md)
