# Real-data plan

What real photographs we need, how they are split, how they will be mixed with
the synthetic set, and what has to be true before the first training run.

**13 real images are in the dataset as of 2026-09-22**, the first through the
intake and far short of every target below. This document is the plan and the
policy; [`real_data_intake.md`](real_data_intake.md) is the workflow that puts
images through it. Written 2026-09-16, after the 12,000-image synthetic batch
was audited and promoted.

Related: [`real_data_source_research.md`](real_data_source_research.md) (which
public sources were investigated and what each was decided to be),
[`dataset_strategy.md`](dataset_strategy.md) (acquisition scenes and
the condition distribution), [`data_sources.md`](data_sources.md) (source
registry and the rights gate), [`annotation_guide.md`](annotation_guide.md)
(how to label), [`../dataset/splits/README.md`](../dataset/splits/README.md)
(the split manifest).

## Where we stand

| | Images | Rows in `dataset/meta.csv` |
| --- | --- | --- |
| Synthetic (generator 2.3.0, seed 2026091401) | 12,000 | 12,000 |
| Real (`wikimedia_commons_curated`, promoted 2026-09-22) | **13** | **13** |

**Acquisition #1 is DEFERRED**, not cancelled. One targeted `type1a` capture
session, `source_id` `team-capture-2026-09`. The participant is in Bulgaria and
Russian rare-class plates are not on the street there; the source record,
staging folder and field checklist stay in place for the day someone can shoot
to them. See [`capture_session_type1a.md`](capture_session_type1a.md).

**Acquisition #2 is complete.** 22 Wikimedia Commons photographs acquired,
21 annotated, then a human review on 2026-09-22 promoted **the 13 marked PASS
only** — 6 `type1` and 7 `type1b` plates, `decision: ACCEPT_FOR_SUBMISSION`.
The 8 QUESTIONABLE images (11 plates) and the 1 FAIL image stay in staging and
are **not** in the dataset. See
[`acquisition_wikimedia_commons.md`](acquisition_wikimedia_commons.md).

### `type1a` is synthetic-only for now — a decision, not a conclusion

**Real `type1a` acquisition has produced zero images, and we are proceeding
without it for the time being.** The reasoning, so it does not have to be
reconstructed later:

- **Online discovery found no `type1a` photograph at all** — 0 of 5,813 Commons
  titles surveyed, after three independent attempts (search and category walk, a
  geometry filter over two-line crops, and a targeted hunt through 179 scenes of
  the Japanese and grey imports most likely to carry a square plate). Every
  readable plate was one-line. See
  [`online_source_discovery.md`](online_source_discovery.md).
- **Self/team capture is deferred for geography**, as above. It is the only
  route that has ever looked credible for this class.
- **So `type1a` remains synthetic-only for now.** The generator produces it, and
  the 12,000-image synthetic batch is rare-class-heavy by design.

Three things this decision is **not**:

- It is **not** a claim that no real `type1a` data exists or can ever be found.
  It says that two specific routes, pursued properly, did not yield any.
- It is **not** permanent. If model evaluation shows the classifier or the OCR
  failing on real `type1a` in a way synthetic data cannot fix, real `type1a`
  acquisition is reopened — a contributor in Russia, a licensed archive, or a
  direct approach to a transport photographer.
- It is **not** a licence to improvise. **Foreign square plates are never
  relabelled as Russian `type1a`.** Japanese, Armenian and other square or
  two-line plates are a different standard, and a square plate photographed
  abroad is not a Russian one. The discovery pass already rejected an Armenian
  plate and a Gostekhnadzor tractor plate on exactly this ground.

## 1. Targets

Two columns, and they are not the same thing.

**A — competition compliance.** The minimums recorded in
[`dataset_strategy.md`](dataset_strategy.md) as the competition's own figures.
These are the documented requirement; nothing here invents one.

**B — preferred engineering target.** Ours. It is higher because a dataset
sized exactly to the minimum leaves nothing for a real holdout: split 150
`type1a` images 70/15/15 and the holdout holds 22 images, which measures
almost nothing.

| Class | A: minimum images | A: minimum unique plates | B: preferred images | B: preferred unique plates |
| --- | --- | --- | --- | --- |
| `type1a` — square / two-line white | **150** | **50** | 300 | 120 |
| `type1b` — yellow one-line | **300** | **100** | 500 | 200 |
| `other` — out-of-scope plates and true negatives | **50** | — | 150 | — |
| `type1` — standard white one-line | *no documented minimum* | — | 400 (opportunistic) | 250 |

Why B is where it is:

- **A real holdout must be able to answer a question.** At ~15 % of a class,
  B gives roughly 45 `type1a` and 75 `type1b` holdout images. Character-level
  accuracy is still readable at that size; plate-level accuracy on 45 images
  carries a 95 % interval of roughly ±9 points, so we report it as a range and
  lean on the character metric. At A's 150 images the holdout is 22 and even
  the character metric wobbles. This is the single strongest argument for B.
- **Unique plates matter more than images.** Twenty photographs of one taxi
  teach the OCR one plate string. The unique-plate targets are what force
  variety; the image targets are what force conditions.
- **`type1` has no documented minimum and we do not go hunting for it.** It is
  the most common plate on the street and accumulates for free in every scene.
  The B figure is a bookkeeping expectation, not a collection effort. Annotate
  every one you photograph.
- **`other` earns its place.** ~60 % out-of-scope plates (motorcycle, trailer,
  military, diplomatic, transit, foreign) and ~40 % true negatives — signage,
  advertising panels, house numbers, text on vehicle bodies. This is what stops
  the detector firing on every rectangle of text.

Rare classes are the priority: `type1a` first, `type1b` second. If field
collection stalls, the shortfall is closed with synthetic images **and recorded
here**, never quietly left empty.

### Condition coverage

The per-axis distribution in [`dataset_strategy.md`](dataset_strategy.md)
stands (day 60–70 %, night 30–40 %, angle 40–50 %, dirt and glare 15–20 %,
rain and snow 10–15 %, motion blur 10–15 %, partial occlusion 10–15 %,
distance near ~30 % / mid ~45 % / far ~25 %). It applies to `type1a` and
`type1b`, the classes we collect deliberately. Three additions for the real
stage:

| Axis | Target | Why |
| --- | --- | --- |
| Cameras / devices | **≥ 3 distinct cameras**, none above 60 % of a class | One phone's sensor, lens and JPEG pipeline is one domain. A model tuned to it fails on the jury's images. |
| Resolution spread | at least two capture resolutions per class | The plate's pixel height is what the OCR actually sees; do not let it be constant. |
| Small / distant plates | ~25 % of a class below 80 px plate width | Matches the far bucket above, and is where detection fails first. |
| Partial occlusion | 10–15 %, labelled with `#` per the guide | Tow bars, bike racks, snow, frames. The synthetic set covers this; real examples validate that the `#` policy transfers. |

Night is the axis most likely to be skipped and the one most likely to decide
the result. Do not let it fall below 25 % of a class.

## 2. Splits

Designed before acquisition, deliberately. Details and the manifest schema are
in [`../dataset/splits/README.md`](../dataset/splits/README.md).

| Split | Share of groups | Purpose |
| --- | --- | --- |
| `train` | 0.70 | Fitting. |
| `val` | 0.15 | Model selection, thresholds, checkpoints, ablations. |
| `holdout` | 0.15 | **Never trained or tuned on.** Approximates the jury's hidden test set. |

- **The unit is a group, not an image.** A group is whatever would leak if it
  were split: one burst or video, one capture session, one vehicle, one
  physical plate, and any crop derived from the same original.
- **Deterministic:** the split comes from `SHA-256("<seed>:<group>")`, so it
  never depends on file order or on when an image arrived.
- **Frozen:** a group in the manifest keeps its split forever. New images are
  assigned around it, and the tooling refuses to move an image between groups.
- **Audited:** `scripts/plan_real_splits.py --check` reports groups spanning
  splits, duplicate rows, identical images across splits (by SHA-256) and
  near-duplicate pairs across splits (thumbnail correlation from the intake
  audit).
- **Class balance is checked, not enforced by the hash.** After each intake,
  read the per-class counts per split; if the holdout is starved of `type1a`,
  fix it by collecting more, not by moving a group.

The holdout must also be *representative*: it is only a stand-in for the jury
set if it carries night, angle, dirt and distance in roughly the proportions
above. Check that when it is first frozen.

## 3. Mixing synthetic and real for training

A plan, not a result. Nothing is trained yet, and every number below is a
starting point to be tested against the real validation split.

**The 12,000 synthetic images are the character-space and rare-layout backbone.**
They cover every legal plate string, both rare layouts, and the adverse
conditions we cannot schedule (snow in September). What they do not carry is a
real camera, real optics, real backgrounds or real dirt.

**Real images decide.** Model selection, thresholds and every reported number
come from the **real** validation split. A synthetic validation set is a
smoke test, never the selection criterion — a model can be excellent on our
own renderer and useless on a phone photo.

Starting points per stage:

| Stage | Initial mix | Reasoning |
| --- | --- | --- |
| Detector | synthetic : real ≈ 3 : 1 by count, with real **oversampled ×3** so an epoch sees them roughly 1 : 1 | Detection transfers from synthetic reasonably well, but backgrounds are exactly what our procedural scenes fake worst. |
| Plate-type classifier | synthetic-heavy early (rare classes are 9,600 of the 12,000), real oversampled to ≈ 1 : 1 at the end | `type1a` / `type1b` real images are the scarcest thing we have; the classifier is what the competition scores on class. |
| OCR | start ≈ 70 % synthetic, anneal towards 40 % as real grows; **final epochs on real only** | The character space comes from synthetic; the appearance must come from real. Fine-tuning last on real is the cheapest domain fix we have. |

- **Do not assume equal weight.** The mix is a hyperparameter. Run at least the
  three-point ablation synthetic-only / mixed / real-only-fine-tune and report
  all three on the real validation split.
- **Never mix the holdout in, at any weight, at any stage**, including for
  early stopping or threshold calibration.
- Synthetic images are not split into train/val/holdout: they are all training
  material. If a synthetic sanity metric is wanted, hold out a fixed slice by
  index and record which — do not let it drift.
- Keep the real:synthetic ratio per *class*, not only overall. An overall 1 : 1
  that is 95 % `type1` real is still starving the rare classes.

## 4. Readiness checklist — before the first training run

Every line must be true, and checkable by someone else.

- [ ] **Real data present.** At least target A for `type1a` and `type1b`, with
      the unique-plate minimums met; the shortfall against B recorded here.
- [ ] **Rights settled.** Every real source has a record with
      `decision: ACCEPT_FOR_SUBMISSION` and evidence; no image from a
      `PENDING`, `REFERENCE_ONLY` or `REJECT` source is in `dataset/`.
      `TRAINING_ONLY_IF_LEGAL` images, if any are ever accepted, are kept out of
      the submitted dataset and tracked separately.
- [ ] **Annotations validated.** `python scripts/validate_dataset_local.py --strict`
      exits 0: no errors, no warnings.
- [ ] **Privacy done.** Every real image has a privacy review row; faces blurred
      or covered where required; images centred on identifiable people rejected;
      the original provenance kept for transformed images.
- [ ] **Intake audits clean.** `scripts/audit_real_source.py` reports nothing
      blocking for every accepted source.
- [ ] **Splits frozen.** `dataset/splits/real_splits.csv` covers every real
      image, and is committed.
- [ ] **Leakage audit clean.** `scripts/plan_real_splits.py --check` reports
      CLEAN: no group spans splits, no identical or near-duplicate image pairs
      across splits.
- [ ] **Holdout protected and representative.** Its size and condition mix are
      recorded; the training code cannot read it (it is selected by split, and
      the check is part of the training script's start-up).
- [ ] **Official debug set excluded.** `data/official_debug/` is referenced by
      no row of `meta.csv` and is not under `dataset/`.
- [ ] **Synthetic batch untouched.** The sealed 2.3.0 batch still verifies.
- [ ] **Baseline defined.** The metric, the split it is measured on (real val)
      and the ablation set are written down *before* the first run, so the first
      number cannot become the target retroactively.

Until every box is ticked, the honest state is "no model", which is better than
a model trained on data we cannot defend.

## 5. What this plan does not claim

- No real image has been collected, downloaded or imported.
- No source has been approved; the registry currently holds one accepted
  source, our own generator.
- No model has been trained.
- The official 30-image debug set remains unavailable and, when it arrives, is
  for checking the pipeline only — never training data, never inside `dataset/`.
