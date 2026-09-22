# Baseline v1 — architecture, training plan and metrics

What the first trained system will be, what it trains on, and how it is judged.
Written **before** training, so the metrics cannot be chosen to flatter a
result and the splits cannot be redrawn after seeing one.

**No model has been trained. No framework is installed.** This document and the
scaffolding beside it make the next session able to start; they do not start it.

Related: [`../dataset/splits/README.md`](../dataset/splits/README.md) (the split
policy), [`real_data_plan.md`](real_data_plan.md) (real-data targets),
[`annotation_guide.md`](annotation_guide.md) (what the labels mean),
[`../configs/baseline_v1.json`](../configs/baseline_v1.json) (every number).

## 1. The constraints that decide everything

### Runtime target

**Not previously recorded anywhere in this repository.** It is written down here
for the first time, from the project brief:

| | |
| --- | --- |
| CPU | Intel i5-7600 (4 cores, no AVX-512) |
| GPU | NVIDIA GTX 1050 Ti, **4 GB** (Pascal, no tensor cores) |
| RAM | 16 GB |
| Budget | **≤ 100 ms/image average** |
| Mode | fully offline — no API, no network, no downloads at inference |

4 GB of Pascal VRAM and a 100 ms budget rule out anything in the
transformer-detector family and any OCR that runs a separate pass per character.
They comfortably allow a small one-stage detector plus one small recogniser.

**Nothing below claims the budget is met.** It is a design target until it is
measured on that hardware, or on a documented proxy.

### Development machine

Verified on this machine rather than assumed:

```
CPU  AMD Ryzen 7 7735HS (8C/16T)
GPU  AMD Radeon 680M (integrated)     nvidia-smi: absent
RAM  27.3 GB
```

**There is no CUDA GPU here**, so the training workflow must not assume one.
Local capability is: data preparation, label export, evaluation of a
predictions file, and CPU inference benchmarking. Not training.

### Training environment — recommendation

Train on a **free or cheap hosted T4/L4 notebook** (Google Colab, Kaggle
Notebooks, or an equivalent), then bring back only the weights.

- 12,000 images at 640 px, YOLOv8n-class, 60 epochs ≈ **1.5–3 h on a T4** —
  inside a free session, and restartable from a checkpoint if it is not.
- The recogniser trains on 48×192 crops and is far smaller: well under an hour.
- Both fit in 16 GB of host RAM and under 8 GB of VRAM at the batch sizes in
  the config.

No account is created, no compute is bought and no external run is started by
this task. The dataset is CC BY 4.0 and the images are already public, so
uploading the training set to a notebook raises no new rights question — but
the **real photographs stay out of it anyway**, because baseline v1 does not
train on them (§5).

## 2. The pipeline

```
image → detector → rectify → recogniser ─┬→ plate type
                                          └→ character sequence
                                                   ↓
                                    validator + confidence → CSV
```

Four stages, three of which are learned, and one — the validator — which
already exists and is already tested.

## 3. Detector

**`yolov8n-pose`, 640×640, one object class (`plate`), four keypoints.**

| | |
| --- | --- |
| Params | ~3.3 M |
| Input | 640×640 letterboxed |
| Output | box + confidence + **4 plate corners** |
| Export | ONNX opset 17 → `onnxruntime` (CUDA provider, CPU fallback) |

**Why a pose model rather than a plain box detector.** Our annotations carry the
plate's four corners exactly — `quad_x1..quad_y4`, and for the 12,000 synthetic
images they are the renderer's own ground truth, not an estimate. A box detector
throws that away and hands the recogniser a crop that is still skewed; 38 % of
the synthetic set carries the `angle` condition. Predicting the corners costs
almost nothing in the same forward pass and buys a perspective rectification
before OCR, which is the single cheapest accuracy win available to us.

**Why one object class, not four.** The type decision needs colour and aspect
resolved at close range, and at 640 px a distant plate is a handful of pixels in
which yellow-vs-white is unreliable. The detector's job is *find the rectangle*;
the recogniser decides what it is, from a 48×192 crop where the evidence is
actually present. This also keeps the detector's rare-class problem from
existing at all: there is no rare class, only plates.

**How the labels are built.** `scripts/export_detector_labels.py` writes
`0 cx cy w h` plus the four normalised corners per row, straight from
`dataset/meta.csv`, clipping to the image. Images are referenced in place — the
exporter writes list files, never copies of 10,200 JPEGs.

> **One open decision, and it is yours.** Ultralytics YOLOv8 is **AGPL-3.0**.
> Using it means our training and inference code, if distributed, is AGPL too.
> This project has been strict about licences, so it should be a deliberate
> choice rather than a default. Permissive alternatives, in order of how little
> else would change: **NanoDet-Plus-m** (Apache-2.0, similar size, no keypoint
> head — corners would need a small separate regressor), or
> **torchvision `ssdlite320_mobilenet_v3_large`** (BSD-3, already a torchvision
> model, weaker on small plates at 320 px). Say which you want before training
> starts; the config changes, nothing else does.

## 4. Type classifier

**Integrated into the recogniser as a second head — not a separate model, and
not folded into the detector.**

The three options, and why this one:

| Option | Verdict |
| --- | --- |
| Separate crop classifier | A third model, a third export, a third thing to keep in sync, for a 4-way decision on a crop the recogniser already has in memory. |
| Classes in the detector | Forces the yellow/white decision at 640 px where a distant plate has no reliable colour. Also creates a rare-class imbalance in the detector that does not need to exist. |
| **Head on the recogniser** ✅ | Shares the backbone, sees the rectified 48×192 crop, costs one small linear layer. The features that read the characters are the features that reveal the layout. |

`type1b` is a **colour** decision and `type1a` is an **aspect and layout**
decision (290×170 mm, two lines, vs 520×112 mm one line). Both are obvious in a
rectified crop and both are ambiguous in a thumbnail. That is the whole argument.

Augmentation is constrained accordingly: `hsv_h = 0.0` in the config, because
hue-shifting a yellow plate towards white would teach the classifier that the
one feature separating `type1b` from `type1` is noise.

## 5. OCR

**CNN → BiGRU → CTC, one model for all three plate types, 48×192 input.**

| | |
| --- | --- |
| Alphabet | **22 symbols + blank** — 10 digits and the 12 letters that exist on Russian plates |
| Max length | 9 (`M000MM` + 3-digit region) |
| Loss | CTC, weight 1.0; type head cross-entropy, weight 0.5 |

**Why CTC rather than character segmentation.** Segmentation needs per-character
boxes, which we do not have and would have to invent; it also fails exactly
where plates are hardest — dirt, glare and occlusion merging adjacent glyphs.
CTC needs only the string, which every row of `meta.csv` already carries.

**Why not a transformer decoder.** 9 characters from a 22-symbol alphabet does
not need attention over a learned vocabulary, and the budget is 100 ms for the
whole pipeline.

**The alphabet is the guardrail.** With 22 symbols the model *cannot* emit a
character no Russian plate carries. A whole family of OCR errors — `0` vs `O`
across scripts, Latin/Cyrillic confusions, punctuation — is removed before
training rather than repaired after it. `src/validator.py` already folds
Cyrillic look-alikes onto Latin, and `src/training/alphabet.py` reuses it.

**Two lines, one reader.** `type1a` is `M 000` on the top line and `MM` + region
below. After rectification the crop is cut at mid-height and the two bands are
placed side by side, so the string becomes one left-to-right sequence in the
correct reading order and a single CTC model reads every class. No second model,
no separate decoder.

**`#` is never predicted.** The annotation guide's "one character I cannot read"
is not in the alphabet, and `alphabet.is_trainable()` excludes any label
containing it from the OCR loss. `#` is produced only at output, by
`alphabet.mask_unconfident()`, from a per-character score below
`char_confidence_for_hash`. Uncertainty is measured, not learned as a glyph.

**What the synthetic labels give us that real data never would.** 12,000 exact
strings, 12,000 distinct plate numbers, exact corners, and — in
`generation.jsonl` — the per-character occlusion record, the rendered contrast,
and the minimum character height in pixels. That last one is a gift: it lets us
plot OCR accuracy against plate resolution and find the size at which the
recogniser actually fails, which is the number that decides the detector's input
resolution in v2.

## 6. Validator and confidence

`src/validator.py` is reused unchanged. It already enforces the two GOST
structures, the 2/3-digit region rule and the 1/2/7 prefix on three-digit
regions.

**The validator must not launder a bad read.** The rule for v1:

- The validator **checks**; it never edits a string into validity. There is no
  "nearest valid plate" step, because a plausible wrong plate is worse than a
  visibly unreadable one — it is wrong *and* confident.
- A read that fails validation is reported with its failure, not silently
  dropped and not repaired.
- Final confidence is `detector_conf × mean(char_scores)`, and the *structural*
  verdict is carried beside it rather than multiplied into it. A structurally
  valid plate read at 0.3 is still a 0.3 read; validity is not evidence.
- Characters below `char_confidence_for_hash` become `#` **before** validation,
  so a `#` cannot be hidden by a lucky structural match.

## 7. What each component trains on

**Baseline v1 trains on synthetic data only.** Every one of the 13 real
photographs is evaluation.

That is a deliberate decision, not an oversight. 10 real images against 10,200
synthetic would not move the weights measurably, and spending them on training
would cost the only real-world measurement the project has. They are worth far
more as a test set than as 0.1 % of a training set.

| Component | Train | Validation (model selection) | Evaluation |
| --- | --- | --- | --- |
| Detector | `synthetic:train` 10,200 | `synthetic:val` 1,200 | `synthetic:test` 600, then real |
| Recogniser (type + OCR) | `synthetic:train` 10,200 | `synthetic:val` 1,200 | `synthetic:test` 600, then real |
| End-to-end | — | — | `synthetic:test`, `real_dev`, `real_holdout` |

**Real evaluation, in two tiers:**

| Tier | Images | What it is for |
| --- | --- | --- |
| `real_dev` (`real:train` + `real:val`) | **11** | Looked at, iterated against, used to find obvious domain failures. |
| `real_holdout` (`real:holdout`) | **2** | Touched **once**, for the final report. |

The frozen group-aware assignment in `dataset/splits/real_splits.csv` is **not
recomputed**. It was written on 2026-09-22 and a group that has been written
down keeps its split forever — that rule is the only thing that makes a holdout
mean anything, and it applies even when we would prefer different numbers.

Because v1 trains on zero real images, all 13 are unseen by the v1 model, and
`real_dev` is a legitimate measurement *for v1*. `real_holdout` stays clean for
the versions after that, when `real:train` becomes fine-tuning data.

**Two guards make this structural rather than aspirational.**
`training.data.load_split("real:holdout")` raises unless given a written reason,
and `scripts/train_baseline.py` refuses to start if any configured training
split contains a real photograph.

### The honest size of the real evaluation

This must be quoted every time a real number is:

| Result on … | 95 % interval |
| --- | --- |
| 2/2 correct on `real_holdout` | **0.34 – 1.00** |
| 13/13 correct on all real | **0.77 – 1.00** |
| 9/13 correct on all real | **0.42 – 0.87** |

**This is an early real holdout and it cannot establish anything.** Two images
distinguish "works" from "catastrophically broken" and nothing finer. Its job in
v1 is to catch a domain collapse — a model that reads synthetic plates perfectly
and real ones not at all — which is a failure large enough for two images to
reveal. Any comparison between two models on it is noise.

`scripts/evaluate_baseline.py` prints the interval next to every proportion and
warns below 30 plates, so the point estimate never appears alone.

### Class coverage in the real set, and its gap

| | `type1` | `type1a` | `type1b` |
| --- | --- | --- | --- |
| `real_dev` (11) | 5 | **0** | 6 |
| `real_holdout` (2) | 1 | **0** | 1 |

**There is no real `type1a` anywhere**, so real-world `type1a` performance —
one of the two classes the competition actually cares about — **cannot be
measured at all in v1**. Synthetic `type1a` accuracy will be reported and must
never be presented as evidence about real `type1a`. `docs/real_data_plan.md`
records why the class is synthetic-only for now.

## 8. Metrics, fixed in advance

Defined in `src/training/metrics.py` and computed by
`scripts/evaluate_baseline.py`. Reported per population, never merged.

**Detection** — precision, recall, F1 at IoU ≥ 0.5; average precision over the
confidence-ranked list; count of images with a prediction where no plate exists.

**Type classification** — precision, recall and F1 **per class**, plus the full
4×4 confusion matrix, plus a **macro** average. Macro, not micro: a
support-weighted average would let `type1b`'s 5,400 synthetic samples bury a
`type1a` failure, which is the failure we most need to see.

**OCR** — exact plate-string accuracy (the competition's own criterion) and CER
beside it. Both per class. Exact accuracy is the result; CER says *how* wrong
the failures are, which decides whether the next move is a better decoder or a
better detector.

**End to end** — the fraction of ground-truth plates that get a correct box, a
correct class **and** a correct string. A missed detection counts as a failure
here, not as an absence. Plus: behaviour on images with no plate, and on images
with several.

**Reported separately, always:** `synthetic:val`, `synthetic:test`, `real_dev`,
`real_holdout`. There is no flag in the evaluator that produces one combined
number, because there is no honest way to combine 600 synthetic images with 2
real ones.

## 9. Readiness

| | |
| --- | --- |
| Dataset frozen and verifiable | ✅ `dataset/splits/dataset_v1.json`, `--check` reproduces |
| Splits deterministic, audited | ✅ seed 2026092201, leakage audit CLEAN |
| Annotations valid | ✅ strict validator, 0 errors, 0 warnings |
| Label export | ✅ `scripts/export_detector_labels.py` |
| Alphabet / CTC coding | ✅ `src/training/alphabet.py`, tested |
| Metrics | ✅ `src/training/metrics.py`, tested |
| Evaluation harness | ✅ `scripts/evaluate_baseline.py`, runs today on a predictions file |
| Configs | ✅ `configs/baseline_v1.json` |
| Holdout guards | ✅ two, both tested |
| Experiment naming, checkpoint policy | ✅ in the config |
| Training loop | ⬜ **not implemented** — needs a framework and a GPU |
| Framework installed | ⬜ **deliberately not** — this task did not authorise training |
| Export to ONNX | ⬜ after training |
| Benchmark on target hardware | ⬜ after export |

The two unticked boxes that matter are the ones the next session opens with, and
both need a person's approval first.

## 10. What v2 does that v1 does not

Recorded now so they are choices later, not discoveries:

- **Fine-tune on `real:train`** once there are enough real images for it to
  mean something. `real_holdout` is already reserved for measuring it.
- **Resolve the 8 QUESTIONABLE staged images** — background plates not yet
  boxed — which would roughly double the real set from material already
  acquired and licence-cleared.
- **Real `type1a`**, if evaluation shows synthetic-only is not enough.
- **Replace the approximate quad** on `wcc_0003`, the one promoted row whose
  corners are a bounding box rather than the plate's true corners.
- **Negative and `other` examples.** Dataset V1 contains **no `other` class
  rows at all**, so the classifier has never seen a non-target plate and the
  detector has never seen a hard negative. `docs/dataset_strategy.md` targets
  50–150 of them. Expect false positives on signage until this is closed.
