# Baseline v1 — architecture, training plan and metrics

What the first trained system will be, what it trains on, and how it is judged.
Written **before** training, so the metrics cannot be chosen to flatter a
result and the splits cannot be redrawn after seeing one.

**Status (2026-09-26): implemented, not yet trained.** Every stage is built,
tested and exported end to end, and has been exercised by *smoke* runs only —
tiny subsets, two epochs, numbers meaningless. The development machine has no
CUDA GPU (§1), so the production training run needs one. The exact procedure is
in [`baseline_v1_runbook.md`](baseline_v1_runbook.md).

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

- 10,200 images at 640 px, SSDlite-MobileNetV3 (~3.4 M params), 60 epochs ≈
  **1.5–3 h on a T4** — inside a free session, and restartable from a
  checkpoint if it is not.
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

**Selected: torchvision `ssdlite320_mobilenet_v3_large`, rebuilt at 640×640,
one object class, fine-tuned from COCO-pretrained weights.**

### 3.1 The licence gate, applied first

**AGPL is excluded.** Ultralytics YOLOv8 — the previous plan — is
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE),
verified 2026-09-23. AGPL's obligation triggers on *distribution*, and a
competition submission is a distribution. Adopting it would force the whole
project's source to AGPL-3.0. That is a licence decision for a person, not a
side effect of picking a detector, so the family is out. YOLOv5 is AGPL-3.0 for
the same reason and is out with it.

### 3.2 Candidates compared

All licences below were read from the projects' own LICENSE files on
2026-09-23, not recalled.

| | **torchvision SSDlite** ✅ | torchvision Faster R-CNN MNv3 | YOLOX-Tiny/Nano | NanoDet-Plus-m | RTMDet-tiny |
| --- | --- | --- | --- | --- | --- |
| Family | SSDLite + MobileNetV3-L | Faster R-CNN + MNv3-L-FPN | YOLOX (anchor-free) | NanoDet-Plus (anchor-free) | RTMDet (MMDetection) |
| **Code licence** | **BSD-3-Clause** | **BSD-3-Clause** | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| **Weights licence** | repo terms + dataset disclaimer (§3.4) | same | Apache-2.0 + same COCO caveat | Apache-2.0 + same COCO caveat | Apache-2.0 + same COCO caveat |
| Params | ~3.4 M | ~19 M | 0.9 / 5.1 M | 0.95–2.4 M | ~4.8 M |
| Input | 320 default, configurable | 320–800 | 416 / 640 | 320–416 | 640 |
| Framework | torchvision (**already required**) | torchvision | +YOLOX repo | +NanoDet repo | +mmcv/mmengine/mmdeploy |
| ONNX export | `torch.onnx.export`, first-class | supported, NMS/RoIAlign caveats | official exporter | official exporter | via MMDeploy |
| Small plates | weakest of the five at 320; helped by 640 | **best recall** | strong (mosaic) | moderate | strong |
| Maintenance | **actively maintained** | actively maintained | stale since 2022 | stale since 2022 | maintained, heavy stack |
| Extra deps | **none** | none | pycocotools, loguru, tabulate | pytorch-lightning (pinned, rot-prone) | large OpenMMLab stack |

### 3.3 Why SSDlite, given it is not the most accurate

Three things decided it, in this order.

**Reproducibility is a stated project value, not a preference.** This repository
seals batches, pins generator requirements and reproduces 12,000 images
byte-for-byte. YOLOX and NanoDet have been effectively unmaintained since 2022;
both need patching against modern NumPy/PyTorch, and NanoDet pins an old
`pytorch-lightning`. Adopting either means inheriting maintenance of a dead
dependency. torchvision is the only candidate that is actively maintained.

**It adds no dependency.** We need `torch` regardless, and `torchvision` comes
with it. Every other candidate is a new third-party repository in the supply
chain of an offline competition submission.

**The task is much easier than COCO, so COCO mAP ranks mislead here.** A
Russian plate is one rigid, high-contrast, near-planar rectangle, and we have
10,200 training images of it. SSDlite's weakness on COCO is a weakness at
80-way classification of deformable objects at 320 px — neither of which is our
problem. Raising the input to 640 addresses the part that *is* our problem.

**The honest disadvantage:** SSD's anchor design and hard-negative mining are
dated, and small-object recall is its weakest axis. So the switch is agreed
*now*, with a measurable trigger rather than a judgement call after seeing
results:

> **If recall on plates narrower than 80 px in `synthetic:val` falls below
> 0.85, switch to `fasterrcnn_mobilenet_v3_large_fpn`** — also BSD-3, also
> torchvision, a one-line config change, at roughly 2× the inference cost.

### 3.4 Pretrained weights are a separate licence question

They are, and the answer is uncomfortable for every candidate equally.

torchvision states it plainly ([models docs](https://docs.pytorch.org/vision/stable/models.html)):

> "The pre-trained models provided in this library may have their own licenses
> or terms and conditions derived from the dataset used for training. It is
> your responsibility to determine whether you have permission to use the
> models for your use case."

The detection weights are COCO-trained, and
[COCO](https://cocodataset.org/#termsofuse) is not a single-licence dataset:
the **annotations** are CC BY 4.0, but the **images** are Flickr images whose
copyright the COCO Consortium does not hold, subject to Flickr's terms.

**This is not a reason to prefer one candidate over another** — YOLOX, NanoDet
and RTMDet weights are all COCO-trained too, so they inherit exactly the same
question behind an Apache-2.0 file header. The code licence differentiates the
candidates; the weights licence does not.

**Pretrained vs from scratch**

| | COCO-pretrained (selected) | From scratch |
| --- | --- | --- |
| Weight provenance | third-party, dataset terms unsettled | **none — fully ours** |
| Synthetic→real transfer | real-photograph priors in the backbone | none; learns only our renderer |
| Convergence | faster, more stable on 10,200 images | slower, needs more augmentation |
| Risk it addresses | the **dominant** risk: domain gap | an **abstract** legal risk |
| Risk it creates | unsettled dataset-derived terms | measurably worse real-world recall |

**Recommendation: fine-tune from the COCO-pretrained weights**, for three
reasons. The synthetic→real gap is our largest and most concrete risk, and we
have only 13 real photographs with which to detect it — a backbone that has
seen real photographs is the cheapest mitigation available. We would ship *our*
fine-tuned weights, not redistribute torchvision's. And the position that model
weights inherit their training data's licence is legally unsettled and
universally relied upon across the industry.

**If you want that uncertainty at zero, it is one flag**:
`detector.pretrained: false` in `configs/baseline_v1.json`. The cost is real —
worse transfer to exactly the images the competition scores — and it is your
call, not mine. The config carries both settings and the comment explaining the
trade.

### 3.5 Consequence: corners move to the recogniser

YOLOv8-pose was chosen partly because it predicted the four plate corners in
one pass. No permissive candidate has an equivalent turnkey keypoint head, so
**corner regression becomes a third head on the recogniser** — which already
receives the crop, so it costs one small output layer and no extra model.

This is arguably the better design regardless: `meta.csv` carries exact corners
for all 12,013 rows, the head is supervised from the first epoch on
ground-truth crops, and the pipeline no longer depends on any detector having a
pose variant. The detector's job narrows to what every candidate does well —
find the rectangle.

Pipeline order becomes: **detect box → crop with margin → corner head →
perspective warp → read**.

### 3.6 Anchor scales — measured, changed, and approved before the first run

Found while implementing, and recorded here because it touches the approved
design. torchvision's SSDlite anchors are **relative to the input size**
(`DefaultBoxGenerator`, `min_ratio=0.2`, `max_ratio=0.95`). Rebuilding at 640
therefore does not make anchors smaller: the smallest is still 0.2 × 640 =
**128 px square**, while a typical Dataset V1 plate, after the 1280×720 frame is
stretched to 640×640, is about 50×25 px.

First screened on every 10th `synthetic:train` plate (1,020 plates):

| Anchor scales | median best IoU | plates with an anchor at IoU ≥ 0.5 |
| --- | --- | --- |
| torchvision default 0.20–0.95 | 0.26 | 25 % |
| **0.05–0.50 (chosen)** | **0.60** | **83 %** |
| 0.04–0.40 | 0.58 | 79 % |
| 0.03–0.35 | 0.57 | 71 % |

Then confirmed on **all 10,200** `synthetic:train` plates (share with an anchor
at IoU ≥ 0.5):

| Plates | n | default 0.20–0.95 | approved 0.05–0.50 |
| --- | --- | --- | --- |
| all | 10,200 | 24.1 % (median best IoU 0.27) | **83.3 %** (0.61) |
| narrower than 80 px (original) | 1,416 | **0.0 %** | **96.6 %** |
| 80–150 px | 4,745 | 9.4 % | 76.8 % |
| 150 px and wider | 4,039 | 49.8 % | 86.2 % |
| `type1` / `type1a` / `type1b` | | 28.0 / 17.3 / 27.7 % | 75.6 / **98.7** / 74.7 % |

Approved anchor scales in pixels at 640: 32, 89.6, 147.2, 204.8, 262.4, 320,
each with a square box, an intermediate square box and 2:1 / 3:1 boxes in both
orientations. Smallest boxes 32×32, 45×23, 55×18; longest side 554 px (the 3:1
box at the 320 px level). Total 12,828 anchors — the same count as the
defaults. Verified from the built model, not only from the config.

With the defaults, three plates in four would be trained from anchors that
barely overlap them, and the small-plate fallback trigger (§3.3) would fire for
a reason unrelated to SSDlite's quality. `configs/baseline_v1.json`
`detector.anchors` therefore sets 0.05–0.50.

**What this does not change:** the model family, the number of anchors per
location (aspect ratios `[2, 3]` on all six levels), every layer shape, and the
COCO-pretrained weights, which load identically (464 tensors; only the 12
class-count-dependent classification convolutions start fresh, exactly as with
the defaults). It changes only the prior box sizes the regression is relative
to. It was chosen from training-split geometry alone, before any training.

**Approved 2026-09-26, before production training**, on these grounds: derived
from training-split geometry only; no validation, test, real or holdout image
and no model metric was used; the SSDlite family, input size, class count,
layer shapes and pretrained-weight loading are unchanged.

Known costs, accepted with the approval: plates filling more than about half
the frame match more loosely (none in Dataset V1; the widest is 315 px at
640); the mean number of well-matched anchors per plate falls from 5.88 to
3.13, because the defaults matched only large plates, and matched them many
times over; the COCO regression head starts calibrated for the old sizes; the
80–150 px band remains the weakest at 76.8 %. **Any further anchor change is a
new experiment, not an edit to Baseline V1.**

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
  *As implemented* (`src/pipeline.py`, reused rather than rewritten): the
  pipeline also multiplies by the type head's probability, and an **invalid**
  read is multiplied by 0.5. Both can only lower a confidence; validity never
  raises one. The validator checks against the structure of the *predicted*
  type, so a one-line `type1` string classified `type1b` is flagged.
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

## 8a. Source-code licence — resolved 2026-09-26: Apache-2.0

The recommendation below was approved and applied: `LICENSE` is the unmodified
Apache License 2.0, covering the project source **outside `dataset/`**.
`dataset/LICENSE` (CC BY 4.0, including the generator) is unchanged, and
third-party software and weights keep their own terms. The full map is in
[`../LICENSING.md`](../LICENSING.md). The original finding follows.

### The finding (2026-09-23)

Discovered while resolving the detector question, and worth recording because
it is what made the AGPL choice consequential.

**`dataset/LICENSE` covers the dataset. There is no LICENSE for the code.**
Under default copyright that means all rights reserved: nobody else may copy,
modify or redistribute this source, and the project has never stated otherwise.

**Is a code licence required for the Volga-IT submission?** *Unknown from this
repository.* The only submission requirement recorded anywhere here is that the
**dataset** is published under CC BY 4.0
(`docs/dataset_strategy.md`, `dataset/README.md`). Nothing in the repository
states whether source must be submitted, published, or licensed. That question
has to be answered from the competition rules, which are not in this repository
— so it is not answered here.

**Why it mattered for the detector.** AGPL-3.0 obliges anyone distributing the
work to license the whole under AGPL-3.0. Had we adopted YOLOv8, submitting the
source would have made that choice for the project by default. Choosing BSD-3
leaves the decision open, which is the right state for a decision nobody has
taken yet.

**Recommendation — add a permissive licence, but not in this commit.**

- **MIT** or **Apache-2.0** are both compatible with everything now selected
  (BSD-3 torchvision). Apache-2.0 additionally grants patent rights and is the
  better default for anything with an ML component.
- It costs nothing and removes an ambiguity that would otherwise surface at
  submission time.
- **Not done here**, because adding a licence is a statement of intent about
  who may use this work, and that belongs to the author, not to a task about
  detector architecture.

The two questions to settle before submission: does the competition require
source, and under what terms do you want it released.

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
| Detector selected, licence cleared | ✅ torchvision SSDlite, BSD-3 (§3) |
| Source-code licence | ✅ Apache-2.0 (`LICENSE`, `LICENSING.md`) |
| Training loops | ✅ `scripts/train_baseline.py`, both components, smoke-tested on CPU |
| Framework | ✅ `requirements-train.txt`, isolated `.venv`, versions pinned |
| Export to ONNX | ✅ `scripts/export_onnx.py`, torch↔ONNX parity checked on every export |
| Deployed pipeline | ✅ `run.py --models`, ONNX Runtime only, refuses smoke exports |
| Evaluation | ✅ `predict_baseline.py` → `evaluate_baseline.py`; `evaluate_recognizer.py` |
| Runtime benchmark | ✅ `scripts/benchmark_runtime.py`, per stage |
| GPU hand-off | ✅ `scripts/pack_training_bundle.py`, `scripts/gpu_train_baseline_v1.sh` |
| **Production training** | ⬜ **needs a CUDA GPU** — measured CPU cost below |
| Benchmark on target hardware | ⬜ needs the i5-7600 + GTX 1050 Ti machine |

Measured on this machine (Ryzen 7 7735HS, CPU only), which is why production
training is not attempted here:

| Component | CPU cost | Configured run |
| --- | --- | --- |
| Recogniser | 3.4 s per 128-plate batch (two views), compute-bound | 40 epochs ≈ **3 h** |
| Detector | ≈ 24 s per 64 images | 60 epochs ≈ **64 h** |

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
