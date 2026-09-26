# Licensing — what covers what

This repository holds four kinds of material under four different sets of
terms. None of them relicenses another.

| Material | Where | Terms |
| --- | --- | --- |
| **Project source code** | everything **outside** `dataset/`: `src/`, `scripts/`, `tests/`, `configs/`, `run.py`, `docs/` | **Apache License 2.0** — [`LICENSE`](LICENSE) |
| **Dataset and its generator** | `dataset/` — annotations, synthetic images, `dataset/generator/`, dataset docs | **CC BY 4.0** for our contributions, as stated in [`dataset/LICENSE`](dataset/LICENSE) |
| **Third-party photographs** | `dataset/images/real/` | each file's **own** licence (CC0 / CC BY 4.0), recorded per row in `dataset/meta.csv` and credited in `dataset/LICENSE` §2a |
| **Third-party software** | installed from `requirements*.txt`, never vendored | each package's own licence (table below) |
| **Pretrained / trained weights** | never committed (`.gitignore`) | see "Model weights" below |

## Project source code — Apache-2.0

`LICENSE` is the unmodified Apache License 2.0 text as published at
<https://www.apache.org/licenses/LICENSE-2.0.txt> (SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
Added on 2026-09-26 following the recommendation recorded in
`docs/baseline_v1.md` §8a (Apache-2.0 over MIT for its explicit patent grant).

**It does not apply to `dataset/`.** `dataset/LICENSE` already released the
synthetic generator under CC BY 4.0 as part of the dataset the competition
requires to be CC BY 4.0, and this file does not change that grant.

## Third-party software

Nothing below is copied into this repository; each is installed by the user and
remains under its own terms. No copyleft (GPL/AGPL) dependency is used —
`scripts/train_baseline.py` refuses one, and Ultralytics YOLO (AGPL-3.0) was
excluded in `docs/baseline_v1.md` §3.1.

| Package | Used for | Licence (from its installed metadata, 2026-09-26) |
| --- | --- | --- |
| numpy | everywhere | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0) |
| Pillow | image I/O, warps | MIT-CMU |
| onnxruntime | deployed inference | MIT |
| torch | training, export only | BSD-3-Clause for PyTorch; bundled components Apache-2.0, BSD-2, BSL-1.0, MIT |
| torchvision | detector architecture, training, export only | BSD-3-Clause |
| onnx | export only | Apache-2.0 |
| pytest | tests only | MIT |

Deployed inference (`run.py --models …`) needs only numpy, Pillow and
onnxruntime.

## Model weights

**COCO-pretrained SSDlite weights (third party).** The detector is fine-tuned
from torchvision's `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`
(`ssdlite320_mobilenet_v3_large_coco-a79551df.pth`, SHA-256
`a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`), downloaded
by torchvision at training time and never committed. torchvision states that
its pretrained models "may have their own licenses or terms and conditions
derived from the dataset used for training"; COCO's annotations are CC BY 4.0
and its images are Flickr images under their owners' terms. That question is
unsettled and is discussed in `docs/baseline_v1.md` §3.4. Setting
`detector.pretrained: false` removes it at a cost in real-image transfer.

**Our trained weights and ONNX exports** are derived from those pretrained
weights and from Dataset V1. They are not committed to git. No licence is
declared for them yet; that is a decision for the project's author, and it
should be taken knowing the pretrained-weights caveat above. The Apache-2.0
grant in `LICENSE` covers source code, not weights.
