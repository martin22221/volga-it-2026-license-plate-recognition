# Baseline V1 — runbook

The exact procedure from "code is ready" to "Baseline V1 is measured", in the
order it must happen. The design is in [`baseline_v1.md`](baseline_v1.md); every
number is in [`../configs/baseline_v1.json`](../configs/baseline_v1.json).

**Order matters.** The real holdout is looked at once, at step 6, and only after
the checkpoints and thresholds are final. Nothing after step 6 may feed back
into a model.

## 0. Before the GPU run: one decision to confirm

`detector.anchors` uses scales 0.05–0.50 instead of torchvision's 0.20–0.95
(measured reason in `baseline_v1.md` §3.6). It is the only departure from
torchvision's own SSDlite configuration. Confirm it, or set `min_ratio: 0.2`,
`max_ratio: 0.95` and commit before packing.

## 1. Pack (on the machine that holds the dataset)

```bash
git status                                    # must be clean
python scripts/pack_training_bundle.py --out ../baseline_v1_training_bundle.zip
```

About 700 MB: every tracked file, the 12,000 synthetic images, the sealed
batch's `generation.jsonl`, and a `TRAINING_BUNDLE.json` marker. **No real
photograph** — training never needs one, and one that is absent cannot leak.

## 2. Train (on a CUDA machine)

```bash
unzip baseline_v1_training_bundle.zip -d baseline_v1 && cd baseline_v1
APPROVER="<name>" bash scripts/gpu_train_baseline_v1.sh
```

The script creates an isolated venv with the pinned versions (CUDA build of
torch 2.14.0 / torchvision 0.29.0), runs both preflights, and then trains the recogniser and the detector. In
bundle mode the preflight verifies `meta.csv`, both split manifests and all
12,000 synthetic images against the Dataset V1 freeze, and **refuses if any real
image is present**. After training it exports to ONNX with parity checks and
evaluates on `synthetic:val`. Result: `baseline_v1_results.tgz`.

What each run records in `runs/<component>/baseline_v1_<date>_2026092201/`:

| File | Content |
| --- | --- |
| `run.json` | experiment id, approver, git commit, Dataset V1 identity (freeze, meta, image digests, seal), seed, config and its hash, versions, hardware, result |
| `metrics.jsonl` | one line per epoch: losses, LR, validation metrics, whether it improved |
| `train.log` | the log |
| `best.pt` / `last.pt` | model weights only (`save_optimizer_state: false`) |

Selection, fixed before any run:

| Component | Selected on | Key | Early stop |
| --- | --- | --- | --- |
| Detector | `synthetic:val` | mAP50, then mAP50-95 | patience 15 |
| Recogniser | `synthetic:val`, deployed two-pass read | type **and** raw string correct, then −CER | patience 10 |

## 3. Bring back and install

Unpack `baseline_v1_results.tgz` into the repository. `models/` and `runs/` are
git-ignored; weights are never committed. `models/baseline_v1/` holds
`detector.onnx`, `recognizer.onnx`, their `.json` sidecars and `parity.json`.

## 4. Fix the operating thresholds — on `synthetic:val` only

`evaluation.detector_confidence` (0.25) and `char_confidence_for_hash` (0.5)
are the values the models were exported with. If they are changed, they are
changed from `synthetic:val` results, then re-exported, **before** step 6.

## 5. Evaluate what may be looked at

```bash
python scripts/evaluate_recognizer.py --models models/baseline_v1 --population synthetic_test --out runs/eval/baseline_v1/recognizer_synthetic_test.json
python scripts/predict_baseline.py   --models models/baseline_v1 --population synthetic_test --out runs/eval/baseline_v1/synthetic_test.predictions.json
python scripts/evaluate_baseline.py  --predictions runs/eval/baseline_v1/synthetic_test.predictions.json --population synthetic_test --out runs/eval/baseline_v1/end_to_end_synthetic_test.json
# real_dev (11 images): the same three commands with --population real_dev
```

## 6. The real holdout — once

```bash
WHY="Baseline V1 final report; checkpoints and thresholds fixed on synthetic:val"
python scripts/evaluate_recognizer.py --models models/baseline_v1 --population real_holdout --final-report "$WHY" --out runs/eval/baseline_v1/recognizer_real_holdout.json
python scripts/predict_baseline.py   --models models/baseline_v1 --population real_holdout --final-report "$WHY" --out runs/eval/baseline_v1/real_holdout.predictions.json
python scripts/evaluate_baseline.py  --predictions runs/eval/baseline_v1/real_holdout.predictions.json --population real_holdout --final-report "$WHY" --out runs/eval/baseline_v1/end_to_end_real_holdout.json
```

Two images. Quote the Wilson interval with every figure (2/2 correct is
0.34–1.00) and report each prediction individually. It detects a domain
collapse and nothing finer.

## 7. Runtime on the competition hardware

On the i5-7600 + GTX 1050 Ti machine, with `onnxruntime-gpu` installed in place
of `onnxruntime`:

```bash
python scripts/benchmark_runtime.py --models models/baseline_v1 --images 200 --warmup 10 \
    --providers CUDAExecutionProvider CPUExecutionProvider --out runs/bench/i5-7600_gtx1050ti_cuda.json
python scripts/benchmark_runtime.py --models models/baseline_v1 --images 200 --warmup 10 \
    --providers CPUExecutionProvider --out runs/bench/i5-7600_cpu.json
```

The ≤ 100 ms/image budget is met only if `pipeline_total.mean_ms` in *that*
report is ≤ 100. A number from any other machine is a proxy, not compliance.

## Smoke runs

`scripts/train_baseline.py --smoke` trains on 64 detector images or 1,024 recognizer
plates for 2 epochs and writes to a `_smoke` directory. Exports from it are
marked `"kind": "smoke"`, and `run.py --models` refuses them without
`--allow-smoke-models`. They prove the plumbing and nothing else.
