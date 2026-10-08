# Baseline V1 — runbook

The exact procedure from "code is ready" to "Baseline V1 is measured", in the
order it must happen. The design is in [`baseline_v1.md`](baseline_v1.md); every
number is in [`../configs/baseline_v1.json`](../configs/baseline_v1.json).

**Order matters.** The real holdout is looked at once, at step 6, and only after
the checkpoints and thresholds are final. Nothing after step 6 may feed back
into a model.

## 0. Anchor scales — approved 2026-09-26

`detector.anchors` = 0.05–0.50 (32–320 px at 640) instead of torchvision's
0.20–0.95, approved before production training (`baseline_v1.md` §3.6). The
trainer builds the detector from these values; `tests/test_baseline_v1_torch.py`
pins them. Nothing to decide before packing.

## 1. Pack (on the machine that holds the dataset)

```bash
git status                                    # must be clean
python scripts/pack_training_bundle.py --out ../baseline_v1_training_bundle.zip
```

About 700 MB: every tracked file, the 12,000 synthetic images, the sealed
batch's `generation.jsonl`, and a `TRAINING_BUNDLE.json` marker. **No real
photograph** — training never needs one, and one that is absent cannot leak.

## 2. Train (on a CUDA machine)

Any Linux machine with an NVIDIA GPU (≥ 4 GB) and internet access (for pip and
the COCO-pretrained weights):

```bash
unzip baseline_v1_training_bundle.zip -d baseline_v1 && cd baseline_v1
sha256sum -c <<< "<bundle sha256>  ../baseline_v1_training_bundle.zip"   # optional
APPROVER="<name>" bash scripts/gpu_train_baseline_v1.sh
```

Variables: `USE_VENV=0` installs into the current interpreter instead of a new
venv (Kaggle / Colab, whose Python often cannot create one); `WORKERS=<n>` sets
data-loader processes (default 4; use 2 on a 2-vCPU Colab);
`TORCH_INDEX=<url>` picks another CUDA build of the same torch version.

**Kaggle (recommended).** Upload the zip as a *private* Dataset (Kaggle unpacks
it). New notebook → Accelerator **GPU T4** (not P100), Internet **on**. In one
cell:

```bash
!cp -r /kaggle/input/<dataset-name>/. /kaggle/working/baseline_v1
!cd /kaggle/working/baseline_v1 && USE_VENV=0 APPROVER="<name>" bash scripts/gpu_train_baseline_v1.sh
```

Then **Save Version → Save & Run All** so it runs in the background (up to 12 h
per session), and download `baseline_v1_results.tgz` from the version's Output.

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
| `best.pt` | the selected weights only: `model`, `epoch`, `config`, `selection` (what `export_onnx.py` reads) |
| `last.pt` | the latest weights plus a `resume` block: optimizer state, shuffle-generator and Python/NumPy/torch/CUDA RNG states, global step (the LR schedule's position), best selection key and epoch (the early-stopping state), and the run's identity. (`checkpoints.save_optimizer_state: false` in the config predates resume support and is not read by code; `best.pt` still carries no optimizer state.) |

Both checkpoints are written atomically (temporary file, then rename). Per
epoch the order is `metrics.jsonl` → `last.pt` → `best.pt` → sync hook; a resume
repairs either possible half-state (a metrics line ahead of `last.pt` is moved
into `run.json` and that epoch retrained; a `best.pt` one write behind is
rebuilt from `last.pt`, which holds the same weights).

Selection, fixed before any run:

| Component | Selected on | Key | Early stop |
| --- | --- | --- | --- |
| Detector | `synthetic:val` | mAP50, then mAP50-95 | patience 15 |
| Recogniser | `synthetic:val`, deployed two-pass read | type **and** raw string correct, then −CER | patience 10 |

### 2a. Interruptions: `--resume` and surviving Kaggle session loss

`python scripts/train_baseline.py --component <c> --i-have-approval "<name>" --resume runs/<c>/<run>/last.pt`
continues **the same run** in its own directory at the next epoch: same LR
schedule position, same optimizer momentum, same shuffle order, same best key
and patience counter. On CPU a resumed run is bit-identical to an uninterrupted
one (`tests/test_train_resume.py`); on CUDA it is as close as the GPU's
non-deterministic kernels allow. It **refuses** (exit 2, nothing trained) a
weights-only file (`best.pt`, or a `last.pt` from before this change), another
component, smoke vs baseline, any configuration difference, another Dataset V1
identity, another code commit (use the same bundle), a run that already
finished, and a run directory whose `run.json`, `metrics.jsonl` or `best.pt`
disagree with the checkpoint. A fresh start into a directory that already holds
a `last.pt` is refused too. Each resume is appended to `run.json` → `resumes`.

**`/kaggle/working` is not persistent.** It dies with the session; a "Save &
Run All" version publishes it only if the run reaches the end. The only place a
notebook can write that survives is Kaggle itself through the API, which needs
an API token. Without a token there is **no safe automatic persistence**; the
fallback is a manual download of the run directory from an interactive session,
which does not survive an unattended crash.

With a token, `scripts/kaggle_persist_run.py` keeps the run in a **private
Kaggle Dataset**, one new version per epoch (~27 MB for the detector; old
versions are kept, so a bad upload can be rolled back):

1. kaggle.com → Settings → API → *Create New Token* (`kaggle.json`). In the
   notebook: Add-ons → Secrets → add `KAGGLE_USERNAME` and `KAGGLE_KEY` with its
   two values, and attach both to the notebook.
2. First cell of every session (exports the token for `!` commands; never print it):

   ```python
   import os
   from kaggle_secrets import UserSecretsClient
   _s = UserSecretsClient()
   os.environ["KAGGLE_USERNAME"] = _s.get_secret("KAGGLE_USERNAME")
   os.environ["KAGGLE_KEY"] = _s.get_secret("KAGGLE_KEY")
   ```

3. Fresh detector run, as a `%%bash` cell (the `init` line only once, ever):

   ```bash
   %%bash
   set -e
   cp -r /kaggle/input/<bundle-dataset>/. /kaggle/working/baseline_v1
   cd /kaggle/working/baseline_v1
   python scripts/kaggle_persist_run.py init --dataset <user>/baseline-v1-detector-ckpt
   USE_VENV=0 COMPONENTS=detector SYNC_DATASET=<user>/baseline-v1-detector-ckpt APPROVER="<name>" \
       bash scripts/gpu_train_baseline_v1.sh
   ```

4. After a session loss, in a new session (after the secrets cell):

   ```bash
   %%bash
   set -e
   cp -r /kaggle/input/<bundle-dataset>/. /kaggle/working/baseline_v1
   cd /kaggle/working/baseline_v1
   python scripts/kaggle_persist_run.py restore --dataset <user>/baseline-v1-detector-ckpt
   LAST=$(ls runs/detector/baseline_v1_*_2026092201/last.pt)
   USE_VENV=0 COMPONENTS=detector SYNC_DATASET=<user>/baseline-v1-detector-ckpt DETECTOR_RESUME="$LAST" \
       APPROVER="<name>" bash scripts/gpu_train_baseline_v1.sh
   ```

   `restore` downloads the *latest* version (an attached `/kaggle/input` copy is
   pinned to the version current when it was attached; `--source <dir>` uses one
   anyway) and puts the run back under its original name. The preflight then
   verifies the checkpoint before any training.

`SYNC_DATASET` makes the script run `kaggle_persist_run.py check` first, so a
missing secret fails before training, not after epoch 20. During training a
failed upload is logged as `SYNC FAILED` in `train.log` and training continues.
`COMPONENTS=detector` skips export (it needs both components) and ends with
`baseline_v1_detector_run.tgz`.

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
