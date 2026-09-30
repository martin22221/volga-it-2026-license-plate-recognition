#!/usr/bin/env bash
# Baseline V1 production training on a CUDA machine (Colab / Kaggle / any Linux GPU).
#
# Run from the root of an unpacked training bundle (scripts/pack_training_bundle.py).
# The bundle holds no real photograph; real-image evaluation happens afterwards, on
# the machine that has them, against the exported ONNX models (docs/baseline_v1_runbook.md).
#
#   APPROVER="<name>" bash scripts/gpu_train_baseline_v1.sh
#
# Options (environment variables):
#   USE_VENV=0     install into the current Python instead of a fresh .venv
#                  (Kaggle / Colab, where venv creation often lacks ensurepip)
#   WORKERS=<n>    data-loader worker processes (default 4; 2 on a 2-vCPU Colab)
#   TORCH_INDEX=<url>  another CUDA build of the same torch version
#   COMPONENTS="recognizer detector"  which components to train (default both, in this order).
#                  Export and validation run only when both are trained in this invocation.
#   DETECTOR_RESUME=<last.pt> / RECOGNIZER_RESUME=<last.pt>
#                  continue that interrupted run from its next epoch (train_baseline.py --resume)
#   SYNC_DATASET=<user>/<slug>
#                  after every epoch, push the run to this private Kaggle Dataset
#                  (scripts/kaggle_persist_run.py; needs KAGGLE_USERNAME / KAGGLE_KEY exported).
#                  Checked before training starts, so a missing token fails in seconds.
#
# Estimated on a T4 (not measured): detector ~2-4 h (60 epochs max, early stop patience 15),
# recogniser ~20-40 min; data loading, not the GPU, is the limit. Both write runs/<component>/baseline_v1_<date>_2026092201/.
set -euo pipefail
: "${APPROVER:?set APPROVER to the name of the person who approved this run}"

test -f TRAINING_BUNDLE.json || { echo "run from the root of an unpacked training bundle"; exit 1; }
COMPONENTS="${COMPONENTS:-recognizer detector}"
for c in $COMPONENTS; do
  case "$c" in detector|recognizer) ;; *) echo "unknown component in COMPONENTS: $c"; exit 1 ;; esac
done
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
SYNC_ARGS=()
if [ -n "${SYNC_DATASET:-}" ]; then
  echo "== checking checkpoint dataset $SYNC_DATASET"
  # Never fatal: a Kaggle API or network problem disables sync, it does not stop training.
  if timeout 90 python scripts/kaggle_persist_run.py check --dataset "$SYNC_DATASET"; then
    SYNC_ARGS=(--sync-command "python scripts/kaggle_persist_run.py push --dataset $SYNC_DATASET")
  else
    echo "WARNING: $SYNC_DATASET not reachable; checkpoint sync DISABLED, training continues"
  fi
fi

if [ "${USE_VENV:-1}" = "1" ]; then
  python -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
fi
WORKERS="${WORKERS:-4}"
# Same versions as requirements-train.txt, CUDA build; pick the index matching the driver.
echo "== installing pinned torch/torchvision (CUDA build)"
python -m pip install --progress-bar off --timeout 60 --retries 3 torch==2.14.0 torchvision==0.29.0 --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
python -m pip install --progress-bar off --timeout 60 --retries 3 numpy==2.5.3 pillow==11.3.0 onnx==1.23.0 onnxruntime==1.30.0 pytest
echo "== verifying CUDA"
timeout 300 python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

resume_of() {
  case "$1" in
    detector) echo "${DETECTOR_RESUME:-}" ;;
    recognizer) echo "${RECOGNIZER_RESUME:-}" ;;
  esac
}

# Preflight alone first: verifies every synthetic image against the Dataset V1 freeze
# (and, when resuming, that the checkpoint belongs to exactly this configuration and code).
for c in $COMPONENTS; do
  RESUME="$(resume_of "$c")"
  python scripts/train_baseline.py --component "$c" ${RESUME:+--resume "$RESUME"}
done

for c in $COMPONENTS; do
  RESUME="$(resume_of "$c")"
  python scripts/train_baseline.py --component "$c" --i-have-approval "$APPROVER" --device cuda \
      --workers "$WORKERS" ${RESUME:+--resume "$RESUME"} ${SYNC_ARGS[@]+"${SYNC_ARGS[@]}"}
done

BOTH=0
[[ " $COMPONENTS " == *" detector "* && " $COMPONENTS " == *" recognizer "* ]] && BOTH=1
if [ "$BOTH" = 0 ]; then
  RUNS=()
  for c in $COMPONENTS; do RUNS+=("$(ls -d runs/"$c"/baseline_v1_*_2026092201 | grep -v smoke | tail -1)"); done
  tar czf "baseline_v1_${COMPONENTS// /_}_run.tgz" "${RUNS[@]}"
  echo "done: baseline_v1_${COMPONENTS// /_}_run.tgz (export and validation need both components)"
  exit 0
fi

DET=$(ls -d runs/detector/baseline_v1_*_2026092201 | grep -v smoke | tail -1)
REC=$(ls -d runs/recognizer/baseline_v1_*_2026092201 | grep -v smoke | tail -1)
python scripts/export_onnx.py --detector "$DET/best.pt" --recognizer "$REC/best.pt" --out models/baseline_v1

# Validation-only evaluation (synthetic:val). Model selection already happened on this split.
mkdir -p runs/eval/baseline_v1
python scripts/evaluate_recognizer.py --models models/baseline_v1 --population synthetic_val \
    --out runs/eval/baseline_v1/recognizer_synthetic_val.json
python scripts/predict_baseline.py --models models/baseline_v1 --population synthetic_val \
    --out runs/eval/baseline_v1/synthetic_val.predictions.json
python scripts/evaluate_baseline.py --predictions runs/eval/baseline_v1/synthetic_val.predictions.json \
    --population synthetic_val --out runs/eval/baseline_v1/end_to_end_synthetic_val.json

# Bring back: runs/detector/<run>/, runs/recognizer/<run>/ (run.json, metrics.jsonl, train.log,
# best.pt), models/baseline_v1/ and runs/eval/baseline_v1/.
tar czf baseline_v1_results.tgz "$DET" "$REC" models/baseline_v1 runs/eval/baseline_v1
echo "done: baseline_v1_results.tgz"
