#!/usr/bin/env python
"""Per-stage latency of the deployed pipeline, on whatever machine runs it.

Measures the ONNX Runtime pipeline exactly as ``run.py`` runs it, split into:

    decode            JPEG -> RGB
    detector_preprocess  stretch to 640, normalise
    detector          the ONNX graph
    detector_postprocess threshold, top-k, NMS, rescale
    recognizer_total  both recogniser passes incl. crop, warp, decode
    recognizer_network   the ONNX graph calls inside recognizer_total
    pipeline_total    everything per image, validator and confidence included

Policy: ``--warmup`` untimed images first (default 10), then every image in the
sample timed once per ``--repeats``; median, mean and p95 per stage.

Hardware, backend (execution provider actually used), input size and the kind
of model (``smoke`` or ``baseline``) are written into the report, so a number
can never be quoted without the machine it came from. **A result on one
machine says nothing about another**: the competition target (i5-7600,
GTX 1050 Ti, <= 100 ms/image) is only met if this script says so *on that
hardware*.

Detection counts matter: an untrained (smoke) detector fires on up to 100
boxes per image, each read twice by the recogniser, so smoke-model timings
over-state the recogniser's share. ``--max-plates`` caps the plates read per
image (default: no cap, as deployed).

Usage::

    python scripts/benchmark_runtime.py --models models/baseline_v1 --images 200 \\
        --providers CUDAExecutionProvider CPUExecutionProvider --out runs/bench/gtx1050ti.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.data import load_split  # noqa: E402


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
        "p95_ms": round(p95, 3),
    }


def _hardware() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        import subprocess

        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False, timeout=10)
        info["nvidia_gpu"] = gpu.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["nvidia_gpu"] = None
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--images", type=int, default=200, help="synthetic:val images to time")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-plates", type=int, default=0)
    parser.add_argument("--providers", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads (0 = its default)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import onnxruntime as ort

    from src.classifier import PlateType
    from src.onnx_backend import StageTimer, load_models
    from src.pipeline import INVALID_FORMAT_PENALTY
    from src.validator import validate_plate

    timer = StageTimer()
    detector, recognizer = load_models(args.models, providers=args.providers, timer=timer)
    if args.threads:
        for stage in (detector, recognizer):
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            stage.session = ort.InferenceSession(
                stage.session._model_path, sess_options=options, providers=stage.session.get_providers()
            )
    images = sorted({s.image for s in load_split("synthetic:val")})
    step = max(1, len(images) // (args.images + args.warmup))
    images = images[::step][: args.images + args.warmup]
    paths = [REPO_ROOT / "dataset" / i for i in images]

    def one(path: Path) -> int:
        started = time.perf_counter()
        detector.cache.path = None                 # force a real decode every time
        detections = list(detector.detect(path))
        if args.max_plates:
            detections = detections[: args.max_plates]
        image = detector.cache.image
        readings = recognizer.read_boxes(image, [(d.box.x1, d.box.y1, d.box.x2, d.box.y2) for d in detections]) if detections else []
        t = time.perf_counter()
        for d, r in zip(detections, readings):
            v = validate_plate(r.text, r.plate_type if PlateType(r.plate_type) != PlateType.OTHER else None)
            _ = d.confidence * r.type_confidence * r.mean_char_score * (1.0 if v.is_valid else INVALID_FORMAT_PENALTY)
        timer.add("validate_and_confidence", time.perf_counter() - t)
        timer.add("pipeline_total", time.perf_counter() - started)
        return len(detections)

    for path in paths[: args.warmup]:
        one(path)
    timer.samples.clear()
    plates = []
    for _ in range(args.repeats):
        for path in paths[args.warmup :]:
            plates.append(one(path))

    report = {
        "hardware": _hardware(),
        "onnxruntime": ort.__version__,
        "providers_used": {"detector": detector.session.get_providers(), "recognizer": recognizer.session.get_providers()},
        "input": {"detector": f"1x3x{detector.size}x{detector.size}", "recognizer": "nx3x48x192, two passes"},
        "source_images": "synthetic:val, strided sample, 1280x720 / 1024x768 JPEG",
        "model_kind": {"detector": detector.meta.get("kind"), "recognizer": recognizer.meta.get("kind")},
        "warmup_images": args.warmup,
        "timed_images": len(plates),
        "plates_read_per_image": _stats([float(p) for p in plates]) if plates else None,
        "stages": {k: _stats(v) for k, v in timer.samples.items()},
        "target": {"hardware": "Intel i5-7600 + GTX 1050 Ti 4 GB, 16 GB RAM", "budget_mean_ms": 100,
                   "met_on_this_machine": None},
    }
    total = report["stages"]["pipeline_total"]["mean_ms"]
    report["target"]["met_on_this_machine"] = total <= 100
    report["target"]["_comment"] = "Only meaningful when this report's hardware IS the target hardware."
    print(json.dumps({k: v for k, v in report.items()}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
