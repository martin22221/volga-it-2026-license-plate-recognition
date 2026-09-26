#!/usr/bin/env python
"""Export Baseline V1 checkpoints to ONNX and prove the export reads like the checkpoint.

Writes, into ``--out``:

* ``detector.onnx``   + ``detector.json``    (sidecar: preprocessing, output
  format, thresholds, NMS, provenance)
* ``recognizer.onnx`` + ``recognizer.json``
* ``parity.json``     torch-vs-ONNX comparison on a fixed ``synthetic:val`` sample

Parity is checked at two levels: raw graph outputs (max absolute difference)
and the *decisions* the deployed code makes from them -- the same boxes after
NMS, the same strings, the same plate types. Only ``synthetic:val`` is used;
the real images are never read here.

The sidecar carries ``kind`` from the checkpoint's ``run.json``. ``run.py``
refuses a ``smoke`` export unless told otherwise, so a plumbing-check model
cannot be mistaken for the baseline.

Usage::

    python scripts/export_onnx.py \\
        --detector runs/detector/<run>/best.pt \\
        --recognizer runs/recognizer/<run>/best.pt \\
        --out models/baseline_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "configs" / "baseline_v1.json"
PARITY_IMAGES = 16
PARITY_PLATES = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_record(checkpoint: Path) -> dict:
    path = checkpoint.parent / "run.json"
    if not path.is_file():
        return {"kind": "unknown"}
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        "kind": record.get("kind", "unknown"),
        "experiment_id": record.get("experiment_id"),
        "git_commit": record.get("git_commit"),
        "dataset": record.get("dataset"),
    }


def _stride(rows: list, n: int) -> list:
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def export_detector(checkpoint: Path, out: Path, config: dict, opset: int) -> dict:
    import torch

    from src.training.models import DetectorExport, build_detector

    section = config["detector"]
    size = int(section["input_size"])
    model = build_detector(section, load_pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    wrapper = DetectorExport(model, size).eval()
    path = out / "detector.onnx"
    torch.onnx.export(
        wrapper, (torch.zeros(1, 3, size, size),), str(path), opset_version=opset,
        input_names=["image"], output_names=["boxes", "scores"], dynamo=False,
    )
    meta = {
        "model": section["family"],
        "license_code": section["license_code"],
        "input_name": "image",
        "input_shape": [1, 3, size, size],
        "input_size": size,
        "preprocessing": "RGB, stretched (not letterboxed) to input_size x input_size with bilinear "
                         "resampling, float32, (x/255 - 0.5)/0.5, NCHW",
        "outputs": {
            "boxes": f"float32 [1, {int(wrapper.anchors.shape[0])}, 4] x1,y1,x2,y2 in input pixels (anchors decoded in-graph)",
            "scores": f"float32 [1, {int(wrapper.anchors.shape[0])}] plate probability (softmax in-graph)",
        },
        "postprocessing": "src/onnx_backend.postprocess: score > confidence_threshold, top-k by score, "
                          "greedy NMS at nms_threshold, keep max_detections, clip to input, scale to original pixels",
        "confidence_threshold": float(config["evaluation"]["detector_confidence"]),
        "nms_threshold": float(section["nms_thresh"]),
        "topk_candidates": int(section["topk_candidates"]),
        "max_detections": int(section["detections_per_img"]),
        "anchors": section.get("anchors"),
        "opset": opset,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.is_relative_to(REPO_ROOT) else str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        **_run_record(checkpoint),
        "onnx_sha256": _sha256(path),
        "onnx_bytes": path.stat().st_size,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return {"model": model, "wrapper": wrapper, "path": path, "meta": meta}


def export_recognizer(checkpoint: Path, out: Path, config: dict, opset: int) -> dict:
    import torch

    from src.recognition import INPUT_HEIGHT, INPUT_WIDTH, TWO_LINE_ASPECT
    from src.training.models import build_recognizer

    section = config["recognizer"]
    model = build_recognizer(section)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    path = out / "recognizer.onnx"
    torch.onnx.export(
        model, (torch.zeros(2, 3, INPUT_HEIGHT, INPUT_WIDTH),), str(path), opset_version=opset,
        input_names=["crop"], output_names=["log_probs", "type_logits", "corners"],
        dynamic_axes={"crop": {0: "n"}, "log_probs": {0: "n"}, "type_logits": {0: "n"}, "corners": {0: "n"}},
        dynamo=False,
    )
    meta = {
        "model": section["family"],
        "input_name": "crop",
        "input_shape": ["n", 3, INPUT_HEIGHT, INPUT_WIDTH],
        "_comment_batch": "dynamic batch: one image can hold several plates. The detector graph is fixed at batch 1.",
        "preprocessing": "two passes (src/recognition.read_plates): (1) detector box widened by 8% per side, "
                         "resized to 48x192 -> corners head; (2) perspective warp from predicted corners to 48x192, "
                         f"two-line unroll when quad aspect < {TWO_LINE_ASPECT}; both RGB float32 (x/255-0.5)/0.5 NCHW",
        "outputs": {
            "log_probs": "float32 [n, 48, 23] CTC log-probabilities, index 0 = blank",
            "type_logits": "float32 [n, 4] over [type1, type1a, type1b, other]",
            "corners": "float32 [n, 8] TL,TR,BR,BL x/y normalised to the pass-1 crop",
        },
        "classes": config["classes"],
        "char_confidence_for_hash": float(config["evaluation"]["char_confidence_for_hash"]),
        "opset": opset,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)) if checkpoint.is_relative_to(REPO_ROOT) else str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        **_run_record(checkpoint),
        "onnx_sha256": _sha256(path),
        "onnx_bytes": path.stat().st_size,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return {"model": model, "path": path, "meta": meta}


def detector_parity(det: dict, config: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from src.onnx_backend import OnnxDetector, postprocess, preprocess_detector
    from src.training.data import load_split
    from src.training.evaluation import to_xywh
    from src.training.metrics import iou

    onnx = OnnxDetector(det["path"], confidence=0.01, providers=["CPUExecutionProvider"])
    size = onnx.size
    images = sorted({s.image: s for s in load_split("synthetic:val")}.values(), key=lambda s: s.image)
    images = _stride(images, PARITY_IMAGES)
    box_diff = score_diff = 0.0
    decisions_equal = 0
    worst_iou = 1.0
    for sample in images:
        with Image.open(sample.path) as handle:
            image = handle.convert("RGB")
        batch, sx, sy = preprocess_detector(image, size)
        with torch.no_grad():
            tb, ts = det["wrapper"](torch.from_numpy(batch))
        ob, os_ = onnx.session.run(None, {"image": batch})
        box_diff = max(box_diff, float(np.abs(tb.numpy() - ob).max()))
        score_diff = max(score_diff, float(np.abs(ts.numpy() - os_).max()))
        kwargs = dict(size=size, scale_x=sx, scale_y=sy, score_threshold=0.01,  # low on purpose: exercise NMS on many boxes
                      nms_threshold=onnx.meta["nms_threshold"], topk=onnx.meta["topk_candidates"],
                      max_detections=onnx.meta["max_detections"])
        a = postprocess(tb.numpy()[0], ts.numpy()[0], **kwargs)
        b = postprocess(ob[0], os_[0], **kwargs)
        if len(a) == len(b):
            decisions_equal += 1
            # Match by geometry, not by rank: near-tied scores may legitimately
            # swap order between backends without changing any decision.
            for ba, _ in a:
                best = max((iou(to_xywh(ba), to_xywh(bb)) for bb, _ in b), default=0.0)
                worst_iou = min(worst_iou, best)
    return {
        "images": len(images),
        "max_abs_box_diff_px": box_diff,
        "max_abs_score_diff": score_diff,
        "same_detection_count": f"{decisions_equal}/{len(images)}",
        "worst_matched_iou": worst_iou,
        "pass": decisions_equal == len(images) and worst_iou > 0.99 and score_diff < 1e-3,
    }


def recognizer_parity(rec: dict, config: dict) -> dict:
    import numpy as np
    import torch

    from src.onnx_backend import OnnxRecognizer
    from src.recognition import read_plates
    from src.training.data import load_split
    from src.training.regions import cut_regions

    threshold = float(config["evaluation"]["char_confidence_for_hash"])
    onnx = OnnxRecognizer(rec["path"], providers=["CPUExecutionProvider"])
    samples = _stride(load_split("synthetic:val"), PARITY_PLATES)
    regions = cut_regions(samples)
    items = [(r.image(), r.box) for r in regions]
    diffs = {"log_probs": 0.0, "type_logits": 0.0, "corners": 0.0}

    def torch_run(x):
        with torch.no_grad():
            out = [o.numpy() for o in rec["model"](torch.from_numpy(x))]
        ref = onnx.session.run(None, {"crop": x})
        for name, a, b in zip(diffs, out, ref):
            diffs[name] = max(diffs[name], float(np.abs(a - b).max()))
        return tuple(out)

    a = read_plates(torch_run, items, char_threshold=threshold)
    b = read_plates(onnx.run_model, items, char_threshold=threshold)
    # The GRU is exported with a dynamic batch; check that no batch size was baked in.
    batch_diff = {}
    for n in (1, 3, 64):
        x = np.random.default_rng(n).standard_normal((n, 3, 48, 192)).astype(np.float32)
        with torch.no_grad():
            ref = [o.numpy() for o in rec["model"](torch.from_numpy(x))]
        got = onnx.session.run(None, {"crop": x})
        batch_diff[str(n)] = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    same_text = sum(x.raw_text == y.raw_text for x, y in zip(a, b))
    same_type = sum(x.plate_type == y.plate_type for x, y in zip(a, b))
    same_layout = sum(x.two_line == y.two_line for x, y in zip(a, b))
    return {
        "plates": len(samples),
        "max_abs_diff": diffs,
        "same_raw_text": f"{same_text}/{len(samples)}",
        "same_type": f"{same_type}/{len(samples)}",
        "same_two_line_decision": f"{same_layout}/{len(samples)}",
        "max_abs_diff_by_batch_size": batch_diff,
        "pass": same_text == same_type == same_layout == len(samples)
        and max(diffs.values()) < 1e-3 and max(batch_diff.values()) < 1e-3,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", type=Path, required=True, help="detector checkpoint (best.pt)")
    parser.add_argument("--recognizer", type=Path, required=True, help="recognizer checkpoint (best.pt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    opset = int(config["export"]["opset"])
    args.out.mkdir(parents=True, exist_ok=True)
    det = export_detector(args.detector.resolve(), args.out, config, opset)
    rec = export_recognizer(args.recognizer.resolve(), args.out, config, opset)
    parity = {
        "detector": detector_parity(det, config),
        "recognizer": recognizer_parity(rec, config),
        "population": "synthetic:val (fixed strided sample)",
        "kinds": {"detector": det["meta"]["kind"], "recognizer": rec["meta"]["kind"]},
    }
    (args.out / "parity.json").write_text(json.dumps(parity, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(parity, indent=1))
    print(f"detector.onnx   {det['meta']['onnx_bytes'] / 1e6:.2f} MB")
    print(f"recognizer.onnx {rec['meta']['onnx_bytes'] / 1e6:.2f} MB")
    ok = parity["detector"]["pass"] and parity["recognizer"]["pass"]
    print("parity: PASS" if ok else "parity: FAIL")
    return 0 if ok else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
