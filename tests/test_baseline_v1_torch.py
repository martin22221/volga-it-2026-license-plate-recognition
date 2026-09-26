"""Baseline V1 tests that need the training stack (torch, onnx, onnxruntime).

Skipped as a whole in an inference-only environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="training stack not installed")

from src.recognition import INPUT_HEIGHT, INPUT_WIDTH  # noqa: E402
from src.training.alphabet import encode  # noqa: E402
from src.training.data import CLASSES, load_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))

def test_recognizer_heads_have_the_configured_shapes() -> None:
    from src.training.models import build_recognizer

    model = build_recognizer(CONFIG["recognizer"]).eval()
    log_probs, types, corners = model(torch.zeros(2, 3, INPUT_HEIGHT, INPUT_WIDTH))
    assert log_probs.shape == (2, 48, CONFIG["recognizer"]["num_chars"])
    assert types.shape == (2, len(CLASSES)) and corners.shape == (2, 8)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(2, 48), atol=1e-5)


def test_detector_is_ssdlite_at_640_with_the_configured_anchors() -> None:
    from src.training.models import DetectorExport, build_detector

    model = build_detector(CONFIG["detector"], load_pretrained=False).eval()
    assert type(model).__name__ == "SSD"
    assert model.transform.fixed_size == (640, 640)
    anchors = CONFIG["detector"]["anchors"]
    scales = model.anchor_generator.scales
    assert scales[0] == pytest.approx(anchors["min_ratio"]) and scales[5] == pytest.approx(anchors["max_ratio"])
    # the approved values, in pixels, exactly as the built model uses them
    assert [round(s * 640, 1) for s in scales[:6]] == anchors["pixel_scales_at_640"]
    assert model.anchor_generator.aspect_ratios == [anchors["aspect_ratios_per_level"]] * 6
    boxes, scores = DetectorExport(model, 640)(torch.zeros(1, 3, 640, 640))
    assert boxes.shape[:2] == scores.shape and boxes.shape[-1] == 4


def test_recognizer_dataset_views_and_targets() -> None:
    from src.training.regions import cut_regions
    from src.training.torch_data import RecognizerDataset

    rows = load_split("synthetic:train")[:4]
    dataset = RecognizerDataset(rows, cut_regions(rows), train=True, seed=1)
    corner_view, strip, corners, cls, target, length = dataset[0]
    assert corner_view.shape == strip.shape == (3, INPUT_HEIGHT, INPUT_WIDTH)
    assert corners.shape == (8,) and 0 <= int(cls) < 4
    assert target[: int(length)].tolist() == encode(rows[0].plate_num)
    again = dataset[0]
    assert torch.equal(again[1], strip), "augmentation must be deterministic per (seed, epoch, index)"


def test_detector_dataset_keeps_boxes_inside_the_image_and_refuses_flips() -> None:
    from src.training.torch_data import DetectorDataset, group_by_image

    images = group_by_image(load_split("synthetic:train")[:6])
    dataset = DetectorDataset(images, 640, augment=CONFIG["detector"]["augmentation"], seed=2)
    for index in range(len(dataset)):
        tensor, target, _ = dataset[index]
        assert tensor.shape == (3, 640, 640)
        boxes = target["boxes"]
        assert bool(((boxes >= 0) & (boxes <= 640)).all())
    with pytest.raises(ValueError):
        DetectorDataset(images, 640, augment={**CONFIG["detector"]["augmentation"], "fliplr": 0.5}, seed=2)


@pytest.fixture(scope="module")
def smoke_export(tmp_path_factory):
    """Untrained models exported through the real export script, marked as smoke."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from scripts import export_onnx
    from src.training.models import build_detector, build_recognizer

    root = tmp_path_factory.mktemp("export")
    torch.manual_seed(0)
    for name, model in (("detector", build_detector(CONFIG["detector"], load_pretrained=False)),
                        ("recognizer", build_recognizer(CONFIG["recognizer"]))):
        (root / name).mkdir()
        torch.save({"model": model.state_dict(), "epoch": 0}, root / name / "best.pt")
        (root / name / "run.json").write_text(json.dumps({"kind": "smoke"}), encoding="utf-8")
    code = export_onnx.main(["--detector", str(root / "detector" / "best.pt"),
                             "--recognizer", str(root / "recognizer" / "best.pt"), "--out", str(root / "models")])
    return code, root / "models"


def test_export_matches_torch(smoke_export) -> None:
    code, models = smoke_export
    parity = json.loads((models / "parity.json").read_text(encoding="utf-8"))
    assert parity["recognizer"]["pass"], parity["recognizer"]
    assert parity["detector"]["max_abs_score_diff"] < 1e-3
    assert code == 0
    sidecar = json.loads((models / "detector.json").read_text(encoding="utf-8"))
    assert sidecar["input_shape"] == [1, 3, 640, 640] and sidecar["kind"] == "smoke"


def test_run_refuses_a_smoke_export_unless_told(smoke_export, tmp_path: Path) -> None:
    import run

    _, models = smoke_export
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (320, 240), "gray").save(images / "a.jpg")
    out = tmp_path / "out.csv"
    assert run.main(["--input", str(images), "--output", str(out), "--models", str(models)]) == 2
    assert run.main(["--input", str(images), "--output", str(out), "--models", str(models),
                     "--allow-smoke-models"]) == 0
    assert out.read_text(encoding="utf-8").startswith("image;plate_num;plate_type;confidence")
