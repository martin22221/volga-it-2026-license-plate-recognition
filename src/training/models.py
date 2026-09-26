"""The two Baseline V1 networks, and the wrappers they are exported through.

Imports torch at module level on purpose: this module is only ever imported by
the training and export scripts. Nothing on the inference path imports it --
deployed inference runs the exported ONNX graphs through ONNX Runtime.

**Detector** -- torchvision ``ssdlite320_mobilenet_v3_large`` (BSD-3-Clause),
rebuilt at 640x640 with one object class, as selected in
``docs/baseline_v1.md`` section 3. Two things are rebuilt, nothing is replaced:

* the input transform, from 320 to 640;
* the anchor *scales* (``anchors.min_ratio`` / ``max_ratio`` in the config).
  torchvision's anchors are relative to the input, so its default smallest
  anchor at 640 is 128 px square -- larger than most plates in Dataset V1.
  The anchor count per location and every layer shape are unchanged, so the
  COCO-pretrained weights load exactly as they would with the defaults. The
  measurement behind this is in ``docs/baseline_v1.md`` section 3.6.

**Recogniser** -- a small CNN, a 2-layer BiGRU and three heads (CTC characters,
plate type, four corners), section 4-5 of the same document.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------


def build_detector(section: dict, *, load_pretrained: bool | None = None) -> nn.Module:
    """SSDlite-MobileNetV3-Large at ``input_size``, one plate class.

    ``load_pretrained`` overrides ``section["pretrained"]`` (the export and the
    tests build the architecture without downloading anything).
    """
    size = int(section["input_size"])
    anchors = section.get("anchors", {})
    pretrained = section.get("pretrained", True) if load_pretrained is None else load_pretrained

    # num_classes counts the background, as torchvision's detectors do.
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=int(section["num_object_classes"]) + 1,
        score_thresh=float(section.get("score_thresh", 0.001)),
        nms_thresh=float(section.get("nms_thresh", 0.55)),
        detections_per_img=int(section.get("detections_per_img", 100)),
        topk_candidates=int(section.get("topk_candidates", 300)),
    )
    model.anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)],
        min_ratio=float(anchors.get("min_ratio", 0.2)),
        max_ratio=float(anchors.get("max_ratio", 0.95)),
    )
    model.transform = GeneralizedRCNNTransform(
        min_size=size,
        max_size=size,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size_divisible=1,
        fixed_size=(size, size),
    )
    if pretrained:
        load_coco_weights(model)
    return model


def load_coco_weights(model: nn.Module) -> dict:
    """Copy every COCO-pretrained tensor whose shape matches; report what did not.

    Only the final classification convolutions differ (91 COCO classes vs 2),
    and those are the only tensors left at their fresh initialisation.
    """
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    source = weights.get_state_dict(progress=False)
    target = model.state_dict()
    loaded, skipped = [], []
    for key, tensor in source.items():
        if key in target and target[key].shape == tensor.shape:
            target[key] = tensor
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(target)
    return {"weights": str(weights), "url": weights.url, "loaded": len(loaded), "skipped": skipped}


class DetectorExport(nn.Module):
    """The detector as an ONNX graph: normalised 640x640 in, decoded boxes out.

    Box decoding and the softmax are inside the graph (the anchors become a
    constant), NMS is not: it runs in NumPy after the graph
    (:func:`src.onnx_backend.nms`), which keeps the exported graph to operators
    every ONNX Runtime build supports identically on CPU and CUDA.

    Input  ``image``  float32 ``[1, 3, S, S]``, ``(x/255 - 0.5)/0.5``
    Output ``boxes``  float32 ``[1, A, 4]``  x1, y1, x2, y2 in input pixels
           ``scores`` float32 ``[1, A]``     plate probability per anchor
    """

    def __init__(self, model: nn.Module, size: int) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.box_coder = model.box_coder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, size, size)
            features = list(self.backbone(dummy).values())
            from torchvision.models.detection.image_list import ImageList

            anchors = model.anchor_generator(ImageList(dummy, [(size, size)]), features)[0]
        self.register_buffer("anchors", anchors)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = list(self.backbone(image).values())
        out = self.head(features)
        boxes = self.box_coder.decode_single(out["bbox_regression"][0], self.anchors)
        scores = torch.softmax(out["cls_logits"][0], dim=-1)[:, 1]
        return boxes.unsqueeze(0), scores.unsqueeze(0)


# ---------------------------------------------------------------------------
# recogniser
# ---------------------------------------------------------------------------


def _conv(cin: int, cout: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class Recognizer(nn.Module):
    """CNN -> BiGRU -> CTC, with a type head and a corner head on the same features.

    48x192 in; the CNN pools height 48 -> 3 and width 192 -> 48, so the reader
    sees 48 time steps for at most 9 characters -- comfortably above the
    ``2 * 9 + 1`` CTC needs when every character repeats.
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        num_chars: int = 23,
        type_classes: int = 4,
        corner_outputs: int = 8,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            *_conv(3, c1), nn.MaxPool2d(2, 2),                  # 24 x 96
            *_conv(c1, c2), nn.MaxPool2d(2, 2),                 # 12 x 48
            *_conv(c2, c3), *_conv(c3, c3), nn.MaxPool2d((2, 1)),  # 6 x 48
            *_conv(c3, c4), *_conv(c4, c4), nn.MaxPool2d((2, 1)),  # 3 x 48
        )
        self.rnn = nn.GRU(
            c4 * 3, rnn_hidden, num_layers=rnn_layers, bidirectional=True, batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0.0,
        )
        self.ctc = nn.Linear(2 * rnn_hidden, num_chars)
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c4, type_classes)
        )
        self.corner_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 12)), nn.Flatten(),
            nn.Linear(c4 * 36, 128), nn.ReLU(inplace=True), nn.Linear(128, corner_outputs),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f = self.features(x)                                   # N, C, 3, 48
        n, c, h, w = f.shape
        seq = f.permute(0, 3, 1, 2).reshape(n, w, c * h)       # N, T, C*H
        seq, _ = self.rnn(seq)
        log_probs = torch.log_softmax(self.ctc(seq), dim=-1)   # N, T, classes
        return log_probs, self.type_head(f), self.corner_head(f)


def build_recognizer(section: dict) -> Recognizer:
    return Recognizer(
        channels=tuple(section["backbone_channels"]),
        rnn_hidden=int(section["rnn_hidden"]),
        rnn_layers=int(section["rnn_layers"]),
        num_chars=int(section["num_chars"]),
        type_classes=int(section["type_head_classes"]),
        corner_outputs=int(section["corner_head_outputs"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "DetectorExport",
    "Recognizer",
    "build_detector",
    "build_recognizer",
    "count_parameters",
    "load_coco_weights",
]
