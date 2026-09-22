"""The metrics the baseline is judged by, defined before it is trained.

Writing these first is deliberate. Metrics chosen after seeing results are
chosen to flatter them, and with a 2-image real holdout the temptation to pick
the number that looks best would be overwhelming.

Three rules are enforced by the shapes here rather than by discipline:

**Synthetic and real are never averaged.** :func:`report` keeps them in
separate blocks. A single headline number over 12,000 synthetic and 13 real
images is a synthetic number wearing a real one's clothes.

**Every proportion carries an interval.** :func:`wilson_interval` is applied to
accuracy figures, so "100 % on 2 images" is reported as what it is --
0.34 to 1.00 -- rather than as success.

**A character-level score never stands in for a plate-level one.** The
competition scores whole plate strings; CER is a diagnostic, not the result.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: An IoU at or above this counts a detection as matching a ground-truth plate.
IOU_MATCH: float = 0.5


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a proportion, Wilson score.

    Wilson rather than the normal approximation because our sample sizes are
    tiny and often at 0 or 1, exactly where the normal approximation returns
    nonsense such as a negative lower bound or a zero-width interval.
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def proportion(successes: int, total: int) -> dict:
    """A proportion reported honestly: value, counts, and its interval."""
    low, high = wilson_interval(successes, total)
    return {
        "value": (successes / total) if total else None,
        "successes": successes,
        "total": total,
        "ci95": [round(low, 4), round(high, 4)],
    }


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    truth: Sequence[Sequence[float]],
    predicted: Sequence[tuple[Sequence[float], float]],
    threshold: float = IOU_MATCH,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy highest-confidence-first matching of boxes to ground truth.

    Returns ``(matches, unmatched_truth, unmatched_predictions)`` where a match
    is ``(truth_index, prediction_index)``. Greedy by confidence is what a
    detection metric conventionally does and what the competition's own scoring
    would effectively do; it is not the optimal assignment, and does not need
    to be.
    """
    order = sorted(range(len(predicted)), key=lambda i: -predicted[i][1])
    used_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    unmatched_pred: list[int] = []
    for pred_index in order:
        box = predicted[pred_index][0]
        best, best_iou = -1, threshold
        for truth_index, truth_box in enumerate(truth):
            if truth_index in used_truth:
                continue
            score = iou(truth_box, box)
            if score >= best_iou:
                best, best_iou = truth_index, score
        if best >= 0:
            used_truth.add(best)
            matches.append((best, pred_index))
        else:
            unmatched_pred.append(pred_index)
    unmatched_truth = [i for i in range(len(truth)) if i not in used_truth]
    return matches, unmatched_truth, unmatched_pred


def detection_scores(true_positives: int, false_positives: int, false_negatives: int) -> dict:
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else None
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }


def average_precision(scored: Sequence[tuple[float, bool]], n_truth: int) -> float:
    """Average precision over a confidence-ranked list, all-points interpolation.

    ``scored`` is ``(confidence, is_true_positive)``. This is the 11-point-free
    definition used by COCO-style evaluation: the area under the interpolated
    precision-recall curve.
    """
    if n_truth == 0:
        return 0.0
    ordered = sorted(scored, key=lambda item: -item[0])
    tp = fp = 0
    points: list[tuple[float, float]] = []
    for _, is_tp in ordered:
        tp, fp = (tp + 1, fp) if is_tp else (tp, fp + 1)
        points.append((tp / n_truth, tp / (tp + fp)))
    # monotone-decreasing precision envelope, then integrate over recall
    best = 0.0
    envelope: list[tuple[float, float]] = []
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    envelope.reverse()
    area, previous_recall = 0.0, 0.0
    for recall, precision in envelope:
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def confusion(pairs: Iterable[tuple[str, str]], classes: Sequence[str]) -> dict:
    """Confusion matrix as ``{truth: {prediction: count}}``, all classes present."""
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for truth, predicted in pairs:
        if truth in matrix and predicted in matrix[truth]:
            matrix[truth][predicted] += 1
    return matrix


def per_class_scores(matrix: Mapping[str, Mapping[str, int]]) -> dict:
    """Precision, recall and F1 per class, plus macro averages.

    Macro, not micro: with type1a and type1b as the classes that matter, an
    average weighted by support would let a common class hide a rare one's
    failure -- which is exactly the failure we care about.
    """
    classes = list(matrix)
    out: dict[str, dict] = {}
    for name in classes:
        tp = matrix[name][name]
        fn = sum(matrix[name][p] for p in classes if p != name)
        fp = sum(matrix[t][name] for t in classes if t != name)
        out[name] = detection_scores(tp, fp, fn)
        out[name]["support"] = tp + fn
    # A class the model never predicts has undefined precision (0/0). Dropping
    # it from the macro average is exactly how a rare-class failure hides: miss
    # every type1a and the macro score would be computed over the classes that
    # went well. So a class counts here whenever it has support OR received a
    # prediction, and an undefined rate counts as 0 rather than as absent.
    # Only a class that is genuinely absent from both truth and predictions is
    # excluded, because there is nothing to be right or wrong about.
    counted = [
        name
        for name in classes
        if out[name]["support"] > 0 or (out[name]["tp"] + out[name]["fp"]) > 0
    ]
    out["macro"] = {
        "precision": (
            sum(out[n]["precision"] or 0.0 for n in counted) / len(counted) if counted else None
        ),
        "recall": (
            sum(out[n]["recall"] or 0.0 for n in counted) / len(counted) if counted else None
        ),
        "f1": sum(out[n]["f1"] or 0.0 for n in counted) / len(counted) if counted else None,
        "classes_counted": counted,
    }
    return out


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Edit distance, the numerator of CER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def ocr_scores(pairs: Sequence[tuple[str, str]]) -> dict:
    """Exact-plate accuracy and CER over ``(truth, prediction)`` pairs.

    Exact accuracy is the one that matters -- the competition scores the whole
    string -- and CER is reported beside it to show *how* wrong the failures
    are. A system at 0.40 exact / 0.05 CER is one good decoding pass from
    working; at 0.40 exact / 0.45 CER it is not.
    """
    if not pairs:
        return {"exact": proportion(0, 0), "cer": None, "characters": 0, "edits": 0}
    exact = sum(1 for truth, pred in pairs if truth == pred)
    edits = sum(levenshtein(truth, pred) for truth, pred in pairs)
    characters = sum(len(truth) for truth, _ in pairs)
    return {
        "exact": proportion(exact, len(pairs)),
        "cer": (edits / characters) if characters else None,
        "characters": characters,
        "edits": edits,
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """One evaluation, with synthetic and real kept apart on purpose."""

    name: str
    blocks: dict = field(default_factory=dict)

    def add(self, population: str, payload: Mapping) -> None:
        self.blocks[population] = dict(payload)

    def to_dict(self) -> dict:
        return {"evaluation": self.name, "populations": self.blocks}


def report_text(report: EvaluationReport) -> str:
    """The report as a person reads it, with intervals kept visible."""
    lines = [f"Evaluation: {report.name}", ""]
    for population, block in report.blocks.items():
        lines.append(f"[{population}]")
        for section, payload in block.items():
            lines.append(f"  {section}:")
            for key, value in (payload or {}).items():
                if isinstance(value, dict) and "ci95" in value:
                    v = value["value"]
                    shown = "n/a" if v is None else f"{v:.4f}"
                    lines.append(
                        f"    {key:<22} {shown}  "
                        f"[{value['ci95'][0]:.3f}, {value['ci95'][1]:.3f}]  "
                        f"n={value['total']}"
                    )
                else:
                    lines.append(f"    {key:<22} {value}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "EvaluationReport",
    "IOU_MATCH",
    "average_precision",
    "confusion",
    "detection_scores",
    "iou",
    "levenshtein",
    "match_detections",
    "ocr_scores",
    "per_class_scores",
    "proportion",
    "report_text",
    "wilson_interval",
]
