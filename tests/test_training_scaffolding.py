"""Tests for the training-side scaffolding.

None of this imports a deep-learning framework, which is the point: the
alphabet, the split guards, the label export and the metrics are all testable
before anything is installed and before anything is trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.alphabet import (
    ALPHABET,
    BLANK_INDEX,
    MAX_LENGTH,
    NUM_CLASSES,
    UNREADABLE,
    AlphabetError,
    collapse,
    decode,
    encode,
    is_trainable,
    mask_unconfident,
)
from src.training.data import (
    CLASSES,
    Sample,
    SplitAccessError,
    load_samples,
    load_split,
    summarise,
    training_pool,
    yolo_line,
)
from src.training.metrics import (
    average_precision,
    confusion,
    detection_scores,
    iou,
    levenshtein,
    match_detections,
    ocr_scores,
    per_class_scores,
    proportion,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ alphabet


def test_the_alphabet_is_exactly_the_symbols_a_russian_plate_can_carry() -> None:
    assert len(ALPHABET) == 22
    assert set("0123456789") <= set(ALPHABET)
    assert set("ABEKMHOPCTYX") == set(ALPHABET) - set("0123456789")
    assert NUM_CLASSES == 23  # + CTC blank
    assert UNREADABLE not in ALPHABET


def test_encode_and_decode_round_trip() -> None:
    for plate in ("K362HH977", "A123BC77", "KM34350", "YY75777"):
        assert decode(encode(plate)) == plate
        assert all(i != BLANK_INDEX for i in encode(plate))


def test_an_unreadable_character_is_never_encoded() -> None:
    """`#` has no ground truth, so it cannot supervise a sequence loss."""
    assert not is_trainable("A12#BC77")
    assert is_trainable("A123BC77")
    with pytest.raises(AlphabetError, match="unreadable"):
        encode("A12#BC77")


def test_a_character_no_plate_carries_is_refused() -> None:
    with pytest.raises(AlphabetError, match="not on a Russian plate"):
        encode("A123BZ77")   # Z is not one of the twelve


def test_an_over_long_plate_is_refused() -> None:
    with pytest.raises(AlphabetError, match="longer than"):
        encode("A123BC7777")
    assert len(encode("A123BC777")) == MAX_LENGTH


def test_cyrillic_is_folded_not_rejected() -> None:
    """Character-set normalisation, not OCR correction."""
    assert decode(encode("А123ВС77")) == "A123BC77"  # Cyrillic А, В, С


def test_greedy_ctc_decode_drops_repeats_then_blanks() -> None:
    a = encode("A")[0]
    b = encode("B")[0]
    assert collapse([a, a, BLANK_INDEX, a, b, b]) == "AAB"
    assert collapse([BLANK_INDEX, BLANK_INDEX]) == ""


def test_low_confidence_characters_become_hashes_at_output_only() -> None:
    masked = mask_unconfident("K362HH977", [0.9] * 4 + [0.1] + [0.9] * 4, 0.5)
    assert masked == "K362#H977"
    assert masked.count(UNREADABLE) == 1


def test_scores_must_correspond_to_characters() -> None:
    with pytest.raises(AlphabetError, match="correspond"):
        mask_unconfident("ABC", [0.9, 0.9], 0.5)


# ------------------------------------------------------------------ split access


def _sample(image: str, split: str, plate_type: str = "type1", plate: str = "A123BC77",
            synthetic: bool = True) -> Sample:
    return Sample(
        image=image, plate_num=plate, plate_type=plate_type,
        bbox=(10.0, 20.0, 40.0, 12.0),
        quad=((10.0, 20.0), (50.0, 20.0), (50.0, 32.0), (10.0, 32.0)),
        conditions=("day",), is_synthetic=synthetic, source="s", split=split,
    )


def test_the_real_holdout_cannot_be_read_by_accident() -> None:
    pool = [_sample("a.jpg", "real:holdout", synthetic=False)]
    with pytest.raises(SplitAccessError, match="once-only"):
        load_split("real:holdout", samples=pool)
    assert len(load_split("real:holdout", samples=pool, unlock_holdout="final report")) == 1


def test_a_blank_reason_does_not_unlock_the_holdout() -> None:
    pool = [_sample("a.jpg", "real:holdout", synthetic=False)]
    with pytest.raises(SplitAccessError):
        load_split("real:holdout", samples=pool, unlock_holdout="   ")


def test_other_splits_need_no_ceremony() -> None:
    pool = [_sample("a.jpg", "synthetic:train"), _sample("b.jpg", "real:train", synthetic=False)]
    assert len(load_split("synthetic:train", samples=pool)) == 1
    assert len(load_split("real:train", samples=pool)) == 1


def test_the_training_pool_is_synthetic_only() -> None:
    """Baseline v1 fits on no real photograph; the pool makes that structural."""
    pool = [
        _sample("a.jpg", "synthetic:train"),
        _sample("b.jpg", "real:train", synthetic=False),
        _sample("c.jpg", "real:holdout", synthetic=False),
        _sample("d.jpg", "synthetic:val"),
    ]
    got = training_pool(pool)
    assert [s.image for s in got] == ["a.jpg"]
    assert all(s.is_synthetic for s in got)


def test_ocr_trainability_excludes_other_and_unreadable() -> None:
    assert _sample("a.jpg", "synthetic:train").ocr_trainable
    assert not _sample("b.jpg", "synthetic:train", plate_type="other", plate="").ocr_trainable
    assert not _sample("c.jpg", "synthetic:train", plate="A12#BC77").ocr_trainable


# ------------------------------------------------------------------ label export


def test_a_yolo_line_is_normalised_and_clipped() -> None:
    sample = _sample("a.jpg", "synthetic:train")
    line = yolo_line(sample, 100, 100, 0)
    parts = line.split()
    assert parts[0] == "0"
    values = [float(v) for v in parts[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[0] == pytest.approx(0.30)   # cx = (10 + 50) / 2 / 100
    assert values[2] == pytest.approx(0.40)   # w  = 40 / 100


def test_a_box_running_past_the_edge_is_clipped_not_rejected() -> None:
    sample = Sample(
        image="a.jpg", plate_num="A123BC77", plate_type="type1",
        bbox=(90.0, 90.0, 40.0, 40.0),
        quad=((90.0, 90.0), (130.0, 90.0), (130.0, 130.0), (90.0, 130.0)),
        conditions=(), is_synthetic=True, source="s", split="synthetic:train",
    )
    values = [float(v) for v in yolo_line(sample, 100, 100, 0).split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_a_box_entirely_outside_the_image_is_an_error() -> None:
    sample = Sample(
        image="a.jpg", plate_num="A123BC77", plate_type="type1",
        bbox=(200.0, 200.0, 10.0, 10.0),
        quad=((200.0, 200.0), (210.0, 200.0), (210.0, 210.0), (200.0, 210.0)),
        conditions=(), is_synthetic=True, source="s", split="synthetic:train",
    )
    with pytest.raises(ValueError, match="degenerate"):
        yolo_line(sample, 100, 100, 0)


# ------------------------------------------------------------------ metrics


def test_a_tiny_sample_gets_a_wide_interval_not_a_flattering_number() -> None:
    """2/2 correct is not 100 % accuracy, and the report must not say it is."""
    low, high = wilson_interval(2, 2)
    assert low < 0.40 and high == 1.0
    result = proportion(2, 2)
    assert result["value"] == 1.0 and result["ci95"][0] < 0.40


def test_an_empty_sample_is_total_ignorance() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert proportion(0, 0)["value"] is None


def test_a_larger_sample_narrows_the_interval() -> None:
    narrow = wilson_interval(900, 1000)
    wide = wilson_interval(9, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


@pytest.mark.parametrize("a,b,expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
    ((0, 0, 10, 10), (20, 20, 5, 5), 0.0),
    ((0, 0, 10, 10), (5, 0, 10, 10), 1 / 3),
])
def test_iou(a, b, expected) -> None:
    assert iou(a, b) == pytest.approx(expected, abs=1e-6)


def test_detections_are_matched_highest_confidence_first() -> None:
    truth = [(0, 0, 10, 10)]
    predicted = [((0, 0, 10, 10), 0.4), ((0, 0, 10, 10), 0.9)]
    matches, unmatched_truth, unmatched_pred = match_detections(truth, predicted)
    assert matches == [(0, 1)]          # the 0.9 box won
    assert unmatched_truth == []
    assert unmatched_pred == [0]        # the 0.4 box is a false positive


def test_a_missed_plate_is_a_false_negative() -> None:
    matches, unmatched_truth, _ = match_detections([(0, 0, 10, 10)], [])
    assert matches == [] and unmatched_truth == [0]


def test_detection_scores_handle_the_empty_case_without_lying() -> None:
    assert detection_scores(0, 0, 0)["precision"] is None
    scores = detection_scores(3, 1, 1)
    assert scores["precision"] == 0.75 and scores["recall"] == 0.75


def test_average_precision_is_one_for_a_perfect_ranking() -> None:
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert average_precision([], 0) == 0.0


def test_the_confusion_matrix_always_has_every_class() -> None:
    matrix = confusion([("type1", "type1"), ("type1b", "type1")], CLASSES)
    assert set(matrix) == set(CLASSES)
    assert all(set(row) == set(CLASSES) for row in matrix.values())
    assert matrix["type1b"]["type1"] == 1


def test_macro_average_does_not_let_a_common_class_hide_a_rare_one() -> None:
    """type1 perfect, type1a entirely wrong: the macro F1 must not look fine."""
    matrix = confusion(
        [("type1", "type1")] * 100 + [("type1a", "type1")] * 4, CLASSES
    )
    scores = per_class_scores(matrix)
    assert scores["type1a"]["recall"] == 0.0
    assert scores["macro"]["recall"] < 0.6


def test_levenshtein_and_cer() -> None:
    assert levenshtein("ABC", "ABC") == 0
    assert levenshtein("ABC", "ABD") == 1
    assert levenshtein("", "AB") == 2


def test_ocr_reports_exact_accuracy_and_cer_separately() -> None:
    scores = ocr_scores([("A123BC77", "A123BC77"), ("K362HH977", "K362HH978")])
    assert scores["exact"]["successes"] == 1 and scores["exact"]["total"] == 2
    assert scores["cer"] == pytest.approx(1 / 17)
    assert scores["exact"]["ci95"][0] < 0.5 < scores["exact"]["ci95"][1]


def test_ocr_on_nothing_is_not_an_error() -> None:
    assert ocr_scores([])["cer"] is None


# ------------------------------------------------------------------ the real dataset


def test_the_committed_dataset_loads_with_its_splits_attached() -> None:
    samples = load_samples()
    assert len(samples) == 12_013
    assert all(s.split != "?:?" for s in samples), "some image has no split"

    pool = training_pool(samples)
    assert len(pool) == 10_200
    assert all(s.is_synthetic for s in pool), "a real photograph reached the training pool"

    summary = summarise(pool)
    assert summary["by_class"] == {"type1": 2040, "type1a": 3570, "type1b": 4590}
    assert summary["unique_plates"] == 10_200


def test_no_real_photograph_is_in_any_training_split() -> None:
    samples = load_samples()
    for split in ("synthetic:train", "synthetic:val", "synthetic:test"):
        rows = load_split(split, samples=samples)
        assert rows, f"{split} is empty"
        assert all(s.is_synthetic for s in rows), f"{split} contains a real photograph"


def test_the_baseline_config_never_points_training_at_real_data() -> None:
    config = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))
    for component in ("detector", "recognizer"):
        section = config[component]
        for key in ("train_split", "val_split", "test_split"):
            assert section[key].startswith("synthetic:"), (
                f"{component}.{key} is {section[key]!r}: baseline v1 trains on synthetic only"
            )
    assert config["evaluation"]["never_merge_synthetic_and_real"] is True
    assert config["checkpoints"]["select_best_on"] == "synthetic:val"


# ------------------------------------------------------------------ licence policy


def test_no_copyleft_dependency_is_configured_for_training() -> None:
    """AGPL/GPL-3 would choose a source licence for a project that has not chosen one.

    `docs/baseline_v1.md` section 8a: this repository declares no source-code
    licence. A copyleft dependency would settle that question by default at
    submission time, so the preflight refuses one outright.
    """
    from scripts.train_baseline import (
        COMPONENT_LICENCES,
        REFUSED_LICENCE_TOKENS,
        REQUIRED_MODULES,
    )

    for component, modules in REQUIRED_MODULES.items():
        for module in modules:
            licence = COMPONENT_LICENCES.get(module)
            assert licence, f"{component} needs {module!r} with no declared licence"
            assert not any(t in licence.lower() for t in REFUSED_LICENCE_TOKENS), (
                f"{component} depends on {module} under {licence}"
            )


def test_the_configured_detector_is_not_from_an_agpl_family() -> None:
    """The decision recorded in docs/baseline_v1.md section 3, locked down."""
    config = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))
    detector = config["detector"]
    family = detector["family"].lower()
    assert "yolov8" not in family and "ultralytics" not in family and "yolov5" not in family
    assert "bsd" in detector["license_code"].lower()
    # the agreed fallback must be permissive too, or the escape hatch is a trap
    assert "bsd" in detector["fallback"]["license_code"].lower()


def test_the_pretrained_weights_choice_is_explicit_and_reversible() -> None:
    """Section 3.4 offers from-scratch as a one-flag alternative; it must exist."""
    config = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))
    detector = config["detector"]
    assert isinstance(detector["pretrained"], bool)
    assert detector["license_weights"], "weights licensing must be stated, not assumed"


def test_corner_regression_moved_to_the_recogniser() -> None:
    """No permissive detector gives keypoints turnkey, so the crop model predicts them."""
    config = json.loads((REPO_ROOT / "configs" / "baseline_v1.json").read_text(encoding="utf-8"))
    recognizer = config["recognizer"]
    assert recognizer["corner_head_outputs"] == 8       # 4 corners, x and y
    assert "keypoints" not in config["detector"]
