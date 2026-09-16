"""Tests for real-image intake: source records and staged-source audits.

The fixtures build a staging folder that mirrors the dataset layout, exactly as
``docs/real_data_intake.md`` describes.  No real photograph is involved: the
images are small generated rectangles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.dataset_meta import CSV_DELIMITER, REQUIRED_COLUMNS
from src.real_intake import (
    DECISIONS,
    RECORD_FIELDS,
    SUBMITTABLE,
    audit_staged_source,
    audit_text,
    blank_record,
    dataset_hashes,
    load_source_record,
    near_duplicate_pairs,
    record_problems,
)

SOURCE_ID = "our-own-photos"
PREFIX = f"images/real/{SOURCE_ID}"

#: A record whose rights actually support ACCEPT_FOR_SUBMISSION.
ACCEPTED_RECORD: dict[str, str] = {
    "source_id": SOURCE_ID,
    "source_name": "Photographs taken by the team",
    "source_reference": "photographed by us, public street parking, 2026-09",
    "original_creator": "the team",
    "upstream_source": "none: original photographs",
    "stated_license": "CC BY 4.0",
    "license_evidence_url": "dataset/LICENSE",
    "redistribution_allowed": "yes",
    "modification_allowed": "yes",
    "commercial_use_allowed": "yes",
    "attribution_required": "yes",
    "attribution_text": "Volga-IT 2026 team",
    "provenance_evidence": "we took the photographs; camera originals kept offline",
    "privacy_review": "completed",
    "decision": SUBMITTABLE,
    "decided_by": "a person",
    "date_checked": "2026-09-16",
    "notes": "",
}


def _row(name: str, **overrides: str) -> dict[str, str]:
    row = {
        "image": f"{PREFIX}/{name}",
        "plate_num": "A123BC77",
        "plate_type": "type1a",
        "bbox_x": "10", "bbox_y": "20", "bbox_w": "40", "bbox_h": "12",
        "quad_x1": "10", "quad_y1": "20",
        "quad_x2": "50", "quad_y2": "20",
        "quad_x3": "50", "quad_y3": "32",
        "quad_x4": "10", "quad_y4": "32",
        "is_vehicle": "true",
        "is_synthetic": "false",
        "source": SOURCE_ID,
        "license": "CC BY 4.0",
        "conditions": "day",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [CSV_DELIMITER.join(columns)]
    lines += [CSV_DELIMITER.join(row.get(column, "") for column in columns) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _image(path: Path, colour: tuple[int, int, int] = (120, 120, 120), size=(100, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    for x in range(10, 50):  # a dark bar, so two colours differ as thumbnails
        for y in range(20, 32):
            image.putpixel((x, y), (20, 20, 20))
    image.save(path, "JPEG", quality=90)


def staged(tmp_path: Path, *, names=("a.jpg", "b.jpg"), record: dict | None = ACCEPTED_RECORD,
           privacy: bool = True, groups: bool = True, annotate: bool = True) -> Path:
    """A complete, clean staging folder."""
    root = tmp_path / "incoming" / SOURCE_ID
    for index, name in enumerate(names):
        _image(root / PREFIX / name, colour=(40 + 60 * index, 90, 150))
    if record is not None:
        (root / "source_record.json").write_text(json.dumps(record), encoding="utf-8")
    if annotate:
        _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS), [_row(name) for name in names])
    if groups:
        _write_csv(root / "groups.csv", ["image", "group"],
                   [{"image": f"{PREFIX}/{name}", "group": f"session-{i}"} for i, name in enumerate(names)])
    if privacy:
        _write_csv(root / "privacy_review.csv", ["image", "faces_present", "action", "reviewer", "date"],
                   [{"image": f"{PREFIX}/{name}", "faces_present": "no", "action": "none_needed",
                     "reviewer": "a person", "date": "2026-09-16"} for name in names])
    return root


# ------------------------------------------------------------------ source records


def test_blank_record_has_every_field_and_decides_nothing() -> None:
    record = blank_record("some-source")
    assert set(record) == set(RECORD_FIELDS)
    assert record["decision"] == "PENDING" and record["decision"] in DECISIONS
    assert record["source_id"] == "some-source"


def test_a_consistent_accept_record_has_no_problems() -> None:
    assert record_problems(ACCEPTED_RECORD) == []


def test_missing_and_empty_fields_are_reported() -> None:
    record = dict(ACCEPTED_RECORD)
    del record["provenance_evidence"]
    record["date_checked"] = ""
    problems = record_problems(record)
    assert any("missing field 'provenance_evidence'" in p for p in problems)
    assert any("date_checked is empty" in p for p in problems)


@pytest.mark.parametrize("field", ["redistribution_allowed", "modification_allowed", "commercial_use_allowed"])
def test_accept_needs_every_permission(field: str) -> None:
    problems = record_problems({**ACCEPTED_RECORD, field: "unclear"})
    assert any(field in p and SUBMITTABLE in p for p in problems)


def test_a_blank_permission_is_not_an_answer() -> None:
    problems = record_problems({**ACCEPTED_RECORD, "redistribution_allowed": ""})
    assert any("record 'unclear' rather than leaving it blank" in p for p in problems)


@pytest.mark.parametrize("licence", ["CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-SA 4.0"])
def test_accept_refuses_licences_that_block_republication(licence: str) -> None:
    problems = record_problems({**ACCEPTED_RECORD, "stated_license": licence})
    assert any("incompatible" in p for p in problems)


def test_accept_refuses_an_unrecognised_licence() -> None:
    problems = record_problems({**ACCEPTED_RECORD, "stated_license": "free to use"})
    assert any("not recognised as redistributable" in p for p in problems)


def test_accept_needs_provenance_evidence_not_a_platform_label() -> None:
    problems = record_problems({**ACCEPTED_RECORD, "provenance_evidence": ""})
    assert any("provenance_evidence is empty" in p for p in problems)


def test_accept_needs_a_finished_privacy_review() -> None:
    problems = record_problems({**ACCEPTED_RECORD, "privacy_review": "in_progress"})
    assert any("privacy_review" in p and SUBMITTABLE in p for p in problems)


def test_accept_needs_the_attribution_text_when_attribution_is_required() -> None:
    problems = record_problems({**ACCEPTED_RECORD, "attribution_text": ""})
    assert any("attribution_text is empty" in p for p in problems)


def test_an_unknown_decision_is_refused() -> None:
    assert any("decision" in p for p in record_problems({**ACCEPTED_RECORD, "decision": "LOOKS_FINE"}))


def test_training_only_contradicts_full_redistribution_rights() -> None:
    record = {**ACCEPTED_RECORD, "decision": "TRAINING_ONLY_IF_LEGAL"}
    assert any("reconsider" in p for p in record_problems(record))


def test_unreadable_record_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_source_record(path).problems == [p for p in load_source_record(path).problems if "unreadable" in p]
    path.write_text("[]", encoding="utf-8")
    assert "not a JSON object" in load_source_record(path).problems[0]


# ------------------------------------------------------------------ near duplicates


def test_near_duplicate_pairs_finds_only_similar_thumbnails() -> None:
    a = [1.0, -1.0, 1.0, -1.0]
    assert near_duplicate_pairs({"x": a, "y": list(a)}) == [("x", "y", 1.0)]
    assert near_duplicate_pairs({"x": a, "y": [-v for v in a]}) == []


# ------------------------------------------------------------------ staged source audit


def test_a_clean_source_has_nothing_blocking(tmp_path: Path) -> None:
    audit = audit_staged_source(staged(tmp_path))
    assert audit.blocking == []
    assert audit.promotable
    summary = audit.summary
    assert summary["images"] == 2
    assert summary["formats"] == {"jpeg": 2}
    assert summary["width"]["max"] == 100 and summary["height"]["max"] == 60
    assert summary["annotations"]["rows"] == 2
    assert summary["annotations"]["by_plate_type"] == {"type1a": 2}
    assert summary["annotations"]["unique_plate_numbers"] == 1
    assert summary["annotations"]["validator_errors"] == []
    assert summary["privacy"]["reviewed"] == 2 and summary["privacy"]["unreviewed"] == []
    assert summary["license"]["redistributable"] is True


def test_the_tool_never_approves_a_source(tmp_path: Path) -> None:
    """Without a person's ACCEPT in the record, nothing is promotable."""
    record = {**ACCEPTED_RECORD, "decision": "PENDING"}
    audit = audit_staged_source(staged(tmp_path, record=record))
    assert audit.blocking == [] and not audit.promotable
    text = audit_text(audit)
    assert "NOT PROMOTABLE" in text and "never promotes" in text
    assert "PENDING" in text


def test_a_missing_source_record_blocks_everything(tmp_path: Path) -> None:
    audit = audit_staged_source(staged(tmp_path, record=None))
    assert any("no source record" in item for item in audit.blocking)
    assert not audit.promotable


def test_record_problems_are_reported_as_blocking(tmp_path: Path) -> None:
    audit = audit_staged_source(staged(tmp_path, record={**ACCEPTED_RECORD, "redistribution_allowed": "unclear"}))
    assert any("source record:" in item for item in audit.blocking)
    assert not audit.promotable


def test_corrupt_images_block(tmp_path: Path) -> None:
    root = staged(tmp_path)
    (root / PREFIX / "c.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    audit = audit_staged_source(root)
    assert any("corrupt image" in item for item in audit.blocking)


def test_identical_images_block(tmp_path: Path) -> None:
    root = staged(tmp_path)
    (root / PREFIX / "copy.jpg").write_bytes((root / PREFIX / "a.jpg").read_bytes())
    audit = audit_staged_source(root)
    assert any("identical images staged" in item for item in audit.blocking)
    assert audit.summary["exact_duplicate_groups"]


def test_an_image_already_in_the_dataset_blocks(tmp_path: Path) -> None:
    root = staged(tmp_path)
    import hashlib

    digest = hashlib.sha256((root / PREFIX / "a.jpg").read_bytes()).hexdigest()
    audit = audit_staged_source(root, known_hashes={digest: "images/real/other/x.jpg"})
    assert any("already in the dataset" in item for item in audit.blocking)


def test_unannotated_images_and_orphan_rows_block(tmp_path: Path) -> None:
    root = staged(tmp_path, annotate=False)
    _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS), [_row("a.jpg"), _row("ghost.jpg")])
    audit = audit_staged_source(root)
    assert any("without an annotation row" in item for item in audit.blocking)
    assert any("without an image" in item for item in audit.blocking)


def test_annotation_errors_block(tmp_path: Path) -> None:
    root = staged(tmp_path, annotate=False)
    _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS),
               [_row("a.jpg", plate_type="type1a", plate_num=""), _row("b.jpg")])
    audit = audit_staged_source(root)
    assert any("validator error" in item for item in audit.blocking)


def test_synthetic_rows_in_a_real_source_block(tmp_path: Path) -> None:
    root = staged(tmp_path, annotate=False)
    _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS),
               [_row("a.jpg", is_synthetic="true"), _row("b.jpg")])
    audit = audit_staged_source(root)
    assert any("is_synthetic=true" in item for item in audit.blocking)


def test_missing_privacy_review_blocks(tmp_path: Path) -> None:
    audit = audit_staged_source(staged(tmp_path, privacy=False))
    assert any("without a privacy review" in item for item in audit.blocking)


def test_a_face_left_untreated_blocks(tmp_path: Path) -> None:
    root = staged(tmp_path)
    _write_csv(root / "privacy_review.csv", ["image", "faces_present", "action", "reviewer", "date"],
               [{"image": f"{PREFIX}/a.jpg", "faces_present": "yes", "action": "none_needed",
                 "reviewer": "a person", "date": "2026-09-16"},
                {"image": f"{PREFIX}/b.jpg", "faces_present": "yes", "action": "blurred",
                 "reviewer": "a person", "date": "2026-09-16"}])
    audit = audit_staged_source(root)
    assert any("faces and no privacy action" in item for item in audit.blocking)
    assert audit.summary["privacy"]["actions"] == {"none_needed": 1, "blurred": 1}


def test_rejected_images_are_a_note_not_a_block(tmp_path: Path) -> None:
    root = staged(tmp_path)
    _write_csv(root / "privacy_review.csv", ["image", "faces_present", "action", "reviewer", "date"],
               [{"image": f"{PREFIX}/a.jpg", "faces_present": "yes", "action": "rejected",
                 "reviewer": "a person", "date": "2026-09-16"},
                {"image": f"{PREFIX}/b.jpg", "faces_present": "no", "action": "none_needed",
                 "reviewer": "a person", "date": "2026-09-16"}])
    audit = audit_staged_source(root)
    assert audit.blocking == []
    assert any("rejected by the privacy review" in note for note in audit.notes)


def test_a_group_spanning_splits_blocks(tmp_path: Path) -> None:
    root = staged(tmp_path)
    _write_csv(root / "groups.csv", ["image", "group"],
               [{"image": f"{PREFIX}/a.jpg", "group": "one-car"},
                {"image": f"{PREFIX}/b.jpg", "group": "one-car"}])
    audit = audit_staged_source(
        root,
        group_of={f"{PREFIX}/a.jpg": "one-car", f"{PREFIX}/b.jpg": "one-car"},
        split_assignment={f"{PREFIX}/a.jpg": "train", f"{PREFIX}/b.jpg": "holdout"},
    )
    assert any("span more than one split" in item for item in audit.blocking)


def test_near_duplicates_are_reported_as_a_note(tmp_path: Path) -> None:
    root = tmp_path / "incoming" / SOURCE_ID
    for name in ("a.jpg", "b.jpg"):
        _image(root / PREFIX / name, colour=(70, 90, 150))  # same picture, different files
    (root / "source_record.json").write_text(json.dumps(ACCEPTED_RECORD), encoding="utf-8")
    _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS), [_row("a.jpg"), _row("b.jpg")])
    _write_csv(root / "privacy_review.csv", ["image", "faces_present", "action", "reviewer", "date"],
               [{"image": f"{PREFIX}/{n}", "faces_present": "no", "action": "none_needed",
                 "reviewer": "p", "date": "2026-09-16"} for n in ("a.jpg", "b.jpg")])
    audit = audit_staged_source(root)
    # identical bytes, so it is caught as an exact duplicate too; the pair must be visible
    assert audit.summary["near_duplicate_pairs"]
    assert any("near-duplicate" in note for note in audit.notes)


def test_an_empty_folder_blocks(tmp_path: Path) -> None:
    root = tmp_path / "incoming" / SOURCE_ID
    root.mkdir(parents=True)
    (root / "source_record.json").write_text(json.dumps(ACCEPTED_RECORD), encoding="utf-8")
    audit = audit_staged_source(root)
    assert any("no images found" in item for item in audit.blocking)


def test_a_missing_folder_is_reported() -> None:
    audit = audit_staged_source(Path("does/not/exist"))
    assert audit.blocking and not audit.promotable


def test_audit_is_serialisable(tmp_path: Path) -> None:
    audit = audit_staged_source(staged(tmp_path))
    text = json.dumps(audit.to_dict())
    assert SOURCE_ID in text and '"promotable": true' in text


def test_dataset_hashes_indexes_existing_images(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _image(root / PREFIX / "a.jpg")
    _write_csv(root / "meta.csv", list(REQUIRED_COLUMNS), [_row("a.jpg")])
    hashes = dataset_hashes(root / "meta.csv", root)
    assert list(hashes.values()) == [f"{PREFIX}/a.jpg"]


# ------------------------------------------------------------------ the CLI


def test_cli_reports_a_clean_source(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import audit_real_source as cli

    root = staged(tmp_path)
    report, machine = tmp_path / "out" / "audit.txt", tmp_path / "out" / "audit.json"
    code = cli.main([str(root), "--dataset", str(tmp_path / "no-dataset"),
                     "--report", str(report), "--json", str(machine)])
    assert code == cli.EXIT_OK
    assert "READY FOR A PERSON'S PROMOTION DECISION" in capsys.readouterr().out
    assert report.exists() and json.loads(machine.read_text(encoding="utf-8"))["promotable"] is True


def test_cli_exits_2_when_something_blocks(tmp_path: Path) -> None:
    from scripts import audit_real_source as cli

    root = staged(tmp_path, privacy=False)
    assert cli.main([str(root), "--dataset", str(tmp_path / "no-dataset")]) == cli.EXIT_BLOCKING


def test_cli_rejects_a_missing_folder(tmp_path: Path) -> None:
    from scripts import audit_real_source as cli

    assert cli.main([str(tmp_path / "nope")]) == cli.EXIT_UNUSABLE


def test_cli_prints_a_blank_record(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import audit_real_source as cli

    assert cli.main(["--blank-record", "new-source"]) == cli.EXIT_OK
    printed = json.loads(capsys.readouterr().out)
    assert printed["source_id"] == "new-source" and printed["decision"] == "PENDING"


def test_the_staging_area_holds_no_images_yet() -> None:
    """Nothing has been acquired: the committed staging tree is empty."""
    staging = Path(__file__).resolve().parents[1] / "data" / "real_staging"
    images = [p for p in staging.rglob("*") if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    assert images == []
