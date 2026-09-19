"""The candidate manifest's consistency rules.

The point of these tests is that an ``ACCEPT_FOR_SUBMISSION`` row cannot be
written unless the evidence beside it actually supports republishing the
photograph in a CC BY 4.0 dataset.
"""

from __future__ import annotations

import pytest

from src.online_discovery import (
    MANIFEST_FIELDS,
    Candidate,
    audit_manifest,
    audit_text,
    candidate_problems,
    gate_verdict,
    read_manifest,
    write_manifest,
)


def accepted(**overrides) -> dict:
    """A row that should pass every rule, so a test can break exactly one thing."""
    row = Candidate(
        candidate_id="cw123",
        source_platform="wikimedia_commons",
        original_url="https://upload.wikimedia.org/x.jpg",
        file_page_url="https://commons.wikimedia.org/wiki/File:X.jpg",
        creator="A. Photographer",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="A. Photographer, CC BY 4.0",
        redistribution_status="yes",
        modification_status="yes",
        plate_class="CONFIRMED_TYPE1B",
        class_confidence="high",
        is_real_photo="yes",
        full_scene_or_crop="full_scene",
        resolution="4000x2252",
        source_group="commons:file:X",
        provenance_status="VERIFIED_ON_SOURCE",
        decision="ACCEPT_FOR_SUBMISSION",
        notes="",
    ).to_row()
    row.update({k: str(v) for k, v in overrides.items()})
    return row


def test_a_fully_evidenced_row_is_clean():
    assert candidate_problems(accepted()) == []


def test_every_manifest_field_is_written_and_read_back(tmp_path):
    path = tmp_path / "m.csv"
    assert write_manifest(path, [accepted()]) == 1
    rows = read_manifest(path)
    assert len(rows) == 1
    assert list(rows[0]) == list(MANIFEST_FIELDS)
    assert rows[0]["license"] == "CC BY 4.0"
    assert candidate_problems(rows[0]) == []


# --- the rights gate ---------------------------------------------------------

@pytest.mark.parametrize(
    "licence",
    ["CC BY-SA 4.0", "CC BY-NC 2.0", "CC BY-ND 4.0", "CC BY-NC-SA 3.0"],
)
def test_share_alike_and_nc_and_nd_cannot_be_accepted(licence):
    problems = candidate_problems(accepted(license=licence))
    assert any("ACCEPT_FOR_SUBMISSION" in p for p in problems), problems


def test_share_alike_is_rejected_even_though_it_permits_redistribution():
    # SA is the trap: it allows republication but forbids relicensing the
    # aggregate under plain CC BY 4.0, which publication requires.
    assert gate_verdict("CC BY-SA 4.0") == "INCOMPATIBLE"
    assert gate_verdict("CC BY 4.0") == "COMPATIBLE"
    assert gate_verdict("CC0") == "COMPATIBLE"


@pytest.mark.parametrize(
    "licence", ["GFDL", "GFDL 1.2", "FAL", "GPL v2", "LGPL-3.0"]
)
def test_non_creative_commons_copyleft_is_incompatible(licence):
    # Copyleft outside the CC family is the ShareAlike problem renamed: it
    # requires the result to carry the same licence, and our dataset is
    # published under plain CC BY 4.0. LGPL-3.0 is the licence that got the
    # Kaggle Nomeroff mirror rejected, so the gate should now catch it itself.
    assert gate_verdict(licence) == "INCOMPATIBLE"
    assert any("ACCEPT_FOR_SUBMISSION" in p for p in candidate_problems(accepted(license=licence)))


def test_an_unrecognised_licence_is_unknown_not_a_soft_yes():
    assert gate_verdict("Some Museum Open Licence v2") == "UNKNOWN"
    problems = candidate_problems(accepted(license="Some Museum Open Licence v2"))
    assert any("confirmed redistributable list" in p for p in problems), problems


def test_empty_licence_blocks_acceptance():
    assert gate_verdict("") == "UNKNOWN"
    assert any("license is empty" in p for p in candidate_problems(accepted(license="")))


def test_missing_licence_url_blocks_acceptance():
    assert any("license_url is empty" in p for p in candidate_problems(accepted(license_url="")))


@pytest.mark.parametrize("field", ["redistribution_status", "modification_status"])
def test_accept_requires_redistribution_and_modification(field):
    problems = candidate_problems(accepted(**{field: "unclear"}))
    assert any(field in p for p in problems), problems


# --- provenance --------------------------------------------------------------

@pytest.mark.parametrize(
    "state", ["PLATFORM_METADATA_ONLY", "UNRESOLVED", "UNKNOWN"]
)
def test_accept_requires_the_source_page_to_have_been_read(state):
    problems = candidate_problems(accepted(provenance_status=state))
    assert any("provenance_status" in p for p in problems), problems


def test_platform_metadata_alone_is_not_provenance():
    # An index such as Openverse reports what the platform claims; that is a
    # lead, not evidence.
    row = accepted(source_platform="openverse", provenance_status="PLATFORM_METADATA_ONLY")
    assert any("its own page has been read" in p for p in candidate_problems(row))


# --- class verification ------------------------------------------------------

@pytest.mark.parametrize("plate_class", ["AMBIGUOUS", "REJECT"])
def test_ambiguous_candidates_cannot_be_accepted(plate_class):
    problems = candidate_problems(accepted(plate_class=plate_class))
    assert any("plate_class" in p for p in problems), problems


def test_a_graphic_is_not_a_photograph():
    problems = candidate_problems(accepted(is_real_photo="no"))
    assert any("is not 'yes'" in p for p in problems), problems


def test_unknown_plate_class_is_rejected_as_a_value():
    assert any("plate_class" in p for p in candidate_problems(accepted(plate_class="TYPE1B")))


# --- attribution -------------------------------------------------------------

def test_cc_by_requires_a_credit_line():
    problems = candidate_problems(accepted(attribution=""))
    assert any("attribution is empty" in p for p in problems), problems


def test_cc0_does_not_require_a_credit_line():
    row = accepted(license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                   attribution="")
    assert candidate_problems(row) == []


def test_public_domain_does_not_require_a_credit_line():
    row = accepted(license="Public domain", license_url="https://example.org/pd", attribution="")
    assert candidate_problems(row) == []


# --- leakage groups ----------------------------------------------------------

def test_accept_requires_a_source_group():
    problems = candidate_problems(accepted(source_group=""))
    assert any("source_group" in p for p in problems), problems


# --- structural --------------------------------------------------------------

def test_a_blank_tristate_is_reported_rather_than_assumed():
    problems = candidate_problems(accepted(redistribution_status=""))
    assert any("record 'unclear'" in p for p in problems), problems


def test_missing_column_is_reported():
    row = accepted()
    del row["source_group"]
    assert any("missing column 'source_group'" in p for p in candidate_problems(row))


def test_unknown_decision_is_reported():
    assert any("decision=" in p for p in candidate_problems(accepted(decision="MAYBE")))


def test_pending_rows_are_allowed_without_full_evidence():
    row = accepted(decision="PENDING", provenance_status="PLATFORM_METADATA_ONLY",
                   plate_class="AMBIGUOUS", license="", license_url="", attribution="")
    assert candidate_problems(row) == []


# --- the audit ---------------------------------------------------------------

def test_audit_counts_and_reports_a_clean_manifest():
    audit = audit_manifest([accepted(), accepted(candidate_id="cw124",
                                                 file_page_url="https://commons.wikimedia.org/wiki/File:Y.jpg",
                                                 source_group="commons:file:Y")])
    assert audit.rows == 2
    assert audit.blocking == []
    assert audit.accepted_by_class == {"CONFIRMED_TYPE1B": 2}
    assert audit.groups == 2
    assert "CONSISTENT" in audit_text(audit)


def test_audit_catches_duplicate_ids_and_urls():
    audit = audit_manifest([accepted(), accepted()])
    assert audit.duplicate_ids == ["cw123"]
    assert audit.duplicate_urls == ["https://commons.wikimedia.org/wiki/File:X.jpg"]
    assert any("duplicate candidate_id" in b for b in audit.blocking)
    assert "BLOCKED" in audit_text(audit)


def test_audit_reports_the_largest_source_group():
    rows = [accepted(candidate_id=f"cw{n}", file_page_url=f"https://c/{n}",
                     source_group="one-burst") for n in range(4)]
    audit = audit_manifest(rows)
    assert audit.groups == 1
    assert audit.largest_group == 4


def test_audit_surfaces_a_bad_row_by_candidate_id():
    audit = audit_manifest([accepted(candidate_id="bad", license="CC BY-SA 4.0")])
    assert any(b.startswith("bad:") for b in audit.blocking)


def test_audit_never_claims_to_approve():
    text = audit_text(audit_manifest([accepted()]))
    assert "never approves" in text
