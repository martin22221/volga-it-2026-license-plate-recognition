"""Acquisition of already-approved online candidates.

The behaviour that matters here is refusal: only approved rows are fetched, and
a file whose licence has moved since it was approved is held rather than taken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.online_acquisition import (
    ACQUISITION_FIELDS,
    USER_AGENT,
    acquire,
    acquire_one,
    approved_rows,
    check_against_source,
    fetch_commons_metadata,
    filename_from_url,
    licenses_match,
    normalise_license,
    sha256_file,
    title_from_file_page,
)

UPLOAD = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Moscow_taxi.jpg"


def _row(**over):
    row = {
        "candidate_id": "cw1",
        "file_page_url": "https://commons.wikimedia.org/wiki/File:Moscow_taxi.jpg",
        "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "creator": "Someone",
        "attribution": "Someone",
        "plate_class": "CONFIRMED_TYPE1B",
        "source_group": "commons:someone:2025-01-01",
        "decision": "ACCEPT_FOR_SUBMISSION",
    }
    row.update(over)
    return row


def _live(**over):
    live = {
        "title": "File:Moscow taxi.jpg",
        "url": UPLOAD,
        "descriptionurl": "https://commons.wikimedia.org/wiki/File:Moscow_taxi.jpg",
        "width": 100,
        "height": 50,
        "size": 4,
        "mime": "image/jpeg",
        "sha1": "",
        "license": "CC0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "artist": "Someone",
        "credit": "",
        "attribution_required": "false",
        "restrictions": "",
    }
    live.update(over)
    return live


# --------------------------------------------------------------------------
# what may be acquired at all
# --------------------------------------------------------------------------


def test_only_approved_rows_are_considered() -> None:
    rows = [
        _row(candidate_id="a"),
        _row(candidate_id="b", decision="PENDING"),
        _row(candidate_id="c", decision="REJECT"),
        _row(candidate_id="d", decision="TRAINING_ONLY_IF_LEGAL"),
    ]
    assert [r["candidate_id"] for r in approved_rows(rows)] == ["a"]


def test_an_empty_decision_is_not_an_approval() -> None:
    assert approved_rows([_row(decision=""), _row(decision="  ")]) == []


# --------------------------------------------------------------------------
# licence comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("CC BY 4.0", "cc-by-4.0"),
        ("CC0", "CC0 1.0"),
        ("Public domain", "PD"),
        ("Public domain", "No restrictions"),
        ("CC BY 4.0 ", "CC BY 4.0"),
    ],
)
def test_cosmetic_licence_differences_are_not_relicensing(a: str, b: str) -> None:
    assert licenses_match(a, b)


@pytest.mark.parametrize(
    "a,b",
    [
        ("CC BY 4.0", "CC BY-SA 4.0"),
        ("CC0", "CC BY 4.0"),
        ("CC BY 4.0", "CC BY-NC 4.0"),
        ("CC BY 3.0", "CC BY-SA 3.0"),
    ],
)
def test_a_real_licence_change_is_detected(a: str, b: str) -> None:
    assert not licenses_match(a, b)


def test_normalise_license_is_stable_on_empty() -> None:
    assert normalise_license("") == ""
    assert normalise_license(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the source check, which is the whole point
# --------------------------------------------------------------------------


def test_an_unchanged_file_has_no_problems() -> None:
    assert check_against_source(_row(), _live()) == []


def test_a_file_that_vanished_is_held() -> None:
    problems = check_against_source(_row(), None)
    assert len(problems) == 1 and "not found at the source" in problems[0]


def test_a_relicensed_file_is_held() -> None:
    problems = check_against_source(_row(license="CC0"), _live(license="CC BY-SA 4.0"))
    assert any("licence changed at the source" in p for p in problems)


def test_a_file_that_lost_its_licence_is_held() -> None:
    problems = check_against_source(_row(), _live(license=""))
    assert any("no licence" in p for p in problems)


def test_a_newly_restricted_file_is_held() -> None:
    problems = check_against_source(_row(), _live(restrictions="trademarked"))
    assert any("restriction" in p for p in problems)


def test_a_file_that_is_no_longer_an_image_is_held() -> None:
    problems = check_against_source(_row(), _live(mime="application/pdf"))
    assert any("media type" in p for p in problems)


def test_a_file_with_no_download_url_is_held() -> None:
    problems = check_against_source(_row(), _live(url=""))
    assert any("no download URL" in p for p in problems)


def test_a_licence_that_matches_but_is_not_redistributable_is_held() -> None:
    problems = check_against_source(_row(license="CC BY-SA 4.0"), _live(license="CC BY-SA 4.0"))
    assert any("not redistributable" in p for p in problems)


# --------------------------------------------------------------------------
# filenames: a URL's query string is not part of the name
# --------------------------------------------------------------------------


def test_tracking_parameters_are_not_part_of_the_filename() -> None:
    url = UPLOAD + "?utm_source=commons.wikimedia.org&utm_campaign=imageinfo"
    assert filename_from_url(url) == "Moscow_taxi.jpg"


def test_percent_escapes_are_decoded() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/1/12/%D0%9F%D0%90%D0%97.jpg"
    assert filename_from_url(url) == "ПАЗ.jpg"


@pytest.mark.parametrize("nasty", ["../../evil.jpg", "a/b.jpg", "C:evil.jpg", "x?y.jpg"])
def test_a_filename_can_never_redirect_the_write(nasty: str) -> None:
    name = filename_from_url("https://upload.wikimedia.org/x/" + nasty)
    assert "/" not in name and "\\" not in name and ":" not in name
    assert not name.startswith("..")


def test_title_is_recovered_from_a_file_page_url() -> None:
    assert title_from_file_page(_row()["file_page_url"]) == "File:Moscow taxi.jpg"


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def test_acquire_one_writes_the_file_and_records_its_digest(tmp_path: Path) -> None:
    payload = b"jpeg"
    result = acquire_one(
        _row(), _live(), originals_dir=tmp_path, opener=lambda url, timeout: payload
    )
    assert result.acquired
    written = tmp_path / "Moscow_taxi.jpg"
    assert written.read_bytes() == payload
    assert result.record is not None
    assert result.record["sha256"] == sha256_file(written)
    assert result.record["bytes"] == len(payload)
    assert set(result.record) == set(ACQUISITION_FIELDS)


def test_a_short_download_is_a_failure_not_an_acquisition(tmp_path: Path) -> None:
    result = acquire_one(
        _row(), _live(size=999), originals_dir=tmp_path, opener=lambda url, timeout: b"jpeg"
    )
    assert result.status == "FAILED" and "size mismatch" in result.notes
    assert list(tmp_path.iterdir()) == []


def test_a_checksum_mismatch_removes_the_file(tmp_path: Path) -> None:
    result = acquire_one(
        _row(),
        _live(sha1="0" * 40),
        originals_dir=tmp_path,
        opener=lambda url, timeout: b"jpeg",
    )
    assert result.status == "FAILED" and "SHA-1" in result.notes
    assert list(tmp_path.iterdir()) == []


def test_a_download_error_is_reported_not_raised(tmp_path: Path) -> None:
    def boom(url: str, timeout: float) -> bytes:
        raise OSError("connection reset")

    result = acquire_one(_row(), _live(), originals_dir=tmp_path, opener=boom)
    assert result.status == "FAILED" and "connection reset" in result.notes


def test_a_held_candidate_never_reaches_the_disk(tmp_path: Path) -> None:
    def fail(url: str, timeout: float) -> bytes:  # pragma: no cover - must not run
        raise AssertionError("a held candidate must not be downloaded")

    result = acquire_one(
        _row(), _live(license="CC BY-SA 4.0"), originals_dir=tmp_path, opener=fail
    )
    assert result.status == "HELD"
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_one_held_candidate_does_not_stop_the_others(tmp_path: Path) -> None:
    rows = [
        _row(candidate_id="ok", file_page_url="https://commons.wikimedia.org/wiki/File:A.jpg"),
        _row(
            candidate_id="moved",
            file_page_url="https://commons.wikimedia.org/wiki/File:B.jpg",
            license="CC0",
        ),
    ]

    def opener(url: str, timeout: float) -> bytes:
        if "api.php" in url:
            return json.dumps(
                {
                    "query": {
                        "pages": {
                            "1": {
                                "title": "File:A.jpg",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/x/A.jpg",
                                        "size": 4,
                                        "mime": "image/jpeg",
                                        "width": 10,
                                        "height": 10,
                                        "extmetadata": {"LicenseShortName": {"value": "CC0"}},
                                    }
                                ],
                            },
                            "2": {
                                "title": "File:B.jpg",
                                "imageinfo": [
                                    {
                                        "url": "https://upload.wikimedia.org/x/B.jpg",
                                        "size": 4,
                                        "mime": "image/jpeg",
                                        "width": 10,
                                        "height": 10,
                                        "extmetadata": {
                                            "LicenseShortName": {"value": "CC BY-SA 4.0"}
                                        },
                                    }
                                ],
                            },
                        }
                    }
                }
            ).encode()
        return b"jpeg"

    results = acquire(rows, originals_dir=tmp_path, opener=opener, log=lambda _: None)
    assert [r.status for r in results] == ["ACQUIRED", "HELD"]
    assert [p.name for p in tmp_path.iterdir()] == ["A.jpg"]


def test_metadata_html_is_stripped_from_the_credit_fields() -> None:
    def opener(url: str, timeout: float) -> bytes:
        return json.dumps(
            {
                "query": {
                    "pages": {
                        "1": {
                            "title": "File:A.jpg",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/x/A.jpg",
                                    "size": 1,
                                    "mime": "image/jpeg",
                                    "width": 1,
                                    "height": 1,
                                    "extmetadata": {
                                        "Artist": {"value": '<a href="/wiki/User:X">X</a>'},
                                        "LicenseShortName": {"value": "CC0"},
                                    },
                                }
                            ],
                        }
                    }
                }
            }
        ).encode()

    live = fetch_commons_metadata(["File:A.jpg"], opener=opener)
    assert live["File:A.jpg"]["artist"] == "X"


def test_a_missing_page_is_simply_absent() -> None:
    def opener(url: str, timeout: float) -> bytes:
        return json.dumps({"query": {"pages": {"-1": {"title": "File:Gone.jpg", "missing": ""}}}}).encode()

    assert fetch_commons_metadata(["File:Gone.jpg"], opener=opener) == {}


def test_the_client_identifies_itself() -> None:
    assert "volga-it-2026" in USER_AGENT and "http" in USER_AGENT
