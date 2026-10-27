from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from brighter.curation import (
    FROZEN_REVISION,
    analyze_duplicate_groups,
    curate_track_a,
    identify_track,
    parse_track_id,
    verify_curated_fidelity,
    read_raw_snapshot_frames,
    verify_interim_against_raw,
)
from brighter.manifest import sha256_file
from brighter.validation import EXPECTED_LABELS


def _row(identifier: str, text: str, labels: tuple[int, ...]) -> dict[str, object]:
    positives = [label for label, value in zip(EXPECTED_LABELS, labels, strict=True) if value]
    return {
        "id": identifier,
        "text": text,
        "text_raw": text,
        **dict(zip(EXPECTED_LABELS, labels, strict=True)),
        "emotions": positives,
    }


def test_identifies_track_a_and_track_c_from_original_id_pattern() -> None:
    track_a = parse_track_id("ptbr_dev_track_a_00017")
    track_c = parse_track_id("ptbr_test_track_c_00201")
    assert track_a is not None
    assert (track_a.configuration, track_a.split, track_a.track, track_a.number) == (
        "ptbr",
        "dev",
        "A",
        17,
    )
    assert track_c is not None and track_c.track == "C"
    assert identify_track("unrelated") is None


def test_duplicate_groups_compare_labels_and_detect_conflicts() -> None:
    rows = [
        _row("ptbr_dev_track_a_00001", "mesmo", (1, 0, 0, 0, 0, 0)),
        _row("ptbr_dev_track_c_00002", "mesmo", (1, 0, 0, 0, 0, 0)),
        _row("ptbr_dev_track_a_00002", "conflito", (0, 0, 0, 1, 0, 0)),
        _row("ptbr_dev_track_c_00001", "conflito", (0, 0, 1, 0, 0, 0)),
    ]
    summary, groups = analyze_duplicate_groups(pd.DataFrame(rows), "dev")
    assert summary["duplicate_groups"] == 2
    assert summary["same_text_same_labels"] == 1
    assert summary["same_text_different_labels"] == 1
    assert summary["track_a_track_c_pairs"] == 2
    by_text = {group["text"]: group for group in groups}
    assert by_text["mesmo"]["labels_identical"] is True
    assert by_text["conflito"]["labels_identical"] is False


def test_exact_row_duplicates_have_a_distinct_classification() -> None:
    row = _row("ptbr_dev_track_a_00001", "igual", (0, 0, 0, 0, 0, 0))
    summary, groups = analyze_duplicate_groups(pd.DataFrame([row, row.copy()]), "dev")
    assert summary["exact_row_duplicate"] == 1
    assert groups[0]["classification"] == "exact_row_duplicate"


def test_curation_is_deterministic_and_uses_explicit_track_ids() -> None:
    source = pd.DataFrame(
        [
            _row("ptbr_dev_track_c_00001", "um", (1, 0, 0, 0, 0, 0)),
            _row("ptbr_dev_track_a_00002", "dois", (0, 0, 0, 1, 0, 0)),
            _row("ptbr_dev_track_a_00001", "um", (1, 0, 0, 0, 0, 0)),
            _row("ptbr_dev_track_c_00002", "dois", (0, 0, 0, 1, 0, 0)),
        ]
    )
    curated = curate_track_a({"dev": source}, FROZEN_REVISION)["dev"]
    assert curated["id"].tolist() == [
        "ptbr_dev_track_a_00002",
        "ptbr_dev_track_a_00001",
    ]
    assert curated["source_track"].eq("A").all()
    assert curated["source_split"].eq("dev").all()
    assert curated["source_revision"].eq(FROZEN_REVISION).all()
    fidelity = verify_curated_fidelity({"dev": source}, {"dev": curated})
    assert fidelity["dev"]["id_sets_equal"] is True
    assert fidelity["dev"]["texts_unchanged"] is True
    assert fidelity["dev"]["labels_unchanged"] is True


def test_frozen_curated_artifacts_and_manifest_are_faithful() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "metadata" / "brighter_ptbr" / "track_a_manifest.json"
    assert manifest_path.exists(), "Run scripts/curate_track_a.py before the test suite"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["revision"] == FROZEN_REVISION
    assert manifest["curated_rows"] == {"train": 2226, "dev": 200, "test": 2226}
    assert manifest["track_c_rows_excluded"] == {"train": 0, "dev": 200, "test": 2226}
    assert manifest["label_conflict_groups"] == 0
    assert manifest["cross_split_overlap_detected"] == 1
    assert manifest["cross_split_overlap_removed"] == 0
    assert manifest["text_processing"] == "none"
    integrity = manifest["source_integrity"]
    assert integrity["immutable_snapshot_hashes_verified"] is True
    assert integrity["raw_to_interim_all_columns_equal"] is True
    assert integrity["curated_ids_equal_explicit_track_a_ids"] is True
    assert integrity["curated_original_columns_equal_source"] is True
    assert integrity["texts_unchanged"] is True
    assert integrity["labels_unchanged"] is True
    assert integrity["synthetic_rows_added"] == 0
    snapshot = (
        project_root
        / "data"
        / "raw"
        / "brighter_ptbr"
        / FROZEN_REVISION
        / "hf_snapshot"
    )
    raw_frames = read_raw_snapshot_frames(snapshot)
    interim_frames = {
        split: pd.read_parquet(
            project_root
            / "data"
            / "interim"
            / "brighter_ptbr"
            / FROZEN_REVISION
            / f"{split}.parquet"
        )
        for split in ("train", "dev", "test")
    }
    verify_interim_against_raw(raw_frames, interim_frames)
    for split, details in manifest["curated_files"].items():
        curated_path = project_root / details["path"]
        interim_path = (
            project_root
            / "data"
            / "interim"
            / "brighter_ptbr"
            / FROZEN_REVISION
            / f"{split}.parquet"
        )
        assert sha256_file(curated_path) == details["sha256"]
        curated = pd.read_parquet(curated_path)
        interim = pd.read_parquet(interim_path)
        expected = interim[interim["id"].map(identify_track).eq("A")].reset_index(drop=True)
        assert set(curated["id"]) == set(expected["id"])
        assert curated["text_raw"].equals(expected["text_raw"])
        assert curated[list(EXPECTED_LABELS)].equals(expected[list(EXPECTED_LABELS)])
