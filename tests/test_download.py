from __future__ import annotations

import json
from pathlib import Path

import pytest

from brighter.download import verify_snapshot
from brighter.manifest import build_manifest, sha256_file, write_json_atomic


def test_manifest_contains_reproducibility_fields_and_hash(tmp_path: Path) -> None:
    parquet = tmp_path / "train.parquet"
    parquet.write_bytes(b"stable parquet test fixture")
    manifest = build_manifest(
        dataset_id="brighter-dataset/BRIGHTER-emotion-categories",
        configuration="ptbr",
        revision="a" * 40,
        split_rows={"train": 3},
        source_columns=["id", "text", "anger"],
        local_columns=["id", "text", "text_raw", "anger"],
        label_columns=["anger"],
        parquet_paths={"train": parquet},
        snapshot_path=Path("data/raw/snapshot"),
        text_source_column="text",
        id_column="id",
        download_timestamp_utc="2026-01-01T00:00:00+00:00",
    )
    assert manifest["huggingface_revision"] == "a" * 40
    assert manifest["processing"] == "none"
    assert manifest["parquet_files"]["train"]["sha256"]
    assert manifest["software_versions"]["python"]


def test_atomic_json_writer_preserves_unicode(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    write_json_atomic(destination, {"texto": "emoção 😊"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"texto": "emoção 😊"}


def test_snapshot_verification_detects_content_change(tmp_path: Path) -> None:
    snapshot = tmp_path / "hf_snapshot"
    (snapshot / "ptbr").mkdir(parents=True)
    (snapshot / "README.md").write_text("card", encoding="utf-8")
    parquet = snapshot / "ptbr" / "train.parquet"
    parquet.write_bytes(b"original")
    files = {
        "README.md": sha256_file(snapshot / "README.md"),
        "ptbr/train.parquet": sha256_file(parquet),
    }
    write_json_atomic(
        snapshot / ".brighter_snapshot.json",
        {"huggingface_revision": "a" * 40, "files": files},
    )
    parquet.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="immutable file hashes"):
        verify_snapshot(snapshot, "a" * 40)
