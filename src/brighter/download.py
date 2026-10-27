"""Revision-pinned acquisition and local Parquet materialization."""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .manifest import build_manifest, load_json, sha256_file, utc_now_iso, write_json_atomic
from .validation import (
    EXPECTED_LABELS,
    analyze_dataset,
    build_schema_document,
    infer_id_column,
    infer_text_column,
    render_inventory,
    write_inventory,
    write_schema,
)

DATASET_ID = "brighter-dataset/BRIGHTER-emotion-categories"
CONFIGURATION = "ptbr"
LOGGER = logging.getLogger(__name__)


def discover_revision(requested_revision: str | None = None) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(DATASET_ID, revision=requested_revision)
    except Exception as exc:
        target = requested_revision or "latest"
        raise RuntimeError(f"Unable to resolve Hugging Face dataset revision {target!r}: {exc}") from exc
    if not info.sha:
        raise RuntimeError("Hugging Face returned dataset metadata without a commit SHA")
    revision = str(info.sha)
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise RuntimeError(f"Hugging Face returned an unexpected revision SHA: {revision!r}")
    LOGGER.info("Latest BRIGHTER revision: %s", revision)
    return revision


def _snapshot_files(snapshot: Path) -> dict[str, str]:
    return {
        path.relative_to(snapshot).as_posix(): sha256_file(path)
        for path in sorted(snapshot.rglob("*"))
        if path.is_file() and ".cache" not in path.parts and path.name != ".brighter_snapshot.json"
    }


def verify_snapshot(snapshot: Path, revision: str) -> dict[str, str]:
    if not snapshot.is_dir():
        raise RuntimeError(f"Raw snapshot directory is missing: {snapshot}")
    parquet_files = list((snapshot / CONFIGURATION).glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"Corrupt/incomplete snapshot: no {CONFIGURATION} Parquet files in {snapshot}")
    if not (snapshot / "README.md").is_file():
        raise RuntimeError(f"Corrupt/incomplete snapshot: README.md is missing from {snapshot}")
    marker = snapshot / ".brighter_snapshot.json"
    recorded_files = None
    if marker.exists():
        recorded = load_json(marker)
        if recorded.get("huggingface_revision") != revision:
            raise RuntimeError("Raw snapshot marker revision does not match its directory revision")
        recorded_files = recorded.get("files")
    files = _snapshot_files(snapshot)
    empty = [name for name in files if (snapshot / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Corrupt/incomplete snapshot: empty files: {', '.join(empty)}")
    if recorded_files is not None and files != recorded_files:
        raise RuntimeError("Raw snapshot content differs from its recorded immutable file hashes")
    return files


def ensure_snapshot(raw_revision_dir: Path, revision: str, force: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = raw_revision_dir / "hf_snapshot"
    if snapshot.exists() and not force:
        verify_snapshot(snapshot, revision)
        LOGGER.info("Immutable raw snapshot already exists; reusing %s", snapshot)
        return snapshot
    raw_revision_dir.mkdir(parents=True, exist_ok=True)
    temporary = raw_revision_dir / f".hf_snapshot-download-{uuid.uuid4().hex}"
    LOGGER.info("Downloading ptbr snapshot pinned to %s...", revision)
    try:
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            revision=revision,
            local_dir=temporary,
            allow_patterns=[f"{CONFIGURATION}/**", "README.md", ".gitattributes"],
            force_download=force,
        )
        downloaded_files = verify_snapshot(temporary, revision)
        if snapshot.exists():
            existing_files = verify_snapshot(snapshot, revision)
            if existing_files != downloaded_files:
                raise RuntimeError(
                    "Forced download differs from the immutable existing snapshot; original was preserved"
                )
            LOGGER.info("Forced verification matched the immutable existing snapshot")
            shutil.rmtree(temporary)
            return snapshot
        os.replace(temporary, snapshot)
        write_json_atomic(
            snapshot / ".brighter_snapshot.json",
            {
                "dataset": DATASET_ID,
                "configuration": CONFIGURATION,
                "huggingface_revision": revision,
                "download_timestamp_utc": utc_now_iso(),
                "files": downloaded_files,
            },
        )
        verify_snapshot(snapshot, revision)
        return snapshot
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Failed to download the pinned Hugging Face snapshot: {exc}") from exc


def load_huggingface_dataset(revision: str) -> Mapping[str, Any]:
    try:
        from datasets import load_dataset

        dataset = load_dataset(DATASET_ID, CONFIGURATION, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load configuration {CONFIGURATION!r} at revision {revision}: {exc}"
        ) from exc
    if not dataset.keys():
        raise RuntimeError("Hugging Face returned the ptbr configuration without splits")
    return dataset


def materialize_frames(
    dataset: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[str], str, str, dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    source_columns: list[str] | None = None
    text_column: str | None = None
    id_column: str | None = None
    huggingface_features: dict[str, Any] = {}
    for split in dataset.keys():
        LOGGER.info("Loading %s split...", split)
        frame = dataset[split].to_pandas()
        features = dataset[split].features
        huggingface_features[str(split)] = (
            features.to_dict() if hasattr(features, "to_dict") else str(features)
        )
        current_source_columns = list(frame.columns)
        if source_columns is None:
            source_columns = current_source_columns
            text_column = infer_text_column(frame)
            id_column = infer_id_column(frame)
        elif current_source_columns != source_columns:
            raise RuntimeError(
                f"Split {split!r} has a different schema; refusing to normalize it silently"
            )
        assert text_column is not None
        if "text_raw" in frame.columns and text_column != "text_raw":
            if not frame["text_raw"].equals(frame[text_column]):
                raise RuntimeError("Existing text_raw column differs from the inferred source text")
        elif "text_raw" not in frame.columns:
            insert_at = frame.columns.get_loc(text_column) + 1
            frame.insert(insert_at, "text_raw", frame[text_column].copy(deep=True))
        if not frame["text_raw"].equals(frame[text_column]):
            raise RuntimeError("Text preservation invariant failed while creating text_raw")
        frames[str(split)] = frame
        LOGGER.info("%s rows: %d", split, len(frame))
    assert source_columns is not None and text_column is not None and id_column is not None
    return frames, source_columns, text_column, id_column, huggingface_features


def save_parquet_splits(frames: Mapping[str, pd.DataFrame], interim_dir: Path) -> dict[str, Path]:
    interim_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, frame in frames.items():
        destination = interim_dir / f"{split}.parquet"
        if destination.exists():
            existing = pd.read_parquet(destination)
            if list(existing.columns) != list(frame.columns) or len(existing) != len(frame):
                raise RuntimeError(
                    f"Existing Parquet conflicts with revision-pinned data and was preserved: {destination}"
                )
            if "text_raw" in frame and not existing["text_raw"].equals(frame["text_raw"]):
                raise RuntimeError(f"Existing Parquet has different raw text and was preserved: {destination}")
            LOGGER.info("Parquet already exists; reusing %s", destination)
        else:
            LOGGER.info("Saving %s...", destination.name)
            temporary = destination.with_suffix(".parquet.tmp")
            try:
                frame.to_parquet(temporary, index=False, engine="pyarrow")
                readback = pd.read_parquet(temporary, engine="pyarrow")
                if list(readback.columns) != list(frame.columns) or len(readback) != len(frame):
                    raise RuntimeError("Parquet readback schema/row-count check failed")
                if not readback["text_raw"].equals(frame["text_raw"]):
                    raise RuntimeError("Parquet readback changed the preserved raw text")
                os.replace(temporary, destination)
            except Exception as exc:
                if temporary.exists():
                    temporary.unlink()
                raise RuntimeError(f"Unable to save verified Parquet {destination}: {exc}") from exc
        paths[split] = destination
    return paths


def _summary(analysis: Mapping[str, Any], revision: str, artifacts: list[Path]) -> str:
    labels = analysis["label_columns"]
    positive_totals = {
        label: sum(item["labels"][label]["positive"] for item in analysis["splits"].values())
        for label in labels
    }
    duplicated_ids_total = sum(item["duplicated_ids"] for item in analysis["splits"].values())
    duplicated_texts_total = sum(item["duplicated_texts"] for item in analysis["splits"].values())
    cross_total = sum(
        values["shared_ids"] + values["shared_texts"]
        for values in analysis["cross_split_overlaps"].values()
    )
    invalid_total = sum(
        metric["invalid_count"] + metric["null"]
        for item in analysis["splits"].values()
        for metric in item["labels"].values()
    )
    lines = ["BRIGHTER PT-BR DATASET READY", "", f"Revision: {revision}", "", "Splits:"]
    lines.extend(f"{split}: {item['rows']}" for split, item in analysis["splits"].items())
    lines.extend(["", "Labels:"])
    lines.extend(f"{label}: {positive_totals[label]} positives" for label in labels)
    lines.extend(
        [
            "",
            "Integrity:",
            f"duplicated IDs: {duplicated_ids_total}",
            f"duplicated texts: {duplicated_texts_total}",
            f"cross-split duplicates: {cross_total}",
            f"invalid labels: {invalid_total}",
            "",
            "Artifacts:",
        ]
    )
    lines.extend(str(path) for path in artifacts)
    lines.extend(["", "No preprocessing, balancing or augmentation was applied."])
    return "\n".join(lines)


def run_download(project_root: Path, requested_revision: str | None = None, force: bool = False) -> tuple[str, dict[str, Any]]:
    revision = discover_revision(requested_revision)
    raw_dir = project_root / "data" / "raw" / "brighter_ptbr" / revision
    interim_dir = project_root / "data" / "interim" / "brighter_ptbr" / revision
    metadata_dir = project_root / "metadata" / "brighter_ptbr"
    report_dir = project_root / "reports" / "brighter_ptbr"
    manifest_path = metadata_dir / "dataset_manifest.json"
    old_manifest = load_json(manifest_path) if manifest_path.exists() else None
    if old_manifest and old_manifest.get("huggingface_revision") == revision:
        for split, details in old_manifest.get("parquet_files", {}).items():
            recorded_path = Path(details["path"])
            path = recorded_path if recorded_path.is_absolute() else project_root / recorded_path
            if path.exists() and sha256_file(path) != details["sha256"]:
                raise RuntimeError(
                    f"Existing Parquet for split {split} differs from its recorded SHA-256; file was preserved"
                )
    snapshot = ensure_snapshot(raw_dir, revision, force=force)
    dataset = load_huggingface_dataset(revision)
    frames, source_columns, text_source, id_column, huggingface_features = materialize_frames(dataset)
    analysis = analyze_dataset(frames)
    parquet_paths = save_parquet_splits(frames, interim_dir)
    if old_manifest and old_manifest.get("huggingface_revision") == revision:
        timestamp = old_manifest.get("download_timestamp_utc")
    else:
        timestamp = None
    manifest = build_manifest(
        dataset_id=DATASET_ID,
        configuration=CONFIGURATION,
        revision=revision,
        split_rows={split: len(frame) for split, frame in frames.items()},
        source_columns=source_columns,
        local_columns=list(next(iter(frames.values())).columns),
        label_columns=list(EXPECTED_LABELS),
        parquet_paths=parquet_paths,
        snapshot_path=snapshot.relative_to(project_root),
        text_source_column=text_source,
        id_column=id_column,
        download_timestamp_utc=timestamp,
        project_root=project_root,
    )
    write_json_atomic(manifest_path, manifest)
    roles = {id_column: "id", text_source: "text", "text_raw": "text"}
    roles.update({label: "label" for label in EXPECTED_LABELS})
    schema = build_schema_document(frames, source_columns=source_columns, roles=roles)
    schema.update(
        {
            "dataset": DATASET_ID,
            "configuration": CONFIGURATION,
            "huggingface_revision": revision,
            "huggingface_features_by_split": huggingface_features,
        }
    )
    schema_path = metadata_dir / "dataset_schema.json"
    write_schema(schema_path, schema)
    report_path = report_dir / "data_inventory.md"
    write_inventory(report_path, render_inventory(manifest, analysis))
    for warning in analysis["warnings"]:
        LOGGER.warning("%s", warning)
    for error in analysis["errors"]:
        LOGGER.error("%s", error)
    artifacts = [snapshot, interim_dir, manifest_path, schema_path, report_path]
    return _summary(analysis, revision, artifacts), analysis


def validate_existing(project_root: Path, requested_revision: str | None = None) -> tuple[str, dict[str, Any]]:
    metadata_dir = project_root / "metadata" / "brighter_ptbr"
    manifest_path = metadata_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found; run download first: {manifest_path}")
    manifest = load_json(manifest_path)
    revision = requested_revision or manifest["huggingface_revision"]
    if revision != manifest["huggingface_revision"]:
        raise RuntimeError(
            "Requested revision does not match dataset_manifest.json; run download_brighter.py --revision first"
        )
    interim_dir = project_root / "data" / "interim" / "brighter_ptbr" / revision
    from .validation import read_parquet_splits

    frames = read_parquet_splits(interim_dir)
    expected_splits = list(manifest["splits"])
    if set(frames) != set(expected_splits):
        raise RuntimeError(
            f"Local split names {sorted(frames)} differ from manifest split names {expected_splits}"
        )
    frames = {split: frames[split] for split in expected_splits}
    for split, details in manifest["parquet_files"].items():
        path = project_root / details["path"]
        if not path.exists() or sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"Parquet integrity check failed for split {split}: {path}")
    analysis = analyze_dataset(frames)
    report_path = project_root / "reports" / "brighter_ptbr" / "data_inventory.md"
    write_inventory(report_path, render_inventory(manifest, analysis))
    artifacts = [project_root / manifest["snapshot_path"], interim_dir, manifest_path, metadata_dir / "dataset_schema.json", report_path]
    return _summary(analysis, revision, artifacts), analysis
