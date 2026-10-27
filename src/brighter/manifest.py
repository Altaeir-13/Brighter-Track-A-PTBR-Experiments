"""Reproducibility metadata helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for distribution in ("datasets", "huggingface_hub", "pandas", "pyarrow"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def build_manifest(
    *,
    dataset_id: str,
    configuration: str,
    revision: str,
    split_rows: Mapping[str, int],
    source_columns: list[str],
    local_columns: list[str],
    label_columns: list[str],
    parquet_paths: Mapping[str, Path],
    snapshot_path: Path,
    text_source_column: str,
    id_column: str,
    download_timestamp_utc: str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    def recorded_path(path: Path) -> str:
        if project_root is not None:
            try:
                return path.relative_to(project_root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    return {
        "dataset": dataset_id,
        "configuration": configuration,
        "huggingface_revision": revision,
        "download_timestamp_utc": download_timestamp_utc or utc_now_iso(),
        "huggingface_url": f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}",
        "splits": dict(split_rows),
        "columns": local_columns,
        "source_columns": source_columns,
        "label_columns": label_columns,
        "id_column": id_column,
        "text_source_column": text_source_column,
        "text_raw_column": "text_raw",
        "format": "parquet",
        "processing": "none",
        "text_preservation": "text_raw is an exact copy of the source text column",
        "snapshot_path": snapshot_path.as_posix(),
        "parquet_files": {
            split: {
                "path": recorded_path(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for split, path in parquet_paths.items()
        },
        "software_versions": package_versions(),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
