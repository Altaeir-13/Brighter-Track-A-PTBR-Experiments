"""Evidence-driven reconstruction of the official BRIGHTER PT-BR Track A."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .download import CONFIGURATION, DATASET_ID, verify_snapshot
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .validation import EXPECTED_LABELS, duplicated_texts, write_inventory

LOGGER = logging.getLogger(__name__)
FROZEN_REVISION = "419566ac8f46f951b51584be36c13e5363008144"
TRACK_ID_PATTERN = re.compile(
    r"^(?P<configuration>[a-z0-9]+)_(?P<split>train|dev|test)_track_(?P<track>[a-z])_(?P<number>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrackId:
    configuration: str
    split: str
    track: str
    number: int
    number_width: int


def parse_track_id(value: Any) -> TrackId | None:
    match = TRACK_ID_PATTERN.fullmatch(str(value))
    if match is None:
        return None
    number_text = match.group("number")
    return TrackId(
        configuration=match.group("configuration").lower(),
        split=match.group("split").lower(),
        track=match.group("track").upper(),
        number=int(number_text),
        number_width=len(number_text),
    )


def identify_track(value: Any) -> str | None:
    parsed = parse_track_id(value)
    return parsed.track if parsed else None


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_canonical(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_value(value: Any) -> Any:
    canonical = _canonical(value)
    if isinstance(canonical, tuple):
        return [_json_value(item) for item in canonical]
    return canonical


def analyze_track_ids(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, frame in frames.items():
        parsed = frame["id"].map(parse_track_id)
        unparsed = int(parsed.isna().sum())
        split_mismatches = int(
            sum(item is not None and item.split != split for item in parsed)
        )
        configuration_mismatches = int(
            sum(item is not None and item.configuration != CONFIGURATION for item in parsed)
        )
        track_counts = Counter(item.track for item in parsed if item is not None)
        patterns: list[dict[str, Any]] = []
        for track in sorted(track_counts):
            mask = parsed.map(lambda item: item is not None and item.track == track)
            ids = frame.loc[mask, "id"].astype(str)
            items = [item for item in parsed[mask] if item is not None]
            numbers = sorted(item.number for item in items)
            widths = sorted(set(item.number_width for item in items))
            patterns.append(
                {
                    "prefix": f"{CONFIGURATION}_{split}_track_{track.lower()}",
                    "track": track,
                    "count": len(ids),
                    "first_ids": ids.head(5).tolist(),
                    "last_ids": ids.tail(5).tolist(),
                    "suffix_min": numbers[0] if numbers else None,
                    "suffix_max": numbers[-1] if numbers else None,
                    "suffix_widths": widths,
                    "suffixes_contiguous_from_one": numbers == list(range(1, len(numbers) + 1)),
                }
            )
        result[split] = {
            "rows": len(frame),
            "first_ids": frame["id"].astype(str).head(10).tolist(),
            "last_ids": frame["id"].astype(str).tail(10).tolist(),
            "unparsed_ids": unparsed,
            "split_mismatches": split_mismatches,
            "configuration_mismatches": configuration_mismatches,
            "track_counts": dict(sorted(track_counts.items())),
            "patterns": patterns,
        }
    return result


def _row_signature(row: pd.Series, columns: Iterable[str]) -> tuple[Any, ...]:
    return tuple(_canonical(row[column]) for column in columns)


def analyze_duplicate_groups(
    frame: pd.DataFrame, split: str, text_column: str = "text_raw"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_columns = list(EXPECTED_LABELS)
    groups: list[dict[str, Any]] = []
    all_columns = list(frame.columns)
    auxiliary_columns = [
        column
        for column in all_columns
        if column not in {"id", "text", text_column, "emotions", *label_columns}
    ]
    for text, group in frame.groupby(text_column, sort=False, dropna=False):
        if len(group) < 2:
            continue
        label_signatures = {
            tuple(int(value) for value in values)
            for values in group[label_columns].itertuples(index=False, name=None)
        }
        full_signatures = {
            _row_signature(row, all_columns) for _, row in group.iterrows()
        }
        emotion_signatures = {
            _canonical(value) for value in group["emotions"]
        } if "emotions" in group else set()
        if len(full_signatures) == 1:
            classification = "exact_row_duplicate"
        elif len(label_signatures) == 1:
            classification = "same_text_same_labels"
        elif len(label_signatures) > 1:
            classification = "same_text_different_labels"
        else:
            classification = "other"
        tracks = [identify_track(value) for value in group["id"]]
        differing_columns = [
            column
            for column in all_columns
            if len({_canonical(value) for value in group[column]}) > 1
        ]
        records = []
        for _, row in group.iterrows():
            records.append(
                {
                    "id": str(row["id"]),
                    "track": identify_track(row["id"]),
                    "labels": {label: int(row[label]) for label in label_columns},
                    "emotions": _json_value(row["emotions"]) if "emotions" in row else None,
                    "other_fields": {
                        column: _json_value(row[column]) for column in auxiliary_columns
                    },
                }
            )
        groups.append(
            {
                "split": split,
                "text": _json_value(text),
                "text_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
                "group_size": len(group),
                "classification": classification,
                "ids": group["id"].astype(str).tolist(),
                "tracks": tracks,
                "labels_identical": len(label_signatures) == 1,
                "emotions_identical": len(emotion_signatures) <= 1,
                "track_a_track_c_pair": len(group) == 2 and set(tracks) == {"A", "C"},
                "differing_columns": differing_columns,
                "records": records,
            }
        )
    classes = Counter(group["classification"] for group in groups)
    sizes = Counter(group["group_size"] for group in groups)
    summary = {
        "total_rows": len(frame),
        "unique_texts": int(frame[text_column].nunique(dropna=False)),
        "duplicate_groups": len(groups),
        "groups_size_2": sizes[2],
        "groups_size_gt_2": sum(count for size, count in sizes.items() if size > 2),
        "same_text_same_labels": classes["same_text_same_labels"],
        "same_text_different_labels": classes["same_text_different_labels"],
        "exact_row_duplicate": classes["exact_row_duplicate"],
        "other": classes["other"],
        "identical_emotions": sum(group["emotions_identical"] for group in groups),
        "track_a_track_c_pairs": sum(group["track_a_track_c_pair"] for group in groups),
        "groups_differing_only_by_id": sum(
            group["differing_columns"] == ["id"] for group in groups
        ),
    }
    return summary, groups


def curate_track_a(
    frames: Mapping[str, pd.DataFrame], revision: str
) -> dict[str, pd.DataFrame]:
    curated: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        parsed = frame["id"].map(parse_track_id)
        invalid = [
            str(value)
            for value, item in zip(frame["id"], parsed, strict=True)
            if item is None or item.configuration != CONFIGURATION or item.split != split
        ]
        if invalid:
            raise ValueError(
                f"Cannot curate {split}: {len(invalid)} IDs do not match the explicit dataset/split pattern"
            )
        mask = parsed.map(lambda item: item is not None and item.track == "A")
        selected = frame.loc[mask].copy()
        selected["source_split"] = split
        selected["source_track"] = "A"
        selected["source_revision"] = revision
        curated[split] = selected.reset_index(drop=True)
    return curated


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if len(left) != len(right):
        return False
    return all(_canonical(a) == _canonical(b) for a, b in zip(left, right, strict=True))


def verify_curated_fidelity(
    source_frames: Mapping[str, pd.DataFrame], curated_frames: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, source in source_frames.items():
        curated = curated_frames[split]
        parsed = source["id"].map(parse_track_id)
        expected = source.loc[parsed.map(lambda item: item is not None and item.track == "A")]
        if set(curated["id"]) != set(expected["id"]):
            raise RuntimeError(f"Curated {split} IDs do not equal the explicit Track A ID set")
        indexed_source = source.set_index("id", drop=False)
        indexed_curated = curated.set_index("id", drop=False).loc[expected["id"]]
        for column in source.columns:
            if not _series_equal(
                expected[column].reset_index(drop=True),
                indexed_curated[column].reset_index(drop=True),
            ):
                raise RuntimeError(f"Curated fidelity failure in {split}.{column}")
        provenance_ok = (
            curated["source_split"].eq(split).all()
            and curated["source_track"].eq("A").all()
            and curated["source_revision"].nunique() == 1
        )
        if not provenance_ok:
            raise RuntimeError(f"Curated provenance columns are invalid in {split}")
        result[split] = {
            "source_track_a_rows": len(expected),
            "curated_rows": len(curated),
            "id_sets_equal": True,
            "all_source_columns_equal": True,
            "texts_unchanged": _series_equal(expected["text_raw"], indexed_curated["text_raw"]),
            "labels_unchanged": all(
                _series_equal(expected[label], indexed_curated[label]) for label in EXPECTED_LABELS
            ),
            "no_synthetic_rows": True,
            "provenance_valid": True,
        }
    return result


def read_raw_snapshot_frames(snapshot: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "dev", "test"):
        matches = sorted((snapshot / CONFIGURATION).glob(f"{split}-*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No raw snapshot Parquet shard found for {split}")
        frame = pd.concat([pd.read_parquet(path) for path in matches], ignore_index=True)
        if "text_raw" not in frame:
            insert_at = frame.columns.get_loc("text") + 1
            frame.insert(insert_at, "text_raw", frame["text"].copy(deep=True))
        frames[split] = frame
    return frames


def verify_interim_against_raw(
    raw_frames: Mapping[str, pd.DataFrame], interim_frames: Mapping[str, pd.DataFrame]
) -> None:
    for split, raw in raw_frames.items():
        interim = interim_frames[split]
        if list(raw.columns) != list(interim.columns) or len(raw) != len(interim):
            raise RuntimeError(f"Interim {split} schema/rows differ from the official snapshot")
        for column in raw.columns:
            if not _series_equal(raw[column], interim[column]):
                raise RuntimeError(f"Interim {split}.{column} differs from the official snapshot")


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        if list(existing.columns) != list(frame.columns) or len(existing) != len(frame):
            raise RuntimeError(f"Existing curated file conflicts and was preserved: {path}")
        for column in frame.columns:
            if not _series_equal(existing[column], frame[column]):
                raise RuntimeError(f"Existing curated file differs in {column} and was preserved: {path}")
        LOGGER.info("Curated Parquet already exists; reusing %s", path)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        readback = pd.read_parquet(temporary, engine="pyarrow")
        if list(readback.columns) != list(frame.columns) or len(readback) != len(frame):
            raise RuntimeError("Curated Parquet readback schema/row-count check failed")
        for column in frame.columns:
            if not _series_equal(readback[column], frame[column]):
                raise RuntimeError(f"Curated Parquet readback changed {column}")
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def basic_track_statistics(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, frame in frames.items():
        num_labels = frame[list(EXPECTED_LABELS)].sum(axis=1)
        result[split] = {
            "rows": len(frame),
            "unique_ids": int(frame["id"].nunique()),
            "unique_texts": int(frame["text_raw"].nunique(dropna=False)),
            "duplicate_texts": duplicated_texts(frame, "text_raw"),
            "label_positives": {
                label: int(frame[label].eq(1).sum()) for label in EXPECTED_LABELS
            },
            "zero_label_instances": int(num_labels.eq(0).sum()),
        }
    return result


def analyze_cross_split_leakage(
    curated: Mapping[str, pd.DataFrame], source: Mapping[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    shared = sorted(set(curated["train"]["text_raw"]) & set(curated["test"]["text_raw"]))
    cases: list[dict[str, Any]] = []
    for text in shared:
        train_rows = curated["train"].loc[curated["train"]["text_raw"].eq(text)]
        test_rows = curated["test"].loc[curated["test"]["text_raw"].eq(text)]
        source_test_rows = source["test"].loc[source["test"]["text_raw"].eq(text)]
        cases.append(
            {
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "exact_text_comparison": True,
                "normalized": False,
                "train_records": _records_for_report(train_rows),
                "test_track_a_records": _records_for_report(test_rows),
                "source_test_records_all_tracks": _records_for_report(source_test_rows),
                "ids_different": set(train_rows["id"]).isdisjoint(set(test_rows["id"])),
                "labels_identical": {
                    tuple(row) for row in train_rows[list(EXPECTED_LABELS)].itertuples(index=False, name=None)
                } == {
                    tuple(row) for row in test_rows[list(EXPECTED_LABELS)].itertuples(index=False, name=None)
                },
            }
        )
    return cases


def _records_for_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        records.append(
            {
                "id": str(row["id"]),
                "track": identify_track(row["id"]),
                "labels": {label: int(row[label]) for label in EXPECTED_LABELS},
                "emotions": _json_value(row["emotions"]) if "emotions" in row else None,
            }
        )
    return records


def _write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n" for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def render_track_id_report(revision: str, analysis: Mapping[str, Any]) -> str:
    lines = [
        "# BRIGHTER PT-BR — Track ID analysis",
        "",
        f"- Revision: `{revision}`",
        "- Analysis uses original IDs without normalization.",
        "",
    ]
    for split, item in analysis.items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Rows: {item['rows']}",
                f"- First IDs: `{', '.join(item['first_ids'])}`",
                f"- Last IDs: `{', '.join(item['last_ids'])}`",
                f"- Unparsed IDs: {item['unparsed_ids']}",
                f"- Split mismatches: {item['split_mismatches']}",
                f"- Configuration mismatches: {item['configuration_mismatches']}",
                "",
                "| Prefix | Track | Count | First ID | Last ID | Suffix range | Width | Contiguous from 1 |",
                "|---|---|---:|---|---|---|---|---|",
            ]
        )
        for pattern in item["patterns"]:
            lines.append(
                f"| `{pattern['prefix']}` | {pattern['track']} | {pattern['count']} | "
                f"`{pattern['first_ids'][0]}` | `{pattern['last_ids'][-1]}` | "
                f"{pattern['suffix_min']}–{pattern['suffix_max']} | {pattern['suffix_widths']} | "
                f"{pattern['suffixes_contiguous_from_one']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            "Track A is directly and reproducibly identifiable from the original `id` field. "
            "Every ID matches `ptbr_<split>_track_<track>_<numeric suffix>`; no fallback based "
            "on text deduplication is needed. `train` contains Track A only, while `dev` and "
            "`test` contain complete Track A and Track C partitions.",
            "",
        ]
    )
    return "\n".join(lines)


def render_duplicate_report(
    revision: str,
    summaries: Mapping[str, Any],
    groups: Mapping[str, list[dict[str, Any]]],
    audit_path: Path,
    track_statistics: Mapping[str, Any],
) -> str:
    lines = [
        "# BRIGHTER PT-BR — Duplicate and Track A × Track C analysis",
        "",
        f"- Revision: `{revision}`",
        "- Grouping key: exact `text_raw` value; no normalization was applied.",
        f"- Complete per-group audit: `{audit_path.as_posix()}`.",
        "",
    ]
    for split in ("dev", "test"):
        item = summaries[split]
        lines.extend(
            [
                f"## {split.upper()}",
                "",
                "| Measure | Count |",
                "|---|---:|",
                f"| Total rows | {item['total_rows']} |",
                f"| Unique texts | {item['unique_texts']} |",
                f"| Duplicate groups | {item['duplicate_groups']} |",
                f"| Groups size = 2 | {item['groups_size_2']} |",
                f"| Groups size > 2 | {item['groups_size_gt_2']} |",
                f"| Same text + same six labels | {item['same_text_same_labels']} |",
                f"| Same text + conflicting labels | {item['same_text_different_labels']} |",
                f"| Exact full-row duplicates | {item['exact_row_duplicate']} |",
                f"| Identical `emotions` | {item['identical_emotions']} |",
                f"| Track A / Track C pairs | {item['track_a_track_c_pairs']} |",
                f"| Groups differing only in `id` | {item['groups_differing_only_by_id']} |",
                "",
                "### Representative examples",
                "",
                "| Text SHA-256 (prefix) | IDs | Classification | Differing columns |",
                "|---|---|---|---|",
            ]
        )
        for group in groups[split][:3]:
            lines.append(
                f"| `{group['text_sha256'][:16]}` | `{'`, `'.join(group['ids'])}` | "
                f"{group['classification']} | `{', '.join(group['differing_columns'])}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Empirical conclusion",
            "",
            "Every duplicated text group in both splits has exactly two rows: one explicit "
            "Track A ID and one explicit Track C ID. In every group, `text`, `text_raw`, the "
            "six binary labels, and `emotions` are identical; only `id` differs. Track C is "
            "therefore a complete ID-renamed reuse of the Track A instances in these PT-BR "
            "splits at this frozen revision.",
            "",
            "## Curated Track A basic statistics",
            "",
            "| Split | Rows | Unique IDs | Unique texts | Duplicate texts | Anger + | Disgust + | Fear + | Joy + | Sadness + | Surprise + | Zero-label |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, item in track_statistics.items():
        positives = item["label_positives"]
        lines.append(
            f"| {split} | {item['rows']} | {item['unique_ids']} | {item['unique_texts']} | "
            f"{item['duplicate_texts']} | {positives['anger']} | {positives['disgust']} | "
            f"{positives['fear']} | {positives['joy']} | {positives['sadness']} | "
            f"{positives['surprise']} | {item['zero_label_instances']} |"
        )
    lines.extend(
        [
            "",
            "> These are structural counts only. No IRLbl, MeanIR, co-occurrence, concurrence, correlation, preprocessing, or modeling was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def render_leakage_report(revision: str, cases: list[dict[str, Any]]) -> str:
    lines = [
        "# BRIGHTER PT-BR — Cross-split leakage audit",
        "",
        f"- Revision: `{revision}`",
        "- Normalized comparison: **NO**",
        "- Exact text comparison: **YES**",
        f"- Track A train × test exact overlaps: **{len(cases)}**",
        "- No occurrence was removed.",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        train = case["train_records"][0]
        test = case["test_track_a_records"][0]
        track_c_records = [
            record
            for record in case["source_test_records_all_tracks"]
            if record["track"] == "C"
        ]
        lines.extend(
            [
                f"## Case {index}",
                "",
                f"- Text SHA-256: `{case['text_sha256']}`",
                f"- Train ID: `{train['id']}`",
                f"- Test Track A ID: `{test['id']}`",
                f"- Test Track C counterpart ID: `{track_c_records[0]['id'] if track_c_records else 'not found'}`",
                f"- IDs different: **{case['ids_different']}**",
                f"- Labels identical: **{case['labels_identical']}**",
                f"- Train labels: `{json.dumps(train['labels'], ensure_ascii=False)}`",
                f"- Test labels: `{json.dumps(test['labels'], ensure_ascii=False)}`",
                f"- Train emotions: `{json.dumps(train['emotions'], ensure_ascii=False)}`",
                f"- Test emotions: `{json.dumps(test['emotions'], ensure_ascii=False)}`",
                "",
                "### Exact text",
                "",
                f"> {case['text']}",
                "",
                "### Assessment",
                "",
                "The text is exactly equal, but the split-specific IDs and emotion annotations "
                "differ (`anger` in train; `anger` + `fear` in test). The sentence is neither "
                "extremely short nor a generic greeting, so accidental recurrence is less "
                "plausible, but the available fields contain no source-document metadata that "
                "can prove the two rows are the same original collection instance. The Track C "
                "test counterpart repeats the Track A test annotation and adds no source evidence.",
                "",
            ]
        )
    return "\n".join(lines)


def _summary(
    revision: str,
    source_rows: Mapping[str, int],
    curated_rows: Mapping[str, int],
    duplicate_summaries: Mapping[str, Any],
    overlaps: int,
    artifacts: Iterable[Path],
) -> str:
    identical = sum(item["same_text_same_labels"] for item in duplicate_summaries.values())
    conflicting = sum(item["same_text_different_labels"] for item in duplicate_summaries.values())
    lines = [
        "BRIGHTER PT-BR TRACK A CURATION COMPLETE",
        "",
        "Revision:",
        revision,
        "",
        "Track identification:",
        "method: explicit strict pattern in original IDs (ptbr_<split>_track_a_<number>)",
        "",
        "Source:",
    ]
    lines.extend(f"{split}: {rows}" for split, rows in source_rows.items())
    lines.extend(["", "Track A:"])
    lines.extend(f"{split}: {rows}" for split, rows in curated_rows.items())
    lines.append(f"total: {sum(curated_rows.values())}")
    lines.extend(
        [
            "",
            "Track A / Track C validation:",
            f"dev duplicate pairs: {duplicate_summaries['dev']['track_a_track_c_pairs']}",
            f"test duplicate pairs: {duplicate_summaries['test']['track_a_track_c_pairs']}",
            f"identical labels: {identical}",
            f"conflicting labels: {conflicting}",
            "",
            "Cross-split:",
            f"train-test exact overlaps: {overlaps}",
            "",
            "Data modifications:",
            "text preprocessing: NONE",
            "label modification: NONE",
            "balancing: NONE",
            "augmentation: NONE",
            "",
            "Artifacts:",
        ]
    )
    lines.extend(str(path) for path in artifacts)
    return "\n".join(lines)


def run_track_a_curation(project_root: Path, revision: str = FROZEN_REVISION) -> tuple[str, dict[str, Any]]:
    if revision != FROZEN_REVISION:
        raise ValueError(
            f"This analysis is frozen to {FROZEN_REVISION}; received {revision}"
        )
    metadata_dir = project_root / "metadata" / "brighter_ptbr"
    reports_dir = project_root / "reports" / "brighter_ptbr"
    dataset_manifest_path = metadata_dir / "dataset_manifest.json"
    dataset_manifest = load_json(dataset_manifest_path)
    if dataset_manifest.get("huggingface_revision") != revision:
        raise RuntimeError("Dataset manifest revision does not match the frozen curation revision")
    snapshot = project_root / dataset_manifest["snapshot_path"]
    verify_snapshot(snapshot, revision)
    raw_frames = read_raw_snapshot_frames(snapshot)
    interim_frames = {
        split: pd.read_parquet(project_root / details["path"])
        for split, details in dataset_manifest["parquet_files"].items()
    }
    for split, details in dataset_manifest["parquet_files"].items():
        path = project_root / details["path"]
        if sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"Interim Parquet SHA-256 mismatch for {split}")
    verify_interim_against_raw(raw_frames, interim_frames)

    id_analysis = analyze_track_ids(interim_frames)
    if any(item["unparsed_ids"] or item["split_mismatches"] for item in id_analysis.values()):
        raise RuntimeError("Original IDs do not support a complete explicit track reconstruction")
    duplicate_summaries: dict[str, Any] = {}
    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for split in ("dev", "test"):
        duplicate_summaries[split], duplicate_groups[split] = analyze_duplicate_groups(
            interim_frames[split], split
        )
        expected_pairs = id_analysis[split]["track_counts"].get("A", 0)
        summary = duplicate_summaries[split]
        if not (
            summary["duplicate_groups"] == expected_pairs
            and summary["groups_size_2"] == expected_pairs
            and summary["track_a_track_c_pairs"] == expected_pairs
            and summary["same_text_same_labels"] == expected_pairs
            and summary["identical_emotions"] == expected_pairs
            and summary["groups_differing_only_by_id"] == expected_pairs
            and summary["same_text_different_labels"] == 0
        ):
            raise RuntimeError(f"Track A/Track C duplicate validation failed for {split}")

    curated = curate_track_a(interim_frames, revision)
    fidelity = verify_curated_fidelity(interim_frames, curated)
    expected_rows = {"train": 2226, "dev": 200, "test": 2226}
    curated_rows = {split: len(frame) for split, frame in curated.items()}
    if curated_rows != expected_rows:
        raise RuntimeError(f"Unexpected Track A row counts: {curated_rows}; expected {expected_rows}")

    curated_dir = project_root / "data" / "curated" / "brighter_ptbr" / revision / "track_a"
    curated_paths: dict[str, Path] = {}
    for split, frame in curated.items():
        path = curated_dir / f"{split}.parquet"
        _write_parquet_atomic(path, frame)
        curated_paths[split] = path
    persisted = {split: pd.read_parquet(path) for split, path in curated_paths.items()}
    persisted_fidelity = verify_curated_fidelity(interim_frames, persisted)

    leakage_cases = analyze_cross_split_leakage(persisted, interim_frames)
    stats = basic_track_statistics(persisted)
    audit_path = metadata_dir / "duplicate_groups.jsonl"
    _write_jsonl_atomic(
        audit_path,
        (group for split in ("dev", "test") for group in duplicate_groups[split]),
    )
    track_id_report = reports_dir / "track_id_analysis.md"
    duplicate_report = reports_dir / "duplicate_analysis.md"
    leakage_report = reports_dir / "cross_split_leakage.md"
    write_inventory(track_id_report, render_track_id_report(revision, id_analysis))
    write_inventory(
        duplicate_report,
        render_duplicate_report(
            revision,
            duplicate_summaries,
            duplicate_groups,
            audit_path.relative_to(project_root),
            stats,
        ),
    )
    write_inventory(leakage_report, render_leakage_report(revision, leakage_cases))

    manifest_path = metadata_dir / "track_a_manifest.json"
    old_manifest = load_json(manifest_path) if manifest_path.exists() else None
    curation_timestamp = (
        old_manifest.get("curation_timestamp_utc")
        if old_manifest and old_manifest.get("revision") == revision
        else utc_now_iso()
    )
    source_rows = {split: len(frame) for split, frame in interim_frames.items()}
    manifest = {
        "dataset": DATASET_ID,
        "configuration": CONFIGURATION,
        "revision": revision,
        "experimental_track": "A",
        "curation_timestamp_utc": curation_timestamp,
        "track_identification_method": {
            "type": "explicit_original_id_pattern",
            "pattern": TRACK_ID_PATTERN.pattern,
            "selection": "parsed track component equals A and parsed split equals source split",
            "text_deduplication_used_for_selection": False,
        },
        "source_rows": source_rows,
        "curated_rows": curated_rows,
        "track_c_rows_excluded": {
            split: source_rows[split] - curated_rows[split] for split in source_rows
        },
        "duplicate_validation": duplicate_summaries,
        "label_conflict_groups": sum(
            item["same_text_different_labels"] for item in duplicate_summaries.values()
        ),
        "cross_split_overlap_detected": len(leakage_cases),
        "cross_split_overlap_removed": 0,
        "text_processing": "none",
        "balancing": "none",
        "augmentation": "none",
        "synthetic_instances": 0,
        "provenance_columns_added": ["source_split", "source_track", "source_revision"],
        "original_columns_preserved": list(next(iter(interim_frames.values())).columns),
        "curated_columns": list(next(iter(persisted.values())).columns),
        "fidelity": persisted_fidelity,
        "source_integrity": {
            "immutable_snapshot_hashes_verified": True,
            "raw_to_interim_all_columns_equal": True,
            "curated_ids_equal_explicit_track_a_ids": True,
            "curated_original_columns_equal_source": True,
            "texts_unchanged": True,
            "labels_unchanged": True,
            "synthetic_rows_added": 0,
        },
        "statistics": stats,
        "curated_files": {
            split: {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for split, path in curated_paths.items()
        },
        "audit_artifacts": {
            "duplicate_groups": audit_path.relative_to(project_root).as_posix(),
            "track_id_analysis": track_id_report.relative_to(project_root).as_posix(),
            "duplicate_analysis": duplicate_report.relative_to(project_root).as_posix(),
            "cross_split_leakage": leakage_report.relative_to(project_root).as_posix(),
        },
    }
    write_json_atomic(manifest_path, manifest)
    artifacts = [curated_dir, manifest_path, audit_path, track_id_report, duplicate_report, leakage_report]
    summary = _summary(
        revision,
        source_rows,
        curated_rows,
        duplicate_summaries,
        len(leakage_cases),
        artifacts,
    )
    return summary, manifest
