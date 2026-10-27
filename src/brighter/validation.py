"""Structural, label, text, and split-integrity checks."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .manifest import write_json_atomic

EXPECTED_LABELS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")


class SchemaError(ValueError):
    """Raised when a required dataset role or label cannot be identified."""


def identify_label_columns(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    missing = [label for label in EXPECTED_LABELS if label not in available]
    if missing:
        raise SchemaError(f"Missing required emotion label columns: {', '.join(missing)}")
    return list(EXPECTED_LABELS)


def infer_id_column(frame: pd.DataFrame) -> str:
    for name in ("id", "ID", "uid", "uuid", "identifier"):
        if name in frame.columns:
            return name
    candidates = [name for name in frame.columns if name.lower().endswith("id")]
    if len(candidates) == 1:
        return candidates[0]
    raise SchemaError("Could not identify an ID column from the real dataset schema")


def infer_text_column(frame: pd.DataFrame) -> str:
    for name in ("text", "text_raw", "sentence", "content", "tweet"):
        if name in frame.columns and (pd.api.types.is_string_dtype(frame[name]) or frame[name].dtype == object):
            return name
    string_columns = [
        name
        for name in frame.columns
        if pd.api.types.is_string_dtype(frame[name]) or frame[name].dtype == object
    ]
    string_columns = [name for name in string_columns if name.lower() not in {"id", "emotions"}]
    if len(string_columns) == 1:
        return string_columns[0]
    raise SchemaError("Could not identify a text column from the real dataset schema")


def invalid_binary_values(series: pd.Series) -> list[Any]:
    non_null = series.dropna()
    invalid = non_null[~non_null.isin([0, 1, False, True])]
    values: list[Any] = []
    for value in invalid.unique().tolist():
        if isinstance(value, np.generic):
            value = value.item()
        values.append(value)
    return values


def validate_binary_labels(frame: pd.DataFrame, labels: Iterable[str] = EXPECTED_LABELS) -> dict[str, dict[str, Any]]:
    label_columns = identify_label_columns(frame.columns)
    requested = list(labels)
    if set(requested) != set(label_columns):
        missing = [label for label in requested if label not in frame.columns]
        if missing:
            raise SchemaError(f"Missing required emotion label columns: {', '.join(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for label in requested:
        series = frame[label]
        nulls = int(series.isna().sum())
        invalid = invalid_binary_values(series)
        positive = int(series.eq(1).sum())
        negative = int(series.eq(0).sum())
        result[label] = {
            "dtype": str(series.dtype),
            "positive": positive,
            "negative": negative,
            "positive_proportion": positive / len(frame) if len(frame) else 0.0,
            "null": nulls,
            "invalid_values": invalid,
            "invalid_count": int((~series.isna() & ~series.isin([0, 1, False, True])).sum()),
        }
    return result


def duplicated_ids(frame: pd.DataFrame, id_column: str) -> int:
    return int(frame.loc[frame[id_column].notna(), id_column].duplicated(keep="first").sum())


def duplicated_texts(frame: pd.DataFrame, text_column: str) -> int:
    return int(frame.loc[frame[text_column].notna(), text_column].duplicated(keep="first").sum())


def calculate_num_labels(frame: pd.DataFrame, labels: Iterable[str] = EXPECTED_LABELS) -> pd.Series:
    label_columns = list(labels)
    metrics = validate_binary_labels(frame, label_columns)
    problems = [name for name, values in metrics.items() if values["null"] or values["invalid_count"]]
    if problems:
        raise ValueError(f"Cannot calculate num_labels with null/invalid values in: {', '.join(problems)}")
    return frame[label_columns].astype("int64").sum(axis=1)


def label_cardinality(frame: pd.DataFrame, labels: Iterable[str] = EXPECTED_LABELS) -> float:
    if len(frame) == 0:
        return 0.0
    return float(calculate_num_labels(frame, labels).sum() / len(frame))


def cross_split_overlaps(
    frames: Mapping[str, pd.DataFrame], id_column: str, text_column: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for left, right in combinations(frames, 2):
        left_ids = set(frames[left][id_column].dropna().tolist())
        right_ids = set(frames[right][id_column].dropna().tolist())
        left_texts = set(frames[left][text_column].dropna().tolist())
        right_texts = set(frames[right][text_column].dropna().tolist())
        result[f"{left}__{right}"] = {
            "shared_ids": len(left_ids & right_ids),
            "shared_texts": len(left_texts & right_texts),
        }
    return result


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": int(values.max()),
    }


def analyze_dataset(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    if not frames:
        raise SchemaError("No dataset splits were found")
    first = next(iter(frames.values()))
    labels = identify_label_columns(first.columns)
    id_column = infer_id_column(first)
    text_column = "text_raw" if "text_raw" in first.columns else infer_text_column(first)
    analyses: dict[str, Any] = {}
    warnings: list[str] = []
    errors: list[str] = []
    for split, frame in frames.items():
        identify_label_columns(frame.columns)
        if id_column not in frame or text_column not in frame:
            raise SchemaError(f"Split {split!r} does not contain the inferred ID/text columns")
        label_metrics = validate_binary_labels(frame, labels)
        null_text = int(frame[text_column].isna().sum())
        string_text = frame[text_column].dropna().astype(str)
        empty = int(string_text.eq("").sum())
        whitespace = int(string_text.str.fullmatch(r"\s+").sum())
        dup_ids = duplicated_ids(frame, id_column)
        dup_text = duplicated_texts(frame, text_column)
        valid_label_frame = pd.DataFrame({label: frame[label].eq(1).astype("int8") for label in labels})
        counts = valid_label_frame.sum(axis=1)
        invalid_total = sum(metric["invalid_count"] for metric in label_metrics.values())
        label_null_total = sum(metric["null"] for metric in label_metrics.values())
        analyses[split] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
            "missing_ids": int(frame[id_column].isna().sum()),
            "duplicated_ids": dup_ids,
            "duplicated_texts": dup_text,
            "null_texts": null_text,
            "empty_texts": empty,
            "whitespace_only_texts": whitespace,
            "character_length": _distribution(string_text.str.len()),
            "word_length": _distribution(string_text.str.findall(r"\S+").str.len()),
            "labels": label_metrics,
            "num_labels": {str(number): int(counts.eq(number).sum()) for number in range(7)},
            "label_cardinality": float(counts.sum() / len(frame)) if len(frame) else 0.0,
        }
        if dup_ids:
            warnings.append(f"{split}: {dup_ids} duplicated ID rows beyond the first occurrence")
        if analyses[split]["missing_ids"]:
            warnings.append(f"{split}: {analyses[split]['missing_ids']} missing IDs")
        if dup_text:
            warnings.append(f"{split}: {dup_text} exact duplicated text rows beyond the first occurrence")
        if null_text or empty or whitespace:
            warnings.append(f"{split}: null={null_text}, empty={empty}, whitespace-only={whitespace} texts")
        if invalid_total or label_null_total:
            errors.append(f"{split}: invalid labels={invalid_total}, null labels={label_null_total}")
        for label, metric in label_metrics.items():
            if not (pd.api.types.is_integer_dtype(frame[label]) or pd.api.types.is_bool_dtype(frame[label])):
                warnings.append(f"{split}: label {label} has non-integer dtype {metric['dtype']}")
    overlaps = cross_split_overlaps(frames, id_column, text_column)
    for pair, values in overlaps.items():
        if values["shared_ids"] or values["shared_texts"]:
            warnings.append(
                f"{pair}: shared IDs={values['shared_ids']}, shared exact texts={values['shared_texts']}"
            )
    return {
        "id_column": id_column,
        "text_column": text_column,
        "label_columns": labels,
        "splits": analyses,
        "cross_split_overlaps": overlaps,
        "warnings": warnings,
        "errors": errors,
        "valid": not errors,
    }


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    return str(value)


def build_schema_document(
    frames: Mapping[str, pd.DataFrame], *, source_columns: list[str], roles: Mapping[str, str]
) -> dict[str, Any]:
    first = next(iter(frames.values()))
    columns: list[dict[str, Any]] = []
    for name in first.columns:
        example = None
        for frame in frames.values():
            non_null = frame[name].dropna()
            if not non_null.empty:
                example = _json_safe(non_null.iloc[0])
                break
        missing_by_split = {split: int(frame[name].isna().sum()) for split, frame in frames.items()}
        columns.append(
            {
                "name": name,
                "dtype": str(first[name].dtype),
                "missing_values": sum(missing_by_split.values()),
                "missing_values_by_split": missing_by_split,
                "example": example,
                "inferred_role": roles.get(name, "metadata"),
                "source_column": name in source_columns,
            }
        )
    return {
        "source_columns": source_columns,
        "local_columns": list(first.columns),
        "columns": columns,
    }


def write_schema(path: Path, document: Mapping[str, Any]) -> None:
    write_json_atomic(path, document)


def render_inventory(manifest: Mapping[str, Any], analysis: Mapping[str, Any]) -> str:
    split_names = list(analysis["splits"])
    lines = [
        "# BRIGHTER PT-BR — Data inventory",
        "",
        "## Dataset",
        "",
        f"- Dataset: `{manifest['dataset']}`",
        f"- Configuration: `{manifest['configuration']}`",
        f"- Revision SHA: `{manifest['huggingface_revision']}`",
        f"- Download timestamp (UTC): `{manifest['download_timestamp_utc']}`",
        "- Processing: `none`",
        "",
        "## Splits",
        "",
        "| Split | Number of instances |",
        "|---|---:|",
    ]
    lines.extend(f"| {split} | {analysis['splits'][split]['rows']} |" for split in split_names)
    lines.extend(["", "## Columns", "", "| Split | Column | Type | Missing values |", "|---|---|---|---:|"])
    for split in split_names:
        item = analysis["splits"][split]
        for column in item["columns"]:
            lines.append(f"| {split} | `{column}` | `{item['dtypes'][column]}` | {item['missing_values'][column]} |")
    lines.extend(["", "## Basic label distribution", ""])
    for split in split_names:
        lines.extend([f"### {split}", "", "| Emotion | Positive | Negative | Positive % |", "|---|---:|---:|---:|"])
        for label in analysis["label_columns"]:
            metric = analysis["splits"][split]["labels"][label]
            lines.append(
                f"| {label} | {metric['positive']} | {metric['negative']} | {metric['positive_proportion'] * 100:.2f}% |"
            )
        lines.append("")
    lines.extend(["## Number of emotions per text", "", "| Num. labels | " + " | ".join(split_names) + " |", "|---:" + "|---:" * len(split_names) + "|"])
    for number in range(7):
        lines.append(
            f"| {number} | "
            + " | ".join(str(analysis["splits"][split]["num_labels"][str(number)]) for split in split_names)
            + " |"
        )
    lines.extend(["", "### Label cardinality", "", "| Split | Cardinality |", "|---|---:|"])
    lines.extend(
        f"| {split} | {analysis['splits'][split]['label_cardinality']:.6f} |" for split in split_names
    )
    lines.extend(
        [
            "",
            "## Text length distributions",
            "",
            "| Split | Unit | Min | Mean | Median | P95 | Max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in split_names:
        for key, unit in (("character_length", "characters"), ("word_length", "words")):
            metric = analysis["splits"][split][key]
            lines.append(
                f"| {split} | {unit} | {metric['min']} | {metric['mean']:.2f} | "
                f"{metric['median']:.2f} | {metric['p95']:.2f} | {metric['max']} |"
            )
    lines.extend(["", "## Integrity", ""])
    for split in split_names:
        item = analysis["splits"][split]
        invalid = sum(label["invalid_count"] + label["null"] for label in item["labels"].values())
        lines.extend(
            [
                f"### {split}",
                "",
                f"- Missing IDs: {item['missing_ids']}",
                f"- Duplicated ID rows beyond the first occurrence: {item['duplicated_ids']}",
                f"- Exact duplicated text rows beyond the first occurrence: {item['duplicated_texts']}",
                f"- Null texts: {item['null_texts']}",
                f"- Empty texts: {item['empty_texts']}",
                f"- Whitespace-only texts: {item['whitespace_only_texts']}",
                f"- Invalid or null label values: {invalid}",
                "",
            ]
        )
    lines.extend(["### Cross-split overlap", "", "| Pair | Shared IDs | Shared exact texts |", "|---|---:|---:|"])
    for pair, values in analysis["cross_split_overlaps"].items():
        lines.append(f"| {pair.replace('__', ' × ')} | {values['shared_ids']} | {values['shared_texts']} |")
    lines.extend(["", "### Warnings and errors", ""])
    messages = list(analysis["warnings"]) + list(analysis["errors"])
    lines.extend(f"- {message}" for message in messages)
    if not messages:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "> Factual inventory only. No preprocessing, balancing, augmentation, removal, or automatic correction was applied.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def read_parquet_splits(directory: Path) -> dict[str, pd.DataFrame]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet splits found in {directory}")
    return {path.stem: pd.read_parquet(path) for path in files}


def write_analysis_json(path: Path, analysis: Mapping[str, Any]) -> None:
    write_json_atomic(path, json.loads(json.dumps(analysis, ensure_ascii=False)))
