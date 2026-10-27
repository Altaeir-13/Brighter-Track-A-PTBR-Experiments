"""Deterministic multilabel EDA for the frozen BRIGHTER PT-BR Track A."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .curation import FROZEN_REVISION
from .download import CONFIGURATION, DATASET_ID
from .manifest import load_json, sha256_file, write_json_atomic
from .validation import EXPECTED_LABELS

LOGGER = logging.getLogger(__name__)
TOKENIZER_MODEL = "neuralmind/bert-base-portuguese-cased"
N_LABELS = len(EXPECTED_LABELS)


def _matrix(data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS) -> np.ndarray:
    values = data[list(labels)].to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError(f"Expected a two-dimensional binary matrix with {len(labels)} labels")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Label matrix contains values outside {0, 1}")
    return values.astype(np.int64, copy=False)


def calculate_num_labels(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> np.ndarray:
    return _matrix(data, labels).sum(axis=1)


def label_cardinality(
    data: pd.DataFrame | np.ndarray,
    labels: Sequence[str] = EXPECTED_LABELS,
    *,
    emotional_only: bool = False,
) -> float:
    counts = calculate_num_labels(data, labels)
    if emotional_only:
        counts = counts[counts > 0]
    return float(counts.mean()) if len(counts) else 0.0


def label_density(
    data: pd.DataFrame | np.ndarray,
    labels: Sequence[str] = EXPECTED_LABELS,
    *,
    emotional_only: bool = False,
) -> float:
    return label_cardinality(data, labels, emotional_only=emotional_only) / len(labels)


def labelset_strings(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> pd.Series:
    values = _matrix(data, labels)
    result = [
        "+".join(label for label, active in zip(labels, row, strict=True) if active) or "none"
        for row in values
    ]
    return pd.Series(result, dtype="string")


def cooccurrence_matrix(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> pd.DataFrame:
    values = _matrix(data, labels)
    return pd.DataFrame(values.T @ values, index=labels, columns=labels, dtype="int64")


def conditional_probability_matrix(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> pd.DataFrame:
    cooccurrence = cooccurrence_matrix(data, labels).to_numpy(dtype=float)
    denominators = np.diag(cooccurrence)[:, None]
    result = np.divide(
        cooccurrence,
        denominators,
        out=np.full_like(cooccurrence, np.nan),
        where=denominators != 0,
    )
    return pd.DataFrame(result, index=labels, columns=labels)


def jaccard_label_matrix(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> pd.DataFrame:
    cooccurrence = cooccurrence_matrix(data, labels).to_numpy(dtype=float)
    positives = np.diag(cooccurrence)
    union = positives[:, None] + positives[None, :] - cooccurrence
    result = np.divide(
        cooccurrence,
        union,
        out=np.full_like(cooccurrence, np.nan),
        where=union != 0,
    )
    return pd.DataFrame(result, index=labels, columns=labels)


def lift_matrix(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> pd.DataFrame:
    values = _matrix(data, labels)
    n_rows = len(values)
    if n_rows == 0:
        return pd.DataFrame(np.nan, index=labels, columns=labels)
    joint = (values.T @ values) / n_rows
    prevalence = values.mean(axis=0)
    independent = prevalence[:, None] * prevalence[None, :]
    result = np.divide(
        joint,
        independent,
        out=np.full_like(joint, np.nan, dtype=float),
        where=independent != 0,
    )
    return pd.DataFrame(result, index=labels, columns=labels)


def imbalance_ratios(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> dict[str, float]:
    values = _matrix(data, labels)
    positives = values.sum(axis=0)
    if np.any(positives == 0):
        absent = [label for label, count in zip(labels, positives, strict=True) if count == 0]
        raise ValueError(f"IRLbl is undefined for labels with zero positives: {', '.join(absent)}")
    maximum = float(positives.max())
    return {
        label: maximum / float(count)
        for label, count in zip(labels, positives, strict=True)
    }


def mean_imbalance_ratio(irlbl: Mapping[str, float]) -> float:
    if not irlbl:
        raise ValueError("MeanIR requires at least one label")
    return float(np.mean(list(irlbl.values())))


def scumble_metrics(
    data: pd.DataFrame | np.ndarray, labels: Sequence[str] = EXPECTED_LABELS
) -> dict[str, Any]:
    """Compute original SCUMBLE/SCUMBLELbl on non-empty labelsets.

    For an instance, SCUMBLEins = 1 - geometric_mean(active IRLbl) /
    arithmetic_mean(active IRLbl). Zero-label rows have no active-label
    population, so they are excluded from the dataset mean and reported.
    """

    values = _matrix(data, labels)
    irlbl = imbalance_ratios(values, labels)
    ratios = np.array([irlbl[label] for label in labels], dtype=float)
    scores = np.full(len(values), np.nan, dtype=float)
    for index, row in enumerate(values):
        active = ratios[row == 1]
        if not len(active):
            continue
        geometric = float(np.exp(np.log(active).mean()))
        arithmetic = float(active.mean())
        scores[index] = float(np.clip(1.0 - geometric / arithmetic, 0.0, 1.0))
    emotional = ~np.isnan(scores)
    scumble = float(scores[emotional].mean()) if emotional.any() else 0.0
    scumble_lbl = {
        label: float(np.nanmean(scores[values[:, position] == 1]))
        for position, label in enumerate(labels)
    }
    return {
        "irlbl": irlbl,
        "mean_ir": mean_imbalance_ratio(irlbl),
        "scumble": scumble,
        "scumble_lbl": scumble_lbl,
        "instances_used": int(emotional.sum()),
        "zero_label_instances_excluded": int((~emotional).sum()),
        "instance_scores": scores,
    }


def truncation_statistics(
    lengths: Sequence[int], thresholds: Sequence[int] = (64, 128, 256, 512)
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(lengths, dtype=int)
    result: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        truncated = int((values > threshold).sum())
        result[str(threshold)] = {
            "truncated_examples": truncated,
            "truncated_percentage": 100.0 * truncated / len(values) if len(values) else 0.0,
        }
    return result


def split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    values = _matrix(frame)
    counts = values.sum(axis=1)
    rows = len(frame)
    label_metrics: dict[str, Any] = {}
    for index, label in enumerate(EXPECTED_LABELS):
        positive = int(values[:, index].sum())
        negative = rows - positive
        label_metrics[label] = {
            "positive": positive,
            "negative": negative,
            "prevalence_percentage": 100.0 * positive / rows if rows else 0.0,
            "positive_negative_ratio": positive / negative if negative else None,
        }
    distribution = {str(number): int((counts == number).sum()) for number in range(N_LABELS + 1)}
    zero = distribution["0"]
    single = distribution["1"]
    multi = int((counts >= 2).sum())
    emotional = counts > 0
    return {
        "rows": rows,
        "unique_texts": int(frame["text_raw"].nunique(dropna=False)),
        "unique_ids": int(frame["id"].nunique(dropna=False)),
        "labels": label_metrics,
        "num_labels_distribution": distribution,
        "zero_label_examples": zero,
        "zero_label_percentage": 100.0 * zero / rows if rows else 0.0,
        "single_label_examples": single,
        "single_label_percentage": 100.0 * single / rows if rows else 0.0,
        "multi_label_examples": multi,
        "multi_label_percentage": 100.0 * multi / rows if rows else 0.0,
        "label_cardinality": float(counts.mean()) if rows else 0.0,
        "label_density": float(counts.mean() / N_LABELS) if rows else 0.0,
        "emotional_examples": int(emotional.sum()),
        "emotional_label_cardinality": float(counts[emotional].mean()) if emotional.any() else 0.0,
        "emotional_label_density": float(counts[emotional].mean() / N_LABELS) if emotional.any() else 0.0,
    }


def labelset_analysis(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = {
        split: labelset_strings(frame).value_counts().to_dict() for split, frame in frames.items()
    }
    all_labelsets = set().union(*(set(item) for item in counts.values()))
    ordered = sorted(all_labelsets, key=lambda item: (-counts["train"].get(item, 0), item))
    rows: list[dict[str, Any]] = []
    for labelset in ordered:
        row: dict[str, Any] = {"labelset": labelset}
        for split, frame in frames.items():
            count = int(counts[split].get(labelset, 0))
            row[f"{split}_count"] = count
            row[f"{split}_pct"] = 100.0 * count / len(frame) if len(frame) else 0.0
        rows.append(row)
    summary = {
        split: {
            "observed": len(counts[split]),
            "possible": 2**N_LABELS,
            "appearing_once": sum(value == 1 for value in counts[split].values()),
            "appearing_fewer_than_5": sum(value < 5 for value in counts[split].values()),
            "top_20": [
                {"labelset": labelset, "count": int(count)}
                for labelset, count in sorted(
                    counts[split].items(), key=lambda item: (-item[1], item[0])
                )[:20]
            ],
        }
        for split in frames
    }
    return pd.DataFrame(rows), summary


def minority_concurrence_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    values = _matrix(frame)
    irlbl = imbalance_ratios(values)
    mean_ir = mean_imbalance_ratio(irlbl)
    minority = [label for label in EXPECTED_LABELS if irlbl[label] > mean_ir]
    majority = [label for label in EXPECTED_LABELS if irlbl[label] <= mean_ir]
    counts = values.sum(axis=1)
    rows: list[dict[str, Any]] = []
    for label in minority:
        position = EXPECTED_LABELS.index(label)
        present = values[:, position] == 1
        positive = int(present.sum())
        alone = int((present & (counts == 1)).sum())
        majority_positions = [EXPECTED_LABELS.index(item) for item in majority if item != label]
        with_majority = int(
            (present & (values[:, majority_positions].sum(axis=1) > 0)).sum()
        ) if majority_positions else 0
        co_counts = {
            other: int((present & (values[:, other_position] == 1)).sum())
            for other_position, other in enumerate(EXPECTED_LABELS)
            if other != label
        }
        most_frequent, most_count = sorted(
            co_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        rows.append(
            {
                "minority_label": label,
                "positive_examples": positive,
                "alone": alone,
                "with_majority_label": with_majority,
                "most_frequent_co_label": most_frequent,
                "most_frequent_co_label_count": most_count,
                "p_most_frequent_co_label_given_minority": most_count / positive if positive else 0.0,
            }
        )
    return pd.DataFrame(rows), minority, majority


def distribution_summary(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99])
    return {
        "min": int(array.min()),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p25": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p97_5": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": int(array.max()),
    }


def text_length_analysis(texts: Sequence[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    characters = np.array([len(text) for text in texts], dtype=int)
    words = np.array([len(re.findall(r"\S+", text)) for text in texts], dtype=int)
    lines = np.array([text.count("\n") + 1 for text in texts], dtype=int)
    arrays = {"characters": characters, "words": words, "lines": lines}
    return {name: distribution_summary(values) for name, values in arrays.items()}, arrays


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
MENTION_RE = re.compile(r"(?<!\w)@[\w_]+", re.UNICODE)
HASHTAG_RE = re.compile(r"(?<!\w)#[\w_]+", re.UNICODE)
EMOTICON_RE = re.compile(
    r"(?:(?:[:;=8xX][-^']?[)(/\\DPpOo])|(?:[)(][-^']?[:;=8])|(?:<3))"
)
HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);")


def social_media_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    import emoji

    texts = frame["text_raw"].astype(str).tolist()
    emoji_lists = [emoji.emoji_list(text) for text in texts]
    uppercase_word = lambda text: any(
        len(word) >= 2 and word.isupper()
        for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    )
    predicates: dict[str, list[bool]] = {
        "url": [bool(URL_RE.search(text)) for text in texts],
        "mention": [bool(MENTION_RE.search(text)) for text in texts],
        "hashtag": [bool(HASHTAG_RE.search(text)) for text in texts],
        "emoji": [bool(items) for items in emoji_lists],
        "text_emoticon": [bool(EMOTICON_RE.search(text)) for text in texts],
        "uppercase_character": [any(character.isupper() for character in text) for text in texts],
        "uppercase_word": [uppercase_word(text) for text in texts],
        "repeated_exclamation": [bool(re.search(r"!{2,}", text)) for text in texts],
        "repeated_question": [bool(re.search(r"\?{2,}", text)) for text in texts],
        "mixed_question_exclamation": ["?!" in text or "!?" in text for text in texts],
        "line_break": ["\n" in text or "\r" in text for text in texts],
        "number": [bool(re.search(r"\d", text)) for text in texts],
        "non_ascii": [any(ord(character) > 127 for character in text) for text in texts],
        "possible_html_entity": [bool(HTML_ENTITY_RE.search(text)) for text in texts],
    }
    rows = [
        {
            "feature": feature,
            "count": int(sum(mask)),
            "percentage": 100.0 * sum(mask) / len(frame),
        }
        for feature, mask in predicates.items()
    ]
    emoji_counter = Counter(item["emoji"] for items in emoji_lists for item in items)
    emoji_present = np.array(predicates["emoji"], dtype=bool)
    association: dict[str, Any] = {}
    for label in EXPECTED_LABELS:
        label_values = frame[label].to_numpy(dtype=int)
        with_emoji = label_values[emoji_present]
        without_emoji = label_values[~emoji_present]
        association[label] = {
            "emoji_text_positive": int(with_emoji.sum()),
            "emoji_text_total": int(len(with_emoji)),
            "emoji_text_prevalence_percentage": 100.0 * with_emoji.mean() if len(with_emoji) else 0.0,
            "non_emoji_text_positive": int(without_emoji.sum()),
            "non_emoji_text_total": int(len(without_emoji)),
            "non_emoji_text_prevalence_percentage": 100.0 * without_emoji.mean() if len(without_emoji) else 0.0,
        }
    emoji_summary = {
        "texts_with_emoji": int(emoji_present.sum()),
        "texts_with_emoji_percentage": 100.0 * emoji_present.mean(),
        "total_emojis": int(sum(emoji_counter.values())),
        "top_20": [
            {"emoji": symbol, "count": count}
            for symbol, count in sorted(emoji_counter.items(), key=lambda item: (-item[1], item[0]))[:20]
        ],
        "label_association": association,
    }
    return pd.DataFrame(rows), emoji_summary


def tokenize_lengths(texts: Sequence[str], tokenizer: Any) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return np.array([len(ids) for ids in encoded["input_ids"]], dtype=int)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    content = frame.to_csv(index=index, float_format="%.8f", lineterminator="\n")
    _write_text(path, content)


def _save_figure(figure: Any, base_path: Path, aliases: Sequence[Path] = ()) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [base_path, *aliases]
    for target in targets:
        for suffix in (".pdf", ".png"):
            destination = target.with_suffix(suffix)
            metadata = {"Creator": "BRIGHTER PT-BR deterministic EDA"}
            if suffix == ".pdf":
                metadata["CreationDate"] = None
            figure.savefig(
                destination,
                dpi=300,
                bbox_inches="tight",
                metadata=metadata,
                facecolor="white",
            )


def _heatmap(matrix: pd.DataFrame, title: str, value_format: str = ".2f") -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    values = matrix.to_numpy(dtype=float)
    image = axis.imshow(values, cmap="viridis", aspect="equal")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_xlabel("Conditioned/co-occurring label")
    axis.set_ylabel("Reference label")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = "NA" if np.isnan(value) else format(value, value_format)
            color = "white" if not np.isnan(value) and value > np.nanmedian(values) else "black"
            axis.text(column, row, label, ha="center", va="center", fontsize=8, color=color)
    figure.colorbar(image, ax=axis, shrink=0.82)
    figure.tight_layout()
    return figure


def _bar_figure(labels: Sequence[str], values: Sequence[float], title: str, ylabel: str) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values)
    axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            f"{value:.2f}" if isinstance(value, float) and not float(value).is_integer() else f"{value:g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    figure.tight_layout()
    return figure


def _histogram(values: Sequence[int], title: str, xlabel: str) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.hist(values, bins="fd", edgecolor="white", linewidth=0.4)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Texts")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def _upset_figure(train: pd.DataFrame) -> Any:
    import matplotlib.pyplot as plt
    from upsetplot import UpSet, from_memberships

    memberships = [
        [label for label, active in zip(EXPECTED_LABELS, row, strict=True) if active]
        for row in _matrix(train)
    ]
    memberships = [membership for membership in memberships if membership]
    # UpSetPlot 0.9 emits pandas-deprecation warnings but remains numerically
    # correct on the pinned pandas 2.x stack. Keep pipeline logs researcher-facing.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = from_memberships(memberships).groupby(level=list(range(N_LABELS))).sum()
        figure = plt.figure(figsize=(10.5, 6.0))
        upset = UpSet(
            data,
            subset_size="sum",
            show_counts=True,
            sort_by="cardinality",
            sort_categories_by=None,
            min_subset_size=5,
        )
        upset.plot(fig=figure)
    figure.suptitle("Training labelset intersections (non-empty, n ≥ 5)", y=1.01)
    return figure


def _latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def latex_tabular(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None) -> str:
    alignment = align or ("l" + "r" * (len(headers) - 1))
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(_latex_escape(item) for item in headers) + r" \\",
        "\\midrule",
    ]
    lines.extend(" & ".join(_latex_escape(item) for item in row) + r" \\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _read_curated(project_root: Path, manifest: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split, details in manifest["curated_files"].items():
        path = project_root / details["path"]
        if sha256_file(path) != details["sha256"]:
            raise RuntimeError(f"Frozen curated input hash mismatch for {split}: {path}")
        frame = pd.read_parquet(path)
        if not frame["source_revision"].eq(FROZEN_REVISION).all() or not frame["source_track"].eq("A").all():
            raise RuntimeError(f"Invalid frozen Track A provenance in {split}")
        frames[split] = frame
    return frames


def _resolve_tokenizer(project_root: Path, requested_revision: str | None = None) -> tuple[Any, str]:
    # This stage intentionally loads only tokenizer assets, never a model backend.
    os.environ.setdefault("USE_TORCH", "0")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    summary_path = project_root / "metadata" / "brighter_ptbr" / "eda" / "eda_summary.json"
    revision = requested_revision
    if revision is None and summary_path.exists():
        previous = load_json(summary_path)
        tokenizer_meta = previous.get("tokenizer", {})
        if tokenizer_meta.get("model") == TOKENIZER_MODEL:
            revision = tokenizer_meta.get("revision")
    if revision is None:
        info = HfApi().model_info(TOKENIZER_MODEL)
        if not info.sha:
            raise RuntimeError("Tokenizer repository metadata did not include a commit SHA")
        revision = str(info.sha)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, revision=revision, use_fast=True)
    return tokenizer, revision


def _generate_figures(
    figure_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    split_stats: Mapping[str, Any],
    cooccurrence: Mapping[str, pd.DataFrame],
    conditional: pd.DataFrame,
    jaccard: pd.DataFrame,
    lift: pd.DataFrame,
    irlbl: Mapping[str, float],
    text_arrays: Mapping[str, np.ndarray],
    token_lengths: np.ndarray,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    generated: list[Path] = []

    def save(fig: Any, name: str, aliases: Sequence[str] = ()) -> None:
        bases = [figure_dir / alias for alias in aliases]
        _save_figure(fig, figure_dir / name, bases)
        generated.extend(
            path.with_suffix(suffix)
            for path in [figure_dir / name, *bases]
            for suffix in (".pdf", ".png")
        )
        plt.close(fig)

    train_prevalence = [split_stats["train"]["labels"][label]["prevalence_percentage"] for label in EXPECTED_LABELS]
    save(_bar_figure(EXPECTED_LABELS, train_prevalence, "Training label prevalence", "Prevalence (%)"), "01_label_distribution_train")
    label_counts = [split_stats["train"]["num_labels_distribution"][str(number)] for number in range(7)]
    save(_bar_figure([str(number) for number in range(7)], label_counts, "Labels per training text", "Examples"), "02_label_count_distribution_train")
    save(_upset_figure(frames["train"]), "labelset_upset_train", ["03_labelset_upset_train"])
    save(_heatmap(cooccurrence["train"], "Training label co-occurrence", ".0f"), "cooccurrence_train", ["04_cooccurrence_train"])
    save(_heatmap(cooccurrence["dev"], "Development label co-occurrence", ".0f"), "cooccurrence_dev")
    save(_heatmap(conditional, "Training conditional probabilities", ".2f"), "conditional_probability_train", ["05_conditional_probability_train"])
    save(_heatmap(jaccard, "Training label Jaccard", ".2f"), "jaccard_labels_train")
    save(_heatmap(lift, "Training label lift", ".2f"), "lift_train")
    save(_bar_figure(EXPECTED_LABELS, [irlbl[label] for label in EXPECTED_LABELS], "Training label imbalance", "IRLbl"), "irlbl_train", ["06_irlbl_train"])
    save(_histogram(text_arrays["characters"], "Training text length", "Characters"), "text_length_chars_train")
    save(_histogram(text_arrays["words"], "Training text length", "Whitespace-delimited words"), "text_length_words_train")
    save(_histogram(token_lengths, "BERTimbau token length", "Tokens including special tokens"), "07_text_length_tokens_train")
    return generated


def _render_report(summary: Mapping[str, Any]) -> str:
    splits = summary["splits"]
    imbalance = summary["imbalance"]
    train_dev = summary["train_dev_comparison"]
    tokenizer = summary["tokenizer"]
    token = summary["token_lengths"]
    social = summary["social_media_features"]
    emoji = summary["emoji"]
    minority = imbalance["minority_labels"]
    max5 = summary["recommended_max_length_candidates"]
    label_rows = [
        [
            label,
            f"{splits['train']['labels'][label]['prevalence_percentage']:.2f}",
            f"{splits['dev']['labels'][label]['prevalence_percentage']:.2f}",
            f"{splits['test']['labels'][label]['prevalence_percentage']:.2f}",
        ]
        for label in EXPECTED_LABELS
    ]
    count_rows = [
        [split, splits[split]["zero_label_examples"], splits[split]["single_label_examples"], splits[split]["multi_label_examples"]]
        for split in ("train", "dev", "test")
    ]
    card_rows = [
        [
            split,
            f"{splits[split]['label_cardinality']:.4f}",
            f"{splits[split]['label_density']:.4f}",
            f"{splits[split]['emotional_label_cardinality']:.4f}",
            f"{splits[split]['emotional_label_density']:.4f}",
        ]
        for split in ("train", "dev", "test")
    ]
    rare_dev = [
        f"{label}={splits['dev']['labels'][label]['positive']}"
        for label in EXPECTED_LABELS
        if splits["dev"]["labels"][label]["positive"] < 20
    ]
    social_by_name = {item["feature"]: item for item in social}
    lines = [
        "# BRIGHTER PT-BR Track A — Multilabel EDA",
        "",
        "## 1. Dataset Version",
        "",
        f"Dataset: `{summary['dataset']}`; configuration: `{summary['configuration']}`; Track A; revision `{summary['revision']}`.",
        "",
        "## 2. Experimental Protocol",
        "",
        "`train + dev` may support methodological development. **TEST SET — DESCRIPTIVE ONLY.** "
        "Test statistics below document the corpus and must not guide model, loss, threshold, preprocessing, augmentation, sampling, or hyperparameter choices.",
        "",
        "No dataset content was modified by this analysis.",
        "",
        "## 3. Label Distribution",
        "",
        _markdown_table(["Emotion", "Train %", "Dev %", "Test % (descriptive only)"], label_rows),
        "",
    ]
    for split in ("train", "dev", "test"):
        lines.extend(
            [
                f"### {split}" + (" — TEST SET — DESCRIPTIVE ONLY" if split == "test" else ""),
                "",
                _markdown_table(
                    ["Emotion", "Positive", "Negative", "Prevalence (%)", "Pos/Neg ratio"],
                    [
                        [
                            label,
                            splits[split]["labels"][label]["positive"],
                            splits[split]["labels"][label]["negative"],
                            f"{splits[split]['labels'][label]['prevalence_percentage']:.2f}",
                            f"{splits[split]['labels'][label]['positive_negative_ratio']:.4f}",
                        ]
                        for label in EXPECTED_LABELS
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 4. Zero-, Single- and Multi-Label Instances",
            "",
            _markdown_table(["Split", "Zero", "Single", "Multi (≥2)"], count_rows),
            "",
            "The six all-zero outputs are treated as `none`; no seventh `neutral` class was created.",
            "",
            "## 5. Label Cardinality and Density",
            "",
            _markdown_table(["Split", "Cardinality", "Density", "Emotional cardinality", "Emotional density"], card_rows),
            "",
            "## 6. Labelset Distribution",
            "",
            f"Training observes **{summary['labelsets']['train']['observed']} / 64** possible labelsets; "
            f"{summary['labelsets']['train']['appearing_once']} occur once and "
            f"{summary['labelsets']['train']['appearing_fewer_than_5']} occur fewer than five times. "
            f"The most common labelset is `{summary['labelsets']['train']['top_20'][0]['labelset']}` "
            f"({summary['labelsets']['train']['top_20'][0]['count']} examples). See `labelsets.csv` and the training UpSet plot.",
            "",
            "## 7. Label Co-occurrence",
            "",
            "Co-occurrence matrices were computed for train and dev only. Diagonals equal label-positive counts. "
            "No test co-occurrence analysis is used for methodological interpretation.",
            "",
            "## 8. Conditional Relationships",
            "",
            "The ordered training matrix reports `P(column label | row label)`, so opposing directions are intentionally distinct. "
            "Training Jaccard and lift matrices provide symmetric overlap and independence-relative views.",
            "",
            "## 9. Label Imbalance",
            "",
            f"Training MeanIR is **{imbalance['mean_ir']:.4f}**. Labels above MeanIR are: **{', '.join(minority)}**. "
            f"SCUMBLE is **{imbalance['scumble']:.6f}**, computed on {imbalance['scumble_instances_used']} emotional examples; "
            f"{imbalance['scumble_zero_label_instances_excluded']} zero-label examples are excluded because the original active-label formula is undefined for an empty labelset.",
            "",
            "SCUMBLE uses `1 − geometric_mean(IRLbl_active) / arithmetic_mean(IRLbl_active)` per non-empty instance, "
            "then averages instance scores. SCUMBLELbl averages those scores over instances containing each label. "
            "This follows Charte et al., [original HAIS formulation](https://doi.org/10.1007/978-3-319-07617-1_10) "
            "and the [extended analysis](https://doi.org/10.1016/j.neucom.2016.08.158).",
            "",
            "## 10. Minority/Majority Concurrence",
            "",
            _markdown_table(
                ["Minority label", "Positive", "Alone", "With majority", "Most frequent co-label", "P(co-label | minority)"],
                [
                    [
                        item["minority_label"], item["positive_examples"], item["alone"],
                        item["with_majority_label"], item["most_frequent_co_label"],
                        f"{item['p_most_frequent_co_label_given_minority']:.4f}",
                    ]
                    for item in imbalance["minority_concurrence"]
                ],
            ),
            "",
            "## 11. Text Characteristics",
            "",
            f"Training text lengths are summarized without changing text. Emoji occur in "
            f"{emoji['texts_with_emoji']} texts ({emoji['texts_with_emoji_percentage']:.2f}%), "
            f"with {emoji['total_emojis']} emoji occurrences. Social-media indicators include "
            f"URLs in {social_by_name['url']['count']} texts, mentions in {social_by_name['mention']['count']}, "
            f"hashtags in {social_by_name['hashtag']['count']}, and non-ASCII characters in {social_by_name['non_ascii']['count']}.",
            "",
            "## 12. Token Length Analysis",
            "",
            f"Tokenizer: `{tokenizer['model']}` at revision `{tokenizer['revision']}`, with special tokens and no truncation. "
            f"Training token p95={token['distribution']['p95']:.1f}, p99={token['distribution']['p99']:.1f}, max={token['distribution']['max']}. "
            + "; ".join(
                f">{threshold}: {item['truncated_examples']} ({item['truncated_percentage']:.2f}%)"
                for threshold, item in token["truncation"].items()
            )
            + ".",
            "",
            "## 13. Train–Dev Distribution Comparison",
            "",
            f"Largest absolute label-prevalence difference: **{train_dev['largest_label']}** at "
            f"**{train_dev['largest_absolute_difference_percentage_points']:.2f} percentage points**. "
            f"Rare dev positive counts requiring caution: {', '.join(rare_dev) if rare_dev else 'none below 20'}.",
            "",
            "## 14. Data Quality Notes",
            "",
            "The known exact train–test overlap remains untouched; see `../cross_split_leakage.md`. "
            "It was not used for any methodological selection. Curated-input SHA-256 hashes were verified before analysis.",
            "",
            "## 15. Implications for Training",
            "",
            f"- Imbalance is relevant: `{', '.join(minority)}` exceed MeanIR; the maximum IRLbl is {max(imbalance['irlbl'].values()):.2f}.",
            f"- The observed SCUMBLE and per-minority concurrence counts indicate that frequency and concurrence should both be monitored; simple oversampling can replicate majority co-labels and is therefore not automatically safe.",
            "- Weighted BCE merits controlled train/dev evaluation because positive frequencies differ substantially. Focal Loss and Asymmetric Loss also merit comparison, but EDA alone does not establish a winner.",
            "- Augmentation is not yet justified by descriptive statistics alone. If tested later, it should be an ablation based only on train/dev.",
            "- Aggressive text preprocessing is not justified by this EDA: textual emoticons, repeated punctuation, numbers, HTML-like entities, and non-ASCII characters are present and may carry emotion information; emoji, URLs, mentions, hashtags, and uppercase signals were not observed in train.",
            f"- Token-length evidence supports evaluating `MAX_LENGTH` candidates {max5}; the final value must be selected on train/dev efficiency and validation behavior, never test.",
            f"- The official dev set has only 200 examples and very small counts for some labels ({', '.join(rare_dev)}). Per-label threshold calibration may be unstable; out-of-fold train predictions should be considered alongside the official dev protocol.",
            "",
            "### Dataset modifications",
            "",
            "- text preprocessing: NONE",
            "- label modification: NONE",
            "- balancing: NONE",
            "- sampling: NONE",
            "- augmentation: NONE",
            "- synthetic data: NONE",
            "",
        ]
    )
    return "\n".join(lines)


def _write_latex_tables(latex_dir: Path, summary: Mapping[str, Any], labelsets: pd.DataFrame) -> None:
    splits = summary["splits"]
    _write_text(
        latex_dir / "dataset_statistics.tex",
        latex_tabular(
            ["Split", "Rows", "Unique IDs", "Unique texts", "Zero-label (%)"],
            [
                [split.capitalize(), item["rows"], item["unique_ids"], item["unique_texts"], f"{item['zero_label_percentage']:.2f}"]
                for split, item in splits.items()
            ],
        ),
    )
    _write_text(
        latex_dir / "label_distribution.tex",
        latex_tabular(
            ["Emotion", "Train (%)", "Dev (%)", "Test (%, descriptive)"],
            [
                [
                    label.capitalize(),
                    f"{splits['train']['labels'][label]['prevalence_percentage']:.2f}",
                    f"{splits['dev']['labels'][label]['prevalence_percentage']:.2f}",
                    f"{splits['test']['labels'][label]['prevalence_percentage']:.2f}",
                ]
                for label in EXPECTED_LABELS
            ],
        ),
    )
    _write_text(
        latex_dir / "multilabel_statistics.tex",
        latex_tabular(
            ["Split", "Zero", "Single", "Multi", "Cardinality", "Density"],
            [
                [
                    split.capitalize(), item["zero_label_examples"], item["single_label_examples"],
                    item["multi_label_examples"], f"{item['label_cardinality']:.4f}", f"{item['label_density']:.4f}",
                ]
                for split, item in splits.items()
            ],
        ),
    )
    imbalance = summary["imbalance"]
    _write_text(
        latex_dir / "imbalance_statistics.tex",
        latex_tabular(
            ["Emotion", "Positive", "Prevalence (%)", "IRLbl", "SCUMBLELbl"],
            [
                [
                    label.capitalize(), splits["train"]["labels"][label]["positive"],
                    f"{splits['train']['labels'][label]['prevalence_percentage']:.2f}",
                    f"{imbalance['irlbl'][label]:.4f}", f"{imbalance['scumble_lbl'][label]:.6f}",
                ]
                for label in EXPECTED_LABELS
            ]
            + [["Mean / SCUMBLE", "--", "--", f"{imbalance['mean_ir']:.4f}", f"{imbalance['scumble']:.6f}"]],
        ),
    )
    top = labelsets.head(20)
    _write_text(
        latex_dir / "top_labelsets.tex",
        latex_tabular(
            ["Labelset", "Train", "Train (%)", "Dev", "Test (descriptive)"],
            [
                [row.labelset, row.train_count, f"{row.train_pct:.2f}", row.dev_count, row.test_count]
                for row in top.itertuples(index=False)
            ],
        ),
    )


def run_eda(
    project_root: Path,
    *,
    revision: str = FROZEN_REVISION,
    tokenizer_revision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if revision != FROZEN_REVISION:
        raise ValueError(f"EDA is frozen to revision {FROZEN_REVISION}")
    track_manifest_path = project_root / "metadata" / "brighter_ptbr" / "track_a_manifest.json"
    track_manifest = load_json(track_manifest_path)
    if track_manifest.get("revision") != revision:
        raise RuntimeError("Track A manifest revision mismatch")
    frames = _read_curated(project_root, track_manifest)
    split_order = ("train", "dev", "test")
    frames = {split: frames[split] for split in split_order}
    reports_dir = project_root / "reports" / "brighter_ptbr" / "eda"
    figures_dir = project_root / "figures" / "brighter_ptbr" / "eda"
    metadata_dir = project_root / "metadata" / "brighter_ptbr" / "eda"
    latex_dir = reports_dir / "latex"
    for directory in (reports_dir, figures_dir, metadata_dir, latex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    # Matplotlib otherwise tries to create its font cache in the user's profile,
    # which is outside the reproducible project workspace in sandboxed runs.
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache.resolve())

    split_stats = {split: split_statistics(frame) for split, frame in frames.items()}
    labelsets, labelset_summary = labelset_analysis(frames)
    cooccurrence = {split: cooccurrence_matrix(frames[split]) for split in ("train", "dev")}
    conditional = conditional_probability_matrix(frames["train"])
    jaccard = jaccard_label_matrix(frames["train"])
    lift = lift_matrix(frames["train"])
    scumble = scumble_metrics(frames["train"])
    concurrence, minority, majority = minority_concurrence_table(frames["train"])
    text_summary, text_arrays = text_length_analysis(frames["train"]["text_raw"].astype(str).tolist())
    social_table, emoji_summary = social_media_analysis(frames["train"])
    tokenizer, resolved_tokenizer_revision = _resolve_tokenizer(project_root, tokenizer_revision)
    token_lengths = tokenize_lengths(frames["train"]["text_raw"].astype(str).tolist(), tokenizer)
    token_distribution = distribution_summary(token_lengths)
    token_truncation = truncation_statistics(token_lengths)
    recommended = [
        int(threshold)
        for threshold, item in token_truncation.items()
        if item["truncated_percentage"] <= 5.0
    ]
    if not recommended:
        recommended = [512]

    train_dev_rows: list[dict[str, Any]] = []
    for label in EXPECTED_LABELS:
        train_pct = split_stats["train"]["labels"][label]["prevalence_percentage"]
        dev_pct = split_stats["dev"]["labels"][label]["prevalence_percentage"]
        train_dev_rows.append(
            {
                "emotion": label,
                "train_prevalence_percentage": train_pct,
                "dev_prevalence_percentage": dev_pct,
                "absolute_difference_percentage_points": abs(train_pct - dev_pct),
            }
        )
    largest = sorted(
        train_dev_rows,
        key=lambda item: (-item["absolute_difference_percentage_points"], item["emotion"]),
    )[0]
    train_dev_summary = {
        "label_prevalence_differences": train_dev_rows,
        "largest_label": largest["emotion"],
        "largest_absolute_difference_percentage_points": largest["absolute_difference_percentage_points"],
        "cardinality_absolute_difference": abs(split_stats["train"]["label_cardinality"] - split_stats["dev"]["label_cardinality"]),
        "density_absolute_difference": abs(split_stats["train"]["label_density"] - split_stats["dev"]["label_density"]),
        "dev_label_positive_counts": {
            label: split_stats["dev"]["labels"][label]["positive"] for label in EXPECTED_LABELS
        },
    }
    input_hashes = {
        split: details["sha256"] for split, details in track_manifest["curated_files"].items()
    }
    summary: dict[str, Any] = {
        "dataset": DATASET_ID,
        "configuration": CONFIGURATION,
        "revision": revision,
        "track": "A",
        "experimental_protocol": {
            "development_data": ["train", "dev"],
            "test_policy": "TEST SET — DESCRIPTIVE ONLY",
            "test_not_used_for_method_selection": True,
        },
        "input_curated_sha256": input_hashes,
        "splits": split_stats,
        "label_prevalence": {
            split: {
                label: split_stats[split]["labels"][label]["prevalence_percentage"]
                for label in EXPECTED_LABELS
            }
            for split in split_order
        },
        "zero_label_rate": {split: split_stats[split]["zero_label_percentage"] for split in split_order},
        "multi_label_rate": {split: split_stats[split]["multi_label_percentage"] for split in split_order},
        "label_cardinality": {split: split_stats[split]["label_cardinality"] for split in split_order},
        "label_density": {split: split_stats[split]["label_density"] for split in split_order},
        "labelsets": labelset_summary,
        "imbalance": {
            "irlbl": scumble["irlbl"],
            "mean_ir": scumble["mean_ir"],
            "scumble": scumble["scumble"],
            "scumble_lbl": scumble["scumble_lbl"],
            "scumble_instances_used": scumble["instances_used"],
            "scumble_zero_label_instances_excluded": scumble["zero_label_instances_excluded"],
            "minority_labels": minority,
            "majority_labels": majority,
            "minority_concurrence": concurrence.to_dict(orient="records"),
        },
        "irlbl": scumble["irlbl"],
        "mean_ir": scumble["mean_ir"],
        "scumble": scumble["scumble"],
        "text_lengths": text_summary,
        "social_media_features": social_table.to_dict(orient="records"),
        "emoji": emoji_summary,
        "tokenizer": {
            "model": TOKENIZER_MODEL,
            "revision": resolved_tokenizer_revision,
            "add_special_tokens": True,
            "truncation": False,
        },
        "token_lengths": {
            "distribution": token_distribution,
            "truncation": token_truncation,
        },
        "recommended_max_length_candidates": recommended,
        "train_dev_comparison": train_dev_summary,
        "known_train_test_overlap": {
            "count": 1,
            "removed": 0,
            "report": "reports/brighter_ptbr/cross_split_leakage.md",
            "used_for_method_selection": False,
        },
        "dataset_modifications": {
            "text_preprocessing": "none",
            "label_modification": "none",
            "balancing": "none",
            "sampling": "none",
            "augmentation": "none",
            "synthetic_data": "none",
        },
        "warnings": [
            "TEST SET — DESCRIPTIVE ONLY; do not use test statistics for methodological choices.",
            "SCUMBLE excludes zero-label instances because the active-label formulation is undefined for empty labelsets.",
            "The official dev split is small; rare-label threshold calibration may be unstable.",
            "One exact train-test text overlap is documented and left untouched.",
        ],
    }

    _write_csv(reports_dir / "labelsets.csv", labelsets)
    label_distribution_rows = []
    for split in split_order:
        for label in EXPECTED_LABELS:
            label_distribution_rows.append(
                {
                    "split": split,
                    "emotion": label,
                    **split_stats[split]["labels"][label],
                    "test_usage": "descriptive_only" if split == "test" else "development",
                }
            )
    _write_csv(reports_dir / "label_distribution.csv", pd.DataFrame(label_distribution_rows))
    for split, matrix in cooccurrence.items():
        _write_csv(reports_dir / f"cooccurrence_{split}.csv", matrix, index=True)
    _write_csv(reports_dir / "conditional_probability_train.csv", conditional, index=True)
    _write_csv(reports_dir / "jaccard_labels_train.csv", jaccard, index=True)
    _write_csv(reports_dir / "lift_train.csv", lift, index=True)
    imbalance_table = pd.DataFrame(
        [
            {
                "emotion": label,
                "positive_samples": split_stats["train"]["labels"][label]["positive"],
                "prevalence_percentage": split_stats["train"]["labels"][label]["prevalence_percentage"],
                "irlbl": scumble["irlbl"][label],
                "above_mean_ir": scumble["irlbl"][label] > scumble["mean_ir"],
                "scumble_lbl": scumble["scumble_lbl"][label],
            }
            for label in EXPECTED_LABELS
        ]
    )
    _write_csv(reports_dir / "imbalance_train.csv", imbalance_table)
    _write_csv(reports_dir / "minority_concurrence_train.csv", concurrence)
    _write_csv(
        reports_dir / "label_count_distribution.csv",
        pd.DataFrame(
            [
                {
                    "num_labels": number,
                    **{split: split_stats[split]["num_labels_distribution"][str(number)] for split in split_order},
                }
                for number in range(7)
            ]
        ),
    )
    _write_csv(reports_dir / "train_dev_prevalence_comparison.csv", pd.DataFrame(train_dev_rows))
    _write_csv(
        reports_dir / "text_length_statistics_train.csv",
        pd.DataFrame([{"unit": unit, **metrics} for unit, metrics in text_summary.items()]),
    )
    _write_csv(reports_dir / "social_media_features_train.csv", social_table)
    _write_csv(reports_dir / "emoji_top20_train.csv", pd.DataFrame(emoji_summary["top_20"]))
    _write_csv(
        reports_dir / "emoji_label_association_train.csv",
        pd.DataFrame([{"emotion": label, **metrics} for label, metrics in emoji_summary["label_association"].items()]),
    )
    _write_csv(
        reports_dir / "token_truncation_train.csv",
        pd.DataFrame([{"max_length": int(limit), **metrics} for limit, metrics in token_truncation.items()]),
    )
    write_json_atomic(
        reports_dir / "imbalance_metrics.json",
        {
            "irlbl": scumble["irlbl"],
            "mean_ir": scumble["mean_ir"],
            "scumble": scumble["scumble"],
            "scumble_lbl": scumble["scumble_lbl"],
            "zero_label_policy": "excluded from SCUMBLE denominator; no active labels",
            "minority_labels": minority,
        },
    )
    write_json_atomic(metadata_dir / "eda_summary.json", summary)
    _write_latex_tables(latex_dir, summary, labelsets)
    _write_text(reports_dir / "multilabel_eda.md", _render_report(summary))
    figure_paths = _generate_figures(
        figures_dir,
        frames,
        split_stats,
        cooccurrence,
        conditional,
        jaccard,
        lift,
        scumble["irlbl"],
        text_arrays,
        token_lengths,
    )
    summary["artifacts"] = {
        "report": (reports_dir / "multilabel_eda.md").relative_to(project_root).as_posix(),
        "figures": [path.relative_to(project_root).as_posix() for path in figure_paths],
        "latex_directory": latex_dir.relative_to(project_root).as_posix(),
        "machine_readable_summary": (metadata_dir / "eda_summary.json").relative_to(project_root).as_posix(),
    }
    write_json_atomic(metadata_dir / "eda_summary.json", summary)

    most_common = labelset_summary["train"]["top_20"][0]
    largest_diff = train_dev_summary["largest_absolute_difference_percentage_points"]
    lines = [
        "BRIGHTER PT-BR MULTILABEL EDA COMPLETE",
        "",
        "Dataset:",
        f"revision: {revision}",
        "track: A",
        "",
        "Train:",
        f"rows: {split_stats['train']['rows']}",
        f"zero-label: {split_stats['train']['zero_label_examples']}",
        f"single-label: {split_stats['train']['single_label_examples']}",
        f"multi-label: {split_stats['train']['multi_label_examples']}",
        "",
        "Label distribution (train positives):",
    ]
    lines.extend(
        f"{label}: {split_stats['train']['labels'][label]['positive']}"
        for label in EXPECTED_LABELS
    )
    lines.extend(
        [
            "",
            "Imbalance:",
            "IRLbl: " + ", ".join(f"{label}={scumble['irlbl'][label]:.4f}" for label in EXPECTED_LABELS),
            f"MeanIR: {scumble['mean_ir']:.4f}",
            f"SCUMBLE: {scumble['scumble']:.6f}",
            "",
            "Labelsets:",
            f"observed: {labelset_summary['train']['observed']} / 64",
            f"most common: {most_common['labelset']} ({most_common['count']})",
            "",
            "Token length:",
            f"p95: {token_distribution['p95']:.2f}",
            f"p99: {token_distribution['p99']:.2f}",
            f">128: {token_truncation['128']['truncated_examples']}",
            f">256: {token_truncation['256']['truncated_examples']}",
            "",
            "Train x Dev:",
            f"largest label prevalence difference: {train_dev_summary['largest_label']} ({largest_diff:.2f} pp)",
            "",
            "Artifacts:",
            f"reports: {reports_dir}",
            f"figures: {figures_dir}",
            f"latex tables: {latex_dir}",
            f"machine-readable summary: {metadata_dir / 'eda_summary.json'}",
            "",
            "Dataset modifications:",
            "text preprocessing: NONE",
            "label modification: NONE",
            "balancing: NONE",
            "sampling: NONE",
            "augmentation: NONE",
            "synthetic data: NONE",
        ]
    )
    return "\n".join(lines), summary
