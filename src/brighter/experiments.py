"""Reproducible development-only baselines for BRIGHTER PT-BR Track A."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .curation import FROZEN_REVISION
from .losses import FoldClassWeights, build_loss, calculate_fold_class_weights
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .validation import EXPECTED_LABELS

LOGGER = logging.getLogger(__name__)
N_FOLDS = 5
SEED = 42
THRESHOLD = 0.5
DEV_POOL_SIZE = 2426
TEST_SIZE = 2226


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"experiment_id", "model_type", "seed", "folds", "threshold"}
    missing = sorted(required - set(config or {}))
    if missing:
        raise ValueError(f"Experiment config missing fields: {missing}")
    if config["seed"] != SEED or config["folds"] != N_FOLDS or config["threshold"] != THRESHOLD:
        raise ValueError("Baseline protocol requires seed=42, folds=5, threshold=0.5")
    permitted_imbalance = {"none", "loss_weighting"}
    if config.get("imbalance_method", "none") not in permitted_imbalance:
        raise ValueError("sampling, augmentation, and dataset balancing are prohibited")
    if config.get("model_type") == "transformer":
        permitted_losses = {"bce", "weighted_bce", "focal", "weighted_focal", "asl", "BCEWithLogitsLoss"}
        if config.get("loss") not in permitted_losses:
            raise ValueError(f"Unsupported transformer loss: {config.get('loss')}")
        training = config.get("training", {})
        physical = int(training.get("physical_batch_size", 0))
        accumulation = int(training.get("gradient_accumulation_steps", 0))
        if physical * accumulation != int(training.get("effective_batch_size", -1)):
            raise ValueError("physical batch * accumulation must equal effective batch")
    return config


def _track_dir(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "curated"
        / "brighter_ptbr"
        / FROZEN_REVISION
        / "track_a"
    )


def construct_development_pool(project_root: Path) -> pd.DataFrame:
    """Load only train+dev. The sealed test file is deliberately never opened."""
    track_dir = _track_dir(project_root)
    parts = []
    for split in ("train", "dev"):
        frame = pd.read_parquet(track_dir / f"{split}.parquet")
        if not frame["source_split"].eq(split).all():
            raise RuntimeError(f"Invalid source_split provenance in {split}")
        parts.append(frame)
    pool = pd.concat(parts, ignore_index=True, copy=False)
    if len(pool) != DEV_POOL_SIZE:
        raise RuntimeError(f"Expected {DEV_POOL_SIZE} development rows, found {len(pool)}")
    if pool["id"].duplicated().any():
        raise RuntimeError("Duplicated IDs in development pool")
    if pool["text_raw"].duplicated().any():
        duplicates = int(pool["text_raw"].duplicated(keep=False).sum())
        raise RuntimeError(f"Exact duplicated texts in development pool: {duplicates}")
    if not np.isin(pool[list(EXPECTED_LABELS)].to_numpy(), [0, 1]).all():
        raise RuntimeError("Non-binary labels in development pool")
    return pool


def development_input_hashes(project_root: Path) -> dict[str, str]:
    track_dir = _track_dir(project_root)
    return {split: sha256_file(track_dir / f"{split}.parquet") for split in ("train", "dev")}


def persist_cv_folds(project_root: Path, pool: pd.DataFrame) -> dict[str, int]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    path = project_root / "metadata" / "brighter_ptbr" / "experiments" / "cv_folds.json"
    ids = pool["id"].astype(str).tolist()
    hashes = development_input_hashes(project_root)
    if path.exists():
        payload = load_json(path)
        if (
            payload.get("revision") != FROZEN_REVISION
            or payload.get("seed") != SEED
            or payload.get("n_splits") != N_FOLDS
            or payload.get("source_sha256") != hashes
        ):
            raise RuntimeError("Persisted CV folds do not match the frozen protocol")
        assignments = {item["id"]: int(item["fold"]) for item in payload["assignments"]}
        if set(assignments) != set(ids) or len(assignments) != DEV_POOL_SIZE:
            raise RuntimeError("Persisted CV assignments do not cover the development pool exactly")
        return assignments

    y = pool[list(EXPECTED_LABELS)].to_numpy(dtype=np.int64)
    splitter = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(pool), -1, dtype=np.int64)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(pool)), y)):
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise RuntimeError("Fold construction left unassigned examples")
    assignments = dict(zip(ids, folds.tolist(), strict=True))
    payload = {
        "revision": FROZEN_REVISION,
        "track": "A",
        "development_splits": ["train", "dev"],
        "test_policy": "SEALED — NOT LOADED OR EVALUATED",
        "method": "iterative-stratification.MultilabelStratifiedKFold",
        "package_version": importlib.metadata.version("iterative-stratification"),
        "seed": SEED,
        "n_splits": N_FOLDS,
        "source_sha256": hashes,
        "assignments": [{"id": identifier, "fold": int(fold)} for identifier, fold in zip(ids, folds, strict=True)],
    }
    write_json_atomic(path, payload)
    return assignments


def attach_folds(pool: pd.DataFrame, assignments: Mapping[str, int]) -> pd.DataFrame:
    result = pool.copy()
    result["fold"] = result["id"].astype(str).map(assignments)
    if result["fold"].isna().any():
        raise RuntimeError("Missing persisted fold for a development example")
    result["fold"] = result["fold"].astype(int)
    return result


def validate_folds(pool: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_ids = set(pool["id"].astype(str))
    for fold in range(N_FOLDS):
        validation = pool.loc[pool["fold"] == fold]
        training = pool.loc[pool["fold"] != fold]
        val_ids = set(validation["id"].astype(str))
        train_ids = set(training["id"].astype(str))
        if train_ids & val_ids or train_ids | val_ids != all_ids:
            raise RuntimeError(f"ID leakage or incomplete partition in fold {fold}")
        shared_texts = set(training["text_raw"]) & set(validation["text_raw"])
        if shared_texts:
            raise RuntimeError(f"Exact text leakage in fold {fold}: {len(shared_texts)} texts")
        labels = validation[list(EXPECTED_LABELS)].to_numpy(dtype=int)
        counts = labels.sum(axis=1)
        row = {
            "fold": fold,
            "examples": len(validation),
            "zero_label": int((counts == 0).sum()),
            "multilabel": int((counts >= 2).sum()),
        }
        for index, label in enumerate(EXPECTED_LABELS):
            row[f"{label}_positive"] = int(labels[:, index].sum())
            row[f"{label}_prevalence"] = float(labels[:, index].mean())
        rows.append(row)
    if sum(row["examples"] for row in rows) != DEV_POOL_SIZE:
        raise RuntimeError("Fold validation counts do not sum to the development pool")
    return rows


def write_cv_distribution(project_root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = project_root / "reports" / "brighter_ptbr" / "experiments" / "cv_distribution.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BRIGHTER PT-BR development-fold distribution",
        "",
        f"Frozen revision: `{FROZEN_REVISION}`. Development pool: official train + dev ({DEV_POOL_SIZE} examples).",
        "",
        "The official test set is **SEALED — NOT LOADED OR EVALUATED**. The documented train–test overlap is not used here.",
        "",
        "Method: 5-fold iterative multilabel stratification; seed 42. Exact ID and text disjointness was verified for every train/validation partition.",
        "",
        "| Fold | N | Zero-label | Multilabel | " + " | ".join(f"{label} +" for label in EXPECTED_LABELS) + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(EXPECTED_LABELS),
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['examples']} | {row['zero_label']} | {row['multilabel']} | "
            + " | ".join(str(row[f"{label}_positive"]) for label in EXPECTED_LABELS)
            + " |"
        )
    lines.extend(["", "## Prevalence (%)", "", "| Fold | " + " | ".join(EXPECTED_LABELS) + " |", "|---:|" + "---:|" * len(EXPECTED_LABELS)])
    for row in rows:
        lines.append(f"| {row['fold']} | " + " | ".join(f"{100*row[f'{label}_prevalence']:.2f}" for label in EXPECTED_LABELS) + " |")
    _write_text(path, "\n".join(lines) + "\n")
    return path


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def threshold_probabilities(probabilities: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return (values >= threshold).astype(np.int64)


def zero_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true_zero = np.asarray(y_true).sum(axis=1) == 0
    predicted_zero = np.asarray(y_pred).sum(axis=1) == 0
    support = int(true_zero.sum())
    exact = float(predicted_zero[true_zero].mean()) if support else float("nan")
    return {
        "support": support,
        "exact_neutral_accuracy": exact,
        "neutral_false_positive_rate": 1.0 - exact if support else float("nan"),
    }


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss, jaccard_score

    support = int(mask.sum())
    if not support:
        return {"support": 0, "macro_f1": float("nan"), "micro_f1": float("nan"), "subset_accuracy": float("nan"), "jaccard_samples": float("nan"), "hamming_loss": float("nan")}
    return {
        "support": support,
        "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true[mask], y_pred[mask], average="micro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
        "jaccard_samples": float(jaccard_score(y_true[mask], y_pred[mask], average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true[mask], y_pred[mask])),
    }


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = THRESHOLD) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        hamming_loss,
        jaccard_score,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = threshold_probabilities(probabilities, threshold)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    counts = y_true.sum(axis=1)
    result: dict[str, Any] = {
        "threshold": threshold,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_average_precision": float(average_precision_score(y_true, probabilities, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "gold_label_cardinality": float(y_true.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred.sum(axis=1).mean()),
        "per_label": {},
        "zero_label": zero_label_metrics(y_true, y_pred),
        "groups": {
            "single_label": _group_metrics(y_true, y_pred, counts == 1),
            "multi_label": _group_metrics(y_true, y_pred, counts >= 2),
        },
    }
    for index, label in enumerate(EXPECTED_LABELS):
        tn = int(((y_true[:, index] == 0) & (y_pred[:, index] == 0)).sum())
        fp = int(((y_true[:, index] == 0) & (y_pred[:, index] == 1)).sum())
        fn = int(((y_true[:, index] == 1) & (y_pred[:, index] == 0)).sum())
        tp = int(((y_true[:, index] == 1) & (y_pred[:, index] == 1)).sum())
        result["per_label"][label] = {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
            "average_precision": float(average_precision_score(y_true[:, index], probabilities[:, index])) if support[index] else 0.0,
            "gold_prevalence": float(y_true[:, index].mean()),
            "predicted_prevalence": float(y_pred[:, index].mean()),
            "prevalence_delta": float(y_pred[:, index].mean() - y_true[:, index].mean()),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    return result


def _oof_frame(validation: pd.DataFrame, probabilities: np.ndarray, fold: int) -> pd.DataFrame:
    result = pd.DataFrame({"id": validation["id"].astype(str).to_numpy(), "fold": fold})
    for index, label in enumerate(EXPECTED_LABELS):
        result[f"{label}_prob"] = probabilities[:, index].astype(np.float64)
    for label in EXPECTED_LABELS:
        result[f"{label}_true"] = validation[label].to_numpy(dtype=np.int64)
    return result


def validate_oof(oof: pd.DataFrame, pool: pd.DataFrame) -> None:
    expected_ids = pool["id"].astype(str).tolist()
    if len(oof) != DEV_POOL_SIZE or oof["id"].duplicated().any() or set(oof["id"]) != set(expected_ids):
        raise RuntimeError("OOF predictions do not cover each development example exactly once")
    expected_folds = pool.set_index("id")["fold"].astype(int)
    observed = oof.set_index("id")["fold"].astype(int)
    if not observed.sort_index().equals(expected_folds.sort_index()):
        raise RuntimeError("OOF fold assignments differ from persisted CV folds")
    probability_columns = [f"{label}_prob" for label in EXPECTED_LABELS]
    values = oof[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError("OOF probabilities outside [0, 1]")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _save_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_fold_class_weights(project_root: Path, config: Mapping[str, Any], fold_metrics: Sequence[Mapping[str, Any]]) -> None:
    """Upsert auditable train-only fold weights for weighted objectives."""

    if config.get("loss") not in {"weighted_bce", "weighted_focal"}:
        return
    rows = []
    for metrics in fold_metrics:
        statistics = metrics["cost"]["training_fold_class_statistics"]
        for index, label in enumerate(EXPECTED_LABELS):
            rows.append({
                "experiment_id": config["experiment_id"], "fold": int(metrics["fold"]), "emotion": label,
                "training_positive": int(statistics["positive"][index]),
                "training_negative": int(statistics["negative"][index]),
                "pos_weight": float(statistics["pos_weight"][index]),
                "focal_alpha": float(statistics["focal_alpha"][index]),
            })
    path = project_root / "reports" / "brighter_ptbr" / "experiments" / "class_weights_by_fold.csv"
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    addition = pd.DataFrame(rows)
    if not current.empty:
        current = current.loc[current["experiment_id"] != config["experiment_id"]]
        addition = pd.concat([current, addition], ignore_index=True)
    addition = addition.sort_values(["experiment_id", "fold", "emotion"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    addition.to_csv(temporary, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(temporary, path)


def _fold_paths(project_root: Path, slug: str, fold: int) -> tuple[Path, Path]:
    base = project_root / "artifacts" / "predictions" / "folds"
    return base / f"{slug}_fold_{fold}.parquet", base / f"{slug}_fold_{fold}_metrics.json"


def _completed_fold(
    project_root: Path,
    slug: str,
    fold: int,
    expected_ids: set[str],
    config_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    prediction_path, metric_path = _fold_paths(project_root, slug, fold)
    if not prediction_path.exists() or not metric_path.exists():
        return None
    frame = pd.read_parquet(prediction_path)
    if set(frame["id"].astype(str)) != expected_ids or len(frame) != len(expected_ids):
        raise RuntimeError(f"Stale or incomplete cached fold {fold} for {slug}")
    metrics = load_json(metric_path)
    if metrics.get("config_sha256") != config_sha256:
        # Results generated before per-fold fingerprints were introduced may be
        # upgraded only when their completed experiment summary proves the same
        # config hash. This prevents silent reuse after a config change.
        summary_path = project_root / "reports" / "brighter_ptbr" / "experiments" / f"{slug}_results.json"
        compatible_legacy = (
            "config_sha256" not in metrics
            and summary_path.exists()
            and load_json(summary_path).get("config_sha256") == config_sha256
        )
        if not compatible_legacy:
            raise RuntimeError(f"Cached fold {fold} for {slug} was produced by a different config")
        metrics["config_sha256"] = config_sha256
        write_json_atomic(metric_path, metrics)
    return frame, metrics


def run_tfidf_fold(config: Mapping[str, Any], training: pd.DataFrame, validation: pd.DataFrame, model_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = config["features"]
    classifier = config["classifier"]
    union = FeatureUnion(
        [
            ("word", TfidfVectorizer(lowercase=False, strip_accents=None, ngram_range=tuple(features["word_ngram_range"]), min_df=features["min_df"], max_features=features["word_max_features"], sublinear_tf=True, dtype=np.float64)),
            ("char", TfidfVectorizer(lowercase=False, strip_accents=None, analyzer="char_wb", ngram_range=tuple(features["char_ngram_range"]), min_df=features["min_df"], max_features=features["char_max_features"], sublinear_tf=True, dtype=np.float64)),
        ]
    )
    logistic = LogisticRegression(C=float(classifier["C"]), solver=classifier["solver"], max_iter=int(classifier["max_iter"]), random_state=SEED, class_weight=None)
    pipeline = Pipeline([("features", union), ("classifier", OneVsRestClassifier(logistic, n_jobs=1))])
    y_train = training[list(EXPECTED_LABELS)].to_numpy(dtype=int)
    started = time.perf_counter()
    pipeline.fit(training["text_raw"].tolist(), y_train)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = pipeline.predict_proba(validation["text_raw"].tolist())
    inference_seconds = time.perf_counter() - started
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib", compress=3)
    return np.asarray(probabilities), {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "examples_per_second": len(validation) / inference_seconds,
        "parameters": None,
        "peak_gpu_memory_bytes": None,
    }


class _TextDataset:
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        encoded = self.tokenizer(self.frame.at[index, "text_raw"], truncation=True, max_length=self.max_length)
        encoded["labels"] = torch.tensor(self.frame.loc[index, list(EXPECTED_LABELS)].to_numpy(dtype=np.float32))
        return encoded


class _MultilabelCollator:
    """Apply tokenizer dynamic padding while stacking six-dimensional labels."""

    def __init__(self, tokenizer: Any):
        from transformers import DataCollatorWithPadding

        self.padding_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        labels = torch.stack([feature.pop("labels") for feature in features])
        batch = self.padding_collator(features)
        batch["labels"] = labels
        return batch


def _resolve_model_revision(project_root: Path, model_name: str) -> str:
    from huggingface_hub import HfApi

    path = project_root / "metadata" / "brighter_ptbr" / "experiments" / "model_revisions.json"
    revisions = load_json(path) if path.exists() else {}
    if model_name not in revisions:
        info = HfApi().model_info(model_name)
        if not info.sha:
            raise RuntimeError(f"No revision SHA returned for {model_name}")
        revisions[model_name] = str(info.sha)
        write_json_atomic(path, revisions)
    return str(revisions[model_name])


def run_transformer_fold(
    project_root: Path,
    config: Mapping[str, Any],
    training: pd.DataFrame,
    validation: pd.DataFrame,
    model_dir: Path,
    fold_weights: FoldClassWeights,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("Transformer baselines require CUDA in this experimental protocol")
    seed_everything(SEED)
    training_config = config["training"]
    model_name = config["model_name"]
    revision = _resolve_model_revision(project_root, model_name)
    project_cache = project_root / ".cache" / "huggingface" / "hub"
    repo_cache_name = "models--" + model_name.replace("/", "--")
    # New encoder experiments are isolated in the project cache. Historical B2
    # was produced from the standard HF cache; retain that exact cached revision
    # when it is not duplicated locally instead of attempting a network fetch.
    cache_dir = project_cache if (project_cache / repo_cache_name / "snapshots" / revision).exists() else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        revision=revision,
        num_labels=len(EXPECTED_LABELS),
        problem_type="multi_label_classification",
        id2label={index: label for index, label in enumerate(EXPECTED_LABELS)},
        label2id={label: index for index, label in enumerate(EXPECTED_LABELS)},
        cache_dir=cache_dir,
        dtype=torch.float32,
    )
    if training_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    device = torch.device("cuda")
    model.to(device)
    configured_loss = "bce" if config["loss"] == "BCEWithLogitsLoss" else str(config["loss"])
    criterion = build_loss(configured_loss, fold_weights=fold_weights, loss_config=config.get("loss_config"))
    if hasattr(criterion, "to"):
        criterion = criterion.to(device)
    collator = _MultilabelCollator(tokenizer)
    train_dataset = _TextDataset(training, tokenizer, int(config["max_length"]))
    validation_dataset = _TextDataset(validation, tokenizer, int(config["max_length"]))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=int(training_config["physical_batch_size"]), shuffle=True, generator=generator, collate_fn=collator, num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=int(training_config["eval_batch_size"]), shuffle=False, collate_fn=collator, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]), weight_decay=float(training_config["weight_decay"]), fused=True)
    accumulation = int(training_config["gradient_accumulation_steps"])
    epochs = int(training_config["epochs"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = optimizer_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=round(total_steps * float(training_config["warmup_ratio"])), num_training_steps=total_steps)
    precision = str(training_config.get("mixed_precision", "fp16"))
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported mixed precision mode: {precision}")
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else None
    autocast_context = (lambda: torch.autocast(device_type="cuda", dtype=autocast_dtype)) if autocast_dtype is not None else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            labels = batch.pop("labels")
            with autocast_context():
                logits = model(**batch).logits
                loss = criterion(logits, labels) / accumulation
            scaler.scale(loss).backward()
            final_microbatch = step + 1 == len(train_loader)
            if (step + 1) % accumulation == 0 or final_microbatch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    training_seconds = time.perf_counter() - started
    model.eval()
    logits = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in validation_loader:
            batch.pop("labels")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with autocast_context():
                logits.append(model(**batch).logits.float().cpu())
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    peak = int(torch.cuda.max_memory_allocated())
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    weight_payload = {
        "positive": fold_weights.positive.tolist(), "negative": fold_weights.negative.tolist(),
        "pos_weight": fold_weights.pos_weight.tolist(), "focal_alpha": fold_weights.focal_alpha.tolist(),
    }
    write_json_atomic(model_dir / "training_provenance.json", {"model_name": model_name, "model_revision": revision, "seed": SEED, "config": dict(config), "training_fold_class_statistics": weight_payload})
    del model, optimizer, scheduler, scaler, train_loader, validation_loader
    torch.cuda.empty_cache()
    return probabilities, {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "examples_per_second": len(validation) / inference_seconds,
        "parameters": parameters,
        "peak_gpu_memory_bytes": peak,
        "model_revision": revision, "training_fold_class_statistics": weight_payload,
    }


def _environment_metadata() -> dict[str, Any]:
    import psutil
    import torch

    cuda = torch.version.cuda
    cudnn = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "scikit-learn", "iterative-stratification")},
        "cuda": cuda,
        "cudnn": cudnn,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "cpu": platform.processor() or "unavailable",
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "determinism": "Seeds fixed; deterministic algorithms requested with warn_only=True. Exact GPU bitwise reproducibility is not guaranteed across hardware, drivers, CUDA, cuDNN, or library versions.",
    }


def _mean_std(fold_metrics: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    values = np.array([float(item[key]) for item in fold_metrics])
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def run_experiment(project_root: Path, config_path: Path, *, final_test: bool = False) -> dict[str, Any]:
    if final_test:
        raise RuntimeError("--final-test is intentionally disabled: the official test set remains sealed")
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    seed_everything(config["seed"])
    pool = construct_development_pool(project_root)
    assignments = persist_cv_folds(project_root, pool)
    pool = attach_folds(pool, assignments)
    fold_distribution = validate_folds(pool)
    write_cv_distribution(project_root, fold_distribution)
    slug = config["artifact_slug"]
    fold_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        validation = pool.loc[pool["fold"] == fold].copy()
        training = pool.loc[pool["fold"] != fold].copy()
        fold_weights = calculate_fold_class_weights(training[list(EXPECTED_LABELS)].to_numpy(dtype=int))
        cached = _completed_fold(project_root, slug, fold, set(validation["id"].astype(str)), config_sha256)
        if cached is not None:
            LOGGER.info("Reusing completed %s fold %d", config["experiment_id"], fold)
            predictions, metrics = cached
            fold_predictions.append(predictions)
            fold_metrics.append(metrics)
            continue
        LOGGER.info("Training %s fold %d: train=%d validation=%d", config["experiment_id"], fold, len(training), len(validation))
        model_dir = project_root / "artifacts" / "models" / slug / f"fold_{fold}"
        if config["model_type"] == "tfidf_logreg":
            probabilities, cost = run_tfidf_fold(config, training, validation, model_dir)
        elif config["model_type"] == "transformer":
            probabilities, cost = run_transformer_fold(project_root, config, training, validation, model_dir, fold_weights)
        else:
            raise ValueError(f"Unsupported model_type: {config['model_type']}")
        y_true = validation[list(EXPECTED_LABELS)].to_numpy(dtype=int)
        metrics = {"fold": fold, "config_sha256": config_sha256, **calculate_metrics(y_true, probabilities), "cost": cost}
        predictions = _oof_frame(validation, probabilities, fold)
        prediction_path, metric_path = _fold_paths(project_root, slug, fold)
        _save_parquet_atomic(predictions, prediction_path)
        write_json_atomic(metric_path, metrics)
        fold_predictions.append(predictions)
        fold_metrics.append(metrics)
        LOGGER.info("Completed %s fold %d macro-F1=%.4f", config["experiment_id"], fold, metrics["macro_f1"])
    oof = pd.concat(fold_predictions, ignore_index=True)
    order = {identifier: index for index, identifier in enumerate(pool["id"].astype(str))}
    oof["_order"] = oof["id"].map(order)
    oof = oof.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    validate_oof(oof, pool)
    _write_fold_class_weights(project_root, config, fold_metrics)
    oof_path = project_root / "artifacts" / "predictions" / f"{slug}_oof.parquet"
    _save_parquet_atomic(oof, oof_path)
    overall = calculate_metrics(
        oof[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int),
        oof[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float),
    )
    previous_summary_path = project_root / "reports" / "brighter_ptbr" / "experiments" / f"{slug}_results.json"
    previous_timestamp = None
    if previous_summary_path.exists():
        previous = load_json(previous_summary_path)
        if previous.get("config_sha256") == config_sha256:
            previous_timestamp = previous.get("completed_timestamp_utc")
    summary = {
        "experiment_id": config["experiment_id"], "artifact_slug": slug,
        "revision": FROZEN_REVISION, "development_examples": len(pool),
        "test_policy": "SEALED — NOT LOADED OR EVALUATED", "config": config,
        "config_sha256": config_sha256, "fold_metrics": fold_metrics,
        "cross_validation": {key: _mean_std(fold_metrics, key) for key in ("macro_f1", "micro_f1", "macro_average_precision", "macro_precision", "macro_recall", "hamming_loss", "jaccard_samples", "subset_accuracy", "gold_label_cardinality", "predicted_label_cardinality")},
        "oof_aggregate": overall,
        "cost": {
            "training_seconds_total": float(sum(item["cost"]["training_seconds"] for item in fold_metrics)),
            "inference_seconds_total": float(sum(item["cost"]["inference_seconds"] for item in fold_metrics)),
            "peak_gpu_memory_bytes": max((item["cost"].get("peak_gpu_memory_bytes") or 0 for item in fold_metrics), default=0) or None,
            "parameters": next((item["cost"].get("parameters") for item in fold_metrics if item["cost"].get("parameters") is not None), None),
        },
        "oof_path": oof_path.relative_to(project_root).as_posix(), "oof_sha256": sha256_file(oof_path),
        "environment": _environment_metadata(), "completed_timestamp_utc": previous_timestamp or utc_now_iso(),
        "methods": {"loss": config.get("loss"), "class_weights": config.get("loss") in {"weighted_bce", "weighted_focal"}, "threshold": 0.5},
        "prohibited_methods": {"threshold_optimization": False, "sampling": False, "augmentation": False, "preprocessing": False},
    }
    summary_path = project_root / "reports" / "brighter_ptbr" / "experiments" / f"{slug}_results.json"
    write_json_atomic(summary_path, summary)
    return summary
