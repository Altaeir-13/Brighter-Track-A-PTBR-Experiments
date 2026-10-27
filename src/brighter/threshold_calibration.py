"""Leakage-safe nested, cross-fitted per-label threshold calibration."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, hamming_loss,
    jaccard_score, precision_recall_fscore_support, precision_score, recall_score,
)

from .experiments import (
    DEV_POOL_SIZE, EXPECTED_LABELS, attach_folds, construct_development_pool,
    development_input_hashes, persist_cv_folds, run_transformer_fold, validate_oof,
    zero_label_metrics,
)
from .losses import calculate_fold_class_weights
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic

GRID = np.arange(1, 100, dtype=np.float64) / 100.0
SYSTEMS = {
    "T1": {"source": "bertweetbr_weighted_focal", "output": "bertweetbr_weighted_focal_thresholded", "short": "E2"},
    "T2": {"source": "bertweetbr_asl", "output": "bertweetbr_asl_thresholded", "short": "E3"},
    "T3": {"source": "bertimbau_large_bce", "output": "bertimbau_large_bce_thresholded", "short": "B2"},
}


def _id_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(temporary, path)


def inner_fold_assignments(outer_train: pd.DataFrame, seed: int = 42) -> dict[str, int]:
    """Create deterministic 3-fold iterative assignments using outer-train only."""

    ids = outer_train["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("outer_train IDs must be unique")
    y = outer_train[list(EXPECTED_LABELS)].to_numpy(dtype=int)
    splitter = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = np.full(len(outer_train), -1, dtype=int)
    for fold, (_, validation_indices) in enumerate(splitter.split(np.zeros(len(outer_train)), y)):
        folds[validation_indices] = fold
    if np.any(folds < 0):
        raise RuntimeError("inner OOF assignment is incomplete")
    return dict(zip(ids, folds.tolist(), strict=True))


def persist_inner_folds(root: Path, system: str, outer_fold: int, outer_train: pd.DataFrame,
                        outer_validation: pd.DataFrame, seed: int = 42) -> dict[str, int]:
    assignments = inner_fold_assignments(outer_train, seed)
    path = root / "metadata" / "brighter_ptbr" / "experiments" / "threshold_inner_folds" / system.lower() / f"outer_{outer_fold}.json"
    train_ids = outer_train["id"].astype(str).tolist()
    validation_ids = outer_validation["id"].astype(str).tolist()
    payload = {
        "system": system, "outer_fold": outer_fold, "inner_folds": 3, "seed": seed,
        "calibration_scope": "outer_train_only", "outer_train_count": len(train_ids),
        "outer_validation_count": len(validation_ids), "outer_train_ids_sha256": _id_digest(train_ids),
        "outer_validation_ids_sha256": _id_digest(validation_ids),
        "outer_validation_excluded_from_assignments": not bool(set(validation_ids) & set(assignments)),
        "assignments": [{"id": identifier, "inner_fold": assignments[identifier]} for identifier in train_ids],
        "test_policy": "SEALED — NOT LOADED OR EVALUATED",
    }
    if path.exists():
        existing = load_json(path)
        if existing != payload:
            raise RuntimeError(f"Persisted inner folds differ for {system} outer fold {outer_fold}")
    else:
        write_json_atomic(path, payload)
    return assignments


def search_threshold(y_true: np.ndarray, probabilities: np.ndarray, grid: np.ndarray = GRID) -> tuple[float, float]:
    """Maximize binary F1; exact/numerical ties select the largest threshold."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    candidates = np.asarray(grid, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or candidates.ndim != 1:
        raise ValueError("threshold search expects aligned one-dimensional arrays")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("probabilities must be finite in [0,1]")
    if np.any(candidates < .01) or np.any(candidates > .99):
        raise ValueError("threshold grid must remain within [0.01,0.99]")
    scores = np.array([f1_score(y, p >= threshold, zero_division=0) for threshold in candidates])
    best = float(scores.max())
    tied = candidates[np.isclose(scores, best, rtol=0.0, atol=1e-12)]
    return float(tied.max()), best


def search_per_label_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds, scores = zip(*(search_threshold(y_true[:, i], probabilities[:, i]) for i in range(len(EXPECTED_LABELS))), strict=True)
    return np.asarray(thresholds), np.asarray(scores)


def decision_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Metrics for already cross-fitted decisions; AP is computed from unchanged probabilities."""

    y_true, y_pred, probabilities = np.asarray(y_true, int), np.asarray(y_pred, int), np.asarray(probabilities, float)
    if y_true.shape != y_pred.shape or y_true.shape != probabilities.shape:
        raise ValueError("truth, decisions, and probabilities must have identical shapes")
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    multi = y_true.sum(axis=1) >= 2
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_average_precision": float(average_precision_score(y_true, probabilities, average="macro")),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "gold_label_cardinality": float(y_true.sum(axis=1).mean()),
        "predicted_label_cardinality": float(y_pred.sum(axis=1).mean()),
        "zero_label": zero_label_metrics(y_true, y_pred), "per_label": {},
        "groups": {"multi_label": {
            "support": int(multi.sum()),
            "macro_f1": float(f1_score(y_true[multi], y_pred[multi], average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true[multi], y_pred[multi], average="micro", zero_division=0)),
            "jaccard_samples": float(jaccard_score(y_true[multi], y_pred[multi], average="samples", zero_division=0)),
            "subset_accuracy": float(accuracy_score(y_true[multi], y_pred[multi])),
        }},
    }
    for i, label in enumerate(EXPECTED_LABELS):
        predicted_positive = int(y_pred[:, i].sum())
        result["per_label"][label] = {
            "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            "support": int(support[i]), "predicted_positive": predicted_positive,
            "average_precision": float(average_precision_score(y_true[:, i], probabilities[:, i])),
            "gold_prevalence": float(y_true[:, i].mean()),
            "predicted_prevalence": float(y_pred[:, i].mean()),
            "prevalence_delta": float(y_pred[:, i].mean() - y_true[:, i].mean()),
        }
    return result


def _load_source(root: Path, system: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    spec = SYSTEMS[system]
    summary = load_json(root / "reports" / "brighter_ptbr" / "experiments" / f"{spec['source']}_results.json")
    oof = pd.read_parquet(root / "artifacts" / "predictions" / f"{spec['source']}_oof.parquet")
    pool = construct_development_pool(root)
    pool = attach_folds(pool, persist_cv_folds(root, pool))
    validate_oof(oof, pool)
    if sha256_file(root / "artifacts" / "predictions" / f"{spec['source']}_oof.parquet") != summary["oof_sha256"]:
        raise RuntimeError(f"Source OOF hash mismatch for {system}")
    return summary, oof, pool


def _inner_prediction_path(root: Path, output: str, outer: int, inner: int) -> Path:
    return root / "artifacts" / "predictions" / "threshold_inner" / output / f"outer_{outer}_inner_{inner}.parquet"


def run_nested_system(root: Path, system: str) -> dict[str, Any]:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system}")
    spec = SYSTEMS[system]
    summary, source_oof, pool = _load_source(root, system)
    config = dict(summary["config"])
    all_outer, threshold_rows, costs = [], [], []
    started = time.perf_counter()
    for outer_fold in range(5):
        outer_train = pool.loc[pool.fold != outer_fold].copy()
        outer_validation = pool.loc[pool.fold == outer_fold].copy()
        assignments = persist_inner_folds(root, system, outer_fold, outer_train, outer_validation, seed=42)
        outer_train["inner_fold"] = outer_train.id.astype(str).map(assignments).astype(int)
        inner_predictions = []
        for inner_fold in range(3):
            inner_train = outer_train.loc[outer_train.inner_fold != inner_fold].copy()
            inner_validation = outer_train.loc[outer_train.inner_fold == inner_fold].copy()
            path = _inner_prediction_path(root, spec["output"], outer_fold, inner_fold)
            if path.exists():
                frame = pd.read_parquet(path)
                if frame.id.astype(str).tolist() != inner_validation.id.astype(str).tolist():
                    raise RuntimeError(f"Stale inner prediction IDs for {system}/{outer_fold}/{inner_fold}")
                cost_path = path.with_suffix(".metrics.json")
                if not cost_path.exists():
                    raise RuntimeError(f"Missing cached cost for {system}/{outer_fold}/{inner_fold}")
                cost = load_json(cost_path)
            else:
                inner_config = json.loads(json.dumps(config))
                inner_config["experiment_id"] = f"{system}_{spec['output'].upper()}_OUTER{outer_fold}_INNER{inner_fold}"
                weights = calculate_fold_class_weights(inner_train[list(EXPECTED_LABELS)].to_numpy(dtype=int))
                model_dir = root / "artifacts" / "models" / "threshold_inner" / spec["output"] / f"outer_{outer_fold}" / f"inner_{inner_fold}"
                probabilities, cost = run_transformer_fold(root, inner_config, inner_train, inner_validation, model_dir, weights)
                frame = pd.DataFrame({"id": inner_validation.id.astype(str).to_numpy(), "inner_fold": inner_fold})
                for i, label in enumerate(EXPECTED_LABELS):
                    frame[f"{label}_prob"] = probabilities[:, i].astype(np.float64)
                    frame[f"{label}_true"] = inner_validation[label].to_numpy(dtype=np.int64)
                _atomic_parquet(frame, path)
                write_json_atomic(path.with_suffix(".metrics.json"), cost)
            inner_predictions.append(frame)
            costs.append({"system": system, "outer_fold": outer_fold, "inner_fold": inner_fold, **cost})
        inner_oof = pd.concat(inner_predictions, ignore_index=True)
        if len(inner_oof) != len(outer_train) or inner_oof.id.duplicated().any() or set(inner_oof.id) != set(outer_train.id.astype(str)):
            raise RuntimeError(f"Incomplete inner OOF for {system} outer fold {outer_fold}")
        y_inner = inner_oof[[f"{x}_true" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
        p_inner = inner_oof[[f"{x}_prob" for x in EXPECTED_LABELS]].to_numpy(dtype=float)
        thresholds, calibration_f1 = search_per_label_thresholds(y_inner, p_inner)
        threshold_rows.append({"system": system, "outer_fold": outer_fold,
                               **{label: thresholds[i] for i, label in enumerate(EXPECTED_LABELS)}})
        outer = source_oof.loc[source_oof.fold == outer_fold].copy()
        if set(outer.id.astype(str)) != set(outer_validation.id.astype(str)):
            raise RuntimeError("Source outer OOF does not match isolated outer validation")
        for i, label in enumerate(EXPECTED_LABELS):
            outer[f"{label}_pred"] = (outer[f"{label}_prob"].to_numpy() >= thresholds[i]).astype(np.int8)
            outer[f"{label}_threshold"] = thresholds[i]
            outer[f"{label}_inner_f1"] = calibration_f1[i]
        all_outer.append(outer)
    thresholded = pd.concat(all_outer, ignore_index=True)
    thresholded = thresholded.set_index("id").loc[source_oof.id.astype(str)].reset_index()
    if len(thresholded) != DEV_POOL_SIZE or thresholded.id.duplicated().any():
        raise RuntimeError("Final thresholded OOF is incomplete")
    for label in EXPECTED_LABELS:
        if not np.array_equal(thresholded[f"{label}_prob"].to_numpy(), source_oof[f"{label}_prob"].to_numpy()):
            raise RuntimeError(f"Probabilities changed during thresholding for {label}")
    artifact = root / "artifacts" / "predictions" / f"{spec['output']}_oof.parquet"
    _atomic_parquet(thresholded, artifact)
    thresholds_frame = pd.DataFrame(threshold_rows)
    threshold_path = root / "reports" / "brighter_ptbr" / "experiments" / f"{spec['output']}_thresholds_by_fold.csv"
    _atomic_csv(thresholds_frame, threshold_path)
    cost_path = root / "reports" / "brighter_ptbr" / "experiments" / f"{spec['output']}_inner_cost.csv"
    _atomic_csv(pd.DataFrame(costs), cost_path)
    y = thresholded[[f"{x}_true" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
    p = thresholded[[f"{x}_prob" for x in EXPECTED_LABELS]].to_numpy(dtype=float)
    fixed = decision_metrics(y, (p >= .5).astype(int), p)
    calibrated = decision_metrics(y, thresholded[[f"{x}_pred" for x in EXPECTED_LABELS]].to_numpy(dtype=int), p)
    if fixed["macro_average_precision"] != calibrated["macro_average_precision"]:
        raise RuntimeError("Average Precision changed despite invariant probabilities")
    result = {
        "system": system, "source_system": spec["short"], "source_oof": summary["oof_path"],
        "thresholded_oof": artifact.relative_to(root).as_posix(), "thresholded_oof_sha256": sha256_file(artifact),
        "fixed_metrics": fixed, "thresholded_metrics": calibrated,
        "thresholds_by_fold": thresholds_frame.to_dict(orient="records"),
        "inner_training_seconds_total": float(sum(float(x["training_seconds"]) for x in costs)),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "dataset_hashes": development_input_hashes(root), "completed_timestamp_utc": utc_now_iso(),
        "protocol": {"outer_folds": 5, "inner_folds": 3, "inner_seed": 42, "grid": [0.01, 0.99, 0.01],
                     "tie_break": "largest_threshold", "objective": "independent_per_label_f1"},
        "test_policy": "SEALED — NOT LOADED OR EVALUATED",
    }
    write_json_atomic(root / "metadata" / "brighter_ptbr" / "experiments" / f"{spec['output']}_summary.json", result)
    return result


def estimate_t3_seconds(root: Path) -> dict[str, float]:
    """Scale observed B2 outer-fold cost by exact inner/outer training sizes."""

    summary, _, pool = _load_source(root, "T3")
    observed = float(summary["cost"]["training_seconds_total"])
    observed_example_epochs = sum(int((pool.fold != fold).sum()) * int(summary["config"]["training"]["epochs"]) for fold in range(5))
    inner_example_epochs = 0
    for outer in range(5):
        outer_train = pool.loc[pool.fold != outer].copy()
        assignments = inner_fold_assignments(outer_train, 42)
        fold_values = outer_train.id.astype(str).map(assignments)
        for inner in range(3):
            inner_example_epochs += int((fold_values != inner).sum()) * int(summary["config"]["training"]["epochs"])
    estimated = observed * inner_example_epochs / observed_example_epochs
    return {"observed_outer_seconds": observed, "estimated_inner_seconds": estimated,
            "estimated_inner_hours": estimated / 3600, "inner_models": 15}
