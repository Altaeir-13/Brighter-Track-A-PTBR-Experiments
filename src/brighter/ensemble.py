"""Leakage-safe T2/T3 probability ensembles using persisted nested OOF predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .experiments import DEV_POOL_SIZE, EXPECTED_LABELS
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .threshold_calibration import decision_metrics, search_per_label_thresholds

ALPHA_GRID = np.arange(21, dtype=np.float64) / 20.0
INNER_SLUGS = {
    "T2": "bertweetbr_asl_thresholded",
    "T3": "bertimbau_large_bce_thresholded",
}
OUTER_SLUGS = {
    "T2": "bertweetbr_asl_thresholded",
    "T3": "bertimbau_large_bce_thresholded",
}
OUTPUTS = {
    "EN1": "ensemble_t2_t3_equal_weight",
    "EN2": "ensemble_t2_t3_tuned_weight",
}


def fuse_probabilities(t2: np.ndarray, t3: np.ndarray, alpha_t2: float) -> np.ndarray:
    """Convex probability fusion; endpoints reproduce the source probabilities."""
    a, b = np.asarray(t2, dtype=float), np.asarray(t3, dtype=float)
    if a.shape != b.shape:
        raise ValueError("T2/T3 probability arrays must have identical shapes")
    if not 0.0 <= float(alpha_t2) <= 1.0:
        raise ValueError("alpha_t2 must be in [0,1]")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(a > 1) or np.any(b < 0) or np.any(b > 1):
        raise ValueError("source probabilities must be finite in [0,1]")
    if float(alpha_t2) == 1.0:
        return a.copy()
    if float(alpha_t2) == 0.0:
        return b.copy()
    return float(alpha_t2) * a + (1.0 - float(alpha_t2)) * b


def select_alpha(y_true: np.ndarray, t2: np.ndarray, t3: np.ndarray,
                 grid: np.ndarray = ALPHA_GRID) -> tuple[float, np.ndarray, float, pd.DataFrame]:
    """Select alpha by inner Macro-F1; ties prefer closest 0.5, then larger alpha."""
    candidates = np.asarray(grid, dtype=float)
    expected = np.arange(21, dtype=np.float64) / 20.0
    if candidates.shape != expected.shape or not np.allclose(candidates, expected, rtol=0, atol=1e-12):
        raise ValueError("alpha grid must be exactly 0.00..1.00 in steps of 0.05")
    rows: list[dict[str, Any]] = []
    for alpha in candidates:
        probabilities = fuse_probabilities(t2, t3, float(alpha))
        thresholds, per_label_f1 = search_per_label_thresholds(y_true, probabilities)
        rows.append({"alpha_T2": float(alpha), "weight_T3": float(1-alpha),
                     "inner_macro_f1": float(per_label_f1.mean()),
                     **{label: float(thresholds[i]) for i, label in enumerate(EXPECTED_LABELS)}})
    trace = pd.DataFrame(rows)
    chosen = choose_alpha_row(trace)
    best = float(chosen.inner_macro_f1)
    thresholds = chosen[list(EXPECTED_LABELS)].to_numpy(dtype=float)
    return float(chosen.alpha_T2), thresholds, best, trace


def choose_alpha_row(trace: pd.DataFrame) -> pd.Series:
    """Apply the predeclared alpha tie-break to an evaluated search trace."""
    best = float(trace.inner_macro_f1.max())
    tied = trace[np.isclose(trace.inner_macro_f1, best, rtol=0.0, atol=1e-12)].copy()
    tied["distance_to_half"] = (tied.alpha_T2 - .5).abs()
    return tied.sort_values(["distance_to_half", "alpha_T2"], ascending=[True, False]).iloc[0]


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(tmp, path)


def _load_inner_system(root: Path, system: str, outer_fold: int) -> pd.DataFrame:
    slug = INNER_SLUGS[system]
    directory = root / "artifacts" / "predictions" / "threshold_inner" / slug
    parts: list[pd.DataFrame] = []
    metadata = load_json(root / "metadata" / "brighter_ptbr" / "experiments" / "threshold_inner_folds" / system.lower() / f"outer_{outer_fold}.json")
    assignments = {str(row["id"]): int(row["inner_fold"]) for row in metadata["assignments"]}
    if not metadata.get("outer_validation_excluded_from_assignments") or metadata.get("calibration_scope") != "outer_train_only":
        raise RuntimeError(f"Invalid leakage metadata for {system}/outer {outer_fold}")
    for inner_fold in range(3):
        path = directory / f"outer_{outer_fold}_inner_{inner_fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing persisted inner OOF: {path}")
        frame = pd.read_parquet(path)
        if frame.id.duplicated().any() or not (frame.inner_fold == inner_fold).all():
            raise RuntimeError(f"Invalid inner OOF structure for {system}/{outer_fold}/{inner_fold}")
        if any(assignments.get(str(identifier)) != inner_fold for identifier in frame.id):
            raise RuntimeError(f"In-sample or stale prediction detected for {system}/{outer_fold}/{inner_fold}")
        parts.append(frame)
    output = pd.concat(parts, ignore_index=True)
    if output.id.duplicated().any() or set(output.id.astype(str)) != set(assignments):
        raise RuntimeError(f"Incomplete inner OOF for {system}/outer {outer_fold}")
    return output


def load_paired_inner_oof(root: Path, outer_fold: int) -> pd.DataFrame:
    """Pair leakage-safe T2/T3 inner-OOF probabilities strictly by ID."""
    t2, t3 = _load_inner_system(root, "T2", outer_fold), _load_inner_system(root, "T3", outer_fold)
    true_cols = [f"{label}_true" for label in EXPECTED_LABELS]
    pcols = [f"{label}_prob" for label in EXPECTED_LABELS]
    left = t2[["id", *true_cols, *pcols]].rename(columns={x: x.replace("_prob", "_t2_prob") for x in pcols})
    right = t3[["id", *true_cols, *pcols]].rename(columns={x: x.replace("_prob", "_t3_prob") for x in pcols})
    paired = left.merge(right, on="id", how="inner", validate="one_to_one", suffixes=("_t2_true", "_t3_true"))
    if len(paired) != len(t2) or len(paired) != len(t3):
        raise RuntimeError(f"T2/T3 inner ID mismatch for outer {outer_fold}")
    for label in EXPECTED_LABELS:
        a, b = paired.pop(f"{label}_true_t2_true"), paired.pop(f"{label}_true_t3_true")
        if not np.array_equal(a.to_numpy(), b.to_numpy()):
            raise RuntimeError(f"T2/T3 truth mismatch for {label}/outer {outer_fold}")
        paired[f"{label}_true"] = a.to_numpy(dtype=np.int8)
    probability_cols = [f"{label}_{system}_prob" for system in ("t2", "t3") for label in EXPECTED_LABELS]
    values = paired[probability_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise RuntimeError("Invalid paired inner probabilities")
    return paired


def _load_outer_pair(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    t2 = pd.read_parquet(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T2']}_oof.parquet")
    t3 = pd.read_parquet(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T3']}_oof.parquet")
    if len(t2) != DEV_POOL_SIZE or t2.id.nunique() != DEV_POOL_SIZE:
        raise RuntimeError("T2 outer OOF is incomplete")
    if t2.id.tolist() != t3.id.tolist() or t2.fold.tolist() != t3.fold.tolist():
        raise RuntimeError("T2/T3 outer OOF ID/fold mismatch")
    for label in EXPECTED_LABELS:
        if not np.array_equal(t2[f"{label}_true"].to_numpy(), t3[f"{label}_true"].to_numpy()):
            raise RuntimeError(f"Outer truth mismatch for {label}")
    return t2, t3


def run_ensemble(root: Path) -> dict[str, Any]:
    """Run EN1 and EN2 from persisted probabilities; never trains or loads a model."""
    t2, t3 = _load_outer_pair(root)
    source_hashes = {
        "t2": sha256_file(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T2']}_oof.parquet"),
        "t3": sha256_file(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T3']}_oof.parquet"),
    }
    outputs: dict[str, list[pd.DataFrame]] = {"EN1": [], "EN2": []}
    weights, thresholds_rows, traces, fold_metrics = [], [], [], []
    for outer_fold in range(5):
        paired = load_paired_inner_oof(root, outer_fold)
        y_inner = paired[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int)
        p2_inner = paired[[f"{label}_t2_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
        p3_inner = paired[[f"{label}_t3_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
        alpha, en2_thresholds, en2_inner_f1, trace = select_alpha(y_inner, p2_inner, p3_inner)
        trace.insert(0, "outer_fold", outer_fold); traces.append(trace)
        en1_prob_inner = fuse_probabilities(p2_inner, p3_inner, .5)
        en1_thresholds, en1_per_label = search_per_label_thresholds(y_inner, en1_prob_inner)
        settings = {
            "EN1": (.5, en1_thresholds, float(en1_per_label.mean())),
            "EN2": (alpha, en2_thresholds, en2_inner_f1),
        }
        weights.append({"outer_fold": outer_fold, "alpha_T2": alpha, "weight_T3": 1-alpha,
                        "inner_macro_f1": en2_inner_f1})
        outer_mask = t2.fold.to_numpy() == outer_fold
        outer2, outer3 = t2.loc[outer_mask].copy(), t3.loc[outer_mask].copy()
        p2_outer = outer2[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
        p3_outer = outer3[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
        y_outer = outer2[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int)
        for system, (chosen_alpha, thresholds, inner_score) in settings.items():
            probabilities = fuse_probabilities(p2_outer, p3_outer, chosen_alpha)
            predictions = (probabilities >= thresholds).astype(np.int8)
            frame = outer2[["id", "fold", *[f"{label}_true" for label in EXPECTED_LABELS]]].copy()
            frame["alpha_T2"] = chosen_alpha; frame["weight_T3"] = 1-chosen_alpha
            for i, label in enumerate(EXPECTED_LABELS):
                frame[f"{label}_prob"] = probabilities[:, i]
                frame[f"{label}_pred"] = predictions[:, i]
                frame[f"{label}_threshold"] = thresholds[i]
            outputs[system].append(frame)
            thresholds_rows.append({"system": system, "outer_fold": outer_fold, "alpha": chosen_alpha,
                                    **{label: thresholds[i] for i, label in enumerate(EXPECTED_LABELS)}})
            metrics = decision_metrics(y_outer, predictions, probabilities)
            fold_metrics.append({"system": system, "outer_fold": outer_fold, "alpha_T2": chosen_alpha,
                                 "inner_macro_f1": inner_score,
                                 **{key: metrics[key] for key in ("macro_f1", "micro_f1", "macro_precision", "macro_recall", "hamming_loss", "jaccard_samples", "subset_accuracy", "macro_average_precision")}})
    final: dict[str, Any] = {}
    for system, parts in outputs.items():
        frame = pd.concat(parts, ignore_index=True).set_index("id").loc[t2.id.astype(str)].reset_index()
        if len(frame) != DEV_POOL_SIZE or frame.id.nunique() != DEV_POOL_SIZE or frame.fold.tolist() != t2.fold.tolist():
            raise RuntimeError(f"Invalid final OOF for {system}")
        path = root / "artifacts" / "predictions" / f"{OUTPUTS[system]}_oof.parquet"
        _atomic_parquet(frame, path)
        y = frame[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int)
        p = frame[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
        pred = frame[[f"{label}_pred" for label in EXPECTED_LABELS]].to_numpy(dtype=int)
        final[system] = {"artifact": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                         "metrics": decision_metrics(y, pred, p)}
    report_dir = root / "reports" / "brighter_ptbr" / "experiments"
    _atomic_csv(pd.DataFrame(weights), report_dir / "ensemble_alpha_by_fold.csv")
    _atomic_csv(pd.DataFrame(thresholds_rows), report_dir / "ensemble_thresholds_by_fold.csv")
    _atomic_csv(pd.DataFrame(fold_metrics), report_dir / "ensemble_fold_metrics.csv")
    _atomic_csv(pd.concat(traces, ignore_index=True), report_dir / "ensemble_alpha_search_trace.csv")
    if source_hashes != {
        "t2": sha256_file(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T2']}_oof.parquet"),
        "t3": sha256_file(root / "artifacts" / "predictions" / f"{OUTER_SLUGS['T3']}_oof.parquet"),
    }:
        raise RuntimeError("Source outer OOF was modified")
    audit = {"paired_inner_complete": True, "outer_validation_excluded": True,
             "inner_predictions_out_of_sample": True, "training_performed": False,
             "source_oof_hashes": source_hashes, "test_policy": "SEALED — NOT LOADED OR EVALUATED"}
    result = {"protocol": {"outer_folds": 5, "inner_folds": 3, "alpha_grid": [0.0, 1.0, 0.05],
                           "threshold_grid": [0.01, 0.99, 0.01],
                           "alpha_tie_break": "closest_to_0.5_then_larger_alpha_T2",
                           "threshold_tie_break": "largest_threshold",
                           "selection_objective": "inner_oof_macro_f1"},
              "audit": audit, "systems": final, "alpha_by_fold": weights,
              "thresholds_by_fold": thresholds_rows, "completed_timestamp_utc": utc_now_iso()}
    write_json_atomic(root / "metadata" / "brighter_ptbr" / "experiments" / "ensemble_run.json", result)
    return result
