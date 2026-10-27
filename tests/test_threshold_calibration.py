from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from brighter.experiments import EXPECTED_LABELS, attach_folds, construct_development_pool, persist_cv_folds
from brighter.threshold_calibration import (
    SYSTEMS, decision_metrics, inner_fold_assignments, search_threshold,
)

ROOT = Path(__file__).resolve().parents[1]


def test_inner_folds_are_complete_disjoint_deterministic_and_outer_validation_excluded() -> None:
    pool = construct_development_pool(ROOT)
    pool = attach_folds(pool, persist_cv_folds(ROOT, pool))
    outer_train, outer_validation = pool[pool.fold != 0], pool[pool.fold == 0]
    first, second = inner_fold_assignments(outer_train, 42), inner_fold_assignments(outer_train, 42)
    assert first == second
    assert len(first) == len(outer_train) == len(set(first))
    assert set(first.values()) == {0, 1, 2}
    assert not set(first) & set(outer_validation.id.astype(str))
    for fold in range(3):
        validation = {x for x, value in first.items() if value == fold}
        training = set(first) - validation
        assert not training & validation
        assert training | validation == set(first)


def test_threshold_search_is_bounded_deterministic_and_largest_tie_wins() -> None:
    y = np.array([1, 0, 0, 0])
    p = np.array([.8, .2, .1, .05])
    first = search_threshold(y, p)
    assert first == search_threshold(y, p)
    assert .01 <= first[0] <= .99
    # Any threshold 0.21..0.80 yields perfect F1; the predeclared rule chooses 0.80.
    assert first == pytest.approx((.80, 1.0))


def test_ap_is_invariant_to_decision_threshold() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=(100, 6))
    p = rng.random((100, 6))
    low = decision_metrics(y, (p >= .2).astype(int), p)
    high = decision_metrics(y, (p >= .8).astype(int), p)
    assert low["macro_average_precision"] == high["macro_average_precision"]
    assert low["macro_average_precision"] == pytest.approx(average_precision_score(y, p, average="macro"))


@pytest.mark.parametrize("system", SYSTEMS)
def test_thresholded_artifact_integrity_when_completed(system: str) -> None:
    spec = SYSTEMS[system]
    source_path = ROOT / "artifacts" / "predictions" / f"{spec['source']}_oof.parquet"
    calibrated_path = ROOT / "artifacts" / "predictions" / f"{spec['output']}_oof.parquet"
    if not calibrated_path.exists():
        pytest.skip("nested calibration not completed yet")
    source, calibrated = pd.read_parquet(source_path), pd.read_parquet(calibrated_path)
    assert len(calibrated) == 2426
    assert calibrated.id.nunique() == 2426
    assert calibrated.id.tolist() == source.id.tolist()
    assert calibrated.fold.tolist() == source.fold.tolist()
    for label in EXPECTED_LABELS:
        assert np.array_equal(calibrated[f"{label}_prob"].to_numpy(), source[f"{label}_prob"].to_numpy())
        assert calibrated[f"{label}_threshold"].between(.01, .99).all()
        assert set(calibrated[f"{label}_pred"].unique()) <= {0, 1}


@pytest.mark.parametrize("system", SYSTEMS)
def test_fixed_point_five_reproduces_persisted_source_metrics(system: str) -> None:
    spec = SYSTEMS[system]
    source = pd.read_parquet(ROOT / "artifacts" / "predictions" / f"{spec['source']}_oof.parquet")
    summary_path = ROOT / "metadata" / "brighter_ptbr" / "experiments" / f"{spec['output']}_summary.json"
    if not summary_path.exists():
        pytest.skip("nested calibration not completed yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    y = source[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int)
    probabilities = source[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(dtype=float)
    reproduced = decision_metrics(y, (probabilities >= .5).astype(int), probabilities)
    for metric in ("macro_f1", "micro_f1", "macro_average_precision", "hamming_loss", "subset_accuracy"):
        assert reproduced[metric] == pytest.approx(summary["fixed_metrics"][metric], abs=1e-12)


def test_persisted_t3_inner_folds_exclude_outer_validation_and_are_complete() -> None:
    pool = construct_development_pool(ROOT)
    pool = attach_folds(pool, persist_cv_folds(ROOT, pool))
    metadata_dir = ROOT / "metadata" / "brighter_ptbr" / "experiments" / "threshold_inner_folds" / "t3"
    for outer_fold in range(5):
        payload = json.loads((metadata_dir / f"outer_{outer_fold}.json").read_text(encoding="utf-8"))
        assigned = {row["id"] for row in payload["assignments"]}
        expected_train = set(pool.loc[pool.fold != outer_fold, "id"].astype(str))
        expected_validation = set(pool.loc[pool.fold == outer_fold, "id"].astype(str))
        assert assigned == expected_train
        assert not assigned & expected_validation
        assert payload["outer_validation_excluded_from_assignments"] is True
        assert {row["inner_fold"] for row in payload["assignments"]} == {0, 1, 2}


def test_t3_smoke_manifest_records_real_forward_backward_and_six_logits() -> None:
    path = ROOT / "metadata" / "brighter_ptbr" / "experiments" / "threshold_t3_smoke.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["system"] == "T3"
    assert payload["source_scope"] == "outer_train_only"
    assert payload["output_shape"] == [1, 6]
    assert payload["forward"] == payload["backward"] == "ok"
    assert payload["peak_gpu_memory_bytes"] > 0
    assert payload["test_policy"] == "SEALED — NOT LOADED OR EVALUATED"
