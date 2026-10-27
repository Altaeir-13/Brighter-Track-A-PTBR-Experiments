from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brighter.experiments import (
    DEV_POOL_SIZE,
    EXPECTED_LABELS,
    attach_folds,
    calculate_metrics,
    construct_development_pool,
    load_config,
    persist_cv_folds,
    run_experiment,
    threshold_probabilities,
    validate_folds,
    validate_oof,
    zero_label_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_development_pool_contains_only_train_and_dev() -> None:
    pool = construct_development_pool(PROJECT_ROOT)
    assert len(pool) == 2426
    assert set(pool["source_split"]) == {"train", "dev"}
    assert not pool["id"].str.contains("_test_").any()


def test_persisted_folds_are_complete_disjoint_and_stratified() -> None:
    pool = construct_development_pool(PROJECT_ROOT)
    assignments = persist_cv_folds(PROJECT_ROOT, pool)
    with_folds = attach_folds(pool, assignments)
    rows = validate_folds(with_folds)
    assert len(assignments) == DEV_POOL_SIZE
    assert set(assignments.values()) == set(range(5))
    assert sorted(row["examples"] for row in rows) == [485, 485, 485, 485, 486]
    # Rare disgust is preserved exactly across all validation folds.
    assert [row["disgust_positive"] for row in rows] == [16] * 5


def test_thresholding_is_fixed_inclusive_and_rejects_invalid_values() -> None:
    probabilities = np.array([[0.49, 0.5, 1.0], [0.0, 0.51, 0.2]])
    assert threshold_probabilities(probabilities).tolist() == [[0, 1, 1], [0, 1, 0]]
    with pytest.raises(ValueError):
        threshold_probabilities(np.array([[1.01]]))


def test_metrics_have_known_values() -> None:
    y_true = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    probabilities = np.array([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1], [0.4, 0.8, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1, 0.1, 0.1], [0.7, 0.4, 0.1, 0.1, 0.1, 0.1]])
    result = calculate_metrics(y_true, probabilities)
    assert result["macro_f1"] == pytest.approx((1.0 + 2 / 3) / 6)
    assert result["micro_f1"] == pytest.approx(6 / 7)
    assert result["hamming_loss"] == pytest.approx(1 / 24)
    assert result["subset_accuracy"] == pytest.approx(3 / 4)
    assert result["per_label"]["anger"]["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_zero_label_metrics_known_values() -> None:
    y_true = np.array([[0, 0], [0, 0], [1, 0]])
    y_pred = np.array([[0, 0], [1, 0], [1, 0]])
    result = zero_label_metrics(y_true, y_pred)
    assert result == {"support": 2, "exact_neutral_accuracy": 0.5, "neutral_false_positive_rate": 0.5}


def _synthetic_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"id_{index:04d}" for index in range(DEV_POOL_SIZE)]
    pool = pd.DataFrame({"id": ids, "fold": np.arange(DEV_POOL_SIZE) % 5})
    oof = pd.DataFrame({"id": ids, "fold": pool["fold"]})
    for label in EXPECTED_LABELS:
        oof[f"{label}_prob"] = 0.5
        oof[f"{label}_true"] = 0
    return pool, oof


def test_oof_completeness_order_and_probability_ranges() -> None:
    pool, oof = _synthetic_oof()
    validate_oof(oof, pool)
    assert oof["id"].tolist() == pool["id"].tolist()
    with pytest.raises(RuntimeError):
        validate_oof(oof.iloc[:-1], pool)
    invalid = oof.copy()
    invalid.loc[0, "anger_prob"] = -0.01
    with pytest.raises(RuntimeError):
        validate_oof(invalid, pool)


def test_config_loading_and_prohibited_imbalance(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "baseline_tfidf_logreg.yaml")
    assert config["seed"] == 42
    assert config["threshold"] == 0.5
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment_id: bad\nmodel_type: x\nseed: 42\nfolds: 5\nthreshold: 0.5\nimbalance_method: oversampling\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prohibited"):
        load_config(bad)


def test_final_test_guard_fails_before_data_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sealed"):
        run_experiment(tmp_path, Path("not-read.yaml"), final_test=True)
