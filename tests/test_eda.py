from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from brighter.eda import (
    calculate_num_labels,
    conditional_probability_matrix,
    imbalance_ratios,
    jaccard_label_matrix,
    label_cardinality,
    label_density,
    labelset_strings,
    lift_matrix,
    mean_imbalance_ratio,
    scumble_metrics,
    split_statistics,
    truncation_statistics,
)
from brighter.validation import EXPECTED_LABELS


def two_label_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 0],
            [1, 1],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
        dtype=int,
    )


def test_num_labels_cardinality_density_and_labelsets() -> None:
    values = two_label_matrix()
    labels = ("a", "b")
    assert calculate_num_labels(values, labels).tolist() == [1, 2, 1, 1, 0]
    assert label_cardinality(values, labels) == pytest.approx(1.0)
    assert label_density(values, labels) == pytest.approx(0.5)
    assert label_cardinality(values, labels, emotional_only=True) == pytest.approx(1.25)
    assert label_density(values, labels, emotional_only=True) == pytest.approx(0.625)
    assert labelset_strings(values, labels).tolist() == ["a", "a+b", "a", "b", "none"]


def test_conditional_probability_is_directional() -> None:
    values = np.array([[1, 1], [1, 0], [1, 0], [0, 1]], dtype=int)
    matrix = conditional_probability_matrix(values, ("a", "b"))
    assert matrix.loc["a", "b"] == pytest.approx(1 / 3)
    assert matrix.loc["b", "a"] == pytest.approx(1 / 2)
    assert matrix.loc["a", "a"] == pytest.approx(1.0)


def test_jaccard_and_lift_have_expected_values() -> None:
    values = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=int)
    jaccard = jaccard_label_matrix(values, ("a", "b"))
    lift = lift_matrix(values, ("a", "b"))
    assert jaccard.loc["a", "b"] == pytest.approx(1 / 3)
    assert jaccard.loc["a", "a"] == pytest.approx(1.0)
    assert lift.loc["a", "b"] == pytest.approx(1.0)
    assert lift.loc["a", "a"] == pytest.approx(2.0)


def test_irlbl_and_mean_ir() -> None:
    ratios = imbalance_ratios(two_label_matrix(), ("a", "b"))
    assert ratios == {"a": pytest.approx(1.0), "b": pytest.approx(1.5)}
    assert mean_imbalance_ratio(ratios) == pytest.approx(1.25)


def test_scumble_matches_arithmetic_geometric_mean_definition() -> None:
    metrics = scumble_metrics(two_label_matrix(), ("a", "b"))
    joint_score = 1.0 - math.sqrt(1.5) / 1.25
    assert metrics["instances_used"] == 4
    assert metrics["zero_label_instances_excluded"] == 1
    assert metrics["scumble"] == pytest.approx(joint_score / 4)
    assert metrics["scumble_lbl"]["a"] == pytest.approx(joint_score / 3)
    assert metrics["scumble_lbl"]["b"] == pytest.approx(joint_score / 2)


def test_scumble_is_zero_when_active_labels_have_equal_imbalance() -> None:
    values = np.array([[1, 1], [1, 0], [0, 1]], dtype=int)
    metrics = scumble_metrics(values, ("a", "b"))
    assert metrics["scumble"] == pytest.approx(0.0)
    assert metrics["scumble_lbl"] == {"a": pytest.approx(0.0), "b": pytest.approx(0.0)}


def test_zero_single_and_multilabel_detection() -> None:
    rows = []
    for active in ((), ("anger",), ("fear", "sadness")):
        row = {label: int(label in active) for label in EXPECTED_LABELS}
        row.update({"id": str(len(rows)), "text_raw": "texto " + str(len(rows))})
        rows.append(row)
    stats = split_statistics(pd.DataFrame(rows))
    assert stats["zero_label_examples"] == 1
    assert stats["single_label_examples"] == 1
    assert stats["multi_label_examples"] == 1
    assert stats["num_labels_distribution"]["2"] == 1
    assert stats["zero_label_percentage"] == pytest.approx(100 / 3)


def test_truncation_statistics_use_strictly_greater_than_max_length() -> None:
    result = truncation_statistics([10, 64, 65, 128, 129], (64, 128, 256))
    assert result["64"] == {
        "truncated_examples": 3,
        "truncated_percentage": pytest.approx(60.0),
    }
    assert result["128"]["truncated_examples"] == 1
    assert result["128"]["truncated_percentage"] == pytest.approx(20.0)
    assert result["256"]["truncated_examples"] == 0

