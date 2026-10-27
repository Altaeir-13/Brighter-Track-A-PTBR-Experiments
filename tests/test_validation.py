from __future__ import annotations

import pandas as pd
import pytest

from brighter.validation import (
    EXPECTED_LABELS,
    SchemaError,
    analyze_dataset,
    calculate_num_labels,
    cross_split_overlaps,
    duplicated_ids,
    duplicated_texts,
    identify_label_columns,
    label_cardinality,
    validate_binary_labels,
)


def frame(ids=("a", "b", "c"), texts=("um", "dois", "três")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(ids),
            "text": list(texts),
            "text_raw": list(texts),
            "anger": [1, 0, 1],
            "disgust": [0, 0, 0],
            "fear": [0, 1, 0],
            "joy": [0, 0, 1],
            "sadness": [0, 0, 0],
            "surprise": [0, 0, 1],
            "emotions": [["anger"], ["fear"], ["anger", "joy", "surprise"]],
        }
    )


def test_identifies_expected_labels_in_canonical_order() -> None:
    assert identify_label_columns(frame().columns) == list(EXPECTED_LABELS)


def test_missing_required_label_raises() -> None:
    with pytest.raises(SchemaError, match="surprise"):
        identify_label_columns(frame().drop(columns="surprise").columns)


def test_binary_validation_reports_invalid_and_null_values() -> None:
    data = frame()
    data.loc[0, "anger"] = 2
    data.loc[1, "fear"] = None
    metrics = validate_binary_labels(data)
    assert metrics["anger"]["invalid_values"] == [2]
    assert metrics["anger"]["invalid_count"] == 1
    assert metrics["fear"]["null"] == 1


def test_duplicate_detection_counts_rows_beyond_first_occurrence() -> None:
    data = frame(ids=("a", "a", "c"), texts=("igual", "igual", "outro"))
    assert duplicated_ids(data, "id") == 1
    assert duplicated_texts(data, "text_raw") == 1


def test_cross_split_overlap_detects_ids_and_exact_texts() -> None:
    train = frame()
    dev = frame(ids=("a", "d", "e"), texts=("novo", "dois", "x"))
    overlap = cross_split_overlaps({"train": train, "dev": dev}, "id", "text_raw")
    assert overlap["train__dev"] == {"shared_ids": 1, "shared_texts": 1}


def test_num_labels_and_cardinality() -> None:
    data = frame()
    assert calculate_num_labels(data).tolist() == [1, 1, 3]
    assert label_cardinality(data) == pytest.approx(5 / 3)


def test_num_labels_rejects_invalid_input() -> None:
    data = frame()
    data.loc[0, "joy"] = 9
    with pytest.raises(ValueError, match="joy"):
        calculate_num_labels(data)


def test_analysis_fails_when_text_column_is_absent() -> None:
    data = frame().drop(columns=["text", "text_raw"])
    with pytest.raises(SchemaError, match="text column"):
        analyze_dataset({"train": data})


def test_analysis_fails_when_id_column_is_absent() -> None:
    data = frame().drop(columns="id")
    with pytest.raises(SchemaError, match="ID column"):
        analyze_dataset({"train": data})


def test_analysis_reports_zero_through_six_label_buckets() -> None:
    data = frame()
    result = analyze_dataset({"train": data})
    assert result["splits"]["train"]["num_labels"] == {
        "0": 0,
        "1": 2,
        "2": 0,
        "3": 1,
        "4": 0,
        "5": 0,
        "6": 0,
    }
