from __future__ import annotations

from pathlib import Path

from brighter.final_reporting import EXPECTED_OUTPUT_HASHES, validate_preserved_outputs
from brighter.generalization_analysis import (
    ALLOWED_PARQUETS, SUMMARY_PATH, THRESHOLDS, run_analysis,
)
from brighter.manifest import load_json, sha256_file

ROOT=Path(__file__).resolve().parents[1]


def test_analysis_has_no_model_training_inference_or_raw_test_access() -> None:
    source=(ROOT/"src/brighter/generalization_analysis.py").read_text(encoding="utf-8").lower()
    for forbidden in ("load_dataset(","data/curated", "transformers", "import torch", "automodelfor", "tokenizer", "optimizer", "backward(", "search_per_label_thresholds", "optimize_threshold"):
        assert forbidden not in source


def test_only_frozen_prediction_artifacts_are_allowed() -> None:
    assert len(ALLOWED_PARQUETS)==9
    assert all(str(path).startswith("artifacts\\predictions") or path.as_posix().startswith("artifacts/predictions") for path in ALLOWED_PARQUETS.values())
    assert all("test.parquet" not in path.name for path in ALLOWED_PARQUETS.values())


def test_frozen_thresholds_are_read_only_constants() -> None:
    assert THRESHOLDS["EN2"]=={"anger":.67,"disgust":.36,"fear":.43,"joy":.65,"sadness":.58,"surprise":.49}
    assert THRESHOLDS["T2"]=={"anger":.66,"disgust":.53,"fear":.52,"joy":.75,"sadness":.65,"surprise":.57}


def test_existing_final_hashes_and_metrics_unchanged() -> None:
    assert validate_preserved_outputs(ROOT)==EXPECTED_OUTPUT_HASHES
    result_path=ROOT/"reports/brighter_ptbr/final/final_test_results.json"
    summary=load_json(ROOT/SUMMARY_PATH)
    assert sha256_file(result_path)==summary["final_metrics_sha256"]
    result=load_json(result_path)
    assert result["primary"]["metrics"]["macro_f1"]==0.2851799354682137
    assert result["secondary"]["metrics"]["macro_f1"]==0.15691169825594717


def test_analysis_summary_declares_read_only_scope() -> None:
    summary=load_json(ROOT/SUMMARY_PATH)
    for key in ("models_loaded","raw_test_loaded","new_inference","new_training","threshold_tuning","parameters_modified"):
        assert summary[key] is False
    assert summary["read_only"] is True and summary["search_space"]=="CLOSED"


def test_reports_tables_and_figures_exist() -> None:
    assert (ROOT/"reports/brighter_ptbr/final/generalization_gap_analysis.md").exists()
    for name in ("main","per_label","cardinality","prevalence","probability_shift","threshold_crossing","calibration"):
        assert (ROOT/f"reports/brighter_ptbr/final/generalization_{name}.csv").exists()
        assert (ROOT/f"reports/brighter_ptbr/final/latex/generalization_{name}.tex").exists()
    figures=list((ROOT/"figures/brighter_ptbr/final/generalization_gap").glob("*"))
    assert len([path for path in figures if path.suffix==".png"])==9
    assert len([path for path in figures if path.suffix==".pdf"])==9


def test_analysis_is_deterministic_and_preserves_state_and_outputs() -> None:
    tracked=[ROOT/SUMMARY_PATH,ROOT/"reports/brighter_ptbr/final/generalization_gap_analysis.md",ROOT/"metadata/brighter_ptbr/final/experiment_state.json",*[(ROOT/path) for path in EXPECTED_OUTPUT_HASHES]]
    before={path:sha256_file(path) for path in tracked}
    run_analysis(ROOT)
    after={path:sha256_file(path) for path in tracked}
    assert after==before
    state=load_json(ROOT/"metadata/brighter_ptbr/final/experiment_state.json")
    assert state["state"]=="TEST_EVALUATED" and state["further_tuning_allowed"] is False
