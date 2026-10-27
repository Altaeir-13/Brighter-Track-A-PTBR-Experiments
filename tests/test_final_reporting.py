from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from brighter.final_reporting import (
    EXPECTED_MACRO, EXPECTED_OUTPUT_HASHES, configure_matplotlib,
    generate_final_figures, generate_latex, generate_markdown_report,
    load_frozen_results, update_registry, validate_completed_reporting,
    validate_preserved_outputs,
)
from brighter.manifest import load_json, sha256_file

ROOT=Path(__file__).resolve().parents[1]


def test_recovery_source_cannot_load_models_test_or_run_inference() -> None:
    source=(ROOT/"src/brighter/final_reporting.py").read_text(encoding="utf-8").lower()
    for forbidden in ("load_dataset(", "test.parquet", "transformers", "import torch", "_predict(", "read_parquet("):
        assert forbidden not in source


def test_recovery_script_requires_reporting_only() -> None:
    source=(ROOT/"scripts/recover_final_reporting.py").read_text(encoding="utf-8")
    assert '"--reporting-only"' in source and "required=True" in source
    assert "run_final_test" not in source


def test_preserved_probability_prediction_and_metric_hashes() -> None:
    assert validate_preserved_outputs(ROOT)==EXPECTED_OUTPUT_HASHES
    result=load_frozen_results(ROOT)
    assert result["primary"]["metrics"]["macro_f1"]==EXPECTED_MACRO["EN2"]
    assert result["secondary"]["metrics"]["macro_f1"]==EXPECTED_MACRO["T2_STANDALONE"]


def test_matplotlib_uses_project_local_cache_and_agg(tmp_path: Path) -> None:
    cache=configure_matplotlib(tmp_path)
    assert cache==tmp_path/".cache/matplotlib"
    assert os.environ["MPLCONFIGDIR"]==str(cache.resolve())
    assert os.environ["MPLBACKEND"]=="Agg"


def test_reporting_generators_use_frozen_json_only(tmp_path: Path) -> None:
    result=load_frozen_results(ROOT)
    figures=generate_final_figures(tmp_path,result)
    latex=generate_latex(tmp_path,result)
    report=generate_markdown_report(tmp_path,result)
    assert len([path for path in figures if path.suffix==".png"])==4
    assert len([path for path in figures if path.suffix==".pdf"])==4
    assert len(latex)==4 and report.exists()
    assert "0.2852" in report.read_text(encoding="utf-8")


def test_registry_update_has_no_t3_standalone_metric(tmp_path: Path) -> None:
    directory=tmp_path/"reports/brighter_ptbr/experiments"; directory.mkdir(parents=True)
    path=directory/"experiment_registry.csv"
    pd.DataFrame({"experiment_id":["FINAL_EN2","FINAL_T2_BERTWEETBR_ASL","FINAL_T3_BERTIMBAU_LARGE_BCE"],"status":["x"]*3,"test_policy":["x"]*3}).to_csv(path,index=False)
    update_registry(tmp_path,load_frozen_results(ROOT)); frame=pd.read_csv(path).set_index("experiment_id")
    assert frame.loc["FINAL_EN2","status"]=="TEST_EVALUATED"
    assert frame.loc["FINAL_T2_BERTWEETBR_ASL","status"]=="TEST_EVALUATED"
    assert frame.loc["FINAL_T3_BERTIMBAU_LARGE_BCE","status"]=="FINAL_EN2_COMPONENT"
    assert pd.isna(frame.loc["FINAL_T3_BERTIMBAU_LARGE_BCE","test_macro_f1"])


def test_completed_recovery_when_final_state_exists() -> None:
    state=load_json(ROOT/"metadata/brighter_ptbr/final/experiment_state.json")
    if state["state"]!="TEST_EVALUATED": pytest.skip("reporting recovery not finalized yet")
    validate_completed_reporting(ROOT)
    assert state["test_accessed"] is True and state["reporting_completed"] is True
