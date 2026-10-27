from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from brighter.ensemble import fuse_probabilities
from brighter.experiments import EXPECTED_LABELS, construct_development_pool
from brighter.final_test import (
    EN2_THRESHOLDS, EXPECTED_ALPHA, EXPECTED_MODEL_HASHES, EXPECTED_T3_WEIGHT,
    T2_THRESHOLDS, assert_final_test_authorized, fuse_en2, preflight,
    transition_state, validate_authorization, validate_track_a_test,
    _write_figures, _write_latex, _write_report,
)
from brighter.manifest import write_json_atomic
from brighter.finalization import FINAL_SPECS, audit_selection_artifacts
from brighter.manifest import load_json, sha256_file

ROOT=Path(__file__).resolve().parents[1]


def test_global_oof_pairing_and_ranges() -> None:
    audit=audit_selection_artifacts(ROOT)
    assert audit["rows"]==audit["unique_ids"]==2426
    assert audit["ordered_ids_match"] and audit["labels_match"] and audit["outer_folds_match"]
    assert audit["test_content_read"] is False


def test_deployment_config_schema_weights_thresholds_and_labels() -> None:
    config=yaml.safe_load((ROOT/"configs"/"final_en2.yaml").read_text(encoding="utf-8"))
    assert config["system"]=="EN2" and config["role"]=="PRIMARY" and config["frozen"] is True
    assert 0<=config["alpha_t2"]<=1 and 0<=config["weight_t3"]<=1
    assert config["alpha_t2"]+config["weight_t3"]==pytest.approx(1)
    assert config["label_order"]==list(EXPECTED_LABELS)
    assert list(config["thresholds"])==list(EXPECTED_LABELS)
    assert all(.01<=x<=.99 for x in config["thresholds"].values())
    assert config["preprocessing"]=="none"


def test_t2_standalone_config_and_thresholds_are_frozen() -> None:
    config=yaml.safe_load((ROOT/"configs"/"final_t2_standalone.yaml").read_text(encoding="utf-8"))
    assert config["role"]=="SECONDARY_PRE_REGISTERED"
    assert config["label_order"]==list(EXPECTED_LABELS)
    assert all(.01<=x<=.99 for x in config["thresholds"].values())


def test_final_training_pool_is_exact_train_plus_dev_without_test_ids() -> None:
    pool=construct_development_pool(ROOT)
    assert len(pool)==pool.id.nunique()==2426
    assert pool.source_split.value_counts().to_dict()=={"train":2226,"dev":200}
    assert not pool.id.astype(str).str.contains("_test_").any()


def test_ensemble_arithmetic_uses_frozen_alpha() -> None:
    decision=load_json(ROOT/"metadata"/"brighter_ptbr"/"final"/"deployment_decision_parameters.json")
    t2=np.array([[.1,.2,.3,.4,.5,.6]]); t3=np.array([[.6,.5,.4,.3,.2,.1]])
    fused=fuse_probabilities(t2,t3,decision["alpha_t2"])
    assert np.allclose(fused,decision["alpha_t2"]*t2+decision["weight_t3"]*t3)


def test_final_test_is_blocked_and_has_no_cli_overrides() -> None:
    with pytest.raises(RuntimeError,match="AUTHORIZED|ALREADY EVALUATED"):
        assert_final_test_authorized(ROOT)
    source=(ROOT/"scripts"/"run_final_test.py").read_text(encoding="utf-8")
    assert "argparse" not in source
    for forbidden in ("--alpha","--threshold","--model","--loss"):
        assert forbidden not in source


@pytest.mark.parametrize("system",("T2","T3"))
def test_final_model_manifest_and_hashes_when_completed(system:str) -> None:
    directory=ROOT/"artifacts"/"models"/"final"/FINAL_SPECS[system]["directory"]
    manifest_path=directory/"final_training_manifest.json"
    if not manifest_path.exists(): pytest.skip("final model training not complete")
    manifest=load_json(manifest_path)
    assert manifest["completed"] and manifest["development_examples"]==2426
    assert manifest["source_split_counts"]=={"train":2226,"dev":200}
    assert manifest["test_ids_loaded"] is False and manifest["label_order"]==list(EXPECTED_LABELS)
    assert manifest["epochs"]==4 and manifest["seed"]==42
    assert manifest["parameters"]>0 and manifest["peak_vram_bytes"]>0
    assert all(sha256_file(directory/name)==digest for name,digest in manifest["file_hashes"].items())


def test_dry_run_six_logits_when_completed() -> None:
    path=ROOT/"metadata"/"brighter_ptbr"/"final"/"dry_run_validation.json"
    if not path.exists(): pytest.skip("final dry run not complete")
    payload=load_json(path)
    assert payload["official_test_used"] is False
    assert payload["t2_shape"]==payload["t3_shape"]==payload["ensemble_shape"]==payload["predictions_shape"]==[2,6]


def test_final_manifest_and_frozen_state_when_completed() -> None:
    path=ROOT/"metadata"/"brighter_ptbr"/"final"/"final_experiment_manifest.json"
    if not path.exists(): pytest.skip("freeze manifest not complete")
    manifest=load_json(path); state=load_json(ROOT/"metadata"/"brighter_ptbr"/"final"/"experiment_state.json")
    assert manifest["experiment_status"]=="FROZEN_PRE_TEST"
    assert state["state"] in {"FROZEN_PRE_TEST","FINAL_TEST_AUTHORIZED","FINAL_TEST_RUNNING","FINAL_TEST_FAILED_AFTER_ACCESS","TEST_EVALUATED"}
    assert manifest["test_accessed"] is False
    assert manifest["primary"]=="EN2" and manifest["secondary"]=="T2_STANDALONE"
    assert manifest["label_order"]==list(EXPECTED_LABELS)
    assert all(item["match"] for stage in manifest["dataset"]["hashes"].values() for item in stage.values())


def test_final_preflight_is_sealed_and_hashes_match() -> None:
    state=load_json(ROOT/"metadata/brighter_ptbr/final/experiment_state.json")
    if state["state"]!="FROZEN_PRE_TEST": pytest.skip("one-time final evaluation has already advanced state")
    audit=preflight(ROOT)
    assert audit["test_content_read"] is False
    assert audit["model_hashes"]==EXPECTED_MODEL_HASHES
    assert audit["label_order"]==list(EXPECTED_LABELS)


def test_authorization_schema() -> None:
    validate_authorization({
        "authorized":True,"scope":"FINAL_TEST_ONCE","primary_system":"EN2",
        "secondary_system":"T2_STANDALONE","search_space_closed":True,
        "config_changes_after_test":False,"timestamp_utc":"fixture",
        "config_hashes":{},"model_hashes":{},"manifest_sha256":"fixture",
        "dataset_revision":"fixture",
    })
    with pytest.raises(RuntimeError,match="scope"):
        validate_authorization({"authorized":True,"scope":"REPEAT"})


def test_state_transition_is_atomic_and_cannot_repeat(tmp_path: Path) -> None:
    path=tmp_path/"metadata/brighter_ptbr/final/experiment_state.json"
    write_json_atomic(path,{"state":"FROZEN_PRE_TEST","test_accessed":False})
    transition_state(tmp_path,"FROZEN_PRE_TEST","FINAL_TEST_AUTHORIZED",test_accessed=False)
    assert load_json(path)["state"]=="FINAL_TEST_AUTHORIZED"
    with pytest.raises(RuntimeError,match="Invalid final-test state transition"):
        transition_state(tmp_path,"FROZEN_PRE_TEST","FINAL_TEST_AUTHORIZED",test_accessed=False)


def test_track_a_test_validation_exactly_2226_unique_ids() -> None:
    frame=pd.DataFrame({"id":[f"ptbr_test_track_a_{i:05d}" for i in range(2226)],"text_raw":["fixture"]*2226})
    for label in EXPECTED_LABELS: frame[label]=0
    validate_track_a_test(frame)
    bad=frame.copy(); bad.loc[0,"id"]="ptbr_test_track_c_00000"
    with pytest.raises(RuntimeError,match="Track-A"):
        validate_track_a_test(bad)


def test_frozen_ensemble_arithmetic_and_thresholds() -> None:
    p2=np.linspace(0,1,12).reshape(2,6); p3=np.flip(p2,axis=1)
    assert np.allclose(fuse_en2(p2,p3),EXPECTED_ALPHA*p2+EXPECTED_T3_WEIGHT*p3)
    assert EXPECTED_ALPHA==.70 and EXPECTED_T3_WEIGHT==.30
    assert EN2_THRESHOLDS=={"anger":.67,"disgust":.36,"fear":.43,"joy":.65,"sadness":.58,"surprise":.49}
    assert T2_THRESHOLDS=={"anger":.66,"disgust":.53,"fear":.52,"joy":.75,"sadness":.65,"surprise":.57}


def test_evaluated_state_refuses_second_execution(tmp_path: Path) -> None:
    path=tmp_path/"metadata/brighter_ptbr/final/experiment_state.json"
    write_json_atomic(path,{"state":"TEST_EVALUATED","test_accessed":True})
    with pytest.raises(RuntimeError,match="FINAL TEST ALREADY EVALUATED"):
        assert_final_test_authorized(tmp_path)


def test_final_reporting_smoke_with_synthetic_data(tmp_path: Path) -> None:
    from brighter.threshold_calibration import decision_metrics
    y=np.array([[1,0,0,0,0,0],[0,1,1,0,0,0],[0,0,0,1,1,1],[0,0,0,0,0,0]],dtype=int)
    p=np.array([[.9,.1,.2,.1,.1,.1],[.1,.8,.7,.1,.2,.2],[.1,.1,.1,.8,.7,.9],[.1,.1,.1,.1,.1,.1]])
    pred=(p>=.5).astype(int); metrics=decision_metrics(y,pred,p)
    result={"primary":{"metrics":metrics},"secondary":{"metrics":metrics}}
    _write_latex(tmp_path,result); _write_figures(tmp_path,result,y,pred); _write_report(tmp_path,result)
    assert len(list((tmp_path/"reports/brighter_ptbr/final/latex").glob("*.tex")))==4
    assert len(list((tmp_path/"figures/brighter_ptbr/final").glob("*.png")))==4
    assert (tmp_path/"reports/brighter_ptbr/final/final_test_results.md").exists()
