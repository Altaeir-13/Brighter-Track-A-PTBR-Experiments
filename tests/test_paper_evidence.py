from __future__ import annotations

import hashlib
import json
from pathlib import Path

from brighter.paper_evidence import LABELS, STATUSES, build_claims, build_master_results, generate_paper_package


ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_master_has_exact_relevant_systems_and_statuses():
    rows = build_master_results(ROOT)
    assert [row["experiment_id"] for row in rows] == [
        "B0", "B1", "B2", "L1", "L2", "L3", "L4", "E1", "E2", "E3", "E4", "E5", "E6",
        "T1", "T2", "T3", "EN1", "EN2", "FINAL_EN2", "FINAL_T2",
    ]
    assert all(row["scientific_status"] in STATUSES for row in rows)
    assert not any(row["experiment_id"] == "FINAL_T3" for row in rows)


def test_frozen_key_values_and_fold_mean_oof_distinction():
    rows = {row["experiment_id"]: row for row in build_master_results(ROOT)}
    assert abs(rows["B2"]["macro_f1"] - 0.3957662267) < 1e-10
    assert abs(rows["T3"]["macro_f1"] - 0.5022780104) < 1e-10
    assert abs(rows["EN2"]["macro_f1"] - 0.5691254631) < 1e-10
    assert abs(rows["FINAL_EN2"]["macro_f1"] - 0.2851799355) < 1e-10
    assert rows["B2"]["evaluation_protocol"].endswith("fold mean")


def test_claim_manifest_statuses_are_explicit():
    claims = build_claims(ROOT, build_master_results(ROOT))
    assert claims
    assert all(claim["scientific_status"] in STATUSES for claim in claims)
    calibration = [c for c in claims if c["claim_id"].startswith("C_DEPLOY")]
    assert calibration and all(c["scientific_status"] == "DEPLOYMENT_CALIBRATION_ONLY" for c in calibration)


def test_paper_generation_is_idempotent_and_preserves_frozen_sources():
    protected = [
        ROOT / "reports/brighter_ptbr/final/final_test_results.json",
        ROOT / "metadata/brighter_ptbr/final/experiment_state.json",
        ROOT / "artifacts/predictions/final/en2_test_probabilities.parquet",
        ROOT / "artifacts/predictions/final/en2_test_predictions.parquet",
    ]
    before = {p: sha(p) for p in protected}
    generate_paper_package(ROOT)
    first = sha(ROOT / "reports/brighter_ptbr/paper/master_results.csv")
    generate_paper_package(ROOT)
    assert sha(ROOT / "reports/brighter_ptbr/paper/master_results.csv") == first
    assert {p: sha(p) for p in protected} == before


def test_paper_code_has_no_training_inference_or_raw_test_access():
    source = (ROOT / "src/brighter/paper_evidence.py").read_text(encoding="utf-8").lower()
    forbidden = ["load_dataset(", "automodelfor", "from_pretrained(", "model.safetensors", "test.parquet", "threshold_grid", "fit("]
    assert not any(token in source for token in forbidden)


def test_label_order_and_final_state_unchanged():
    assert LABELS == ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    state = json.loads((ROOT / "metadata/brighter_ptbr/final/experiment_state.json").read_text(encoding="utf-8"))
    assert state["state"] == "TEST_EVALUATED"
    assert state["test_accessed"] is True
    assert state["search_space"] == "CLOSED"
    assert state["further_tuning_allowed"] is False
