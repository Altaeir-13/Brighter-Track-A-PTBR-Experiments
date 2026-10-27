from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def close(a: float, b: float, tol: float = 5e-5) -> bool:
    return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    state = json.loads((ROOT / "metadata/brighter_ptbr/final/experiment_state.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "reports/brighter_ptbr/final/final_test_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "metadata/brighter_ptbr/paper/paper_evidence_manifest.json").read_text(encoding="utf-8"))
    with (ROOT / "reports/brighter_ptbr/paper/master_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["experiment_id"]: row for row in csv.DictReader(handle)}

    assert state["state"] == "TEST_EVALUATED"
    assert state["primary"] == "EN2" and state["search_space"] == "CLOSED"
    assert state["test_accessed"] is True and state["further_tuning_allowed"] is False
    assert final["test_rows"] == 2226 and final["unique_ids"] == 2226
    assert final["dataset_revision"] == "419566ac8f46f951b51584be36c13e5363008144"
    assert close(final["development"]["EN2_nested_macro_f1"], .5691)
    assert close(final["primary"]["metrics"]["macro_f1"], .2852)
    assert close(final["development"]["T2_nested_macro_f1"], .5585)
    assert close(final["secondary"]["metrics"]["macro_f1"], .1569)
    assert rows["FINAL_EN2"]["scientific_status"] == "FINAL_TEST"
    assert rows["FINAL_T2"]["scientific_status"] == "FINAL_TEST"
    assert close(rows["EN2"]["macro_f1"], final["development"]["EN2_nested_macro_f1"])
    assert close(rows["FINAL_EN2"]["macro_f1"], final["primary"]["metrics"]["macro_f1"])
    assert close(rows["T2"]["macro_f1"], final["development"]["T2_nested_macro_f1"])
    assert close(rows["FINAL_T2"]["macro_f1"], final["secondary"]["metrics"]["macro_f1"])
    assert rows["B2"]["evaluation_protocol"].endswith("fold mean")
    assert close(rows["B2"]["macro_f1"], .3958)
    assert close(rows["T3"]["macro_f1"], .5023)
    assert "FINAL_T3" not in rows
    claim_list = manifest["claims"]
    claims = {c["claim_id"]: c for c in claim_list}
    assert len(claims) == len(claim_list), "Duplicate claim_id in paper evidence manifest"
    assert claims["C_DEPLOY_EN2_SCORE"]["scientific_status"] == "DEPLOYMENT_CALIBRATION_ONLY"
    assert claims["C_DEPLOY_T2_SCORE"]["scientific_status"] == "DEPLOYMENT_CALIBRATION_ONLY"
    assert claims["C_FINAL_EN2_MACRO_F1"]["scientific_status"] == "FINAL_TEST"
    assert close(claims["C_EN2_MACRO_F1"]["value"], rows["EN2"]["macro_f1"])
    assert close(claims["C_FINAL_EN2_MACRO_F1"]["value"], rows["FINAL_EN2"]["macro_f1"])
    labels = json.loads((ROOT / "metadata/brighter_ptbr/final/labels.json").read_text(encoding="utf-8"))
    order = labels.get("label_order", labels.get("labels")) if isinstance(labels, dict) else labels
    assert order == ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    eda = json.loads((ROOT / "metadata/brighter_ptbr/eda/eda_summary.json").read_text(encoding="utf-8"))
    assert [eda["splits"][x]["rows"] for x in ["train", "dev", "test"]] == [2226, 200, 2226]
    source_hashes = json.loads((ROOT / "metadata/brighter_ptbr/paper/source_artifact_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in source_hashes.items():
        assert sha256(ROOT / relative) == expected, f"Frozen source changed: {relative}"
    generalization = json.loads((ROOT / "metadata/brighter_ptbr/final/generalization_gap_summary.json").read_text(encoding="utf-8"))
    for relative, expected in generalization["frozen_output_hashes"].items():
        assert sha256(ROOT / relative) == expected, f"Frozen predictive output changed: {relative}"
    text = (ROOT / "reports/brighter_ptbr/paper/README.md").read_text(encoding="utf-8")
    assert "0.5897" in text and "deployment calibration" in text.lower()
    print(f"PASS: {len(rows)} master systems and {len(claims)} evidence claims are consistent")


if __name__ == "__main__":
    main()
