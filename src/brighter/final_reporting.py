"""Post-test reporting recovery using only frozen outputs from the one-time evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic

FINAL_ROOT = Path("metadata/brighter_ptbr/final")
REPORT_ROOT = Path("reports/brighter_ptbr/final")
FIGURE_ROOT = Path("figures/brighter_ptbr/final")
RESULT_PATH = REPORT_ROOT / "final_test_results.json"
LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
EXPECTED_MACRO = {"EN2": 0.2851799354682137, "T2_STANDALONE": 0.15691169825594717}
EXPECTED_OUTPUT_HASHES = {
    "artifacts/predictions/final/t2_test_probabilities.parquet": "debc6061276d022c955d610218e7a6e0c13ba3d4d90c0e693b9d8efba071231e",
    "artifacts/predictions/final/t3_component_test_probabilities.parquet": "f33aed250c959d757234297b0d06e58e768c31d0397e2edd63817f2e9dee5ad0",
    "artifacts/predictions/final/en2_test_probabilities.parquet": "6d54bbb511a00f12c5d1de9f31295ec74170cc60ac1d7ba5e6c12e67fbb5f2b9",
    "artifacts/predictions/final/t2_test_predictions.parquet": "23ca4212656ab02f57ccb783723081889d7f8f967730fe87466153b41ed3c3bd",
    "artifacts/predictions/final/en2_test_predictions.parquet": "5ed3c25279fd41fc6328fb6802c7ccc5400002f919c73388ee05e90463aabf65",
    "artifacts/predictions/final/en2_test_submission.csv": "123458dc1a18251a23eb1f063608d02c6fd718039fbf2551aca9a9e5e4b52abd",
    "artifacts/predictions/final/t2_test_submission.csv": "257bf6e20f4861266cdfb6173f9cba80b8ef7a7df6ccb22177607b936cd6d0b5",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.rstrip() + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(normalized, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def validate_preserved_outputs(root: Path) -> dict[str, str]:
    """Hash final outputs without opening models or the official test dataset."""
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_OUTPUT_HASHES.items():
        path = root / relative
        if not path.exists():
            raise RuntimeError(f"Missing frozen final output: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"Frozen final output hash mismatch: {relative}")
        observed[relative] = digest
    return observed


def validate_frozen_inputs(root: Path) -> dict[str, Any]:
    """Validate hashes only; never instantiate models and never open test data."""
    manifest = load_json(root / FINAL_ROOT / "final_experiment_manifest.json")
    authorization = load_json(root / FINAL_ROOT / "final_test_authorization.json")
    model_files = {
        "T2": root / manifest["models"]["T2"]["path"] / "model.safetensors",
        "T3": root / manifest["models"]["T3"]["path"] / "model.safetensors",
    }
    model_hashes = {name: sha256_file(path) for name, path in model_files.items()}
    if model_hashes != authorization["model_hashes"]:
        raise RuntimeError("Frozen model hashes changed")
    config_paths = {"final_en2.yaml": root / "configs/final_en2.yaml", "final_t2_standalone.yaml": root / "configs/final_t2_standalone.yaml"}
    config_hashes = {name: sha256_file(path) for name, path in config_paths.items()}
    if config_hashes != authorization["config_hashes"]:
        raise RuntimeError("Frozen config hashes changed")
    for stage in manifest["dataset"]["hashes"].values():
        for item in stage.values():
            if not item["match"]:
                raise RuntimeError("Frozen dataset integrity manifest is not valid")
    # The official test file is deliberately not opened or re-hashed in recovery mode.
    return {"model_hashes": model_hashes, "config_hashes": config_hashes, "dataset_hashes": manifest["dataset"]["hashes"]}


def load_frozen_results(root: Path) -> dict[str, Any]:
    result = load_json(root / RESULT_PATH)
    if result.get("test_rows") != 2226 or result.get("unique_ids") != 2226 or result.get("track") != "A":
        raise RuntimeError("Frozen result identity mismatch")
    if not np.isclose(result["primary"]["metrics"]["macro_f1"], EXPECTED_MACRO["EN2"], atol=0, rtol=0):
        raise RuntimeError("Frozen EN2 metric changed")
    if not np.isclose(result["secondary"]["metrics"]["macro_f1"], EXPECTED_MACRO["T2_STANDALONE"], atol=0, rtol=0):
        raise RuntimeError("Frozen T2 metric changed")
    if result.get("artifact_hashes") != EXPECTED_OUTPUT_HASHES:
        raise RuntimeError("Frozen result artifact hashes changed")
    return result


def configure_matplotlib(root: Path) -> Path:
    cache = root / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    os.environ["MPLBACKEND"] = "Agg"
    return cache.resolve()


def _save_figure(fig: Any, directory: Path, name: str) -> None:
    png, pdf = directory / f"{name}.png", directory / f"{name}.pdf"
    if png.exists() and pdf.exists() and png.stat().st_size > 0 and pdf.stat().st_size > 0:
        return
    fig.tight_layout()
    fig.savefig(png, dpi=180, bbox_inches="tight", metadata={"Software": "BRIGHTER PT-BR frozen reporting recovery"})
    fig.savefig(pdf, bbox_inches="tight", metadata={"Creator": "BRIGHTER PT-BR frozen reporting recovery", "CreationDate": None, "ModDate": None})


def generate_final_figures(root: Path, result: Mapping[str, Any]) -> list[Path]:
    """Generate figures solely from the persisted metrics JSON."""
    configure_matplotlib(root)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out = root / FIGURE_ROOT
    out.mkdir(parents=True, exist_ok=True)
    en2, t2 = result["primary"]["metrics"], result["secondary"]["metrics"]

    fig, ax = plt.subplots(figsize=(5, 4))
    values = [t2["macro_f1"], en2["macro_f1"]]
    bars = ax.bar(["T2", "EN2"], values, color=["#4c78a8", "#f58518"])
    ax.bar_label(bars, fmt="%.4f"); ax.set_ylim(0, max(values) * 1.25); ax.set_ylabel("Macro-F1"); ax.set_title("Final test Macro-F1")
    _save_figure(fig, out, "final_macro_f1"); plt.close(fig)

    x = np.arange(len(LABELS)); width = .36
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x-width/2, [t2["per_label"][label]["f1"] for label in LABELS], width, label="T2")
    ax.bar(x+width/2, [en2["per_label"][label]["f1"] for label in LABELS], width, label="EN2")
    ax.set_xticks(x, LABELS); ax.set_ylabel("F1"); ax.set_title("Final test F1 by emotion"); ax.legend()
    _save_figure(fig, out, "final_per_label_f1"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    gold = [en2["per_label"][label]["gold_prevalence"] for label in LABELS]
    predicted = [en2["per_label"][label]["predicted_prevalence"] for label in LABELS]
    ax.bar(x-width/2, gold, width, label="Gold"); ax.bar(x+width/2, predicted, width, label="EN2 predicted")
    ax.set_xticks(x, LABELS); ax.set_ylabel("Prevalence"); ax.set_title("Gold vs predicted prevalence — EN2"); ax.legend()
    _save_figure(fig, out, "final_en2_prevalence"); plt.close(fig)

    confusions = result["primary"].get("confusion_matrices", [])
    if len(confusions) == len(LABELS):
        by_label = {item["label"]: item for item in confusions}
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for ax, label in zip(axes.flat, LABELS, strict=True):
            item = by_label[label]; matrix = np.array([[item["tn"], item["fp"]], [item["fn"], item["tp"]]])
            ax.imshow(matrix, cmap="Blues"); ax.set_title(label); ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
            for row in range(2):
                for column in range(2): ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        _save_figure(fig, out, "final_en2_confusion_matrices"); plt.close(fig)
    else:
        _atomic_text(root / REPORT_ROOT / "final_confusion_figure_status.txt", "SKIPPED — would require reopening raw test data")
    return sorted(out.glob("final_*.*"))


def generate_latex(root: Path, result: Mapping[str, Any]) -> list[Path]:
    out = root / REPORT_ROOT / "latex"; out.mkdir(parents=True, exist_ok=True); br = r"\\"
    systems = [("EN2", result["primary"]["metrics"]), ("T2 standalone", result["secondary"]["metrics"])]
    main = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", "System & Macro-F1 & Micro-F1 & Macro-P & Macro-R & Hamming & Jaccard & Subset Acc. " + br, r"\midrule"]
    for name,m in systems: main.append(f"{name} & {m['macro_f1']:.4f} & {m['micro_f1']:.4f} & {m['macro_precision']:.4f} & {m['macro_recall']:.4f} & {m['hamming_loss']:.4f} & {m['jaccard_samples']:.4f} & {m['subset_accuracy']:.4f} " + br)
    main += [r"\bottomrule", r"\end{tabular}"]
    per = [r"\begin{tabular}{llrrrrr}", r"\toprule", "System & Label & Precision & Recall & F1 & AP & Support " + br, r"\midrule"]
    for name,m in systems:
        for label in LABELS:
            item=m["per_label"][label]; per.append(f"{name} & {label} & {item['precision']:.4f} & {item['recall']:.4f} & {item['f1']:.4f} & {item['average_precision']:.4f} & {item['support']} " + br)
    per += [r"\bottomrule", r"\end{tabular}"]
    neutral = [r"\begin{tabular}{lrrrrr}", r"\toprule", "System & Neutral acc. & Neutral FP & Gold card. & Pred. card. & Delta " + br, r"\midrule"]
    for name,m in systems: neutral.append(f"{name} & {m['zero_label']['exact_neutral_accuracy']:.4f} & {m['zero_label']['neutral_false_positive_rate']:.4f} & {m['gold_label_cardinality']:.4f} & {m['predicted_label_cardinality']:.4f} & {m['predicted_label_cardinality']-m['gold_label_cardinality']:+.4f} " + br)
    neutral += [r"\bottomrule", r"\end{tabular}"]
    multi = [r"\begin{tabular}{lrrrrr}", r"\toprule", "System & N & Macro-F1 & Micro-F1 & Jaccard & Subset Acc. " + br, r"\midrule"]
    for name,m in systems:
        item=m["groups"]["multi_label"]; multi.append(f"{name} & {item['support']} & {item['macro_f1']:.4f} & {item['micro_f1']:.4f} & {item['jaccard_samples']:.4f} & {item['subset_accuracy']:.4f} " + br)
    multi += [r"\bottomrule", r"\end{tabular}"]
    for filename, lines in (("final_main_results.tex",main),("final_per_label_results.tex",per),("final_neutral_cardinality.tex",neutral),("final_multilabel_results.tex",multi)):
        _atomic_text(out/filename,"\n".join(lines))
    return sorted(out.glob("final_*.tex"))


def generate_markdown_report(root: Path, result: Mapping[str, Any]) -> Path:
    en2,t2=result["primary"]["metrics"],result["secondary"]["metrics"]
    lines=["# BRIGHTER PT-BR — final test results","","## 1. Frozen protocol","",f"BRIGHTER Track A PT-BR, revision `{result['dataset_revision']}`, 2,226 official test examples. Primary EN2 fused frozen T2/T3 probabilities with alpha 0.70/0.30 and thresholds `anger=.67, disgust=.36, fear=.43, joy=.65, sadness=.58, surprise=.49`. Secondary T2 used thresholds `anger=.66, disgust=.53, fear=.52, joy=.75, sadness=.65, surprise=.57`. No preprocessing, data modification, recalibration, or post-test parameter change occurred.","","## 2. Execution integrity","","Final inference was executed exactly once. T2/T3 probabilities, EN2 fusion, frozen decisions, submissions, and metrics were persisted before a reporting-only Matplotlib cache permission failure. The pipeline stopped as pre-registered. Recovery consumed only frozen result/output artifacts: no model was loaded, no test dataset was reopened, no inference was repeated, and no hyperparameter changed.","","## 3. Development estimates","","Leakage-safe nested Macro-F1: T2 = **0.5585**; EN2 = **0.5691**.","","## 4. Deployment calibration","","T2 = 0.5787; EN2 = 0.5897. **DEPLOYMENT CALIBRATION ONLY — NOT AN UNBIASED PERFORMANCE ESTIMATE.**","","## 5. Final test","","| System | Macro-F1 | Micro-F1 | Macro Precision | Macro Recall | Hamming | Jaccard | Subset Accuracy |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in (("EN2 (primary)",en2),("T2 standalone (secondary)",t2)): lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['hamming_loss']:.4f} | {m['jaccard_samples']:.4f} | {m['subset_accuracy']:.4f} |")
    lines += ["","## 6. Per-label results","","| System | Label | Precision | Recall | F1 | AP | Support | Predicted prevalence |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in (("EN2",en2),("T2",t2)):
        for label in LABELS:
            item=m["per_label"][label]; lines.append(f"| {name} | {label} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['average_precision']:.4f} | {item['support']} | {item['predicted_prevalence']:.4f} |")
    lines += ["","## 7. Minority labels","",f"EN2 retained the strongest minority result for surprise (F1 {en2['per_label']['surprise']['f1']:.4f}) but recovered very few disgust (recall {en2['per_label']['disgust']['recall']:.4f}) and fear (recall {en2['per_label']['fear']['recall']:.4f}) positives. T2 recovered no disgust positives and only one fear true positive. These are frozen descriptive findings, not grounds for post-test adjustment.","","## 8. Neutral behavior","","| System | Exact Neutral Accuracy | Neutral FP |","|---|---:|---:|",f"| EN2 | {en2['zero_label']['exact_neutral_accuracy']:.4f} | {en2['zero_label']['neutral_false_positive_rate']:.4f} |",f"| T2 | {t2['zero_label']['exact_neutral_accuracy']:.4f} | {t2['zero_label']['neutral_false_positive_rate']:.4f} |","","## 9. Cardinality","","| System | Gold | Predicted | Delta |","|---|---:|---:|---:|",f"| EN2 | {en2['gold_label_cardinality']:.4f} | {en2['predicted_label_cardinality']:.4f} | {en2['predicted_label_cardinality']-en2['gold_label_cardinality']:+.4f} |",f"| T2 | {t2['gold_label_cardinality']:.4f} | {t2['predicted_label_cardinality']:.4f} | {t2['predicted_label_cardinality']-t2['gold_label_cardinality']:+.4f} |","",f"Development EN2 predicted cardinality was approximately 1.0639; final test cardinality was {en2['predicted_label_cardinality']:.4f} against gold {en2['gold_label_cardinality']:.4f}. The final system emitted substantially fewer positive labels, consistent with very low recalls for disgust ({en2['per_label']['disgust']['recall']:.4f}), fear ({en2['per_label']['fear']['recall']:.4f}), and sadness ({en2['per_label']['sadness']['recall']:.4f}). No recalibration was performed.","","## 10. Development-to-test gap","",f"EN2 changed from 0.5691 nested development Macro-F1 to {en2['macro_f1']:.4f} on final test, a difference of {en2['macro_f1']-.5691254630769899:+.4f}. T2 changed from 0.5585 to {t2['macro_f1']:.4f}, a difference of {t2['macro_f1']-.5585138660403824:+.4f}.","","## 11. Interpretation","","The gaps are compatible with several non-exclusive hypotheses: train/dev-to-test distribution shift; probability calibration shift after final retraining; threshold-transfer failure; seed/model variance; rare-label uncertainty; and differences between cross-fitted OOF models and a single final model trained on all 2,426 development examples. This recovery does not test or privilege any causal explanation. The result is valid under the frozen protocol and is not treated as erroneous merely because it is lower than development.","","## 12. Threats to validity","","- The development dataset is small for neural multilabel estimation.","- Label imbalance is severe; the original training split contains only 75 disgust positives.","- One train–test text overlap was known before final evaluation and retained under the official split protocol.","- Deployment alpha 0.70 differed from the nested alpha range/mean, increasing transfer uncertainty.","- Development and test scores/distributions differ markedly.","- Final models used a single frozen seed.","- Per-label thresholds may be sensitive to prevalence and score-distribution changes.","- OOF estimates aggregate multiple fold-specific models, whereas deployment used one model per component trained on the full development pool.","","The search space remains closed. EN2 remains the primary pre-registered system and T2 the secondary baseline regardless of relative test performance."]
    path=root/REPORT_ROOT/"final_test_results.md"; _atomic_text(path,"\n".join(lines)); return path


def generate_incident_report(root: Path) -> Path:
    text="""# Final-test reporting incident

1. The one-time frozen final evaluation started normally after successful pre-flight and explicit authorization.
2. The official Track A test was accessed once.
3. T2 and T3 probabilities, EN2 probabilities, frozen binary predictions, and submission artifacts were persisted with SHA-256 hashes.
4. Final EN2 and T2 metrics were calculated and persisted in `final_test_results.json`.
5. The subsequent figure step failed exclusively because Matplotlib could not create `<MATPLOTLIB_CACHE>/fontlist-v330.json.matplotlib-lock`.
6. The pipeline stopped and recorded `FINAL_TEST_FAILED_AFTER_ACCESS`, as required by the frozen protocol.
7. No inference was repeated and no model or official test dataset was reopened during recovery.
8. Recovery was restricted to reporting from frozen outputs, with a project-local Matplotlib cache and the Agg backend.
9. No model, alpha, threshold, preprocessing rule, prediction, probability, or hyperparameter was changed.
"""
    path=root/REPORT_ROOT/"final_test_incident.md"; _atomic_text(path,text); return path


def update_registry(root: Path, result: Mapping[str, Any]) -> Path:
    path=root/"reports/brighter_ptbr/experiments/experiment_registry.csv"; frame=pd.read_csv(path)
    for column in ("test_macro_f1","test_micro_f1"):
        if column not in frame.columns: frame[column]=np.nan
    for column in ("test_evaluated_timestamp","final_role"):
        if column not in frame.columns: frame[column]=pd.Series(pd.NA,index=frame.index,dtype="object")
        else: frame[column]=frame[column].astype("object")
    mappings={"FINAL_EN2":("primary",result["primary"]["metrics"]),"FINAL_T2_BERTWEETBR_ASL":("secondary",result["secondary"]["metrics"])}
    for experiment_id,(role,metrics) in mappings.items():
        mask=frame.experiment_id.eq(experiment_id); frame.loc[mask,"status"]="TEST_EVALUATED"; frame.loc[mask,"test_policy"]="TEST_EVALUATED — SEARCH SPACE CLOSED"; frame.loc[mask,"test_macro_f1"]=metrics["macro_f1"]; frame.loc[mask,"test_micro_f1"]=metrics["micro_f1"]; frame.loc[mask,"test_evaluated_timestamp"]=result["timestamp_end_utc"]; frame.loc[mask,"final_role"]=role
    component=frame.experiment_id.eq("FINAL_T3_BERTIMBAU_LARGE_BCE"); frame.loc[component,"status"]="FINAL_EN2_COMPONENT"; frame.loc[component,"test_policy"]="COMPONENT INFERENCE ONLY — NO STANDALONE TEST METRIC"; frame.loc[component,"final_role"]="component"; frame.loc[component,"test_macro_f1"]=np.nan; frame.loc[component,"test_micro_f1"]=np.nan
    temporary=path.with_suffix(".csv.tmp"); frame.to_csv(temporary,index=False,float_format="%.8f",lineterminator="\n"); os.replace(temporary,path); return path


def _reporting_artifact_hashes(root: Path) -> dict[str,str]:
    paths=[root/REPORT_ROOT/"final_test_results.md",root/REPORT_ROOT/"final_test_incident.md",root/"reports/brighter_ptbr/experiments/experiment_registry.csv",*sorted((root/REPORT_ROOT/"latex").glob("final_*.tex")),*sorted((root/FIGURE_ROOT).glob("final_*.*"))]
    return {path.relative_to(root).as_posix():sha256_file(path) for path in paths}


def write_run_manifest(root: Path, result: Mapping[str,Any], outputs: Mapping[str,str], integrity: Mapping[str,Any]) -> Path:
    authorization_path=root/FINAL_ROOT/"final_test_authorization.json"
    payload={"state_before":"FROZEN_PRE_TEST","state_at_recovery_start":"FINAL_TEST_FAILED_AFTER_ACCESS","state_after":"TEST_EVALUATED","test_accessed":True,"inference_completed":True,"metrics_completed":True,"reporting_failure":True,"reporting_failure_reason":"Matplotlib global font-cache permission error","reporting_recovered":True,"recovery_mode":"REPORTING_ONLY","models_loaded_during_recovery":False,"test_dataset_loaded_during_recovery":False,"inference_executed_during_recovery":False,"parameters_modified":False,"authorization_sha256":sha256_file(authorization_path),"model_hashes":integrity["model_hashes"],"config_hashes":integrity["config_hashes"],"dataset_revision":result["dataset_revision"],"test_instances":result["test_rows"],"probability_and_prediction_hashes":dict(outputs),"final_metrics_sha256":sha256_file(root/RESULT_PATH),"timestamp_final_test_start_utc":result.get("timestamp_start_utc"),"timestamp_final_test_metrics_utc":result.get("timestamp_end_utc"),"timestamp_reporting_recovery_utc":utc_now_iso(),"reporting_artifact_hashes":_reporting_artifact_hashes(root),"search_space":"CLOSED","further_tuning_allowed":False}
    path=root/FINAL_ROOT/"final_test_run_manifest.json"; write_json_atomic(path,payload); return path


def validate_completed_reporting(root: Path) -> dict[str,Any]:
    outputs=validate_preserved_outputs(root); result=load_frozen_results(root); integrity=validate_frozen_inputs(root)
    required=[root/REPORT_ROOT/"final_test_results.md",root/REPORT_ROOT/"final_test_incident.md",root/FINAL_ROOT/"final_test_run_manifest.json",*[(root/REPORT_ROOT/"latex"/name) for name in ("final_main_results.tex","final_per_label_results.tex","final_neutral_cardinality.tex","final_multilabel_results.tex")]]
    required += [root/FIGURE_ROOT/f"{stem}.{suffix}" for stem in ("final_macro_f1","final_per_label_f1","final_en2_prevalence","final_en2_confusion_matrices") for suffix in ("png","pdf")]
    if any(not path.exists() or path.stat().st_size==0 for path in required): raise RuntimeError("Reporting recovery artifacts are incomplete")
    return {"outputs":outputs,"result":result,"integrity":integrity}


def recover_reporting(root: Path) -> dict[str,Any]:
    state=load_json(root/FINAL_ROOT/"experiment_state.json")
    if state.get("state")=="TEST_EVALUATED":
        validated=validate_completed_reporting(root); return {"mode":"REPORTING_ONLY","idempotent_reuse":True,**validated}
    if state.get("state")!="FINAL_TEST_FAILED_AFTER_ACCESS" or state.get("test_accessed") is not True:
        raise RuntimeError("Reporting recovery requires FINAL_TEST_FAILED_AFTER_ACCESS with test_accessed=true")
    if (root/FINAL_ROOT/"final_test_run_manifest.json").exists():
        validated=validate_completed_reporting(root); return {"mode":"REPORTING_ONLY","idempotent_reuse":True,**validated}
    outputs=validate_preserved_outputs(root); result=load_frozen_results(root); integrity=validate_frozen_inputs(root)
    generate_final_figures(root,result); generate_latex(root,result); generate_markdown_report(root,result); generate_incident_report(root); update_registry(root,result); write_run_manifest(root,result,outputs,integrity)
    return {"mode":"REPORTING_ONLY","idempotent_reuse":False,"outputs":outputs,"result":result,"integrity":integrity}


def finalize_recovery_state(root: Path) -> dict[str,Any]:
    validated=validate_completed_reporting(root); path=root/FINAL_ROOT/"experiment_state.json"; state=load_json(path)
    if state.get("state")=="TEST_EVALUATED": return state
    if state.get("state")!="FINAL_TEST_FAILED_AFTER_ACCESS" or state.get("test_accessed") is not True: raise RuntimeError("Invalid reporting-recovery state transition")
    state.update({"state":"TEST_EVALUATED","test_accessed":True,"inference_completed":True,"metrics_completed":True,"reporting_completed":True,"search_space":"CLOSED","space_of_search":"CLOSED","further_tuning_allowed":False,"test_policy":"TEST_EVALUATED — SEARCH SPACE CLOSED","reporting_recovery_mode":"REPORTING_ONLY","updated_at":utc_now_iso()}); write_json_atomic(path,state); return state
