"""One-time immutable final-test evaluation for the frozen BRIGHTER PT-BR protocol."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix

from .curation import FROZEN_REVISION
from .experiments import EXPECTED_LABELS, TEST_SIZE
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .threshold_calibration import decision_metrics

FINAL_ROOT = Path("metadata/brighter_ptbr/final")
REPORT_ROOT = Path("reports/brighter_ptbr/final")
PREDICTION_ROOT = Path("artifacts/predictions/final")
FIGURE_ROOT = Path("figures/brighter_ptbr/final")
AUTHORIZATION_NAME = "final_test_authorization.json"
RUN_MANIFEST_NAME = "final_test_run_manifest.json"
LABELS = list(EXPECTED_LABELS)
EN2_THRESHOLDS = {"anger": .67, "disgust": .36, "fear": .43, "joy": .65, "sadness": .58, "surprise": .49}
T2_THRESHOLDS = {"anger": .66, "disgust": .53, "fear": .52, "joy": .75, "sadness": .65, "surprise": .57}
EXPECTED_ALPHA = .70
EXPECTED_T3_WEIGHT = .30
EXPECTED_MODEL_HASHES = {
    "T2": "5b845ab4988c02ed8bbf1d1fee7551cd64cdbdac213c19f5ebd24823e79a612a",
    "T3": "aa25ca2517e262281ab60fee0fd161a48b57fc46967c0bba7955e03c98f4255d",
}
OUTPUT_FILES = (
    "t2_test_probabilities.parquet", "t3_component_test_probabilities.parquet",
    "en2_test_probabilities.parquet", "t2_test_predictions.parquet",
    "en2_test_predictions.parquet", "en2_test_submission.csv", "t2_test_submission.csv",
)


def _state_path(root: Path) -> Path:
    return root / FINAL_ROOT / "experiment_state.json"


def transition_state(root: Path, expected: str, target: str, *, test_accessed: bool) -> dict[str, Any]:
    """Atomically transition the protocol state only from its declared predecessor."""
    path = _state_path(root)
    state = load_json(path)
    if state.get("state") != expected:
        if state.get("state") == "TEST_EVALUATED":
            raise RuntimeError("FINAL TEST ALREADY EVALUATED. RE-EXECUTION IS DISABLED BY THE FROZEN EXPERIMENT PROTOCOL.")
        raise RuntimeError(f"Invalid final-test state transition: expected {expected}, found {state.get('state')}")
    state.update({
        "state": target, "primary": "EN2", "secondary": "T2_STANDALONE",
        "test_accessed": bool(test_accessed), "search_space": "CLOSED",
        "space_of_search": "CLOSED", "further_tuning_allowed": False,
        "updated_at": utc_now_iso(),
    })
    write_json_atomic(path, state)
    return state


def validate_authorization(payload: Mapping[str, Any]) -> None:
    required = {
        "authorized": True, "scope": "FINAL_TEST_ONCE", "primary_system": "EN2",
        "secondary_system": "T2_STANDALONE", "search_space_closed": True,
        "config_changes_after_test": False,
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Invalid final-test authorization field: {key}")
    for key in ("timestamp_utc", "config_hashes", "model_hashes", "manifest_sha256", "dataset_revision"):
        if key not in payload:
            raise RuntimeError(f"Missing final-test authorization field: {key}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _check_outputs_absent(root: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (root / PREDICTION_ROOT / name).exists()]
    if (root / FINAL_ROOT / RUN_MANIFEST_NAME).exists():
        existing.append(RUN_MANIFEST_NAME)
    if existing:
        raise RuntimeError("Final-test artifacts already exist; overwrite is forbidden: " + ", ".join(existing))


def preflight(root: Path) -> dict[str, Any]:
    """Validate every frozen input without opening any test data file."""
    state = load_json(_state_path(root))
    if state.get("state") == "TEST_EVALUATED":
        raise RuntimeError("FINAL TEST ALREADY EVALUATED. RE-EXECUTION IS DISABLED BY THE FROZEN EXPERIMENT PROTOCOL.")
    if state.get("state") != "FROZEN_PRE_TEST" or state.get("test_accessed") is not False:
        raise RuntimeError("Final experiment is not in the required FROZEN_PRE_TEST state")
    manifest_path = root / FINAL_ROOT / "final_experiment_manifest.json"
    manifest = load_json(manifest_path)
    en2_path, t2_path = root / "configs/final_en2.yaml", root / "configs/final_t2_standalone.yaml"
    en2, t2 = _load_yaml(en2_path), _load_yaml(t2_path)
    decision = load_json(root / FINAL_ROOT / "deployment_decision_parameters.json")
    t2_decision = load_json(root / FINAL_ROOT / "t2_standalone_thresholds.json")
    labels = load_json(root / FINAL_ROOT / "labels.json")
    if manifest["dataset"]["revision"] != FROZEN_REVISION or en2["dataset_revision"] != FROZEN_REVISION or t2["dataset_revision"] != FROZEN_REVISION:
        raise RuntimeError("Frozen dataset revision mismatch")
    if labels.get("label_order") != LABELS or manifest.get("label_order") != LABELS or en2.get("label_order") != LABELS or t2.get("label_order") != LABELS:
        raise RuntimeError("Frozen label order mismatch")
    if not np.isclose(float(en2["alpha_t2"]), EXPECTED_ALPHA) or not np.isclose(float(en2["weight_t3"]), EXPECTED_T3_WEIGHT) or not np.isclose(float(en2["alpha_t2"] + en2["weight_t3"]), 1.0):
        raise RuntimeError("Frozen ensemble weights mismatch")
    if en2.get("thresholds") != EN2_THRESHOLDS or decision.get("thresholds") != EN2_THRESHOLDS:
        raise RuntimeError("Frozen EN2 thresholds mismatch")
    if t2.get("thresholds") != T2_THRESHOLDS or t2_decision.get("thresholds") != T2_THRESHOLDS:
        raise RuntimeError("Frozen T2 thresholds mismatch")
    config_hashes = {"final_en2.yaml": sha256_file(en2_path), "final_t2_standalone.yaml": sha256_file(t2_path)}
    if config_hashes != manifest.get("config_hashes"):
        raise RuntimeError("Frozen config hash mismatch")
    model_paths = {
        "T2": root / en2["models"]["t2"]["path"] / "model.safetensors",
        "T3": root / en2["models"]["t3"]["path"] / "model.safetensors",
    }
    model_hashes = {key: sha256_file(path) for key, path in model_paths.items()}
    if model_hashes != EXPECTED_MODEL_HASHES:
        raise RuntimeError("Frozen model hash mismatch")
    for key in ("T2", "T3"):
        if model_hashes[key] != manifest["models"][key]["file_hashes"]["model.safetensors"]:
            raise RuntimeError(f"Manifest model hash mismatch for {key}")
    _check_outputs_absent(root)
    return {
        "state_before": state["state"], "model_hashes": model_hashes,
        "config_hashes": config_hashes, "manifest_sha256": sha256_file(manifest_path),
        "dataset_revision": FROZEN_REVISION, "en2": en2, "t2": t2,
        "label_order": LABELS, "preflight_timestamp_utc": utc_now_iso(),
        "test_content_read": False,
    }


def create_final_test_authorization(root: Path) -> dict[str, Any]:
    """Persist explicit one-time authorization after a successful sealed pre-flight."""
    audit = preflight(root)
    path = root / FINAL_ROOT / AUTHORIZATION_NAME
    if path.exists():
        raise RuntimeError("Final-test authorization already exists")
    payload = {
        "authorized": True, "scope": "FINAL_TEST_ONCE", "primary_system": "EN2",
        "secondary_system": "T2_STANDALONE", "search_space_closed": True,
        "config_changes_after_test": False, "timestamp_utc": utc_now_iso(),
        "config_hashes": audit["config_hashes"], "model_hashes": audit["model_hashes"],
        "manifest_sha256": audit["manifest_sha256"], "dataset_revision": FROZEN_REVISION,
    }
    validate_authorization(payload)
    write_json_atomic(path, payload)
    transition_state(root, "FROZEN_PRE_TEST", "FINAL_TEST_AUTHORIZED", test_accessed=False)
    return payload


def assert_final_test_authorized(root: Path) -> dict[str, Any]:
    state = load_json(_state_path(root))
    if state.get("state") == "TEST_EVALUATED":
        raise RuntimeError("FINAL TEST ALREADY EVALUATED. RE-EXECUTION IS DISABLED BY THE FROZEN EXPERIMENT PROTOCOL.")
    if state.get("state") != "FINAL_TEST_AUTHORIZED" or state.get("test_accessed") is not False:
        raise RuntimeError("Final test is not in FINAL_TEST_AUTHORIZED state")
    payload = load_json(root / FINAL_ROOT / AUTHORIZATION_NAME)
    validate_authorization(payload)
    return payload


def validate_track_a_test(frame: pd.DataFrame) -> None:
    required = {"id", "text_raw", *LABELS}
    if len(frame) != TEST_SIZE or frame.id.nunique() != TEST_SIZE or frame.id.duplicated().any():
        raise RuntimeError("Official Track A test must contain exactly 2226 unique IDs")
    if not required.issubset(frame.columns):
        raise RuntimeError("Official Track A test schema mismatch")
    if not frame.id.astype(str).str.fullmatch(r"ptbr_test_track_a_\d{5}").all():
        raise RuntimeError("Non-Track-A identifier detected in final test")
    if not np.isin(frame[LABELS].to_numpy(), [0, 1]).all():
        raise RuntimeError("Invalid final-test labels")


def _validate_probabilities(probabilities: np.ndarray, rows: int) -> None:
    if probabilities.shape != (rows, len(LABELS)) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("Invalid final-test probability tensor")


def fuse_en2(t2: np.ndarray, t3: np.ndarray) -> np.ndarray:
    if t2.shape != t3.shape:
        raise ValueError("T2/T3 probability shapes differ")
    return EXPECTED_ALPHA * np.asarray(t2, float) + EXPECTED_T3_WEIGHT * np.asarray(t3, float)


def _predict(directory: Path, texts: list[str], max_length: int, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen final inference requires the validated CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True).to("cuda").eval()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter(); values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                values.append(torch.sigmoid(model(**batch).logits.float()).cpu().numpy())
    torch.cuda.synchronize(); duration = time.perf_counter() - started
    payload = {"seconds": duration, "examples_per_second": len(texts) / duration, "batch_size": batch_size, "peak_vram_bytes": int(torch.cuda.max_memory_allocated())}
    del model, tokenizer; torch.cuda.empty_cache()
    return np.concatenate(values).astype(np.float64), payload


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"Refusing to overwrite final artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False); os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"Refusing to overwrite final artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n"); os.replace(temporary, path)


def _probability_frame(ids: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"id": ids.astype(str).to_numpy()})
    for i, label in enumerate(LABELS): frame[f"{label}_prob"] = probabilities[:, i]
    return frame


def _prediction_frame(test: pd.DataFrame, probabilities: np.ndarray, predictions: np.ndarray, thresholds: Mapping[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame({"id": test.id.astype(str).to_numpy()})
    for i, label in enumerate(LABELS):
        frame[f"{label}_true"] = test[label].to_numpy(np.int8)
        frame[f"{label}_prob"] = probabilities[:, i]
        frame[f"{label}_pred"] = predictions[:, i].astype(np.int8)
        frame[f"{label}_threshold"] = float(thresholds[label])
    return frame


def _confusions(y: np.ndarray, predictions: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i, label in enumerate(LABELS):
        tn, fp, fn, tp = confusion_matrix(y[:, i], predictions[:, i], labels=[0, 1]).ravel()
        rows.append({"label": label, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return rows


def _write_latex(root: Path, metrics: Mapping[str, Any]) -> None:
    out = root / REPORT_ROOT / "latex"; out.mkdir(parents=True, exist_ok=True)
    systems = [("EN2", metrics["primary"]["metrics"]), ("T2 standalone", metrics["secondary"]["metrics"])]
    br = r"\\"
    main = [r"\begin{tabular}{lrrrrrrr}", r"\toprule", "System & Macro-F1 & Micro-F1 & Macro-P & Macro-R & Hamming & Jaccard & Subset Acc. " + br, r"\midrule"]
    for name, m in systems: main.append(f"{name} & {m['macro_f1']:.4f} & {m['micro_f1']:.4f} & {m['macro_precision']:.4f} & {m['macro_recall']:.4f} & {m['hamming_loss']:.4f} & {m['jaccard_samples']:.4f} & {m['subset_accuracy']:.4f} " + br)
    main += [r"\bottomrule", r"\end{tabular}"]
    per = [r"\begin{tabular}{llrrrr}", r"\toprule", "System & Label & Precision & Recall & F1 & Support " + br, r"\midrule"]
    for name, m in systems:
        for label in LABELS:
            x = m["per_label"][label]; per.append(f"{name} & {label} & {x['precision']:.4f} & {x['recall']:.4f} & {x['f1']:.4f} & {x['support']} " + br)
    per += [r"\bottomrule", r"\end{tabular}"]
    neutral = [r"\begin{tabular}{lrrrrr}", r"\toprule", "System & Exact neutral acc. & Neutral FP & Gold card. & Pred. card. & Delta " + br, r"\midrule"]
    for name, m in systems: neutral.append(f"{name} & {m['zero_label']['exact_neutral_accuracy']:.4f} & {m['zero_label']['neutral_false_positive_rate']:.4f} & {m['gold_label_cardinality']:.4f} & {m['predicted_label_cardinality']:.4f} & {m['predicted_label_cardinality']-m['gold_label_cardinality']:+.4f} " + br)
    neutral += [r"\bottomrule", r"\end{tabular}"]
    multi = [r"\begin{tabular}{lrrrrr}", r"\toprule", "System & N & Macro-F1 & Micro-F1 & Jaccard & Subset Acc. " + br, r"\midrule"]
    for name, m in systems:
        x=m["groups"]["multi_label"]; multi.append(f"{name} & {x['support']} & {x['macro_f1']:.4f} & {x['micro_f1']:.4f} & {x['jaccard_samples']:.4f} & {x['subset_accuracy']:.4f} " + br)
    multi += [r"\bottomrule", r"\end{tabular}"]
    for name, lines in (("final_main_results.tex", main), ("final_per_label_results.tex", per), ("final_neutral_cardinality.tex", neutral), ("final_multilabel_results.tex", multi)):
        (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_figures(root: Path, metrics: Mapping[str, Any], y: np.ndarray, predictions: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    out = root / FIGURE_ROOT; out.mkdir(parents=True, exist_ok=True)
    primary, secondary = metrics["primary"]["metrics"], metrics["secondary"]["metrics"]
    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight"); fig.savefig(out / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); vals=[secondary["macro_f1"],primary["macro_f1"]]; bars=ax.bar(["T2","EN2"],vals,color=["#4c78a8","#f58518"]); ax.bar_label(bars,fmt="%.4f"); ax.set_ylim(0,max(vals)*1.2); ax.set_ylabel("Macro-F1"); ax.set_title("Final test Macro-F1"); save(fig,"final_macro_f1")
    x=np.arange(6); fig,ax=plt.subplots(figsize=(9,4)); w=.36; ax.bar(x-w/2,[secondary["per_label"][l]["f1"] for l in LABELS],w,label="T2"); ax.bar(x+w/2,[primary["per_label"][l]["f1"] for l in LABELS],w,label="EN2"); ax.set_xticks(x,LABELS); ax.set_ylabel("F1"); ax.set_title("Final test F1 by emotion"); ax.legend(); save(fig,"final_per_label_f1")
    gold=y.mean(0); pred=predictions.mean(0); fig,ax=plt.subplots(figsize=(9,4)); ax.bar(x-w/2,gold,w,label="Gold"); ax.bar(x+w/2,pred,w,label="EN2 predicted"); ax.set_xticks(x,LABELS); ax.set_ylabel("Prevalence"); ax.set_title("Gold vs predicted prevalence — EN2"); ax.legend(); save(fig,"final_en2_prevalence")
    fig,axes=plt.subplots(2,3,figsize=(9,6))
    for i,(ax,label) in enumerate(zip(axes.flat,LABELS,strict=True)):
        cm=confusion_matrix(y[:,i],predictions[:,i],labels=[0,1]); ax.imshow(cm,cmap="Blues"); ax.set_title(label); ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        for row in range(2):
            for col in range(2): ax.text(col,row,str(cm[row,col]),ha="center",va="center")
    save(fig,"final_en2_confusion_matrices")


def _write_report(root: Path, result: Mapping[str, Any]) -> None:
    en2, t2 = result["primary"]["metrics"], result["secondary"]["metrics"]
    lines = ["# BRIGHTER PT-BR — final test results", "", "## 1. Frozen protocol", "", "The one-time Track A PT-BR final evaluation used the frozen EN2 primary system (T2/T3 soft fusion 0.70/0.30) and the pre-registered T2 standalone secondary baseline. Models, alpha, thresholds, max length, preprocessing, label order, and dataset revision passed pre-flight hash validation and were not changed.", "", "## 2. Development estimates", "", "Leakage-safe nested development: T2 = **0.5585**, EN2 = **0.5691** Macro-F1. Deployment calibration: T2 = 0.5787, EN2 = 0.5897 — **DEPLOYMENT CALIBRATION ONLY; NOT AN UNBIASED PERFORMANCE ESTIMATE.**", "", "## 3. Final test", "", "| System | Macro-F1 | Micro-F1 | Macro Precision | Macro Recall | Hamming | Jaccard | Subset Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in (("EN2 (primary)",en2),("T2 standalone (secondary)",t2)): lines.append(f"| {name} | {m['macro_f1']:.4f} | {m['micro_f1']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['hamming_loss']:.4f} | {m['jaccard_samples']:.4f} | {m['subset_accuracy']:.4f} |")
    lines += ["", "## 4. Per-label results", "", "| System | Label | Precision | Recall | F1 | AP | Support | Predicted prevalence |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in (("EN2",en2),("T2",t2)):
        for label in LABELS:
            x=m["per_label"][label]; lines.append(f"| {name} | {label} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} | {x['average_precision']:.4f} | {x['support']} | {x['predicted_prevalence']:.4f} |")
    lines += ["", "## 5. Minority labels", "", "The pre-registered minority-label results for disgust, fear, and surprise are reported above without any post-test adjustment.", "", "## 6. Neutral behavior", "", "| System | Exact Neutral Accuracy | Neutral FP |", "|---|---:|---:|", f"| EN2 | {en2['zero_label']['exact_neutral_accuracy']:.4f} | {en2['zero_label']['neutral_false_positive_rate']:.4f} |", f"| T2 | {t2['zero_label']['exact_neutral_accuracy']:.4f} | {t2['zero_label']['neutral_false_positive_rate']:.4f} |", "", "## 7. Multi-label behavior", "", "| System | N | Macro-F1 | Micro-F1 | Jaccard | Subset Accuracy |", "|---|---:|---:|---:|---:|---:|"]
    for name,m in (("EN2",en2),("T2",t2)):
        x=m["groups"]["multi_label"]; lines.append(f"| {name} | {x['support']} | {x['macro_f1']:.4f} | {x['micro_f1']:.4f} | {x['jaccard_samples']:.4f} | {x['subset_accuracy']:.4f} |")
    lines += ["", "## 8. Cardinality and prevalence", "", "| System | Gold cardinality | Predicted cardinality | Delta |", "|---|---:|---:|---:|", f"| EN2 | {en2['gold_label_cardinality']:.4f} | {en2['predicted_label_cardinality']:.4f} | {en2['predicted_label_cardinality']-en2['gold_label_cardinality']:+.4f} |", f"| T2 | {t2['gold_label_cardinality']:.4f} | {t2['predicted_label_cardinality']:.4f} | {t2['predicted_label_cardinality']-t2['gold_label_cardinality']:+.4f} |", "", "## 9. Development-to-test comparison", "", f"EN2: {en2['macro_f1']:.4f} - 0.5691 = **{en2['macro_f1']-.5691254630769899:+.4f}**. T2: {t2['macro_f1']:.4f} - 0.5585 = **{t2['macro_f1']-.5585138660403824:+.4f}**.", "", "## 10. Final observations", "", "EN2 remains the primary pre-registered system and T2 remains the secondary pre-registered baseline regardless of their relative test scores. The search space is closed and further tuning is forbidden.", ""]
    (root / REPORT_ROOT / "final_test_results.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")


def _update_registry(root: Path, result: Mapping[str, Any]) -> None:
    path=root/"reports/brighter_ptbr/experiments/experiment_registry.csv"; frame=pd.read_csv(path)
    for column in ("test_macro_f1","test_micro_f1","test_evaluated_timestamp","final_role"):
        if column not in frame: frame[column]=np.nan
    mapping={"FINAL_EN2":("primary",result["primary"]["metrics"]),"FINAL_T2_BERTWEETBR_ASL":("secondary",result["secondary"]["metrics"])}
    for experiment_id,(role,metrics) in mapping.items():
        mask=frame.experiment_id.eq(experiment_id); frame.loc[mask,"status"]="TEST_EVALUATED"; frame.loc[mask,"test_policy"]="TEST_EVALUATED — SEARCH SPACE CLOSED"; frame.loc[mask,"test_macro_f1"]=metrics["macro_f1"]; frame.loc[mask,"test_micro_f1"]=metrics["micro_f1"]; frame.loc[mask,"test_evaluated_timestamp"]=result["timestamp_end_utc"]; frame.loc[mask,"final_role"]=role
    mask=frame.experiment_id.eq("FINAL_T3_BERTIMBAU_LARGE_BCE"); frame.loc[mask,"status"]="FINAL_EN2_COMPONENT"; frame.loc[mask,"test_policy"]="COMPONENT INFERENCE ONLY — NO STANDALONE TEST METRIC"; frame.loc[mask,"final_role"]="component"
    temporary=path.with_suffix(".csv.tmp"); frame.to_csv(temporary,index=False,float_format="%.8f",lineterminator="\n"); os.replace(temporary,path)


def _package_versions() -> dict[str, str]:
    import importlib.metadata
    versions={"python":platform.python_version()}
    for name in ("torch","transformers","tokenizers","numpy","pandas","scikit-learn","pyarrow"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]="not-installed"
    return versions


def run_final_test(root: Path) -> dict[str, Any]:
    """Run the authorized Track A final test exactly once and freeze all outputs."""
    authorization=assert_final_test_authorized(root); started=utc_now_iso(); accessed=False
    try:
        transition_state(root,"FINAL_TEST_AUTHORIZED","FINAL_TEST_RUNNING",test_accessed=False)
        en2_config=_load_yaml(root/"configs/final_en2.yaml")
        test_path=root/"data"/"curated"/"brighter_ptbr"/FROZEN_REVISION/"track_a"/"test.parquet"
        test=pd.read_parquet(test_path); accessed=True
        running=load_json(_state_path(root)); running["test_accessed"]=True; running["test_first_access_timestamp_utc"]=utc_now_iso(); write_json_atomic(_state_path(root),running)
        validate_track_a_test(test)
        ids=test.id.astype(str).reset_index(drop=True); texts=test.text_raw.astype(str).tolist(); rows=len(test)
        p2,t2_cost=_predict(root/en2_config["models"]["t2"]["path"],texts,int(en2_config["max_length"]),batch_size=8); _validate_probabilities(p2,rows)
        t2_prob_path=root/PREDICTION_ROOT/"t2_test_probabilities.parquet"; _atomic_parquet(_probability_frame(ids,p2),t2_prob_path)
        p3,t3_cost=_predict(root/en2_config["models"]["t3"]["path"],texts,int(en2_config["max_length"]),batch_size=1); _validate_probabilities(p3,rows)
        t3_prob_path=root/PREDICTION_ROOT/"t3_component_test_probabilities.parquet"; _atomic_parquet(_probability_frame(ids,p3),t3_prob_path)
        ensemble=fuse_en2(p2,p3); _validate_probabilities(ensemble,rows)
        en2_prob_path=root/PREDICTION_ROOT/"en2_test_probabilities.parquet"; _atomic_parquet(_probability_frame(ids,ensemble),en2_prob_path)
        en2_thresholds=np.array([EN2_THRESHOLDS[label] for label in LABELS]); t2_thresholds=np.array([T2_THRESHOLDS[label] for label in LABELS]); en2_pred=(ensemble>=en2_thresholds).astype(np.int8); t2_pred=(p2>=t2_thresholds).astype(np.int8); y=test[LABELS].to_numpy(np.int8)
        t2_pred_path=root/PREDICTION_ROOT/"t2_test_predictions.parquet"; en2_pred_path=root/PREDICTION_ROOT/"en2_test_predictions.parquet"; _atomic_parquet(_prediction_frame(test,p2,t2_pred,T2_THRESHOLDS),t2_pred_path); _atomic_parquet(_prediction_frame(test,ensemble,en2_pred,EN2_THRESHOLDS),en2_pred_path)
        _atomic_csv(pd.DataFrame({"id":ids,**{label:en2_pred[:,i] for i,label in enumerate(LABELS)}}),root/PREDICTION_ROOT/"en2_test_submission.csv"); _atomic_csv(pd.DataFrame({"id":ids,**{label:t2_pred[:,i] for i,label in enumerate(LABELS)}}),root/PREDICTION_ROOT/"t2_test_submission.csv")
        artifact_paths=[root/PREDICTION_ROOT/name for name in OUTPUT_FILES]; artifact_hashes={path.relative_to(root).as_posix():sha256_file(path) for path in artifact_paths}
        ended=utc_now_iso(); result={"primary":{"system":"EN2","role":"PRIMARY_PRE_REGISTERED_SYSTEM","metrics":decision_metrics(y,en2_pred,ensemble),"confusion_matrices":_confusions(y,en2_pred)},"secondary":{"system":"T2_STANDALONE","role":"SECONDARY_PRE_REGISTERED_BASELINE","metrics":decision_metrics(y,t2_pred,p2),"confusion_matrices":_confusions(y,t2_pred)},"development":{"EN2_nested_macro_f1":.5691254630769899,"T2_nested_macro_f1":.5585138660403824,"EN2_deployment_calibration_macro_f1":.5896568184803863,"T2_deployment_calibration_macro_f1":.5787498908447},"alpha_t2":EXPECTED_ALPHA,"weight_t3":EXPECTED_T3_WEIGHT,"en2_thresholds":EN2_THRESHOLDS,"t2_thresholds":T2_THRESHOLDS,"test_rows":rows,"unique_ids":int(test.id.nunique()),"track":"A","dataset_revision":FROZEN_REVISION,"known_overlap_retained":True,"single_pre_registered_evaluation":True,"inference":{"T2":t2_cost,"T3":t3_cost},"artifact_hashes":artifact_hashes,"timestamp_start_utc":started,"timestamp_end_utc":ended}
        result_path=root/REPORT_ROOT/"final_test_results.json"; write_json_atomic(result_path,result)
        pd.DataFrame([{"system":"EN2",**x} for x in result["primary"]["confusion_matrices"]]+[{"system":"T2_STANDALONE",**x} for x in result["secondary"]["confusion_matrices"]]).to_csv(root/REPORT_ROOT/"final_confusion_matrices.csv",index=False,lineterminator="\n")
        _write_latex(root,result); _write_figures(root,result,y,en2_pred); _write_report(root,result); _update_registry(root,result)
        report_artifacts=[result_path,root/REPORT_ROOT/"final_test_results.md",root/REPORT_ROOT/"final_confusion_matrices.csv",root/"reports/brighter_ptbr/experiments/experiment_registry.csv"]+list((root/REPORT_ROOT/"latex").glob("*.tex"))+list((root/FIGURE_ROOT).glob("*"))
        report_hashes={path.relative_to(root).as_posix():sha256_file(path) for path in report_artifacts}
        track_manifest=load_json(root/"metadata/brighter_ptbr/track_a_manifest.json"); expected_test_hash=track_manifest["curated_files"]["test"]["sha256"]; observed_test_hash=sha256_file(test_path)
        if observed_test_hash!=expected_test_hash: raise RuntimeError("Curated Track A test hash changed")
        manifest={"authorization_sha256":sha256_file(root/FINAL_ROOT/AUTHORIZATION_NAME),"authorization":authorization,"model_hashes":authorization["model_hashes"],"config_hashes":authorization["config_hashes"],"dataset_revision":FROZEN_REVISION,"dataset_integrity":{"curated_test_expected_sha256":expected_test_hash,"curated_test_observed_sha256":observed_test_hash,"match":True},"number_test_examples":rows,"unique_test_ids":int(test.id.nunique()),"label_order":LABELS,"alpha_t2":EXPECTED_ALPHA,"weight_t3":EXPECTED_T3_WEIGHT,"en2_thresholds":EN2_THRESHOLDS,"t2_thresholds":T2_THRESHOLDS,"timestamp_start_utc":started,"timestamp_end_utc":ended,"package_versions":_package_versions(),"gpu":"NVIDIA GeForce RTX 3060 Ti","test_prediction_hashes":artifact_hashes,"metric_artifact_hashes":report_hashes,"state_before":"FROZEN_PRE_TEST","state_after":"TEST_EVALUATED","search_space":"CLOSED","further_tuning_allowed":False}
        write_json_atomic(root/FINAL_ROOT/RUN_MANIFEST_NAME,manifest)
        transition_state(root,"FINAL_TEST_RUNNING","TEST_EVALUATED",test_accessed=True)
        return result
    except Exception:
        state=load_json(_state_path(root))
        if state.get("state") not in ("TEST_EVALUATED","FINAL_TEST_FAILED_AFTER_ACCESS","FINAL_TEST_ABORTED_PRE_ACCESS"):
            state.update({"state":"FINAL_TEST_FAILED_AFTER_ACCESS" if accessed else "FINAL_TEST_ABORTED_PRE_ACCESS","test_accessed":accessed,"search_space":"CLOSED","space_of_search":"CLOSED","further_tuning_allowed":False,"updated_at":utc_now_iso()}); write_json_atomic(_state_path(root),state)
        raise
