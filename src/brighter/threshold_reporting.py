"""Scientific consolidation for nested threshold calibration T1--T3."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import DEV_POOL_SIZE, EXPECTED_LABELS
from .manifest import load_json, sha256_file, write_json_atomic

SYSTEMS = (
    ("T1", "E2 BERTweetBR + Weighted Focal", "bertweetbr_weighted_focal_thresholded", "bertweetbr_weighted_focal"),
    ("T2", "E3 BERTweetBR + ASL", "bertweetbr_asl_thresholded", "bertweetbr_asl"),
    ("T3", "B2 BERTimbau Large + BCE", "bertimbau_large_bce_thresholded", "bertimbau_large_bce"),
)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _save(fig: Any, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load(root: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for system, display, slug, source in SYSTEMS:
        summary = load_json(root / "metadata" / "brighter_ptbr" / "experiments" / f"{slug}_summary.json")
        thresholded = pd.read_parquet(root / "artifacts" / "predictions" / f"{slug}_oof.parquet")
        original = pd.read_parquet(root / "artifacts" / "predictions" / f"{source}_oof.parquet")
        if len(thresholded) != DEV_POOL_SIZE or thresholded.id.nunique() != DEV_POOL_SIZE:
            raise RuntimeError(f"Incomplete thresholded OOF for {system}")
        if thresholded.id.tolist() != original.id.tolist() or thresholded.fold.tolist() != original.fold.tolist():
            raise RuntimeError(f"ID/fold mismatch for {system}")
        for label in EXPECTED_LABELS:
            if not np.array_equal(thresholded[f"{label}_prob"].to_numpy(), original[f"{label}_prob"].to_numpy()):
                raise RuntimeError(f"Probability mutation for {system}/{label}")
        if sha256_file(root / "artifacts" / "predictions" / f"{slug}_oof.parquet") != summary["thresholded_oof_sha256"]:
            raise RuntimeError(f"Thresholded OOF hash mismatch for {system}")
        output[system] = {"display": display, "slug": slug, "source": source,
                          "summary": summary, "oof": thresholded, "original": original}
    reference = output["T1"]["oof"]
    for system in ("T2", "T3"):
        current = output[system]["oof"]
        if current.id.tolist() != reference.id.tolist() or current.fold.tolist() != reference.fold.tolist():
            raise RuntimeError("Cross-system OOF ordering differs")
    return output


def paired_bootstrap(y: np.ndarray, fixed: np.ndarray, thresholded: np.ndarray,
                     samples: int = 10_000, seed: int = 42) -> dict[str, float | int]:
    """Instance-paired percentile bootstrap of thresholded minus fixed Macro-F1."""
    rng, n = np.random.default_rng(seed), len(y)
    differences = np.empty(samples, dtype=float)
    chunk = 100
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        macro = []
        for predictions in (fixed, thresholded):
            class_scores = []
            for label in range(y.shape[1]):
                truth = y[indices, label]
                pred = predictions[indices, label]
                tp = ((truth == 1) & (pred == 1)).sum(axis=1)
                fp = ((truth == 0) & (pred == 1)).sum(axis=1)
                fn = ((truth == 1) & (pred == 0)).sum(axis=1)
                denominator = 2 * tp + fp + fn
                class_scores.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
            macro.append(np.mean(class_scores, axis=0))
        differences[start:start + count] = macro[1] - macro[0]
    observed_fixed = np.mean([_binary_f1(y[:, i], fixed[:, i]) for i in range(y.shape[1])])
    observed_thresholded = np.mean([_binary_f1(y[:, i], thresholded[:, i]) for i in range(y.shape[1])])
    return {"samples": samples, "seed": seed, "delta": float(observed_thresholded - observed_fixed),
            "ci_2_5": float(np.percentile(differences, 2.5)), "ci_97_5": float(np.percentile(differences, 97.5))}


def _binary_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def _tables(data: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, ...]:
    thresholds, main, per_label, neutral, multilabel, boot = [], [], [], [], [], []
    for system, display, _, _ in SYSTEMS:
        s = data[system]["summary"]; fixed, calibrated = s["fixed_metrics"], s["thresholded_metrics"]
        tf = pd.DataFrame(s["thresholds_by_fold"])
        for row in tf.to_dict(orient="records"): thresholds.append(row)
        y = data[system]["oof"][[f"{x}_true" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
        probabilities = data[system]["oof"][[f"{x}_prob" for x in EXPECTED_LABELS]].to_numpy(dtype=float)
        fixed_pred = (probabilities >= .5).astype(int)
        calibrated_pred = data[system]["oof"][[f"{x}_pred" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
        bootstrap = paired_bootstrap(y, fixed_pred, calibrated_pred)
        boot.append({"system": system, **bootstrap})
        main.append({"system": system, "display": display, "fixed_macro_f1": fixed["macro_f1"],
                     "thresholded_macro_f1": calibrated["macro_f1"], "delta_macro_f1": calibrated["macro_f1"]-fixed["macro_f1"],
                     "bootstrap_ci_low": bootstrap["ci_2_5"], "bootstrap_ci_high": bootstrap["ci_97_5"],
                     "fixed_micro_f1": fixed["micro_f1"], "thresholded_micro_f1": calibrated["micro_f1"],
                     "fixed_macro_precision": fixed["macro_precision"], "thresholded_macro_precision": calibrated["macro_precision"],
                     "fixed_macro_recall": fixed["macro_recall"], "thresholded_macro_recall": calibrated["macro_recall"],
                     "fixed_hamming_loss": fixed["hamming_loss"], "thresholded_hamming_loss": calibrated["hamming_loss"],
                     "fixed_jaccard": fixed["jaccard_samples"], "thresholded_jaccard": calibrated["jaccard_samples"],
                     "fixed_subset_accuracy": fixed["subset_accuracy"], "thresholded_subset_accuracy": calibrated["subset_accuracy"],
                     "macro_average_precision": fixed["macro_average_precision"],
                     "neutral_fp_thresholded": calibrated["zero_label"]["neutral_false_positive_rate"],
                     "predicted_cardinality_thresholded": calibrated["predicted_label_cardinality"],
                     "disgust_f1": calibrated["per_label"]["disgust"]["f1"],
                     "fear_f1": calibrated["per_label"]["fear"]["f1"], "surprise_f1": calibrated["per_label"]["surprise"]["f1"]})
        for label in EXPECTED_LABELS:
            a, b, values = fixed["per_label"][label], calibrated["per_label"][label], tf[label]
            per_label.append({"system": system, "label": label, "average_precision": a["average_precision"],
                              "f1_fixed": a["f1"], "f1_thresholded": b["f1"], "delta_f1": b["f1"]-a["f1"],
                              "precision_fixed": a["precision"], "precision_thresholded": b["precision"],
                              "recall_fixed": a["recall"], "recall_thresholded": b["recall"],
                              "support": a["support"], "predicted_positive_fixed": a["predicted_positive"],
                              "predicted_positive_thresholded": b["predicted_positive"],
                              "gold_prevalence": a["gold_prevalence"], "predicted_prevalence_fixed": a["predicted_prevalence"],
                              "predicted_prevalence_thresholded": b["predicted_prevalence"],
                              "threshold_mean": values.mean(), "threshold_std": values.std(ddof=1),
                              "threshold_min": values.min(), "threshold_max": values.max()})
        gold = fixed["gold_label_cardinality"]
        neutral.append({"system": system, "neutral_fp_fixed": fixed["zero_label"]["neutral_false_positive_rate"],
                        "neutral_fp_thresholded": calibrated["zero_label"]["neutral_false_positive_rate"],
                        "neutral_fp_delta": calibrated["zero_label"]["neutral_false_positive_rate"]-fixed["zero_label"]["neutral_false_positive_rate"],
                        "neutral_accuracy_fixed": fixed["zero_label"]["exact_neutral_accuracy"],
                        "neutral_accuracy_thresholded": calibrated["zero_label"]["exact_neutral_accuracy"],
                        "gold_cardinality": gold, "predicted_cardinality_fixed": fixed["predicted_label_cardinality"],
                        "predicted_cardinality_thresholded": calibrated["predicted_label_cardinality"],
                        "absolute_error_fixed": abs(fixed["predicted_label_cardinality"]-gold),
                        "absolute_error_thresholded": abs(calibrated["predicted_label_cardinality"]-gold)})
        a, b = fixed["groups"]["multi_label"], calibrated["groups"]["multi_label"]
        multilabel.append({"system": system, "support": a["support"],
                           **{f"fixed_{k}": a[k] for k in ("macro_f1","micro_f1","jaccard_samples","subset_accuracy")},
                           **{f"thresholded_{k}": b[k] for k in ("macro_f1","micro_f1","jaccard_samples","subset_accuracy")}})
    return tuple(pd.DataFrame(x) for x in (main, thresholds, per_label, neutral, multilabel, boot))


def _complementarity(data: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions, probabilities = {}, {}
    y = data["T1"]["oof"][[f"{x}_true" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
    multi = y.sum(axis=1) >= 2
    for system, _, _, _ in SYSTEMS:
        oof = data[system]["oof"]
        predictions[system] = oof[[f"{x}_pred" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
        probabilities[system] = oof[[f"{x}_prob" for x in EXPECTED_LABELS]].to_numpy(dtype=float)
    pair_rows, label_rows = [], []
    exact = {s: np.all(predictions[s] == y, axis=1) for s in predictions}
    for a, b in combinations(predictions, 2):
        diff = predictions[a] != predictions[b]
        error_a, error_b = predictions[a] != y, predictions[b] != y
        pair_rows.append({"system_a": a, "system_b": b, "examples_with_any_disagreement": int(diff.any(axis=1).sum()),
                          "example_disagreement_rate": float(diff.any(axis=1).mean()), "label_disagreement_rate": float(diff.mean()),
                          "a_exact_correct_b_wrong": int((exact[a] & ~exact[b]).sum()), "b_exact_correct_a_wrong": int((exact[b] & ~exact[a]).sum()),
                          "both_exact_correct": int((exact[a] & exact[b]).sum()), "both_exact_wrong": int((~exact[a] & ~exact[b]).sum()),
                          "probability_correlation_flat": float(np.corrcoef(probabilities[a].ravel(), probabilities[b].ravel())[0,1]),
                          "error_correlation_flat": float(np.corrcoef(error_a.ravel(), error_b.ravel())[0,1]),
                          "multilabel_example_disagreement_rate": float(diff[multi].any(axis=1).mean()),
                          "multilabel_a_correct_b_wrong": int((exact[a][multi] & ~exact[b][multi]).sum()),
                          "multilabel_b_correct_a_wrong": int((exact[b][multi] & ~exact[a][multi]).sum())})
        for i, label in enumerate(EXPECTED_LABELS):
            ea, eb = error_a[:,i].astype(int), error_b[:,i].astype(int)
            error_corr = float(np.corrcoef(ea, eb)[0,1]) if ea.std() and eb.std() else float("nan")
            label_rows.append({"system_a": a, "system_b": b, "label": label,
                               "disagreement_rate": float(diff[:,i].mean()),
                               "a_correct_b_wrong": int(((ea==0)&(eb==1)).sum()), "b_correct_a_wrong": int(((eb==0)&(ea==1)).sum()),
                               "probability_correlation": float(np.corrcoef(probabilities[a][:,i], probabilities[b][:,i])[0,1]),
                               "error_correlation": error_corr,
                               "multilabel_disagreement_rate": float(diff[multi,i].mean())})
    exclusive = []
    for system in predictions:
        others = [x for x in predictions if x != system]
        exclusive.append({"system": system, "exact_correct": int(exact[system].sum()),
                          "exclusive_exact_correct": int((exact[system] & ~exact[others[0]] & ~exact[others[1]]).sum()),
                          "exclusive_error": int((~exact[system] & exact[others[0]] & exact[others[1]]).sum()),
                          "multilabel_exact_correct": int(exact[system][multi].sum()),
                          "multilabel_exclusive_exact_correct": int((exact[system][multi] & ~exact[others[0]][multi] & ~exact[others[1]][multi]).sum())})
    return pd.DataFrame(pair_rows), pd.DataFrame(label_rows), pd.DataFrame(exclusive)


def _figures(root: Path, main: pd.DataFrame, thresholds: pd.DataFrame, per_label: pd.DataFrame,
             neutral: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    out = root / "figures" / "brighter_ptbr" / "experiments"
    x = np.arange(3); width=.35
    fig, ax=plt.subplots(figsize=(8,5)); ax.bar(x-width/2,main.fixed_macro_f1,width,label="Fixed 0.50"); ax.bar(x+width/2,main.thresholded_macro_f1,width,label="Nested thresholded"); ax.set_xticks(x,main.system); ax.set(ylabel="Macro-F1",title="Fixed vs nested cross-fitted thresholds",ylim=(0,.65)); ax.legend(); ax.grid(axis="y",alpha=.2); _save(fig,out,"fixed_vs_thresholded_macro_f1")
    for labels,name,title in ((EXPECTED_LABELS,"per_label_f1_before_after","Per-label F1 before/after"),(("disgust","fear","surprise"),"minority_f1_before_after","Minority F1 before/after")):
        sub=per_label[per_label.label.isin(labels)]; fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
        for ax,(system,g) in zip(axes,sub.groupby("system",sort=False),strict=True):
            q=np.arange(len(g)); ax.bar(q-width/2,g.f1_fixed,width,label="Fixed"); ax.bar(q+width/2,g.f1_thresholded,width,label="Thresholded"); ax.set_xticks(q,g.label,rotation=35,ha="right"); ax.set_title(system); ax.grid(axis="y",alpha=.2)
        axes[0].set_ylabel("F1"); axes[0].legend(); fig.suptitle(title); fig.tight_layout(); _save(fig,out,name)
    stats=per_label[["system","label","threshold_mean","threshold_std"]]; fig,ax=plt.subplots(figsize=(10,5));
    for j,(system,g) in enumerate(stats.groupby("system",sort=False)): ax.errorbar(np.arange(6)+(j-1)*.12,g.threshold_mean,yerr=g.threshold_std,fmt="o",capsize=3,label=system)
    ax.axhline(.5,color="gray",ls="--",lw=1); ax.set_xticks(np.arange(6),EXPECTED_LABELS); ax.set(ylabel="Threshold",title="Nested thresholds by label",ylim=(0,1)); ax.legend(); ax.grid(axis="y",alpha=.2); _save(fig,out,"thresholds_by_label")
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(x-width/2,neutral.neutral_fp_fixed,width,label="Fixed"); ax.bar(x+width/2,neutral.neutral_fp_thresholded,width,label="Thresholded"); ax.set_xticks(x,neutral.system); ax.set(ylabel="Neutral FP",title="Neutral false-positive rate",ylim=(0,1)); ax.legend(); ax.grid(axis="y",alpha=.2); _save(fig,out,"neutral_fp_before_after")
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(x-width/2,neutral.predicted_cardinality_fixed,width,label="Fixed"); ax.bar(x+width/2,neutral.predicted_cardinality_thresholded,width,label="Thresholded"); ax.axhline(neutral.gold_cardinality.iloc[0],color="black",ls="--",label="Gold"); ax.set_xticks(x,neutral.system); ax.set(ylabel="Labels/example",title="Predicted label cardinality"); ax.legend(); ax.grid(axis="y",alpha=.2); _save(fig,out,"cardinality_before_after")
    long=thresholds.melt(id_vars=["system","outer_fold"],value_vars=EXPECTED_LABELS,var_name="label",value_name="threshold"); fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,(system,g) in zip(axes,long.groupby("system",sort=False),strict=True):
        for fold,fg in g.groupby("outer_fold"): ax.plot(np.arange(6),fg.threshold,marker="o",alpha=.65,label=f"fold {fold}")
        ax.set_xticks(np.arange(6),EXPECTED_LABELS,rotation=35,ha="right"); ax.set_title(system); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Threshold"); axes[-1].legend(fontsize=8); fig.suptitle("Threshold stability across outer folds"); fig.tight_layout(); _save(fig,out,"threshold_stability")
    matrix=np.eye(3); names=list(main.system)
    for r in pairwise.itertuples(): i,j=names.index(r.system_a),names.index(r.system_b); matrix[i,j]=matrix[j,i]=r.example_disagreement_rate
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(matrix,vmin=0,vmax=max(.01,matrix[np.triu_indices(3,1)].max()),cmap="Blues"); ax.set_xticks(range(3),names); ax.set_yticks(range(3),names); [ax.text(j,i,"—" if i==j else f"{matrix[i,j]:.3f}",ha="center",va="center") for i in range(3) for j in range(3)]; ax.set_title("Example-level prediction disagreement"); fig.colorbar(im,ax=ax,label="Rate"); _save(fig,out,"threshold_error_complementarity")


def _latex(root: Path, main: pd.DataFrame, per_label: pd.DataFrame, thresholds: pd.DataFrame, neutral: pd.DataFrame) -> None:
    report=root/"reports"/"brighter_ptbr"/"experiments"
    main_tex=main[["system","fixed_macro_f1","thresholded_macro_f1","delta_macro_f1","bootstrap_ci_low","bootstrap_ci_high"]]
    label_tex=per_label[["system","label","f1_fixed","f1_thresholded","precision_thresholded","recall_thresholded"]]
    values=per_label[["system","label","threshold_mean","threshold_std","threshold_min","threshold_max"]]
    neutral_tex=neutral[["system","neutral_fp_fixed","neutral_fp_thresholded","predicted_cardinality_fixed","predicted_cardinality_thresholded","gold_cardinality"]]
    for name,frame in (("threshold_main_results",main_tex),("threshold_per_label",label_tex),("threshold_values",values),("threshold_neutral_cardinality",neutral_tex)):
        _atomic_text(frame.to_latex(index=False,float_format=lambda x:f"{x:.4f}",caption=None,label=None),report/f"{name}.tex")


def _registry(root: Path, data: dict[str, dict[str, Any]], main: pd.DataFrame) -> None:
    path=root/"reports"/"brighter_ptbr"/"experiments"/"experiment_registry.csv"; old=pd.read_csv(path)
    rows=[]
    for system,display,slug,source in SYSTEMS:
        summary=data[system]["summary"]; source_summary=load_json(root/"reports"/"brighter_ptbr"/"experiments"/f"{source}_results.json"); cfg=source_summary["config"]; row=main[main.system==system].iloc[0]
        rows.append({"experiment_id":f"{system}_NESTED_THRESHOLD_CALIBRATION","model":display,"encoder":cfg["model_name"],"hugging_face_checkpoint":cfg["model_name"],
                     "loss":cfg["loss"],"threshold_strategy":"nested_5x3_per_label_f1","threshold":"per_label_cross_fitted","fold":"all","folds":5,"seed":42,
                     "learning_rate":cfg["training"]["learning_rate"],"epochs":cfg["training"]["epochs"],"physical_batch":cfg["training"]["physical_batch_size"],
                     "accumulation":cfg["training"]["gradient_accumulation_steps"],"effective_batch_size":cfg["training"]["effective_batch_size"],"max_length":cfg["max_length"],
                     "training_time":summary["inner_training_seconds_total"],"macro_f1_mean":row.thresholded_macro_f1,"micro_f1_mean":row.thresholded_micro_f1,
                     "oof_path":summary["thresholded_oof"],"oof_sha256":summary["thresholded_oof_sha256"],"status":"complete","timestamp":summary["completed_timestamp_utc"],
                     "train_sha256":summary["dataset_hashes"]["train"],"dev_sha256":summary["dataset_hashes"]["dev"],"git_commit":"NOT_A_GIT_REPOSITORY","test_policy":"SEALED — NOT LOADED OR EVALUATED"})
    add=pd.DataFrame(rows); old=old[~old.experiment_id.isin(add.experiment_id)]; _atomic_csv(pd.concat([old,add],ignore_index=True),path)


def _dataset_integrity(root: Path) -> dict[str, Any]:
    revision="419566ac8f46f951b51584be36c13e5363008144"; dataset=load_json(root/"metadata"/"brighter_ptbr"/"dataset_manifest.json"); track=load_json(root/"metadata"/"brighter_ptbr"/"track_a_manifest.json"); snapshot=load_json(root/"data"/"raw"/"brighter_ptbr"/revision/"hf_snapshot"/".brighter_snapshot.json")
    checks={}
    for level,records in (("raw",{"train":snapshot["files"]["ptbr/train-00000-of-00001.parquet"],"dev":snapshot["files"]["ptbr/dev-00000-of-00001.parquet"]}),
                          ("interim",{x:dataset["parquet_files"][x]["sha256"] for x in ("train","dev")}),
                          ("curated",{x:track["curated_files"][x]["sha256"] for x in ("train","dev")})):
        checks[level]={}
        for split,expected in records.items():
            if level=="raw": path=root/"data"/"raw"/"brighter_ptbr"/revision/"hf_snapshot"/"ptbr"/f"{split}-00000-of-00001.parquet"
            elif level=="interim": path=root/dataset["parquet_files"][split]["path"]
            else: path=root/track["curated_files"][split]["path"]
            observed=sha256_file(path); checks[level][split]={"expected":expected,"observed":observed,"match":observed==expected}
    if not all(v["match"] for level in checks.values() for v in level.values()): raise RuntimeError("Dataset integrity check failed")
    return {"revision":revision,"verified_train_dev_only":True,"test_content_read":False,"hashes":checks,
            "data_policy":{"text_preprocessing":"none","label_modification":"none","sampling":"none","augmentation":"none","translation":"none","synthetic_data":"none","external_data":"none","track_b":"OUT OF SCOPE BY EXPERIMENTAL DESIGN"}}


def _report(main: pd.DataFrame, per_label: pd.DataFrame, neutral: pd.DataFrame, multilabel: pd.DataFrame,
            bootstrap: pd.DataFrame, pairwise: pd.DataFrame, label_comp: pd.DataFrame, exclusive: pd.DataFrame) -> str:
    t3=main[main.system=="T3"].iloc[0]; t3labels=per_label[per_label.system=="T3"]
    lines=["# BRIGHTER PT-BR — leakage-safe threshold calibration", "", "## 1. Motivation", "", "Evaluate whether per-emotion decision thresholds improve the three selected systems without changing encoders, losses, data, probabilities, or ranking.", "",
           "## 2. Leakage-safe protocol", "", "For each frozen outer fold, thresholds were learned exclusively from three-fold iterative-stratified inner-OOF predictions over outer-train. Outer-validation labels never entered model training, grid search, or tie-breaking. Existing outer OOF probabilities were reused unchanged. The test remained **SEALED — NOT LOADED OR EVALUATED**.", "",
           "## 3. Nested calibration design", "", "The grid was 0.01–0.99 in steps of 0.01. Each label threshold independently maximized its F1; since Macro-F1 is the unweighted mean of six label-wise F1 terms and each term depends only on its label decisions, this separately maximizes every term on calibration data. Numerical ties selected the largest threshold, predeclared to favor fewer false positives. Seed 42 and exact source-system hyperparameters were retained.", "",
           "## 4. Threshold distributions", "", "| System | Label | Mean | SD | Min | Max |", "|---|---|---:|---:|---:|---:|"]
    for r in per_label.itertuples(): lines.append(f"| {r.system} | {r.label} | {r.threshold_mean:.3f} | {r.threshold_std:.3f} | {r.threshold_min:.2f} | {r.threshold_max:.2f} |")
    lines += ["", "## 5. Main results", "", "| System | Fixed Macro-F1 | Thresholded Macro-F1 | Delta | 95% bootstrap CI | Micro-F1 after | Neutral FP after | Cardinality after | disgust F1 | fear F1 | surprise F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in main.itertuples(): lines.append(f"| {r.system} | {r.fixed_macro_f1:.4f} | {r.thresholded_macro_f1:.4f} | {r.delta_macro_f1:+.4f} | [{r.bootstrap_ci_low:+.4f}, {r.bootstrap_ci_high:+.4f}] | {r.thresholded_micro_f1:.4f} | {r.neutral_fp_thresholded:.4f} | {r.predicted_cardinality_thresholded:.4f} | {r.disgust_f1:.4f} | {r.fear_f1:.4f} | {r.surprise_f1:.4f} |")
    lines += ["", "AP was exactly invariant for every system because probabilities were copied unchanged; only binary decisions varied. The earlier published B2 value 0.3958 is the mean of the five outer-fold Macro-F1 values; 0.3976 is the Macro-F1 recomputed once on the concatenated 2,426-row OOF and is therefore the correct paired reference for this cross-fitted comparison.", "", "## 6. Minority labels", "", "| System | Label | AP | F1 fixed → thresholded | Precision after | Recall after | Predicted prevalence after |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in per_label[per_label.label.isin(["disgust","fear","surprise"])].itertuples(): lines.append(f"| {r.system} | {r.label} | {r.average_precision:.4f} | {r.f1_fixed:.4f} → {r.f1_thresholded:.4f} | {r.precision_thresholded:.4f} | {r.recall_thresholded:.4f} | {r.predicted_prevalence_thresholded:.4f} |")
    disgust=t3labels[t3labels.label=="disgust"].iloc[0]; fear=t3labels[t3labels.label=="fear"].iloc[0]; surprise=t3labels[t3labels.label=="surprise"].iloc[0]
    lines += ["", "T3 recovers **22 true-positive disgust examples** (recall 0.2750 of support 80), versus zero at threshold 0.50. Its mean disgust threshold is " f"{disgust.threshold_mean:.3f} ± {disgust.threshold_std:.3f} (range {disgust.threshold_min:.2f}–{disgust.threshold_max:.2f}); predicted prevalence becomes {disgust.predicted_prevalence_thresholded:.4f} against gold {disgust.gold_prevalence:.4f}. This confirms usable ranking signal, but F1 {disgust.f1_thresholded:.4f} remains below T1/T2 and the between-fold range indicates moderate instability.", "",
              f"Fear improves {fear.f1_fixed:.4f} → {fear.f1_thresholded:.4f} at threshold {fear.threshold_mean:.3f} ± {fear.threshold_std:.3f}; surprise improves {surprise.f1_fixed:.4f} → {surprise.f1_thresholded:.4f} at {surprise.threshold_mean:.3f} ± {surprise.threshold_std:.3f}.", "",
              "## 7. ASL behavior", "", "T2 remains the best overall system. Calibration raises majority-label thresholds while keeping minority thresholds near 0.50, reducing ASL overprediction and increasing Macro-F1.", "", "## 8. Weighted Focal behavior", "", "T1's low, highly stable thresholds convert strong minority ranking into recall and F1, but overshoot gold cardinality and sharply increase Neutral FP.", "", "## 9. BERTimbau Large behavior", "", f"Thresholding alone raises T3 Macro-F1 by {t3.delta_macro_f1:+.4f}, close to T1 but below T2. It substantially recovers every minority label, proving that fixed 0.50 hid ranking signal. Nevertheless, specialized losses remain useful: T1/T2 retain higher minority AP/F1, and T2 retains higher overall Macro-F1.", "", "## 10. Neutral examples", "", "| System | Neutral FP fixed → thresholded | Exact neutral accuracy fixed → thresholded |", "|---|---:|---:|"]
    for r in neutral.itertuples(): lines.append(f"| {r.system} | {r.neutral_fp_fixed:.4f} → {r.neutral_fp_thresholded:.4f} | {r.neutral_accuracy_fixed:.4f} → {r.neutral_accuracy_thresholded:.4f} |")
    lines += ["", "## 11. Label cardinality", "", "| System | Fixed | Thresholded | Gold | Absolute error before → after |", "|---|---:|---:|---:|---:|"]
    for r in neutral.itertuples(): lines.append(f"| {r.system} | {r.predicted_cardinality_fixed:.4f} | {r.predicted_cardinality_thresholded:.4f} | {r.gold_cardinality:.4f} | {r.absolute_error_fixed:.4f} → {r.absolute_error_thresholded:.4f} |")
    lines += ["", "## 12. Multilabel-only evaluation", "", "| System | Macro-F1 fixed → after | Micro-F1 fixed → after | Jaccard fixed → after | Subset accuracy fixed → after |", "|---|---:|---:|---:|---:|"]
    for r in multilabel.itertuples(): lines.append(f"| {r.system} | {r.fixed_macro_f1:.4f} → {r.thresholded_macro_f1:.4f} | {r.fixed_micro_f1:.4f} → {r.thresholded_micro_f1:.4f} | {r.fixed_jaccard_samples:.4f} → {r.thresholded_jaccard_samples:.4f} | {r.fixed_subset_accuracy:.4f} → {r.thresholded_subset_accuracy:.4f} |")
    lines += ["", "## 13. Bootstrap comparison", "", "Paired instance bootstrap used 10,000 resamples, seed 42. Intervals are percentile intervals and are reported descriptively, without automatic significance language.", "", "| System | Delta Macro-F1 | 95% CI |", "|---|---:|---:|"]
    for r in bootstrap.itertuples(): lines.append(f"| {r.system} | {r.delta:+.4f} | [{r.ci_2_5:+.4f}, {r.ci_97_5:+.4f}] |")
    lines += ["", "## 14. Threshold stability", "", "T1 is the most stable. T2 is stable for the minority thresholds and moderately variable for anger/joy. T3 has stable fear/surprise thresholds but moderate disgust variability (0.10–0.20) and high anger variability (0.21–0.47), a validity caveat.", "", "## 15. Error complementarity", "", "| Pair | Example disagreement | Label disagreement | A correct/B wrong | B correct/A wrong | Probability corr. | Error corr. | Multilabel disagreement |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pairwise.itertuples(): lines.append(f"| {r.system_a}/{r.system_b} | {r.example_disagreement_rate:.4f} | {r.label_disagreement_rate:.4f} | {r.a_exact_correct_b_wrong} | {r.b_exact_correct_a_wrong} | {r.probability_correlation_flat:.4f} | {r.error_correlation_flat:.4f} | {r.multilabel_example_disagreement_rate:.4f} |")
    lines += ["", "Exclusive exact-correct/error counts:", "", "| System | Exact correct | Exclusively correct | Exclusive error | Multilabel exclusively correct |", "|---|---:|---:|---:|---:|"]
    for r in exclusive.itertuples(): lines.append(f"| {r.system} | {r.exact_correct} | {r.exclusive_exact_correct} | {r.exclusive_error} | {r.multilabel_exclusive_exact_correct} |")
    minority_comp=label_comp[label_comp.label.isin(["disgust","fear","surprise"])]
    lines += ["", "Minority disagreement rates range from " f"{minority_comp.disagreement_rate.min():.4f} to {minority_comp.disagreement_rate.max():.4f}; the nonzero exclusive-correct counts justify studying an ensemble later, but do not by themselves authorize building one.", "", "## 16. Candidate selection", "", "Select **T2 (BERTweetBR + ASL + nested thresholds)** as the primary system and **T3 (BERTimbau Large + BCE + nested thresholds)** as the complementary secondary candidate. T2 has the best Macro-F1 and best balance of neutral/cardinality behavior. T1 exceeds T3 in Macro-F1 and minority AP/F1 at much lower calibration cost, but its Neutral FP (0.6180) and cardinality (1.2370) are materially worse. T3 is more complementary to T2: it is exclusively exact-correct on 202 examples overall (versus 121 for T1) and wins 284 paired exact decisions where T2 fails (versus 203 for T1), while retaining better neutral/cardinality behavior. T1 remains an important calibrated ablation, but not one of the two selected systems.", "", "## 17. Next experiment", "", "Recommendation only — **do not execute automatically**: freeze T2 as the primary system and perform a narrowly scoped, leakage-safe ensemble feasibility study using T2/T3 predictions. T3's additional exclusive correct decisions and lower error correlation with T2 justify this analysis despite its higher computational cost. Do not train BERTimbau Large + ASL/Weighted Focal unless a later, separately authorized study specifically targets encoder–loss interaction.", "", "## Validation", "", "All thresholded OOF files contain 2,426 unique ordered IDs with identical outer folds and bitwise-identical source probabilities. Train/dev raw, interim, and curated hashes match frozen records. No preprocessing, label modification, sampling, augmentation, translation, synthetic/external data, or Track B was used. Test policy: **SEALED — NOT LOADED OR EVALUATED**.", ""]
    return "\n".join(lines)


def generate_threshold_reports(root: Path) -> dict[str, Any]:
    data=_load(root); main,thresholds,per_label,neutral,multilabel,bootstrap=_tables(data); pairwise,label_comp,exclusive=_complementarity(data)
    report=root/"reports"/"brighter_ptbr"/"experiments"
    for name,frame in (("threshold_main_results",main),("thresholds_by_fold",thresholds),("threshold_per_label",per_label),("threshold_neutral_cardinality",neutral),("threshold_multilabel",multilabel),("threshold_bootstrap",bootstrap),("threshold_complementarity_pairs",pairwise),("threshold_complementarity_per_label",label_comp),("threshold_complementarity_exclusive",exclusive)):
        _atomic_csv(frame,report/f"{name}.csv")
    integrity=_dataset_integrity(root); write_json_atomic(root/"metadata"/"brighter_ptbr"/"experiments"/"threshold_dataset_integrity.json",integrity)
    _figures(root,main,thresholds,per_label,neutral,pairwise); _latex(root,main,per_label,thresholds,neutral); _registry(root,data,main)
    _atomic_text(_report(main,per_label,neutral,multilabel,bootstrap,pairwise,label_comp,exclusive),report/"threshold_calibration_results.md")
    summary={"development_examples":DEV_POOL_SIZE,"outer_folds":5,"inner_folds":3,"threshold_grid":[.01,.99,.01],"bootstrap_samples":10_000,"bootstrap_seed":42,"test_policy":"SEALED — NOT LOADED OR EVALUATED","dataset_integrity":integrity,"selected_systems":["T2","T3"],"main_results":main.to_dict(orient="records"),"per_label":per_label.to_dict(orient="records"),"neutral_cardinality":neutral.to_dict(orient="records"),"multilabel_only":multilabel.to_dict(orient="records"),"bootstrap":bootstrap.to_dict(orient="records"),"complementarity":{"pairs":pairwise.to_dict(orient="records"),"per_label":label_comp.to_dict(orient="records"),"exclusive":exclusive.to_dict(orient="records")},"systems":{system:data[system]["summary"] for system,_,_,_ in SYSTEMS}}
    write_json_atomic(root/"metadata"/"brighter_ptbr"/"experiments"/"threshold_calibration_summary.json",summary)
    return summary
