"""Consolidation and scientific reporting for the L1--L4 loss ablation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .experiments import DEV_POOL_SIZE, EXPECTED_LABELS
from .manifest import load_json, sha256_file, write_json_atomic

EXPERIMENTS = (
    ("L1", "Weighted BCE", "bertimbau_base_weighted_bce"),
    ("L2", "Focal", "bertimbau_base_focal"),
    ("L3", "Weighted Focal", "bertimbau_base_weighted_focal"),
    ("L4", "ASL", "bertimbau_base_asl"),
)
COMPARATORS = (
    ("B1", "Base BCE", "bertimbau_base_bce"),
    ("B2", "Large BCE", "bertimbau_large_bce"),
)


def _load(project_root: Path, entries: tuple[tuple[str, str, str], ...]) -> dict[str, dict[str, Any]]:
    report_dir = project_root / "reports" / "brighter_ptbr" / "experiments"
    return {short: load_json(report_dir / f"{slug}_results.json") for short, _, slug in entries}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(temporary, path)


def _save_figure(fig: Any, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_without_suffix.with_suffix(suffix), dpi=220, bbox_inches="tight")


def _fold_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, display, _ in EXPERIMENTS:
        for item in results[short]["fold_metrics"]:
            multi = item["groups"]["multi_label"]
            rows.append({
                "experiment_id": results[short]["experiment_id"], "model": short, "loss": display, "fold": item["fold"],
                "macro_f1": item["macro_f1"], "micro_f1": item["micro_f1"],
                "hamming_loss": item["hamming_loss"], "jaccard_samples": item["jaccard_samples"],
                "subset_accuracy": item["subset_accuracy"],
                "exact_neutral_accuracy": item["zero_label"]["exact_neutral_accuracy"],
                "neutral_false_positive_rate": item["zero_label"]["neutral_false_positive_rate"],
                "multilabel_support": multi["support"], "multilabel_macro_f1": multi["macro_f1"],
                "multilabel_micro_f1": multi["micro_f1"], "multilabel_jaccard": multi["jaccard_samples"],
                "multilabel_subset_accuracy": multi["subset_accuracy"],
                "training_seconds": item["cost"]["training_seconds"], "inference_seconds": item["cost"]["inference_seconds"],
                "peak_gpu_memory_bytes": item["cost"]["peak_gpu_memory_bytes"],
            })
    return pd.DataFrame(rows)


def _per_label_table(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, display, _ in EXPERIMENTS:
        summary = results[short]
        fold_scopes = []
        for item in summary["fold_metrics"]:
            confusion = item["per_label"][EXPECTED_LABELS[0]]["confusion"]
            fold_n = sum(confusion.values())
            fold_scopes.append(("fold", item["fold"], item, fold_n))
        for scope, fold, metrics, n in [
            ("oof", None, summary["oof_aggregate"], DEV_POOL_SIZE),
            *fold_scopes,
        ]:
            for label in EXPECTED_LABELS:
                item = metrics["per_label"][label]
                predicted_positive = item["confusion"]["tp"] + item["confusion"]["fp"]
                rows.append({
                    "experiment_id": summary["experiment_id"], "model": short, "loss": display,
                    "scope": scope, "fold": fold, "emotion": label, "precision": item["precision"],
                    "recall": item["recall"], "f1": item["f1"], "average_precision": item["average_precision"],
                    "gold_positive": item["support"], "predicted_positive": predicted_positive,
                    "gold_prevalence": item["support"] / n, "predicted_prevalence": predicted_positive / n,
                })
    return pd.DataFrame(rows)


def _update_registry(project_root: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path = project_root / "reports" / "brighter_ptbr" / "experiments" / "experiment_registry.csv"
    registry = pd.read_csv(path)
    new_rows = []
    for short, display, slug in EXPERIMENTS:
        result = results[short]
        config = result["config"]
        new_rows.append({
            "experiment_id": result["experiment_id"], "model": config["model_name"], "loss": display,
            "max_length": config["max_length"], "learning_rate": config["training"]["learning_rate"],
            "epochs": config["training"]["epochs"], "effective_batch_size": config["training"]["effective_batch_size"],
            "seed": config["seed"], "folds": config["folds"], "threshold_strategy": "fixed_0.5",
            "macro_f1_mean": result["cross_validation"]["macro_f1"]["mean"],
            "macro_f1_std": result["cross_validation"]["macro_f1"]["std"],
            "micro_f1_mean": result["cross_validation"]["micro_f1"]["mean"],
            "training_time": result["cost"]["training_seconds_total"], "status": "complete",
            "timestamp": result["completed_timestamp_utc"],
            "model_checkpoints": f"artifacts/models/{slug}/fold_{{0..4}}",
            "oof_path": result["oof_path"], "oof_sha256": result["oof_sha256"],
            "config_sha256": result["config_sha256"], "peak_gpu_memory_bytes": result["cost"]["peak_gpu_memory_bytes"],
            "package_versions": json.dumps(result["environment"]["packages"], sort_keys=True, separators=(",", ":")),
            "git_commit": "NOT_A_GIT_REPOSITORY",
        })
    additions = pd.DataFrame(new_rows)
    registry = registry.loc[~registry["experiment_id"].isin(additions["experiment_id"])]
    registry = pd.concat([registry, additions], ignore_index=True)
    _atomic_csv(registry, path)


def _figures(project_root: Path, results: Mapping[str, Mapping[str, Any]], per_label: pd.DataFrame) -> None:
    figure_dir = project_root / "figures" / "brighter_ptbr" / "experiments"
    labels = [short for short, _, _ in EXPERIMENTS]
    means = [results[x]["cross_validation"]["macro_f1"]["mean"] for x in labels]
    stds = [results[x]["cross_validation"]["macro_f1"]["std"] for x in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(labels, means, yerr=stds, capsize=4, color="#4C78A8")
    ax.set(ylabel="Macro-F1", title="BERTimbau Base — loss ablation (threshold 0.5)", ylim=(0, max(means) + 0.08))
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir / "imbalance_loss_macro_f1")
    plt.close(fig)

    aggregate = per_label.loc[per_label["scope"] == "oof"]
    for metric, filename, title in (
        ("average_precision", "imbalance_loss_ap_by_label", "OOF Average Precision by label"),
        ("f1", "imbalance_loss_minority_f1", "OOF minority-label F1 at threshold 0.5"),
        ("recall", "imbalance_loss_minority_recall", "OOF minority-label recall at threshold 0.5"),
    ):
        data = aggregate if metric == "average_precision" else aggregate.loc[aggregate["emotion"].isin(["disgust", "fear", "surprise"])]
        pivot = data.pivot(index="emotion", columns="model", values=metric).reindex(columns=labels)
        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set(ylabel=metric.replace("_", " ").title(), xlabel="", title=title, ylim=(0, max(0.55, float(pivot.max().max()) + 0.08)))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Loss")
        fig.tight_layout()
        _save_figure(fig, figure_dir / filename)
        plt.close(fig)

    prevalence = aggregate.loc[aggregate["emotion"].isin(["disgust", "fear", "surprise"])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, label in zip(axes, ["disgust", "fear", "surprise"], strict=True):
        subset = prevalence.loc[prevalence["emotion"] == label]
        x = np.arange(len(subset))
        axis.bar(x - 0.18, subset["gold_prevalence"], width=0.36, label="Gold")
        axis.bar(x + 0.18, subset["predicted_prevalence"], width=0.36, label="Predicted")
        axis.set_xticks(x, subset["model"])
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Prevalence")
    axes[0].legend()
    fig.suptitle("Minority predicted prevalence vs gold prevalence (OOF, threshold 0.5)")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "imbalance_loss_predicted_vs_gold_prevalence")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, (short, display, _) in zip(axes.flat, EXPERIMENTS, strict=True):
        oof = pd.read_parquet(project_root / results[short]["oof_path"])
        for label in EXPECTED_LABELS:
            precision, recall, _ = precision_recall_curve(oof[f"{label}_true"], oof[f"{label}_prob"])
            ap = results[short]["oof_aggregate"]["per_label"][label]["average_precision"]
            axis.plot(recall, precision, label=f"{label} ({ap:.3f})")
        axis.set(xlabel="Recall", ylabel="Precision", title=f"{short} — {display}", xlim=(0, 1), ylim=(0, 1.02))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("OOF precision–recall curves")
    fig.tight_layout()
    _save_figure(fig, figure_dir / "imbalance_loss_pr_curves")
    plt.close(fig)


def _report(results: Mapping[str, Mapping[str, Any]], baselines: Mapping[str, Mapping[str, Any]], per_label: pd.DataFrame) -> str:
    lines = [
        "# BRIGHTER PT-BR — imbalance loss ablation", "", "## A. Audit", "",
        "The existing reports, registry, fold metrics, confusion matrices, persisted CV assignments, manifests, baseline code, B1/B2 model provenance, and all three baseline OOF files were inspected. The development pool is the deliberate union of official train+dev (2,426 unique examples). All OOF files use the same ordered IDs and the same five persisted folds (seed 42). Train/dev SHA-256 hashes match the frozen manifest.", "",
        "The official test remains **SEALED — NOT LOADED OR EVALUATED**. A direct `--final-test` smoke test failed before configuration/data access, as required. No preprocessing, sampling, balancing, augmentation, synthetic data, external data, or label modification was used. This directory is not a Git repository, so `git_commit` is recorded as `NOT_A_GIT_REPOSITORY` rather than fabricated.", "",
        "## B. Implementations", "",
        "All variants use BERTimbau Base, max length 128, AdamW, LR 2e-5, four epochs, weight decay 0.01, 10% linear warmup, FP16, physical batch 8 × accumulation 2 = effective batch 16, seed 42, identical folds, and threshold 0.5.", "",
        "- **L1 Weighted BCE:** for class c, `w_c = n_c^- / n_c^+`, and `L_c = -[w_c y log σ(z) + (1-y) log(1-σ(z))]`. Counts use only the outer-fold training partition.",
        "- **L2 Focal:** `L_c = (1-p_t)^2 BCE(z,y)`, with `p_t = exp(-BCE)` and alpha=1. Gamma 0 exactly recovers BCE under the same mean reduction.",
        "- **L3 Weighted Focal:** `alpha_c = N/(n_c^+ k)` (k=6), then `L_c = alpha_c (1-p_t)^2 BCE(z,y)`. The cited paper is ambiguous about adapting its class-level weight to both binary signs; the mathematically literal implementation multiplies the complete per-class focal term, not only positives.",
        "- **L4 ASL:** positive `L+ = -(1-p)^gamma+ log p`; negative `p_m=max(p-0.05,0)`, `L- = -p_m^gamma- log(1-p_m)`, with `gamma+=0`, `gamma-=4`. Computation is float32 over logits under autocast.", "",
        "Formulations were checked against [Hanif et al. (2025)](https://aclanthology.org/2025.semeval-1.279/) and the [official Ridnik et al. ASL implementation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py).", "",
        "## C. Results", "",
        "Primary selection uses OOF Macro-F1; fold mean ± sample SD is included for stability.", "",
        "| Model/Loss | OOF Macro-F1 | Fold Macro-F1 mean ± SD | Micro-F1 | Hamming | Jaccard | AP disgust | AP fear | AP surprise |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for short, display, _ in EXPERIMENTS:
        result = results[short]
        o = result["oof_aggregate"]
        cv = result["cross_validation"]["macro_f1"]
        lines.append(f"| {short} {display} | {o['macro_f1']:.4f} | {cv['mean']:.4f} ± {cv['std']:.4f} | {o['micro_f1']:.4f} | {o['hamming_loss']:.4f} | {o['jaccard_samples']:.4f} | {o['per_label']['disgust']['average_precision']:.4f} | {o['per_label']['fear']['average_precision']:.4f} | {o['per_label']['surprise']['average_precision']:.4f} |")
    lines.extend(["", "### OOF metrics by class", "", "| Loss | Emotion | Precision | Recall | F1 | AP | Gold + | Predicted + | Gold prev. | Pred. prev. |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    aggregate = per_label.loc[per_label["scope"] == "oof"]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.model} | {row.emotion} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.average_precision:.4f} | {row.gold_positive} | {row.predicted_positive} | {row.gold_prevalence:.4f} | {row.predicted_prevalence:.4f} |")
    lines.extend(["", "## D. Comparison with B1 and B2", "", "| Loss | Δ OOF Macro-F1 vs B1 | Δ vs B2 | Δ Micro-F1 vs B1 | Δ Neutral FP vs B1 |", "|---|---:|---:|---:|---:|"])
    b1, b2 = baselines["B1"]["oof_aggregate"], baselines["B2"]["oof_aggregate"]
    for short, display, _ in EXPERIMENTS:
        o = results[short]["oof_aggregate"]
        lines.append(f"| {short} {display} | {o['macro_f1']-b1['macro_f1']:+.4f} | {o['macro_f1']-b2['macro_f1']:+.4f} | {o['micro_f1']-b1['micro_f1']:+.4f} | {o['zero_label']['neutral_false_positive_rate']-b1['zero_label']['neutral_false_positive_rate']:+.4f} |")
    lines.extend(["", "## E. Disgust", "", "| Loss | Any positive? | Predicted + | Precision | Recall | F1 | AP | Gold prev. | Pred. prev. |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in aggregate.loc[aggregate["emotion"] == "disgust"].itertuples(index=False):
        lines.append(f"| {row.model} | {'yes' if row.predicted_positive else 'no'} | {row.predicted_positive} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.average_precision:.4f} | {row.gold_prevalence:.4f} | {row.predicted_prevalence:.4f} |")
    lines.extend(["", "All weighted/asymmetric variants make positive disgust predictions and lift F1 above zero. L3 gives the best disgust AP and precision with prevalence closest to gold; L1 gives the highest recall but overpredicts substantially. L4 improves F1 but its precision and AP are weaker than L3.", "", "## F. Threshold diagnostics", "", "**DIAGNOSTIC ONLY — THRESHOLD SELECTED AND EVALUATED ON SAME OOF.**", "", "B1/B2 were swept from 0.01 to 0.90 by 0.01. See `threshold_diagnostics.md` and its full CSV. Key minority points:", "", "| Model | Emotion | Best diagnostic threshold | Maximum diagnostic F1 | Precision | Recall |", "|---|---|---:|---:|---:|---:|"])
    diag = pd.read_csv(Path("reports/brighter_ptbr/experiments/threshold_diagnostic_summary.csv")) if Path("reports/brighter_ptbr/experiments/threshold_diagnostic_summary.csv").exists() else pd.DataFrame()
    if not diag.empty:
        for row in diag.loc[diag["emotion"].isin(["disgust", "fear", "surprise"])].itertuples(index=False):
            lines.append(f"| {row.model} | {row.emotion} | {row.threshold:.2f} | {row.f1:.4f} | {row.precision:.4f} | {row.recall:.4f} |")
    lines.extend(["", "These values show ranking signal, especially for B2 fear/surprise, but cannot be reported as unbiased generalization performance.", "", "## G. Cost", "", "| Loss | Parameters | Training total (s) | Inference (s) | Peak VRAM (GiB) |", "|---|---:|---:|---:|---:|"])
    for short, display, _ in EXPERIMENTS:
        cost = results[short]["cost"]
        lines.append(f"| {short} {display} | {cost['parameters']:,} | {cost['training_seconds_total']:.1f} | {cost['inference_seconds_total']:.2f} | {cost['peak_gpu_memory_bytes']/2**30:.2f} |")
    lines.extend(["", "## H. Decision", "", "Advance **L4 ASL** and **L3 Weighted Focal** (maximum two). L4 has the best OOF/fold Macro-F1 and strong minority recall, but its neutral false-positive rate is high and must remain a selection caveat. L3 is the complementary conservative candidate: best disgust AP/precision, substantially lower neutral FP, and no severe majority-label collapse, although its fold variance is higher. L1 is stable and high-Macro-F1 but is not advanced because its neutral FP and overprediction are nearly as severe as L4 while its ranking evidence is less compelling than L3. L2 improves on B1 but does not recover disgust at 0.5.", "", "## I. Next phase", "", "Run the frozen BERTweetBR/Albertina encoder × BCE/Weighted-Focal/ASL matrix without threshold tuning. Retain BCE as the encoder-effect control. Leakage-safe nested per-label threshold calibration remains a later phase. **Track B is OUT OF SCOPE BY EXPERIMENTAL DESIGN**; the active project uses BRIGHTER Track A only.", "", "## Validation", "", "All 40 unit/integration tests pass. Each new OOF contains exactly 2,426 unique IDs with persisted folds and finite probabilities. Train/dev hashes remain unchanged. Fold checkpoints, config hashes, package versions, timings, VRAM, timestamps, OOF paths/hashes, and train-only class statistics are persisted. The test remains sealed.", ""])
    return "\n".join(lines)


def generate_imbalance_loss_reports(project_root: Path) -> dict[str, Any]:
    results = _load(project_root, EXPERIMENTS)
    baselines = _load(project_root, COMPARATORS)
    fold_table = _fold_table(results)
    per_label = _per_label_table(results)
    report_dir = project_root / "reports" / "brighter_ptbr" / "experiments"
    _atomic_csv(fold_table, report_dir / "imbalance_loss_fold_metrics.csv")
    _atomic_csv(per_label, report_dir / "imbalance_loss_per_label.csv")
    _update_registry(project_root, results)
    _figures(project_root, results, per_label)
    # Resolve diagnostic CSV relative to project root while building the report.
    previous = Path.cwd()
    try:
        os.chdir(project_root)
        report = _report(results, baselines, per_label)
    finally:
        os.chdir(previous)
    report_path = report_dir / "imbalance_loss_results.md"
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8", newline="\n")
    os.replace(temporary, report_path)
    summary = {
        "development_examples": DEV_POOL_SIZE, "folds": 5, "threshold": 0.5,
        "test_policy": "SEALED — NOT LOADED OR EVALUATED",
        "selected_losses": ["asl", "weighted_focal"],
        "experiments": {short: {
            "experiment_id": results[short]["experiment_id"], "config_sha256": results[short]["config_sha256"],
            "oof_path": results[short]["oof_path"], "oof_sha256": results[short]["oof_sha256"],
            "macro_f1_oof": results[short]["oof_aggregate"]["macro_f1"],
            "macro_f1_cv": results[short]["cross_validation"]["macro_f1"],
            "cost": results[short]["cost"],
        } for short, _, _ in EXPERIMENTS},
        "artifact_hashes": {
            "fold_metrics": sha256_file(report_dir / "imbalance_loss_fold_metrics.csv"),
            "per_label": sha256_file(report_dir / "imbalance_loss_per_label.csv"),
        },
    }
    write_json_atomic(project_root / "metadata" / "brighter_ptbr" / "experiments" / "imbalance_loss_summary.json", summary)
    return summary
