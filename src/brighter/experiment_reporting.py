"""Publication-ready consolidation of BRIGHTER baseline OOF results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .curation import FROZEN_REVISION
from .experiments import DEV_POOL_SIZE, EXPECTED_LABELS
from .manifest import load_json, write_json_atomic

EXPERIMENTS = (
    ("B0_TFIDF_LOGREG", "tfidf_logreg", "TF-IDF + Logistic Regression"),
    ("B1_BERTIMBAU_BASE_BCE", "bertimbau_base_bce", "BERTimbau Base + BCE"),
    ("B2_BERTIMBAU_LARGE_BCE", "bertimbau_large_bce", "BERTimbau Large + BCE"),
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_text(path, frame.to_csv(index=False, float_format="%.8f", lineterminator="\n"))


def _latex_escape(value: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(character, character) for character in str(value))


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], alignment: str | None = None) -> str:
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule", " & ".join(_latex_escape(item) for item in headers) + " \\\\", "\\midrule"]
    lines.extend(" & ".join(_latex_escape(item) for item in row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        metadata = {"Creator": "BRIGHTER PT-BR baseline experiments"}
        if suffix == ".pdf":
            metadata["CreationDate"] = None
        figure.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white", metadata=metadata)


def _load_results(project_root: Path) -> list[dict[str, Any]]:
    report_dir = project_root / "reports" / "brighter_ptbr" / "experiments"
    results = []
    for experiment_id, slug, display in EXPERIMENTS:
        path = report_dir / f"{slug}_results.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed experiment: {experiment_id}")
        item = load_json(path)
        if item["development_examples"] != DEV_POOL_SIZE or item["test_policy"] != "SEALED — NOT LOADED OR EVALUATED":
            raise RuntimeError(f"Invalid development/test protocol in {path}")
        item["display_name"] = display
        results.append(item)
    return results


def _load_oof(project_root: Path, result: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.read_parquet(project_root / result["oof_path"])
    if len(frame) != DEV_POOL_SIZE or frame["id"].duplicated().any():
        raise RuntimeError(f"Incomplete OOF file for {result['experiment_id']}")
    return frame


def _overall_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        cv = result["cross_validation"]
        rows.append({
            "experiment_id": result["experiment_id"], "model": result["display_name"],
            "macro_f1_mean": cv["macro_f1"]["mean"], "macro_f1_std": cv["macro_f1"]["std"],
            "micro_f1_mean": cv["micro_f1"]["mean"], "micro_f1_std": cv["micro_f1"]["std"],
            "macro_precision_mean": cv["macro_precision"]["mean"], "macro_recall_mean": cv["macro_recall"]["mean"],
            "hamming_loss_mean": cv["hamming_loss"]["mean"], "jaccard_mean": cv["jaccard_samples"]["mean"],
            "subset_accuracy_mean": cv["subset_accuracy"]["mean"],
        })
    return pd.DataFrame(rows)


def _per_label_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        for label in EXPECTED_LABELS:
            metrics = result["oof_aggregate"]["per_label"][label]
            rows.append({"experiment_id": result["experiment_id"], "model": result["display_name"], "emotion": label, "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "support": metrics["support"], "average_precision": metrics["average_precision"]})
    return pd.DataFrame(rows)


def _confusion_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        for label in EXPECTED_LABELS:
            confusion = result["oof_aggregate"]["per_label"][label]["confusion"]
            rows.append({"experiment_id": result["experiment_id"], "model": result["display_name"], "emotion": label, **confusion})
    return pd.DataFrame(rows)


def _cost_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        cost = result["cost"]
        config = result["config"]
        training = config.get("training", {})
        rows.append({
            "experiment_id": result["experiment_id"], "model": result["display_name"],
            "parameters": cost["parameters"], "macro_f1_mean": result["cross_validation"]["macro_f1"]["mean"],
            "training_seconds": cost["training_seconds_total"], "inference_seconds": cost["inference_seconds_total"],
            "inference_examples_per_second": DEV_POOL_SIZE / cost["inference_seconds_total"],
            "peak_vram_bytes": cost["peak_gpu_memory_bytes"],
            "physical_batch_size": training.get("physical_batch_size"), "gradient_accumulation": training.get("gradient_accumulation_steps"),
            "effective_batch_size": training.get("effective_batch_size"),
        })
    return pd.DataFrame(rows)


def _registry(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        config = result["config"]
        training = config.get("training", {})
        rows.append({
            "experiment_id": result["experiment_id"], "model": config["model_name"], "loss": config["loss"],
            "max_length": config.get("max_length"), "learning_rate": training.get("learning_rate"), "epochs": training.get("epochs"),
            "effective_batch_size": training.get("effective_batch_size"), "seed": config["seed"], "folds": config["folds"],
            "threshold_strategy": "fixed_0.5", "macro_f1_mean": result["cross_validation"]["macro_f1"]["mean"],
            "macro_f1_std": result["cross_validation"]["macro_f1"]["std"], "micro_f1_mean": result["cross_validation"]["micro_f1"]["mean"],
            "training_time": result["cost"]["training_seconds_total"], "status": "complete", "timestamp": result["completed_timestamp_utc"],
        })
    return pd.DataFrame(rows)


def _fold_metrics_table(results: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        for item in result["fold_metrics"]:
            cost = item["cost"]
            rows.append({
                "experiment_id": result["experiment_id"], "fold": item["fold"],
                "macro_f1": item["macro_f1"], "micro_f1": item["micro_f1"],
                "macro_precision": item["macro_precision"], "macro_recall": item["macro_recall"],
                "hamming_loss": item["hamming_loss"], "jaccard_samples": item["jaccard_samples"],
                "subset_accuracy": item["subset_accuracy"],
                "exact_neutral_accuracy": item["zero_label"]["exact_neutral_accuracy"],
                "neutral_false_positive_rate": item["zero_label"]["neutral_false_positive_rate"],
                "training_seconds": cost["training_seconds"], "inference_seconds": cost["inference_seconds"],
                "inference_examples_per_second": cost["examples_per_second"],
                "peak_gpu_memory_bytes": cost.get("peak_gpu_memory_bytes"),
            })
    return pd.DataFrame(rows)


def _generate_figures(project_root: Path, results: Sequence[Mapping[str, Any]], overall: pd.DataFrame, per_label: pd.DataFrame) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache.resolve())
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    figure_dir = project_root / "figures" / "brighter_ptbr" / "experiments"
    x = np.arange(len(overall))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, overall["macro_f1_mean"], yerr=overall["macro_f1_std"], capsize=5)
    ax.set_xticks(x, ["B0", "B1", "B2"])
    ax.set_ylabel("Macro-F1 (5-fold mean ± SD)")
    ax.set_title("Baseline Macro-F1 at threshold 0.5")
    ax.set_ylim(0, max(overall["macro_f1_mean"] + overall["macro_f1_std"]) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "baseline_macro_f1")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    width = 0.25
    positions = np.arange(len(EXPECTED_LABELS))
    for index, (_, _, display) in enumerate(EXPERIMENTS):
        values = per_label.loc[per_label["model"] == display].set_index("emotion").loc[list(EXPECTED_LABELS), "f1"]
        ax.bar(positions + (index - 1) * width, values, width=width, label=f"B{index}")
    ax.set_xticks(positions, EXPECTED_LABELS)
    ax.set_ylabel("OOF F1")
    ax.set_title("Per-label baseline F1 at threshold 0.5")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "baseline_per_label_f1")
    plt.close(fig)

    for result in results[1:]:
        oof = _load_oof(project_root, result)
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        for label in EXPECTED_LABELS:
            precision, recall, _ = precision_recall_curve(oof[f"{label}_true"], oof[f"{label}_prob"])
            ap = result["oof_aggregate"]["per_label"][label]["average_precision"]
            ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{result['display_name']} — OOF precision–recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figure_slug = result["artifact_slug"].removesuffix("_bce")
        _save_figure(fig, figure_dir / f"{figure_slug}_pr_curves")
        plt.close(fig)


def _write_latex(report_dir: Path, overall: pd.DataFrame, per_label: pd.DataFrame, cost: pd.DataFrame) -> None:
    latex = report_dir / "latex"
    _write_text(latex / "overall_baselines.tex", _latex_table(
        ["Model", "Macro-F1 mean (SD)", "Micro-F1", "Hamming loss", "Jaccard"],
        [[row.experiment_id.split("_")[0], f"{row.macro_f1_mean:.4f} ({row.macro_f1_std:.4f})", f"{row.micro_f1_mean:.4f}", f"{row.hamming_loss_mean:.4f}", f"{row.jaccard_mean:.4f}"] for row in overall.itertuples(index=False)],
    ))
    _write_text(latex / "per_label_baselines.tex", _latex_table(
        ["Model", "Emotion", "Precision", "Recall", "F1", "Support"],
        [[row.experiment_id.split("_")[0], row.emotion, f"{row.precision:.4f}", f"{row.recall:.4f}", f"{row.f1:.4f}", row.support] for row in per_label.itertuples(index=False)],
    ))
    _write_text(latex / "computational_cost.tex", _latex_table(
        ["Model", "Parameters", "Train (s)", "Inference (s)", "Peak VRAM (GiB)"],
        [[row.experiment_id.split("_")[0], "--" if pd.isna(row.parameters) else f"{int(row.parameters):,}", f"{row.training_seconds:.1f}", f"{row.inference_seconds:.2f}", "--" if pd.isna(row.peak_vram_bytes) else f"{row.peak_vram_bytes / 2**30:.2f}"] for row in cost.itertuples(index=False)],
    ))


def _report(results: Sequence[Mapping[str, Any]], overall: pd.DataFrame, per_label: pd.DataFrame, cost: pd.DataFrame) -> str:
    lines = [
        "# BRIGHTER PT-BR — First baseline battery", "", "## 1. Experimental protocol", "",
        f"Track A at frozen revision `{FROZEN_REVISION}`. The development pool contains official train+dev ({DEV_POOL_SIZE} examples). The official test set is **SEALED — NOT LOADED OR EVALUATED**.", "",
        "Raw `text_raw` was used without manual lowercasing, cleaning, normalization, balancing, sampling, augmentation, class weights, or threshold optimization.", "",
        "## 2. Cross-validation strategy", "", "Five persisted iterative multilabel-stratified folds use seed 42. Every development ID has exactly one OOF prediction from each model; train/validation IDs and exact texts are disjoint in every fold. See `cv_distribution.md`.", "",
        "## 3. Models", "", "- B0: word (1–2) + character `char_wb` (3–5) TF-IDF, One-vs-Rest Logistic Regression.", "- B1: `neuralmind/bert-base-portuguese-cased`, six independent logits, unweighted BCE.", "- B2: `neuralmind/bert-large-portuguese-cased`, same objective and folds.", "",
        "## 4. Training configuration", "", "Transformers: max length 128, dynamic padding, AdamW, learning rate 2e-5, four epochs, weight decay 0.01, 10% linear warmup, fp16, effective batch size 16. Base uses physical batch 8 × accumulation 2; Large uses physical batch 1 × accumulation 16.", "",
        "## 5. Overall results", "", "| Model | Macro-F1 mean ± SD | Micro-F1 mean | Macro precision | Macro recall | Hamming loss | Jaccard |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(f"| {row.experiment_id.split('_')[0]} | {row.macro_f1_mean:.4f} ± {row.macro_f1_std:.4f} | {row.micro_f1_mean:.4f} | {row.macro_precision_mean:.4f} | {row.macro_recall_mean:.4f} | {row.hamming_loss_mean:.4f} | {row.jaccard_mean:.4f} |")
    lines.extend(["", "All values use the fixed threshold 0.5. Mean and sample SD are calculated over the five validation folds.", "", "## 6. Per-label results", "", "| Model | Emotion | Precision | Recall | F1 | Support | AP |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in per_label.itertuples(index=False):
        lines.append(f"| {row.experiment_id.split('_')[0]} | {row.emotion} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.support} | {row.average_precision:.4f} |")
    lines.extend(["", "## 7. Zero-label behavior", "", "| Model | Zero support | Exact Neutral Accuracy | Neutral False Positive Rate |", "|---|---:|---:|---:|"])
    for result in results:
        zero = result["oof_aggregate"]["zero_label"]
        lines.append(f"| {result['experiment_id'].split('_')[0]} | {zero['support']} | {zero['exact_neutral_accuracy']:.4f} | {zero['neutral_false_positive_rate']:.4f} |")
    lines.extend(["", "## 8. Multi-label behavior", "", "| Model | Group | N | Subset accuracy | Jaccard | Hamming loss |", "|---|---|---:|---:|---:|---:|"])
    for result in results:
        for group in ("single_label", "multi_label"):
            item = result["oof_aggregate"]["groups"][group]
            lines.append(f"| {result['experiment_id'].split('_')[0]} | {group} | {item['support']} | {item['subset_accuracy']:.4f} | {item['jaccard_samples']:.4f} | {item['hamming_loss']:.4f} |")
    lines.extend(["", "## 9. Precision–Recall analysis", "", "OOF precision–recall curves and per-label Average Precision were generated for Base and Large. They are diagnostic only; no thresholds were selected.", "", "## 10. Computational cost", "", "| Model | Parameters | Training (s) | Inference (s) | Peak VRAM (GiB) |", "|---|---:|---:|---:|---:|"])
    for row in cost.itertuples(index=False):
        parameters = "n/a" if pd.isna(row.parameters) else f"{int(row.parameters):,}"
        vram = "n/a" if pd.isna(row.peak_vram_bytes) else f"{row.peak_vram_bytes / 2**30:.2f}"
        lines.append(f"| {row.experiment_id.split('_')[0]} | {parameters} | {row.training_seconds:.1f} | {row.inference_seconds:.2f} | {vram} |")
    minority = per_label.loc[per_label["emotion"].isin(["disgust", "fear", "surprise"])]
    detected = minority.loc[minority["f1"] > 0]
    lines.extend(["", "GPU timings are hardware-specific. Seeds and deterministic settings were fixed, but PyTorch reported a non-bitwise-deterministic memory-efficient attention backward kernel; exact reproduction can vary across GPU/driver/library stacks.", "", "## 11. Observations", ""])
    best = overall.sort_values("macro_f1_mean", ascending=False).iloc[0]
    lines.append(f"- The highest five-fold Macro-F1 in this battery is {best['experiment_id'].split('_')[0]} at {best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}.")
    if detected.empty:
        lines.append("- At threshold 0.5, none of the three minority labels obtained non-zero OOF F1 in any baseline aggregate.")
    else:
        detected_text = ", ".join(f"{row.experiment_id.split('_')[0]}:{row.emotion}={row.f1:.3f}" for row in detected.itertuples(index=False))
        lines.append(f"- Minority-label OOF F1 above zero was observed for {detected_text}; remaining zeros must be interpreted jointly with AP/PR curves, not as absence of ranking signal.")
    lines.extend(["- Differences between thresholded F1 and Average Precision indicate that ranking quality and the fixed operating point must be kept conceptually separate.", "- Neutral and genuinely multilabel cases exhibit different failure profiles; both should remain explicit in later ablations.", "", "## 12. Next experimental questions", "", "The current evidence motivates controlled train/dev-only ablations of loss weighting, alternative losses, threshold calibration, maximum length, and OOF-based calibration. It does not establish that any particular method will improve performance. The official test remains sealed until the final protocol is frozen.", "", "### Data modifications", "", "- preprocessing: NONE", "- balancing: NONE", "- sampling: NONE", "- augmentation: NONE", "- synthetic data: NONE", ""])
    return "\n".join(lines)


def generate_baseline_reports(project_root: Path) -> dict[str, Any]:
    results = _load_results(project_root)
    oof_keys = []
    for result in results:
        oof = _load_oof(project_root, result)
        oof_keys.append(oof[["id", "fold"]].reset_index(drop=True))
    if any(not candidate.equals(oof_keys[0]) for candidate in oof_keys[1:]):
        raise RuntimeError("Baseline OOF files do not have exactly the same ordered IDs and folds")
    report_dir = project_root / "reports" / "brighter_ptbr" / "experiments"
    overall = _overall_table(results)
    per_label = _per_label_table(results)
    confusion = _confusion_table(results)
    cost = _cost_table(results)
    registry = _registry(results)
    fold_metrics = _fold_metrics_table(results)
    _write_csv(report_dir / "overall_baselines.csv", overall)
    _write_csv(report_dir / "per_label_baselines.csv", per_label)
    _write_csv(report_dir / "confusion_matrices.csv", confusion)
    write_json_atomic(report_dir / "confusion_matrices.json", {"rows": confusion.to_dict(orient="records")})
    _write_csv(report_dir / "computational_cost.csv", cost)
    _write_csv(report_dir / "experiment_registry.csv", registry)
    _write_csv(report_dir / "fold_metrics.csv", fold_metrics)
    _write_latex(report_dir, overall, per_label, cost)
    _generate_figures(project_root, results, overall, per_label)
    _write_text(report_dir / "baseline_results.md", _report(results, overall, per_label, cost))
    summary = {"revision": FROZEN_REVISION, "development_examples": DEV_POOL_SIZE, "folds": 5, "threshold": 0.5, "test_policy": "SEALED — NOT LOADED OR EVALUATED", "experiments": {result["experiment_id"]: {"macro_f1": result["cross_validation"]["macro_f1"], "micro_f1": result["cross_validation"]["micro_f1"], "oof_path": result["oof_path"], "oof_rows": DEV_POOL_SIZE} for result in results}}
    write_json_atomic(project_root / "metadata" / "brighter_ptbr" / "experiments" / "baseline_summary.json", summary)
    return summary
