"""OOF threshold diagnostics that are explicitly descriptive, not unbiased estimates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support

from .experiments import EXPECTED_LABELS

WARNING = "DIAGNOSTIC ONLY — THRESHOLD SELECTED AND EVALUATED ON SAME OOF"
MODELS = {
    "B1": "bertimbau_base_bce_oof.parquet",
    "B2": "bertimbau_large_bce_oof.parquet",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.8f", lineterminator="\n")
    os.replace(temporary, path)


def _save_figure(fig: Any, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path_without_suffix.with_suffix(suffix), dpi=220, bbox_inches="tight")


def generate_threshold_diagnostics(project_root: Path) -> dict[str, Any]:
    report_dir = project_root / "reports" / "brighter_ptbr" / "experiments"
    figure_dir = project_root / "figures" / "brighter_ptbr" / "experiments"
    thresholds = np.round(np.arange(0.01, 0.91, 0.01), 2)
    sweep_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for model, filename in MODELS.items():
        oof = pd.read_parquet(project_root / "artifacts" / "predictions" / filename)
        fig_curves, axes_curves = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        fig_distributions, axes_distributions = plt.subplots(3, 2, figsize=(12, 10))
        fig_pr, ax_pr = plt.subplots(figsize=(7.2, 5.4))
        for index, label in enumerate(EXPECTED_LABELS):
            y_true = oof[f"{label}_true"].to_numpy(dtype=int)
            probability = oof[f"{label}_prob"].to_numpy(dtype=float)
            label_rows = []
            for threshold in thresholds:
                predicted = (probability >= threshold).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, predicted, average="binary", zero_division=0
                )
                row = {
                    "warning": WARNING, "model": model, "emotion": label,
                    "threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
                    "predicted_positive": int(predicted.sum()),
                    "predicted_prevalence": float(predicted.mean()),
                    "gold_positive": int(y_true.sum()), "gold_prevalence": float(y_true.mean()),
                }
                label_rows.append(row)
                sweep_rows.append(row)
            best = max(label_rows, key=lambda row: (row["f1"], -row["threshold"]))
            summary_rows.append({
                **best, "average_precision": float(average_precision_score(y_true, probability)),
                "maximum_probability": float(probability.max()),
                "max_true_positive_probability": float(probability[y_true == 1].max()),
            })
            for truth_name, mask in (("true_negative", y_true == 0), ("true_positive", y_true == 1)):
                values = probability[mask]
                distribution_rows.append({
                    "model": model, "emotion": label, "group": truth_name, "n": len(values),
                    "mean": values.mean(), "std": values.std(ddof=1), "min": values.min(),
                    "q05": np.quantile(values, 0.05), "q25": np.quantile(values, 0.25),
                    "median": np.median(values), "q75": np.quantile(values, 0.75),
                    "q95": np.quantile(values, 0.95), "max": values.max(),
                })

            axis = axes_curves.flat[index]
            curve = pd.DataFrame(label_rows)
            axis.plot(curve["threshold"], curve["precision"], label="Precision")
            axis.plot(curve["threshold"], curve["recall"], label="Recall")
            axis.plot(curve["threshold"], curve["f1"], label="F1")
            axis.axvline(best["threshold"], color="black", linestyle="--", alpha=0.45)
            axis.set_title(f"{label} — best diagnostic F1={best['f1']:.3f} @ {best['threshold']:.2f}")
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.2)
            if index == 0:
                axis.legend(fontsize=8)

            axis = axes_distributions.flat[index]
            bins = np.linspace(0, max(0.1, probability.max()), 31)
            axis.hist(probability[y_true == 0], bins=bins, density=True, alpha=0.55, label="gold negative")
            axis.hist(probability[y_true == 1], bins=bins, density=True, alpha=0.55, label="gold positive")
            axis.axvline(0.5, color="black", linestyle=":", alpha=0.6)
            axis.set_title(label)
            axis.set_xlabel("Predicted probability")
            axis.set_ylabel("Density")
            if index == 0:
                axis.legend(fontsize=8)

            precision_curve, recall_curve, _ = precision_recall_curve(y_true, probability)
            ap = average_precision_score(y_true, probability)
            ax_pr.plot(recall_curve, precision_curve, label=f"{label} (AP={ap:.3f})")

        axes_curves[-1, 0].set_xlabel("Threshold")
        axes_curves[-1, 1].set_xlabel("Threshold")
        fig_curves.suptitle(f"{model} OOF threshold diagnostics\n{WARNING}", fontsize=12)
        fig_curves.tight_layout()
        _save_figure(fig_curves, figure_dir / f"{model.lower()}_threshold_diagnostic_curves")
        plt.close(fig_curves)
        fig_distributions.suptitle(f"{model} OOF probability distributions", fontsize=12)
        fig_distributions.tight_layout()
        _save_figure(fig_distributions, figure_dir / f"{model.lower()}_probability_distributions")
        plt.close(fig_distributions)
        ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title=f"{model} OOF precision–recall")
        ax_pr.grid(alpha=0.2)
        ax_pr.legend(fontsize=8)
        fig_pr.tight_layout()
        _save_figure(fig_pr, figure_dir / f"{model.lower()}_threshold_diagnostic_pr")
        plt.close(fig_pr)

    sweep = pd.DataFrame(sweep_rows)
    summary = pd.DataFrame(summary_rows)
    distributions = pd.DataFrame(distribution_rows)
    _atomic_csv(sweep, report_dir / "threshold_diagnostic_sweep.csv")
    _atomic_csv(summary, report_dir / "threshold_diagnostic_summary.csv")
    _atomic_csv(distributions, report_dir / "threshold_probability_distributions.csv")

    lines = [
        "# B1/B2 OOF threshold diagnostics", "", f"**{WARNING}**", "",
        "The sweep spans 0.01–0.90 in increments of 0.01. These values describe the existing OOF predictions; they are not leakage-safe estimates of a calibrated model.", "",
        "| Model | Emotion | AP | Best diagnostic threshold | Maximum diagnostic F1 | Precision | Recall | Predicted + | Gold + | Max score |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.emotion} | {row.average_precision:.4f} | {row.threshold:.2f} | {row.f1:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.predicted_positive} | {row.gold_positive} | {row.maximum_probability:.4f} |")
    lines.extend(["", "Thresholds selected here must not be carried into outer-fold evaluation. Confirmatory calibration requires an inner calibration split/OOF process inside each outer training fold.", ""])
    path = report_dir / "threshold_diagnostics.md"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return {"warning": WARNING, "rows": len(sweep), "summary": summary.to_dict(orient="records")}
