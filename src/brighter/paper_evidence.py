from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
STATUSES = {
    "DESCRIPTIVE",
    "LEAKAGE_SAFE_ESTIMATE",
    "DEPLOYMENT_CALIBRATION_ONLY",
    "FINAL_TEST",
    "POST_HOC_DIAGNOSTIC",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: Any) -> float | None:
    if value in (None, "", "NA", "n/a"):
        return None
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(rows: Iterable[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next(row for row in rows if row[key] == value)


def _minority(rows: list[dict[str, str]], system_key: str, system: str, label: str,
              label_key: str = "emotion", f1_key: str = "f1") -> float | None:
    for row in rows:
        if row.get(system_key) == system and row.get(label_key) == label and row.get("scope", "oof") == "oof":
            return _f(row.get(f1_key))
    return None


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#")]:
        text = text.replace(old, new)
    return text


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _latex_table(headers: list[str], rows: list[list[Any]], caption: str, label: str,
                 aligns: str | None = None) -> str:
    aligns = aligns or ("l" + "r" * (len(headers) - 1))
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(_latex_escape(x) for x in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(_latex_escape(_fmt(x)) for x in row) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(body)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_master_results(root: Path) -> list[dict[str, Any]]:
    exp = root / "reports/brighter_ptbr/experiments"
    meta = root / "metadata/brighter_ptbr/experiments"
    final_json = _json(root / "reports/brighter_ptbr/final/final_test_results.json")
    baseline = _json(meta / "baseline_summary.json")["experiments"]
    baseline_labels = _csv(exp / "per_label_baselines.csv")
    loss_summary = _json(meta / "imbalance_loss_summary.json")["experiments"]
    loss_labels = _csv(exp / "imbalance_loss_per_label.csv")
    encoder_summary = _json(meta / "encoder_summary.json")
    encoder = encoder_summary["experiments"]
    encoder_cost = {x["model"]: x for x in _csv(exp / "encoder_cost.csv")}
    threshold = {x["system"]: x for x in _json(meta / "threshold_calibration_summary.json")["main_results"]}
    ensemble_summary = _json(meta / "ensemble_summary.json")
    ensemble = {x["system"]: x for x in ensemble_summary["main_results"]}
    ensemble_labels = _csv(exp / "ensemble_per_label.csv")
    cardinality = {x["model"]: x for x in _csv(exp / "encoder_cardinality.csv")}
    baseline_neutral = {"B0": 0.0885, "B1": 0.2847, "B2": 0.2935}
    systems: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        base = dict(macro_f1=None, macro_f1_std=None, micro_f1=None, neutral_fp=None,
                    predicted_cardinality=None, disgust_f1=None, fear_f1=None,
                    surprise_f1=None, training_seconds=None, parameter_count=None)
        base.update(kwargs)
        systems.append(base)

    baseline_meta = {
        "B0": ("B0_TFIDF_LOGREG", "TF-IDF word+char", "Logistic Regression"),
        "B1": ("B1_BERTIMBAU_BASE_BCE", "BERTimbau Base", "BCE"),
        "B2": ("B2_BERTIMBAU_LARGE_BCE", "BERTimbau Large", "BCE"),
    }
    costs = {x["experiment_id"]: x for x in _csv(exp / "computational_cost.csv")}
    for short, (key, enc, loss) in baseline_meta.items():
        item = baseline[key]
        add(experiment_id=short, encoder=enc, loss=loss, decision_strategy="global threshold 0.50",
            evaluation_protocol="5-fold iterative multilabel CV; fold mean",
            macro_f1=item["macro_f1"]["mean"], macro_f1_std=item["macro_f1"]["std"],
            micro_f1=item["micro_f1"]["mean"], neutral_fp=baseline_neutral[short],
            predicted_cardinality=_f(cardinality.get(short, {}).get("predicted_label_cardinality")),
            disgust_f1=_minority(baseline_labels, "experiment_id", key, "disgust"),
            fear_f1=_minority(baseline_labels, "experiment_id", key, "fear"),
            surprise_f1=_minority(baseline_labels, "experiment_id", key, "surprise"),
            training_seconds=_f(costs[key]["training_seconds"]),
            parameter_count=_f(costs[key]["parameters"]),
            scientific_status="LEAKAGE_SAFE_ESTIMATE",
            source_artifact="metadata/brighter_ptbr/experiments/baseline_summary.json; reports/brighter_ptbr/experiments/per_label_baselines.csv")

    loss_names = {"L1": "Weighted BCE", "L2": "Focal", "L3": "Weighted Focal", "L4": "ASL"}
    for short, loss in loss_names.items():
        item = loss_summary[short]
        key = item["experiment_id"]
        result = _json(exp / f"bertimbau_base_{'weighted_bce' if short == 'L1' else 'focal' if short == 'L2' else 'weighted_focal' if short == 'L3' else 'asl'}_results.json")
        oof = result["oof_aggregate"]
        add(experiment_id=short, encoder="BERTimbau Base", loss=loss,
            decision_strategy="global threshold 0.50", evaluation_protocol="5-fold OOF aggregate",
            macro_f1=item["macro_f1_oof"], macro_f1_std=item["macro_f1_cv"]["std"],
            micro_f1=oof["micro_f1"], neutral_fp=oof["zero_label"]["neutral_false_positive_rate"],
            predicted_cardinality=sum(float(r["predicted_prevalence"]) for r in loss_labels if r["model"] == short and r["scope"] == "oof"),
            disgust_f1=_minority(loss_labels, "model", short, "disgust"),
            fear_f1=_minority(loss_labels, "model", short, "fear"),
            surprise_f1=_minority(loss_labels, "model", short, "surprise"),
            training_seconds=item["cost"]["training_seconds_total"],
            parameter_count=item["cost"]["parameters"],
            scientific_status="LEAKAGE_SAFE_ESTIMATE",
            source_artifact="metadata/brighter_ptbr/experiments/imbalance_loss_summary.json; reports/brighter_ptbr/experiments/imbalance_loss_per_label.csv")

    encoder_names = {
        "E1": ("BERTweetBR", "BCE"), "E2": ("BERTweetBR", "Weighted Focal"),
        "E3": ("BERTweetBR", "ASL"), "E4": ("Albertina 100M PT-BR", "BCE"),
        "E5": ("Albertina 100M PT-BR", "Weighted Focal"), "E6": ("Albertina 100M PT-BR", "ASL"),
    }
    for short, (enc, loss) in encoder_names.items():
        item = encoder[short]
        m = item["metrics"]
        add(experiment_id=short, encoder=enc, loss=loss, decision_strategy="global threshold 0.50",
            evaluation_protocol="5-fold OOF aggregate", macro_f1=m["macro_f1"], micro_f1=m["micro_f1"],
            neutral_fp=m["zero_label"]["neutral_false_positive_rate"],
            predicted_cardinality=m["predicted_label_cardinality"],
            disgust_f1=m["per_label"]["disgust"]["f1"], fear_f1=m["per_label"]["fear"]["f1"],
            surprise_f1=m["per_label"]["surprise"]["f1"],
            training_seconds=_f(encoder_cost[short]["training_seconds_total"]),
            parameter_count=_f(encoder_cost[short]["parameters"]),
            scientific_status="LEAKAGE_SAFE_ESTIMATE",
            source_artifact="metadata/brighter_ptbr/experiments/encoder_summary.json")

    threshold_meta = {
        "T1": ("BERTweetBR", "Weighted Focal"), "T2": ("BERTweetBR", "ASL"),
        "T3": ("BERTimbau Large", "BCE"),
    }
    for short, (enc, loss) in threshold_meta.items():
        m = threshold[short]
        add(experiment_id=short, encoder=enc, loss=loss,
            decision_strategy="nested per-label thresholds",
            evaluation_protocol="nested 5x3 cross-fitted OOF aggregate",
            macro_f1=m["thresholded_macro_f1"], micro_f1=m["thresholded_micro_f1"],
            neutral_fp=m["neutral_fp_thresholded"], predicted_cardinality=m["predicted_cardinality_thresholded"],
            disgust_f1=m["disgust_f1"], fear_f1=m["fear_f1"], surprise_f1=m["surprise_f1"],
            scientific_status="LEAKAGE_SAFE_ESTIMATE",
            source_artifact="metadata/brighter_ptbr/experiments/threshold_calibration_summary.json")

    for short, strategy in [("EN1", "equal-weight soft ensemble + nested thresholds"),
                            ("EN2", "tuned-weight soft ensemble + nested thresholds")]:
        m = ensemble[short]
        add(experiment_id=short, encoder="BERTweetBR + BERTimbau Large", loss="ASL + BCE",
            decision_strategy=strategy, evaluation_protocol="nested 5x3 cross-fitted ensemble OOF",
            macro_f1=m["macro_f1"], micro_f1=m["micro_f1"], neutral_fp=m["neutral_fp"],
            predicted_cardinality=m["predicted_cardinality"],
            disgust_f1=_minority(ensemble_labels, "system", short, "disgust", "label"),
            fear_f1=_minority(ensemble_labels, "system", short, "fear", "label"),
            surprise_f1=_minority(ensemble_labels, "system", short, "surprise", "label"),
            parameter_count=ensemble_summary["cost"]["ensemble"]["parameters_across_two_networks"],
            scientific_status="LEAKAGE_SAFE_ESTIMATE",
            source_artifact="metadata/brighter_ptbr/experiments/ensemble_summary.json")

    for short, block, enc, loss, strategy in [
        ("FINAL_EN2", final_json["primary"], "BERTweetBR + BERTimbau Large", "ASL + BCE", "frozen alpha 0.70/0.30 + frozen thresholds"),
        ("FINAL_T2", final_json["secondary"], "BERTweetBR", "ASL", "frozen standalone thresholds"),
    ]:
        m = block["metrics"]
        add(experiment_id=short, encoder=enc, loss=loss, decision_strategy=strategy,
            evaluation_protocol="one-time official Track A test",
            macro_f1=m["macro_f1"], micro_f1=m["micro_f1"],
            neutral_fp=m["zero_label"]["neutral_false_positive_rate"],
            predicted_cardinality=m["predicted_label_cardinality"],
            disgust_f1=m["per_label"]["disgust"]["f1"], fear_f1=m["per_label"]["fear"]["f1"],
            surprise_f1=m["per_label"]["surprise"]["f1"],
            parameter_count=469310220 if short == "FINAL_EN2" else 134907654,
            scientific_status="FINAL_TEST",
            source_artifact="reports/brighter_ptbr/final/final_test_results.json")
    return systems


def build_claims(root: Path, master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eda = _json(root / "metadata/brighter_ptbr/eda/eda_summary.json")
    final = _json(root / "reports/brighter_ptbr/final/final_test_results.json")
    gen = _json(root / "metadata/brighter_ptbr/final/generalization_gap_summary.json")
    ensemble = _json(root / "metadata/brighter_ptbr/experiments/ensemble_summary.json")
    claims: list[dict[str, Any]] = []

    def claim(cid: str, description: str, value: Any, unit: str, source: str, field: str,
              protocol: str, status: str) -> None:
        assert status in STATUSES
        claims.append({"claim_id": cid, "description": description, "value": value, "unit": unit,
                       "source_file": source, "source_field": field,
                       "experimental_protocol": protocol, "scientific_status": status})

    for split in ["train", "dev", "test"]:
        claim(f"C_DATA_{split.upper()}_N", f"Official Track A {split} examples", eda["splits"][split]["rows"],
              "examples", "metadata/brighter_ptbr/eda/eda_summary.json", f"splits.{split}.rows", "corpus audit", "DESCRIPTIVE")
    train = eda["splits"]["train"]
    for key, desc, unit in [
        ("zero_label_examples", "Train zero-label examples", "examples"),
        ("zero_label_percentage", "Train zero-label rate", "percent"),
        ("single_label_examples", "Train single-label examples", "examples"),
        ("multi_label_examples", "Train multilabel examples", "examples"),
        ("multi_label_percentage", "Train multilabel rate", "percent"),
        ("label_cardinality", "Train label cardinality", "labels/example"),
        ("label_density", "Train label density", "ratio"),
    ]:
        claim(f"C_EDA_{key.upper()}", desc, train[key], unit, "metadata/brighter_ptbr/eda/eda_summary.json",
              f"splits.train.{key}", "descriptive train EDA", "DESCRIPTIVE")
    claim("C_EDA_MEANIR", "Training MeanIR", eda["mean_ir"], "ratio", "metadata/brighter_ptbr/eda/eda_summary.json", "mean_ir", "descriptive train EDA", "DESCRIPTIVE")
    claim("C_EDA_SCUMBLE", "Training SCUMBLE", eda["scumble"], "score", "metadata/brighter_ptbr/eda/eda_summary.json", "scumble", "descriptive train EDA", "DESCRIPTIVE")
    claim("C_EDA_LABELSETS_OBSERVED", "Observed train labelsets", eda["labelsets"]["train"]["observed"], "labelsets", "metadata/brighter_ptbr/eda/eda_summary.json", "labelsets.train.observed", "descriptive train EDA", "DESCRIPTIVE")
    claim("C_EDA_LABELSETS_POSSIBLE", "Possible labelsets for six labels", eda["labelsets"]["train"]["possible"], "labelsets", "metadata/brighter_ptbr/eda/eda_summary.json", "labelsets.train.possible", "descriptive train EDA", "DESCRIPTIVE")
    disgust_concurrence = next(x for x in eda["imbalance"]["minority_concurrence"] if x["minority_label"] == "disgust")
    for key, description in [
        ("positive_examples", "Train disgust positives"),
        ("with_majority_label", "Train disgust positives co-occurring with a majority label"),
        ("most_frequent_co_label_count", "Train disgust positives co-occurring with anger"),
    ]:
        claim(f"C_EDA_DISGUST_{key.upper()}", description, disgust_concurrence[key], "examples",
              "metadata/brighter_ptbr/eda/eda_summary.json", f"imbalance.minority_concurrence[disgust].{key}",
              "descriptive train EDA", "DESCRIPTIVE")

    def metric_source(row: dict[str, Any], metric: str) -> tuple[str, str]:
        eid = row["experiment_id"]
        if eid in {"B0", "B1", "B2"}:
            key = {"B0": "B0_TFIDF_LOGREG", "B1": "B1_BERTIMBAU_BASE_BCE", "B2": "B2_BERTIMBAU_LARGE_BCE"}[eid]
            if metric in {"macro_f1", "macro_f1_std", "micro_f1"}:
                field = "macro_f1.std" if metric == "macro_f1_std" else f"{metric}.mean"
                return "metadata/brighter_ptbr/experiments/baseline_summary.json", f"experiments.{key}.{field}"
            if metric in {"disgust_f1", "fear_f1", "surprise_f1"}:
                return "reports/brighter_ptbr/experiments/per_label_baselines.csv", f"experiment_id={key}; emotion={metric.removesuffix('_f1')}; f1"
            if metric == "predicted_cardinality":
                return "reports/brighter_ptbr/experiments/encoder_cardinality.csv", f"model={eid}; predicted_label_cardinality"
            if metric in {"training_seconds", "parameter_count"}:
                return "reports/brighter_ptbr/experiments/computational_cost.csv", f"experiment_id={key}; {'training_seconds' if metric == 'training_seconds' else 'parameters'}"
            return "reports/brighter_ptbr/experiments/baseline_results.md", f"{eid}.{metric}"
        if eid.startswith("L"):
            if metric in {"macro_f1", "macro_f1_std", "training_seconds", "parameter_count"}:
                return "metadata/brighter_ptbr/experiments/imbalance_loss_summary.json", f"experiments.{eid}.{metric}"
            return "reports/brighter_ptbr/experiments/imbalance_loss_per_label.csv", f"model={eid}; {metric}"
        if eid.startswith("E") and eid[1:].isdigit():
            return "metadata/brighter_ptbr/experiments/encoder_summary.json", f"experiments.{eid}.metrics.{metric}"
        if eid in {"T1", "T2", "T3"}:
            return "metadata/brighter_ptbr/experiments/threshold_calibration_summary.json", f"main_results[system={eid}].{metric}"
        if eid in {"EN1", "EN2"}:
            return "metadata/brighter_ptbr/experiments/ensemble_summary.json", f"main_results[system={eid}].{metric}"
        block = "primary" if eid == "FINAL_EN2" else "secondary"
        return "reports/brighter_ptbr/final/final_test_results.json", f"{block}.metrics.{metric}"

    for row in master:
        for metric in ["macro_f1", "macro_f1_std", "micro_f1", "neutral_fp", "predicted_cardinality",
                       "disgust_f1", "fear_f1", "surprise_f1", "training_seconds", "parameter_count"]:
            if row.get(metric) is not None:
                source_file, source_field = metric_source(row, metric)
                claim(f"C_{row['experiment_id']}_{metric.upper()}", f"{row['experiment_id']} {metric}", row[metric], "score",
                      source_file, source_field, row["evaluation_protocol"], row["scientific_status"])
    for key, value in final["en2_thresholds"].items():
        claim(f"C_DEPLOY_EN2_T_{key.upper()}", f"Frozen EN2 threshold for {key}", value, "probability",
              "reports/brighter_ptbr/final/final_test_results.json", f"en2_thresholds.{key}", "full-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    claim("C_DEPLOY_ALPHA_T2", "Frozen EN2 T2 weight", final["alpha_t2"], "weight", "reports/brighter_ptbr/final/final_test_results.json", "alpha_t2", "full-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    claim("C_DEPLOY_WEIGHT_T3", "Frozen EN2 T3 weight", final["weight_t3"], "weight", "reports/brighter_ptbr/final/final_test_results.json", "weight_t3", "full-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    for key, value in final["t2_thresholds"].items():
        claim(f"C_DEPLOY_T2_T_{key.upper()}", f"Frozen T2 standalone threshold for {key}", value, "probability",
              "reports/brighter_ptbr/final/final_test_results.json", f"t2_thresholds.{key}",
              "full-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    claim("C_DEPLOY_EN2_SCORE", "EN2 full-OOF deployment calibration Macro-F1", final["development"]["EN2_deployment_calibration_macro_f1"], "Macro-F1", "reports/brighter_ptbr/final/final_test_results.json", "development.EN2_deployment_calibration_macro_f1", "same-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    claim("C_DEPLOY_T2_SCORE", "T2 full-OOF deployment calibration Macro-F1", final["development"]["T2_deployment_calibration_macro_f1"], "Macro-F1", "reports/brighter_ptbr/final/final_test_results.json", "development.T2_deployment_calibration_macro_f1", "same-OOF deployment calibration", "DEPLOYMENT_CALIBRATION_ONLY")
    for item in gen["main_results"]:
        claim(f"C_GAP_{item['system']}", f"{item['system']} development-to-test Macro-F1 difference", item["difference"], "Macro-F1", "metadata/brighter_ptbr/final/generalization_gap_summary.json", "main_results", "frozen development vs final test", "POST_HOC_DIAGNOSTIC")
    for item in gen["predicted_cardinality"]:
        claim(f"C_CARD_RATIO_{item['system']}", f"{item['system']} predicted-cardinality test/development ratio", item["test_dev_ratio"], "ratio", "metadata/brighter_ptbr/final/generalization_gap_summary.json", "predicted_cardinality", "post-hoc frozen artifact comparison", "POST_HOC_DIAGNOSTIC")
    for item in gen["macro_calibration"]:
        claim(f"C_AP_{item['system']}_{item['split'].upper()}", f"{item['system']} Macro-AP on {item['split']}", item["average_precision"], "Average Precision", "metadata/brighter_ptbr/final/generalization_gap_summary.json", "macro_calibration", "post-hoc threshold-free diagnostic", "POST_HOC_DIAGNOSTIC")
    comparison = next(x for x in ensemble["bootstrap"] if x["comparison"] == "EN2_vs_T2")
    for key, description in [("delta", "EN2 minus T2 nested Macro-F1"), ("ci_2_5", "EN2 minus T2 bootstrap CI lower bound"), ("ci_97_5", "EN2 minus T2 bootstrap CI upper bound")]:
        claim(f"C_EN2_BOOT_{key.upper()}", description, comparison[key], "Macro-F1",
              "metadata/brighter_ptbr/experiments/ensemble_summary.json", f"bootstrap.EN2_vs_T2.{key}",
              "nested cross-fitted paired bootstrap", "LEAKAGE_SAFE_ESTIMATE")
    alpha_values = [float(x["alpha_T2"]) for x in ensemble["alpha_by_fold"]]
    alpha_summary = {
        "mean": sum(alpha_values) / len(alpha_values),
        "std": (sum((x - sum(alpha_values) / len(alpha_values)) ** 2 for x in alpha_values) / (len(alpha_values) - 1)) ** 0.5,
        "min": min(alpha_values), "max": max(alpha_values),
    }
    for key in ["mean", "std", "min", "max"]:
        claim(f"C_EN2_ALPHA_{key.upper()}", f"Nested EN2 T2 alpha {key}", alpha_summary[key], "weight",
              "metadata/brighter_ptbr/experiments/ensemble_summary.json", f"alpha_by_fold.{key}",
              "nested cross-fitted ensemble", "LEAKAGE_SAFE_ESTIMATE")

    for system, block in [("EN2", final["primary"]), ("T2", final["secondary"])]:
        metrics = block["metrics"]
        for key in ["macro_precision", "macro_recall", "macro_average_precision", "hamming_loss", "jaccard_samples", "subset_accuracy", "gold_label_cardinality"]:
            claim(f"C_FINAL_{system}_{key.upper()}", f"Final test {system} {key}", metrics[key], "score",
                  "reports/brighter_ptbr/final/final_test_results.json",
                  f"{'primary' if system == 'EN2' else 'secondary'}.metrics.{key}",
                  "one-time official Track A test", "FINAL_TEST")
        for label in LABELS:
            for key in ["precision", "recall", "f1", "average_precision", "support", "predicted_prevalence"]:
                claim(f"C_FINAL_{system}_PERLABEL_{label.upper()}_{key.upper()}", f"Final test {system} {label} {key}",
                      metrics["per_label"][label][key], "score", "reports/brighter_ptbr/final/final_test_results.json",
                      f"{'primary' if system == 'EN2' else 'secondary'}.metrics.per_label.{label}.{key}",
                      "one-time official Track A test", "FINAL_TEST")

    prevalence_rows = _csv(root / "reports/brighter_ptbr/final/generalization_prevalence.csv")
    for row in prevalence_rows:
        for field in ["gold_dev", "pred_dev_en2_nested", "gold_test", "pred_test_en2", "prediction_error_dev", "prediction_error_test"]:
            claim(f"C_POSTHOC_PREV_EN2_{row['label'].upper()}_{field.upper()}",
                  f"Post-hoc EN2 {row['label']} {field}", _f(row[field]), "prevalence",
                  "reports/brighter_ptbr/final/generalization_prevalence.csv",
                  f"label={row['label']}; {field}",
                  "post-hoc frozen artifact comparison", "POST_HOC_DIAGNOSTIC")
    crossing_rows = _csv(root / "reports/brighter_ptbr/final/generalization_threshold_crossing.csv")
    for row in crossing_rows:
        if row["system"] != "EN2":
            continue
        for field in ["threshold", "development_crossing_rate", "test_crossing_rate", "delta"]:
            claim(f"C_POSTHOC_CROSS_EN2_{row['label'].upper()}_{field.upper()}",
                  f"Post-hoc EN2 {row['label']} threshold crossing {field}", _f(row[field]), "rate",
                  "reports/brighter_ptbr/final/generalization_threshold_crossing.csv",
                  f"system=EN2; label={row['label']}; {field}",
                  "post-hoc frozen probability comparison", "POST_HOC_DIAGNOSTIC")
    return claims


def _paper_texts(root: Path, master: list[dict[str, Any]], claims: list[dict[str, Any]]) -> dict[str, str]:
    figures = [
        ("Figure 1", "figures/brighter_ptbr/paper/paper_experimental_pipeline", "paper synthesis generated only from frozen protocol metadata"),
        ("Figure 2", "figures/brighter_ptbr/eda/01_label_distribution_train", "training label imbalance"),
        ("Figure 3", "figures/brighter_ptbr/paper/macro_f1_evolution", "main experimental-stage evolution"),
        ("Figure 4", "figures/brighter_ptbr/experiments/encoder_f1_by_label", "per-label encoder/loss behavior"),
        ("Figure 5", "figures/brighter_ptbr/experiments/fixed_vs_thresholded_macro_f1", "nested threshold effect"),
        ("Figure 6", "figures/brighter_ptbr/final/generalization_gap/01_macro_f1_dev_vs_test", "development-to-test gap"),
        ("Figure 7", "figures/brighter_ptbr/final/generalization_gap/06_threshold_crossing_dev_vs_test", "decision-transfer shift"),
    ]
    fig_lines = "\n".join(f"- **{n}:** `{p}.png` / `{p}.pdf` — {why}." for n, p, why in figures)
    rqs = """1. **RQ1:** How do imbalance-aware loss functions affect minority emotion recognition in Brazilian Portuguese multi-label classification?
2. **RQ2:** How does encoder choice interact with imbalance-aware objectives?
3. **RQ3:** How much does leakage-safe class-specific threshold calibration improve performance over a global 0.5 threshold?
4. **RQ4:** Can complementary encoders improve performance through probability-level ensemble?
5. **RQ5:** How robust are cross-validated decision thresholds when transferred to final retrained models?"""
    contributions = """- **C1.** A data-centric audit of the current BRIGHTER PT-BR release, including the documented Track A/Track C duplication issue.
- **C2.** A controlled comparison of BCE, Weighted BCE, Focal, Weighted Focal and ASL.
- **C3.** A controlled comparison of BERTimbau, BERTweetBR and Albertina under a fixed protocol.
- **C4.** Leakage-safe nested evaluation of class-specific thresholds.
- **C5.** A leakage-safe soft ensemble of BERTweetBR+ASL and BERTimbau Large+BCE.
- **C6.** An auditable analysis of failure when OOF-derived decision rules are transferred to final retrained models.

These are empirical contributions, not state-of-the-art claims."""
    threats = """| Validity dimension | Threat | Mitigation / interpretation |
|---|---|---|
| Internal | Threshold/model-selection complexity | Frozen folds, nested 5×3 selection, explicit scientific-status labels |
| Internal | One final seed | Reported as a limitation; no post-test seed search |
| Internal | One known train–test textual overlap | Retained under the official split and documented before final evaluation |
| Internal | Final retraining changed probability distributions | Quantified post hoc without retuning |
| Construct | Macro-F1 is the official primary metric | Secondary micro, per-label, neutral and cardinality metrics are reported |
| Construct | Hard aggregated emotion labels | Interpretations are limited to Track A label semantics |
| Construct | Neutral is an all-zero vector | Exact neutral accuracy and neutral FP are reported explicitly |
| External | PT-BR social-media/personal-narrative domain only | No cross-language or cross-domain claim |
| External | Small dataset and large final gap | Final test is disclosed and robustness is a central result |
| Conclusion | Rare-class support, especially 75 train disgust positives | Per-label metrics, AP and threshold stability are reported |
| Conclusion | Rare-label threshold uncertainty | Nested calibration mitigates but cannot eliminate variance |"""
    related = """| Work/system | Role in manuscript | Current project evidence | Follow-up |
|---|---|---|---|
| BRIGHTER dataset / SemEval-2025 Task 11 | Dataset and task definition | Dataset README citation entry and frozen revision | Verify final BibTeX metadata |
| BERTimbau | Portuguese BERT baseline | Model checkpoint and project reports | Verify canonical paper/citation |
| Focal Loss | Imbalance-aware objective | Implemented L2/L3 and method report | Verify canonical Lin et al. citation |
| Asymmetric Loss | Imbalance-aware multilabel objective | Official implementation link and L4/E3 reports | Verify final paper metadata |
| RSSN | SemEval Task 11 related system | Named in prior project notes/request | Locate and verify proceedings entry |
| NITK-VITAL | SemEval Task 11 related system | Named in prior project notes/request | Locate and verify proceedings entry |
| University of Indonesia | SemEval Task 11 related system | Named in prior project notes/request | Resolve exact system-paper title |
| AKCIT | SemEval Task 11 related system | Named in prior project notes/request | Locate and verify proceedings entry |
| LATE-GIL-NLP | SemEval Task 11 related system | Named in prior project notes/request | Locate and verify proceedings entry |
| QMUL | SemEval Task 11 related system | Named in prior project notes/request | Locate and verify proceedings entry |

No new literature claims are introduced here; unresolved entries are explicitly marked for later bibliographic verification."""
    narrative = f"""# Paper narrative

## 1. Problem

Multi-label emotion classification in Brazilian Portuguese combines severe class imbalance, label concurrence, and a decision problem in which one global threshold can conceal useful minority-class ranking.

## 2. Dataset challenges

The Track A training split contains 2,226 texts. Of these, 632 (28.39%) are zero-label, 333 (14.96%) are multilabel, and only 75 contain `disgust`; 65 of those 75 co-occur with a majority label and 60 with `anger`. MeanIR is 4.2198 and SCUMBLE is 0.037882.

## 3. Experimental hypotheses

{rqs}

## 4. Controlled experimentation

The study fixes the original PT-BR Track A data, five iterative multilabel folds, max length 128, optimizer schedule, effective batch size 16, seed policy, and no preprocessing/augmentation/sampling. It varies baselines, losses, encoders, leakage-safe thresholds, and finally a probability ensemble.

## 5. Main development findings

BERTimbau Large+BCE improved the baseline fold mean to 0.3958. ASL was the strongest BERTimbau Base loss (OOF Macro-F1 0.4617) but overpredicted. BERTweetBR+BCE did not materially exceed BERTimbau Base+BCE; the largest fixed-threshold gain emerged from the BERTweetBR×ASL interaction (E3=0.5258).

## 6. Threshold and ensemble gains

Nested thresholds raised T2 from 0.5258 to 0.5585 and T3 from concatenated OOF 0.3976 to 0.5023. EN2 reached the leakage-safe estimate 0.5691, +0.0106 over T2 with percentile CI [+0.0011,+0.0202], while improving neutral FP and predicted cardinality.

## 7. Final generalization failure

The pre-registered EN2 final test Macro-F1 was 0.2852, not 0.5691. T2 fell from 0.5585 to 0.1569. These results are retained without post-test adjustment.

## 8. Post-hoc evidence

Gold cardinality remained essentially unchanged (0.882111→0.882300), but EN2 predicted cardinality fell 1.0639→0.3459. Threshold crossings collapsed for most labels and EN2 Macro-AP fell 0.6002→0.5042. The evidence therefore supports both decision-transfer/score-scale shift and ranking degradation, but does not identify one causal mechanism.

## 9. Scientific interpretation

The main methodological lesson is that leakage-safe selection within cross-validation is necessary but may be insufficient when thresholds calibrated on fold-model OOF score distributions are transferred to single models retrained on the full development pool. T3 was comparatively more stable and mitigated T2's degradation, so the ensemble itself is not supported as the primary failure source.

## 10. Contributions

{contributions}

## 11. Limitations

{threats}
"""
    outline = """# Paper outline

1. **Introduction** — task, imbalance, decision transfer, contributions.
2. **Related Work** — multilingual emotion detection; Portuguese encoders; imbalance-aware multilabel learning; threshold calibration.
3. **Dataset and Task** — BRIGHTER Track A PT-BR, label semantics, EDA, integrity and overlap note.
4. **Methodology**
   1. Data Audit
   2. Baselines
   3. Imbalance-aware Losses
   4. Encoders
   5. Threshold Calibration
   6. Ensemble
5. **Experimental Protocol** — frozen folds, hyperparameters, leakage controls, metrics, bootstrap, test preregistration.
6. **Results**
   1. Baselines
   2. Loss Functions
   3. Encoder Comparison
   4. Threshold Calibration
   5. Ensemble
   6. Final Test
7. **Generalization Analysis** — prevalence, cardinality, score distributions, threshold crossings, ranking and calibration diagnostics.
8. **Threats to Validity** — internal, construct, external, conclusion.
9. **Conclusion** — controlled findings and robustness lesson without post-test tuning.
"""
    facts = """# Results facts

Every statement below must be copied with its source/status intact.

## Baselines

- **Statement:** Larger BERTimbau capacity improved the fixed-threshold baseline. **Evidence:** B1 fold mean 0.3238±0.0204; B2 fold mean 0.3958±0.0212. **Table/Figure:** `paper_baselines_losses.tex`, baseline Macro-F1 figure. **Source:** `baseline_summary.json`. **Status:** LEAKAGE_SAFE_ESTIMATE.

## Loss functions

- **Statement:** ASL was the strongest BERTimbau Base loss by OOF Macro-F1 (0.4617) but had Neutral FP 0.7729. **Evidence:** L4 report/summary. **Table/Figure:** `paper_baselines_losses.tex`, loss Macro-F1 figure. **Status:** LEAKAGE_SAFE_ESTIMATE.
- **Statement:** Weighted Focal retained high disgust precision (0.4762) and AP (0.1987) at fixed 0.5. **Evidence:** L3 per-label OOF. **Status:** LEAKAGE_SAFE_ESTIMATE.

## Encoders

- **Statement:** BERTweetBR alone did not explain the gain: E1=0.3263 was near B1, while E3=0.5258. **Evidence:** `encoder_summary.json`. **Table/Figure:** `paper_encoder_results.tex`, encoder figures. **Status:** LEAKAGE_SAFE_ESTIMATE.
- **Statement:** The evidence indicates encoder×loss interaction; Albertina did not reproduce BERTweetBR's ASL/Weighted-Focal behavior. **Evidence:** E1–E6 controlled matrix. **Status:** LEAKAGE_SAFE_ESTIMATE.

## Threshold calibration

- **Statement:** Nested class thresholds improved T1 0.3974→0.5152, T2 0.5258→0.5585 and T3 0.3976→0.5023. **Evidence:** `threshold_calibration_summary.json`. **Table/Figure:** `paper_threshold_ensemble.tex`, fixed-vs-thresholded figure. **Status:** LEAKAGE_SAFE_ESTIMATE.
- **Statement:** B2's 0.3958 is a five-fold mean; 0.3976 is the concatenated OOF paired reference. **Evidence:** threshold report. **Status:** LEAKAGE_SAFE_ESTIMATE.

## Ensemble

- **Statement:** EN2 reached 0.5691, +0.0106 vs T2, percentile CI [+0.0011,+0.0202]. **Evidence:** `ensemble_summary.json`. **Table/Figure:** `paper_threshold_ensemble.tex`. **Status:** LEAKAGE_SAFE_ESTIMATE.
- **Statement:** The nested T2 alpha was 0.870±0.076 (0.75–0.95). **Evidence:** ensemble alpha by fold. **Status:** LEAKAGE_SAFE_ESTIMATE.

## Deployment calibration

- **Statement:** Deployment fixed alpha 0.70 and EN2 thresholds [.67,.36,.43,.65,.58,.49]. **Evidence:** final test JSON/config. **Status:** DEPLOYMENT_CALIBRATION_ONLY.
- **Statement:** EN2 deployment calibration Macro-F1 0.5897 is not an unbiased estimate. **Evidence:** freeze/final reports. **Status:** DEPLOYMENT_CALIBRATION_ONLY.

## Final test

- **Statement:** Pre-registered primary EN2 obtained Macro-F1 0.2852 and Micro-F1 0.4221; T2 obtained 0.1569 and 0.2405. **Evidence:** `final_test_results.json`. **Table/Figure:** `paper_final_results.tex`. **Status:** FINAL_TEST.
- **Statement:** EN2 final predicted cardinality was 0.3459 vs gold 0.8823, with Neutral FP 0.0942. **Evidence:** final test JSON. **Status:** FINAL_TEST.

## Generalization

- **Statement:** EN2 changed 0.5691→0.2852 (-0.2839); T2 changed 0.5585→0.1569 (-0.4016). **Evidence:** generalization summary. **Status:** POST_HOC_DIAGNOSTIC.
- **Statement:** Gold cardinality was stable (0.882111→0.882300), whereas EN2 predicted cardinality fell 1.0639→0.3459. **Evidence:** generalization summary. **Status:** POST_HOC_DIAGNOSTIC.
- **Statement:** EN2 Macro-AP fell 0.6002→0.5042. **Evidence:** generalization summary. **Status:** POST_HOC_DIAGNOSTIC.
"""
    readme = f"""# BRIGHTER PT-BR paper evidence package

This directory is the writing-time source of truth. It consolidates frozen artifacts; it does not replace them.

## Core files

- `master_results.csv`: one row per relevant experimental system, with protocol and scientific status.
- `metadata/brighter_ptbr/paper/paper_evidence_manifest.json`: provenance for quantitative claims.
- `results_facts.md`: claim-level writing support.
- `paper_narrative.md`: cautious scientific narrative.
- `paper_outline.md`: manuscript structure.
- `threats_to_validity.md`: structured validity assessment.
- `related_work_inventory.md`: already-used work plus verification backlog.
- `figure_selection.md`: the seven selected paper figures.
- `latex/`: compact paper-ready tables.

## Scientific-status rules

- **DESCRIPTIVE:** corpus/property statements, not predictive estimates.
- **LEAKAGE_SAFE_ESTIMATE:** cross-validated or nested cross-fitted development evidence.
- **DEPLOYMENT_CALIBRATION_ONLY:** parameters/scores chosen on the same full OOF pool; never present as unbiased performance.
- **FINAL_TEST:** one-time pre-registered test outcomes.
- **POST_HOC_DIAGNOSTIC:** read-only analyses performed after test; never selection evidence.

The manuscript must use EN2=0.5691 as the leakage-safe development estimate and EN2=0.2852 as the final test result. It must not substitute deployment calibration 0.5897 for either.

## Selected figures

{fig_lines}

Run `python scripts/validate_paper_evidence.py` before manuscript edits or submission. The experiment remains `TEST_EVALUATED`; the search space is closed.
"""
    return {
        "paper_narrative.md": narrative,
        "paper_outline.md": outline,
        "results_facts.md": facts,
        "threats_to_validity.md": "# Threats to validity\n\n" + threats + "\n",
        "related_work_inventory.md": "# Related work inventory\n\n" + related + "\n",
        "figure_selection.md": "# Selected paper figures\n\n" + fig_lines + "\n\nOnly the pipeline and Macro-F1 evolution figures are new syntheses; all quantitative inputs are frozen.\n",
        "research_questions_and_contributions.md": "# Research questions and contributions\n\n## Research questions\n\n" + rqs + "\n\n## Potential contributions\n\n" + contributions + "\n",
        "README.md": readme,
    }


def _write_tables(root: Path, master: list[dict[str, Any]]) -> None:
    paper = root / "reports/brighter_ptbr/paper"
    latex = paper / "latex"
    latex.mkdir(parents=True, exist_ok=True)
    eda = _json(root / "metadata/brighter_ptbr/eda/eda_summary.json")
    by_id = {x["experiment_id"]: x for x in master}
    dataset_rows = [["Train", 2226, 632, 1261, 333, 0.8796, 0.1466], ["Dev", 200, 46, 129, 25, 0.9100, 0.1517], ["Test", 2226, 616, 1280, 330, 0.8823, 0.1471]]
    tables = {
        "paper_dataset_statistics.tex": _latex_table(["Split", "N", "Zero", "Single", "Multi", "Card.", "Density"], dataset_rows, "BRIGHTER PT-BR Track A statistics.", "tab:dataset"),
        "paper_baselines_losses.tex": _latex_table(["ID", "Encoder", "Loss", "Macro-F1", "Neutral FP", "Disgust F1"], [[i, by_id[i]["encoder"], by_id[i]["loss"], by_id[i]["macro_f1"], by_id[i]["neutral_fp"], by_id[i]["disgust_f1"]] for i in ["B0","B1","B2","L1","L2","L3","L4"]], "Baselines and imbalance-aware losses.", "tab:baselines-losses"),
        "paper_encoder_results.tex": _latex_table(["ID", "Encoder", "Loss", "Macro-F1", "Disgust", "Fear", "Surprise"], [[i, by_id[i]["encoder"], by_id[i]["loss"], by_id[i]["macro_f1"], by_id[i]["disgust_f1"], by_id[i]["fear_f1"], by_id[i]["surprise_f1"]] for i in ["E1","E2","E3","E4","E5","E6"]], "Controlled encoder-by-loss comparison at threshold 0.50.", "tab:encoders"),
        "paper_threshold_ensemble.tex": _latex_table(["ID", "Decision", "Macro-F1", "Micro-F1", "Neutral FP", "Cardinality"], [[i, by_id[i]["decision_strategy"], by_id[i]["macro_f1"], by_id[i]["micro_f1"], by_id[i]["neutral_fp"], by_id[i]["predicted_cardinality"]] for i in ["T1","T2","T3","EN1","EN2"]], "Leakage-safe threshold calibration and ensemble results.", "tab:threshold-ensemble"),
        "paper_final_results.tex": _latex_table(["System", "Role", "Macro-F1", "Micro-F1", "Neutral FP", "Pred. card."], [["EN2", "Primary", by_id["FINAL_EN2"]["macro_f1"], by_id["FINAL_EN2"]["micro_f1"], by_id["FINAL_EN2"]["neutral_fp"], by_id["FINAL_EN2"]["predicted_cardinality"]], ["T2", "Secondary", by_id["FINAL_T2"]["macro_f1"], by_id["FINAL_T2"]["micro_f1"], by_id["FINAL_T2"]["neutral_fp"], by_id["FINAL_T2"]["predicted_cardinality"]]], "One-time pre-registered final Track A test results.", "tab:final"),
        "paper_generalization_analysis.tex": _latex_table(["System", "Nested dev", "Final test", "Difference", "Pred. card. dev", "Pred. card. test"], [["EN2", .5691254631, .2851799355, -.2839455276, 1.0638911789, .3459119497], ["T2", .5585138660, .1569116983, -.4016021678, 1.1014014839, .3616352201]], "Development-to-test generalization gap (post-hoc descriptive comparison).", "tab:generalization"),
        "paper_computational_costs.tex": _latex_table(["System", "Parameters", "Train s", "Inference s", "Peak VRAM GiB"], [["B1",108927750,506.94,4.29,1.92],["B2",334402566,6803.45,43.73,5.58],["E2",134907654,530.56,4.99,2.30],["E3",134907654,509.92,4.62,2.30],["E4",139196934,974.29,10.47,2.32],["Final T2",134907654,138.01,4.65,2.30],["Final T3",334402566,2173.47,51.80,5.58]], "Observed computational cost on the frozen hardware; times are hardware-specific.", "tab:cost"),
    }
    for name, text in tables.items():
        (latex / name).write_text(text, encoding="utf-8")


def _write_synthesis_figures(root: Path, master: list[dict[str, Any]]) -> None:
    import os
    cache = root / ".cache/matplotlib-paper"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    out = root / "figures/brighter_ptbr/paper"
    out.mkdir(parents=True, exist_ok=True)
    by_id = {x["experiment_id"]: x for x in master}
    stages = ["B0", "B1", "B2", "L4", "E3", "T2", "EN2", "FINAL_EN2"]
    values = [by_id[x]["macro_f1"] for x in stages]
    colors = ["#4c78a8"] * 7 + ["#c44e52"]
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(stages, values, color=colors)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Experimental stage / system")
    ax.set_ylim(0, 0.65)
    ax.set_title("BRIGHTER PT-BR: controlled development progression and final test")
    for i, value in enumerate(values):
        ax.text(i, value + .012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(6.5, color="#777777", linewidth=.9, linestyle="--")
    ax.text(5.55, .625, "Leakage-safe development", ha="center", fontsize=8)
    ax.text(7.0, .625, "Final test", ha="center", fontsize=8, color="#8b1a1a")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"macro_f1_evolution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    nodes = [
        (0.06, "Data audit\nTrack A PT-BR"), (0.22, "5-fold CV\nbaselines"),
        (0.38, "Loss\nablation"), (0.54, "Encoder × loss\nmatrix"),
        (0.70, "Nested 5×3\nthresholds"), (0.84, "Nested\nensemble"),
        (0.96, "Frozen\nfinal test"),
    ]
    for x, label in nodes:
        ax.text(x, .55, label, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=.45", facecolor="#e9f1f7", edgecolor="#42677f"))
    for (x1, _), (x2, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(x2 - .052, .55), xytext=(x1 + .052, .55), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(.5, .15, "Frozen data/folds/protocol  →  leakage-safe selection  →  one-time pre-registered test", ha="center", fontsize=9)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"paper_experimental_pipeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_paper_package(root: Path) -> dict[str, Any]:
    root = root.resolve()
    state = _json(root / "metadata/brighter_ptbr/final/experiment_state.json")
    if state["state"] != "TEST_EVALUATED" or state["search_space"] != "CLOSED" or state["further_tuning_allowed"]:
        raise RuntimeError("Paper freeze requires TEST_EVALUATED with CLOSED search space.")
    paper = root / "reports/brighter_ptbr/paper"
    meta_paper = root / "metadata/brighter_ptbr/paper"
    paper.mkdir(parents=True, exist_ok=True); meta_paper.mkdir(parents=True, exist_ok=True)
    master = build_master_results(root)
    fields = ["experiment_id", "encoder", "loss", "decision_strategy", "evaluation_protocol", "macro_f1",
              "macro_f1_std", "micro_f1", "neutral_fp", "predicted_cardinality", "disgust_f1", "fear_f1",
              "surprise_f1", "training_seconds", "parameter_count", "scientific_status", "source_artifact"]
    _write_csv(paper / "master_results.csv", master, fields)
    claims = build_claims(root, master)
    manifest = {
        "package": "BRIGHTER_PTBR_PAPER_EVIDENCE_FREEZE",
        "experiment_state": "TEST_EVALUATED", "search_space": "CLOSED", "further_tuning_allowed": False,
        "scientific_status_vocabulary": sorted(STATUSES), "claims": claims,
        "master_results": "reports/brighter_ptbr/paper/master_results.csv",
    }
    (meta_paper / "paper_evidence_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, text in _paper_texts(root, master, claims).items():
        (paper / name).write_text(text.rstrip() + "\n", encoding="utf-8")
    _write_tables(root, master)
    _write_synthesis_figures(root, master)
    source_files = [
        "metadata/brighter_ptbr/eda/eda_summary.json",
        "metadata/brighter_ptbr/experiments/baseline_summary.json",
        "metadata/brighter_ptbr/experiments/imbalance_loss_summary.json",
        "metadata/brighter_ptbr/experiments/encoder_summary.json",
        "metadata/brighter_ptbr/experiments/threshold_calibration_summary.json",
        "metadata/brighter_ptbr/experiments/ensemble_summary.json",
        "metadata/brighter_ptbr/final/final_experiment_manifest.json",
        "reports/brighter_ptbr/final/final_test_results.json",
        "metadata/brighter_ptbr/final/generalization_gap_summary.json",
        "reports/brighter_ptbr/experiments/experiment_registry.csv",
    ]
    hashes = {path: _sha256(root / path) for path in source_files}
    (meta_paper / "source_artifact_hashes.json").write_text(json.dumps(hashes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"master_rows": len(master), "claims": len(claims), "paper_dir": str(paper)}
