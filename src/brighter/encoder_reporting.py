"""Consolidate the frozen BRIGHTER PT-BR encoder x loss experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import (
    DEV_POOL_SIZE, EXPECTED_LABELS, attach_folds, calculate_metrics,
    construct_development_pool, development_input_hashes, persist_cv_folds,
    validate_oof,
)
from .manifest import load_json, sha256_file, write_json_atomic

NEW = (
    ("E1", "BERTweetBR", "BCE", "bertweetbr_bce"),
    ("E2", "BERTweetBR", "Weighted Focal", "bertweetbr_weighted_focal"),
    ("E3", "BERTweetBR", "ASL", "bertweetbr_asl"),
    ("E4", "Albertina 100M PT-BR", "BCE", "albertina100m_bce"),
    ("E5", "Albertina 100M PT-BR", "Weighted Focal", "albertina100m_weighted_focal"),
    ("E6", "Albertina 100M PT-BR", "ASL", "albertina100m_asl"),
)
COMPARATORS = (
    ("B1", "BERTimbau Base", "BCE", "bertimbau_base_bce"),
    ("L3", "BERTimbau Base", "Weighted Focal", "bertimbau_base_weighted_focal"),
    ("L4", "BERTimbau Base", "ASL", "bertimbau_base_asl"),
    ("B2", "BERTimbau Large", "BCE", "bertimbau_large_bce"),
)
ALL = COMPARATORS + NEW


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


def _save(fig: Any, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(root / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_all(root: Path) -> dict[str, dict[str, Any]]:
    reports = root / "reports" / "brighter_ptbr" / "experiments"
    pool = attach_folds(construct_development_pool(root), persist_cv_folds(root, construct_development_pool(root)))
    output: dict[str, dict[str, Any]] = {}
    reference_ids: list[str] | None = None
    reference_folds: list[int] | None = None
    for short, encoder, loss, slug in ALL:
        summary = load_json(reports / f"{slug}_results.json")
        oof_path = root / "artifacts" / "predictions" / f"{slug}_oof.parquet"
        oof = pd.read_parquet(oof_path)
        validate_oof(oof, pool)
        ids, folds = oof["id"].astype(str).tolist(), oof["fold"].astype(int).tolist()
        if reference_ids is None:
            reference_ids, reference_folds = ids, folds
        elif ids != reference_ids or folds != reference_folds:
            raise RuntimeError(f"Ordered OOF IDs/folds differ for {short}")
        y = oof[[f"{x}_true" for x in EXPECTED_LABELS]].to_numpy(dtype=int)
        p = oof[[f"{x}_prob" for x in EXPECTED_LABELS]].to_numpy(dtype=float)
        output[short] = {
            "short": short, "encoder": encoder, "loss": loss, "slug": slug,
            "summary": summary, "oof": oof, "metrics": calculate_metrics(y, p),
            "oof_sha256": sha256_file(oof_path),
        }
        if summary["oof_sha256"] != output[short]["oof_sha256"]:
            raise RuntimeError(f"OOF hash mismatch for {short}")
    return output


def _main_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, encoder, loss, _ in ALL:
        m = data[short]["metrics"]
        row = {"model": short, "encoder": encoder, "loss": loss,
               "macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"],
               "macro_average_precision": m["macro_average_precision"],
               "neutral_false_positive_rate": m["zero_label"]["neutral_false_positive_rate"]}
        for label in ("disgust", "fear", "surprise"):
            row[f"ap_{label}"] = m["per_label"][label]["average_precision"]
            row[f"f1_{label}"] = m["per_label"][label]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def _fold_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, encoder, loss, _ in NEW:
        for m in data[short]["summary"]["fold_metrics"]:
            multi = m["groups"]["multi_label"]
            rows.append({
                "model": short, "experiment_id": data[short]["summary"]["experiment_id"],
                "encoder": encoder, "loss": loss, "fold": m["fold"],
                "macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"],
                "macro_average_precision": m["macro_average_precision"],
                "hamming_loss": m["hamming_loss"], "jaccard": m["jaccard_samples"],
                "subset_accuracy": m["subset_accuracy"],
                "exact_neutral_accuracy": m["zero_label"]["exact_neutral_accuracy"],
                "neutral_false_positive_rate": m["zero_label"]["neutral_false_positive_rate"],
                "multilabel_support": multi["support"], "multilabel_macro_f1": multi["macro_f1"],
                "multilabel_micro_f1": multi["micro_f1"], "multilabel_jaccard": multi["jaccard_samples"],
                "multilabel_subset_accuracy": multi["subset_accuracy"],
                "gold_label_cardinality": m["gold_label_cardinality"],
                "predicted_label_cardinality": m["predicted_label_cardinality"],
                "training_seconds": m["cost"]["training_seconds"],
                "inference_seconds": m["cost"]["inference_seconds"],
                "peak_vram_bytes": m["cost"]["peak_gpu_memory_bytes"],
            })
    return pd.DataFrame(rows)


def _per_label(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, encoder, loss, _ in NEW:
        summary = data[short]["summary"]
        scopes = [("oof", None, data[short]["metrics"])] + [("fold", x["fold"], x) for x in summary["fold_metrics"]]
        for scope, fold, metrics in scopes:
            for label in EXPECTED_LABELS:
                x = metrics["per_label"][label]
                predicted = x["confusion"]["tp"] + x["confusion"]["fp"]
                rows.append({
                    "model": short, "experiment_id": summary["experiment_id"], "encoder": encoder,
                    "loss": loss, "scope": scope, "fold": fold, "label": label,
                    "precision": x["precision"], "recall": x["recall"], "f1": x["f1"],
                    "average_precision": x["average_precision"], "support": x["support"],
                    "predicted_positive": predicted, "gold_prevalence": x["gold_prevalence"],
                    "predicted_prevalence": x["predicted_prevalence"], "prevalence_delta": x["prevalence_delta"],
                })
    return pd.DataFrame(rows)


def _cardinality(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "model": short, "encoder": encoder, "loss": loss,
        "gold_label_cardinality": data[short]["metrics"]["gold_label_cardinality"],
        "predicted_label_cardinality": data[short]["metrics"]["predicted_label_cardinality"],
        "delta": data[short]["metrics"]["predicted_label_cardinality"] - data[short]["metrics"]["gold_label_cardinality"],
    } for short, encoder, loss, _ in ALL])


def _delta(data: dict[str, dict[str, Any]], left: str, right: str) -> dict[str, float | str]:
    a, b = data[left]["metrics"], data[right]["metrics"]
    return {"comparison": f"{left} - {right}", "macro_f1": a["macro_f1"] - b["macro_f1"],
            "micro_f1": a["micro_f1"] - b["micro_f1"],
            "neutral_fp": a["zero_label"]["neutral_false_positive_rate"] - b["zero_label"]["neutral_false_positive_rate"],
            **{f"ap_{x}": a["per_label"][x]["average_precision"] - b["per_label"][x]["average_precision"] for x in ("disgust", "fear", "surprise")}}


def _cost(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for short, encoder, loss, _ in NEW:
        s, cfg = data[short]["summary"], data[short]["summary"]["config"]
        c = s["cost"]
        training = c["training_seconds_total"]
        baseline = data["B1"]["metrics"]["macro_f1"] if loss == "BCE" else data[{"Weighted Focal": "L3", "ASL": "L4"}[loss]]["metrics"]["macro_f1"]
        rows.append({"model": short, "encoder": encoder, "loss": loss, "parameters": c["parameters"],
                     "macro_f1": data[short]["metrics"]["macro_f1"],
                     "training_seconds_total": training, "training_seconds_mean_fold": training / 5,
                     "inference_seconds_oof": c["inference_seconds_total"],
                     "peak_vram_bytes": c["peak_gpu_memory_bytes"],
                     "physical_batch": cfg["training"]["physical_batch_size"],
                     "accumulation": cfg["training"]["gradient_accumulation_steps"],
                     "effective_batch": cfg["training"]["effective_batch_size"],
                     "examples_per_second": DEV_POOL_SIZE * 4 * 4 / training,
                     "macro_f1_gain_per_training_minute": (data[short]["metrics"]["macro_f1"] - baseline) / (training / 60)})
    return pd.DataFrame(rows)


def _figures(root: Path, data: dict[str, dict[str, Any]], main: pd.DataFrame,
             per_label: pd.DataFrame, cardinality: pd.DataFrame, cost: pd.DataFrame) -> None:
    out = root / "figures" / "brighter_ptbr" / "experiments"
    new_main = main[main.model.str.startswith("E")]
    fig, ax = plt.subplots(figsize=(9, 5)); colors = {"BCE":"#4C78A8", "Weighted Focal":"#F58518", "ASL":"#54A24B"}
    ax.bar(new_main.model, new_main.macro_f1, color=[colors[x] for x in new_main.loss])
    for row in new_main.itertuples(index=False): ax.text(int(row.model[1:])-1, row.macro_f1+.008, f"{row.macro_f1:.3f}", ha="center", fontsize=9)
    handles = [plt.Rectangle((0,0),1,1,color=color,label=loss) for loss,color in colors.items()]
    ax.legend(handles=handles, title="Loss"); ax.set(title="Encoder × loss — OOF Macro-F1 (E1–E3 BERTweetBR; E4–E6 Albertina)", ylabel="Macro-F1", ylim=(0, .6)); ax.grid(axis="y", alpha=.25)
    _save(fig, out, "encoder_macro_f1")
    agg = per_label[per_label.scope == "oof"]
    for metric, name, title, labels in (
        ("f1", "encoder_f1_by_label", "OOF F1 by class", EXPECTED_LABELS),
        ("average_precision", "encoder_ap_by_label", "OOF AP by class", EXPECTED_LABELS),
        ("f1", "encoder_minority_f1", "Minority-label F1", ("disgust", "fear", "surprise")),
        ("average_precision", "encoder_minority_ap", "Minority-label AP", ("disgust", "fear", "surprise")),
    ):
        pivot = agg[agg.label.isin(labels)].pivot(index="label", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(10, 5)); pivot.plot.bar(ax=ax); ax.set(title=title, xlabel="", ylabel=metric, ylim=(0, max(.55, pivot.max().max()+.08))); ax.grid(axis="y", alpha=.2)
        _save(fig, out, name)
    fig, ax = plt.subplots(figsize=(12, 5)); p = agg.pivot(index="label", columns="model", values="predicted_prevalence"); p.plot.bar(ax=ax); ax.plot(np.arange(len(EXPECTED_LABELS)), agg.groupby("label", sort=False).gold_prevalence.first().reindex(EXPECTED_LABELS), "ko--", label="gold"); ax.set(title="Gold vs predicted prevalence", ylabel="Prevalence", xlabel=""); ax.legend(ncol=4)
    _save(fig, out, "encoder_gold_vs_predicted_prevalence")
    fig, ax = plt.subplots(figsize=(10, 5)); c = cardinality[cardinality.model.str.startswith("E")]; x=np.arange(len(c)); ax.bar(x-.18,c.gold_label_cardinality,.36,label="Gold"); ax.bar(x+.18,c.predicted_label_cardinality,.36,label="Predicted"); ax.set_xticks(x,c.model); ax.set(title="Gold vs predicted label cardinality", ylabel="Labels/example"); ax.legend(); ax.grid(axis="y",alpha=.2)
    _save(fig, out, "encoder_cardinality")
    fig, ax = plt.subplots(figsize=(9,5)); ax.bar(new_main.model,new_main.neutral_false_positive_rate,color="#E45756"); ax.set(title="Neutral false-positive rate",ylabel="Neutral FP",ylim=(0,1)); ax.grid(axis="y",alpha=.2)
    _save(fig,out,"encoder_neutral_fp")
    fig, ax=plt.subplots(figsize=(8,5)); ax.scatter(cost.training_seconds_total/60, new_main.macro_f1, s=80); [ax.annotate(r.model,(r.training_seconds_total/60,r.macro_f1),xytext=(5,5),textcoords="offset points") for r in cost.itertuples()]; ax.set(title="Computational cost × OOF Macro-F1",xlabel="Training minutes",ylabel="Macro-F1"); ax.grid(alpha=.25)
    _save(fig,out,"encoder_cost_vs_macro_f1")


def _update_registry(root: Path, data: dict[str, dict[str, Any]]) -> None:
    path = root / "reports" / "brighter_ptbr" / "experiments" / "experiment_registry.csv"
    old = pd.read_csv(path)
    hashes = development_input_hashes(root)
    rows = []
    for short, encoder, loss, slug in NEW:
        s, cfg, m = data[short]["summary"], data[short]["summary"]["config"], data[short]["metrics"]
        rows.append({
            "experiment_id": s["experiment_id"], "model": encoder, "encoder": encoder,
            "hugging_face_checkpoint": cfg["model_name"], "architecture_family": cfg["architecture_family"],
            "loss": loss, "threshold": .5, "threshold_strategy": "fixed_0.5", "fold": "all", "folds": 5,
            "seed": cfg["seed"], "learning_rate": cfg["training"]["learning_rate"], "epochs": cfg["training"]["epochs"],
            "physical_batch": cfg["training"]["physical_batch_size"], "accumulation": cfg["training"]["gradient_accumulation_steps"],
            "effective_batch_size": cfg["training"]["effective_batch_size"], "max_length": cfg["max_length"],
            "parameter_count": s["cost"]["parameters"], "training_time": s["cost"]["training_seconds_total"],
            "peak_gpu_memory_bytes": s["cost"]["peak_gpu_memory_bytes"], "macro_f1_mean": s["cross_validation"]["macro_f1"]["mean"],
            "macro_f1_std": s["cross_validation"]["macro_f1"]["std"], "micro_f1_mean": s["cross_validation"]["micro_f1"]["mean"],
            "oof_path": s["oof_path"], "oof_sha256": s["oof_sha256"], "config_sha256": s["config_sha256"],
            "status": "complete", "timestamp": s["completed_timestamp_utc"], "train_sha256": hashes["train"],
            "dev_sha256": hashes["dev"], "git_commit": "NOT_A_GIT_REPOSITORY",
            "mixed_precision": cfg["training"]["mixed_precision"],
        })
    add = pd.DataFrame(rows)
    old = old[~old.experiment_id.isin(add.experiment_id)]
    _atomic_csv(pd.concat([old, add], ignore_index=True), path)


def _report(data: dict[str, dict[str, Any]], main: pd.DataFrame, cardinality: pd.DataFrame,
            cost: pd.DataFrame, deltas: pd.DataFrame) -> str:
    def mm(short: str) -> dict[str, Any]: return data[short]["metrics"]
    lines = ["# BRIGHTER PT-BR — encoder × loss experiment", "", "## 1. Experimental question", "",
             "We tested whether social-media PT-BR pretraining (BERTweetBR) or a DeBERTa-family PT-BR encoder (Albertina) improves Track A multilabel emotion recognition, and whether Weighted Focal/ASL transfer across encoders.", "",
             "## 2. Protocol", "", "Only encoder and loss varied. The immutable 2,426-example development pool, exact persisted five folds (seed 42), AdamW/LR 2e-5, four epochs, weight decay 0.01, linear warmup, max length 128, effective batch 16, checkpoint rule and fixed threshold 0.50 were retained. No preprocessing, sampling, augmentation, translation, external/synthetic data, label changes, Track B, or test access occurred. Weighted Focal weights use only each fold's training subset; ASL is gamma_pos=0, gamma_neg=4, clip=0.05.", "",
             "Albertina was run in FP32 because its DeBERTa attention mask overflows under both FP16 and BF16 autocast in the installed PyTorch/Transformers combination. This technical exception was established by real-batch forward/backward smoke tests before E4; all other protocol choices stayed fixed.", "",
             "## 3. Models", "", "BERTimbau Base/Large are Portuguese BERT controls. [BERTweetBR](https://huggingface.co/melll-uff/bertweetbr) is RoBERTa-like and pretrained on Brazilian tweets; its frozen tokenizer had optional normalization disabled, so raw text was preserved. [Albertina 100M PT-BR](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder) is a PT-BR DeBERTa-family encoder. All models emit six independent logits; sigmoid is used once only for probabilities.", "",
             "## 4. Main results", "", "| Encoder | Loss | Macro-F1 | Micro-F1 | AP disgust | F1 disgust | AP fear | F1 fear | AP surprise | F1 surprise | Neutral FP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in main.itertuples(index=False):
        lines.append(f"| {r.encoder} ({r.model}) | {r.loss} | {r.macro_f1:.4f} | {r.micro_f1:.4f} | {r.ap_disgust:.4f} | {r.f1_disgust:.4f} | {r.ap_fear:.4f} | {r.f1_fear:.4f} | {r.ap_surprise:.4f} | {r.f1_surprise:.4f} | {r.neutral_false_positive_rate:.4f} |")
    lines += ["", "## 5. Minority labels", "",
              f"E2 improved ranking over L3 for disgust ({mm('E2')['per_label']['disgust']['average_precision']:.4f} vs {mm('L3')['per_label']['disgust']['average_precision']:.4f}), fear ({mm('E2')['per_label']['fear']['average_precision']:.4f} vs {mm('L3')['per_label']['fear']['average_precision']:.4f}), and surprise ({mm('E2')['per_label']['surprise']['average_precision']:.4f} vs {mm('L3')['per_label']['surprise']['average_precision']:.4f}). Its disgust F1 fell to {mm('E2')['per_label']['disgust']['f1']:.4f}: precision stayed {mm('E2')['per_label']['disgust']['precision']:.4f}, but recall was only {mm('E2')['per_label']['disgust']['recall']:.4f}. E5 did not transfer this benefit reliably and was fold-unstable.", "",
              f"E3 obtained disgust F1 {mm('E3')['per_label']['disgust']['f1']:.4f}, fear F1 {mm('E3')['per_label']['fear']['f1']:.4f}, and surprise F1 {mm('E3')['per_label']['surprise']['f1']:.4f}; its strong Macro-F1 is therefore not just a majority-label effect.", "",
              "## 6. Encoder effect", "", "Under BCE, E1-B1 and E4-B1 isolate the approximate encoder changes. BERTweetBR was essentially neutral; Albertina was negative under the frozen optimization protocol.", "",
              "## 7. Loss effect", "", "Within BERTweetBR, Weighted Focal improves over BCE but ASL is much stronger. Within Albertina, Weighted Focal is negative and unstable, while ASL substantially rescues performance. Therefore loss efficacy is not encoder-invariant.", "", "## 8. Interaction", "", "| Contrast | Δ Macro-F1 | Δ Micro-F1 | Δ AP disgust | Δ AP fear | Δ AP surprise | Δ Neutral FP |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in deltas.itertuples(index=False): lines.append(f"| {r.comparison} | {r.macro_f1:+.4f} | {r.micro_f1:+.4f} | {r.ap_disgust:+.4f} | {r.ap_fear:+.4f} | {r.ap_surprise:+.4f} | {r.neutral_fp:+.4f} |")
    lines += ["", "The divergent E2/E5 and E3/E6 contrasts demonstrate a material encoder × loss interaction rather than an additive encoder-only or loss-only effect.", "",
              "## 9. Calibration behavior", "", "This section describes threshold-decision prevalence, not formal probability calibration.", "", "| Model | Gold cardinality | Predicted cardinality | Delta |", "|---|---:|---:|---:|"]
    for r in cardinality.itertuples(index=False): lines.append(f"| {r.model} | {r.gold_label_cardinality:.4f} | {r.predicted_label_cardinality:.4f} | {r.delta:+.4f} |")
    for short, base in (("E3", "L4"), ("E6", "L4")):
        m, b = mm(short), mm(base)
        contributors = sorted(EXPECTED_LABELS, key=lambda x: m["per_label"][x]["prevalence_delta"], reverse=True)[:3]
        lines += ["", f"**{short} ASL vs {base}:** Neutral FP is {m['zero_label']['neutral_false_positive_rate']:.4f} vs {b['zero_label']['neutral_false_positive_rate']:.4f}; predicted cardinality is {m['predicted_label_cardinality']:.4f} vs gold {m['gold_label_cardinality']:.4f}. Largest prevalence deltas: " + ", ".join(f"{x} {m['per_label'][x]['prevalence_delta']:+.4f}" for x in contributors) + ". The recall gain must therefore be read together with precision and neutral behavior."]
    lines += ["", "**ASL conclusion.** E3 still superpredicts, but less than L4 (both cardinality delta and Neutral FP improve). Its recall increase is accompanied by higher minority F1/AP and higher Micro-F1, so the precision loss is compensated in aggregate, though decision behavior remains aggressive. E6 superpredicts slightly more than L4, ranks every minority label worse, and loses Macro/Micro-F1; its recall does not compensate for the precision and neutral-behavior degradation.", "",
              "**Weighted Focal conclusion.** E2 raises AP for all three minorities and keeps Neutral FP below L3. Disgust precision remains high, but recall drops enough that disgust F1 falls; fear F1 improves, while surprise F1 falls despite improved AP. Thus BERTweetBR improves minority ranking more consistently than fixed-threshold recovery. E5 reduces minority AP/F1 and is not a successful transfer of L3."]
    lines += ["", "## 10. Computational cost", "", "| Model | Params | Training min | Mean/fold min | OOF inference s | Peak VRAM GiB | Physical × accum = effective | Examples/s |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in cost.itertuples(index=False): lines.append(f"| {r.model} | {int(r.parameters):,} | {r.training_seconds_total/60:.2f} | {r.training_seconds_mean_fold/60:.2f} | {r.inference_seconds_oof:.2f} | {r.peak_vram_bytes/2**30:.2f} | {r.physical_batch} × {r.accumulation} = {r.effective_batch} | {r.examples_per_second:.2f} |")
    lines += ["", "## 11. Selection", "", "Advance at most three systems:", "",
              "1. **E3 BERTweetBR + ASL** — best Macro-F1 and strong fold stability/minority F1; retain the explicit overprediction caveat.",
              "2. **E2 BERTweetBR + Weighted Focal** — lower Macro-F1 but best minority ranking and controlled Neutral FP, complementary to E3.",
              "3. **B2 BERTimbau Large + BCE** — retained as a stronger-encoder, calibrated-behavior comparator and likely error-complementary control.", "",
              "E4–E6 are not selected: Albertina does not offset its slower FP32 cost or lower performance. E5 is additionally unstable across folds.", "",
              "## 12. Next experiment", "", "Recommended but **not executed**: BERTimbau Large + Weighted Focal, followed by leakage-safe per-label threshold calibration for the selected systems. No ensemble, augmentation, translation, Track B, or final-test evaluation was run.", "",
              "## Validation", "", f"All six new OOF artifacts contain exactly {DEV_POOL_SIZE:,} unique examples with identical ordered IDs/folds, finite [0,1] probabilities, matching saved SHA-256 hashes, and unchanged train/dev hashes. The official test remains **SEALED — NOT LOADED OR EVALUATED**.", ""]
    return "\n".join(lines)


def generate_encoder_reports(root: Path) -> dict[str, Any]:
    data = _load_all(root)
    main, fold, per_label = _main_table(data), _fold_table(data), _per_label(data)
    cardinality, cost = _cardinality(data), _cost(data)
    comparisons = (("E1","B1"),("E2","E1"),("E3","E1"),("E4","B1"),("E5","E4"),("E6","E4"),("E2","L3"),("E3","L4"),("E5","L3"),("E6","L4"))
    deltas = pd.DataFrame([_delta(data, a, b) for a, b in comparisons])
    report_dir = root / "reports" / "brighter_ptbr" / "experiments"
    _atomic_csv(fold, report_dir / "encoder_fold_metrics.csv")
    _atomic_csv(per_label, report_dir / "encoder_per_label.csv")
    _atomic_csv(cardinality, report_dir / "encoder_cardinality.csv")
    _atomic_csv(main, report_dir / "encoder_main_results.csv")
    _atomic_csv(deltas, report_dir / "encoder_effect_decomposition.csv")
    _atomic_csv(cost, report_dir / "encoder_cost.csv")
    _figures(root, data, main, per_label, cardinality, cost)
    _update_registry(root, data)
    _atomic_text(_report(data, main, cardinality, cost, deltas), report_dir / "encoder_results.md")
    summary = {
        "development_examples": DEV_POOL_SIZE, "folds": 5, "threshold": .5,
        "dataset_hashes": development_input_hashes(root),
        "test_policy": "SEALED — NOT LOADED OR EVALUATED",
        "selected_systems": ["E3_BERTWEETBR_ASL", "E2_BERTWEETBR_WEIGHTED_FOCAL", "B2_BERTIMBAU_LARGE_BCE"],
        "main_results": main.to_dict(orient="records"), "effect_decomposition": deltas.to_dict(orient="records"),
        "cardinality": cardinality.to_dict(orient="records"), "cost": cost.to_dict(orient="records"),
        "experiments": {short: {"experiment_id": data[short]["summary"]["experiment_id"],
            "oof_path": data[short]["summary"]["oof_path"], "oof_sha256": data[short]["oof_sha256"],
            "metrics": data[short]["metrics"]} for short, _, _, _ in NEW},
    }
    write_json_atomic(root / "metadata" / "brighter_ptbr" / "experiments" / "encoder_summary.json", summary)
    return summary
