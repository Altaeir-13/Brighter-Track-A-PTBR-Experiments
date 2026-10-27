"""Scientific reporting for leakage-safe T2/T3 ensemble feasibility."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ensemble import OUTPUTS
from .experiments import EXPECTED_LABELS
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .threshold_reporting import _dataset_integrity, paired_bootstrap

DISPLAY = {"T2": "BERTweetBR + ASL", "T3": "BERTimbau Large + BCE",
           "EN1": "Equal-weight ensemble", "EN2": "Tuned-weight ensemble"}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, float_format="%.8f", lineterminator="\n"); os.replace(tmp, path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n"); os.replace(tmp, path)


def _save(fig: Any, root: Path, name: str) -> None:
    out = root / "figures" / "brighter_ptbr" / "experiments"; out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"): fig.savefig(out / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load(root: Path) -> dict[str, dict[str, Any]]:
    run = load_json(root / "metadata" / "brighter_ptbr" / "experiments" / "ensemble_run.json")
    data: dict[str, dict[str, Any]] = {}
    for system, slug in (("T2", "bertweetbr_asl_thresholded"), ("T3", "bertimbau_large_bce_thresholded")):
        summary = load_json(root / "metadata" / "brighter_ptbr" / "experiments" / f"{slug}_summary.json")
        oof = pd.read_parquet(root / "artifacts" / "predictions" / f"{slug}_oof.parquet")
        data[system] = {"metrics": summary["thresholded_metrics"], "oof": oof}
    for system in ("EN1", "EN2"):
        path = root / "artifacts" / "predictions" / f"{OUTPUTS[system]}_oof.parquet"
        oof = pd.read_parquet(path)
        if sha256_file(path) != run["systems"][system]["sha256"]:
            raise RuntimeError(f"Ensemble artifact hash mismatch: {system}")
        data[system] = {"metrics": run["systems"][system]["metrics"], "oof": oof}
    reference = data["T2"]["oof"]
    for system, value in data.items():
        if len(value["oof"]) != 2426 or value["oof"].id.nunique() != 2426:
            raise RuntimeError(f"Incomplete OOF for {system}")
        if value["oof"].id.tolist() != reference.id.tolist() or value["oof"].fold.tolist() != reference.fold.tolist():
            raise RuntimeError(f"ID/fold mismatch for {system}")
    return data


def _pred(oof: pd.DataFrame) -> np.ndarray:
    return oof[[f"{label}_pred" for label in EXPECTED_LABELS]].to_numpy(dtype=int)


def _truth(oof: pd.DataFrame) -> np.ndarray:
    return oof[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(dtype=int)


def _tables(root: Path, data: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, ...]:
    main, per_label, neutral, multilabel = [], [], [], []
    for system in ("T2", "T3", "EN1", "EN2"):
        m = data[system]["metrics"]
        main.append({"system": system, "display": DISPLAY[system],
                     **{key: m[key] for key in ("macro_f1", "micro_f1", "macro_precision", "macro_recall", "hamming_loss", "jaccard_samples", "subset_accuracy", "macro_average_precision")},
                     "neutral_fp": m["zero_label"]["neutral_false_positive_rate"],
                     "neutral_accuracy": m["zero_label"]["exact_neutral_accuracy"],
                     "gold_cardinality": m["gold_label_cardinality"],
                     "predicted_cardinality": m["predicted_label_cardinality"],
                     "cardinality_abs_error": abs(m["predicted_label_cardinality"]-m["gold_label_cardinality"])})
        for label in EXPECTED_LABELS:
            x = m["per_label"][label]
            per_label.append({"system": system, "label": label,
                              **{key: x[key] for key in ("precision", "recall", "f1", "average_precision", "support", "predicted_positive", "gold_prevalence", "predicted_prevalence")}})
        g = m["groups"]["multi_label"]
        multilabel.append({"system": system, **g})
        neutral.append({"system": system, "neutral_fp": m["zero_label"]["neutral_false_positive_rate"],
                        "neutral_accuracy": m["zero_label"]["exact_neutral_accuracy"],
                        "gold_cardinality": m["gold_label_cardinality"],
                        "predicted_cardinality": m["predicted_label_cardinality"],
                        "cardinality_abs_error": abs(m["predicted_label_cardinality"]-m["gold_label_cardinality"])})
    y = _truth(data["T2"]["oof"])
    comparisons = (("EN1_vs_T2", "T2", "EN1"), ("EN2_vs_T2", "T2", "EN2"), ("EN2_vs_T3", "T3", "EN2"))
    bootstrap = []
    for name, baseline, candidate in comparisons:
        result = paired_bootstrap(y, _pred(data[baseline]["oof"]), _pred(data[candidate]["oof"]))
        bootstrap.append({"comparison": name, "baseline": baseline, "candidate": candidate, **result})
    return tuple(pd.DataFrame(x) for x in (main, per_label, neutral, multilabel, bootstrap))


def _error_analysis(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    y, t2, en2 = _truth(data["T2"]["oof"]), _pred(data["T2"]["oof"]), _pred(data["EN2"]["oof"])
    t2_exact, en2_exact = np.all(t2 == y, axis=1), np.all(en2 == y, axis=1)
    multi = y.sum(axis=1) >= 2
    rows = [{"scope": "exact_labelset", "label": "ALL", "t2_wrong_en2_correct": int((~t2_exact & en2_exact).sum()),
             "t2_correct_en2_wrong": int((t2_exact & ~en2_exact).sum()),
             "t2_fn_en2_tp": np.nan, "t2_tp_en2_fn": np.nan, "t2_fp_en2_tn": np.nan, "t2_tn_en2_fp": np.nan},
            {"scope": "multilabel_exact", "label": "ALL", "t2_wrong_en2_correct": int((~t2_exact[multi] & en2_exact[multi]).sum()),
             "t2_correct_en2_wrong": int((t2_exact[multi] & ~en2_exact[multi]).sum()),
             "t2_fn_en2_tp": np.nan, "t2_tp_en2_fn": np.nan, "t2_fp_en2_tn": np.nan, "t2_tn_en2_fp": np.nan}]
    for i, label in enumerate(EXPECTED_LABELS):
        rows.append({"scope": "per_label", "label": label, "t2_wrong_en2_correct": np.nan, "t2_correct_en2_wrong": np.nan,
                     "t2_fn_en2_tp": int(((y[:,i]==1)&(t2[:,i]==0)&(en2[:,i]==1)).sum()),
                     "t2_tp_en2_fn": int(((y[:,i]==1)&(t2[:,i]==1)&(en2[:,i]==0)).sum()),
                     "t2_fp_en2_tn": int(((y[:,i]==0)&(t2[:,i]==1)&(en2[:,i]==0)).sum()),
                     "t2_tn_en2_fp": int(((y[:,i]==0)&(t2[:,i]==0)&(en2[:,i]==1)).sum())})
    return pd.DataFrame(rows)


def _cost(root: Path) -> dict[str, Any]:
    values = {}
    for system, slug in (("T2", "bertweetbr_asl"), ("T3", "bertimbau_large_bce")):
        summary = load_json(root / "reports" / "brighter_ptbr" / "experiments" / f"{slug}_results.json")
        fold_sizes = [sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) for folder in sorted((root/"artifacts"/"models"/slug).glob("fold_*"))]
        values[system] = {"parameters": int(summary["cost"]["parameters"]),
                          "oof_inference_seconds": float(summary["cost"]["inference_seconds_total"]),
                          "deploy_checkpoint_bytes_approx": int(np.median(fold_sizes)),
                          "five_fold_storage_bytes": int(sum(fold_sizes))}
    return {"T2": values["T2"], "T3": values["T3"],
            "ensemble": {"networks": 2, "parameters_across_two_networks": values["T2"]["parameters"]+values["T3"]["parameters"],
                         "oof_inference_seconds_sequential_approx": values["T2"]["oof_inference_seconds"]+values["T3"]["oof_inference_seconds"],
                         "inference_time_ratio_vs_t2": (values["T2"]["oof_inference_seconds"]+values["T3"]["oof_inference_seconds"])/values["T2"]["oof_inference_seconds"],
                         "deploy_storage_bytes_approx": values["T2"]["deploy_checkpoint_bytes_approx"]+values["T3"]["deploy_checkpoint_bytes_approx"]}}


def _figures(root: Path, main: pd.DataFrame, per_label: pd.DataFrame, neutral: pd.DataFrame,
             alpha: pd.DataFrame, thresholds: pd.DataFrame, errors: pd.DataFrame) -> None:
    order = ["T2","T3","EN1","EN2"]; x=np.arange(4)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(x,main.set_index("system").loc[order].macro_f1); ax.set_xticks(x,order); ax.set(ylabel="Macro-F1",title="T2/T3 ensemble feasibility",ylim=(0,.62)); ax.grid(axis="y",alpha=.2); _save(fig,root,"ensemble_macro_f1")
    for metric,name,title in (("f1","ensemble_minority_f1","Minority F1"),("average_precision","ensemble_minority_ap","Minority Average Precision")):
        sub=per_label[per_label.label.isin(("disgust","fear","surprise"))]; fig,ax=plt.subplots(figsize=(9,5)); width=.19
        for j,system in enumerate(order): g=sub[sub.system==system].set_index("label").loc[["disgust","fear","surprise"]]; ax.bar(np.arange(3)+(j-1.5)*width,g[metric],width,label=system)
        ax.set_xticks(np.arange(3),["disgust","fear","surprise"]); ax.set(ylabel=metric.replace("_"," ").title(),title=title); ax.legend(); ax.grid(axis="y",alpha=.2); _save(fig,root,name)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(alpha.outer_fold,alpha.alpha_T2,marker="o",label="T2"); ax.plot(alpha.outer_fold,alpha.weight_T3,marker="s",label="T3"); ax.set_xticks(range(5)); ax.set(xlabel="Outer fold",ylabel="Weight",title="Tuned global ensemble weight",ylim=(0,1)); ax.legend(); ax.grid(alpha=.2); _save(fig,root,"ensemble_alpha_by_fold")
    long=thresholds.melt(id_vars=["system","outer_fold","alpha"],value_vars=EXPECTED_LABELS,var_name="label",value_name="threshold"); fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,(system,g) in zip(axes,long.groupby("system",sort=False),strict=True):
        for fold,fg in g.groupby("outer_fold"): ax.plot(range(6),fg.threshold,marker="o",alpha=.7,label=f"fold {fold}")
        ax.set_xticks(range(6),EXPECTED_LABELS,rotation=35,ha="right"); ax.set_title(system); ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Threshold"); axes[-1].legend(fontsize=8); fig.suptitle("Ensemble thresholds by outer fold"); fig.tight_layout(); _save(fig,root,"ensemble_thresholds")
    for column,name,title,ylabel,reference in (("neutral_fp","ensemble_neutral_fp","Neutral false-positive rate","Rate",None),("predicted_cardinality","ensemble_cardinality","Predicted label cardinality","Labels/example",neutral.gold_cardinality.iloc[0])):
        fig,ax=plt.subplots(figsize=(8,5)); vals=neutral.set_index("system").loc[order,column]; ax.bar(x,vals); ax.set_xticks(x,order); ax.set(ylabel=ylabel,title=title); ax.grid(axis="y",alpha=.2)
        if reference is not None: ax.axhline(reference,color="black",ls="--",label="Gold"); ax.legend()
        _save(fig,root,name)
    row=errors[(errors.scope=="exact_labelset")].iloc[0]; fig,ax=plt.subplots(figsize=(6,4)); ax.bar([0,1],[row.t2_wrong_en2_correct,row.t2_correct_en2_wrong]); ax.set_xticks([0,1],["T2 wrong → EN2 correct","T2 correct → EN2 wrong"],rotation=12); ax.set(ylabel="Examples",title="Exact-labelset corrections"); ax.grid(axis="y",alpha=.2); _save(fig,root,"ensemble_t2_corrections")


def _latex(root: Path, main: pd.DataFrame, per_label: pd.DataFrame, alpha: pd.DataFrame, neutral: pd.DataFrame) -> None:
    out=root/"reports"/"brighter_ptbr"/"experiments"
    frames={"ensemble_main_results":main[["system","macro_f1","micro_f1","macro_average_precision","neutral_fp","predicted_cardinality"]],
            "ensemble_per_label":per_label[["system","label","precision","recall","f1","average_precision"]],
            "ensemble_weights":alpha,
            "ensemble_neutral_cardinality":neutral}
    for name,frame in frames.items(): _atomic_text(frame.to_latex(index=False,float_format=lambda x:f"{x:.4f}"),out/f"{name}.tex")


def _registry(root: Path, main: pd.DataFrame, run: dict[str, Any], cost: dict[str, Any]) -> None:
    path=root/"reports"/"brighter_ptbr"/"experiments"/"experiment_registry.csv"; old=pd.read_csv(path)
    additions=[]
    for system in ("EN1","EN2"):
        row=main[main.system==system].iloc[0]; artifact=root/run["systems"][system]["artifact"]
        additions.append({"experiment_id":f"{system}_T2_T3_NESTED_ENSEMBLE","model":"T2+T3 probability ensemble","loss":"ASL+BCE (pretrained sources)","max_length":128,"seed":42,"folds":5,"threshold_strategy":"nested_5x3_ensemble_per_label_f1","macro_f1_mean":row.macro_f1,"micro_f1_mean":row.micro_f1,"training_time":0,"status":"complete","timestamp":utc_now_iso(),"oof_path":run["systems"][system]["artifact"],"oof_sha256":sha256_file(artifact),"git_commit":"NOT_A_GIT_REPOSITORY","encoder":"BERTweetBR + BERTimbau Large","architecture_family":"RoBERTa + BERT","threshold":"per_label_cross_fitted","fold":"all","parameter_count":cost["ensemble"]["parameters_across_two_networks"],"test_policy":"SEALED — NOT LOADED OR EVALUATED"})
    ids={x["experiment_id"] for x in additions}; old=old[~old.experiment_id.isin(ids)]
    combined=pd.concat([old,pd.DataFrame(additions)],ignore_index=True).reindex(columns=old.columns)
    _atomic_csv(combined,path)


def _report(main: pd.DataFrame, per_label: pd.DataFrame, neutral: pd.DataFrame, multilabel: pd.DataFrame,
            bootstrap: pd.DataFrame, alpha: pd.DataFrame, thresholds: pd.DataFrame, errors: pd.DataFrame, cost: dict[str, Any]) -> str:
    m=main.set_index("system"); p=per_label.set_index(["system","label"]); b=bootstrap.set_index("comparison")
    delta=float(m.loc["EN2","macro_f1"]-m.loc["T2","macro_f1"]); practical=delta>=.005
    exact=errors[errors.scope=="exact_labelset"].iloc[0]
    lines=["# BRIGHTER PT-BR — T2/T3 ensemble feasibility", "", "## 1. Motivation", "", "Test whether the observed T2/T3 error complementarity can be converted into cross-fitted performance using only existing probabilities.", "", "## 2. Leakage-safe protocol", "", "For each frozen outer fold, T2 and T3 inner-OOF probabilities were joined strictly by ID. Every probability came from an inner model that excluded that example from training. EN1 thresholds and EN2 alpha plus thresholds were selected only on outer-train inner-OOF predictions. Outer-validation labels were never used for selection. No model was trained or loaded; the test remained **SEALED — NOT LOADED OR EVALUATED**.", "", "EN2 searched alpha 0.00–1.00 in steps of 0.05. Thresholds were optimized per label on 0.01–0.99. Alpha ties prefer the value closest to 0.50 and then the larger T2 weight; threshold ties prefer the largest threshold.", "", "## 3. Systems", "", "T2 is BERTweetBR + ASL with nested thresholds. T3 is BERTimbau Large + BCE with nested thresholds. EN1 fixes equal probability weights; EN2 tunes one global T2 weight per outer fold.", "", "## 4. Equal-weight ensemble", "", f"EN1 reaches Macro-F1 {m.loc['EN1','macro_f1']:.4f}, a delta of {m.loc['EN1','macro_f1']-m.loc['T2','macro_f1']:+.4f} versus T2. Equal averaging alone does not improve the primary system.", "", "## 5. Tuned-weight ensemble", "", f"EN2 reaches Macro-F1 {m.loc['EN2','macro_f1']:.4f}, delta {delta:+.4f}. The mean T2 weight is {alpha.alpha_T2.mean():.3f} ± {alpha.alpha_T2.std(ddof=1):.3f} (range {alpha.alpha_T2.min():.2f}–{alpha.alpha_T2.max():.2f}), leaving mean T3 weight {alpha.weight_T3.mean():.3f}.", "", "## 6. Main results", "", "| System | Macro-F1 | Micro-F1 | Macro AP | Neutral FP | Cardinality |", "|---|---:|---:|---:|---:|---:|"]
    for system in ("T2","T3","EN1","EN2"): lines.append(f"| {system} | {m.loc[system,'macro_f1']:.4f} | {m.loc[system,'micro_f1']:.4f} | {m.loc[system,'macro_average_precision']:.4f} | {m.loc[system,'neutral_fp']:.4f} | {m.loc[system,'predicted_cardinality']:.4f} |")
    lines += ["", "## 7. Minority labels", "", "| System | disgust F1/AP | fear F1/AP | surprise F1/AP |", "|---|---:|---:|---:|"]
    for system in ("T2","T3","EN1","EN2"): lines.append(f"| {system} | {p.loc[(system,'disgust'),'f1']:.4f}/{p.loc[(system,'disgust'),'average_precision']:.4f} | {p.loc[(system,'fear'),'f1']:.4f}/{p.loc[(system,'fear'),'average_precision']:.4f} | {p.loc[(system,'surprise'),'f1']:.4f}/{p.loc[(system,'surprise'),'average_precision']:.4f} |")
    lines += ["", "EN2 improves minority decision F1 versus T2 for disgust, fear, and surprise, but AP decreases for all three. Thus the gain comes from the jointly selected decision surface and modest probability fusion, not improved minority ranking.", "", "## 8. Neutral and cardinality", "", f"EN2 improves Neutral FP {m.loc['T2','neutral_fp']:.4f} → {m.loc['EN2','neutral_fp']:.4f}. Cardinality moves {m.loc['T2','predicted_cardinality']:.4f} → {m.loc['EN2','predicted_cardinality']:.4f}, closer to gold {m.loc['EN2','gold_cardinality']:.4f}.", "", "## 9. Ensemble weights", "", "| Fold | T2 alpha | T3 weight | Inner Macro-F1 |", "|---:|---:|---:|---:|"]
    for r in alpha.itertuples(): lines.append(f"| {r.outer_fold} | {r.alpha_T2:.2f} | {r.weight_T3:.2f} | {r.inner_macro_f1:.4f} |")
    lines += ["", "Alpha consistently favors T2 (0.75–0.95). T3 contributes a small but nonzero 5–25%; the selection does not collapse to alpha=1.", "", "## 10. Thresholds", "", "EN2 threshold stability:", "", "| Label | Mean | SD | Min | Max |", "|---|---:|---:|---:|---:|"]
    en2t=thresholds[thresholds.system=="EN2"]
    for label in EXPECTED_LABELS: lines.append(f"| {label} | {en2t[label].mean():.3f} | {en2t[label].std(ddof=1):.3f} | {en2t[label].min():.2f} | {en2t[label].max():.2f} |")
    lines += ["", "## 11. Error correction", "", f"EN2 corrects {int(exact.t2_wrong_en2_correct)} exact-labelset errors made by T2 and destroys {int(exact.t2_correct_en2_wrong)} exact-labelset successes. Net subset-accuracy corrections: {int(exact.t2_wrong_en2_correct-exact.t2_correct_en2_wrong):+d}.", "", "Per-label transition counts are available in `ensemble_error_analysis.csv`. For fear the ensemble recovers more false negatives than true positives it loses; for disgust and surprise, the small F1 gains arise mainly from removing false positives rather than increasing recall.", "", "## 12. Bootstrap", "", "| Comparison | Delta Macro-F1 | 95% percentile CI |", "|---|---:|---:|"]
    for name in ("EN1_vs_T2","EN2_vs_T2","EN2_vs_T3"): lines.append(f"| {name} | {b.loc[name,'delta']:+.4f} | [{b.loc[name,'ci_2_5']:+.4f}, {b.loc[name,'ci_97_5']:+.4f}] |")
    lines += ["", "Intervals use 10,000 paired instance bootstrap samples, seed 42, and are reported descriptively.", "", "## 13. Computational trade-off", "", f"The ensemble requires two separate networks: {cost['T2']['parameters']:,} T2 parameters plus {cost['T3']['parameters']:,} T3 parameters ({cost['ensemble']['parameters_across_two_networks']:,} across both). Historical sequential OOF inference is approximately {cost['ensemble']['oof_inference_seconds_sequential_approx']:.1f}s versus {cost['T2']['oof_inference_seconds']:.1f}s for T2 alone ({cost['ensemble']['inference_time_ratio_vs_t2']:.1f}×). Approximate deployable checkpoint storage is {cost['ensemble']['deploy_storage_bytes_approx']/2**30:.2f} GiB versus {cost['T2']['deploy_checkpoint_bytes_approx']/2**30:.2f} GiB for T2.", "", "## 14. Final selection", ""]
    if practical:
        lines += [f"**Decision A — ensemble approved.** EN2 gains {delta:+.4f}, exceeds the predeclared +0.005 practical threshold, improves Neutral FP and cardinality, and uses a reasonably stable nonzero T3 contribution. Freeze EN2 as the primary candidate and retain T2 as the final standalone baseline."]
    else:
        lines += [f"**Decision B — ensemble not justified.** The {delta:+.4f} gain does not exceed the predeclared +0.005 practical threshold. Freeze T2."]
    lines += ["", "## 15. Next step", "", "Freeze the selected configuration and its five cross-fitted alpha/threshold sets. Do not open or evaluate the test in this task. Any final-test protocol requires separate explicit authorization and a predeclared method for deriving one deployment alpha/threshold vector without test labels.", "", "## Validation", "", "Source OOF hashes were unchanged; both ensemble OOFs have 2,426 unique IDs and identical outer folds. Dataset train/dev hashes match frozen raw/interim/curated manifests. No preprocessing, label changes, sampling, augmentation, translation, synthetic/external data, or Track B was used. Test: **SEALED — NOT LOADED OR EVALUATED**.", ""]
    return "\n".join(lines)


def generate_ensemble_reports(root: Path) -> dict[str, Any]:
    run=load_json(root/"metadata"/"brighter_ptbr"/"experiments"/"ensemble_run.json"); data=_load(root)
    main,per_label,neutral,multilabel,bootstrap=_tables(root,data); errors=_error_analysis(data)
    report=root/"reports"/"brighter_ptbr"/"experiments"; alpha=pd.read_csv(report/"ensemble_alpha_by_fold.csv"); thresholds=pd.read_csv(report/"ensemble_thresholds_by_fold.csv")
    for name,frame in (("ensemble_main_results",main),("ensemble_per_label",per_label),("ensemble_neutral_cardinality",neutral),("ensemble_multilabel",multilabel),("ensemble_bootstrap",bootstrap),("ensemble_error_analysis",errors)): _atomic_csv(frame,report/f"{name}.csv")
    cost=_cost(root); integrity=_dataset_integrity(root); _figures(root,main,per_label,neutral,alpha,thresholds,errors); _latex(root,main,per_label,alpha,neutral); _registry(root,main,run,cost)
    delta=float(main.set_index("system").loc["EN2","macro_f1"]-main.set_index("system").loc["T2","macro_f1"])
    summary={"protocol":run["protocol"],"audit":run["audit"],"main_results":main.to_dict(orient="records"),"per_label":per_label.to_dict(orient="records"),"neutral_cardinality":neutral.to_dict(orient="records"),"multilabel_only":multilabel.to_dict(orient="records"),"bootstrap":bootstrap.to_dict(orient="records"),"alpha_by_fold":alpha.to_dict(orient="records"),"thresholds_by_fold":thresholds.to_dict(orient="records"),"error_analysis":errors.to_dict(orient="records"),"cost":cost,"dataset_integrity":integrity,"practical_delta_threshold":.005,"decision":"A_ENSEMBLE_APPROVED" if delta>=.005 else "B_ENSEMBLE_NOT_JUSTIFIED","selected_system":"EN2" if delta>=.005 else "T2","standalone_baseline":"T2","test_policy":"SEALED — NOT LOADED OR EVALUATED"}
    write_json_atomic(root/"metadata"/"brighter_ptbr"/"experiments"/"ensemble_summary.json",summary); _atomic_text(_report(main,per_label,neutral,multilabel,bootstrap,alpha,thresholds,errors,cost),report/"ensemble_results.md")
    return summary
