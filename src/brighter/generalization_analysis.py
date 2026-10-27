"""Read-only post-hoc generalization-gap diagnostics from frozen predictions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

# pandas.to_latex imports Matplotlib's Styler path, so the project-local cache
# must be configured before importing pandas or any Matplotlib module.
_PROJECT_MPL_CACHE=Path.cwd()/".cache"/"matplotlib"
_PROJECT_MPL_CACHE.mkdir(parents=True,exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(_PROJECT_MPL_CACHE.resolve())
os.environ["MPLBACKEND"]="Agg"

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support

from .final_reporting import EXPECTED_OUTPUT_HASHES, validate_frozen_inputs, validate_preserved_outputs
from .manifest import load_json, sha256_file, write_json_atomic

LABELS=["anger","disgust","fear","joy","sadness","surprise"]
THRESHOLDS={"EN2":{"anger":.67,"disgust":.36,"fear":.43,"joy":.65,"sadness":.58,"surprise":.49},"T2":{"anger":.66,"disgust":.53,"fear":.52,"joy":.75,"sadness":.65,"surprise":.57}}
REPORT_DIR=Path("reports/brighter_ptbr/final")
FIGURE_DIR=Path("figures/brighter_ptbr/final/generalization_gap")
SUMMARY_PATH=Path("metadata/brighter_ptbr/final/generalization_gap_summary.json")
ALLOWED_PARQUETS={
    "t2_dev":Path("artifacts/predictions/bertweetbr_asl_oof.parquet"),
    "t3_dev":Path("artifacts/predictions/bertimbau_large_bce_oof.parquet"),
    "t2_nested":Path("artifacts/predictions/bertweetbr_asl_thresholded_oof.parquet"),
    "en2_nested":Path("artifacts/predictions/ensemble_t2_t3_tuned_weight_oof.parquet"),
    "t2_final_prob":Path("artifacts/predictions/final/t2_test_probabilities.parquet"),
    "t3_final_prob":Path("artifacts/predictions/final/t3_component_test_probabilities.parquet"),
    "en2_final_prob":Path("artifacts/predictions/final/en2_test_probabilities.parquet"),
    "t2_final_pred":Path("artifacts/predictions/final/t2_test_predictions.parquet"),
    "en2_final_pred":Path("artifacts/predictions/final/en2_test_predictions.parquet"),
}
QUANTILES=[.01,.05,.10,.25,.50,.75,.90,.95,.99]


def _write_text(path:Path,content:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); normalized=content.rstrip()+"\n"
    if path.exists() and path.read_text(encoding="utf-8")==normalized:return
    temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(normalized,encoding="utf-8",newline="\n"); os.replace(temporary,path)


def _write_csv(path:Path,frame:pd.DataFrame)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(temporary,index=False,float_format="%.10f",lineterminator="\n")
    if path.exists() and path.read_bytes()==temporary.read_bytes():temporary.unlink();return
    os.replace(temporary,path)


def _load_allowed(root:Path)->dict[str,pd.DataFrame]:
    frames={name:pd.read_parquet(root/path) for name,path in ALLOWED_PARQUETS.items()}
    for name,frame in frames.items():
        if frame.id.duplicated().any():raise RuntimeError(f"Duplicate IDs in frozen artifact {name}")
        probabilities=frame[[f"{label}_prob" for label in LABELS]].to_numpy(float)
        if probabilities.shape[1]!=6 or not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():raise RuntimeError(f"Invalid frozen probabilities in {name}")
    dev_ids=frames["t2_dev"].id.astype(str).tolist()
    for name in ("t3_dev","t2_nested","en2_nested"):
        if frames[name].id.astype(str).tolist()!=dev_ids:raise RuntimeError(f"Development ID/order mismatch: {name}")
    final_ids=frames["t2_final_prob"].id.astype(str).tolist()
    for name in ("t3_final_prob","en2_final_prob","t2_final_pred","en2_final_pred"):
        if frames[name].id.astype(str).tolist()!=final_ids:raise RuntimeError(f"Final output ID/order mismatch: {name}")
    if len(dev_ids)!=2426 or len(final_ids)!=2226:raise RuntimeError("Frozen artifact row-count mismatch")
    return frames


def _matrix(frame:pd.DataFrame,suffix:str)->np.ndarray:
    return frame[[f"{label}_{suffix}" for label in LABELS]].to_numpy(float if suffix=="prob" else int)


def _ece(y:np.ndarray,p:np.ndarray,bins:int=10)->float:
    edges=np.linspace(0,1,bins+1); total=len(y); value=0.0
    for index in range(bins):
        mask=(p>=edges[index]) & ((p<edges[index+1]) if index<bins-1 else (p<=edges[index+1]))
        if mask.any():value+=mask.mean()*abs(float(y[mask].mean())-float(p[mask].mean()))
    return float(value)


def _stats(system:str,split:str,label:str,values:np.ndarray)->dict[str,Any]:
    row={"system":system,"split":split,"label":label,"n":len(values),"mean":float(values.mean()),"std":float(values.std(ddof=0)),"min":float(values.min()),"max":float(values.max())}
    for q,value in zip(QUANTILES,np.quantile(values,QUANTILES),strict=True):row[f"p{int(q*100):02d}"]=float(value)
    return row


def _performance_rows(system:str,dev_frame:pd.DataFrame,final_result:Mapping[str,Any],final_key:str)->list[dict[str,Any]]:
    y=_matrix(dev_frame,"true"); pred=_matrix(dev_frame,"pred"); precision,recall,f1,support=precision_recall_fscore_support(y,pred,average=None,zero_division=0)
    final=final_result[final_key]["metrics"]["per_label"]; rows=[]
    for i,label in enumerate(LABELS):
        rows.append({"system":system,"label":label,"precision_dev":float(precision[i]),"precision_test":float(final[label]["precision"]),"delta_precision":float(final[label]["precision"]-precision[i]),"recall_dev":float(recall[i]),"recall_test":float(final[label]["recall"]),"delta_recall":float(final[label]["recall"]-recall[i]),"f1_dev":float(f1[i]),"f1_test":float(final[label]["f1"]),"delta_f1":float(final[label]["f1"]-f1[i]),"support_dev":int(support[i]),"support_test":int(final[label]["support"])})
    return rows


def compute_analysis(root:Path)->dict[str,Any]:
    state=load_json(root/"metadata/brighter_ptbr/final/experiment_state.json")
    if state.get("state")!="TEST_EVALUATED" or state.get("test_accessed") is not True or state.get("further_tuning_allowed") is not False:raise RuntimeError("Generalization analysis requires immutable TEST_EVALUATED state")
    output_hashes=validate_preserved_outputs(root); integrity=validate_frozen_inputs(root); result=load_json(root/REPORT_DIR/"final_test_results.json"); result_hash=sha256_file(root/REPORT_DIR/"final_test_results.json")
    if result["primary"]["metrics"]["macro_f1"]!=0.2851799354682137 or result["secondary"]["metrics"]["macro_f1"]!=0.15691169825594717:raise RuntimeError("Frozen final metrics changed")
    frames=_load_allowed(root)
    y_dev=_matrix(frames["t2_dev"],"true"); y_test=_matrix(frames["en2_final_pred"],"true")
    for name in ("t3_dev","t2_nested","en2_nested"):
        if not np.array_equal(_matrix(frames[name],"true"),y_dev):raise RuntimeError(f"Development gold mismatch: {name}")
    if not np.array_equal(_matrix(frames["t2_final_pred"],"true"),y_test):raise RuntimeError("Final frozen gold mismatch")
    probabilities={
        "T2":{"development":_matrix(frames["t2_dev"],"prob"),"test":_matrix(frames["t2_final_prob"],"prob")},
        "T3":{"development":_matrix(frames["t3_dev"],"prob"),"test":_matrix(frames["t3_final_prob"],"prob")},
    }
    probabilities["EN2"]={"development":.70*probabilities["T2"]["development"]+.30*probabilities["T3"]["development"],"test":_matrix(frames["en2_final_prob"],"prob")}
    if not np.allclose(probabilities["EN2"]["test"],.70*probabilities["T2"]["test"]+.30*probabilities["T3"]["test"],atol=1e-12):raise RuntimeError("Frozen EN2 arithmetic mismatch")

    prevalence=[]
    nested_en2_pred=_matrix(frames["en2_nested"],"pred"); nested_t2_pred=_matrix(frames["t2_nested"],"pred"); final_en2_pred=_matrix(frames["en2_final_pred"],"pred"); final_t2_pred=_matrix(frames["t2_final_pred"],"pred")
    for i,label in enumerate(LABELS):
        dev_gold=float(y_dev[:,i].mean()); test_gold=float(y_test[:,i].mean()); dev_pred=float(nested_en2_pred[:,i].mean()); test_pred=float(final_en2_pred[:,i].mean())
        prevalence.append({"label":label,"gold_dev":dev_gold,"gold_test":test_gold,"gold_absolute_delta":test_gold-dev_gold,"gold_relative_delta":(test_gold-dev_gold)/dev_gold,"pred_dev_en2_nested":dev_pred,"pred_test_en2":test_pred,"prediction_error_dev":dev_pred-dev_gold,"prediction_error_test":test_pred-test_gold})
    prevalence_df=pd.DataFrame(prevalence)
    cardinality=pd.DataFrame([
        {"system":"EN2","development":float(nested_en2_pred.sum(1).mean()),"test":float(final_en2_pred.sum(1).mean())},
        {"system":"T2","development":float(nested_t2_pred.sum(1).mean()),"test":float(final_t2_pred.sum(1).mean())},
    ]); cardinality["absolute_delta"]=cardinality.test-cardinality.development; cardinality["relative_delta"]=cardinality.absolute_delta/cardinality.development; cardinality["test_dev_ratio"]=cardinality.test/cardinality.development
    gold_card={"development":float(y_dev.sum(1).mean()),"test":float(y_test.sum(1).mean())}; gold_card["absolute_delta"]=gold_card["test"]-gold_card["development"]

    probability_rows=[];distance_rows=[];score_class_rows=[]
    for system in ("T2","T3","EN2"):
        for i,label in enumerate(LABELS):
            dev=probabilities[system]["development"][:,i]; test=probabilities[system]["test"][:,i]
            probability_rows += [_stats(system,"development",label,dev),_stats(system,"test",label,test)]
            hist_dev,_=np.histogram(dev,bins=20,range=(0,1),density=False); hist_test,_=np.histogram(test,bins=20,range=(0,1),density=False); a=(hist_dev+1e-12)/(hist_dev.sum()+20e-12); b=(hist_test+1e-12)/(hist_test.sum()+20e-12)
            distance_rows.append({"system":system,"label":label,"mean_delta":float(test.mean()-dev.mean()),"median_delta":float(np.median(test)-np.median(dev)),"wasserstein":float(wasserstein_distance(dev,test)),"jensen_shannon":float(jensenshannon(a,b,base=2.0)**2)})
            for split,values,gold in (("development",dev,y_dev[:,i]),("test",test,y_test[:,i])):
                positive=values[gold==1];negative=values[gold==0];score_class_rows.append({"system":system,"split":split,"label":label,"positive_n":len(positive),"mean_positive_score":float(positive.mean()),"median_positive_score":float(np.median(positive)),"negative_n":len(negative),"mean_negative_score":float(negative.mean()),"median_negative_score":float(np.median(negative))})
    probability_df=pd.DataFrame(probability_rows); distances=pd.DataFrame(distance_rows); score_by_class=pd.DataFrame(score_class_rows)

    crossings=[]
    for system in ("T2","EN2"):
        for i,label in enumerate(LABELS):
            threshold=THRESHOLDS[system][label]; dev=probabilities[system]["development"][:,i];test=probabilities[system]["test"][:,i];dev_rate=float((dev>=threshold).mean());test_rate=float((test>=threshold).mean())
            crossings.append({"system":system,"label":label,"threshold":threshold,"development_crossing_rate":dev_rate,"test_crossing_rate":test_rate,"delta":test_rate-dev_rate,"development_mean_minus_threshold":float(dev.mean()-threshold),"test_mean_minus_threshold":float(test.mean()-threshold),"development_median_minus_threshold":float(np.median(dev)-threshold),"test_median_minus_threshold":float(np.median(test)-threshold)})
    crossings_df=pd.DataFrame(crossings)

    per_label=pd.DataFrame(_performance_rows("EN2",frames["en2_nested"],result,"primary")+_performance_rows("T2",frames["t2_nested"],result,"secondary"))
    calibration=[]
    for system,dev_prob,test_prob,dev_prob_for_ap in (("T2",probabilities["T2"]["development"],probabilities["T2"]["test"],probabilities["T2"]["development"]),("EN2",probabilities["EN2"]["development"],probabilities["EN2"]["test"],_matrix(frames["en2_nested"],"prob"))):
        # OOF AP for EN2 remains tied to the nested ensemble artifact.
        for i,label in enumerate(LABELS):
            for split,gold,prob in (("development",y_dev,dev_prob_for_ap),("test",y_test,test_prob)):
                calibration.append({"system":system,"split":split,"label":label,"average_precision":float(average_precision_score(gold[:,i],prob[:,i])),"brier_score":float(brier_score_loss(gold[:,i],prob[:,i])),"ece_10_bin":_ece(gold[:,i],prob[:,i],10)})
    calibration_df=pd.DataFrame(calibration)
    macro=calibration_df.groupby(["system","split"],as_index=False)[["average_precision","brier_score","ece_10_bin"]].mean();macro["label"]="MACRO";calibration_df=pd.concat([calibration_df,macro[["system","split","label","average_precision","brier_score","ece_10_bin"]]],ignore_index=True)

    main=pd.DataFrame([{"system":"EN2","nested_development_macro_f1":.5691254630769899,"final_test_macro_f1":result["primary"]["metrics"]["macro_f1"],"difference":result["primary"]["metrics"]["macro_f1"]-.5691254630769899},{"system":"T2","nested_development_macro_f1":.5585138660403824,"final_test_macro_f1":result["secondary"]["metrics"]["macro_f1"],"difference":result["secondary"]["metrics"]["macro_f1"]-.5585138660403824}])
    return {"state":state,"output_hashes":output_hashes,"input_integrity":integrity,"result_hash":result_hash,"result":result,"frames":frames,"probabilities":probabilities,"y_dev":y_dev,"y_test":y_test,"main":main,"prevalence":prevalence_df,"cardinality":cardinality,"gold_cardinality":gold_card,"probability_summary":probability_df,"distances":distances,"score_by_gold_class":score_by_class,"crossings":crossings_df,"per_label":per_label,"calibration":calibration_df}


def _hypotheses(data:Mapping[str,Any])->pd.DataFrame:
    prevalence=data["prevalence"]; distances=data["distances"]; calibration=data["calibration"]
    max_label_delta=float(prevalence.gold_absolute_delta.abs().max())
    mean_shift=distances.groupby("system").mean_delta.mean().to_dict()
    macro=calibration[calibration.label.eq("MACRO")].set_index(["system","split"])
    ap_en2_dev=float(macro.loc[("EN2","development"),"average_precision"]);ap_en2_test=float(macro.loc[("EN2","test"),"average_precision"])
    brier_en2_dev=float(macro.loc[("EN2","development"),"brier_score"]);brier_en2_test=float(macro.loc[("EN2","test"),"brier_score"])
    return pd.DataFrame([
        {"hypothesis":"H1 label prevalence shift","evidence_supporting":f"Largest absolute per-label prevalence change was {max_label_delta:.4f}.","evidence_against":f"Gold cardinality changed only {data['gold_cardinality']['absolute_delta']:+.6f}; prevalence changes are modest relative to the performance gap.","status":"NOT SUPPORTED"},
        {"hypothesis":"H2 probability calibration shift","evidence_supporting":f"Mean score shifts averaged {mean_shift['EN2']:+.4f} for EN2 and predicted cardinality fell {float(data['cardinality'].set_index('system').loc['EN2','absolute_delta']):+.4f}; crossing rates collapsed for most labels.","evidence_against":f"EN2 macro Brier did not deteriorate ({brier_en2_dev:.4f}→{brier_en2_test:.4f}), and score movement conflates retraining with test-distribution effects.","status":"PARTIALLY SUPPORTED"},
        {"hypothesis":"H3 ranking degradation","evidence_supporting":f"EN2 macro AP changed {ap_en2_dev:.4f}→{ap_en2_test:.4f}, so threshold-free ranking also degraded.","evidence_against":"Some labels, especially surprise and joy, retained materially better ranking/recall than the rare conservative labels.","status":"PARTIALLY SUPPORTED"},
        {"hypothesis":"H4 final-model retraining shift","evidence_supporting":f"Frozen final models moved probability scales relative to fold OOF: mean shifts T2={mean_shift['T2']:+.4f}, T3={mean_shift['T3']:+.4f}.","evidence_against":"No matched inference of fold and final models on the same held-out texts exists, so retraining cannot be isolated from test shift.","status":"PARTIALLY SUPPORTED"},
        {"hypothesis":"H5 T2-specific failure","evidence_supporting":f"T2 had the largest standalone test gap and mean score shift {mean_shift['T2']:+.4f}, versus only {mean_shift['T3']:+.4f} for T3.","evidence_against":"EN2 also remained conservative and some T2 labels shifted upward, so the failure was heterogeneous rather than uniformly T2-wide.","status":"PARTIALLY SUPPORTED"},
        {"hypothesis":"H6 ensemble failure","evidence_supporting":"The frozen fusion inherited low component scores and remained far below nested development.","evidence_against":f"EN2 test Macro-F1 {data['result']['primary']['metrics']['macro_f1']:.4f} exceeded T2 {data['result']['secondary']['metrics']['macro_f1']:.4f}; T3 therefore mitigated rather than solely caused T2 conservatism.","status":"NOT SUPPORTED"},
        {"hypothesis":"H7 textual/domain shift","evidence_supporting":"Development-to-test score and ranking changes are compatible with covariate/domain differences.","evidence_against":"No permitted frozen text-level representation supports direct covariate analysis in this task.","status":"INCONCLUSIVE"},
    ])


def _latex(path:Path,frame:pd.DataFrame,columns:list[str],headers:list[str],formats:Mapping[str,str]|None=None)->None:
    formats=formats or {}; view=frame[columns].copy()
    for column,fmt in formats.items():view[column]=view[column].map(lambda value:fmt.format(value))
    content=view.to_latex(index=False,escape=False,header=headers,column_format="l"+"r"*(len(columns)-1),caption=None,label=None)
    _write_text(path,content)


def _save_figure(fig:Any,directory:Path,name:str)->None:
    png,pdf=directory/f"{name}.png",directory/f"{name}.pdf"
    if png.exists() and pdf.exists() and png.stat().st_size and pdf.stat().st_size:return
    fig.tight_layout();fig.savefig(png,dpi=180,bbox_inches="tight",metadata={"Software":"BRIGHTER PT-BR post-hoc analysis"});fig.savefig(pdf,bbox_inches="tight",metadata={"Creator":"BRIGHTER PT-BR post-hoc analysis","CreationDate":None,"ModDate":None})


def generate_figures(root:Path,data:Mapping[str,Any])->list[Path]:
    cache=root/".cache/matplotlib";cache.mkdir(parents=True,exist_ok=True);os.environ["MPLCONFIGDIR"]=str(cache.resolve());os.environ["MPLBACKEND"]="Agg"
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    out=root/FIGURE_DIR;out.mkdir(parents=True,exist_ok=True);labels=LABELS;x=np.arange(6);width=.36
    main=data["main"].set_index("system").loc[["T2","EN2"]];fig,ax=plt.subplots(figsize=(6,4));positions=np.arange(2);ax.bar(positions-width/2,main.nested_development_macro_f1,width,label="Nested development");ax.bar(positions+width/2,main.final_test_macro_f1,width,label="Final test");ax.set_xticks(positions,["T2","EN2"]);ax.set_ylabel("Macro-F1");ax.set_title("Development vs final test");ax.legend();_save_figure(fig,out,"01_macro_f1_dev_vs_test");plt.close(fig)
    card=data["cardinality"].set_index("system").loc[["T2","EN2"]];fig,ax=plt.subplots(figsize=(6,4));ax.bar(positions-width/2,card.development,width,label="Nested development");ax.bar(positions+width/2,card.test,width,label="Final test");ax.axhline(data["gold_cardinality"]["test"],color="black",linestyle="--",label="Gold test");ax.set_xticks(positions,["T2","EN2"]);ax.set_ylabel("Predicted label cardinality");ax.set_title("Predicted cardinality shift");ax.legend();_save_figure(fig,out,"02_predicted_cardinality_dev_vs_test");plt.close(fig)
    prevalence=data["prevalence"]
    for split in ("development","test"):
        gold=prevalence["gold_dev" if split=="development" else "gold_test"];pred=prevalence["pred_dev_en2_nested" if split=="development" else "pred_test_en2"]
        fig,ax=plt.subplots(figsize=(9,4));ax.bar(x-width/2,gold,width,label="Gold");ax.bar(x+width/2,pred,width,label="EN2 predicted");ax.set_xticks(x,labels);ax.set_ylabel("Prevalence");ax.set_title(f"Gold vs predicted prevalence — {split}");ax.legend();_save_figure(fig,out,"03_prevalence_development" if split=="development" else "04_prevalence_test");plt.close(fig)
    fig,axes=plt.subplots(3,6,figsize=(18,9),sharex=True,sharey=True)
    for row,system in enumerate(("T2","T3","EN2")):
        for column,label in enumerate(labels):
            ax=axes[row,column]
            for split,style in (("development","-"),("test","--")):
                values=np.sort(data["probabilities"][system][split][:,column]);ecdf=np.arange(1,len(values)+1)/len(values);ax.plot(values,ecdf,style,label=split if row==0 and column==0 else None)
            ax.axvline(THRESHOLDS[system][label],color="black",alpha=.35,linewidth=.8) if system in THRESHOLDS else None
            if row==0:ax.set_title(label)
            if column==0:ax.set_ylabel(f"{system}\nECDF")
            if row==2:ax.set_xlabel("Probability")
    axes[0,0].legend(loc="lower right");fig.suptitle("Frozen probability distributions: development OOF vs final test",y=1.01);_save_figure(fig,out,"05_probability_ecdf_dev_vs_test");plt.close(fig)
    crossing=data["crossings"];fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,system in zip(axes,("T2","EN2"),strict=True):
        part=crossing[crossing.system.eq(system)].set_index("label").loc[labels];ax.bar(x-width/2,part.development_crossing_rate,width,label="Development OOF");ax.bar(x+width/2,part.test_crossing_rate,width,label="Final test");ax.set_xticks(x,labels,rotation=35,ha="right");ax.set_title(system);ax.set_ylabel("Rate above frozen threshold");ax.legend()
    _save_figure(fig,out,"06_threshold_crossing_dev_vs_test");plt.close(fig)
    primary=data["per_label"];primary=primary[primary.system.eq("EN2")].set_index("label").loc[labels]
    for metric,index in (("F1",7),("Recall",8)):
        low=metric.lower();fig,ax=plt.subplots(figsize=(9,4));ax.bar(x-width/2,primary[f"{low}_dev"],width,label="Nested development");ax.bar(x+width/2,primary[f"{low}_test"],width,label="Final test");ax.set_xticks(x,labels);ax.set_ylabel(metric);ax.set_title(f"EN2 per-label {metric}: development vs final test");ax.legend();_save_figure(fig,out,f"{index:02d}_{low}_dev_vs_test");plt.close(fig)
    calibration=data["calibration"];macro=calibration[calibration.label.eq("MACRO")];fig,ax=plt.subplots(figsize=(6,4));pivot=macro.pivot(index="system",columns="split",values="brier_score").loc[["T2","EN2"]];ax.bar(positions-width/2,pivot.development,width,label="Development OOF");ax.bar(positions+width/2,pivot.test,width,label="Final test");ax.set_xticks(positions,["T2","EN2"]);ax.set_ylabel("Macro Brier score (lower is better)");ax.set_title("Post-hoc calibration diagnostic");ax.legend();_save_figure(fig,out,"09_brier_dev_vs_test");plt.close(fig)
    return sorted(out.glob("*.*"))


def write_tables(root:Path,data:Mapping[str,Any])->dict[str,Path]:
    report=root/REPORT_DIR;latex=report/"latex";latex.mkdir(parents=True,exist_ok=True)
    tables={"generalization_main":data["main"],"generalization_per_label":data["per_label"],"generalization_cardinality":data["cardinality"],"generalization_prevalence":data["prevalence"],"generalization_probability_shift":data["distances"],"generalization_threshold_crossing":data["crossings"],"generalization_calibration":data["calibration"],"generalization_probability_summary":data["probability_summary"],"generalization_score_by_gold_class":data["score_by_gold_class"]}
    paths={}
    for name,frame in tables.items():path=report/f"{name}.csv";_write_csv(path,frame);paths[name]=path
    fmt={column:"{:.4f}" for column in data["main"].columns if column!="system"};_latex(latex/"generalization_main.tex",data["main"],list(data["main"].columns),["System","Nested dev Macro-F1","Final test Macro-F1","Difference"],fmt)
    columns=["system","label","precision_dev","precision_test","recall_dev","recall_test","f1_dev","f1_test"];
    _latex(latex/"generalization_per_label.tex",data["per_label"],columns,["System","Label","P dev","P test","R dev","R test","F1 dev","F1 test"],{c:"{:.4f}" for c in columns[2:]})
    _latex(latex/"generalization_cardinality.tex",data["cardinality"],list(data["cardinality"].columns),["System","Development","Test","Delta","Relative delta","Test/dev"],{c:"{:.4f}" for c in data["cardinality"].columns if c!="system"})
    pcols=["label","gold_dev","gold_test","gold_absolute_delta","pred_dev_en2_nested","pred_test_en2","prediction_error_dev","prediction_error_test"];
    _latex(latex/"generalization_prevalence.tex",data["prevalence"],pcols,["Label","Gold dev","Gold test","Gold delta","Pred dev","Pred test","Error dev","Error test"],{c:"{:.4f}" for c in pcols[1:]})
    _latex(latex/"generalization_probability_shift.tex",data["distances"],list(data["distances"].columns),["System","Label","Mean delta","Median delta","Wasserstein","JS divergence"],{c:"{:.4f}" for c in data["distances"].columns[2:]})
    ccols=["system","label","threshold","development_crossing_rate","test_crossing_rate","delta"];
    _latex(latex/"generalization_threshold_crossing.tex",data["crossings"],ccols,["System","Label","Threshold","Dev crossing","Test crossing","Delta"],{c:"{:.4f}" for c in ccols[2:]})
    _latex(latex/"generalization_calibration.tex",data["calibration"],list(data["calibration"].columns),["System","Split","Label","AP","Brier","ECE-10"],{c:"{:.4f}" for c in data["calibration"].columns[3:]})
    return paths


def write_report(root:Path,data:Mapping[str,Any],hypotheses:pd.DataFrame)->Path:
    prevalence=data["prevalence"];card=data["cardinality"].set_index("system");dist=data["distances"];cross=data["crossings"];per=data["per_label"];cal=data["calibration"];score=data["score_by_gold_class"]
    macro=cal[cal.label.eq("MACRO")].set_index(["system","split"]);en2_per=per[per.system.eq("EN2")].set_index("label")
    mean_shifts=dist.groupby("system").mean_delta.mean();largest_t2=dist[dist.system.eq("T2")].sort_values("mean_delta").iloc[0];largest_t3=dist[dist.system.eq("T3")].sort_values("mean_delta").iloc[0]
    lines=["# BRIGHTER PT-BR — post-hoc generalization gap analysis","","## 1. Objective","","Investigate the frozen development-to-test gap without changing the predictive experiment. This analysis uses only preserved OOF/final probability, prediction, metric, calibration, and ensemble artifacts. It performs no training, model loading, official-test loading, inference, threshold search, or parameter modification.","","## 2. Frozen Experiment Context","","Primary EN2 remains frozen at nested development Macro-F1 0.5691 and final test Macro-F1 0.2852 (difference -0.2839). Secondary T2 remains 0.5585 and 0.1569 (difference -0.4016). EN2 probabilities are fixed as 0.70×T2 + 0.30×T3. For probability-scale transfer diagnostics only, the same frozen fusion is applied to preserved development OOF probabilities; the unbiased performance estimate remains the nested cross-fitted result.","","## 3. Label Distribution","","| Label | Gold dev | Gold test | Absolute delta | Relative delta |","|---|---:|---:|---:|---:|"]
    for _,row in prevalence.iterrows():lines.append(f"| {row.label} | {row.gold_dev:.4f} | {row.gold_test:.4f} | {row.gold_absolute_delta:+.4f} | {row.gold_relative_delta:+.2%} |")
    lines += ["",f"The largest absolute per-label prevalence change was {prevalence.gold_absolute_delta.abs().max():.4f}. These changes are too small, descriptively, to match the magnitude of the Macro-F1 gaps; H1 is therefore not supported by label prevalence alone.","","## 4. Cardinality Shift","",f"Gold cardinality was {data['gold_cardinality']['development']:.6f} in development and {data['gold_cardinality']['test']:.6f} in test (delta {data['gold_cardinality']['absolute_delta']:+.6f}).","","| System | Predicted dev | Predicted test | Delta | Test/dev |","|---|---:|---:|---:|---:|",f"| EN2 | {card.loc['EN2','development']:.4f} | {card.loc['EN2','test']:.4f} | {card.loc['EN2','absolute_delta']:+.4f} | {card.loc['EN2','test_dev_ratio']:.3f} |",f"| T2 | {card.loc['T2','development']:.4f} | {card.loc['T2','test']:.4f} | {card.loc['T2','absolute_delta']:+.4f} | {card.loc['T2','test_dev_ratio']:.3f} |","",f"EN2 emitted only {card.loc['EN2','test_dev_ratio']:.1%} of its nested development label cardinality; T2 emitted {card.loc['T2','test_dev_ratio']:.1%}. This decision-rate collapse occurred while gold cardinality was almost unchanged.","","## 5. Probability Distribution Shift","",f"Across labels, mean probability shifts averaged T2={mean_shifts['T2']:+.4f}, T3={mean_shifts['T3']:+.4f}, and EN2={mean_shifts['EN2']:+.4f}. The largest negative mean shifts were T2/{largest_t2.label}={largest_t2.mean_delta:+.4f} and T3/{largest_t3.label}={largest_t3.mean_delta:+.4f}.","","Full quantiles (min, p01, p05, p10, p25, median, p75, p90, p95, p99, max), Wasserstein distances, and fixed-histogram Jensen–Shannon divergences are in the machine-readable probability tables.","","## 6. Threshold Crossing Analysis","","| System | Label | Threshold | Dev crossing | Test crossing | Delta |","|---|---|---:|---:|---:|---:|"]
    for _,row in cross.iterrows():lines.append(f"| {row.system} | {row.label} | {row.threshold:.2f} | {row.development_crossing_rate:.4f} | {row.test_crossing_rate:.4f} | {row.delta:+.4f} |")
    lines += ["","Frozen-threshold crossing rates fell for most labels, directly explaining the lower predicted cardinality. This is a transfer diagnostic only: no alternative threshold or oracle test F1 was computed.","","## 7. Per-label Generalization","","| Label | P dev | P test | R dev | R test | F1 dev | F1 test |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,row in en2_per.iterrows():lines.append(f"| {label} | {row.precision_dev:.4f} | {row.precision_test:.4f} | {row.recall_dev:.4f} | {row.recall_test:.4f} | {row.f1_dev:.4f} | {row.f1_test:.4f} |")
    lines += ["",f"Final EN2 macro precision remained {data['result']['primary']['metrics']['macro_precision']:.4f}, whereas macro recall fell to {data['result']['primary']['metrics']['macro_recall']:.4f}; the dominant decision-level pattern is conservatism rather than indiscriminate false positives.","","## 8. Minority Classes","",f"Disgust recall changed {en2_per.loc['disgust','recall_dev']:.4f}→{en2_per.loc['disgust','recall_test']:.4f}, fear {en2_per.loc['fear','recall_dev']:.4f}→{en2_per.loc['fear','recall_test']:.4f}, and surprise {en2_per.loc['surprise','recall_dev']:.4f}→{en2_per.loc['surprise','recall_test']:.4f}. Sadness also collapsed ({en2_per.loc['sadness','recall_dev']:.4f}→{en2_per.loc['sadness','recall_test']:.4f}), so the gap is not confined to the three pre-designated minority labels.","","## 9. T2 Probability Behavior","",f"T2 predicted cardinality changed {card.loc['T2','development']:.4f}→{card.loc['T2','test']:.4f}. Its most negative mean score shift was {largest_t2.label} ({largest_t2.mean_delta:+.4f}). Surprise behaved differently at the decision level: final recall remained {data['result']['secondary']['metrics']['per_label']['surprise']['recall']:.4f}, while sadness recall was {data['result']['secondary']['metrics']['per_label']['sadness']['recall']:.4f}. Their distinct score locations relative to frozen thresholds, not a changed decision rule, account descriptively for that contrast.","","## 10. T3 Component Probability Behavior","",f"T3 mean scores also shifted ({mean_shifts['T3']:+.4f} averaged across labels), with the largest negative label shift in {largest_t3.label} ({largest_t3.mean_delta:+.4f}). This argues against a purely T2-specific probability-scale failure. No T3 standalone final Macro-F1 was calculated or reported.","","## 11. EN2 Ensemble Behavior","",f"EN2 final Macro-F1 ({data['result']['primary']['metrics']['macro_f1']:.4f}) exceeded frozen T2 ({data['result']['secondary']['metrics']['macro_f1']:.4f}), and its neutral FP was lower ({data['result']['primary']['metrics']['zero_label']['neutral_false_positive_rate']:.4f} vs {data['result']['secondary']['metrics']['zero_label']['neutral_false_positive_rate']:.4f}). Thus T3 mitigated some T2 failure, but the fixed fusion could not prevent the shared downward score/crossing shift.","","## 12. Threshold-Free Ranking Analysis","","**POST-HOC DIAGNOSTIC METRIC — NOT USED FOR MODEL SELECTION.**","","| System | Macro AP dev | Macro AP test | Delta |","|---|---:|---:|---:|"]
    for system in ("T2","EN2"):
        dev=float(macro.loc[(system,"development"),"average_precision"]);test=float(macro.loc[(system,"test"),"average_precision"]);lines.append(f"| {system} | {dev:.4f} | {test:.4f} | {test-dev:+.4f} |")
    lines += ["","AP changed as well as thresholded F1, indicating that threshold transfer is not the sole issue. The relative size varies by label; the calibration CSV contains every per-label value.","","## 13. Calibration Diagnostics","","**POST-HOC DIAGNOSTIC METRICS — NOT USED FOR MODEL SELECTION.**","","| System | Brier dev | Brier test | ECE-10 dev | ECE-10 test |","|---|---:|---:|---:|---:|"]
    for system in ("T2","EN2"):lines.append(f"| {system} | {macro.loc[(system,'development'),'brier_score']:.4f} | {macro.loc[(system,'test'),'brier_score']:.4f} | {macro.loc[(system,'development'),'ece_10_bin']:.4f} | {macro.loc[(system,'test'),'ece_10_bin']:.4f} |")
    lines += ["","ECE uses ten fixed equal-width bins. Brier has priority; ECE is included only as a simple descriptive supplement. Score-by-gold-class tables separate positive and negative score movement.","","## 14. Potential Explanations","","The evidence is compatible with multiple non-exclusive mechanisms: modest label-prevalence differences, downward probability-scale transfer, ranking degradation, final-model retraining variance, rare-label instability, and covariate/domain shift. The present artifacts do not identify a unique cause.","","## 15. Evidence For/Against Each Hypothesis","","| Hypothesis | Evidence supporting | Evidence against | Status |","|---|---|---|---|"]
    for _,row in hypotheses.iterrows():lines.append(f"| {row.hypothesis} | {row.evidence_supporting} | {row.evidence_against} | **{row.status}** |")
    lines += ["","## 16. Implications for the Article","","The article should report the robust nested estimate (0.5691) and the frozen final result (0.2852) without hiding the gap. The post-hoc evidence makes threshold transfer between cross-validation model distributions and single full-development retrained models a central methodological risk in small imbalanced multilabel tasks.","","Historical context must remain separate: an earlier experiment used a previously balanced/augmented dataset, BERTimbau Large, Focal Loss, demojize, thresholds, and a Safety Net, with leaderboard Macro-F1 0.55746. The current experiment used original Track A, no augmentation/preprocessing/Safety Net, nested selection, ASL/BERTweetBR and a frozen ensemble. The values 0.55746 and 0.2852 are not directly comparable because protocols, data, and possibly evaluation endpoints differ.","","## 17. Limitations","","- The dataset and rare-label supports are small; original training had only 75 disgust positives.","- Final models used one frozen seed.","- OOF distributions aggregate fold-specific models, while deployment uses one full-development model per component.","- Test text was not reopened, so H7 textual/domain shift could not be directly measured.","- Probability distances are descriptive, not causal tests.","- ECE depends on fixed binning and is secondary to Brier.","- One train–test overlap known pre-test remains in the official protocol.","","## 18. Conclusions","",f"The clearest evidence is a heterogeneous probability-scale and frozen-threshold transfer failure after final retraining/application, accompanied by a predicted-cardinality collapse despite nearly unchanged gold cardinality. Threshold-free AP also declined, so the gap cannot be reduced to thresholds alone. T2 shifted much more than T3 on average, making a T2-dominant component effect partially supported; T3 was comparatively stable and its contribution mitigated, but did not eliminate, the final conservatism. Retraining-induced calibration shift cannot be isolated from test/domain shift with the frozen artifacts, so causal attribution remains appropriately limited."]
    path=root/REPORT_DIR/"generalization_gap_analysis.md";_write_text(path,"\n".join(lines));return path


def run_analysis(root:Path)->dict[str,Any]:
    before=validate_preserved_outputs(root);state_before=load_json(root/"metadata/brighter_ptbr/final/experiment_state.json");data=compute_analysis(root);hypotheses=_hypotheses(data);tables=write_tables(root,data);_write_csv(root/REPORT_DIR/"generalization_hypotheses.csv",hypotheses);_latex(root/REPORT_DIR/"latex/generalization_hypotheses.tex",hypotheses,list(hypotheses.columns),["Hypothesis","Evidence supporting","Evidence against","Status"]);figures=generate_figures(root,data);report=write_report(root,data,hypotheses)
    after=validate_preserved_outputs(root);state_after=load_json(root/"metadata/brighter_ptbr/final/experiment_state.json")
    if before!=after:raise RuntimeError("Frozen probability/prediction artifact changed during analysis")
    if state_before!=state_after or state_after.get("state")!="TEST_EVALUATED":raise RuntimeError("Frozen experimental state changed during analysis")
    summary={"analysis":"POST_HOC_GENERALIZATION_GAP","read_only":True,"models_loaded":False,"raw_test_loaded":False,"new_inference":False,"new_training":False,"threshold_tuning":False,"parameters_modified":False,"search_space":"CLOSED","frozen_output_hashes":after,"final_metrics_sha256":data["result_hash"],"gold_cardinality":data["gold_cardinality"],"main_results":data["main"].to_dict("records"),"predicted_cardinality":data["cardinality"].to_dict("records"),"macro_calibration":data["calibration"][data["calibration"].label.eq("MACRO")].to_dict("records"),"hypotheses":hypotheses.to_dict("records"),"report":report.relative_to(root).as_posix(),"tables":{name:path.relative_to(root).as_posix() for name,path in tables.items()},"figures":[path.relative_to(root).as_posix() for path in figures]}
    write_json_atomic(root/SUMMARY_PATH,summary);return {"summary":summary,"data":data,"hypotheses":hypotheses}
