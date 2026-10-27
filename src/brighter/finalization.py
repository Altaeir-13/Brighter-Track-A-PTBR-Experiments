"""Pre-test deployment calibration and final full-development training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .curation import FROZEN_REVISION
from .ensemble import fuse_probabilities, select_alpha
from .experiments import (
    DEV_POOL_SIZE, EXPECTED_LABELS, _MultilabelCollator, _TextDataset,
    _resolve_model_revision, construct_development_pool, development_input_hashes,
    seed_everything,
)
from .losses import build_loss, calculate_fold_class_weights
from .manifest import load_json, sha256_file, utc_now_iso, write_json_atomic
from .threshold_calibration import search_per_label_thresholds
from .threshold_reporting import _dataset_integrity

FINAL_ROOT = Path("metadata/brighter_ptbr/final")
REPORT_ROOT = Path("reports/brighter_ptbr/final")
LABELS = list(EXPECTED_LABELS)
WARNING = "DEPLOYMENT CALIBRATION ONLY — HYPERPARAMETERS SELECTED ON THE SAME OOF SET. NOT AN UNBIASED PERFORMANCE ESTIMATE."
NESTED_PERFORMANCE = 0.5691254630769899
FINAL_SPECS = {
    "T2": {"source": "bertweetbr_asl", "directory": "t2_bertweetbr_asl", "registry": "FINAL_T2_BERTWEETBR_ASL"},
    "T3": {"source": "bertimbau_large_bce", "directory": "t3_bertimbau_large_bce", "registry": "FINAL_T3_BERTIMBAU_LARGE_BCE"},
}


def _id_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _json_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text(path, yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True))


def _software_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn", "pyarrow", "huggingface-hub"):
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name] = "not-installed"
    return result


def _gpu_metadata() -> dict[str, Any]:
    import torch
    return {"cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "ram_gb": 32, "declared_vram_gb": 8}


def audit_selection_artifacts(root: Path) -> dict[str, Any]:
    paths = {
        "t2": root/"artifacts"/"predictions"/"bertweetbr_asl_oof.parquet",
        "t3": root/"artifacts"/"predictions"/"bertimbau_large_bce_oof.parquet",
        "en2": root/"artifacts"/"predictions"/"ensemble_t2_t3_tuned_weight_oof.parquet",
    }
    frames = {key: pd.read_parquet(path) for key,path in paths.items()}
    reference = frames["t2"]
    for key,frame in frames.items():
        if len(frame) != DEV_POOL_SIZE or frame.id.nunique() != DEV_POOL_SIZE:
            raise RuntimeError(f"Incomplete {key} OOF")
        if frame.id.tolist() != reference.id.tolist() or frame.fold.tolist() != reference.fold.tolist():
            raise RuntimeError(f"ID/fold mismatch in {key} OOF")
        for label in EXPECTED_LABELS:
            if not np.array_equal(frame[f"{label}_true"].to_numpy(), reference[f"{label}_true"].to_numpy()):
                raise RuntimeError(f"Label mismatch in {key}/{label}")
            probabilities = frame[f"{label}_prob"].to_numpy(float)
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
                raise RuntimeError(f"Invalid probabilities in {key}/{label}")
    return {"rows": DEV_POOL_SIZE, "unique_ids": DEV_POOL_SIZE, "ordered_ids_match": True,
            "labels_match": True, "outer_folds_match": True,
            "oof_sha256": {key: sha256_file(path) for key,path in paths.items()},
            "test_content_read": False}


def prepare_deployment_freeze(root: Path) -> dict[str, Any]:
    """Select deployment-only parameters and write the pre-registered protocol."""
    audit = audit_selection_artifacts(root)
    t2 = pd.read_parquet(root/"artifacts"/"predictions"/"bertweetbr_asl_oof.parquet")
    t3 = pd.read_parquet(root/"artifacts"/"predictions"/"bertimbau_large_bce_oof.parquet")
    y = t2[[f"{label}_true" for label in EXPECTED_LABELS]].to_numpy(int)
    p2 = t2[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(float)
    p3 = t3[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(float)
    alpha, thresholds, calibration_score, trace = select_alpha(y, p2, p3)
    t2_thresholds, t2_scores = search_per_label_thresholds(y, p2)
    nested_alpha = pd.read_csv(root/"reports"/"brighter_ptbr"/"experiments"/"ensemble_alpha_by_fold.csv").alpha_T2
    nested_thresholds = pd.read_csv(root/"reports"/"brighter_ptbr"/"experiments"/"ensemble_thresholds_by_fold.csv").query("system == 'EN2'")
    alpha_consistent = bool(nested_alpha.min() <= alpha <= nested_alpha.max())
    threshold_comparison = {label: {"final": float(thresholds[i]), "nested_mean": float(nested_thresholds[label].mean()),
                                    "nested_sd": float(nested_thresholds[label].std(ddof=1)), "nested_min": float(nested_thresholds[label].min()),
                                    "nested_max": float(nested_thresholds[label].max()),
                                    "within_nested_range": bool(nested_thresholds[label].min() <= thresholds[i] <= nested_thresholds[label].max())}
                            for i,label in enumerate(EXPECTED_LABELS)}
    pool = construct_development_pool(root)
    ids = pool.id.astype(str).tolist()
    if set(pool.source_split) != {"train","dev"} or any("_test_" in identifier for identifier in ids):
        raise RuntimeError("Final training pool provenance is invalid")
    final_dir = root/FINAL_ROOT; final_dir.mkdir(parents=True, exist_ok=True)
    labels_payload = {"label_order": LABELS, "num_labels": 6, "sha256": hashlib.sha256("\n".join(LABELS).encode()).hexdigest()}
    write_json_atomic(final_dir/"labels.json", labels_payload)
    decision = {"system":"EN2","dataset_revision":FROZEN_REVISION,"development_examples":DEV_POOL_SIZE,
                "alpha_t2":float(alpha),"weight_t3":float(1-alpha),
                "thresholds":{label:float(thresholds[i]) for i,label in enumerate(EXPECTED_LABELS)},
                "selection_source":"paired cross-fitted OOF predictions over full development pool",
                "selection_protocol":{"alpha_grid":[0.0,1.0,0.05],"threshold_grid":[0.01,0.99,0.01],
                                      "objective":"per-label F1 / Macro-F1","alpha_tie_break":"closest_to_0.5_then_larger_T2",
                                      "threshold_tie_break":"largest_threshold"},
                "performance_estimate":{"method":"nested leakage-safe outer evaluation","macro_f1":NESTED_PERFORMANCE},
                "deployment_calibration":{"macro_f1":float(calibration_score),"warning":WARNING},
                "deployment_calibration_score_is_unbiased":False,"nested_consistency":{"alpha_within_range":alpha_consistent,
                "nested_alpha_mean":float(nested_alpha.mean()),"nested_alpha_sd":float(nested_alpha.std(ddof=1)),
                "nested_alpha_min":float(nested_alpha.min()),"nested_alpha_max":float(nested_alpha.max()),
                "thresholds":threshold_comparison},"label_order":LABELS,"test_policy":"SEALED — NOT LOADED OR EVALUATED"}
    write_json_atomic(final_dir/"deployment_decision_parameters.json",decision)
    t2_payload={"system":"T2 standalone final","dataset_revision":FROZEN_REVISION,"development_examples":DEV_POOL_SIZE,
                "thresholds":{label:float(t2_thresholds[i]) for i,label in enumerate(EXPECTED_LABELS)},
                "selection_source":"cross-fitted T2 OOF predictions over full development pool",
                "deployment_calibration":{"macro_f1":float(t2_scores.mean()),"warning":WARNING},
                "deployment_calibration_score_is_unbiased":False,"performance_estimate":{"method":"nested leakage-safe outer evaluation","macro_f1":0.5585138660403824},
                "label_order":LABELS,"test_policy":"SEALED — NOT LOADED OR EVALUATED"}
    write_json_atomic(final_dir/"t2_standalone_thresholds.json",t2_payload)
    (root/REPORT_ROOT).mkdir(parents=True,exist_ok=True)
    trace.to_csv(root/REPORT_ROOT/"deployment_alpha_search.csv",index=False,float_format="%.8f")
    en2_config={"system":"EN2","role":"PRIMARY","frozen":True,"dataset_revision":FROZEN_REVISION,"development_examples":DEV_POOL_SIZE,
                "models":{"t2":{"path":"artifacts/models/final/t2_bertweetbr_asl","checkpoint":"melll-uff/bertweetbr"},
                          "t3":{"path":"artifacts/models/final/t3_bertimbau_large_bce","checkpoint":"neuralmind/bert-large-portuguese-cased"}},
                "alpha_t2":float(alpha),"weight_t3":float(1-alpha),"thresholds":decision["thresholds"],"label_order":LABELS,
                "max_length":128,"preprocessing":"none","test_policy":"SEALED_UNTIL_SEPARATELY_AUTHORIZED"}
    t2_config={"system":"T2","role":"SECONDARY_PRE_REGISTERED","frozen":True,"dataset_revision":FROZEN_REVISION,
               "development_examples":DEV_POOL_SIZE,"model":{"path":"artifacts/models/final/t2_bertweetbr_asl","checkpoint":"melll-uff/bertweetbr"},
               "thresholds":t2_payload["thresholds"],"label_order":LABELS,"max_length":128,"preprocessing":"none",
               "test_policy":"SEALED_UNTIL_SEPARATELY_AUTHORIZED"}
    _write_yaml(root/"configs"/"final_en2.yaml",en2_config); _write_yaml(root/"configs"/"final_t2_standalone.yaml",t2_config)
    state={"state":"PREPARING_FINAL_MODELS","primary":"EN2","secondary":"T2_STANDALONE","test_accessed":False,
           "test_policy":"SEALED — NOT LOADED OR EVALUATED","space_of_search":"CLOSED","updated_at":utc_now_iso()}
    write_json_atomic(final_dir/"experiment_state.json",state)
    _write_protocol(root,decision,t2_payload,audit,ids)
    return {"decision":decision,"t2":t2_payload,"audit":audit,"training_ids_sha256":_id_digest(ids)}


def _write_protocol(root: Path, decision: Mapping[str,Any], t2: Mapping[str,Any], audit: Mapping[str,Any], ids:list[str]) -> None:
    thresholds=decision["thresholds"]
    lines=["# BRIGHTER PT-BR — pre-registered final-test protocol","","Status: **PRE-REGISTERED; TEST SEALED AND NOT ACCESSED**.","",
           "This document was written before final full-development training and before any test access.","","## 1. Declared systems","",
           "- PRIMARY: EN2 final — T2 BERTweetBR+ASL and T3 BERTimbau Large+BCE.","- SECONDARY: T2 standalone final.",
           "- No T3 standalone, EN1, T1, Albertina, or other candidate may be evaluated as a competing final system.","","## 2. Frozen decision rule","",
           f"`p = {decision['alpha_t2']:.2f} * p_T2 + {decision['weight_t3']:.2f} * p_T3`.","",
           "Threshold order: `anger, disgust, fear, joy, sadness, surprise`.","",
           "EN2 thresholds: `"+", ".join(f"{x}={thresholds[x]:.2f}" for x in EXPECTED_LABELS)+"`.","",
           "T2 standalone thresholds: `"+", ".join(f"{x}={t2['thresholds'][x]:.2f}" for x in EXPECTED_LABELS)+"`.","",
           "## 3. Metrics","","Primary: Macro-F1. Secondary: Micro-F1, per-label Precision/Recall/F1, Hamming Loss, sample Jaccard, Subset Accuracy, Exact Neutral Accuracy, Neutral False Positive Rate, and label cardinality.","",
           "## 4. Immutable evaluation rules","","1. No alpha change after test access.","2. No threshold calibration or selection on test.","3. No model/loss/seed change.","4. No choice between T2 and EN2 based on test; EN2 is primary and T2 is always reported secondarily.","5. Exactly one separately authorized final evaluation is permitted.","6. No preprocessing, label modification, sampling, augmentation, translation, synthetic/external data, or Track B.","",
           "## 5. Pre-test provenance","",f"Development IDs: {len(ids)}; SHA-256 `{_id_digest(ids)}`. OOF audit: {audit['unique_ids']} unique IDs with matched labels/folds.","",
           "The unbiased pre-test development estimate remains nested EN2 Macro-F1 = 0.5691. Any full-OOF deployment calibration score is explicitly not an unbiased performance estimate.",""]
    _write_text(root/REPORT_ROOT/"final_test_protocol.md","\n".join(lines))


def _source_config(root:Path,system:str)->dict[str,Any]:
    source=FINAL_SPECS[system]["source"]
    config=dict(load_json(root/"reports"/"brighter_ptbr"/"experiments"/f"{source}_results.json")["config"])
    config["experiment_id"]=FINAL_SPECS[system]["registry"]
    config["artifact_slug"]=f"final_{source}"
    return config


def _model_files_hashes(directory:Path)->dict[str,str]:
    relevant=[]
    for pattern in ("*.safetensors","config.json","tokenizer_config.json","tokenizer.json","vocab.txt","bpe.codes","added_tokens.json","special_tokens_map.json"):
        relevant.extend(directory.glob(pattern))
    return {path.name:sha256_file(path) for path in sorted(set(relevant))}


def train_final_model(root:Path,system:str)->dict[str,Any]:
    """Train one frozen model for four epochs on all 2,426 train+dev examples."""
    if system not in FINAL_SPECS: raise ValueError("system must be T2 or T3")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,get_linear_schedule_with_warmup
    if not torch.cuda.is_available(): raise RuntimeError("Final transformer training requires CUDA")
    pool=construct_development_pool(root); ids=pool.id.astype(str).tolist(); ids_hash=_id_digest(ids)
    if len(pool)!=DEV_POOL_SIZE or set(pool.source_split)!={"train","dev"} or any("_test_" in x for x in ids): raise RuntimeError("Invalid final training pool")
    config=_source_config(root,system); config_hash=_json_digest(config); output=root/"artifacts"/"models"/"final"/FINAL_SPECS[system]["directory"]
    manifest_path=output/"final_training_manifest.json"
    if manifest_path.exists():
        existing=load_json(manifest_path)
        if existing.get("config_sha256")!=config_hash or existing.get("training_ids_sha256")!=ids_hash or existing.get("development_examples")!=DEV_POOL_SIZE: raise RuntimeError(f"Stale final {system} manifest; refusing overwrite")
        if not existing.get("completed") or any(sha256_file(output/name)!=digest for name,digest in existing["file_hashes"].items()): raise RuntimeError(f"Invalid final {system} artifact; refusing overwrite")
        return existing
    if output.exists() and any(output.iterdir()): raise RuntimeError(f"Partial final {system} directory exists; refusing overwrite")
    output.mkdir(parents=True,exist_ok=True); seed_everything(42); training=config["training"]; name=config["model_name"]; revision=_resolve_model_revision(root,name)
    project_cache=root/".cache"/"huggingface"/"hub"; repo="models--"+name.replace("/","--"); cache=project_cache if (project_cache/repo/"snapshots"/revision).exists() else None
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision,use_fast=True,cache_dir=cache)
    model=AutoModelForSequenceClassification.from_pretrained(name,revision=revision,num_labels=6,problem_type="multi_label_classification",id2label={i:x for i,x in enumerate(EXPECTED_LABELS)},label2id={x:i for i,x in enumerate(EXPECTED_LABELS)},cache_dir=cache,dtype=torch.float32)
    if training.get("gradient_checkpointing",False): model.gradient_checkpointing_enable(); model.config.use_cache=False
    device=torch.device("cuda"); model.to(device); weights=calculate_fold_class_weights(pool[list(EXPECTED_LABELS)].to_numpy(int)); configured="bce" if config["loss"]=="BCEWithLogitsLoss" else config["loss"]; criterion=build_loss(configured,fold_weights=weights,loss_config=config.get("loss_config")); criterion=criterion.to(device) if hasattr(criterion,"to") else criterion
    loader=DataLoader(_TextDataset(pool,tokenizer,int(config["max_length"])),batch_size=int(training["physical_batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=_MultilabelCollator(tokenizer),num_workers=0,pin_memory=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(training["learning_rate"]),weight_decay=float(training["weight_decay"]),fused=True); accumulation=int(training["gradient_accumulation_steps"]); epochs=int(training["epochs"]); steps=math.ceil(len(loader)/accumulation)*epochs; scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=round(steps*float(training["warmup_ratio"])),num_training_steps=steps)
    precision=str(training.get("mixed_precision","fp16")); dtype=torch.float16 if precision=="fp16" else torch.bfloat16 if precision=="bf16" else None; autocast=(lambda:torch.autocast(device_type="cuda",dtype=dtype)) if dtype is not None else nullcontext; scaler=torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    torch.cuda.reset_peak_memory_stats(); started=time.perf_counter(); optimizer.zero_grad(set_to_none=True); model.train(); epoch_seconds=[]
    for epoch in range(epochs):
        epoch_start=time.perf_counter()
        for step,batch in enumerate(loader):
            batch={key:value.to(device,non_blocking=True) for key,value in batch.items()}; labels=batch.pop("labels")
            with autocast(): loss=criterion(model(**batch).logits,labels)/accumulation
            scaler.scale(loss).backward()
            if (step+1)%accumulation==0 or step+1==len(loader): scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),float(training.get("max_grad_norm",1.0))); scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); epoch_seconds.append(time.perf_counter()-epoch_start); print(f"{system} epoch {epoch+1}/{epochs} complete ({epoch_seconds[-1]:.1f}s)",flush=True)
    duration=time.perf_counter()-started; peak=int(torch.cuda.max_memory_allocated()); parameters=int(sum(x.numel() for x in model.parameters())); model.save_pretrained(output,safe_serialization=True); tokenizer.save_pretrained(output)
    write_json_atomic(output/"labels.json",{"label_order":LABELS}); write_json_atomic(output/"training_config.json",config)
    payload={"system":system,"experiment_id":FINAL_SPECS[system]["registry"],"completed":True,"trained_on":"official train+dev only","development_examples":DEV_POOL_SIZE,"source_split_counts":{str(k):int(v) for k,v in pool.source_split.value_counts().items()},"training_ids_sha256":ids_hash,"dataset_revision":FROZEN_REVISION,"dataset_hashes":development_input_hashes(root),"config_sha256":config_hash,"seed":42,"epochs":epochs,"label_order":LABELS,"loss":config["loss"],"loss_config":config.get("loss_config",{}),"model_name":name,"model_revision":revision,"training_seconds":duration,"epoch_seconds":epoch_seconds,"peak_vram_bytes":peak,"parameters":parameters,"physical_batch":training["physical_batch_size"],"gradient_accumulation":training["gradient_accumulation_steps"],"effective_batch":training["effective_batch_size"],"software_versions":_software_versions(),"hardware":_gpu_metadata(),"test_ids_loaded":False,"test_policy":"SEALED — NOT LOADED OR EVALUATED","completed_timestamp_utc":utc_now_iso()}
    payload["file_hashes"]=_model_files_hashes(output); write_json_atomic(manifest_path,payload)
    del model,optimizer,scheduler,scaler,loader; torch.cuda.empty_cache(); return payload


def dry_run_final_models(root:Path)->dict[str,Any]:
    """Software-only inference using artificial strings; never opens dataset files."""
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    decision=load_json(root/FINAL_ROOT/"deployment_decision_parameters.json"); texts=["Texto artificial neutro.","Estou feliz e surpreso!"]
    outputs={}
    for system in ("T2","T3"):
        directory=root/"artifacts"/"models"/"final"/FINAL_SPECS[system]["directory"]
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(directory,local_files_only=True).to("cuda").eval()
        batch=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors="pt"); batch={k:v.to("cuda") for k,v in batch.items()}
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16): probabilities=torch.sigmoid(model(**batch).logits.float()).cpu().numpy()
        if probabilities.shape!=(2,6) or not np.isfinite(probabilities).all(): raise RuntimeError(f"Invalid dry-run output for {system}")
        outputs[system]=probabilities; del model; torch.cuda.empty_cache()
    ensemble=fuse_probabilities(outputs["T2"],outputs["T3"],decision["alpha_t2"]); thresholds=np.array([decision["thresholds"][x] for x in EXPECTED_LABELS]); predictions=(ensemble>=thresholds).astype(np.int8)
    if ensemble.shape!=(2,6) or predictions.shape!=(2,6): raise RuntimeError("Invalid ensemble dry-run shape")
    payload={"fixture":"two artificial software-test strings","official_test_used":False,"t2_shape":[2,6],"t3_shape":[2,6],"ensemble_shape":[2,6],"predictions_shape":[2,6],"alpha_t2":decision["alpha_t2"],"thresholds":decision["thresholds"],"finite_probabilities":True,"completed_timestamp_utc":utc_now_iso(),"test_policy":"SEALED — NOT LOADED OR EVALUATED"}
    write_json_atomic(root/FINAL_ROOT/"dry_run_validation.json",payload); return payload


def _update_registry(root:Path,t2:Mapping[str,Any],t3:Mapping[str,Any],decision:Mapping[str,Any])->None:
    path=root/"reports"/"brighter_ptbr"/"experiments"/"experiment_registry.csv"; old=pd.read_csv(path); ids={"FINAL_T2_BERTWEETBR_ASL","FINAL_T3_BERTIMBAU_LARGE_BCE","FINAL_EN2"}; old=old[~old.experiment_id.isin(ids)]
    common={"seed":42,"folds":1,"threshold_strategy":"frozen_deployment","status":"FROZEN_PRE_TEST","timestamp":utc_now_iso(),"git_commit":"NOT_A_GIT_REPOSITORY","fold":"all_development","train_sha256":t2["dataset_hashes"]["train"],"dev_sha256":t2["dataset_hashes"]["dev"],"test_policy":"SEALED — NOT LOADED OR EVALUATED"}
    rows=[{"experiment_id":"FINAL_T2_BERTWEETBR_ASL","model":"BERTweetBR","loss":"ASL","max_length":128,"learning_rate":2e-5,"epochs":4,"effective_batch_size":16,"threshold":json.dumps(load_json(root/FINAL_ROOT/"t2_standalone_thresholds.json")["thresholds"],sort_keys=True),"training_time":t2["training_seconds"],"model_checkpoints":"artifacts/models/final/t2_bertweetbr_asl","parameter_count":t2["parameters"],"peak_gpu_memory_bytes":t2["peak_vram_bytes"],"encoder":"BERTweetBR","hugging_face_checkpoint":t2["model_name"],"architecture_family":"RoBERTa","physical_batch":t2["physical_batch"],"accumulation":t2["gradient_accumulation"],"mixed_precision":"fp16",**common},
          {"experiment_id":"FINAL_T3_BERTIMBAU_LARGE_BCE","model":"BERTimbau Large","loss":"BCEWithLogitsLoss","max_length":128,"learning_rate":2e-5,"epochs":4,"effective_batch_size":16,"threshold":"component_only","training_time":t3["training_seconds"],"model_checkpoints":"artifacts/models/final/t3_bertimbau_large_bce","parameter_count":t3["parameters"],"peak_gpu_memory_bytes":t3["peak_vram_bytes"],"encoder":"BERTimbau Large","hugging_face_checkpoint":t3["model_name"],"architecture_family":"BERT","physical_batch":t3["physical_batch"],"accumulation":t3["gradient_accumulation"],"mixed_precision":"fp16",**common},
          {"experiment_id":"FINAL_EN2","model":"T2+T3 tuned soft ensemble","loss":"ASL+BCE components","max_length":128,"effective_batch_size":16,"threshold":json.dumps(decision["thresholds"],sort_keys=True),"training_time":t2["training_seconds"]+t3["training_seconds"],"model_checkpoints":"artifacts/models/final/t2_bertweetbr_asl + artifacts/models/final/t3_bertimbau_large_bce","parameter_count":t2["parameters"]+t3["parameters"],"encoder":"BERTweetBR + BERTimbau Large","architecture_family":"RoBERTa + BERT","**common":None}]
    # Replace the sentinel used to keep the dense literal readable.
    rows[-1].pop("**common",None); rows[-1].update(common)
    combined=pd.concat([old,pd.DataFrame(rows)],ignore_index=True).reindex(columns=old.columns); tmp=path.with_suffix(".csv.tmp"); combined.to_csv(tmp,index=False,float_format="%.8f",lineterminator="\n"); os.replace(tmp,path)


def finalize_freeze(root:Path)->dict[str,Any]:
    t2=load_json(root/"artifacts"/"models"/"final"/FINAL_SPECS["T2"]["directory"]/"final_training_manifest.json"); t3=load_json(root/"artifacts"/"models"/"final"/FINAL_SPECS["T3"]["directory"]/"final_training_manifest.json"); decision=load_json(root/FINAL_ROOT/"deployment_decision_parameters.json"); dry=load_json(root/FINAL_ROOT/"dry_run_validation.json")
    if not t2["completed"] or not t3["completed"] or dry["official_test_used"]: raise RuntimeError("Final artifacts are incomplete")
    pool=construct_development_pool(root); ids=pool.id.astype(str).tolist(); integrity=_dataset_integrity(root)
    if not all(item["match"] for stage in integrity["hashes"].values() for item in stage.values()): raise RuntimeError("Dataset integrity failure")
    configs={name:sha256_file(root/"configs"/name) for name in ("final_en2.yaml","final_t2_standalone.yaml")}
    manifest={"experiment_status":"FROZEN_PRE_TEST","dataset":{"id":"brighter-dataset/BRIGHTER-emotion-categories","configuration":"ptbr","track":"A","revision":FROZEN_REVISION,"development_examples":DEV_POOL_SIZE,"hashes":integrity["hashes"],"development_ids_sha256":_id_digest(ids)},"primary":"EN2","secondary":"T2_STANDALONE","performance_estimate":{"method":"nested leakage-safe outer evaluation","macro_f1":NESTED_PERFORMANCE},"deployment":{"alpha_t2":decision["alpha_t2"],"weight_t3":decision["weight_t3"],"thresholds":decision["thresholds"],"calibration_score_is_unbiased":False},"label_order":LABELS,"models":{"T2":{"path":"artifacts/models/final/t2_bertweetbr_asl","file_hashes":t2["file_hashes"],"manifest_sha256":sha256_file(root/"artifacts"/"models"/"final"/FINAL_SPECS["T2"]["directory"]/"final_training_manifest.json")},"T3":{"path":"artifacts/models/final/t3_bertimbau_large_bce","file_hashes":t3["file_hashes"],"manifest_sha256":sha256_file(root/"artifacts"/"models"/"final"/FINAL_SPECS["T3"]["directory"]/"final_training_manifest.json")}},"config_hashes":configs,"software_versions":_software_versions(),"hardware":_gpu_metadata(),"seed":42,"test_accessed":False,"test_policy":"SEALED — NOT LOADED OR EVALUATED","space_of_search":"CLOSED","timestamp_utc":utc_now_iso()}
    write_json_atomic(root/FINAL_ROOT/"final_experiment_manifest.json",manifest)
    state={"state":"FROZEN_PRE_TEST","primary":"EN2","secondary":"T2_STANDALONE","test_accessed":False,"test_policy":"SEALED — NOT LOADED OR EVALUATED","space_of_search":"CLOSED","manifest_sha256":sha256_file(root/FINAL_ROOT/"final_experiment_manifest.json"),"updated_at":utc_now_iso()}; write_json_atomic(root/FINAL_ROOT/"experiment_state.json",state)
    _update_registry(root,t2,t3,decision); _write_freeze_report(root,manifest,t2,t3,decision); return manifest


def _write_freeze_report(root:Path,manifest:Mapping[str,Any],t2:Mapping[str,Any],t3:Mapping[str,Any],decision:Mapping[str,Any])->None:
    comparison=decision["nested_consistency"]["thresholds"]
    lines=["# BRIGHTER PT-BR — model freeze report","","State: **FROZEN_PRE_TEST**. Test: **SEALED — NOT LOADED OR EVALUATED**.","","## 1. Frozen dataset","",f"BRIGHTER Track A PT-BR revision `{FROZEN_REVISION}`; official train+dev only, 2,426 examples. Development IDs SHA-256: `{manifest['dataset']['development_ids_sha256']}`.","","Data policy: preprocessing, label changes, sampling, augmentation, translation, synthetic/external data = NONE; Track B = OUT OF SCOPE.","","## 2. Final systems and training","",f"T2: `melll-uff/bertweetbr`, ASL gamma_pos=0, gamma_neg=4, clip=0.05; 4 epochs, LR 2e-5, AdamW, weight decay 0.01, warmup 0.1, effective batch 16, max length 128, seed 42. Trained on {t2['development_examples']} examples in {t2['training_seconds']:.1f}s; peak VRAM {t2['peak_vram_bytes']/2**30:.2f} GiB; {t2['parameters']:,} parameters.","",f"T3: `neuralmind/bert-large-portuguese-cased`, BCEWithLogitsLoss; same frozen optimizer/schedule/epochs/max length/effective batch/seed. Trained on {t3['development_examples']} examples in {t3['training_seconds']:.1f}s; peak VRAM {t3['peak_vram_bytes']/2**30:.2f} GiB; {t3['parameters']:,} parameters.","","No validation split, early stopping, tuning, additional seed, or test observation was used.","","## 3. Frozen deployment decision","",f"Alpha T2 = **{decision['alpha_t2']:.2f}**; T3 weight = **{decision['weight_t3']:.2f}**.","","| Label | Final threshold | Nested mean ± SD | Nested range | Consistent |","|---|---:|---:|---:|---:|"]
    for label in EXPECTED_LABELS:
        x=comparison[label]; lines.append(f"| {label} | {x['final']:.2f} | {x['nested_mean']:.3f} ± {x['nested_sd']:.3f} | {x['nested_min']:.2f}–{x['nested_max']:.2f} | {'yes' if x['within_nested_range'] else 'no'} |")
    lines += ["",f"Alpha nested range was 0.75–0.95; global deployment alpha {decision['alpha_t2']:.2f} lies outside it. This value was not manually changed. The likely reason is the less regularized full-OOF optimization, which uses all 2,426 OOF labels for deployment selection rather than five separate outer-train calibration sets. This is recorded as a deployment stability warning.","","## 4. Scientific estimate versus deployment calibration","",f"The frozen unbiased development estimate remains nested EN2 Macro-F1 = **{NESTED_PERFORMANCE:.4f}**. Global deployment calibration Macro-F1 = {decision['deployment_calibration']['macro_f1']:.4f} is labeled: **{WARNING}** It must never replace the nested estimate.","","## 5. Test protocol","","The pre-registered protocol is in `final_test_protocol.md`. EN2 must be reported as primary and T2 standalone as secondary. Alpha, thresholds, models, label order, and metrics cannot change after test access. T3 is a component, not a third candidate.","","## 6. Hashes and environment","",f"T2 model manifest SHA-256: `{manifest['models']['T2']['manifest_sha256']}`.",f"T3 model manifest SHA-256: `{manifest['models']['T3']['manifest_sha256']}`.",f"Final EN2 config SHA-256: `{manifest['config_hashes']['final_en2.yaml']}`.",f"Final T2 config SHA-256: `{manifest['config_hashes']['final_t2_standalone.yaml']}`.","",f"Hardware: {manifest['hardware']['gpu']}; declared 8 GB VRAM and 32 GB RAM. Software versions are persisted in the final manifest.","","## 7. Immutability statement","","The search space is closed. No decision may change after the test is opened. This task did not authorize or perform test access.",""]
    _write_text(root/REPORT_ROOT/"model_freeze_report.md","\n".join(lines))
