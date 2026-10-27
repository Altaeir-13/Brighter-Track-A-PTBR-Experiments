#!/usr/bin/env python
"""Real-batch CUDA smoke tests for the frozen encoder checkpoints."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from brighter.experiments import (
    EXPECTED_LABELS,
    _MultilabelCollator,
    _TextDataset,
    attach_folds,
    construct_development_pool,
    persist_cv_folds,
    seed_everything,
)
from brighter.losses import build_loss, calculate_fold_class_weights
from brighter.manifest import load_json, utc_now_iso, write_json_atomic

ENCODERS = (
    ("bertweetbr", "melll-uff/bertweetbr"),
    ("albertina100m", "PORTULAN/albertina-100m-portuguese-ptbr-encoder"),
)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for encoder smoke tests")
    seed_everything(42)
    revisions = load_json(PROJECT_ROOT / "metadata" / "brighter_ptbr" / "experiments" / "model_revisions.json")
    cache_dir = PROJECT_ROOT / ".cache" / "huggingface" / "hub"
    pool = construct_development_pool(PROJECT_ROOT)
    pool = attach_folds(pool, persist_cv_folds(PROJECT_ROOT, pool))
    fold_training = pool.loc[pool["fold"] != 0].copy()
    real_batch = fold_training.iloc[:8].copy()
    weights = calculate_fold_class_weights(fold_training[list(EXPECTED_LABELS)].to_numpy(dtype=int))
    records = []

    for slug, model_name in ENCODERS:
        revision = revisions[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            num_labels=len(EXPECTED_LABELS),
            problem_type="multi_label_classification",
            id2label={index: label for index, label in enumerate(EXPECTED_LABELS)},
            label2id={label: index for index, label in enumerate(EXPECTED_LABELS)},
            cache_dir=cache_dir,
            dtype=torch.float32,
        )
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.to("cuda").train()
        collator = _MultilabelCollator(tokenizer)
        loader = DataLoader(_TextDataset(real_batch, tokenizer, 128), batch_size=8, collate_fn=collator)
        torch.cuda.reset_peak_memory_stats()
        precision = "fp32" if slug == "albertina100m" else "fp16"
        autocast_dtype = None if precision == "fp32" else torch.float16
        autocast_context = (lambda: torch.autocast(device_type="cuda", dtype=autocast_dtype)) if autocast_dtype is not None else nullcontext
        loss_records = []
        for loss_name, options in (
            ("bce", {}),
            ("weighted_focal", {"gamma": 2.0}),
            ("asl", {"gamma_pos": 0.0, "gamma_neg": 4.0, "clip": 0.05}),
        ):
            model.zero_grad(set_to_none=True)
            batch = next(iter(loader))
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            targets = batch.pop("labels")
            criterion = build_loss(loss_name, fold_weights=weights, loss_config=options)
            if hasattr(criterion, "to"):
                criterion = criterion.to("cuda")
            with autocast_context():
                logits = model(**batch).logits
                loss = criterion(logits, targets)
            if logits.shape != (8, len(EXPECTED_LABELS)) or not torch.isfinite(loss):
                raise RuntimeError(f"Invalid {model_name}/{loss_name} smoke output")
            loss.backward()
            if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
                raise RuntimeError(f"No gradients for {model_name}/{loss_name}")
            loss_records.append({"loss": loss_name, "value": float(loss.detach().cpu()), "backward": "ok"})
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
        checkpoint_dir = PROJECT_ROOT / ".cache" / "encoder_smoke" / slug
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        del model
        torch.cuda.empty_cache()

        reloaded_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to("cuda").eval()
        encoded = reloaded_tokenizer(real_batch["text_raw"].tolist()[:2], padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode(), autocast_context():
            reload_shape = list(reloaded(**encoded).logits.shape)
        if reload_shape != [2, len(EXPECTED_LABELS)]:
            raise RuntimeError(f"Checkpoint reload failed for {model_name}")
        parameters = int(sum(parameter.numel() for parameter in reloaded.parameters()))
        records.append({
            "model_name": model_name, "revision": revision, "architecture": reloaded.config.model_type,
            "tokenizer_class": type(reloaded_tokenizer).__name__,
            "tokenizer_normalization": getattr(reloaded_tokenizer, "normalization", None),
            "model_input_names": list(reloaded_tokenizer.model_input_names),
            "batch_shape": [8, len(EXPECTED_LABELS)], "losses": loss_records,
            "master_weight_dtype": "float32", "mixed_precision": precision,
            "peak_gpu_memory_bytes": peak, "parameters": parameters,
            "checkpoint_reload": "ok", "reload_shape": reload_shape,
        })
        del reloaded
        torch.cuda.empty_cache()

    output = {
        "timestamp_utc": utc_now_iso(), "real_training_examples": 8, "fold": 0,
        "max_length": 128, "physical_batch": 8, "test_policy": "SEALED — NOT LOADED OR EVALUATED",
        "encoders": records,
    }
    write_json_atomic(PROJECT_ROOT / "metadata" / "brighter_ptbr" / "experiments" / "encoder_smoke_tests.json", output)
    for record in records:
        print(record["model_name"], record["architecture"], record["tokenizer_class"], f"peak={record['peak_gpu_memory_bytes']/2**30:.2f} GiB", "reload=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
