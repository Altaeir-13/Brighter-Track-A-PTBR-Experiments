from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from brighter.experiments import EXPECTED_LABELS, attach_folds, construct_development_pool, persist_cv_folds
from brighter.manifest import utc_now_iso, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
MODEL = "neuralmind/bert-large-portuguese-cased"
REVISION = "aa302f6ea73b759f7df9cad58bd272127b67ec28"

pool = construct_development_pool(ROOT)
pool = attach_folds(pool, persist_cv_folds(ROOT, pool))
example = pool.loc[pool.fold != 0].iloc[0]
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, revision=REVISION, num_labels=6, problem_type="multi_label_classification", dtype=torch.float32,
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.cuda().train()
batch = tokenizer([str(example.text_raw)], padding=True, truncation=True, max_length=128, return_tensors="pt")
batch = {key: value.cuda() for key, value in batch.items()}
labels = torch.tensor([[float(example[label]) for label in EXPECTED_LABELS]], device="cuda")
torch.cuda.reset_peak_memory_stats()
with torch.autocast("cuda", dtype=torch.float16):
    logits = model(**batch).logits
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
loss.backward()
torch.cuda.synchronize()
payload = {
    "timestamp_utc": utc_now_iso(), "system": "T3", "model": MODEL, "revision": REVISION,
    "source_scope": "outer_train_only", "batch": 1, "max_length": 128,
    "output_shape": list(logits.shape), "forward": "ok", "backward": "ok",
    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    "test_policy": "SEALED — NOT LOADED OR EVALUATED",
}
write_json_atomic(ROOT / "metadata" / "brighter_ptbr" / "experiments" / "threshold_t3_smoke.json", payload)
print(payload)
