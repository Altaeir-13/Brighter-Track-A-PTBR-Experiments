from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from transformers import AutoConfig, AutoTokenizer

from brighter.experiments import EXPECTED_LABELS, attach_folds, construct_development_pool, load_config, persist_cv_folds, validate_oof
from brighter.manifest import load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENCODERS = {
    "bertweetbr": ("melll-uff/bertweetbr", "roberta", "BertweetTokenizer", "fp16"),
    "albertina100m": ("PORTULAN/albertina-100m-portuguese-ptbr-encoder", "deberta", "DebertaTokenizer", "fp32"),
}
OOF_SLUGS = (
    "bertweetbr_bce", "bertweetbr_weighted_focal", "bertweetbr_asl",
    "albertina100m_bce", "albertina100m_weighted_focal", "albertina100m_asl",
)


def test_real_encoder_smoke_manifest_covers_forward_backward_losses_and_reload() -> None:
    payload = load_json(PROJECT_ROOT / "metadata" / "brighter_ptbr" / "experiments" / "encoder_smoke_tests.json")
    assert payload["test_policy"] == "SEALED — NOT LOADED OR EVALUATED"
    assert payload["real_training_examples"] == 8
    records = {item["model_name"]: item for item in payload["encoders"]}
    for _, (model_name, architecture, tokenizer_class, precision) in ENCODERS.items():
        record = records[model_name]
        assert record["architecture"] == architecture
        assert record["tokenizer_class"] == tokenizer_class
        assert record["mixed_precision"] == precision
        assert record["batch_shape"] == [8, 6]
        assert record["reload_shape"] == [2, 6]
        assert record["checkpoint_reload"] == "ok"
        assert record["peak_gpu_memory_bytes"] < 8 * 2**30
        assert {item["loss"] for item in record["losses"]} == {"bce", "weighted_focal", "asl"}
        assert all(item["backward"] == "ok" for item in record["losses"])


@pytest.mark.parametrize("slug", ENCODERS)
def test_cached_tokenizer_padding_optional_token_types_and_six_label_checkpoint(slug: str) -> None:
    model_name, architecture, tokenizer_class, _ = ENCODERS[slug]
    checkpoint = PROJECT_ROOT / ".cache" / "encoder_smoke" / slug
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    config = AutoConfig.from_pretrained(checkpoint)
    encoded = tokenizer(["Oi @user 😄", "Texto maior sem qualquer limpeza adicional!"], padding=True, truncation=True, max_length=128, return_tensors="pt")
    assert type(tokenizer).__name__ == tokenizer_class
    assert config.model_type == architecture
    assert config.num_labels == len(EXPECTED_LABELS)
    assert encoded["input_ids"].shape[0] == 2
    assert encoded["input_ids"].shape == encoded["attention_mask"].shape
    assert ("token_type_ids" in encoded) == (slug == "albertina100m")
    if slug == "bertweetbr":
        assert tokenizer.normalization is False


@pytest.mark.parametrize("config_name", [
    "bertweetbr_bce.yaml", "bertweetbr_weighted_focal.yaml", "bertweetbr_asl.yaml",
    "albertina100m_bce.yaml", "albertina100m_weighted_focal.yaml", "albertina100m_asl.yaml",
])
def test_encoder_configs_keep_frozen_protocol(config_name: str) -> None:
    config = load_config(PROJECT_ROOT / "configs" / config_name)
    training = config["training"]
    assert config["threshold"] == 0.5
    assert config["max_length"] == 128
    assert training["learning_rate"] == 2e-5
    assert training["epochs"] == 4
    assert training["physical_batch_size"] * training["gradient_accumulation_steps"] == 16
    assert config["data_policy"] == {"track": "A", "preprocessing": "none", "sampling": "none", "augmentation": "none", "external_data": "none", "track_b": "not_used"}


@pytest.mark.parametrize("slug", OOF_SLUGS)
def test_encoder_oof_schema_and_folds_when_completed(slug: str) -> None:
    path = PROJECT_ROOT / "artifacts" / "predictions" / f"{slug}_oof.parquet"
    if not path.exists():
        pytest.skip("encoder experiment not completed yet")
    oof = pd.read_parquet(path)
    pool = construct_development_pool(PROJECT_ROOT)
    pool = attach_folds(pool, persist_cv_folds(PROJECT_ROOT, pool))
    validate_oof(oof, pool)
    assert oof.columns.tolist() == ["id", "fold", *[f"{label}_prob" for label in EXPECTED_LABELS], *[f"{label}_true" for label in EXPECTED_LABELS]]
