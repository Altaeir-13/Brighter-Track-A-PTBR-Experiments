"""Numerically stable multilabel losses for the BRIGHTER ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FoldClassWeights:
    """Class statistics computed from one outer-fold training partition only."""

    positive: np.ndarray
    negative: np.ndarray
    pos_weight: np.ndarray
    focal_alpha: np.ndarray


def calculate_fold_class_weights(targets: np.ndarray) -> FoldClassWeights:
    """Calculate L1 and L3 weights without using validation observations."""

    values = np.asarray(targets)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("targets must be a non-empty [examples, labels] matrix")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("targets must be binary")
    positive = values.sum(axis=0, dtype=np.float64)
    negative = values.shape[0] - positive
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError("every label needs positive and negative training examples")
    pos_weight = negative / positive
    # Hanif et al. (2025), Eq. 1: w_i = N / (|C_i| * k).
    focal_alpha = values.shape[0] / (positive * values.shape[1])
    if not np.isfinite(pos_weight).all() or not np.isfinite(focal_alpha).all():
        raise ValueError("class weights must be finite")
    return FoldClassWeights(positive, negative, pos_weight, focal_alpha)


def focal_loss_with_logits(
    logits: Any,
    targets: Any,
    *,
    gamma: float = 2.0,
    alpha: Any | None = None,
) -> Any:
    """Mean binary focal loss over logits; alpha may be a per-label vector.

    The stable BCE term is multiplied by ``(1-p_t)**gamma``. With gamma=0
    and alpha=None (or all ones), this is exactly BCE-with-logits under the
    same mean reduction.
    """

    import torch
    import torch.nn.functional as functional

    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    logits32 = logits.float()
    targets32 = targets.float()
    if logits32.shape != targets32.shape:
        raise ValueError("logits and targets must have identical shapes")
    bce = functional.binary_cross_entropy_with_logits(logits32, targets32, reduction="none")
    modulation = torch.exp(-bce).neg().add(1).pow(gamma)
    loss = modulation * bce
    if alpha is not None:
        alpha_tensor = torch.as_tensor(alpha, dtype=loss.dtype, device=loss.device)
        if alpha_tensor.ndim != 1 or alpha_tensor.shape[0] != loss.shape[-1]:
            raise ValueError("alpha must contain one value per label")
        if not torch.isfinite(alpha_tensor).all() or torch.any(alpha_tensor <= 0):
            raise ValueError("alpha values must be finite and positive")
        loss = loss * alpha_tensor
    return loss.mean()


def asymmetric_loss_with_logits(
    logits: Any,
    targets: Any,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> Any:
    """Mean ASL using Ridnik et al.'s shifted negative probability.

    For positives the term is ``-(1-p)^gamma_pos log(p)``. For negatives,
    ``p_m=max(p-clip, 0)`` and the term is
    ``-p_m^gamma_neg log(1-p_m)``. Softplus/log1p keep extreme logits finite.
    """

    import torch
    import torch.nn.functional as functional

    if gamma_pos < 0 or gamma_neg < 0:
        raise ValueError("ASL gammas must be non-negative")
    if not 0 <= clip < 1:
        raise ValueError("ASL clip must be in [0, 1)")
    logits32 = logits.float()
    targets32 = targets.float()
    if logits32.shape != targets32.shape:
        raise ValueError("logits and targets must have identical shapes")
    probability = torch.sigmoid(logits32)
    positive_loss = functional.softplus(-logits32) * (1.0 - probability).pow(gamma_pos)
    shifted_probability = (probability - clip).clamp_min(0.0)
    negative_loss = -torch.log1p(-shifted_probability) * shifted_probability.pow(gamma_neg)
    return torch.where(targets32 > 0.5, positive_loss, negative_loss).mean()


def build_loss(
    loss_name: str,
    *,
    fold_weights: FoldClassWeights,
    loss_config: Mapping[str, Any] | None = None,
) -> Any:
    """Build the configured callable while keeping fold weights explicit."""

    import torch

    options = dict(loss_config or {})
    normalized = loss_name.lower()
    if normalized == "bce":
        return torch.nn.BCEWithLogitsLoss()
    if normalized == "weighted_bce":
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(fold_weights.pos_weight, dtype=torch.float32))
    if normalized == "focal":
        gamma = float(options.get("gamma", 2.0))
        return lambda logits, targets: focal_loss_with_logits(logits, targets, gamma=gamma)
    if normalized == "weighted_focal":
        gamma = float(options.get("gamma", 2.0))
        return lambda logits, targets: focal_loss_with_logits(logits, targets, gamma=gamma, alpha=fold_weights.focal_alpha)
    if normalized == "asl":
        gamma_pos = float(options.get("gamma_pos", 0.0))
        gamma_neg = float(options.get("gamma_neg", 4.0))
        clip = float(options.get("clip", 0.05))
        return lambda logits, targets: asymmetric_loss_with_logits(
            logits, targets, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip
        )
    raise ValueError(f"Unsupported transformer loss: {loss_name}")
