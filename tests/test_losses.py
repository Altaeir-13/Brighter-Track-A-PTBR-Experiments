from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from brighter.losses import (
    asymmetric_loss_with_logits,
    calculate_fold_class_weights,
    focal_loss_with_logits,
)


def test_fold_weights_have_correct_shape_are_finite_and_prioritize_rare_class() -> None:
    targets = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=int)
    weights = calculate_fold_class_weights(targets)
    assert weights.pos_weight.shape == (2,)
    assert weights.focal_alpha.shape == (2,)
    assert np.isfinite(weights.pos_weight).all()
    assert weights.pos_weight[1] > weights.pos_weight[0]
    assert weights.focal_alpha[1] > weights.focal_alpha[0]
    assert weights.pos_weight.tolist() == pytest.approx([1 / 3, 3.0])


def test_fold_weights_depend_only_on_passed_training_targets() -> None:
    training = np.array([[1, 0], [0, 0], [0, 1]], dtype=int)
    validation_a = np.zeros((20, 2), dtype=int)
    validation_b = np.ones((20, 2), dtype=int)
    first = calculate_fold_class_weights(training)
    _ = validation_a, validation_b  # Neither partition is accepted by the API.
    second = calculate_fold_class_weights(training)
    assert np.array_equal(first.pos_weight, second.pos_weight)


def test_focal_gamma_zero_matches_bce_and_weighted_alpha_is_per_class() -> None:
    logits = torch.tensor([[0.2, -1.1], [2.0, 0.3]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert focal_loss_with_logits(logits, targets, gamma=0).item() == pytest.approx(
        functional.binary_cross_entropy_with_logits(logits, targets).item()
    )
    alpha = torch.tensor([1.0, 3.0])
    manual = (functional.binary_cross_entropy_with_logits(logits, targets, reduction="none") * alpha).mean()
    assert focal_loss_with_logits(logits, targets, gamma=0, alpha=alpha).item() == pytest.approx(manual.item())


def test_focal_extreme_logits_is_finite_and_backward_valid() -> None:
    logits = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = focal_loss_with_logits(logits, targets, gamma=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_asl_matches_manual_terms_and_applies_margin_asymmetrically() -> None:
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    loss = asymmetric_loss_with_logits(logits, targets, gamma_pos=0, gamma_neg=4, clip=0.05)
    p = 0.5
    shifted = p - 0.05
    manual = (-math.log(p) - shifted**4 * math.log(1 - shifted)) / 2
    assert loss.item() == pytest.approx(manual)
    no_clip = asymmetric_loss_with_logits(logits, targets, gamma_pos=0, gamma_neg=4, clip=0)
    assert loss < no_clip
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_asl_extreme_logits_is_finite_and_backward_valid() -> None:
    logits = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = asymmetric_loss_with_logits(logits, targets, gamma_pos=0, gamma_neg=4, clip=0.05)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
