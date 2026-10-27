from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brighter.ensemble import (
    ALPHA_GRID, OUTPUTS, choose_alpha_row, fuse_probabilities,
    load_paired_inner_oof, select_alpha,
)
from brighter.experiments import EXPECTED_LABELS
from brighter.threshold_calibration import search_threshold

ROOT = Path(__file__).resolve().parents[1]


def test_alpha_grid_is_exact() -> None:
    assert np.array_equal(ALPHA_GRID, np.arange(21) / 20)
    assert ALPHA_GRID[0] == 0 and ALPHA_GRID[-1] == 1


def test_probability_fusion_and_endpoints() -> None:
    t2 = np.array([[.1, .9], [.3, .7]])
    t3 = np.array([[.8, .2], [.4, .6]])
    assert np.array_equal(fuse_probabilities(t2, t3, 1), t2)
    assert np.array_equal(fuse_probabilities(t2, t3, 0), t3)
    assert np.allclose(fuse_probabilities(t2, t3, .25), .25 * t2 + .75 * t3)


def test_alpha_selection_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=(80, 6))
    t2, t3 = rng.random((80, 6)), rng.random((80, 6))
    first = select_alpha(y, t2, t3)
    second = select_alpha(y, t2, t3)
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_alpha_tie_break_prefers_closest_to_half_then_larger_t2_weight() -> None:
    trace = pd.DataFrame({"alpha_T2": [.4, .6, .9], "inner_macro_f1": [.7, .7, .7]})
    assert choose_alpha_row(trace).alpha_T2 == pytest.approx(.6)
    trace = pd.DataFrame({"alpha_T2": [.3, .5, .7], "inner_macro_f1": [.7, .7, .7]})
    assert choose_alpha_row(trace).alpha_T2 == pytest.approx(.5)


def test_threshold_tie_break_remains_largest_threshold() -> None:
    threshold, score = search_threshold(np.array([1, 0, 0]), np.array([.8, .2, .1]))
    assert threshold == pytest.approx(.8)
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("outer_fold", range(5))
def test_paired_inner_oof_is_complete_finite_and_leakage_safe(outer_fold: int) -> None:
    paired = load_paired_inner_oof(ROOT, outer_fold)
    assert len(paired) in (1940, 1941)
    assert paired.id.nunique() == len(paired)
    probability_columns = [f"{label}_{system}_prob" for system in ("t2", "t3") for label in EXPECTED_LABELS]
    probabilities = paired[probability_columns].to_numpy(float)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    outer_ids = set(pd.read_parquet(ROOT / "artifacts" / "predictions" / "bertweetbr_asl_thresholded_oof.parquet")
                    .query("fold == @outer_fold").id.astype(str))
    assert not set(paired.id.astype(str)) & outer_ids


@pytest.mark.parametrize("system", ("EN1", "EN2"))
def test_final_ensemble_oof_integrity_when_present(system: str) -> None:
    path = ROOT / "artifacts" / "predictions" / f"{OUTPUTS[system]}_oof.parquet"
    if not path.exists():
        pytest.skip("ensemble has not been executed yet")
    frame = pd.read_parquet(path)
    source = pd.read_parquet(ROOT / "artifacts" / "predictions" / "bertweetbr_asl_thresholded_oof.parquet")
    assert len(frame) == frame.id.nunique() == 2426
    assert frame.id.tolist() == source.id.tolist()
    assert frame.fold.tolist() == source.fold.tolist()
    probabilities = frame[[f"{label}_prob" for label in EXPECTED_LABELS]].to_numpy(float)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert frame.alpha_T2.between(0, 1).all()
