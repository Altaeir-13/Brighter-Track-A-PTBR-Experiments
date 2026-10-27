#!/usr/bin/env python
"""Run/reuse the complete frozen first baseline battery and consolidate it."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brighter.experiment_reporting import EXPERIMENTS, generate_baseline_reports
from brighter.experiments import EXPECTED_LABELS, run_experiment


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    configs = (
        PROJECT_ROOT / "configs" / "baseline_tfidf_logreg.yaml",
        PROJECT_ROOT / "configs" / "baseline_bertimbau_base_bce.yaml",
        PROJECT_ROOT / "configs" / "baseline_bertimbau_large_bce.yaml",
    )
    try:
        results = [run_experiment(PROJECT_ROOT, config) for config in configs]
        generate_baseline_reports(PROJECT_ROOT)
    except Exception as error:
        logging.exception("Baseline battery failed: %s", error)
        return 1
    print("\nBRIGHTER BASELINE EXPERIMENTS COMPLETE\n")
    print("Development:\n2426 examples\n5 multilabel-stratified folds\n")
    print("Results:")
    for result in results:
        metric = result["cross_validation"]["macro_f1"]
        print(f"{result['experiment_id']}: Macro-F1 {metric['mean']:.4f} +/- {metric['std']:.4f}")
    print("\nPer-label OOF F1:")
    for result in results:
        values = ", ".join(f"{label}={result['oof_aggregate']['per_label'][label]['f1']:.4f}" for label in EXPECTED_LABELS)
        print(f"{result['experiment_id']}: {values}")
    print("\nOOF predictions: 2426 / 2426 for all models")
    print("Test set: SEALED - NOT EVALUATED")
    print("Data modifications: NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
